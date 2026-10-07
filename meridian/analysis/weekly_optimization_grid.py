# Copyright 2026 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weekly optimization grid information."""

from collections.abc import Callable, Sequence
import dataclasses
from typing import Any
import warnings

from meridian import backend
from meridian import constants as c
from meridian.analysis import analyzer as analyzer_module
from meridian.analysis import optimizer
from meridian.common import errors as common_errors
from meridian.data import time_coordinates as tc
import numpy as np
import xarray as xr

__all__ = [
    'WeeklyOptimizationGrid',
]


@dataclasses.dataclass(frozen=True)
class WeeklyOptimizationGrid:
  """Weekly optimization grid information.

  Attributes:
    incremental_outcome: xr.DataArray of shape `(n_media_channels,
      n_spend_multipliers, n_times)` containing incremental outcome for all
      impression-based paid media channels. `None` if the model has no media
      channels.
    rf_incremental_outcome: xr.DataArray of shape `(n_rf_channels,
      n_frequencies, n_times)` containing the incremental outcome of every
      reach and frequency channel at historical spend (spend multiplier `1.0`)
      for each candidate frequency. Impressions are held fixed as frequency
      varies, so reach is `impressions / frequency`. Since the RF incremental
      outcome is linear in reach (and therefore spend) for a fixed frequency,
      the outcome at any spend multiplier `M` is `M` times this value. The
      optimal frequency for an optimization period is resolved at
      `to_optimization_grid` time by maximizing the outcome summed over the
      selected weeks. If `use_optimal_frequency` is `False`, the frequency
      dimension has a single `NaN` coordinate and holds the outcome at
      historical frequency. `None` if the model has no RF channels.
    nonoptimized_spend: xr.DataArray of shape `(n_paid_channels, n_times)`
      containing non-aggregated spend allocation for all paid channels.
    use_kpi: Whether using generic KPI or revenue.
    use_posterior: Whether posterior distributions were used, or prior.
    multiplier_step: Multiplier step size for the spend multiplier.
    max_budget_percent_decrease: Maximum percentage decrease in budget allowed.
    max_budget_percent_increase: Maximum percentage increase in budget allowed.
    max_constraint_variation: Maximum constraint variation allowed.
    use_optimal_frequency: Whether optimal frequency was used.
    max_frequency: Maximum frequency value used for optimal frequency.
  """

  incremental_outcome: xr.DataArray | None
  rf_incremental_outcome: xr.DataArray | None
  nonoptimized_spend: xr.DataArray
  use_kpi: bool
  use_posterior: bool
  multiplier_step: float
  max_budget_percent_decrease: float
  max_budget_percent_increase: float
  max_constraint_variation: float
  use_optimal_frequency: bool = True
  max_frequency: float | None = None

  @classmethod
  def create(
      cls,
      analyzer: analyzer_module.Analyzer,
      new_data: analyzer_module.DataTensors | None = None,
      *,
      start_date: tc.Date | None = None,
      end_date: tc.Date | None = None,
      use_posterior: bool = True,
      use_kpi: bool = False,
      max_budget_percent_decrease: float = 0.9,
      max_budget_percent_increase: float = 1,
      max_constraint_variation: float = 0.3,
      multiplier_step: float | None = None,
      batch_size: int = 10,
      use_optimal_frequency: bool = True,
      max_frequency: float | None = None,
      chains_per_batch: int | None = None,
  ) -> 'WeeklyOptimizationGrid':
    """Builds a weekly optimization grid using vectorized backend calculations.

    This method pre-calculates weekly incremental outcomes across MCMC draws
    and chains over a discretized spend multiplier grid. This pre-computed grid
    can then be passed to `BudgetOptimizer.optimize`, enabling fast linear
    interpolation lookups during optimization iterations.

    Args:
      analyzer: An `Analyzer` instance with a fitted model.
      new_data: An optional `DataTensors` container with optional counterfactual
        or future data tensors. If any tensors have modified time dimensions,
        all tensors must have the same number of time periods and
        `new_data.time` must be provided. Defaults to None.
      start_date: Optional start date selector, inclusive. Defaults to None.
      end_date: Optional end date selector, inclusive. Defaults to None.
      use_posterior: Boolean. If True, the incremental outcome is derived from
        the posterior distribution of the model. Otherwise, the prior
        distribution is used. Defaults to True.
      use_kpi: Boolean. If True, the incremental outcome is derived from the KPI
        impact. Otherwise, the incremental outcome is derived from the revenue
        impact. Defaults to False.
      max_budget_percent_decrease: Maximum budget decrease allowed, must be in
        the range [0, 1). Defaults to 0.9.
      max_budget_percent_increase: Maximum budget increase allowed, must be
        non-negative. Defaults to 1.0.
      max_constraint_variation: Maximum constraint variation allowed for each
        channel, must be non-negative. Defaults to 0.3.
      multiplier_step: Multiplier step size (delta) for the spend multiplier
        grid. If None, it is dynamically computed based on default tolerance.
        Defaults to None.
      batch_size: Maximum number of grid points to process in each batch to
        avoid memory exhaustion. Grid points are spend multipliers for
        impression-based media channels and candidate frequencies for reach and
        frequency channels. Defaults to 10.
      use_optimal_frequency: Whether to precompute RF outcomes over a grid of
        candidate frequencies so that the optimal frequency can be resolved
        for any optimization period. If `False`, RF outcomes are computed at
        historical frequency. Defaults to True.
      max_frequency: Maximum frequency value used for the candidate frequency
        grid. If `None`, the maximum historical frequency of the model input
        data is used, matching `Analyzer.optimal_freq`. Defaults to None.
      chains_per_batch: Maximum number of MCMC chains to process in each batch.
        The computation is split over the chain dimension of the posterior (or
        prior) parameters and the per-batch means are averaged back together,
        which lowers the peak memory footprint without changing the result. If
        None, all chains are processed at once. Defaults to None.

    Returns:
      A `WeeklyOptimizationGrid` object containing the weekly grid dataset and
      scenario constraints.
    """
    if not 0.0 <= max_budget_percent_decrease < 1.0:
      raise ValueError(
          '`max_budget_percent_decrease` must be in the range [0, 1).'
          f' Got {max_budget_percent_decrease}.'
      )
    if max_budget_percent_increase < 0.0:
      raise ValueError(
          '`max_budget_percent_increase` must be non-negative. Got'
          f' {max_budget_percent_increase}.'
      )
    if max_constraint_variation < 0.0:
      raise ValueError(
          '`max_constraint_variation` must be non-negative. Got'
          f' {max_constraint_variation}.'
      )

    if chains_per_batch is not None and chains_per_batch < 1:
      raise ValueError(
          f'`chains_per_batch` must be positive. Got {chains_per_batch}.'
      )

    dist_type = c.POSTERIOR if use_posterior else c.PRIOR
    if dist_type not in analyzer.inference_data.groups():
      raise common_errors.NotFittedModelError(
          'Running budget optimization scenarios requires fitting the model.'
      )

    if new_data is None:
      new_data = analyzer_module.DataTensors()
    model_context = analyzer.model_context
    required_tensors = c.PERFORMANCE_DATA + (c.TIME,)
    filled_data = new_data.validate_and_fill_missing_data(
        required_tensors_names=required_tensors,
        model_context=model_context,
    )
    assert filled_data.time is not None
    channels = model_context.input_data.get_all_paid_channels()
    # Paid channels are ordered as media channels followed by RF channels.
    media_channels = list(channels[: model_context.n_media_channels])
    rf_channels = list(channels[model_context.n_media_channels :])
    all_times = list(filled_data.time)

    selected_times_opt = optimizer._expand_selected_times(  # pylint: disable=protected-access
        model_context=model_context,
        start_date=start_date,
        end_date=end_date,
        new_data=filled_data,
    )
    if selected_times_opt is not None:
      selected_times_list = [
          t.strftime(c.DATE_FORMAT) if not isinstance(t, str) else t
          for t in selected_times_opt
      ]
      time_indices = backend.to_tensor(
          [all_times.index(t) for t in selected_times_list],
          dtype=backend.int32,
      )
    else:
      selected_times_list = all_times
      time_indices = None

    nonoptimized_spend = analyzer.get_aggregated_spend(
        new_data=filled_data.filter_fields(
            c.PAID_CHANNELS + c.SPEND_DATA + (c.TIME,)
        ),
        selected_times=all_times,
        include_media=model_context.n_media_channels > 0,
        include_rf=model_context.n_rf_channels > 0,
        aggregate_times=False,
    ).transpose(c.CHANNEL, c.TIME)
    n_times = len(all_times)

    bounds = (
        (1 - max_budget_percent_decrease) * (1 - max_constraint_variation),
        (1 + max_budget_percent_increase) * (1 + max_constraint_variation),
    )
    lower, upper = bounds

    if multiplier_step is None:
      multiplier_step = 0.0001 * (1 - max_budget_percent_decrease)

    # Generate steps carefully matching numpy arange bounds.
    mults = np.round(
        np.arange(lower, upper + multiplier_step / 2.0, multiplier_step),
        decimals=6,
    )

    channel_multipliers = {channel: mults for channel in channels}
    unique_multipliers = sorted(set().union(*channel_multipliers.values()))

    inf_data = (
        analyzer.inference_data.posterior  # pyrefly: ignore[missing-attribute]
        if use_posterior
        else analyzer.inference_data.prior  # pyrefly: ignore[missing-attribute]
    )
    eqs = analyzer._model_equations  # pylint: disable=protected-access
    kpi_transformer = model_context.kpi_transformer

    def to_float(tensor_like: backend.Tensor) -> backend.Tensor:
      return backend.cast(
          backend.to_tensor(tensor_like), dtype=backend.float_dtype
      )

    population = to_float(model_context.population)

    if filled_data.revenue_per_kpi is None:
      n_geos = model_context.n_geos
      revenue_per_kpi = backend.ones(
          (n_geos, n_times), dtype=backend.float_dtype
      )
    else:
      revenue_per_kpi = to_float(filled_data.revenue_per_kpi)

    if time_indices is not None:
      revenue_per_kpi = backend.gather(revenue_per_kpi, time_indices, axis=1)

    if model_context.n_media_channels > 0:
      if model_context.media_tensors.media_transformer is None:
        media_base_scaled = to_float(model_context.media_tensors.media_scaled)  # pyrefly: ignore[bad-argument-type]
      else:
        media_base_scaled = to_float(
            model_context.media_tensors.media_transformer.forward(
                filled_data.media
            )
        )
      alpha_m = to_float(inf_data.alpha_m)
      ec_m = to_float(inf_data.ec_m)
      slope_m = to_float(inf_data.slope_m)
      beta_gm = to_float(inf_data.beta_gm)
      decay_m = model_context.adstock_decay_spec.media
      sat_m = model_context.saturation_spec.media
    else:
      media_base_scaled = None
      alpha_m = None
      ec_m = None
      slope_m = None
      beta_gm = None
      decay_m = None
      sat_m = None

    if model_context.n_rf_channels > 0:
      # Impressions are held fixed while frequency varies, so that
      # `reach = impressions / frequency`. The reach transformer is a per-geo,
      # per-channel scaling, so scaling impressions once and dividing by the
      # frequency inside the jitted computation is equivalent to scaling reach.
      rf_impressions = filled_data.reach * filled_data.frequency  # pyrefly: ignore[unsupported-operation]
      if model_context.rf_tensors.reach_transformer is None:
        rf_impressions_scaled = to_float(rf_impressions)
      else:
        rf_impressions_scaled = to_float(
            model_context.rf_tensors.reach_transformer.forward(rf_impressions)
        )
      if use_optimal_frequency:
        # Match the frequency grid used by `Analyzer.optimal_freq`.
        max_freq = max_frequency or np.max(
            np.array(model_context.rf_tensors.frequency)
        )
        freq_grid = np.arange(1, max_freq, 0.1)
        if freq_grid.size == 0:
          raise ValueError(
              'The optimal frequency grid is empty. `max_frequency` must be'
              f' greater than 1. Got {max_freq}.'
          )
        historical_frequency = None
      else:
        # A single placeholder frequency; outcomes are computed at historical
        # frequency.
        freq_grid = np.array([np.nan])
        historical_frequency = to_float(filled_data.frequency)  # pyrefly: ignore[bad-argument-type]
      alpha_rf = to_float(inf_data.alpha_rf)
      ec_rf = to_float(inf_data.ec_rf)
      slope_rf = to_float(inf_data.slope_rf)
      beta_grf = to_float(inf_data.beta_grf)
      decay_rf = model_context.adstock_decay_spec.rf
      sat_rf = model_context.saturation_spec.rf
    else:
      rf_impressions_scaled = None
      freq_grid = None
      historical_frequency = None
      alpha_rf = None
      ec_rf = None
      slope_rf = None
      beta_grf = None
      decay_rf = None
      sat_rf = None

    grid_batch_size = max(1, batch_size)

    # MCMC parameters are shaped `(n_chains, n_draws, ...)`. Splitting the
    # computation over the chain dimension lowers the peak memory footprint.
    if alpha_m is not None:
      n_chains = int(alpha_m.shape[0])
    elif alpha_rf is not None:
      n_chains = int(alpha_rf.shape[0])
    else:
      n_chains = 0

    if chains_per_batch is None or n_chains == 0:
      chain_ranges = [(0, n_chains)]
    else:
      chain_ranges = [
          (start, min(start + chains_per_batch, n_chains))
          for start in range(0, n_chains, chains_per_batch)
      ]

    def slice_chains(
        tensor: backend.Tensor | None, start: int, stop: int
    ) -> backend.Tensor | None:
      """Slices the leading chain dimension, preserving the tensor rank."""
      if tensor is None or (start == 0 and stop == n_chains):
        return tensor
      return tensor[start:stop, ...]

    def run_in_batches(
        values: backend.Tensor,
        compute_fn: Callable[[backend.Tensor, int, int], backend.Tensor],
    ) -> np.ndarray:
      """Runs `compute_fn` over batches of `values` and chain ranges."""
      all_outcomes = []
      for i in range(0, len(values), grid_batch_size):
        batch = values[i : i + grid_batch_size]
        chain_outcomes = []
        chain_weights = []
        for chain_start, chain_stop in chain_ranges:
          chain_outcome = compute_fn(batch, chain_start, chain_stop)
          chain_outcomes.append(np.asarray(chain_outcome))
          chain_weights.append(chain_stop - chain_start)

        if len(chain_outcomes) == 1:
          batch_outcomes = chain_outcomes[0]
        else:
          # The batch computations average over the chain and draw dimensions.
          # Every chain has the same number of draws, so the global mean is the
          # mean of the per-batch means weighted by the number of chains in
          # each batch.
          batch_outcomes = np.average(
              np.stack(chain_outcomes, axis=0), axis=0, weights=chain_weights
          )
        all_outcomes.append(batch_outcomes)
      return np.concatenate(all_outcomes, axis=0)

    incremental_outcome = None
    if model_context.n_media_channels > 0:
      all_multipliers_array = backend.to_tensor(
          unique_multipliers, dtype=backend.float_dtype
      )

      def compute_media(
          batch: backend.Tensor, chain_start: int, chain_stop: int
      ) -> backend.Tensor:
        return cls._compute_batch(
            multiplier_batch=batch,
            media_base_scaled=media_base_scaled,
            alpha_m=slice_chains(alpha_m, chain_start, chain_stop),
            ec_m=slice_chains(ec_m, chain_start, chain_stop),
            slope_m=slice_chains(slope_m, chain_start, chain_stop),
            beta_gm=slice_chains(beta_gm, chain_start, chain_stop),
            revenue_per_kpi=revenue_per_kpi,
            population=population,
            time_indices=time_indices,
            eqs=eqs,
            decay_m=decay_m,
            sat_m=sat_m,
            n_times=n_times,
            kpi_transformer=kpi_transformer,
            use_kpi=use_kpi,
        )

      outcomes = run_in_batches(all_multipliers_array, compute_media)

      final_outcomes = np.transpose(outcomes, (2, 0, 1))

      incremental_outcome = xr.DataArray(
          final_outcomes,
          coords={
              c.CHANNEL: media_channels,
              c.SPEND_MULTIPLIER: unique_multipliers,
              c.TIME: selected_times_list,
          },
          dims=[c.CHANNEL, c.SPEND_MULTIPLIER, c.TIME],
      )

    rf_incremental_outcome = None
    if model_context.n_rf_channels > 0:
      assert freq_grid is not None
      # For the historical frequency case the value is a placeholder that is
      # ignored by `_compute_rf_batch`.
      all_frequencies_array = backend.to_tensor(
          np.nan_to_num(freq_grid, nan=1.0), dtype=backend.float_dtype
      )

      def compute_rf(
          batch: backend.Tensor, chain_start: int, chain_stop: int
      ) -> backend.Tensor:
        return cls._compute_rf_batch(
            batch,
            rf_impressions_scaled,
            historical_frequency,
            slice_chains(alpha_rf, chain_start, chain_stop),
            slice_chains(ec_rf, chain_start, chain_stop),
            slice_chains(slope_rf, chain_start, chain_stop),
            slice_chains(beta_grf, chain_start, chain_stop),
            revenue_per_kpi,
            population,
            time_indices=time_indices,
            eqs=eqs,
            decay_rf=decay_rf,
            sat_rf=sat_rf,
            n_times=n_times,
            kpi_transformer=kpi_transformer,
            use_kpi=use_kpi,
        )

      rf_outcomes = run_in_batches(all_frequencies_array, compute_rf)

      rf_incremental_outcome = xr.DataArray(
          np.transpose(rf_outcomes, (2, 0, 1)),
          coords={
              c.CHANNEL: rf_channels,
              c.FREQUENCY: freq_grid,
              c.TIME: selected_times_list,
          },
          dims=[c.CHANNEL, c.FREQUENCY, c.TIME],
      )

    if selected_times_opt is not None:
      nonoptimized_spend = nonoptimized_spend.sel({c.TIME: selected_times_list})

    return cls(
        incremental_outcome=incremental_outcome,
        nonoptimized_spend=nonoptimized_spend,
        use_kpi=use_kpi,
        use_posterior=use_posterior,
        multiplier_step=multiplier_step,
        max_budget_percent_decrease=max_budget_percent_decrease,
        max_budget_percent_increase=max_budget_percent_increase,
        max_constraint_variation=max_constraint_variation,
        use_optimal_frequency=use_optimal_frequency,
        max_frequency=max_frequency,
        rf_incremental_outcome=rf_incremental_outcome,
    )

  @staticmethod
  def _aggregate_incremental_outcome(
      effect_diff: backend.Tensor,
      beta: backend.Tensor,
      revenue_per_kpi: backend.Tensor,
      time_indices: backend.Tensor | None,
      population: backend.Tensor,
      kpi_transformer: Any,
      use_kpi: bool,
  ) -> backend.Tensor:
    """Converts transformed media effects into a per-week incremental outcome.

    Args:
      effect_diff: Tensor of shape `(n_chains, n_draws, n_geos, n_times,
        n_channels)` with the difference of the adstock/Hill transformed
        effects between the scenario and the zero-media counterfactual.
      beta: Tensor of shape `(n_chains, n_draws, n_geos, n_channels)` with the
        channel coefficients.
      revenue_per_kpi: Tensor of shape `(n_geos, n_selected_times)`.
      time_indices: Optional indices of the selected time periods.
      population: Tensor of shape `(n_geos,)` with the population of each geo.
      kpi_transformer: The KPI transformer of the model.
      use_kpi: Whether to return the outcome in KPI units instead of revenue.

    Returns:
      Tensor of shape `(n_selected_times, n_channels)` with the incremental
      outcome averaged over chains and draws and summed over geos.
    """
    if time_indices is not None:
      effect_diff = backend.gather(effect_diff, time_indices, axis=3)

    incremental_kpi = backend.einsum('...gtm,...gm->...gtm', effect_diff, beta)
    # Inverse transform incremental KPI to the natural scale. Because
    # `kpi_transformer.inverse` adds `population_scaled_mean`, we scale only
    # by `population_scaled_stdev` and `population` to omit the mean
    # intercept/offset and obtain the uncentered incremental KPI on the
    # natural scale.
    incremental_kpi_natural = (
        incremental_kpi
        * kpi_transformer.population_scaled_stdev
        * population[:, backend.newaxis, backend.newaxis]
    )

    if use_kpi:
      incremental_outcome = incremental_kpi_natural
    else:
      incremental_outcome = backend.einsum(
          'gt,...gtm->...gtm', revenue_per_kpi, incremental_kpi_natural
      )

    incremental_outcome_f64 = backend.cast(
        incremental_outcome, backend.np_float_dtype
    )
    mean_incremental_outcome = backend.reduce_mean(
        incremental_outcome_f64, axis=(0, 1)
    )

    return backend.reduce_sum(mean_incremental_outcome, axis=0)

  @classmethod
  @backend.function(
      jit_compile=True,
      static_argnames=[
          'eqs',
          'decay_m',
          'sat_m',
          'n_times',
          'kpi_transformer',
          'use_kpi',
      ],
  )
  def _compute_batch(
      cls,
      *,
      multiplier_batch: backend.Tensor,
      media_base_scaled: backend.Tensor,
      alpha_m: backend.Tensor,
      ec_m: backend.Tensor,
      slope_m: backend.Tensor,
      beta_gm: backend.Tensor,
      revenue_per_kpi: backend.Tensor,
      population: backend.Tensor,
      time_indices: backend.Tensor | None,
      eqs: Any,
      decay_m: Any,
      sat_m: Any,
      n_times: int,
      kpi_transformer: Any,
      use_kpi: bool,
  ) -> backend.Tensor:
    """Computes media incremental outcome for a batch of spend multipliers."""

    def _compute_kpi_for_multiplier(
        multiplier: backend.Tensor,
    ) -> backend.Tensor:
      multiplier_float = backend.cast(multiplier, backend.float_dtype)

      media_t1 = eqs.adstock_hill_media(
          media=media_base_scaled * multiplier_float,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
          decay_functions=decay_m,
          saturation_spec=sat_m,
          n_times_output=n_times,
      )
      media_t0 = eqs.adstock_hill_media(
          media=media_base_scaled * 0.0,
          alpha=alpha_m,
          ec=ec_m,
          slope=slope_m,
          decay_functions=decay_m,
          saturation_spec=sat_m,
          n_times_output=n_times,
      )
      return cls._aggregate_incremental_outcome(
          media_t1 - media_t0,
          beta_gm,
          revenue_per_kpi=revenue_per_kpi,
          time_indices=time_indices,
          population=population,
          kpi_transformer=kpi_transformer,
          use_kpi=use_kpi,
      )

    return backend.vectorized_map(_compute_kpi_for_multiplier, multiplier_batch)

  @classmethod
  @backend.function(
      jit_compile=True,
      static_argnames=[
          'eqs',
          'decay_rf',
          'sat_rf',
          'n_times',
          'kpi_transformer',
          'use_kpi',
      ],
  )
  def _compute_rf_batch(
      cls,
      frequency_batch: backend.Tensor,
      rf_impressions_scaled: backend.Tensor,
      historical_frequency: backend.Tensor | None,
      alpha_rf: backend.Tensor,
      ec_rf: backend.Tensor,
      slope_rf: backend.Tensor,
      beta_grf: backend.Tensor,
      revenue_per_kpi: backend.Tensor,
      population: backend.Tensor,
      time_indices: backend.Tensor | None,
      eqs: Any,
      decay_rf: Any,
      sat_rf: Any,
      n_times: int,
      kpi_transformer: Any,
      use_kpi: bool,
  ) -> backend.Tensor:
    """Computes RF incremental outcome at historical spend for frequencies.

    For each candidate frequency `f`, frequency is set to `f` for all geos and
    times while impressions are held fixed, i.e. `reach = impressions / f`. If
    `historical_frequency` is provided, the candidate frequency is ignored and
    the historical frequency is used instead.

    Args:
      frequency_batch: Tensor of shape `(n_batch,)` with candidate frequencies.
      rf_impressions_scaled: Tensor of shape `(n_geos, n_media_times,
        n_rf_channels)` with impressions scaled by the reach transformer.
      historical_frequency: Optional tensor of shape `(n_geos, n_media_times,
        n_rf_channels)` with historical frequency.
      alpha_rf: Adstock parameter.
      ec_rf: Hill half-saturation parameter.
      slope_rf: Hill slope parameter.
      beta_grf: RF channel coefficients.
      revenue_per_kpi: Tensor of shape `(n_geos, n_selected_times)`.
      population: Tensor of shape `(n_geos,)` with the population of each geo.
      time_indices: Optional indices of the selected time periods.
      eqs: Model equations.
      decay_rf: RF adstock decay functions.
      sat_rf: RF saturation spec.
      n_times: Number of output time periods.
      kpi_transformer: The KPI transformer of the model.
      use_kpi: Whether to return the outcome in KPI units instead of revenue.

    Returns:
      Tensor of shape `(n_batch, n_selected_times, n_rf_channels)`.
    """

    def _compute_kpi_for_frequency(
        frequency_value: backend.Tensor,
    ) -> backend.Tensor:
      if historical_frequency is None:
        frequency = backend.ones_like(rf_impressions_scaled) * backend.cast(
            frequency_value, backend.float_dtype
        )
      else:
        frequency = historical_frequency
      reach = backend.divide_no_nan(rf_impressions_scaled, frequency)

      rf_t1 = eqs.adstock_hill_rf(
          reach=reach,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
          decay_functions=decay_rf,
          saturation_spec=sat_rf,
          n_times_output=n_times,
      )
      rf_t0 = eqs.adstock_hill_rf(
          reach=reach * 0.0,
          frequency=frequency,
          alpha=alpha_rf,
          ec=ec_rf,
          slope=slope_rf,
          decay_functions=decay_rf,
          saturation_spec=sat_rf,
          n_times_output=n_times,
      )
      return cls._aggregate_incremental_outcome(
          rf_t1 - rf_t0,
          beta_grf,
          revenue_per_kpi=revenue_per_kpi,
          time_indices=time_indices,
          population=population,
          kpi_transformer=kpi_transformer,
          use_kpi=use_kpi,
      )

    return backend.vectorized_map(_compute_kpi_for_frequency, frequency_batch)

  @property
  def media_channels(self) -> list[str]:
    """The impression-based media channels in the weekly grid."""
    if self.incremental_outcome is None:
      return []
    return self.incremental_outcome.channel.data.tolist()

  @property
  def rf_channels(self) -> list[str]:
    """The reach and frequency channels in the weekly grid."""
    if self.rf_incremental_outcome is None:
      return []
    return self.rf_incremental_outcome.channel.data.tolist()

  @property
  def n_rf_channels(self) -> int:
    """The number of reach and frequency channels in the weekly grid."""
    return len(self.rf_channels)

  @property
  def channels(self) -> list[str]:
    """The spend channels in the weekly grid (media followed by RF)."""
    return self.media_channels + self.rf_channels

  @property
  def time(self) -> list[str]:
    """The spend times in the weekly grid."""
    return self.nonoptimized_spend.time.data.tolist()

  @property
  def opt_freq_ds(self) -> xr.Dataset | None:
    """Optimal frequency results over the full grid period.

    Derived from `rf_incremental_outcome` without additional model evaluation.
    The schema follows `Analyzer.optimal_freq`, restricted to the variables
    that can be computed from the grid, and with only the `mean` metric (the
    grid stores the mean incremental outcome over draws):

    * `roi`: `(frequency, rf_channel, metric)` ROI curve over the candidate
      frequencies.
    * `optimal_frequency`: `(rf_channel,)` frequency maximizing ROI.
    * `optimized_incremental_outcome`: `(rf_channel, metric)` incremental
      outcome at historical spend and optimal frequency.
    * `optimized_roi`: `(rf_channel, metric)` ROI at optimal frequency.

    The optimal frequency is resolved over all weeks of the grid. Use
    `to_optimization_grid` to resolve it for a sub-period.

    Returns:
      An `xr.Dataset` as described above, or `None` if the grid has no RF
      channels or was created with `use_optimal_frequency=False`.
    """
    if self.rf_incremental_outcome is None or not self.use_optimal_frequency:
      return None

    rf_outcome = self.rf_incremental_outcome
    rf_times = rf_outcome[c.TIME].values
    optimal_frequency, optimized_outcome = self._resolve_rf_outcomes(
        list(rf_times)
    )
    assert optimal_frequency is not None

    rf_spend = (
        self.nonoptimized_spend.sel({c.CHANNEL: rf_outcome[c.CHANNEL].values})
        .sel({c.TIME: rf_times})
        .sum(dim=c.TIME)
        .values
    )
    # Zero historical spend yields NaN ROI, consistent with `Analyzer`.
    safe_spend = np.where(rf_spend == 0, np.nan, rf_spend)
    # Shape `(n_frequencies, n_rf_channels)`.
    summed_outcome = (
        rf_outcome.sum(dim=c.TIME).transpose(c.FREQUENCY, c.CHANNEL).values
    )
    roi = summed_outcome / safe_spend
    optimized_roi = optimized_outcome / safe_spend

    return xr.Dataset(
        data_vars={
            c.ROI: (
                [c.FREQUENCY, c.RF_CHANNEL, c.METRIC],
                roi[..., np.newaxis],
            ),
            c.OPTIMAL_FREQUENCY: ([c.RF_CHANNEL], optimal_frequency),
            c.OPTIMIZED_INCREMENTAL_OUTCOME: (
                [c.RF_CHANNEL, c.METRIC],
                optimized_outcome[:, np.newaxis],
            ),
            c.OPTIMIZED_ROI: (
                [c.RF_CHANNEL, c.METRIC],
                optimized_roi[:, np.newaxis],
            ),
        },
        coords={
            c.FREQUENCY: rf_outcome[c.FREQUENCY].values,
            c.RF_CHANNEL: rf_outcome[c.CHANNEL].values,
            c.METRIC: [c.MEAN],
        },
        attrs={
            c.USE_POSTERIOR: self.use_posterior,
            c.IS_REVENUE_KPI: not self.use_kpi,
        },
    )

  def _resolve_rf_outcomes(
      self, selected_times: Sequence[str]
  ) -> tuple[np.ndarray | None, np.ndarray]:
    """Resolves RF optimal frequency and outcome for the selected weeks.

    The optimal frequency maximizes the RF incremental outcome at historical
    spend summed over the selected weeks. Since spend is fixed across
    candidate frequencies, this is equivalent to maximizing ROI as done in
    `Analyzer.optimal_freq`.

    Args:
      selected_times: The weeks of the optimization period.

    Returns:
      A tuple `(optimal_frequency, rf_outcome)`. `optimal_frequency` is an
      array of shape `(n_rf_channels,)` or `None` if historical frequency is
      used. `rf_outcome` is an array of shape `(n_rf_channels,)` with the
      incremental outcome of each RF channel at historical spend, summed over
      the selected weeks, at the resolved frequency.
    """
    if self.rf_incremental_outcome is None:
      return None, np.zeros(0)

    week_mask = np.isin(
        self.rf_incremental_outcome[c.TIME].values, selected_times
    )
    # Shape `(n_rf_channels, n_frequencies)`.
    summed_outcomes = (
        self.rf_incremental_outcome.isel({c.TIME: week_mask})
        .sum(dim=c.TIME)
        .transpose(c.CHANNEL, c.FREQUENCY)
        .values
    )
    if not self.use_optimal_frequency:
      return None, summed_outcomes[:, 0]

    optimal_freq_idx = np.nanargmax(summed_outcomes, axis=1)
    freq_grid = self.rf_incremental_outcome[c.FREQUENCY].values
    rf_channel_indices = np.arange(len(optimal_freq_idx))
    return (
        np.asarray(freq_grid[optimal_freq_idx], dtype=float),
        summed_outcomes[rf_channel_indices, optimal_freq_idx],
    )

  def _validate_dates(
      self,
      *,
      start_date: tc.Date | None = None,
      end_date: tc.Date | None = None,
  ) -> bool:
    """Checks if the weekly optimization grid covers the scenario date range.

    Args:
      start_date: Start date of the optimization period.
      end_date: End date of the optimization period.

    Returns:
      True if the weekly grid covers the given start and end dates, False
      otherwise.
    """
    grid_times = set(self.time)
    if start_date is not None:
      start_date_str = tc.normalize_date(start_date).strftime(c.DATE_FORMAT)
      if start_date_str not in grid_times:
        warnings.warn(
            f'Given weekly grid does not cover start_date {start_date_str}.'
        )
        return False

    if end_date is not None:
      end_date_str = tc.normalize_date(end_date).strftime(c.DATE_FORMAT)
      if end_date_str not in grid_times:
        warnings.warn(
            f'Given weekly grid does not cover end_date {end_date_str}.'
        )
        return False

    return True

  def _validate_optimization_bounds(
      self,
      *,
      lower_bound: np.ndarray,
      upper_bound: np.ndarray,
      hist_spend: np.ndarray,
      round_factor: int,
  ) -> bool:
    """Checks if the weekly optimization grid covers the optimization bounds.

    Args:
      lower_bound: `np.ndarray` of shape `(n_channels,)` containing the lower
        bound for each channel.
      upper_bound: `np.ndarray` of shape `(n_channels,)` containing the upper
        bound for each channel.
      hist_spend: `np.ndarray` of shape `(n_channels,)` containing the
        historical spend for each channel.
      round_factor: Integer number of digits to round optimization bounds.

    Returns:
      True if the weekly grid covers the optimization bounds, False otherwise.
    """
    errors = []
    rf_channels = set(self.rf_channels)
    rounded_hist_spend = np.round(hist_spend, round_factor).astype(int)
    for i, (channel, channel_spend) in enumerate(
        zip(self.channels, rounded_hist_spend)
    ):
      # RF outcomes are linear in spend, so any bounds are covered.
      if channel_spend == 0 or channel in rf_channels:
        continue

      channel_grid = self.incremental_outcome.sel({c.CHANNEL: channel}).dropna(  # pyrefly: ignore[missing-attribute]
          dim=c.SPEND_MULTIPLIER, how='all'
      )
      channel_mults = channel_grid[c.SPEND_MULTIPLIER].values
      channel_min_spend = float(
          np.round(channel_mults[0] * channel_spend, round_factor).astype(int)
      )
      channel_max_spend = float(
          np.round(channel_mults[-1] * channel_spend, round_factor).astype(int)
      )

      if lower_bound[i] < channel_min_spend:
        errors.append(
            f'Lower bound {lower_bound[i]} for channel {channel} is below the'
            f' mimimum spend of the grid {channel_min_spend}.'
        )
      if upper_bound[i] > channel_max_spend:
        errors.append(
            f'Upper bound {upper_bound[i]} for channel {channel} is above the'
            f' maximum spend of the grid {channel_max_spend}.'
        )

    if errors:
      warnings.warn(
          'Bounds are not within the grid. Error message:\n' + '\n'.join(errors)
      )
      return False

    return True

  def to_optimization_grid(
      self,
      start_date: tc.Date = None,
      end_date: tc.Date = None,
      budget: float | None = None,
      spend_constraint_lower: float | Sequence[float] | None = None,
      spend_constraint_upper: float | Sequence[float] | None = None,
  ) -> optimizer.OptimizationGrid | None:
    """Creates an OptimizationGrid from the weekly grid.

    If validation fails, returns None.

    Args:
      start_date: Start date of the optimization period.
      end_date: End date of the optimization period.
      budget: The total budget for the optimization period. If unspecified, it
        represents historical total spend.
      spend_constraint_lower: Numeric list of size `channels` or float (same
        constraint for all channels) indicating the lower bound of media-level
        spend. Defaults to `max_constraint_variation`.
      spend_constraint_upper: Numeric list of size `channels` or float (same
        constraint for all channels) indicating the upper bound of media-level
        spend. Defaults to `max_constraint_variation`.

    Returns:
      An OptimizationGrid containing the computed spend and outcome grids,
      or None if validation failed.
    """
    if not self._validate_dates(
        start_date=start_date,
        end_date=end_date,
    ):
      return None

    if start_date is not None:
      start_date_str = tc.normalize_date(start_date).strftime(c.DATE_FORMAT)
    else:
      start_date_str = self.time[0]

    if end_date is not None:
      end_date_str = tc.normalize_date(end_date).strftime(c.DATE_FORMAT)
    else:
      end_date_str = self.time[-1]

    selected_times = [
        t for t in self.time if start_date_str <= t <= end_date_str
    ]

    selected_spend = self.nonoptimized_spend.sel(
        time=slice(start_date_str, end_date_str)
    )
    hist_spend = selected_spend.sum(dim=c.TIME).to_numpy()
    n_paid_channels = len(self.channels)
    budget_val = budget if budget is not None else np.sum(hist_spend)

    valid_pct_of_spend = optimizer._validate_pct_of_spend(  # pylint: disable=protected-access
        n_channels=n_paid_channels,
        hist_spend=hist_spend,
        pct_of_spend=None,
    )
    spend = budget_val * valid_pct_of_spend
    round_factor = optimizer.get_round_factor(budget_val, gtol=0.0001)
    if spend_constraint_lower is None:
      spend_constraint_lower = self.max_constraint_variation
    if spend_constraint_upper is None:
      spend_constraint_upper = self.max_constraint_variation
    optimization_lower_bound, optimization_upper_bound = (
        optimizer.get_optimization_bounds(
            n_channels=n_paid_channels,
            spend=spend,
            round_factor=round_factor,
            spend_constraint_lower=spend_constraint_lower,
            spend_constraint_upper=spend_constraint_upper,
        )
    )

    if not self._validate_optimization_bounds(
        lower_bound=optimization_lower_bound,
        upper_bound=optimization_upper_bound,
        hist_spend=hist_spend,
        round_factor=round_factor,
    ):
      return None

    step_size = 10 ** (-round_factor)
    spend_grid, incremental_outcome_grid = self._create_grids(
        spend_bound_lower=optimization_lower_bound,
        spend_bound_upper=optimization_upper_bound,
        step_size=step_size,
        spend=hist_spend,
        selected_times=selected_times,
    )

    grid_dataset = xr.Dataset(
        data_vars={
            c.SPEND_GRID: (
                [c.GRID_SPEND_INDEX, c.CHANNEL],
                spend_grid,
            ),
            c.INCREMENTAL_OUTCOME_GRID: (
                [c.GRID_SPEND_INDEX, c.CHANNEL],
                incremental_outcome_grid,
            ),
        },
        coords={
            c.GRID_SPEND_INDEX: np.arange(0, len(spend_grid)),
            c.CHANNEL: self.channels,
        },
        attrs={c.SPEND_STEP_SIZE: step_size},
    )

    optimal_frequency, _ = self._resolve_rf_outcomes(selected_times)

    return optimizer.OptimizationGrid(
        _grid_dataset=grid_dataset,
        historical_spend=hist_spend,
        use_kpi=self.use_kpi,
        use_posterior=self.use_posterior,
        use_optimal_frequency=self.use_optimal_frequency,
        max_frequency=self.max_frequency,
        start_date=start_date_str,
        end_date=end_date_str,
        gtol=0.0001,
        round_factor=round_factor,
        optimal_frequency=optimal_frequency,
        selected_geos=None,
        selected_times=selected_times,
    )

  def _create_grids(
      self,
      *,
      spend_bound_lower: np.ndarray,
      spend_bound_upper: np.ndarray,
      step_size: int,
      spend: np.ndarray | None = None,
      selected_times: Sequence[str] | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Creates spend and incremental outcome grids from weekly grid."""
    n_grid_rows = int(
        (np.max(np.subtract(spend_bound_upper, spend_bound_lower)) // step_size)
        + 1
    )
    n_grid_columns = len(self.channels)

    spend_grid = np.full([n_grid_rows, n_grid_columns], np.nan)
    for i, (lower_bound, upper_bound) in enumerate(
        zip(spend_bound_lower, spend_bound_upper)
    ):
      spend_grid_m = np.arange(
          lower_bound,
          upper_bound + step_size,
          step_size,
      )
      spend_grid[: len(spend_grid_m), i] = spend_grid_m

    incremental_outcome_grid = np.full([n_grid_rows, n_grid_columns], np.nan)

    if selected_times is None:
      selected_times = self.time

    _, rf_outcomes = self._resolve_rf_outcomes(selected_times)
    rf_outcome_by_channel = dict(zip(self.rf_channels, rf_outcomes))

    nonoptimized_spend = spend if spend is not None else self.nonoptimized_spend
    if isinstance(nonoptimized_spend, xr.DataArray):
      nonoptimized_spend = nonoptimized_spend.sum(dim=c.TIME).data
    for i, (channel, channel_spend) in enumerate(
        zip(self.channels, nonoptimized_spend)
    ):
      spend_column = spend_grid[:, i]
      valid_mask = ~np.isnan(spend_column)
      if channel_spend == 0:
        incremental_outcome_grid[valid_mask, i] = 0.0
        continue

      multipliers = spend_column[valid_mask] / channel_spend

      if channel in rf_outcome_by_channel:
        # RF outcome is linear in spend at a fixed frequency.
        incremental_outcome_grid[valid_mask, i] = (
            multipliers * rf_outcome_by_channel[channel]
        )
        continue

      channel_grid = self.incremental_outcome.sel({c.CHANNEL: channel}).dropna(  # pyrefly: ignore[missing-attribute]
          dim=c.SPEND_MULTIPLIER, how='all'
      )
      channel_mults = channel_grid[c.SPEND_MULTIPLIER].values
      week_mask = np.isin(channel_grid[c.TIME].values, selected_times)
      summed_outcomes = (
          channel_grid.isel({c.TIME: week_mask})
          .sum(dim=c.TIME)
          .transpose(c.SPEND_MULTIPLIER)
          .values
      )

      # Linear interpolation lookup.
      incremental_outcome_grid[valid_mask, i] = np.interp(
          multipliers, channel_mults, summed_outcomes
      )

    if self.n_rf_channels > 0:
      incremental_outcome_grid = backend.stabilize_rf_roi_grid(
          spend_grid, incremental_outcome_grid, self.n_rf_channels
      )

    return spend_grid, incremental_outcome_grid

  @classmethod
  def combine(
      cls, grids: Sequence['WeeklyOptimizationGrid']
  ) -> 'WeeklyOptimizationGrid':
    """Combines a sequence of WeeklyOptimizationGrid objects into one grid by concatenating dates.

    Args:
      grids: A sequence of WeeklyOptimizationGrid objects to be combined.

    Returns:
      A new WeeklyOptimizationGrid object with concatenated dates.

    Raises:
      ValueError: If the grids sequence is empty, or if the grids have
        incompatible attributes (e.g. different channels, spend multipliers, or
        scalar parameters), or if there are overlapping dates, or if the grid
        dates are not continuous.
    """
    if not grids:
      raise ValueError('The grids sequence must not be empty.')

    if len(grids) == 1:
      return grids[0]

    first = grids[0]
    for i, grid in enumerate(grids[1:], start=1):
      if grid.use_kpi != first.use_kpi:
        raise ValueError(f'Grid at index {i} has a different use_kpi value.')
      if grid.use_posterior != first.use_posterior:
        raise ValueError(
            f'Grid at index {i} has a different use_posterior value.'
        )
      if grid.multiplier_step != first.multiplier_step:
        raise ValueError(
            f'Grid at index {i} has a different multiplier_step value.'
        )
      if grid.max_budget_percent_decrease != first.max_budget_percent_decrease:
        raise ValueError(
            f'Grid at index {i} has a different max_budget_percent_decrease'
            ' value.'
        )
      if grid.max_budget_percent_increase != first.max_budget_percent_increase:
        raise ValueError(
            f'Grid at index {i} has a different max_budget_percent_increase'
            ' value.'
        )
      if grid.max_constraint_variation != first.max_constraint_variation:
        raise ValueError(
            f'Grid at index {i} has a different max_constraint_variation value.'
        )
      if grid.use_optimal_frequency != first.use_optimal_frequency:
        raise ValueError(
            f'Grid at index {i} has a different use_optimal_frequency value.'
        )
      if grid.max_frequency != first.max_frequency:
        raise ValueError(
            f'Grid at index {i} has a different max_frequency value.'
        )
      if (grid.rf_incremental_outcome is None) != (
          first.rf_incremental_outcome is None
      ):
        raise ValueError(
            f'Grid at index {i} has different rf_incremental_outcome presence.'
        )
      if (
          grid.rf_incremental_outcome is not None
          and first.rf_incremental_outcome is not None
          and not np.array_equal(
              grid.rf_incremental_outcome[c.FREQUENCY].values,
              first.rf_incremental_outcome[c.FREQUENCY].values,
              equal_nan=True,
          )
      ):
        raise ValueError(f'Grid at index {i} has different frequency grids.')
      if (grid.incremental_outcome is None) != (
          first.incremental_outcome is None
      ):
        raise ValueError(
            f'Grid at index {i} has different incremental_outcome presence.'
        )
      if not np.array_equal(grid.channels, first.channels):
        raise ValueError(f'Grid at index {i} has different channels.')
      if (
          grid.incremental_outcome is not None
          and first.incremental_outcome is not None
          and not np.array_equal(
              grid.incremental_outcome[c.SPEND_MULTIPLIER].values,
              first.incremental_outcome[c.SPEND_MULTIPLIER].values,
          )
      ):
        raise ValueError(f'Grid at index {i} has different spend multipliers.')

    def concat_over_time(
        arrays: Sequence[xr.DataArray | None],
    ) -> xr.DataArray | None:
      if arrays[0] is None:
        return None
      return xr.concat([a for a in arrays if a is not None], dim=c.TIME)

    combined_outcome = concat_over_time([g.incremental_outcome for g in grids])
    combined_rf_outcome = concat_over_time(
        [g.rf_incremental_outcome for g in grids]
    )
    combined_spend = xr.concat(
        [g.nonoptimized_spend for g in grids], dim=c.TIME
    )

    # Check for duplicate dates.
    combined_times = combined_spend[c.TIME].values
    if len(combined_times) != len(set(combined_times)):
      raise ValueError('Combined grids contain duplicate dates.')

    # Sort by time dimension to ensure chronological order.
    sorted_indices = np.argsort(combined_times)
    if combined_outcome is not None:
      combined_outcome = combined_outcome.isel({c.TIME: sorted_indices})
    if combined_rf_outcome is not None:
      combined_rf_outcome = combined_rf_outcome.isel({c.TIME: sorted_indices})
    combined_spend = combined_spend.isel({c.TIME: sorted_indices})

    # Check that dates form a contiguous weekly period.
    sorted_times = combined_spend[c.TIME].values
    sorted_dates = [tc.normalize_date(t) for t in sorted_times]
    for i in range(len(sorted_dates) - 1):
      if (sorted_dates[i + 1] - sorted_dates[i]).days != 7:
        raise ValueError(
            'Combined grids do not form a contiguous weekly period. Gap'
            f' detected between {sorted_times[i]} and {sorted_times[i+1]}.'
        )

    return cls(
        incremental_outcome=combined_outcome,
        nonoptimized_spend=combined_spend,
        use_kpi=first.use_kpi,
        use_posterior=first.use_posterior,
        multiplier_step=first.multiplier_step,
        max_budget_percent_decrease=first.max_budget_percent_decrease,
        max_budget_percent_increase=first.max_budget_percent_increase,
        max_constraint_variation=first.max_constraint_variation,
        use_optimal_frequency=first.use_optimal_frequency,
        max_frequency=first.max_frequency,
        rf_incremental_outcome=combined_rf_outcome,
    )
