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

from collections.abc import Sequence
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


@dataclasses.dataclass(frozen=True)
class WeeklyOptimizationGrid:
  """Weekly optimization grid information.

  Attributes:
    incremental_outcome: xr.DataArray of shape `(n_channels,
      n_spend_multipliers, n_times)` containing incremental outcome for all paid
      channels.
    nonoptimized_spend: xr.DataArray of shape `(n_paid_channels, n_times)`
      containing non-aggregated spend allocation for all paid channels.
    use_kpi: Whether using generic KPI or revenue.
    use_posterior: Whether posterior distributions were used, or prior.
    multiplier_step: Multiplier step size for the spend multiplier.
    max_budget_percent_decrease: Maximum percentage decrease in budget allowed.
    max_budget_percent_increase: Maximum percentage increase in budget allowed.
    max_constraint_variation: Maximum constraint variation allowed.
    n_rf_channels: Number of reach and frequency channels in the grid.
    use_optimal_frequency: Whether optimal frequency was used.
    max_frequency: Maximum frequency value used for optimal frequency.
    optimal_frequency: Optional ndarray of optimal frequency per RF channel.
  """

  incremental_outcome: xr.DataArray
  nonoptimized_spend: xr.DataArray
  use_kpi: bool
  use_posterior: bool
  multiplier_step: float
  max_budget_percent_decrease: float
  max_budget_percent_increase: float
  max_constraint_variation: float
  n_rf_channels: int = 0
  use_optimal_frequency: bool = True
  max_frequency: float | None = None
  optimal_frequency: np.ndarray | None = None

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
  ) -> 'WeeklyOptimizationGrid':
    """Builds a weekly optimization grid using vectorized backend calculations.

    This method pre-calculates weekly incremental outcomes across MCMC draws
    and chains over a discretized spend multiplier grid. This pre-computed grid
    can then be passed to `BudgetOptimizer.optimize`, enabling fast linear
    interpolation lookups during optimization iterations.

    Args:
      analyzer: An `Analyzer` instance with a fitted model.
      new_data: An optional `DataTensors` container with optional counterfactual
        or future data tensors. Defaults to None.
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
      batch_size: Maximum number of spend multipliers to process in each batch
        to avoid memory exhaustion. Defaults to 10.
      use_optimal_frequency: Whether to compute and use optimal frequency for RF
        channels during grid creation. Defaults to True.
      max_frequency: Maximum frequency value used for optimal frequency grid.
        Defaults to None.

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
    channels = model_context.input_data.get_all_paid_channels()
    all_times = np.asarray(filled_data.time).astype(str).tolist()

    selected_times_opt = optimizer._expand_selected_times(  # pylint: disable=protected-access
        model_context=model_context,
        start_date=start_date,
        end_date=end_date,
        new_data=filled_data,
        return_flexible_str=True,
    )
    if selected_times_opt is not None:
      selected_times_list = [
          t.strftime(c.DATE_FORMAT) if not isinstance(t, str) else t  # pyrefly: ignore[missing-attribute]
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
      if use_optimal_frequency:
        opt_freq_data = analyzer_module.DataTensors(
            rf_impressions=filled_data.reach * filled_data.frequency,  # pyrefly: ignore[unsupported-operation]
            rf_spend=filled_data.rf_spend,
            revenue_per_kpi=filled_data.revenue_per_kpi,
        )
        opt_freq_ds = analyzer.optimal_freq(
            new_data=opt_freq_data,
            use_posterior=use_posterior,
            use_kpi=use_kpi,
            max_frequency=max_frequency,
        )
        optimal_frequency_tensor = backend.to_tensor(
            opt_freq_ds.optimal_frequency,
            dtype=backend.float_dtype,
        )
        optimal_frequency = np.asarray(
            opt_freq_ds.optimal_frequency.data, dtype=float
        )
        frequency_base = to_float(
            backend.ones_like(filled_data.frequency) * optimal_frequency_tensor  # pyrefly: ignore[bad-argument-type]
        )
        reach_at_opt_freq = backend.divide_no_nan(
            filled_data.reach * filled_data.frequency,  # pyrefly: ignore[unsupported-operation]
            frequency_base,
        )
        if model_context.rf_tensors.reach_transformer is None:
          reach_base_scaled = to_float(reach_at_opt_freq)
        else:
          reach_base_scaled = to_float(
              model_context.rf_tensors.reach_transformer.forward(
                  reach_at_opt_freq
              )
          )
      else:
        optimal_frequency = None
        frequency_base = to_float(filled_data.frequency)  # pyrefly: ignore[bad-argument-type]
        if model_context.rf_tensors.reach_transformer is None:
          reach_base_scaled = to_float(model_context.rf_tensors.reach_scaled)  # pyrefly: ignore[bad-argument-type]
        else:
          reach_base_scaled = to_float(
              model_context.rf_tensors.reach_transformer.forward(
                  filled_data.reach
              )
          )
      alpha_rf = to_float(inf_data.alpha_rf)
      ec_rf = to_float(inf_data.ec_rf)
      slope_rf = to_float(inf_data.slope_rf)
      beta_grf = to_float(inf_data.beta_grf)
      decay_rf = model_context.adstock_decay_spec.rf
      sat_rf = model_context.saturation_spec.rf
    else:
      optimal_frequency = None
      reach_base_scaled = None
      frequency_base = None
      alpha_rf = None
      ec_rf = None
      slope_rf = None
      beta_grf = None
      decay_rf = None
      sat_rf = None

    all_multipliers_array = backend.to_tensor(
        unique_multipliers, dtype=backend.float_dtype
    )
    all_outcomes = []
    multiplier_batch_size = max(1, batch_size)

    for i in range(0, len(all_multipliers_array), multiplier_batch_size):  # pyrefly: ignore[bad-argument-type]
      batch = all_multipliers_array[i : i + multiplier_batch_size]
      batch_outcomes = cls._compute_batch(
          batch,
          media_base_scaled,
          alpha_m,
          ec_m,
          slope_m,
          beta_gm,
          reach_base_scaled,
          frequency_base,
          alpha_rf,
          ec_rf,
          slope_rf,
          beta_grf,
          revenue_per_kpi,
          time_indices=time_indices,
          eqs=eqs,
          decay_m=decay_m,
          sat_m=sat_m,
          decay_rf=decay_rf,
          sat_rf=sat_rf,
          n_times=n_times,
          kpi_transformer=kpi_transformer,
          use_kpi=use_kpi,
      )
      all_outcomes.append(np.asarray(batch_outcomes))

    outcomes = np.concatenate(all_outcomes, axis=0)

    final_outcomes = np.transpose(outcomes, (2, 0, 1))

    incremental_outcome = xr.DataArray(
        final_outcomes,
        coords={
            c.CHANNEL: channels,
            c.SPEND_MULTIPLIER: unique_multipliers,
            c.TIME: selected_times_list,
        },
        dims=[c.CHANNEL, c.SPEND_MULTIPLIER, c.TIME],
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
        n_rf_channels=model_context.n_rf_channels,
        use_optimal_frequency=use_optimal_frequency,
        max_frequency=max_frequency,
        optimal_frequency=optimal_frequency,
    )

  @classmethod
  @backend.function(
      jit_compile=True,
      static_argnames=[
          'eqs',
          'decay_m',
          'sat_m',
          'decay_rf',
          'sat_rf',
          'n_times',
          'kpi_transformer',
          'use_kpi',
      ],
  )
  def _compute_batch(
      cls,
      multiplier_batch: backend.Tensor,
      media_base_scaled: backend.Tensor | None,
      alpha_m: backend.Tensor | None,
      ec_m: backend.Tensor | None,
      slope_m: backend.Tensor | None,
      beta_gm: backend.Tensor | None,
      reach_base_scaled: backend.Tensor | None,
      frequency_base: backend.Tensor | None,
      alpha_rf: backend.Tensor | None,
      ec_rf: backend.Tensor | None,
      slope_rf: backend.Tensor | None,
      beta_grf: backend.Tensor | None,
      revenue_per_kpi: backend.Tensor,
      time_indices: backend.Tensor | None,
      eqs: Any,
      decay_m: Any,
      sat_m: Any,
      decay_rf: Any,
      sat_rf: Any,
      n_times: int,
      kpi_transformer: Any,
      use_kpi: bool,
  ) -> backend.Tensor:
    """Computes incremental outcome for a batch of spend multipliers."""

    def _compute_kpi_for_multiplier(
        multiplier: backend.Tensor,
    ) -> backend.Tensor:
      multiplier_float = backend.cast(multiplier, backend.float_dtype)

      diffs = []
      betas = []
      if media_base_scaled is not None:
        media_t1 = eqs.adstock_hill_media(
            media=media_base_scaled * multiplier_float,  # pyrefly: ignore[unsupported-operation]
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
            decay_functions=decay_m,
            saturation_spec=sat_m,
            n_times_output=n_times,
        )
        media_t0 = eqs.adstock_hill_media(
            media=media_base_scaled * 0.0,  # pyrefly: ignore[unsupported-operation]
            alpha=alpha_m,
            ec=ec_m,
            slope=slope_m,
            decay_functions=decay_m,
            saturation_spec=sat_m,
            n_times_output=n_times,
        )
        diffs.append(media_t1 - media_t0)
        betas.append(beta_gm)

      if reach_base_scaled is not None:
        rf_t1 = eqs.adstock_hill_rf(
            reach=reach_base_scaled * multiplier_float,  # pyrefly: ignore[unsupported-operation]
            frequency=frequency_base,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
            decay_functions=decay_rf,
            saturation_spec=sat_rf,
            n_times_output=n_times,
        )
        rf_t0 = eqs.adstock_hill_rf(
            reach=reach_base_scaled * 0.0,  # pyrefly: ignore[unsupported-operation]
            frequency=frequency_base,
            alpha=alpha_rf,
            ec=ec_rf,
            slope=slope_rf,
            decay_functions=decay_rf,
            saturation_spec=sat_rf,
            n_times_output=n_times,
        )
        diffs.append(rf_t1 - rf_t0)
        betas.append(beta_grf)

      if len(diffs) > 1:
        media_diff = backend.concatenate(diffs, axis=-1)
        combined_beta = backend.concatenate(betas, axis=-1)
      else:
        media_diff = diffs[0]
        combined_beta = betas[0]

      if time_indices is not None:
        media_diff = backend.gather(media_diff, time_indices, axis=3)

      incremental_kpi = backend.einsum(
          '...gtm,...gm->...gtm', media_diff, combined_beta
      )
      # Inverse transform incremental KPI to natural scale.
      # We calculate the difference between the inverse transformed incremental
      # KPI and the inverse transformed zero to remove any intercept/offset
      # introduced by the KPI transformer, obtaining the uncentered incremental
      # KPI on the natural scale.
      transformed_kpi = kpi_transformer.inverse(
          backend.einsum('...m->m...', incremental_kpi)
      )
      transformed_zero = kpi_transformer.inverse(
          backend.zeros_like(transformed_kpi)
      )
      incremental_kpi_natural = backend.einsum(
          'm...->...m', transformed_kpi - transformed_zero
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

    return backend.vectorized_map(_compute_kpi_for_multiplier, multiplier_batch)

  @property
  def channels(self) -> list[str]:
    """The spend channels in the weekly grid."""
    return self.incremental_outcome.channel.data.tolist()

  @property
  def time(self) -> list[str]:
    """The spend times in the weekly grid."""
    return self.incremental_outcome.time.data.tolist()

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
    rounded_hist_spend = np.round(hist_spend, round_factor).astype(int)
    for i, (channel, channel_spend) in enumerate(
        zip(self.channels, rounded_hist_spend)
    ):
      if channel_spend == 0:
        continue

      channel_grid = self.incremental_outcome.sel({c.CHANNEL: channel}).dropna(
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
        optimal_frequency=self.optimal_frequency,
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
      selected_times: Sequence[str] | Sequence[bool] | None = None,
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

    week_indices = np.where(np.isin(self.time, selected_times))[0]  # pyrefly: ignore[bad-argument-type]

    nonoptimized_spend = spend if spend is not None else self.nonoptimized_spend
    if isinstance(nonoptimized_spend, xr.DataArray):
      nonoptimized_spend = nonoptimized_spend.sum(dim=c.TIME).data
    for i, (channel, channel_spend) in enumerate(
        zip(self.channels, nonoptimized_spend)
    ):
      if channel_spend == 0:
        incremental_outcome_grid[:, i] = 0.0
        continue

      spend_column = spend_grid[:, i]
      valid_mask = ~np.isnan(spend_column)
      multipliers = spend_column[valid_mask] / channel_spend

      channel_grid = self.incremental_outcome.sel({c.CHANNEL: channel}).dropna(
          dim=c.SPEND_MULTIPLIER, how='all'
      )
      channel_mults = channel_grid[c.SPEND_MULTIPLIER].values
      channel_outcomes = channel_grid.values

      summed_outcomes = np.sum(channel_outcomes[:, week_indices], axis=1)

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
      if grid.n_rf_channels != first.n_rf_channels:
        raise ValueError(
            f'Grid at index {i} has a different n_rf_channels value.'
        )
      if grid.use_optimal_frequency != first.use_optimal_frequency:
        raise ValueError(
            f'Grid at index {i} has a different use_optimal_frequency value.'
        )
      if grid.max_frequency != first.max_frequency:
        raise ValueError(
            f'Grid at index {i} has a different max_frequency value.'
        )
      if (grid.optimal_frequency is None) != (first.optimal_frequency is None):
        raise ValueError(
            f'Grid at index {i} has different optimal_frequency values.'
        )
      if grid.optimal_frequency is not None and not np.array_equal(
          grid.optimal_frequency, first.optimal_frequency  # pyrefly: ignore[bad-argument-type]
      ):
        raise ValueError(
            f'Grid at index {i} has different optimal_frequency values.'
        )
      if not np.array_equal(grid.channels, first.channels):
        raise ValueError(f'Grid at index {i} has different channels.')
      if not np.array_equal(
          grid.incremental_outcome[c.SPEND_MULTIPLIER].values,
          first.incremental_outcome[c.SPEND_MULTIPLIER].values,
      ):
        raise ValueError(f'Grid at index {i} has different spend multipliers.')

    combined_outcome = xr.concat(
        [g.incremental_outcome for g in grids], dim=c.TIME
    )
    combined_spend = xr.concat(
        [g.nonoptimized_spend for g in grids], dim=c.TIME
    )

    # Check for duplicate dates.
    combined_times = combined_outcome[c.TIME].values
    if len(combined_times) != len(set(combined_times)):
      raise ValueError('Combined grids contain duplicate dates.')

    # Sort by time dimension to ensure chronological order.
    sorted_indices = np.argsort(combined_times)
    combined_outcome = combined_outcome.isel({c.TIME: sorted_indices})
    combined_spend = combined_spend.isel({c.TIME: sorted_indices})

    # Check that dates form a contiguous weekly period.
    sorted_times = combined_outcome[c.TIME].values
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
        n_rf_channels=first.n_rf_channels,
        use_optimal_frequency=first.use_optimal_frequency,
        max_frequency=first.max_frequency,
        optimal_frequency=first.optimal_frequency,
    )
