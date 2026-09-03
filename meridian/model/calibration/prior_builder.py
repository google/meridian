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

"""Builds a joint prior distribution based on incrementality experiment results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import datetime
import functools
from typing import Any, TYPE_CHECKING
import warnings

from meridian import backend
from meridian import constants
from meridian.data import input_data
from meridian.data import time_coordinates as tc
from meridian.model.calibration import base
from meridian.model.calibration import constants as calibration_constants
from meridian.model.calibration.adapters import experiment as experiment_adapter
from meridian.model.calibration.adapters import meridian_geox as geox_adapter
import numpy as np


if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=g-bad-import-order
  from meridian_geox import api as geox_api
  # pylint: enable=g-import-not-at-top
  # pylint: enable=g-bad-import-order

  _GeoXResults = geox_api.AnalysisResult | Sequence[geox_api.AnalysisResult]
else:
  _GeoXResults = Any
_KpiTypes = str | Sequence[str]
_Floats = float | Sequence[float]
_OptionalFloats = _Floats | None
_AdstockDecaySpec = str | Mapping[str, str]
_Alpha = float | Mapping[str, float]
_PriorDistribution = (
    backend.tfd.Distribution | Mapping[str, backend.tfd.Distribution] | None
)
_SingleDate = str | datetime.datetime | datetime.date | np.datetime64
_Dates = _SingleDate | Sequence[_SingleDate]


def _validate_adstock_decay_spec(
    adstock_decay_spec: _AdstockDecaySpec,
    valid_channels: frozenset[str],
) -> None:
  """Validates that adstock_decay_spec is a string or mapping of strings."""
  if isinstance(adstock_decay_spec, str):
    if adstock_decay_spec not in constants.ADSTOCK_DECAY_FUNCTIONS:
      raise ValueError(
          f"Invalid 'adstock_decay_spec': '{adstock_decay_spec}'. Valid"
          f" options are: {sorted(constants.ADSTOCK_DECAY_FUNCTIONS)}."
      )
  elif isinstance(adstock_decay_spec, Mapping):
    invalid_channels = set(adstock_decay_spec.keys()) - valid_channels
    if invalid_channels:
      raise ValueError(
          "Invalid channels in 'adstock_decay_spec' mapping:"
          f" {sorted(invalid_channels)}. Valid channels are:"
          f" {sorted(valid_channels)}."
      )
    for spec in adstock_decay_spec.values():
      if spec not in constants.ADSTOCK_DECAY_FUNCTIONS:
        raise ValueError(
            f"Invalid 'adstock_decay_spec': '{spec}'. Valid options are:"
            f" {sorted(constants.ADSTOCK_DECAY_FUNCTIONS)}."
        )
  else:
    raise TypeError(
        "adstock_decay_spec must be either a string or a Mapping of channel"
        f" name to string. Got type: {type(adstock_decay_spec)}."
    )


def _validate_alpha(alpha: _Alpha, valid_channels: frozenset[str]) -> None:
  """Validates that alpha is either a valid float, or a mapping of floats."""
  if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(
          f"Invalid 'alpha': {alpha}. Alpha must be between 0.0 and 1.0."
      )
  elif isinstance(alpha, Mapping):
    invalid_channels = set(alpha.keys()) - valid_channels
    if invalid_channels:
      raise ValueError(
          f"Invalid channels in 'alpha' mapping: {sorted(invalid_channels)}. "
          f"Valid channels are: {sorted(valid_channels)}."
      )
    for val in alpha.values():
      if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise TypeError(f"Alpha values must be numeric. Got type: {type(val)}.")
      if not 0.0 <= val <= 1.0:
        raise ValueError(
            f"Invalid 'alpha': {val}. Alpha must be between 0.0 and 1.0."
        )
  else:
    raise TypeError(
        "alpha must be either a float or a Mapping of channel name to float."
        f" Got type: {type(alpha)}."
    )


def _coerce_required_floats(values: _Floats) -> Sequence[float]:
  """Coerces values to a read-only sequence of floats.

  Args:
    values: A float, int, or sequence thereof.

  Returns:
    A read-only sequence of float values.

  Raises:
    TypeError: If any elements are None, boolean, or not float/int values.
  """
  if isinstance(values, (float, int)) and not isinstance(values, bool):
    return [float(values)]

  if isinstance(values, (str, bool)) or not isinstance(values, Iterable):
    raise TypeError(
        "Expected float, int, or sequence thereof, but got type:"
        f" {type(values)}"
    )

  coerced = []
  for v in values:
    if v is None or isinstance(v, bool):
      raise TypeError(
          "None/Booleans are not allowed for required float fields."
      )
    if not isinstance(v, (float, int)):
      raise TypeError(f"Expected float or int, but got type: {type(v)}")
    coerced.append(float(v))
  return coerced


def _coerce_optional_floats(
    values: _OptionalFloats,
) -> Sequence[float | None]:
  """Coerces values to a read-only sequence of float or None values.

  Args:
    values: A float, int, None, or sequence thereof.

  Returns:
    A read-only sequence of float or None values.

  Raises:
    TypeError: If any elements are boolean or not float/int/None values.
  """
  if values is None:
    return []

  if isinstance(values, (float, int)) and not isinstance(values, bool):
    return [float(values)]

  if isinstance(values, (str, bool)) or not isinstance(values, Iterable):
    raise TypeError(
        "Expected float, int, None, or sequence thereof, but got type:"
        f" {type(values)}"
    )

  coerced = []
  for v in values:
    if v is None:
      coerced.append(None)
      continue
    if isinstance(v, bool):
      raise TypeError("Booleans are not allowed for optional float fields.")
    if not isinstance(v, (float, int)):
      raise TypeError(f"Expected float, int, or None, but got type: {type(v)}")
    coerced.append(float(v))
  return coerced


def _coerce_kpi_types(experiment_kpi_types: _KpiTypes) -> Sequence[str]:
  """Coerces experiment_kpi_types to a read-only sequence of strings.

  Args:
    experiment_kpi_types: A string or sequence of strings.

  Returns:
    A read-only sequence of strings.

  Raises:
    ValueError: If `experiment_kpi_types` contains invalid KPI types.
  """
  if experiment_kpi_types is None:
    raise ValueError(
        "`experiment_kpi_types` must be passed as a string or sequence of"
        " strings for all experiments."
    )

  coerced = (
      [experiment_kpi_types]
      if isinstance(experiment_kpi_types, str)
      else list(experiment_kpi_types)
  )

  valid_types = {constants.REVENUE, constants.NON_REVENUE}
  for kpi in coerced:
    if kpi not in valid_types:
      raise ValueError(
          f"Invalid 'experiment_kpi_types': '{kpi}'. Must be one of:"
          f" {sorted(valid_types)}"
      )

  return coerced


def _coerce_required_dates(
    dates: _Dates,
) -> Sequence[datetime.date]:
  """Coerces dates to a read-only sequence of `datetime.date` values.

  Args:
    dates: A single `datetime.date`, `datetime.datetime`, `np.datetime64`,
      string in `'YYYY-MM-DD'` format, or a sequence thereof.

  Returns:
    A read-only sequence of `datetime.date` values.

  Raises:
    ValueError: If `dates` is None, or any elements cannot be parsed or coerced
      to a `datetime.date`.
  """
  dates_seq = (
      dates
      if isinstance(dates, Sequence) and not isinstance(dates, (str, bytes))
      else [dates]
  )
  return [tc.normalize_date(d) for d in dates_seq]


def _validate_sequence_lengths(
    lengths: Mapping[str, int],
) -> None:
  """Validates that all sequence arguments are of matching length.

  Args:
    lengths: A mapping from argument name strings to their sequence lengths.

  Raises:
    ValueError: If the sequence arguments do not all have the same length.
  """
  if len(set(lengths.values())) > 1:
    raise ValueError(
        f"Sequence arguments must have the same length. Got lengths: {lengths}"
    )


def _wrap_as_calibrated(
    dist_list: Sequence[backend.tfd.Distribution],
    *,
    channels: Sequence[str],
    calibrated_channels: set[str],
    calibration_outputs_map: Mapping[str, base.CalibrationOutput],
    name: str,
) -> base.CalibratedDistribution:
  """Wraps univariate distributions into a joint CalibratedDistribution.

  Args:
    dist_list: A sequence of distributions to wrap.
    channels: The sequence of all channel names in the Meridian model.
    calibrated_channels: The set of channel names with registered experiments
      undergoing prior calibration.
    calibration_outputs_map: Mapping from channel name to calibration outputs.
    name: The name prefix for the CalibratedDistribution.

  Returns:
    A `base.CalibratedDistribution` object.
  """
  is_calibrated = [ch in calibrated_channels for ch in channels]
  calibration_outputs = [
      calibration_outputs_map.get(ch) if ch in calibrated_channels else None
      for ch in channels
  ]
  return base.CalibratedDistribution(
      dist_list,
      is_calibrated=is_calibrated,
      calibration_outputs=calibration_outputs,
      name=name,
  )


def _get_model_duration_days(
    time_coords: tc.TimeCoordinates,
) -> int:
  """Calculates the total duration of the modeled period in days."""
  all_dates = time_coords.all_dates
  model_duration_days = (
      all_dates[-1] - all_dates[0]
  ).days + time_coords.interval_days

  if model_duration_days <= 0:
    raise ValueError(
        f"Model duration in days must be positive. Got: {model_duration_days}."
    )
  return model_duration_days


class CalibrationBuilder:
  """Builder for collecting incrementality experiment results and building ROI prior distributions.

  This builder collects incrementality experiment results for media and reach &
  frequency channels, calibrates them, and outputs ROI priors for media and
  reach & frequency channels as `base.CalibratedDistribution` objects.
  """

  def __init__(
      self,
      data: input_data.InputData,
      *,
      custom_prior: _PriorDistribution = None,
      baseline_prior: _PriorDistribution = None,
      adstock_decay_spec: _AdstockDecaySpec = constants.GEOMETRIC_DECAY,
      alpha: _Alpha = calibration_constants.DEFAULT_ALPHA,
      max_lag: int = constants.DEFAULT_MAX_LAG,
      registry: base.CalibrationRegistry | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
      data: `meridian.data.input_data.InputData` object with Meridian dataset.
      custom_prior: Mapping of channel name to ROI prior distribution for media
        and reach & frequency channels that aren't undergoing calibration, or a
        single custom prior for all uncalibrated channels. Entries are ignored
        if the channel's prior is calibrated using experiment results via
        `CalibrationBuilder`. Channels that aren't calibrated using
        `CalibrationBuilder` and aren't included in this Mapping use the default
        ROI prior of `LogNormal(0.2, 0.9)`. Default is `None`.
      baseline_prior: Mapping of channel name to baseline prior distribution, or
        a single baseline prior for all calibrated channels. If provided, the
        baseline prior will regularize the incrementality experiment results.
        Default is None. If None, an improper uniform distribution over the
        positive real numbers is used.
      adstock_decay_spec: Adstock decay specification used in duration
        adjustments. This can be either `'geometric'` or `'binomial'`, or a
        mapping from channel name to the decay function string. Default is
        `'geometric'`. This value should be consistent with the adstock decay
        spec defined in the Meridian model's `ModelSpec` object.
      alpha: The decay rate parameter (between 0 and 1 inclusive) used for
        duration adjustments. It determines the proportion of the total media
        effect captured during the experiment window. This can be either a float
        or a mapping from channel name to float. This value should be the same
        as the point estimate of the alpha prior for each corresponding media
        and reach & frequency channel. Default is `0.5`.
      max_lag: Maximum lag value used in duration adjustments. This value should
        be consistent with the Meridian model's `max_lag` value set in the
        `ModelSpec` object. Default is `8`.
      registry: The `CalibrationRegistry` to use. Primarily for testing. Default
        is a new `CalibrationRegistry` instance.

    Raises:
      TypeError: If `adstock_decay_spec` is not a string or mapping, or if
        `alpha` values or mapping values are not numeric.
      ValueError: If `adstock_decay_spec` contains an invalid decay function or
        if `alpha` values are not within the closed interval `[0.0, 1.0]`.
    """
    self._input_data = data

    _validate_adstock_decay_spec(adstock_decay_spec, self._valid_channels)
    _validate_alpha(alpha, self._valid_channels)

    if (
        not isinstance(max_lag, int)
        or isinstance(max_lag, bool)
        or max_lag < 0
    ):
      raise ValueError(
          f"'max_lag' must be a non-negative integer. Got: {max_lag}"
      )
    self._custom_prior = custom_prior
    self._baseline_prior = baseline_prior
    self._adstock_decay_spec = adstock_decay_spec
    self._alpha = alpha
    self._max_lag = max_lag

    self._registry = registry or base.CalibrationRegistry()
    self._container_by_channel: dict[str, base.CalibrationInput] = {}
    self._total_spend_by_channel: dict[str, float] = {}

  @functools.cached_property
  def _resolved_adstock_decay_specs(self) -> Mapping[str, str]:
    """A cached mapping of channel name to resolved adstock decay spec."""
    adstock_decay_spec = self._adstock_decay_spec
    if isinstance(adstock_decay_spec, str):
      return {channel: adstock_decay_spec for channel in self._valid_channels}
    return {
        channel: adstock_decay_spec.get(channel, constants.GEOMETRIC_DECAY)
        for channel in self._valid_channels
    }

  @functools.cached_property
  def _resolved_alphas(self) -> Mapping[str, float]:
    """A cached mapping of channel name to resolved alpha."""
    alpha = self._alpha
    if isinstance(alpha, (int, float)):
      alpha_val = float(alpha)
      return {channel: alpha_val for channel in self._valid_channels}
    return {
        channel: float(alpha.get(channel, calibration_constants.DEFAULT_ALPHA))
        for channel in self._valid_channels
    }

  @functools.cached_property
  def _resolved_custom_priors(self) -> Mapping[str, backend.tfd.Distribution]:
    """A cached mapping of channel name to resolved custom prior."""
    if self._custom_prior is None:
      return {}
    if isinstance(self._custom_prior, backend.tfd.Distribution):
      return {channel: self._custom_prior for channel in self._valid_channels}
    return {
        channel: prior
        for channel, prior in self._custom_prior.items()
        if channel in self._valid_channels
    }

  @functools.cached_property
  def _revenue_per_kpi(self) -> float | None:
    """The mean of `'revenue_per_kpi'`, cached on first access."""
    if self._input_data.revenue_per_kpi is not None:
      try:
        return float(self._input_data.revenue_per_kpi.mean())
      except (TypeError, ValueError, AttributeError):
        pass
    return None

  @functools.cached_property
  def _valid_channels(self) -> frozenset[str]:
    """A cached set of all valid paid media and RF channels."""
    return frozenset(self._input_data.get_all_paid_channels())

  def with_meridian_geox_experiment_result(
      self,
      channel_name: str,
      geox_results: _GeoXResults,
      experiment_kpi_types: _KpiTypes,
      *,
      point_estimate_adjustments: _OptionalFloats = None,
      standard_error_adjustments: _OptionalFloats = None,
  ) -> CalibrationBuilder:
    """Adds one or more Meridian GeoX experiment results to the specified channel.

    Extracts Meridian GeoX experiment information from one or more
    `meridian_geox.api.AnalysisResult` objects for translation into
    ROI priors.

    Calibration is supported for both revenue and non-revenue(such as
    conversions)-based incrementality experiments. However, calibration is only
    possible when the Meridian incremental outcome is in terms of revenue.
    Non-revenue incrementality experiment results are translated to the unitless
    Meridian ROI scale by using the `revenue_per_kpi` information available in
    Meridian(`meridian.data.input_data.InputData`). The `experiment_kpi_types`
    argument defines which experiments are of revenue and non-revenue type for
    translation.

    If `revenue_per_kpi` information is not available in Meridian,
    translation of non-revenue experiments is not supported. In this
    case, an error is raised.

    Note: When passing sequences for arguments (e.g. experiment_kpi_types,
    point_estimate_adjustments, standard_error_adjustments), elements at the
    same index must correspond to the same experiment.

    Args:
      channel_name: Name of the channel corresponding to `geox_results`.
      geox_results: A single or sequence of `meridian_geox.api.AnalysisResult`
        objects containing experiment-derived KPI estimates and metadata.
      experiment_kpi_types: A string or sequence of strings denoting whether the
        KPI of each experiment is of a `'revenue'` or `'non-revenue'` type.
      point_estimate_adjustments: A float (or sequence of floats for multiple
        experiments) representing an optional point estimate adjustment which is
        added to the computed point estimate adjustment for each corresponding
        experiment.
      standard_error_adjustments: A float (or sequence of floats for multiple
        experiments) representing an optional standard error adjustment which is
        added to the computed standard error adjustment for each corresponding
        experiment.

    Returns:
      The `CalibrationBuilder` instance.

    Raises:
      ImportError: If the 'meridian_geox' library is not installed.
      ValueError: If any of the following are true:
        - The channel name is not found in the model's paid channels.
        - The sequence arguments have mismatched lengths.
        - The experiment kpi_types are not compatible with the channel's KPI
          type or are not supported.
        - An experiment has `kpi_type` set to `'non-revenue'` but
          `revenue_per_kpi` is `None`.
        - An experiment result is statistically significantly negative.
        - An experiment result has a non-positive standard error.
      InvalidGeoXResultError: If any of the GeoX results are invalid or
        incompatible with calibration.
    """
    if not geox_adapter.HAS_GEOX:
      raise ImportError(
          "GeoX calibration is not available because the 'meridian_geox'"
          " library is not installed. Install it via pip: `pip install"
          " google-meridian[geox]`. You can use"
          " `with_incrementality_experiment_result()` alternatively."
      )

    self._validate_channel_name(channel_name)

    results_seq: Sequence[geox_api.AnalysisResult]
    if isinstance(geox_results, geox_adapter.geox_api.AnalysisResult):
      results_seq = [geox_results]
    elif isinstance(geox_results, Iterable):
      results_seq = list(geox_results)
    else:
      raise ValueError(
          "`geox_results` must be a single `AnalysisResult` or sequence."
          f" Got type: {type(geox_results)}"
      )

    kpi_types_seq = _coerce_kpi_types(experiment_kpi_types)

    lengths = {
        calibration_constants.GEOX_RESULTS: len(results_seq),
        calibration_constants.EXPERIMENT_KPI_TYPES: len(kpi_types_seq),
    }

    point_estimate_adjustments_seq = _coerce_optional_floats(
        point_estimate_adjustments
    )
    if point_estimate_adjustments is not None:
      lengths[calibration_constants.POINT_ESTIMATE_ADJUSTMENTS] = len(
          point_estimate_adjustments_seq
      )

    standard_error_adjustments_seq = _coerce_optional_floats(
        standard_error_adjustments
    )
    if standard_error_adjustments is not None:
      lengths[calibration_constants.STANDARD_ERROR_ADJUSTMENTS] = len(
          standard_error_adjustments_seq
      )

    _validate_sequence_lengths(lengths)

    for kpi_type in kpi_types_seq:
      self._validate_kpi_compatibility(channel_name, kpi_type)

    container = self._get_or_create_channel_container(channel_name)
    num_experiments = len(results_seq)
    point_estimate_adjustments_sequence = (
        point_estimate_adjustments_seq or [None] * num_experiments
    )
    standard_error_adjustments_sequence = (
        standard_error_adjustments_seq or [None] * num_experiments
    )

    for (
        result,
        kpi_type,
        gamma_adj,
        tau_adj,
    ) in zip(
        results_seq,
        kpi_types_seq,
        point_estimate_adjustments_sequence,
        standard_error_adjustments_sequence,
    ):
      config = geox_adapter.resolve_meridian_geox_source(
          result=result,
          kpi_type=kpi_type,
          revenue_per_kpi=self._revenue_per_kpi,
          point_estimate_adjustment=gamma_adj,
          standard_error_adjustment=tau_adj,
      )
      container.add_calibration_data(config)
    return self

  def with_incrementality_experiment_result(
      self,
      channel_name: str,
      *,
      point_estimates: _Floats,
      standard_errors: _Floats,
      experiment_kpi_types: _KpiTypes,
      experiment_total_spends: _Floats,
      experiment_start_dates: _Dates,
      experiment_end_dates: _Dates,
      point_estimate_adjustments: _OptionalFloats = None,
      standard_error_adjustments: _OptionalFloats = None,
  ) -> CalibrationBuilder:
    """Adds one or more incrementality experiment results for the specified channel.

    Calibration is supported for both revenue and non-revenue (such as
    conversions)-based incrementality experiments. However, calibration is only
    possible when the Meridian incremental outcome is in terms of revenue.
    Non-revenue incrementality experiment results are translated to the unitless
    Meridian ROI scale by using the `revenue_per_kpi` information available in
    Meridian (`meridian.data.input_data.InputData`). The `experiment_kpi_types`
    argument defines which experiments are of revenue and non-revenue type for
    translation.

    If `revenue_per_kpi` information is not available in Meridian,
    translation of non-revenue experiments is not supported. In this
    case, an error is raised.

    Note: When passing sequences for arguments (e.g. point_estimates,
    standard_errors, experiment_total_spends, experiment_kpi_types,
    experiment_start_dates, experiment_end_dates, point_estimate_adjustments,
    standard_error_adjustments), elements at the same index must correspond to
    the same experiment.

    Args:
      channel_name: Name of the channel corresponding to the incrementality
        experiment result.
      point_estimates: A float or sequence of floats representing ROI or IKPC
        point estimates of the incrementality experiment results.
      standard_errors: A float or sequence of floats representing the standard
        error of the incrementality experiment point estimates.
      experiment_kpi_types: A string or sequence of strings denoting whether the
        KPI of each experiment is of a `'revenue'` or `'non-revenue'` type.
      experiment_total_spends: A float or sequence of floats representing the
        typical total spend for the experiment duration for the channel. Used to
        apply spend adjustments when converting the associated experiment to the
        ROI prior.
      experiment_start_dates: A `datetime.date`, `datetime.datetime`,
        `np.datetime64`, string in `'YYYY-MM-DD'` format, or sequences thereof
        representing the start date of the experiment. Used to apply recency and
        duration adjustments when converting the associated experiment to the
        ROI prior.
      experiment_end_dates: A `datetime.date`, `datetime.datetime`,
        `np.datetime64`, string in `'YYYY-MM-DD'` format, or sequences thereof
        representing the end date of the experiment. Used to apply recency and
        duration adjustments when converting the associated experiment to the
        ROI prior.
      point_estimate_adjustments: A float (or sequence of floats for multiple
        experiments) representing an optional point estimate adjustment which is
        added to the computed point estimate adjustment for each corresponding
        experiment.
      standard_error_adjustments: A float (or sequence of floats for multiple
        experiments) representing an optional standard error adjustment which is
        added to the computed standard error adjustment for each corresponding
        experiment.

    Returns:
      The `CalibrationBuilder` instance.

    Raises:
      TypeError: If any of the required float arguments cannot be coerced due to
        invalid types.
      ValueError: If any of the following are true:
        - Any of the start or end dates are None or cannot be parsed or coerced
          to a `datetime.date`.
        - Any `standard_errors` are non-positive.
        - Any `experiment_total_spends` are not positive.
        - The sequence arguments have mismatched lengths.
        - The experiment start and end dates are not chronological.
        - The channel name is not found in the model's paid channels.
        - The experiment kpi_types are not compatible with the channel's KPI
          type or are not supported.
        - An experiment has statistically significantly negative lift.
        - An experiment's spend exceeds the channel's total spend.
    """
    self._validate_channel_name(channel_name)

    lengths = {}
    point_estimates_seq = _coerce_required_floats(point_estimates)
    lengths[calibration_constants.POINT_ESTIMATES] = len(point_estimates_seq)

    standard_errors_seq = _coerce_required_floats(standard_errors)
    lengths[calibration_constants.STANDARD_ERRORS] = len(standard_errors_seq)

    spends_seq = _coerce_required_floats(experiment_total_spends)
    lengths[calibration_constants.EXPERIMENT_TOTAL_SPENDS] = len(spends_seq)

    kpi_types_seq = _coerce_kpi_types(experiment_kpi_types)
    lengths[calibration_constants.EXPERIMENT_KPI_TYPES] = len(kpi_types_seq)

    start_dates_seq = _coerce_required_dates(experiment_start_dates)
    lengths[calibration_constants.EXPERIMENT_START_DATES] = len(start_dates_seq)

    end_dates_seq = _coerce_required_dates(experiment_end_dates)
    lengths[calibration_constants.EXPERIMENT_END_DATES] = len(end_dates_seq)

    point_estimate_adjustments_seq = _coerce_optional_floats(
        point_estimate_adjustments
    )
    if point_estimate_adjustments is not None:
      lengths[calibration_constants.POINT_ESTIMATE_ADJUSTMENTS] = len(
          point_estimate_adjustments_seq
      )

    standard_error_adjustments_seq = _coerce_optional_floats(
        standard_error_adjustments
    )
    if standard_error_adjustments is not None:
      lengths[calibration_constants.STANDARD_ERROR_ADJUSTMENTS] = len(
          standard_error_adjustments_seq
      )

    _validate_sequence_lengths(lengths)

    for kpi_type in kpi_types_seq:
      self._validate_kpi_compatibility(channel_name, kpi_type)

    container = self._get_or_create_channel_container(channel_name)
    num_experiments = len(point_estimates_seq)
    point_estimate_adjustments_sequence = (
        point_estimate_adjustments_seq or [None] * num_experiments
    )
    standard_error_adjustments_sequence = (
        standard_error_adjustments_seq or [None] * num_experiments
    )

    for (
        point_estimate,
        standard_error,
        spend,
        kpi_type,
        start_date,
        end_date,
        gamma_adj,
        tau_adj,
    ) in zip(
        point_estimates_seq,
        standard_errors_seq,
        spends_seq,
        kpi_types_seq,
        start_dates_seq,
        end_dates_seq,
        point_estimate_adjustments_sequence,
        standard_error_adjustments_sequence,
    ):
      config = experiment_adapter.resolve_experiment_source(
          point_estimate=point_estimate,
          standard_error=standard_error,
          kpi_type=kpi_type,
          revenue_per_kpi=self._revenue_per_kpi,
          total_spend=spend,
          experiment_start_date=start_date,
          experiment_end_date=end_date,
          point_estimate_adjustment=gamma_adj,
          standard_error_adjustment=tau_adj,
      )
      container.add_calibration_data(config)
    return self

  def build(
      self,
  ) -> base.CalibratedPriors:
    """Builds ROI prior distributions for media and reach & frequency channels.

    Transforms incrementality experiment estimates into calibrated priors and
    returns ROI prior distribution objects for media and reach & frequency
    channels.

    Returns:
      A `base.CalibratedPriors` object containing the calibrated prior
      distributions corresponding to ROI priors for media and reach &
      frequency channels.
    """

    calibration_result = self._registry.get_roi_distributions_by_channel(
        last_modeled_date=self._input_data.time_coordinates.all_dates[-1],
        max_lag=self._max_lag,
        interval_days=self._input_data.time_coordinates.interval_days,
        model_duration_days=_get_model_duration_days(
            self._input_data.time_coordinates
        ),
    )
    calibrated_map = calibration_result.distributions
    calibration_outputs_map = calibration_result.outputs
    custom_priors = self._resolved_custom_priors

    overwritten_channels = custom_priors.keys() & calibrated_map.keys()
    if overwritten_channels:
      warnings.warn(
          "Custom prior for channel(s)"
          f" {', '.join(sorted(overwritten_channels))} will be overwritten by"
          " the calibrated prior(s).",
          UserWarning,
      )

    combined_overrides = {**custom_priors, **calibrated_map}
    calibrated_channels = set(calibrated_map.keys())

    if (
        self._input_data.media_channel is not None
        and len(self._input_data.media_channel.values) > 0
    ):
      media_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          calibration_constants.get_default_prior().roi_m
      )
      roi_m = _wrap_as_calibrated(
          media_builder(**combined_overrides),
          channels=list(self._input_data.media_channel.values),
          calibrated_channels=calibrated_channels,
          calibration_outputs_map=calibration_outputs_map,
          name="roi_m",
      )
    else:
      roi_m = None

    if (
        self._input_data.rf_channel is not None
        and len(self._input_data.rf_channel.values) > 0
    ):
      rf_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          calibration_constants.get_default_prior().roi_rf
      )
      roi_rf = _wrap_as_calibrated(
          rf_builder(**combined_overrides),
          channels=list(self._input_data.rf_channel.values),
          calibrated_channels=calibrated_channels,
          calibration_outputs_map=calibration_outputs_map,
          name="roi_rf",
      )
    else:
      roi_rf = None

    return base.CalibratedPriors(roi_m=roi_m, roi_rf=roi_rf)

  def _validate_channel_name(self, channel_name: str) -> None:
    """Verifies that the channel name exists in the model's paid channels."""
    if channel_name not in self._valid_channels:
      raise ValueError(
          f"Channel '{channel_name}' not found in the model's paid media or RF"
          f" channels. Valid channels are: {sorted(self._valid_channels)}"
      )

  def _validate_kpi_compatibility(
      self, channel_name: str, experiment_kpi_type: str
  ) -> None:
    """Verifies compatibility between model KPI and experiment KPI types."""
    model_kpi_type = self._input_data.kpi_type
    has_rev_per_kpi = self._input_data.revenue_per_kpi is not None

    if model_kpi_type == constants.NON_REVENUE and not has_rev_per_kpi:
      raise ValueError(
          "Experiment calibration for models where"
          " the outcome is not in terms of revenue is not supported."
          " Pass `revenue_per_kpi` in `meridian.data.input_data.InputData`."
      )

    if (
        model_kpi_type == constants.REVENUE
        and experiment_kpi_type == constants.NON_REVENUE
        and not has_rev_per_kpi
    ):
      raise ValueError(
          f"Channel '{channel_name}': Experiment calibration for models with"
          " revenue as KPI and experiments with non-revenue KPIs requires"
          " `revenue_per_kpi` to be passed in"
          " `meridian.data.input_data.InputData`."
      )

  def _get_or_create_channel_container(
      self, channel_name: str
  ) -> base.CalibrationInput:
    """Returns the existing channel input container or creates a new one."""
    if channel_name not in self._container_by_channel:
      container = base.CalibrationInput(
          channel_name=channel_name,
          baseline_prior=self._get_baseline_prior(channel_name),
          total_spend=self._get_channel_total_spend(channel_name),
          adstock_decay_spec=self._resolved_adstock_decay_specs.get(  # pyrefly: ignore[bad-argument-type]
              channel_name
          ),
          alpha=self._resolved_alphas.get(channel_name),  # pyrefly: ignore[bad-argument-type]
      )
      self._container_by_channel[channel_name] = container
      self._registry.add_input(container)
    return self._container_by_channel[channel_name]

  def _get_channel_total_spend(self, channel_name: str) -> float:
    """Returns the total channel spend, cached for efficiency.

    Args:
      channel_name: Name of the channel.

    Returns:
      The total spend of the channel.

    Raises:
      ValueError: If channel_name is not found in the model's media or RF
        channels, or if the spend data for the channel is not available.
    """
    if channel_name in self._total_spend_by_channel:
      return self._total_spend_by_channel[channel_name]

    media_channels = (
        list(self._input_data.media_channel.values)
        if self._input_data.media_channel is not None
        else []
    )
    rf_channels = (
        list(self._input_data.rf_channel.values)
        if self._input_data.rf_channel is not None
        else []
    )
    if channel_name in media_channels:
      spend_data = self._input_data.media_spend
      dim = constants.MEDIA_CHANNEL
    elif channel_name in rf_channels:
      spend_data = self._input_data.rf_spend
      dim = constants.RF_CHANNEL
    else:
      raise ValueError(
          f"Channel '{channel_name}' not found in the model's media or RF"
          f" channels. Valid channels are: {sorted(self._valid_channels)}"
      )

    if spend_data is None:
      raise ValueError(
          f"Channel '{channel_name}' spend is required for calibration but is"
          " not available."
      )

    total_channel_spend = float(spend_data.sel({dim: channel_name}).sum())

    self._total_spend_by_channel[channel_name] = total_channel_spend
    return total_channel_spend

  def _get_baseline_prior(
      self, channel_name: str
  ) -> backend.tfd.Distribution | None:
    """Resolves the baseline prior distribution for the specified channel."""
    if self._baseline_prior is None:
      return None
    if isinstance(self._baseline_prior, backend.tfd.Distribution):
      return self._baseline_prior
    return self._baseline_prior.get(channel_name)
