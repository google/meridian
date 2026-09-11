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

"""Defines ModelContext class for Meridian."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import functools
from typing import Any, cast
import warnings

from meridian import backend
from meridian import constants
from meridian.data import input_data as data
from meridian.data import time_coordinates as tc
from meridian.model import adstock_hill
from meridian.model import knots
from meridian.model import media
from meridian.model import prior_distribution
from meridian.model import spec
from meridian.model import transformers
import numpy as np

__all__ = [
    "ChannelParameters",
    "ModelContext",
    "SaturationSpec",
]


@dataclasses.dataclass(frozen=True)
class ChannelParameters:
  """Parameters for a specific channel.

  Attributes:
    index: The index of the channel within its type.
    prefix: The distribution prefix (e.g., 'm', 'rf', 'om', 'orf').
    decay_spec: The model's decay spec for this channel.
    is_rf: Whether the channel is a Reach & Frequency (RF) channel.
  """

  index: int
  prefix: str
  decay_spec: str
  is_rf: bool


def _get_decay(decay_spec: str | Sequence[str], index: int) -> str:
  return decay_spec[index] if not isinstance(decay_spec, str) else decay_spec


def _nearest_coordinates_hint(
    target: datetime.date, dates: Sequence[datetime.date]
) -> str:
  """Describes the time coordinates bracketing `target`, for error messages."""
  earlier = [date for date in dates if date < target]
  later = [date for date in dates if date > target]
  nearest = [str(d) for d in earlier[-1:] + later[:1]]
  if not nearest:
    return ""
  return f" The nearest are {' and '.join(nearest)}."


def _compile_date_range_mask(
    date_ranges: Sequence[spec.DateRange],
    dates: Sequence[datetime.date],
    *,
    spec_name: str,
) -> np.ndarray:
  """Compiles a sequence of `DateRange`s into a boolean mask over `dates`.

  Each `DateRange` is the closed interval `[start_date, end_date]`, matching
  the semantics documented on `spec.DateRange`: both bounds are inclusive. An
  omitted bound leaves that side open, so `DateRange()` selects every date. The
  union of all the given ranges is taken.

  Note: Every bound that is present must be one of `dates`.

  Args:
    date_ranges: The date ranges to compile.
    dates: The date coordinates to compile against, in order.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.

  Returns:
    A boolean array of shape `(len(dates),)`, `True` wherever the date falls
    inside at least one of `date_ranges`.

  Raises:
    ValueError: If a bound is not one of `dates`.
  """
  known_dates = set(dates)
  mask = np.zeros(len(dates), dtype=bool)
  for date_range in date_ranges:
    # `DateRange.__post_init__` already normalizes these, but the declared
    # attribute type stays polymorphic (`Date | None`), so normalize again to
    # compare `date` against `date` rather than against `str`.
    start = (
        tc.normalize_date(date_range.start_date)
        if date_range.start_date is not None
        else None
    )
    end = (
        tc.normalize_date(date_range.end_date)
        if date_range.end_date is not None
        else None
    )
    for bound_name, bound in (("start_date", start), ("end_date", end)):
      if bound is not None and bound not in known_dates:
        raise ValueError(
            f"`{spec_name}` has a `DateRange` whose `{bound_name}` ({bound}) is"
            " not one of the input data's time coordinates. Date range bounds"
            " must name an exact time coordinate."
            + _nearest_coordinates_hint(bound, dates)
        )
    mask |= np.array(
        [
            (start is None or date >= start) and (end is None or date <= end)
            for date in dates
        ],
        dtype=bool,
    )
  return mask


def _resolve_name_indices(
    names: Sequence[str],
    universe: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> list[int]:
  """Resolves coordinate names to their positional indices.

  Args:
    names: The names to resolve.
    universe: The ordered coordinate values to resolve against.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The input data dimension being resolved against, used in error
      messages.

  Returns:
    The index of each name in `universe`, in the order given.

  Raises:
    ValueError: If any name is absent from `universe`.
  """
  index_of = {name: index for index, name in enumerate(universe)}
  unknown = [name for name in names if name not in index_of]
  if unknown:
    raise ValueError(
        f"`{spec_name}` refers to {dim_name} that are not in the input data:"
        f" {sorted(unknown)}. Available {dim_name}: {sorted(universe)}."
    )
  return [index_of[name] for name in names]


def _compile_name_mask(
    names: Sequence[str],
    universe: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> np.ndarray:
  """Compiles a selection of coordinate names into a boolean mask.

  Args:
    names: The selected names.
    universe: The ordered coordinate values to compile against.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The input data dimension being resolved against, used in error
      messages.

  Returns:
    A boolean array of shape `(len(universe),)`, `True` at the selected names.

  Raises:
    ValueError: If any name is absent from `universe`.
  """
  mask = np.zeros(len(universe), dtype=bool)
  mask[
      _resolve_name_indices(
          names, universe, spec_name=spec_name, dim_name=dim_name
      )
  ] = True
  return mask


def _compile_calibration_spec(
    calibration: spec.CalibrationSpec,
    dates: Sequence[datetime.date],
    channels: Sequence[str],
    *,
    spec_name: str,
    dim_name: str,
) -> np.ndarray:
  """Compiles a `CalibrationSpec` into a boolean calibration period array.

  Args:
    calibration: The declarative calibration specification.
    dates: The media time coordinates to compile against, in order.
    channels: The channel coordinates to compile against, in order.
    spec_name: The `ModelSpec` attribute being compiled, used in error messages.
    dim_name: The channel dimension being resolved against, used in error
      messages.

  Returns:
    A boolean array of shape `(len(dates), len(channels))`.

  Raises:
    ValueError: If a channel name is absent from `channels`, or a date range
      bound is not one of `dates`.
  """
  entries = calibration.spec
  n_channels = len(channels)

  # `CalibrationSpec.__post_init__` guarantees the sequence is homogeneous and
  # non-empty, so the first element determines the scope of the whole spec.
  # The type checker cannot carry that guarantee across the sequence, hence the
  # casts.
  if isinstance(entries[0], spec.DateRange):
    global_mask = _compile_date_range_mask(
        cast(Sequence[spec.DateRange], entries), dates, spec_name=spec_name
    )
    return np.tile(global_mask[:, np.newaxis], (1, n_channels))

  # Per-channel scope. A channel that no entry mentions is left *unrestricted*
  # (all `True`), matching the documented meaning of an unset
  # `roi_calibration_period`: "If `None`, all times are used." Defaulting an
  # unmentioned channel to all `False` would instead zero out its aggregated
  # spend, making the denominator of its ROI prior zero.
  compiled = np.zeros((len(dates), n_channels), dtype=bool)
  is_mentioned = np.zeros(n_channels, dtype=bool)
  for entry in cast(Sequence[spec.ChannelCalibrationSpec], entries):
    mask = _compile_date_range_mask(
        entry.date_ranges, dates, spec_name=spec_name
    )
    for index in _resolve_name_indices(
        entry.channels, channels, spec_name=spec_name, dim_name=dim_name
    ):
      compiled[:, index] |= mask
      is_mentioned[index] = True
  compiled[:, ~is_mentioned] = True
  return compiled


def _compile_geo_holdout_specs(
    geo_specs: Sequence[spec.GeoHoldoutSpec],
    dates: Sequence[datetime.date],
    geos: Sequence[str],
) -> np.ndarray:
  """Compiles per-geo holdout specs into a boolean holdout mask.

  Unlike calibration, a geo that no entry mentions is simply not held out, so
  it compiles to all `False`.

  Args:
    geo_specs: The per-geo holdout specifications.
    dates: The time coordinates to compile against, in order.
    geos: The geo coordinates to compile against, in order.

  Returns:
    A boolean array of shape `(len(geos), len(dates))`.

  Raises:
    ValueError: If a geo name is absent from `geos`, or a date range bound is
      not one of `dates`.
  """
  compiled = np.zeros((len(geos), len(dates)), dtype=bool)
  for geo_spec in geo_specs:
    mask = _compile_date_range_mask(
        geo_spec.date_ranges, dates, spec_name="holdout"
    )
    for index in _resolve_name_indices(
        geo_spec.geos, geos, spec_name="holdout", dim_name="geos"
    ):
      compiled[index, :] |= mask
  return compiled


def _draw_random_holdout(
    random_spec: spec.RandomHoldoutSpec,
    n_geos: int,
    n_times: int,
) -> np.ndarray:
  """Draws a random holdout mask, stratified by geo.

  Each geo independently holds out exactly `round(ratio * n_times)` time
  periods, sampled without replacement. Stratifying by geo -- rather than
  drawing over the flattened `n_geos * n_times` cell space -- guarantees that
  every geo contributes both training and test rows.

  Args:
    random_spec: The random holdout specification.
    n_geos: The number of geos.
    n_times: The number of time periods.

  Returns:
    A boolean array of shape `(n_geos, n_times)`.
  """
  rng = np.random.default_rng(random_spec.seed)
  n_holdout = min(int(round(random_spec.ratio * n_times)), n_times)
  mask = np.zeros((n_geos, n_times), dtype=bool)
  if n_holdout > 0:
    for geo_index in range(n_geos):
      mask[geo_index, rng.choice(n_times, size=n_holdout, replace=False)] = True
  return mask


@dataclasses.dataclass(frozen=True)
class SaturationSpec:
  """Specification for each channel's saturation function.

  Attributes:
    media: A string or sequence of strings specifying the saturation function to
      use for media channels.
    rf: A string or sequence of strings specifying the saturation function to
      use for reach and frequency channels.
    organic_media: A string or sequence of strings specifying the saturation
      function to use for organic media channels.
    organic_rf: A string or sequence of strings specifying the saturation
      function to use for organic reach and frequency channels.
  """

  media: str | Sequence[str] = constants.HILL
  rf: str | Sequence[str] = constants.HILL
  organic_media: str | Sequence[str] = constants.HILL
  organic_rf: str | Sequence[str] = constants.HILL


class ModelContext:
  """Model context for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.
  """

  def __init__(
      self,
      input_data: data.InputData,
      model_spec: spec.ModelSpec,
  ):
    self._input_data = input_data
    self._model_spec = model_spec

    self._validate_data_dependent_model_spec()
    self._validate_model_spec_shapes()

    self._set_total_media_contribution_prior = False
    self._warn_setting_ignored_priors()
    self._validate_mroi_priors_non_revenue()
    self._validate_roi_priors_non_revenue()
    self._check_media_prior_support()
    self._validate_geo_invariants()
    self._validate_time_invariants()
    self._validate_media_spend_for_paid_channels()
    self._validate_rf_spend_for_paid_channels()

  # TODO: Deduplicate with `_validate_model_spec_shapes`. Both
  # methods run from `__init__` and validate the same legacy `ModelSpec`
  # arrays (`roi_calibration_period`, `rf_roi_calibration_period`,
  # `holdout_id`, `control_population_scaling_id`) against the same shapes,
  # with near-identical error messages.
  def _validate_data_dependent_model_spec(self):
    """Validates that the data dependent model specs have correct shapes."""

    if self._model_spec.roi_calibration_period is not None and (
        self._model_spec.roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_media_channels,
        )
    ):
      raise ValueError(
          "The shape of `roi_calibration_period`"
          f" {self._model_spec.roi_calibration_period.shape} is different from"
          f" `(n_media_times, n_media_channels) = ({self.n_media_times},"
          f" {self.n_media_channels})`."
      )

    if self._model_spec.rf_roi_calibration_period is not None and (
        self._model_spec.rf_roi_calibration_period.shape
        != (
            self.n_media_times,
            self.n_rf_channels,
        )
    ):
      raise ValueError(
          "The shape of `rf_roi_calibration_period`"
          f" {self._model_spec.rf_roi_calibration_period.shape} is different"
          f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
          f" {self.n_rf_channels})`."
      )

    if self._model_spec.holdout_id is not None:
      if self.is_national and (
          self._model_spec.holdout_id.shape != (self.n_times,)
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_times,) = ({self.n_times},)`."
        )
      elif not self.is_national and (
          self._model_spec.holdout_id.shape
          != (
              self.n_geos,
              self.n_times,
          )
      ):
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            f" different from `(n_geos, n_times) = ({self.n_geos},"
            f" {self.n_times})`."
        )

    if self._model_spec.control_population_scaling_id is not None and (
        self._model_spec.control_population_scaling_id.shape
        != (self.n_controls,)
    ):
      raise ValueError(
          "The shape of `control_population_scaling_id`"
          f" {self._model_spec.control_population_scaling_id.shape} is"
          f" different from `(n_controls,) = ({self.n_controls},)`."
      )

    if self._model_spec.non_media_population_scaling_id is not None and (
        self._model_spec.non_media_population_scaling_id.shape
        != (self.n_non_media_channels,)
    ):
      raise ValueError(
          "The shape of `non_media_population_scaling_id`"
          f" {self._model_spec.non_media_population_scaling_id.shape} is"
          " different from `(n_non_media_channels,) ="
          f" ({self.n_non_media_channels},)`."
      )

  # TODO: Deduplicate with `_validate_data_dependent_model_spec`.
  def _validate_model_spec_shapes(self):
    """Validate shapes of model_spec attributes."""
    if self._model_spec.roi_calibration_period is not None:
      if self._model_spec.roi_calibration_period.shape != (
          self.n_media_times,
          self.n_media_channels,
      ):
        raise ValueError(
            "The shape of `roi_calibration_period`"
            f" {self._model_spec.roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_media_channels) = ({self.n_media_times},"
            f" {self.n_media_channels})`."
        )

    if self._model_spec.rf_roi_calibration_period is not None:
      if self._model_spec.rf_roi_calibration_period.shape != (
          self.n_media_times,
          self.n_rf_channels,
      ):
        raise ValueError(
            "The shape of `rf_roi_calibration_period`"
            f" {self._model_spec.rf_roi_calibration_period.shape} is different"
            f" from `(n_media_times, n_rf_channels) = ({self.n_media_times},"
            f" {self.n_rf_channels})`."
        )

    if self._model_spec.holdout_id is not None:
      expected_shape = (
          (self.n_times,) if self.is_national else (self.n_geos, self.n_times)
      )
      if self._model_spec.holdout_id.shape != expected_shape:
        raise ValueError(
            f"The shape of `holdout_id` {self._model_spec.holdout_id.shape} is"
            " different from"
            f" {'`(n_times,)`' if self.is_national else '`(n_geos, n_times)`'}"
            f" = {expected_shape}."
        )

    if self._model_spec.control_population_scaling_id is not None:
      if self._model_spec.control_population_scaling_id.shape != (
          self.n_controls,
      ):
        raise ValueError(
            "The shape of `control_population_scaling_id`"
            f" {self._model_spec.control_population_scaling_id.shape} is"
            f" different from `(n_controls,) = ({self.n_controls},)`."
        )

  def _validate_geo_invariants(self):
    """Validates non-national model invariants."""
    if self.is_national:
      return

    if self._input_data.controls is not None:
      self._check_if_no_geo_variation(
          self.controls_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_geo_variation(
          self.non_media_treatments_normalized,  # pyrefly: ignore[bad-argument-type]
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_geo_variation(
          self.media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_geo_variation(
          self.rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_geo_variation(
          self.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
          "organic_media",
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_geo_variation(
          self.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
          "organic_reach",
          self._input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
      )

  def _check_if_no_geo_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if `n_knots == n_time` and data lacks geo variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=0) < epsilon  # pyrefly: ignore[bad-argument-type]
    )[:, 1]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    # We use the shape of scaled_data (instead of `n_time`) because the data may
    # be padded to account for lagged effects.
    data_n_time = scaled_data.shape[1]
    mask = backend.equal(counts, data_n_time)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)

    if col_idx_bad.shape[0] and self.knot_info.n_knots == self.n_times:
      raise ValueError(
          f"The following {data_name} variables do not vary across geos, making"
          f" a model with n_knots=n_time unidentifiable: {dims_bad}. This can"
          " lead to poor model convergence. Since these variables only vary"
          " across time and not across geo, they are collinear with time and"
          " redundant in a model with a parameter for each time period.  To"
          " address this, you can either: (1) decrease the number of knots"
          " (n_knots < n_time), or (2) drop the listed variables that do not"
          " vary across geos."
      )

  def _validate_time_invariants(self):
    """Validates model time invariants."""
    if self._input_data.controls is not None:
      self._check_if_no_time_variation(
          self.controls_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.CONTROLS,
          self._input_data.controls.coords[constants.CONTROL_VARIABLE].values,
      )
    if self._input_data.non_media_treatments is not None:
      self._check_if_no_time_variation(
          self.non_media_treatments_normalized,  # pyrefly: ignore[bad-argument-type]
          constants.NON_MEDIA_TREATMENTS,
          self._input_data.non_media_treatments.coords[
              constants.NON_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.media is not None:
      self._check_if_no_time_variation(
          self.media_tensors.media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.MEDIA,
          self._input_data.media.coords[constants.MEDIA_CHANNEL].values,
      )
    if self._input_data.reach is not None:
      self._check_if_no_time_variation(
          self.rf_tensors.reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.REACH,
          self._input_data.reach.coords[constants.RF_CHANNEL].values,
      )
    if self._input_data.organic_media is not None:
      self._check_if_no_time_variation(
          self.organic_media_tensors.organic_media_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.ORGANIC_MEDIA,
          self._input_data.organic_media.coords[
              constants.ORGANIC_MEDIA_CHANNEL
          ].values,
      )
    if self._input_data.organic_reach is not None:
      self._check_if_no_time_variation(
          self.organic_rf_tensors.organic_reach_scaled,  # pyrefly: ignore[bad-argument-type]
          constants.ORGANIC_REACH,
          self._input_data.organic_reach.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values,
      )

  def _validate_media_spend_for_paid_channels(self) -> None:
    self._validate_spend_for_paid_channels(
        self.input_data.aggregate_media_spend(), constants.MEDIA_CHANNEL
    )

  def _validate_rf_spend_for_paid_channels(self) -> None:
    self._validate_spend_for_paid_channels(
        self.input_data.aggregate_rf_spend(), constants.RF_CHANNEL
    )

  def _validate_spend_for_paid_channels(
      self,
      spend: np.ndarray | None,
      dim: str,
  ) -> None:
    """Validates non-zero media spend for paid media channels.

    Args:
      spend: The media spend data to validate.
      dim: The dimension name of the spend data.

    Raises:
      ValueError if any paid media channel has zero total spend.
    """
    if spend is None:
      return
    zero_spend_channels = spend.coords[dim].where(spend == 0, drop=True).values  # pyrefly: ignore[missing-attribute]

    if zero_spend_channels.size > 0:
      raise ValueError(
          "Zero total spend detected for paid channels:"
          f" {', '.join(zero_spend_channels)}. If data is correct and this is"
          " expected, please consider modeling the data as organic media."
      )

  def _check_if_no_time_variation(
      self,
      scaled_data: backend.Tensor,
      data_name: str,
      data_dims: Sequence[str],
      epsilon=1e-4,
  ):
    """Raise an error if data lacks time variation."""

    # Result shape: [n, d], where d is the number of axes of condition.
    col_idx_full = backend.get_indices_where(
        backend.reduce_std(scaled_data, axis=1) < epsilon  # pyrefly: ignore[bad-argument-type]
    )[:, 1]
    col_idx_unique, _, counts = backend.unique_with_counts(col_idx_full)
    mask = backend.equal(counts, self.n_geos)
    col_idx_bad = backend.boolean_mask(col_idx_unique, mask)
    dims_bad = backend.gather(data_dims, col_idx_bad)
    if col_idx_bad.shape[0]:
      if self.is_national:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            " which is equivalent to no signal at all in a national model:"
            f" {dims_bad}.  This can lead to poor model convergence. To address"
            " this, drop the listed variables that do not vary across time."
        )
      else:
        raise ValueError(
            f"The following {data_name} variables do not vary across time,"
            f" making a model with geo main effects unidentifiable: {dims_bad}."
            " This can lead to poor model convergence. Since these variables"
            " only vary across geo and not across time, they are collinear"
            " with geo and redundant in a model with geo main effects. To"
            " address this, drop the listed variables that do not vary across"
            " time."
        )

  @property
  def input_data(self) -> data.InputData:
    return self._input_data

  @property
  def model_spec(self) -> spec.ModelSpec:
    return self._model_spec

  @functools.cached_property
  def media_tensors(self) -> media.MediaTensors:
    return media.build_media_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def rf_tensors(self) -> media.RfTensors:
    return media.build_rf_tensors(self._input_data, self._model_spec)

  @functools.cached_property
  def organic_media_tensors(self) -> media.OrganicMediaTensors:
    return media.build_organic_media_tensors(self._input_data)

  @functools.cached_property
  def organic_rf_tensors(self) -> media.OrganicRfTensors:
    return media.build_organic_rf_tensors(self._input_data)

  @functools.cached_property
  def kpi(self) -> backend.Tensor:
    return backend.to_tensor(self._input_data.kpi, dtype=backend.float_dtype)

  @functools.cached_property
  def revenue_per_kpi(self) -> backend.Tensor | None:
    if self._input_data.revenue_per_kpi is None:
      return None
    return backend.to_tensor(
        self._input_data.revenue_per_kpi, dtype=backend.float_dtype
    )

  @functools.cached_property
  def controls(self) -> backend.Tensor | None:
    if self._input_data.controls is None:
      return None
    return backend.to_tensor(
        self._input_data.controls, dtype=backend.float_dtype
    )

  @functools.cached_property
  def non_media_treatments(self) -> backend.Tensor | None:
    if self._input_data.non_media_treatments is None:
      return None
    return backend.to_tensor(
        self._input_data.non_media_treatments, dtype=backend.float_dtype
    )

  @functools.cached_property
  def population(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.population, dtype=backend.float_dtype
    )

  @functools.cached_property
  def total_spend(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_spend(), dtype=backend.float_dtype
    )

  @functools.cached_property
  def total_outcome(self) -> backend.Tensor:
    return backend.to_tensor(
        self._input_data.get_total_outcome(), dtype=backend.float_dtype
    )

  @property
  def n_geos(self) -> int:
    return len(self._input_data.geo)

  @property
  def n_media_channels(self) -> int:
    if self._input_data.media_channel is None:
      return 0
    return len(self._input_data.media_channel)

  @property
  def n_rf_channels(self) -> int:
    if self._input_data.rf_channel is None:
      return 0
    return len(self._input_data.rf_channel)

  @property
  def n_organic_media_channels(self) -> int:
    if self._input_data.organic_media_channel is None:
      return 0
    return len(self._input_data.organic_media_channel)

  @property
  def n_organic_rf_channels(self) -> int:
    if self._input_data.organic_rf_channel is None:
      return 0
    return len(self._input_data.organic_rf_channel)

  @property
  def n_controls(self) -> int:
    if self._input_data.control_variable is None:
      return 0
    return len(self._input_data.control_variable)

  @property
  def n_non_media_channels(self) -> int:
    if self._input_data.non_media_channel is None:
      return 0
    return len(self._input_data.non_media_channel)

  @property
  def n_times(self) -> int:
    return len(self._input_data.time)

  @property
  def n_media_times(self) -> int:
    return len(self._input_data.media_time)

  @property
  def is_national(self) -> bool:
    return self.n_geos == 1

  @functools.cached_property
  def knot_info(self) -> knots.KnotInfo:
    return knots.get_knot_info(
        n_times=self.n_times,
        knots=self._model_spec.knots,
        enable_aks=self._model_spec.enable_aks,
        data=self._input_data,
        is_national=self.is_national,
    )

  def _inject_legacy_knot_info_for_serde(self, legacy_knots: list[int]):
    """Injects legacy knots by bypassing the AKS check in get_knot_info."""
    legacy_knots_arr = np.array(legacy_knots, dtype=int)
    n_knots = len(legacy_knots_arr)
    if n_knots == 1:
      weights = np.ones((1, self.n_times), dtype=backend.np_float_dtype)
    else:
      weights = knots.l1_distance_weights(self.n_times, legacy_knots_arr)  # pyrefly: ignore[bad-argument-type]
    self.__dict__["knot_info"] = knots.KnotInfo(
        n_knots=n_knots,
        knot_locations=legacy_knots_arr,  # pyrefly: ignore[bad-argument-type]
        weights=weights,  # pyrefly: ignore[bad-argument-type]
    )

  @functools.cached_property
  def controls_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for controls, if it exists."""
    if self.controls is None:
      return None

    if self._model_spec.control_population_scaling_id is not None:
      controls_population_scaling_id = backend.to_tensor(
          self._model_spec.control_population_scaling_id, dtype=backend.bool_
      )
    else:
      controls_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.controls,
        population=self.population,
        population_scaling_id=controls_population_scaling_id,
    )

  @functools.cached_property
  def non_media_transformer(
      self,
  ) -> transformers.CenteringAndScalingTransformer | None:
    """Returns a `CenteringAndScalingTransformer` for non-media treatments."""
    if self.non_media_treatments is None:
      return None
    if self._model_spec.non_media_population_scaling_id is not None:
      non_media_population_scaling_id = backend.to_tensor(
          self._model_spec.non_media_population_scaling_id, dtype=backend.bool_
      )
    else:
      non_media_population_scaling_id = None

    return transformers.CenteringAndScalingTransformer(
        tensor=self.non_media_treatments,
        population=self.population,
        population_scaling_id=non_media_population_scaling_id,
    )

  @functools.cached_property
  def kpi_transformer(self) -> transformers.KpiTransformer:
    return transformers.KpiTransformer(self.kpi, self.population)

  @functools.cached_property
  def controls_scaled(self) -> backend.Tensor | None:
    if self.controls is not None:
      # If `controls` is defined, then `controls_transformer` is also defined.
      return self.controls_transformer.forward(self.controls)  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def non_media_treatments_normalized(self) -> backend.Tensor | None:
    """Normalized non-media treatments.

    The non-media treatments values are scaled by population (for channels where
    `non_media_population_scaling_id` is `True`) and normalized by centering and
    scaling with means and standard deviations.
    """
    if self.non_media_transformer is not None:
      return self.non_media_transformer.forward(
          self.non_media_treatments
      )  # pytype: disable=attribute-error
    else:
      return None

  @functools.cached_property
  def kpi_scaled(self) -> backend.Tensor:
    return self.kpi_transformer.forward(self.kpi)

  @functools.cached_property
  def media_effects_dist(self) -> str:
    if self.is_national:
      return constants.NATIONAL_MODEL_SPEC_ARGS[constants.MEDIA_EFFECTS_DIST]  # pytype: disable=bad-return-type
    else:
      return self._model_spec.media_effects_dist

  @functools.cached_property
  def unique_sigma_for_each_geo(self) -> bool:
    if self.is_national:
      # Should evaluate to False.
      return constants.NATIONAL_MODEL_SPEC_ARGS[  # pytype: disable=bad-return-type
          constants.UNIQUE_SIGMA_FOR_EACH_GEO
      ]
    else:
      return self._model_spec.unique_sigma_for_each_geo

  @functools.cached_property
  def baseline_geo_idx(self) -> int:
    """Returns the index of the baseline geo."""
    if isinstance(self._model_spec.baseline_geo, int):
      if (
          self._model_spec.baseline_geo < 0
          or self._model_spec.baseline_geo >= self.n_geos
      ):
        raise ValueError(
            f"Baseline geo index {self._model_spec.baseline_geo} out of range"
            f" [0, {self.n_geos - 1}]."
        )
      return self._model_spec.baseline_geo
    elif isinstance(self._model_spec.baseline_geo, str):
      # np.where returns a 1-D tuple, its first element is an array of found
      # elements.
      index = np.where(self._input_data.geo == self._model_spec.baseline_geo)[0]
      if index.size == 0:
        raise ValueError(
            f"Baseline geo '{self._model_spec.baseline_geo}' not found."
        )
      # Geos are unique, so index is a 1-element array.
      return index[0]
    else:
      return backend.argmax(self.population)

  @functools.cached_property
  def holdout_id(self) -> backend.Tensor | None:
    if self._model_spec.holdout_id is None:
      return None
    tensor = backend.to_tensor(self._model_spec.holdout_id, dtype=backend.bool_)
    return tensor[backend.newaxis, ...] if self.is_national else tensor

  # --------------------------------------------------------------------------
  # Compiled model spec properties.
  #
  # These resolve `ModelSpec`'s declarative attributes against this context's
  # `InputData` coordinates, producing the positional boolean arrays the model
  # engine consumes. They are the single place where a channel name becomes a
  # column index and a date range becomes a row mask.
  #
  # Precedence is *legacy first*: when both a declarative attribute and its
  # deprecated array counterpart are set, the deprecated array wins. This
  # matches the contract `ModelSpec.__post_init__` currently advertises in its
  # conflict warning ("<legacy> takes precedence for backward compatibility").
  # --------------------------------------------------------------------------

  def _coordinate_names(self, coordinate: Any) -> list[str]:
    """Returns an input data coordinate's values as a list of strings."""
    if coordinate is None:
      return []
    return [str(value) for value in coordinate.values]

  @functools.cached_property
  def compiled_roi_calibration_period(self) -> np.ndarray | None:
    """The effective ROI calibration period for media channels.

    Resolved from the declarative `ModelSpec.roi_calibration` against the input
    data's media time and media channel coordinates, or taken as-is from the
    deprecated `ModelSpec.roi_calibration_period`.

    Returns:
      A boolean array of shape `(n_media_times, n_media_channels)`, or `None`
      if neither attribute is set.

    Raises:
      ValueError: If the spec names a media channel not in the input data, or
        a date range bound that is not a time coordinate.
    """
    if self._model_spec.roi_calibration_period is not None:
      return self._model_spec.roi_calibration_period
    if self._model_spec.roi_calibration is None:
      return None
    return _compile_calibration_spec(
        self._model_spec.roi_calibration,
        self._input_data.media_time_coordinates.all_dates,
        self._coordinate_names(self._input_data.media_channel),
        spec_name="roi_calibration",
        dim_name="media channels",
    )

  @functools.cached_property
  def compiled_rf_roi_calibration_period(self) -> np.ndarray | None:
    """The effective ROI calibration period for reach & frequency channels.

    Resolved from the declarative `ModelSpec.rf_roi_calibration` against the
    input data's media time and RF channel coordinates, or taken as-is from the
    deprecated `ModelSpec.rf_roi_calibration_period`.

    Returns:
      A boolean array of shape `(n_media_times, n_rf_channels)`, or `None` if
      neither attribute is set.

    Raises:
      ValueError: If the spec names an RF channel not in the input data, or a
        date range bound that is not a time coordinate.
    """
    if self._model_spec.rf_roi_calibration_period is not None:
      return self._model_spec.rf_roi_calibration_period
    if self._model_spec.rf_roi_calibration is None:
      return None
    return _compile_calibration_spec(
        self._model_spec.rf_roi_calibration,
        self._input_data.media_time_coordinates.all_dates,
        self._coordinate_names(self._input_data.rf_channel),
        spec_name="rf_roi_calibration",
        dim_name="RF channels",
    )

  @functools.cached_property
  def compiled_holdout_id(self) -> np.ndarray | None:
    """The effective holdout mask.

    Resolved from the declarative `ModelSpec.holdout` against the input data's
    time and geo coordinates, or taken as-is from the deprecated
    `ModelSpec.holdout_id`.

    For a declarative holdout, a `resolved` draw always wins and is never
    re-drawn; see `spec.RandomHoldoutSpec` for why a seed alone cannot
    reproduce a draw. Only when a `RandomHoldoutSpec` carries no `resolved`
    draw is one made here, once, and memoized for the lifetime of this context.

    Returns:
      A boolean array of shape `(n_times,)` for a national model or
      `(n_geos, n_times)` otherwise -- the same convention as the deprecated
      `ModelSpec.holdout_id` -- or `None` if neither attribute is set.

    Raises:
      ValueError: If the spec names a geo not in the input data, or a date
        range bound that is not a time coordinate.
    """
    if self._model_spec.holdout_id is not None:
      return self._model_spec.holdout_id
    holdout = self._model_spec.holdout
    if holdout is None:
      return None

    dates = self._input_data.time_coordinates.all_dates
    geos = self._coordinate_names(self._input_data.geo)

    if holdout.resolved is not None:
      compiled = _compile_geo_holdout_specs(holdout.resolved, dates, geos)
    elif isinstance(holdout.spec, spec.RandomHoldoutSpec):
      compiled = _draw_random_holdout(holdout.spec, len(geos), len(dates))
    elif isinstance(holdout.spec[0], spec.DateRange):
      # A global holdout applies the same date mask to every geo.
      global_mask = _compile_date_range_mask(
          cast(Sequence[spec.DateRange], holdout.spec),
          dates,
          spec_name="holdout",
      )
      compiled = np.tile(global_mask[np.newaxis, :], (len(geos), 1))
    else:
      compiled = _compile_geo_holdout_specs(
          cast(Sequence[spec.GeoHoldoutSpec], holdout.spec), dates, geos
      )

    # National models carry a 1-D holdout, matching the legacy convention that
    # `_validate_model_spec_shapes` enforces.
    return compiled[0] if self.is_national else compiled

  @functools.cached_property
  def compiled_control_population_scaling_id(self) -> np.ndarray | None:
    """The effective population-scaling selection for control variables.

    Resolved from the declarative `ModelSpec.population_scaled_controls`
    against the input data's control variable coordinates, or taken as-is from
    the deprecated `ModelSpec.control_population_scaling_id`.

    Returns:
      A boolean array of shape `(n_controls,)`, or `None` if neither attribute
      is set.

    Raises:
      ValueError: If the spec names a control variable not in the input data.
    """
    if self._model_spec.control_population_scaling_id is not None:
      return self._model_spec.control_population_scaling_id
    if self._model_spec.population_scaled_controls is None:
      return None
    return _compile_name_mask(
        self._model_spec.population_scaled_controls,
        self._coordinate_names(self._input_data.control_variable),
        spec_name="population_scaled_controls",
        dim_name="control variables",
    )

  @functools.cached_property
  def compiled_non_media_population_scaling_id(self) -> np.ndarray | None:
    """The effective population-scaling selection for non-media channels.

    Resolved from the declarative
    `ModelSpec.population_scaled_non_media_channels` against the input data's
    non-media channel coordinates, or taken as-is from the deprecated
    `ModelSpec.non_media_population_scaling_id`.

    Returns:
      A boolean array of shape `(n_non_media_channels,)`, or `None` if neither
      attribute is set.

    Raises:
      ValueError: If the spec names a non-media channel not in the input data.
    """
    if self._model_spec.non_media_population_scaling_id is not None:
      return self._model_spec.non_media_population_scaling_id
    if self._model_spec.population_scaled_non_media_channels is None:
      return None
    return _compile_name_mask(
        self._model_spec.population_scaled_non_media_channels,
        self._coordinate_names(self._input_data.non_media_channel),
        spec_name="population_scaled_non_media_channels",
        dim_name="non-media channels",
    )

  def resolve_non_media_baseline_values(
      self,
      values: Mapping[str, float | str] | Sequence[float | str] | None,
  ) -> list[float | str] | None:
    """Resolves non-media baseline values into positional channel order.

    A mapping only needs to name the channels whose baseline differs from the
    default; any channel it omits falls back to `'min'`.

    Args:
      values: A mapping from non-media channel name to baseline value, a
        sequence already in channel order, or `None`.

    Returns:
      A list of length `n_non_media_channels` in channel order, or `None` if
      `values` is `None`.

    Raises:
      ValueError: If a mapping key is not a known non-media channel.
    """
    if values is None:
      return None
    if not isinstance(values, Mapping):
      return list(values)
    channels = self._coordinate_names(self._input_data.non_media_channel)
    _resolve_name_indices(
        list(values.keys()),
        channels,
        spec_name="non_media_baseline_values",
        dim_name="non-media channels",
    )
    return [
        values.get(channel, constants.NON_MEDIA_BASELINE_MIN)
        for channel in channels
    ]

  @functools.cached_property
  def compiled_non_media_baseline_values(self) -> list[float | str] | None:
    """`ModelSpec.non_media_baseline_values`, in positional channel order.

    Returns:
      A list of length `n_non_media_channels`, or `None` if the attribute is
      unset.

    Raises:
      ValueError: If the attribute is a mapping naming an unknown channel.
    """
    return self.resolve_non_media_baseline_values(
        self._model_spec.non_media_baseline_values
    )

  def _warn_setting_ignored_priors(self):
    """Raises a warning if ignored priors are set."""
    default_distribution = prior_distribution.PriorDistribution()
    for ignored_priors_dict, prior_type, prior_type_name in (
        (
            constants.IGNORED_PRIORS_MEDIA,
            self._model_spec.effective_media_prior_type,
            "media_prior_type",
        ),
        (
            constants.IGNORED_PRIORS_RF,
            self._model_spec.effective_rf_prior_type,
            "rf_prior_type",
        ),
    ):
      ignored_custom_priors = []
      for prior in ignored_priors_dict.get(prior_type, []):
        self_prior = getattr(self._model_spec.prior, prior)
        default_prior = getattr(default_distribution, prior)
        if not prior_distribution.distributions_are_equal(
            self_prior, default_prior
        ):
          ignored_custom_priors.append(prior)
      if ignored_custom_priors:
        ignored_priors_str = ", ".join(ignored_custom_priors)
        warnings.warn(
            f"Custom prior(s) `{ignored_priors_str}` are ignored when"
            f' `{prior_type_name}` is set to "{prior_type}".'
        )

  def _validate_mroi_priors_non_revenue(self):
    """Validates mroi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      if (
          self.n_media_channels > 0
          and (
              self._model_spec.effective_media_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.mroi_m, default_distribution.mroi_m
          )
      ):
        raise ValueError(
            f"Custom priors should be set on `{constants.MROI_M}` when"
            ' `media_prior_type` is "mroi", KPI is non-revenue and revenue per'
            " kpi data is missing."
        )
      if (
          self.n_rf_channels > 0
          and (
              self._model_spec.effective_rf_prior_type
              == constants.TREATMENT_PRIOR_TYPE_MROI
          )
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.mroi_rf, default_distribution.mroi_rf
          )
      ):
        raise ValueError(
            f"Custom priors should be set on `{constants.MROI_RF}` when"
            ' `rf_prior_type` is "mroi", KPI is non-revenue and revenue per kpi'
            " data is missing."
        )

  def _validate_roi_priors_non_revenue(self):
    """Validates roi priors in the non-revenue outcome case."""
    if (
        self._input_data.kpi_type == constants.NON_REVENUE
        and self._input_data.revenue_per_kpi is None
    ):
      default_distribution = prior_distribution.PriorDistribution()
      default_roi_m_used = (
          self._model_spec.effective_media_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.roi_m, default_distribution.roi_m
          )
      )
      default_roi_rf_used = (
          self._model_spec.effective_rf_prior_type
          == constants.TREATMENT_PRIOR_TYPE_ROI
          and prior_distribution.distributions_are_equal(
              self._model_spec.prior.roi_rf, default_distribution.roi_rf
          )
      )
      # If ROI priors are used with the default prior distribution for all paid
      # channels (media and RF), then use the "total paid media contribution
      # prior" procedure.
      if (
          (default_roi_m_used and default_roi_rf_used)
          or (self.n_media_channels == 0 and default_roi_rf_used)
          or (self.n_rf_channels == 0 and default_roi_m_used)
      ):
        self._set_total_media_contribution_prior = True
        warnings.warn(
            "Consider setting custom ROI priors, as kpi_type was specified as"
            " `non_revenue` with no `revenue_per_kpi` being set. Otherwise, the"
            " total media contribution prior will be used with"
            f" `p_mean={constants.P_MEAN}` and `p_sd={constants.P_SD}`. Further"
            " documentation available at "
            " https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom#set-total-paid-media-contribution-prior",
        )
      elif self.n_media_channels > 0 and default_roi_m_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_M}` when"
            ' `media_prior_type` is "roi", custom priors are assigned on'
            ' `{constants.ROI_RF}` or `rf_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )
      elif self.n_rf_channels > 0 and default_roi_rf_used:
        raise ValueError(
            f"Custom priors should be set on `{constants.ROI_RF}` when"
            ' `rf_prior_type` is "roi", custom priors are assigned on'
            ' `{constants.ROI_M}` or `media_prior_type` is not "roi", KPI is'
            " non-revenue and revenue per kpi data is missing."
        )

  def _check_media_prior_support(self) -> None:
    """Checks ROI, mROI, and Contribution prior support when random effects are log-normal.

    Priors for ROI, mROI, and Contribution can only have negative support if the
    random effects follow a normal distribution. This check enforces that priors
    have non-negative support when random effects follow a log-normal
    distribution. This check only applies to geo-level models with log-normal
    random effects since national models do not have random effects.
    """
    prior = self._model_spec.prior
    if self.n_media_channels > 0:
      self._check_for_negative_support(
          prior.roi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
      )
      self._check_for_negative_support(
          prior.mroi_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
      )
      self._check_for_negative_support(
          prior.contribution_m,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_rf_channels > 0:
      self._check_for_negative_support(
          prior.roi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_ROI,
      )
      self._check_for_negative_support(
          prior.mroi_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_MROI,
      )
      self._check_for_negative_support(
          prior.contribution_rf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_organic_media_channels > 0:
      self._check_for_negative_support(
          prior.contribution_om,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )
    if self.n_organic_rf_channels > 0:
      self._check_for_negative_support(
          prior.contribution_orf,
          self.media_effects_dist,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      )

  def _check_for_negative_support(
      self,
      dist: backend.tfd.Distribution,
      media_effects_dist: str,
      prior_type: str,
  ) -> None:
    """Checks for negative support in prior distributions.

    When `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`, prior distributions
    for media effects must be non-negative. This function raises a ValueError if
    any part of the distribution's CDF is greater than 0 at 0, indicating some
    probability mass below zero.

    Args:
      dist: The distribution to check.
      media_effects_dist: The type of media effects distribution.
      prior_type: The prior type that corresponds with current prior under test.

    Raises:
      ValueError: If the prior distribution has negative support when
      `media_effects_dist` is `MEDIA_EFFECTS_LOG_NORMAL`.
    """
    if (
        prior_type == self._model_spec.media_prior_type
        and media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL
        and np.any(dist.cdf(0) > 0)
    ):
      raise ValueError(
          "Media priors must have non-negative support when"
          f' `media_effects_dist`="{media_effects_dist}". Found negative prior'
          f" distribution support for {dist.name}."
      )

  @functools.cached_property
  def prior_broadcast(self) -> prior_distribution.PriorDistribution:
    """Returns broadcasted `PriorDistribution` object."""
    total_spend = self._input_data.get_total_spend()
    # Total spend can have 1, 2 or 3 dimensions. Aggregate by channel.
    if len(total_spend.shape) == 1:
      # Already aggregated by channel.
      agg_total_spend = total_spend
    elif len(total_spend.shape) == 2:
      agg_total_spend = np.sum(total_spend, axis=(0,))
    else:
      agg_total_spend = np.sum(total_spend, axis=(0, 1))

    return self._model_spec.prior.broadcast(
        n_geos=self.n_geos,
        n_media_channels=self.n_media_channels,
        n_rf_channels=self.n_rf_channels,
        n_organic_media_channels=self.n_organic_media_channels,
        n_organic_rf_channels=self.n_organic_rf_channels,
        n_controls=self.n_controls,
        n_non_media_channels=self.n_non_media_channels,
        unique_sigma_for_each_geo=self.unique_sigma_for_each_geo,
        n_knots=self.knot_info.n_knots,
        is_national=self.is_national,
        set_total_media_contribution_prior=self._set_total_media_contribution_prior,
        kpi=np.sum(self._input_data.kpi.values),
        total_spend=agg_total_spend,
    )

  @functools.cached_property
  def adstock_decay_spec(self) -> adstock_hill.AdstockDecaySpec:
    """Returns `AdstockDecaySpec` object with correctly mapped channels."""
    if isinstance(self._model_spec.adstock_decay_spec, str):
      return adstock_hill.AdstockDecaySpec.from_consistent_type(
          self._model_spec.adstock_decay_spec
      )

    try:
      return self._create_adstock_decay_functions_from_channel_map(
          self._model_spec.adstock_decay_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `adstock_decay_spec` keys"
          f" {tuple(self._model_spec.adstock_decay_spec.keys())}. Keys should"
          " contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())}."
      ) from e

  def _create_adstock_decay_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> adstock_hill.AdstockDecaySpec:
    """Create `AdstockDecaySpec` from mapping from channels to decay functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      media_adstock_function = media_channel_builder(**channel_function_map)
    else:
      media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      rf_adstock_function = rf_channel_builder(**channel_function_map)
    else:
      rf_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_media_adstock_function = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_adstock_function = constants.GEOMETRIC_DECAY

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.GEOMETRIC_DECAY
      )
      organic_rf_adstock_function = organic_rf_channel_builder(
          **channel_function_map
      )
    else:
      organic_rf_adstock_function = constants.GEOMETRIC_DECAY

    return adstock_hill.AdstockDecaySpec(
        media=media_adstock_function,
        rf=rf_adstock_function,
        organic_media=organic_media_adstock_function,
        organic_rf=organic_rf_adstock_function,
    )

  @functools.cached_property
  def saturation_spec(self) -> SaturationSpec:
    """The SaturationSpec object with correctly mapped channels."""
    if isinstance(self._model_spec.saturation_spec, str):
      return SaturationSpec(
          media=self._model_spec.saturation_spec,
          rf=self._model_spec.saturation_spec,
          organic_media=self._model_spec.saturation_spec,
          organic_rf=self._model_spec.saturation_spec,
      )

    try:
      return self._create_saturation_functions_from_channel_map(
          self._model_spec.saturation_spec
      )
    except KeyError as e:
      raise ValueError(
          "Unrecognized channel names found in `saturation_spec` keys"
          f" {tuple(self._model_spec.saturation_spec.keys())}. Keys should"
          " contain only channel_names"
          f" {tuple(self._input_data.get_all_adstock_hill_channels().tolist())}."
      ) from e

  def _create_saturation_functions_from_channel_map(
      self, channel_function_map: Mapping[str, str]
  ) -> SaturationSpec:
    """Creates `SaturationSpec` from mapping from channels to saturation functions."""

    for channel in channel_function_map:
      if channel not in self._input_data.get_all_adstock_hill_channels():
        raise KeyError(f"Channel {channel} not found in data.")

    if self._input_data.media_channel is not None:
      media_channel_builder = self._input_data.get_paid_media_channels_argument_builder().with_default_value(
          constants.HILL
      )
      media_saturation = media_channel_builder(**channel_function_map)
    else:
      media_saturation = constants.HILL

    if self._input_data.rf_channel is not None:
      rf_channel_builder = self._input_data.get_paid_rf_channels_argument_builder().with_default_value(
          constants.HILL
      )
      rf_saturation = rf_channel_builder(**channel_function_map)
    else:
      rf_saturation = constants.HILL

    if self._input_data.organic_media_channel is not None:
      organic_media_channel_builder = self._input_data.get_organic_media_channels_argument_builder().with_default_value(
          constants.HILL
      )
      organic_media_saturation = organic_media_channel_builder(
          **channel_function_map
      )
    else:
      organic_media_saturation = constants.HILL

    if self._input_data.organic_rf_channel is not None:
      organic_rf_channel_builder = self._input_data.get_organic_rf_channels_argument_builder().with_default_value(
          constants.HILL
      )
      organic_rf_saturation = organic_rf_channel_builder(**channel_function_map)
    else:
      organic_rf_saturation = constants.HILL

    return SaturationSpec(
        media=media_saturation,
        rf=rf_saturation,
        organic_media=organic_media_saturation,
        organic_rf=organic_rf_saturation,
    )

  def create_inference_data_coords(
      self, n_chains: int, n_draws: int
  ) -> Mapping[str, np.ndarray | Sequence[str]]:
    """Creates data coordinates for inference data."""
    media_channel_names = (
        self.input_data.media_channel
        if self.input_data.media_channel is not None
        else np.array([])
    )
    rf_channel_names = (
        self.input_data.rf_channel
        if self.input_data.rf_channel is not None
        else np.array([])
    )
    organic_media_channel_names = (
        self.input_data.organic_media_channel
        if self.input_data.organic_media_channel is not None
        else np.array([])
    )
    organic_rf_channel_names = (
        self.input_data.organic_rf_channel
        if self.input_data.organic_rf_channel is not None
        else np.array([])
    )
    non_media_channel_names = (
        self.input_data.non_media_channel
        if self.input_data.non_media_channel is not None
        else np.array([])
    )
    control_variable_names = (
        self.input_data.control_variable
        if self.input_data.control_variable is not None
        else np.array([])
    )
    return {  # pyrefly: ignore[bad-return]
        constants.CHAIN: np.arange(n_chains),
        constants.DRAW: np.arange(n_draws),
        constants.GEO: self.input_data.geo,  # pyrefly: ignore[bad-assignment]
        constants.TIME: self.input_data.time,  # pyrefly: ignore[bad-assignment]
        constants.MEDIA_TIME: self.input_data.media_time,  # pyrefly: ignore[bad-assignment]
        constants.KNOTS: np.arange(self.knot_info.n_knots),
        constants.CONTROL_VARIABLE: control_variable_names,  # pyrefly: ignore[bad-assignment]
        constants.NON_MEDIA_CHANNEL: non_media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.MEDIA_CHANNEL: media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.RF_CHANNEL: rf_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.ORGANIC_MEDIA_CHANNEL: organic_media_channel_names,  # pyrefly: ignore[bad-assignment]
        constants.ORGANIC_RF_CHANNEL: organic_rf_channel_names,  # pyrefly: ignore[bad-assignment]
    }

  def create_inference_data_dims(self) -> Mapping[str, Sequence[str]]:
    """Creates data dimensions for inference data."""
    inference_dims = dict(constants.INFERENCE_DIMS)
    if self.unique_sigma_for_each_geo:
      inference_dims[constants.SIGMA] = [constants.GEO]  # pyrefly: ignore[unsupported-operation]
    else:
      inference_dims[constants.SIGMA] = []  # pyrefly: ignore[unsupported-operation]

    return {
        param: [constants.CHAIN, constants.DRAW] + list(dims)
        for param, dims in inference_dims.items()
    }

  def populate_cached_properties(self):
    """Eagerly activates all cached properties.

    This is useful for creating a `tf.function` computation graph with this
    Meridian object as part of a captured closure. Within the computation graph,
    internal state mutations are problematic, and so this method freezes the
    object's states before the computation graph is created.
    """
    cls = self.__class__
    # "Freeze" all @cached_property attributes by simply accessing them (with
    # `getattr()`).
    cached_properties = [
        attr
        for attr in dir(self)
        if isinstance(getattr(cls, attr, cls), functools.cached_property)
    ]
    for attr in cached_properties:
      _ = getattr(self, attr)

  def expand_selected_time_dims(
      self,
      start_date: tc.Date = None,
      end_date: tc.Date = None,
  ) -> list[str] | None:
    """Validates and returns time dimension values based on the selected times.

    If both `start_date` and `end_date` are None, returns None. If specified,
    both `start_date` and `end_date` are inclusive, and must be present in the
    time coordinates of the input data.

    Args:
      start_date: Start date of the selected time period. If None, implies the
        earliest time dimension value in the input data.
      end_date: End date of the selected time period. If None, implies the
        latest time dimension value in the input data.

    Returns:
      A list of time dimension values (as Meridian-formatted strings) in the
      input data within the selected time period, or do nothing and pass through
      None if both arguments are Nones, or if `start_date` and `end_date`
      correspond to the entire time range in the input data.

    Raises:
      ValueError if `start_date` or `end_date` is not in the input data time
      dimensions.
    """
    expanded = self.input_data.time_coordinates.expand_selected_time_dims(
        start_date=start_date, end_date=end_date
    )
    if expanded is None:
      return None
    return [date.strftime(constants.DATE_FORMAT) for date in expanded]

  def get_media_scaling_factor(
      self, channel_name: str
  ) -> backend.Tensor:
    """Retrieves the population-scaled median used to scale a channel's volume.

    For Reach & Frequency (RF) channels, this returns the scaling factor applied
    to the 'reach' component, as 'frequency' is not transformed.

    Args:
      channel_name: The string name of the paid or organic channel.

    Returns:
      A tensor of shape (n_geos,) representing the scaling factor.

    Raises:
      ValueError: If the channel is not found, or the transformer is
        uninitialized.
    """
    input_data = self.input_data

    if (
        input_data.non_media_channel is not None
        and channel_name in input_data.non_media_channel.values
    ):
      raise ValueError(
          "Cannot return a scaling factor for non-media treatment"
          f" '{channel_name}'."
      )

    if (
        input_data.control_variable is not None
        and channel_name in input_data.control_variable.values
    ):
      raise ValueError(
          "Cannot return a scaling factor for control variable"
          f" '{channel_name}'."
      )

    configs = [
        (
            input_data.media_channel,
            self.media_tensors.media_transformer,
            "media",
        ),
        (
            input_data.rf_channel,
            self.rf_tensors.reach_transformer,
            "RF",
        ),
        (
            input_data.organic_media_channel,
            self.organic_media_tensors.organic_media_transformer,
            "organic media",
        ),
        (
            input_data.organic_rf_channel,
            self.organic_rf_tensors.organic_reach_transformer,
            "organic RF",
        ),
    ]

    for channel_data, transformer, channel_type_name in configs:
      if channel_data is None or channel_name not in channel_data.values:
        continue

      if transformer is None:
        raise ValueError(
            f"Transformer for {channel_type_name} channel '{channel_name}'"
            " is missing."
        )
      (indices,) = np.where(channel_data.values == channel_name)
      idx = indices[0]
      return transformer.scale_factors_gm[:, idx]

    raise ValueError(f"Channel '{channel_name}' not found in any model inputs.")

  def get_channel_parameters(
      self, channel_name: str
  ) -> ChannelParameters:
    """Maps channel names to index, prefix, decay spec, and is_rf.

    Args:
      channel_name: Name of the channel.

    Returns:
      A ChannelParameters object containing the channel's metadata.

    Raises:
      ValueError: If the channel is not found.
    """
    input_data = self.input_data
    decay_spec = self.adstock_decay_spec

    configs = [
        (input_data.media_channel, "m", decay_spec.media, False),
        (input_data.rf_channel, "rf", decay_spec.rf, True),
        (
            input_data.organic_media_channel,
            "om",
            decay_spec.organic_media,
            False,
        ),
        (input_data.organic_rf_channel, "orf", decay_spec.organic_rf, True),
    ]

    for channel_data, prefix, channel_decay_spec, is_rf in configs:
      if channel_data is not None and channel_name in channel_data.values:
        index = list(channel_data.values).index(channel_name)
        return ChannelParameters(
            index=index,
            prefix=prefix,
            decay_spec=_get_decay(channel_decay_spec, index),
            is_rf=is_rf,
        )

    raise ValueError(f"Channel '{channel_name}' not found in the model.")

  def get_channel_parameter_tensor(
      self,
      dist_tensors: Any,
      *,
      param_base_name: str,
      channel_name: str,
  ) -> backend.Tensor:
    """Safely extracts a channel's parameter tensor (e.g., 'alpha', 'beta_g').

    Args:
      dist_tensors: An object containing batched distribution tensors (e.g.,
        DistributionTensors).
      param_base_name: The base name of the parameter (e.g., 'alpha', 'ec',
        'beta_g').
      channel_name: The name of the channel.

    Returns:
      The sliced parameter tensor for the specific channel.

    Raises:
      ValueError: If the parameter or channel is not found.
    """
    params = self.get_channel_parameters(channel_name)
    if param_base_name == constants.BETA_G:
      full_param_name = f"{param_base_name}{params.prefix}"
    else:
      full_param_name = f"{param_base_name}_{params.prefix}"
    try:
      tensor_block = getattr(dist_tensors, full_param_name)
    except AttributeError:
      raise ValueError(
          f"Parameter '{full_param_name}' not found in the distribution"
          " tensors."
      ) from None
    return tensor_block[..., params.index]
