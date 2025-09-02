"""Helpers to export data used in Meridian visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import tensorflow as tf
import xarray as xr

from meridian.analysis import analyzer
from meridian import constants as C


# ---------------------------------------------------------------------------
# 1. Curve-parameter (Hill/response-curve) data
# ---------------------------------------------------------------------------
def get_curve_parameter_data(
    mmm,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
  """Returns a tidy table of Hill curve data."""
  ana = analyzer.Analyzer(mmm)
  return ana.hill_curves(confidence_level=confidence_level)


# ---------------------------------------------------------------------------
# 2. Budget-optimisation (optimal frequency) data
# ---------------------------------------------------------------------------
def get_budget_optimisation_data(
    mmm,
    *,
    selected_channels: Sequence[str] | None = None,
    selected_times: Sequence[str | int] | None = None,
    use_kpi: bool = False,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
  """Returns optimal frequency metrics as a table."""
  ana = analyzer.Analyzer(mmm)

  rf_tensors = getattr(mmm, "rf_tensors", None)
  input_data = getattr(mmm, "input_data", None)
  new_data = analyzer.DataTensors(
      rf_impressions=tf.cast(rf_tensors.rf_impressions, tf.float32)
      if getattr(rf_tensors, "rf_impressions", None) is not None
      else None,
      rf_spend=tf.cast(rf_tensors.rf_spend, tf.float32)
      if getattr(rf_tensors, "rf_spend", None) is not None
      else None,
      revenue_per_kpi=tf.cast(input_data.revenue_per_kpi, tf.float32)
      if getattr(input_data, "revenue_per_kpi", None) is not None
      else None,
  )

  rf_ds: xr.Dataset = ana.optimal_freq(
      new_data=new_data,
      selected_times=selected_times,
      use_kpi=use_kpi,
      confidence_level=confidence_level,
  )

  channels = (
      selected_channels if selected_channels is not None else rf_ds.rf_channel.values
  )

  perf_df = (
      rf_ds[[C.ROI]]
      .sel(metric=[C.MEAN])
      .sel(rf_channel=channels)
      .to_dataframe()
      .reset_index()
      .pivot(index=[C.RF_CHANNEL, C.FREQUENCY], columns=C.METRIC, values=C.ROI)
      .reset_index()
      .rename(columns={C.MEAN: C.ROI})
  )

  opt_df = (
      rf_ds[[C.OPTIMAL_FREQUENCY]].sel(rf_channel=channels).to_dataframe().reset_index()
  )

  return perf_df.merge(opt_df, on=C.RF_CHANNEL)


# ---------------------------------------------------------------------------
# 3. Actual-vs-fitted outcome data
# ---------------------------------------------------------------------------
def get_actual_vs_fitted_data(
    mmm,
    *,
    confidence_level: float = 0.90,
    aggregate_geos: bool = False,
    aggregate_times: bool = False,
    selected_geos: Sequence[str] | None = None,
    selected_times: Sequence[str | int] | None = None,
) -> pd.DataFrame:
  """Returns outcome data comparing actual and expected values."""
  ana = analyzer.Analyzer(mmm)
  fit_ds: xr.Dataset = ana.expected_vs_actual_data(
      confidence_level=confidence_level,
      aggregate_geos=aggregate_geos,
      aggregate_times=aggregate_times,
      selected_geos=selected_geos,
      selected_times=selected_times,
  )

  return fit_ds.to_dataframe().reset_index()


# ---------------------------------------------------------------------------
# 4. Actual-vs-fitted outcome data (API-compatible wrapper)
# ---------------------------------------------------------------------------
def get_actual_vs_fitted_data_fixed(
    mmm,
    *,
    confidence_level: float = 0.90,
    aggregate_geos: bool = False,
    aggregate_times: bool = False,
    selected_times: Sequence[str | int] | None = None,
) -> pd.DataFrame:
  """Wrapper for ``Analyzer.expected_vs_actual_data`` removing deprecated args."""

  ana = analyzer.Analyzer(mmm)
  fit_ds: xr.Dataset = ana.expected_vs_actual_data(
      confidence_level=confidence_level,
      aggregate_geos=aggregate_geos,
      aggregate_times=aggregate_times,
  )

  if selected_times is not None:
    fit_ds = fit_ds.sel(time=selected_times)

  return fit_ds.to_dataframe().reset_index()


if __name__ == "__main__":
  curves = get_curve_parameter_data(mmm)
  curves.to_csv("hill_curve_parameters.csv", index=False)

  opt = get_budget_optimisation_data(mmm, selected_channels=["YouTube"])
  opt.to_csv("rf_budget_optimisation.csv", index=False)

  avf = get_actual_vs_fitted_data_fixed(mmm, aggregate_geos=True)
  avf.to_csv("actual_vs_fitted.csv", index=False)
