"""Helpers to export data used in Meridian visualizations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Union, Sequence as Seq

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from meridian.analysis import analyzer
from meridian import constants as C


# ---------------------------------------------------------------------------
# Utility functions for working with Hill parameters
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, "tf.Tensor"]


def _to_numpy(x: ArrayLike | None) -> np.ndarray | None:
  """Convert ``tf.Tensor`` or Python containers to ``np.ndarray`` without copy."""
  if x is None:
    return None
  if isinstance(x, np.ndarray):
    return x
  if isinstance(x, (tf.Tensor, tf.Variable)):
    return x.numpy()
  return np.asarray(x)


def extract_hill_posterior(
    transformer: Any | None = None,
    *,
    ec: ArrayLike | None = None,
    slope: ArrayLike | None = None,
    channel: int | str = 0,
    channel_names: Seq[str] | None = None,
    flatten_samples: bool = True,
    quantiles: Seq[float] = (0.05, 0.5, 0.95),
) -> Dict[str, Any]:
  """Extract posterior samples and summaries for Hill ``ec`` and ``slope``.

  Parameters ``ec`` and ``slope`` are arrays whose final dimension indexes the
  media channel.  Leading dimensions correspond to arbitrary sample batches
  (e.g., ``[n_chains, n_draws]``).  ``channel`` may be an integer index or a
  channel name when ``channel_names`` is provided.
  """

  if transformer is not None:
    if ec is None and hasattr(transformer, "ec"):
      ec = getattr(transformer, "ec")
    if slope is None and hasattr(transformer, "slope"):
      slope = getattr(transformer, "slope")

  if ec is None or slope is None:
    raise ValueError(
        "Provide either a trained `transformer` or both `ec` and `slope` arrays/tensors."
    )

  ec_np = _to_numpy(ec)
  slope_np = _to_numpy(slope)

  if ec_np.shape != slope_np.shape:
    raise ValueError("`ec` and `slope` shapes must match; got " f"{ec_np.shape} vs {slope_np.shape}.")
  if ec_np.ndim < 1:
    raise ValueError("`ec`/`slope` must have at least 1 dimension; last dimension is channels.")

  n_channels = ec_np.shape[-1]

  if isinstance(channel, (str, bytes)):
    if channel_names is None:
      raise ValueError("When `channel` is a string, `channel_names` must be given.")
    try:
      channel_index = int(list(channel_names).index(channel))
    except ValueError as e:
      raise ValueError(
          f"Channel name {channel!r} not found in channel_names={list(channel_names)!r}."
      ) from e
  else:
    channel_index = int(channel)
    if channel_index < 0:
      channel_index += n_channels
    if not 0 <= channel_index < n_channels:
      raise ValueError(
          f"`channel` index out of range: {channel_index} for n_channels={n_channels}."
      )

  ec_ch = ec_np[..., channel_index]
  slope_ch = slope_np[..., channel_index]

  if flatten_samples:
    ec_samples = ec_ch.reshape(-1)
    slope_samples = slope_ch.reshape(-1)
  else:
    ec_samples = ec_ch
    slope_samples = slope_ch

  def _summ(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=0)),
        "quantiles": {float(q): float(np.quantile(x, q)) for q in quantiles},
    }

  return {
      "channel_index": channel_index,
      "channel_name": (list(channel_names)[channel_index] if channel_names is not None else None),
      "samples": {"ec": ec_samples, "slope": slope_samples},
      "summary": {"ec": _summ(ec_samples), "slope": _summ(slope_samples)},
  }


def hill_curve_quantiles(
    media_grid: ArrayLike,
    ec_samples: ArrayLike,
    slope_samples: ArrayLike,
    quantiles: Seq[float] = (0.05, 0.5, 0.95),
) -> Dict[str, Any]:
  """Evaluate Hill curves for posterior samples and summarise over a media grid."""
  m = np.atleast_1d(_to_numpy(media_grid)).astype(float)
  if m.ndim != 1:
    raise ValueError("`media_grid` must be 1D.")
  if np.any(m < 0):
    raise ValueError("`media_grid` must be non-negative.")

  ec_s = np.ravel(_to_numpy(ec_samples)).astype(float)
  sl_s = np.ravel(_to_numpy(slope_samples)).astype(float)
  if ec_s.shape != sl_s.shape:
    raise ValueError("`ec_samples` and `slope_samples` must have the same length.")

  log_ec = np.log(np.maximum(ec_s, 1e-300))[:, None]
  log_m = np.where(m[None, :] > 0.0, np.log(m[None, :]), -np.inf)
  logit = sl_s[:, None] * (log_ec - log_m)

  out = np.empty_like(logit)
  pos = logit >= 0
  neg = ~pos
  out[pos] = 1.0 / (1.0 + np.exp(logit[pos]))
  expx = np.exp(logit[neg])
  out[neg] = expx / (1.0 + expx)

  zero_mask = (m == 0.0)[None, :]
  if np.any(zero_mask):
    sl_b = sl_s[:, None]
    y_zero = np.where(sl_b == 0.0, 0.5, 0.0)
    out = np.where(zero_mask, y_zero, out)

  mean = np.mean(out, axis=0)
  qdict = {float(q): np.quantile(out, q, axis=0) for q in quantiles}
  return {"media": m, "mean": mean, "quantiles": qdict}


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
  # Construct a float32 frequency grid to avoid mixed-precision errors in
  # ``Analyzer.optimal_freq``.  Use the maximum frequency from the model data when
  # available, otherwise fall back to a reasonable default.
  rf_freq = getattr(rf_tensors, "frequency", None)
  max_frequency = None
  if rf_freq is not None:
    try:
      max_frequency = float(tf.reduce_max(tf.cast(rf_freq, tf.float32)).numpy())
    except Exception:  # numpy arrays or other array-like
      max_frequency = float(np.max(np.array(rf_freq, dtype=np.float32)))
  if not max_frequency or not np.isfinite(max_frequency):
    max_frequency = 50.0
  freq_grid = np.arange(1.0, max_frequency, 0.1, dtype=np.float32)

  rf_ds: xr.Dataset = ana.optimal_freq(
      new_data=new_data,
      freq_grid=freq_grid,
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
