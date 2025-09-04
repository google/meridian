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
  """Returns optimal-frequency ROI (RF) and ROI for non-RF paid media as one table.

  Output schema (preserved):
    - rf_channel: channel name (applies to both RF and non-RF)
    - frequency: RF frequency grid value (NaN for non-RF rows)
    - roi: mean ROI
    - optimal_frequency: channel-level optimal frequency (NaN for non-RF rows)

  Notes:
    - For RF channels we compute ROI across the provided frequency grid and attach
      that channel's optimal_frequency on each row.
    - For non-RF paid media channels we compute ROI from `Analyzer.summary_metrics`
      (mean), evaluated at the RF channels' optimal frequency (so the system is
      consistent), and add a single row per non-RF channel with frequency/optimal_frequency = NaN.
  """
  ana = analyzer.Analyzer(mmm)

  # Build a minimal DataTensors consistent with the original code, so both
  # optimal_freq() and summary_metrics() can reuse mmm data where not provided.
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

  # Frequency grid for RF part (if any RF channels exist).
  rf_freq = getattr(rf_tensors, "frequency", None)
  max_frequency = None
  if rf_freq is not None:
    try:
      max_frequency = float(tf.reduce_max(tf.cast(rf_freq, tf.float32)).numpy())
    except Exception:
      max_frequency = float(np.max(np.array(rf_freq, dtype=np.float32)))
  if not max_frequency or not np.isfinite(max_frequency):
    max_frequency = 50.0
  freq_grid = np.arange(1.0, max_frequency, 0.1, dtype=np.float32)

  tables: list[pd.DataFrame] = []

  # ---- 1) RF block: ROI over frequency grid + optimal_frequency per RF channel.
  rf_ds = None
  rf_available: list[str] = []
  try:
    rf_ds = ana.optimal_freq(
        new_data=new_data,
        freq_grid=freq_grid,
        selected_times=selected_times,
        use_kpi=use_kpi,
        confidence_level=confidence_level,
    )
    rf_available = [str(c) for c in rf_ds.coords[C.RF_CHANNEL].values]
  except Exception:
    rf_ds = None
    rf_available = []

  # Select RF channels, tolerating unknown labels.
  if selected_channels is None:
    rf_selected = rf_available
  else:
    rf_selected = [c for c in selected_channels if c in rf_available]

  # If we have RF data and selection yields channels, build the RF table.
  opt_freq_vec = None
  if rf_ds is not None and rf_available:
    # ROI(grid) for selected RF channels.
    if rf_selected:
      perf_df = (
          rf_ds[[C.ROI]]
          .sel(metric=[C.MEAN])
          .sel(rf_channel=rf_selected)
          .to_dataframe()
          .reset_index()
          .pivot(index=[C.RF_CHANNEL, C.FREQUENCY], columns=C.METRIC, values=C.ROI)
          .reset_index()
          .rename(columns={C.MEAN: C.ROI})
      )
      opt_df = (
          rf_ds[[C.OPTIMAL_FREQUENCY]]
          .sel(rf_channel=rf_selected)
          .to_dataframe()
          .reset_index()
      )
      rf_table = perf_df.merge(opt_df, on=C.RF_CHANNEL)
      tables.append(rf_table)

    # Keep the full optimal-frequency vector (for all RF channels) to make
    # summary_metrics internally consistent; do not subset here.
    opt_freq_vec = np.asarray(rf_ds[C.OPTIMAL_FREQUENCY].values, dtype=np.float32)

  # ---- 2) Non-RF paid media block: ROI at (RF) optimal frequencies.
  # Compute summary metrics with optimal_frequency applied (if known).
  sum_ds = ana.summary_metrics(
      new_data=new_data,
      optimal_frequency=opt_freq_vec,
      selected_times=selected_times,
      use_kpi=use_kpi,
      confidence_level=confidence_level,
      include_non_paid_channels=False,
  )

  # All paid channels (media + rf) live on `channel`. Remove aggregate total.
  all_paid = [str(c) for c in sum_ds.coords[C.CHANNEL].values]
  non_total = [c for c in all_paid if c != C.ALL_CHANNELS]

  # Non-RF = paid channels that are not in rf_available.
  non_rf_channels = [c for c in non_total if c not in rf_available]
  if selected_channels is not None:
    non_rf_channels = [c for c in non_rf_channels if c in selected_channels]

  if non_rf_channels:
    mdf = (
        sum_ds[[C.ROI]]
        .sel(metric=[C.MEAN])
        .sel(channel=non_rf_channels)
        .to_dataframe()
        .reset_index()
        .pivot(index=[C.CHANNEL], columns=C.METRIC, values=C.ROI)
        .reset_index()
        .rename(columns={C.MEAN: C.ROI})
    )
    mdf[C.FREQUENCY] = np.nan
    mdf[C.OPTIMAL_FREQUENCY] = np.nan
    mdf = mdf.rename(columns={C.CHANNEL: C.RF_CHANNEL})
    mdf = mdf[[C.RF_CHANNEL, C.FREQUENCY, C.ROI, C.OPTIMAL_FREQUENCY]]
    tables.append(mdf)

  # ---- 3) Combine.
  if tables:
    out = pd.concat(tables, ignore_index=True)
  else:
    out = pd.DataFrame(columns=[C.RF_CHANNEL, C.FREQUENCY, C.ROI, C.OPTIMAL_FREQUENCY])

  return out


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


# ---------------------------------------------------------------------------
# High-level, notebook-friendly Hill helpers
# ---------------------------------------------------------------------------

def _autodetect_hill_params_and_names(mmm):
  """Best-effort discovery of Hill params (ec, slope) and channel names on `mmm`."""
  ht = getattr(mmm, "hill_transformer", None) or getattr(mmm, "hill", None)
  ec = getattr(ht, "ec", None) if ht is not None else getattr(mmm, "ec", None)
  slope = getattr(ht, "slope", None) if ht is not None else getattr(mmm, "slope", None)

  channel_names = (
      getattr(mmm, "media_channel_names", None)
      or getattr(mmm, "channel_names", None)
      or getattr(getattr(mmm, "input_data", None), "media_channel_names", None)
  )
  try:
    if channel_names is not None:
      channel_names = list(map(str, list(channel_names)))
  except Exception:
    channel_names = None

  return ec, slope, channel_names


def hill_for_channel(
    mmm,
    channel: int | str,
    *,
    media_grid: ArrayLike | str | None = "auto",
    quantiles: Seq[float] = (0.05, 0.5, 0.95),
    flatten_samples: bool = True,
    grid_multiplier: float = 5.0,
    grid_points: int = 200,
    channel_names: Seq[str] | None = None,
) -> Dict[str, Any]:
  """All-in-one: posterior + curve for a single channel.

  Returns:
    {
      "channel_index": int,
      "channel_name": Optional[str],
      "posterior": <output of extract_hill_posterior>,
      "curve": Optional[dict(media, mean, quantiles)],
      "curve_df": Optional[pd.DataFrame]
    }
  """
  ec, slope, auto_names = _autodetect_hill_params_and_names(mmm)
  if channel_names is None:
    channel_names = auto_names

  post = extract_hill_posterior(
      transformer=None,
      ec=ec,
      slope=slope,
      channel=channel,
      channel_names=channel_names,
      flatten_samples=flatten_samples,
      quantiles=quantiles,
  )

  # Choose a sensible grid if requested
  curve = None
  curve_df = None
  if isinstance(media_grid, str) and media_grid == "auto":
    median_ec = post["summary"]["ec"]["quantiles"].get(0.5, float(np.median(post["samples"]["ec"])))
    grid_max = float(grid_multiplier * median_ec) if np.isfinite(median_ec) and median_ec > 0 else 100.0
    media_grid = np.linspace(0.0, grid_max, grid_points, dtype=float)

  if media_grid is not None:
    curve = hill_curve_quantiles(
        media_grid=media_grid,
        ec_samples=post["samples"]["ec"],
        slope_samples=post["samples"]["slope"],
        quantiles=quantiles,
    )
    # Tidy frame with conventional column names
    cols = {
        "media": curve["media"],
        "response_mean": curve["mean"],
    }
    for q in quantiles:
        cols[f"response_q{int(round(100*q)):02d}"] = curve["quantiles"][float(q)]
    curve_df = pd.DataFrame(cols)

  return {
      "channel_index": post["channel_index"],
      "channel_name": post["channel_name"],
      "posterior": post,
      "curve": curve,
      "curve_df": curve_df,
  }


def hill_curve_df(mmm, channel: int | str, **kwargs) -> pd.DataFrame:
  """One-liner: just give me a tidy curve DataFrame for a channel."""
  out = hill_for_channel(mmm, channel, **kwargs)
  if out["curve_df"] is None:
    raise ValueError("No curve grid was produced. Pass media_grid='auto' or an array.")
  return out["curve_df"]


def plot_hill(
    mmm,
    channel: int | str,
    *,
    media_grid: ArrayLike | str | None = "auto",
    quantiles: Seq[float] = (0.05, 0.95),
    title: Optional[str] = None,
    ax: Any | None = None,
    **kwargs,
):
  """One-liner plot: mean + interval for a channel."""
  try:
    import matplotlib.pyplot as plt
  except Exception as e:
    raise ImportError("matplotlib is required for plot_hill(...)") from e

  # Ensure we also compute the median for labeling even if not requested
  qs = sorted(set([0.5, *quantiles]))
  out = hill_for_channel(mmm, channel, media_grid=media_grid, quantiles=qs, **kwargs)
  curve = out["curve"]
  if curve is None:
    raise ValueError("No curve data available; supply media_grid='auto' or an explicit grid.")

  low, high = min(quantiles), max(quantiles)
  if ax is None:
    _, ax = plt.subplots(figsize=(6, 4))

  ax.plot(curve["media"], curve["mean"], label="Mean response")
  ax.fill_between(curve["media"], curve["quantiles"][float(low)], curve["quantiles"][float(high)],
                  alpha=0.25, label=f"{int(round((high-low)*100))}% interval")

  ax.set_xlabel("Media (same scale as Hill input)")
  ax.set_ylabel("Saturated response (0..1)")
  label = out["channel_name"] if out["channel_name"] is not None else f"channel {out['channel_index']}"
  ax.set_title(title or f"Hill curve – {label}")
  ax.legend()
  return ax


if __name__ == "__main__":
  curves = get_curve_parameter_data(mmm)
  curves.to_csv("hill_curve_parameters.csv", index=False)
  rf_tensors = getattr(mmm, "rf_tensors", None)
  has_rf = rf_tensors is not None and (
      getattr(rf_tensors, "rf_impressions", None) is not None
      or getattr(rf_tensors, "rf_spend", None) is not None
  )
  if has_rf:
    opt = get_budget_optimisation_data(mmm, selected_channels=["YouTube"])
    opt.to_csv("rf_budget_optimisation.csv", index=False)
  else:
    print(
        "Skipping RF budget optimisation export: no reach/frequency tensors on the model."
    )

  avf = get_actual_vs_fitted_data_fixed(mmm, aggregate_geos=True)
  avf.to_csv("actual_vs_fitted.csv", index=False)
