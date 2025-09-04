"""Convenience helpers to run common notebook snippets with one call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Optional

import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from meridian import constants as C

c = C  # convenience alias
from meridian.analysis import analyzer

try:
  # If helpers.py is part of a package (e.g., mypkg.helpers)
  from . import betas, diagnostics, values
except ImportError:
  # If helpers.py is imported as a top-level module
  import importlib
  import os
  import sys

  sys.path.append(os.path.dirname(__file__))
  betas = importlib.import_module("betas")
  diagnostics = importlib.import_module("diagnostics")
  values = importlib.import_module("values")


# ---------------------------------------------------------------------------
# 1. Beta transformations and plots
# ---------------------------------------------------------------------------

def _resolve_channel(name: str, media_in_model: Mapping[str, str] | None) -> str:
  """Resolve ``name`` via ``media_in_model`` when available."""
  if media_in_model and name in media_in_model:
    return media_in_model[name]
  return name


def helper_view_transformed_variable(
    mmm: Any,
    feature_key: str,
    media_in_model: Mapping[str, str] | None = None,
    **kwargs: Any,
):
  """Return transformed data and chart for a media variable."""
  channel = _resolve_channel(feature_key, media_in_model)
  return betas.view_transformed_variable(mmm, channel=channel, **kwargs)


def helper_view_transformed_and_raw_variable(
    mmm: Any,
    feature_key: str,
    media_in_model: Mapping[str, str] | None = None,
    **kwargs: Any,
):
  """Like :func:`helper_view_transformed_variable` but also returns raw values."""
  channel = _resolve_channel(feature_key, media_in_model)
  return betas.view_transformed_variable(
      mmm, channel=channel, show_raw=True, **kwargs
  )


def helper_plot_posterior_coef(
    mmm: Any,
    channels: Sequence[str],
    media_in_model: Mapping[str, str] | None = None,
):
  """Plot posterior coefficients for ``channels``."""
  labels = [_resolve_channel(ch, media_in_model) for ch in channels]
  all_channels = betas.get_beta_channel_names(mmm)
  idxs = [all_channels.index(ch) for ch in labels]

  rows = []
  for ch, i in zip(labels, idxs):
    log_chart = betas.plot_posterior_coef(mmm, c.BETA_M, i).properties(
        title=f"{ch} — {c.BETA_M}"
    )
    if hasattr(mmm.inference_data.posterior, c.BETA_GM):
      lin_chart = betas.plot_posterior_coef(mmm, c.BETA_GM, i).properties(
          title=f"{ch} — {c.BETA_GM}"
      )
      rows.append(alt.hconcat(log_chart, lin_chart))
    else:
      rows.append(log_chart)
  return alt.vconcat(*rows).resolve_scale(x="independent", y="independent")


# ---------------------------------------------------------------------------
# 2. t-statistics and related diagnostics
# ---------------------------------------------------------------------------

def helper_inspect_t_stat(
    mmm: Any,
    feature_key: str,
    media_in_model: Mapping[str, str] | None = None,
    *,
    scale: str = "linear",
    geo_index: int | None = None,
    **kwargs: Any,
) -> pd.Series:
  """Compute t-statistics for ``feature_key``."""
  channel_name = _resolve_channel(feature_key, media_in_model)
  all_channels = betas.get_beta_channel_names(mmm)
  channel_idx = all_channels.index(channel_name)
  stats = betas.inspect_t_stat(
      mmm,
      channel_index=channel_idx,
      scale=scale,
      geo_index=geo_index,
      **kwargs,
  )
  return pd.Series(stats, name=channel_name)


def helper_plot_normal_posterior_coef(
    mmm: Any,
    channels: Sequence[str],
    media_in_model: Mapping[str, str] | None = None,
    *,
    channel_dim: str = "media_channel",
):
  """Plot z-scored normal posterior coefficients for ``channels``."""
  labels = [_resolve_channel(ch, media_in_model) for ch in channels]
  all_channels = betas.get_beta_channel_names(mmm)
  idxs = [all_channels.index(ch) for ch in labels]

  def z_chart(var_name: str, ch_idx: int, title: str) -> alt.Chart:
    da = mmm.inference_data.posterior[var_name]
    label = da.coords[channel_dim].values[ch_idx]
    stats = diagnostics.posterior_summary(
        mmm.inference_data, var_name, coord={channel_dim: label}
    )
    samples = da.sel({channel_dim: label}).values.reshape(-1)
    z = (samples - stats["mean"]) / (stats["sd"] + 1e-12)
    df = pd.DataFrame({"z": z})
    density = (
        alt.Chart(df)
        .transform_density("z", as_=["z", "density"])
        .mark_area()
        .encode(x="z:Q", y="density:Q")
        .properties(title=title)
    )
    zero_line = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule().encode(x="z:Q")
    return density + zero_line

  has_beta_gm = c.BETA_GM in mmm.inference_data.posterior

  rows = []
  for ch, i in zip(labels, idxs):
    left = z_chart(c.BETA_M, i, f"{ch} — {c.BETA_M} (z)")
    if has_beta_gm:
      right = z_chart(c.BETA_GM, i, f"{ch} — {c.BETA_GM} (z)")
      rows.append(alt.hconcat(left, right))
    else:
      rows.append(left)
  return alt.vconcat(*rows).resolve_scale(x="independent", y="independent")


def helper_inspect_normal_t_stat(
    mmm: Any,
    feature_key: str,
    media_in_model: Mapping[str, str] | None = None,
    *,
    channel_dim: str = "media_channel",
    var_name: str = c.BETA_M,
) -> pd.Series:
  """Normal-scale t-statistic and p-value for ``feature_key``."""
  from math import erf, sqrt

  channel_name = _resolve_channel(feature_key, media_in_model)
  s = diagnostics.posterior_summary(
      mmm.inference_data, var_name=var_name, coord={channel_dim: channel_name}
  )

  def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

  t_stat = s["t_stat"]
  p_value = 2.0 * (1.0 - _phi(abs(t_stat)))
  return pd.Series(
    {"t_stat": t_stat, "p_value": p_value, "mean": s["mean"], "sd": s["sd"]},
    name=channel_name,
  )


# ---------------------------------------------------------------------------
# 3. Multicollinearity and residual diagnostics
# ---------------------------------------------------------------------------

def helper_compute_vif(
    mmm: Any,
    media_in_model: Mapping[str, str] | None = None,
) -> pd.Series:
  """Compute VIF values for regressors used by ``mmm``."""
  media_da = None
  for cand in ("media", "media_impressions", "media_predictors"):
    if hasattr(mmm.input_data, cand):
      media_da = getattr(mmm.input_data, cand)
      break
  if media_da is None:
    raise AttributeError("Could not locate media predictors on mmm.input_data.")

  controls_da = None
  for cand in ("controls", "control_predictors", "organic"):
    if hasattr(mmm.input_data, cand):
      controls_da = getattr(mmm.input_data, cand)
      break

  X = diagnostics.build_design_matrix(media_da, controls_da, add_constant=True)
  rename_map = {k: v for k, v in (media_in_model or {}).items() if k in X.columns}
  if rename_map:
    X = X.rename(columns=rename_map)
  X_noconst = X.drop(columns=["const"], errors="ignore")
  vif = diagnostics.compute_vif(X_noconst)
  return vif.sort_values(ascending=False)


def helper_durbin_watson(
    mmm: Any,
    *,
    aggregate_geos: bool = False,
    aggregate_times: bool = False,
    by: Iterable[str] | None = None,
    time_dim: str | None = None,
):
  """Durbin–Watson statistic for model residuals."""
  an = analyzer.Analyzer(mmm)
  ds = an.expected_vs_actual_data(
      aggregate_geos=aggregate_geos, aggregate_times=aggregate_times
  )
  yobs = ds["actual"]
  yhat = ds["expected"].sel(metric=C.MEAN)
  yobs, yhat = xr.align(yobs, yhat, join="inner")
  dw = diagnostics.durbin_watson((yobs, yhat), time_dim=time_dim, by=by, reduce_over=())
  return dw if by is None else dw.to_pandas()


# ---------------------------------------------------------------------------
# 4. Probability mass and Hill curve utilities
# ---------------------------------------------------------------------------

def helper_mass_above_threshold(
    mmm: Any,
    feature_key: str,
    media_in_model: Mapping[str, str] | None = None,
    *,
    var_name: str = C.BETA_M,
    channel_dim: str = "media_channel",
    cred_mass: float = 0.95,
) -> dict[str, Any]:
  """Probability mass above threshold on lognormal and normal scales."""
  from meridian import constants as constants

  idata = mmm.inference_data
  channel_label = _resolve_channel(feature_key, media_in_model)
  coord = {channel_dim: channel_label}
  res_logn = diagnostics.mass_above_threshold(
      idata, var_name, coord=coord, cred_mass=cred_mass, log_space=True
  )
  res_norm = diagnostics.mass_above_threshold(
      idata, var_name, coord=coord, cred_mass=cred_mass, threshold=0.0, log_space=False
  )
  return {"lognormal": res_logn, "normal": res_norm}


def helper_get_curve_parameter_data(
    mmm: Any,
    *,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
  """Return Hill/response-curve parameter table."""
  return values.get_curve_parameter_data(mmm, confidence_level=confidence_level)


def helper_get_curve_parameter_values(
    mmm: Any,
    channel_friendly: str,
    media_in_model: Mapping[str, str] | None = None,
):
  """Extract Hill parameter draws or summaries for ``channel_friendly``."""
  from .values import _autodetect_hill_params_and_names, extract_hill_posterior

  friendly_to_raw = {v: k for k, v in (media_in_model or {}).items()}
  ec, slope, channel_names = _autodetect_hill_params_and_names(mmm)
  candidates = [channel_friendly]
  if channel_friendly in friendly_to_raw:
    candidates.append(friendly_to_raw[channel_friendly])

  if (ec is not None) and (slope is not None):
    chan = None
    if channel_names:
      for c in candidates:
        if c in channel_names:
          chan = c
          break
    if chan is None:
      chan = candidates[0]
    post = extract_hill_posterior(
        ec=ec, slope=slope, channel=chan, channel_names=channel_names
    )
    summary = post["summary"]
    samples = pd.DataFrame(post["samples"])
    return {
        "channel": post["channel_name"] or f"#{post['channel_index']}",
        "summary": summary,
        "samples": samples,
    }

  curves = values.get_curve_parameter_data(mmm)
  ch_col = next(
      (c for c in ["channel", "rf_channel", "media_channel", "Channel"] if c in curves.columns),
      None,
  )
  if ch_col is None:
    raise ValueError("No channel column in hill_curves table.")
  mask = pd.Series(False, index=curves.index)
  for cnd in candidates:
    mask = mask | curves[ch_col].astype(str).eq(cnd)
  return curves.loc[mask]


# ---------------------------------------------------------------------------
# 5. Optimisation and actual vs fitted
# ---------------------------------------------------------------------------

def helper_get_optimized_data(
    mmm: Any,
    *,
    desired: Sequence[str] | None = None,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
  """Return ROI tables for selected channels."""
  import warnings

  ana = analyzer.Analyzer(mmm)

  rf_available: list[str] = []
  freq_grid = None
  try:
    rf_tensors = getattr(mmm, "rf_tensors", None)
    rf_freq = getattr(rf_tensors, "frequency", None)
    mf = None
    if rf_freq is not None:
      try:
        mf = float(tf.reduce_max(tf.cast(rf_freq, tf.float32)).numpy())
      except Exception:
        mf = float(np.max(np.array(rf_freq, dtype=np.float32)))
    if not mf or not np.isfinite(mf) or mf <= 1.0:
      mf = 50.0
    freq_grid = np.arange(1.0, mf, 0.1, dtype=np.float32)
    rf_probe = ana.optimal_freq(freq_grid=freq_grid, use_kpi=False, confidence_level=confidence_level)
    rf_available = [str(c) for c in rf_probe.coords[C.RF_CHANNEL].values]
  except Exception:
    rf_available = []

  sum_ds = ana.summary_metrics(include_non_paid_channels=False, use_kpi=False, confidence_level=confidence_level)
  paid_channels = [str(c) for c in sum_ds.coords[C.CHANNEL].values if str(c) != C.ALL_CHANNELS]
  non_rf_paid = [c for c in paid_channels if c not in rf_available]

  if desired is None:
    sel_non_rf = non_rf_paid
    sel_rf = rf_available
  else:
    sel_non_rf = [c for c in desired if c in non_rf_paid]
    sel_rf = [c for c in desired if c in rf_available]
    if not sel_non_rf and not sel_rf:
      warnings.warn("None of the desired channels are present; using all paid channels.")
      sel_non_rf = non_rf_paid
      sel_rf = rf_available

  non_rf_full_df = pd.DataFrame()
  if sel_non_rf:
    df = sum_ds[[C.ROI]].to_dataframe().reset_index()
    metrics_avail = set(df[C.METRIC].astype(str).unique())
    metrics_keep = [m for m in [C.MEAN, C.MEDIAN, C.CI_LO, C.CI_HI] if m in metrics_avail]
    sub = df[(df[C.CHANNEL].isin(sel_non_rf)) & (df[C.METRIC].isin(metrics_keep))].copy()
    id_cols = [c for c in sub.columns if c not in {C.METRIC, C.ROI}]
    wide = (
        sub
        .pivot_table(index=id_cols, columns=C.METRIC, values=C.ROI, aggfunc="mean")
        .reset_index()
    )
    rename_map = {
        C.MEAN: "roi_mean",
        C.MEDIAN: "roi_median",
        C.CI_LO: "roi_ci_lo",
        C.CI_HI: "roi_ci_hi",
    }
    wide = wide.rename(columns=rename_map)
    wide = wide.rename(columns={C.CHANNEL: C.RF_CHANNEL})
    wide[C.FREQUENCY] = np.nan
    wide[C.OPTIMAL_FREQUENCY] = np.nan
    roi_cols = [c for c in ["roi_mean", "roi_median", "roi_ci_lo", "roi_ci_hi"] if c in wide.columns]
    other_id_cols = [c for c in id_cols if c not in {C.CHANNEL}]
    col_order = [C.RF_CHANNEL, *[c for c in other_id_cols if c != C.RF_CHANNEL], *roi_cols, C.FREQUENCY, C.OPTIMAL_FREQUENCY]
    non_rf_full_df = wide[[c for c in col_order if c in wide.columns]]

  rf_df = pd.DataFrame(columns=[C.RF_CHANNEL, C.FREQUENCY, C.ROI, C.OPTIMAL_FREQUENCY])
  if sel_rf:
    rf_ds = ana.optimal_freq(freq_grid=freq_grid, use_kpi=False, confidence_level=confidence_level)
    perf = rf_ds[[C.ROI]].to_dataframe().reset_index()
    perf = perf[(perf[C.METRIC] == C.MEAN) & (perf[C.RF_CHANNEL].isin(sel_rf))]
    perf = perf[[C.RF_CHANNEL, C.FREQUENCY, C.ROI]]
    opt = rf_ds[[C.OPTIMAL_FREQUENCY]].to_dataframe().reset_index()
    opt = opt[opt[C.RF_CHANNEL].isin(sel_rf)][[C.RF_CHANNEL, C.OPTIMAL_FREQUENCY]].drop_duplicates()
    rf_df = perf.merge(opt, on=C.RF_CHANNEL, how="left")

  maybe_roi_cols = [c for c in ["roi_mean", "roi_median", "roi_ci_lo", "roi_ci_hi"] if c in non_rf_full_df.columns]
  if not non_rf_full_df.empty:
    rf_df = rf_df.rename(columns={C.ROI: "roi_mean"})
    for col in maybe_roi_cols:
      if col not in rf_df.columns:
        rf_df[col] = np.nan
    wanted_cols = [C.RF_CHANNEL] + [c for c in non_rf_full_df.columns if c not in [C.RF_CHANNEL]]
    rf_df = rf_df[[c for c in wanted_cols if c in rf_df.columns]]

  budget_df = pd.concat([non_rf_full_df, rf_df], ignore_index=True)
  return budget_df


def helper_get_actual_vs_fitted_data(
    mmm: Any,
    *,
    confidence_level: float = 0.90,
    aggregate_geos: bool = True,
    aggregate_times: bool = False,
    selected_times: Sequence[str] | None = None,
) -> pd.DataFrame:
  """Return actual vs fitted data."""
  return values.get_actual_vs_fitted_data_fixed(
      mmm,
      confidence_level=confidence_level,
      aggregate_geos=aggregate_geos,
      aggregate_times=aggregate_times,
      selected_times=selected_times,
  )

