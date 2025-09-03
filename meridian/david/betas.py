"""Utilities for exploring beta posteriors from a Meridian model.

This module provides helper functions to extract posterior draws for
parameters of a :class:`meridian.model.Meridian` model and to fit a simple
probability distribution matching the type of the corresponding prior.
It also exposes a plotting helper which returns an Altair chart so that
posteriors can easily be visualised.

These functions only operate on a fitted model. Make sure you call
``Meridian.sample_posterior`` beforehand.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence
import types

import pandas as pd
import numpy as np
import altair as alt
import tensorflow_probability as tfp

from meridian import constants as c
from meridian.model import model as meridian_model


tfpd = tfp.distributions


def _check_fitted(mmm: meridian_model.Meridian) -> None:
  if not hasattr(mmm.inference_data, c.POSTERIOR):
    raise meridian_model.NotFittedModelError(
        "sample_posterior() must be run before using this utility")


def get_scalar_prior_names(mmm: meridian_model.Meridian) -> list[str]:
  """Returns names of scalar parameters in the posterior."""
  _check_fitted(mmm)
  names = [
      name
      for name, da in mmm.inference_data.posterior.data_vars.items()
      if da.values.ndim == 2
  ]
  return names


def get_beta_channel_names(mmm: meridian_model.Meridian) -> list[str]:
  """Returns media channel names for ``beta_m`` coefficients."""
  channels = getattr(getattr(mmm, "input_data", None), "media_channel", None)
  if channels is None:
    return []
  try:
    return list(channels.values)
  except Exception:  # pragma: no cover - fallback for simple sequences
    return list(channels)


def get_posterior_samples(
    mmm: meridian_model.Meridian, parameter: str) -> np.ndarray:
  """Returns flattened posterior draws for ``parameter``."""
  _check_fitted(mmm)
  if parameter not in mmm.inference_data.posterior.data_vars:
    raise ValueError(f"Unknown parameter: {parameter}")
  return mmm.inference_data.posterior[parameter].values.reshape(-1)


def estimate_distribution(samples: np.ndarray,
                          prior_dist: tfpd.Distribution) -> Mapping[str, Any]:
  """Estimates distribution parameters for the posterior samples.

  This fits parameters of the same distribution class as ``prior_dist`` using
  simple moment matching. Returned values depend on the distribution type.
  """
  dist_type = type(prior_dist)

  if dist_type is tfpd.Normal:
    loc = float(np.mean(samples))
    scale = float(np.std(samples, ddof=1))
    return {"distribution": "Normal", "loc": loc, "scale": scale}
  if dist_type is tfpd.LogNormal:
    logs = np.log(samples)
    loc = float(np.mean(logs))
    scale = float(np.std(logs, ddof=1))
    return {"distribution": "LogNormal", "loc": loc, "scale": scale}
  if dist_type is tfpd.HalfNormal:
    scale = float(np.sqrt(np.mean(samples**2)))
    return {"distribution": "HalfNormal", "scale": scale}
  if dist_type is tfpd.Uniform:
    low = float(samples.min())
    high = float(samples.max())
    return {"distribution": "Uniform", "low": low, "high": high}
  if dist_type is tfpd.Beta:
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)
    return {
        "distribution": "Beta",
        "concentration1": float(alpha),
        "concentration0": float(beta),
    }
  if dist_type is tfpd.Deterministic:
    return {"distribution": "Deterministic", "loc": float(samples.mean())}

  # Generic fallback.
  return {
      "distribution": dist_type.__name__,
      "mean": float(np.mean(samples)),
      "stddev": float(np.std(samples, ddof=1)),
  }


def fit_parameter_distribution(
    mmm: meridian_model.Meridian, parameter: str) -> Mapping[str, Any]:
  """Returns estimated distribution parameters for ``parameter``."""
  prior = getattr(mmm.prior_broadcast, parameter, None)
  if prior is None:
    raise ValueError(f"No prior information found for parameter: {parameter}")
  samples = get_posterior_samples(mmm, parameter)
  return estimate_distribution(samples, prior)


def plot_posterior(mmm: meridian_model.Meridian, parameter: str) -> alt.Chart:
  """Creates an Altair histogram for the posterior of ``parameter``."""
  samples = get_posterior_samples(mmm, parameter)
  df = pd.DataFrame({parameter: samples})
  chart = (
      alt.Chart(df)
      .mark_bar(opacity=0.7)
      .encode(x=alt.X(f"{parameter}:Q", bin=True), y="count()")
      .properties(title=f"Posterior of {parameter}")
  )
  return chart


def get_posterior_coef_samples(
    mmm: meridian_model.Meridian, parameter: str, index: int
) -> np.ndarray:
  """Returns flattened draws for one coefficient of ``parameter``.

  Parameters
  ----------
  mmm:
    Fitted :class:`Meridian` model instance.
  parameter:
    Name of the parameter that contains multiple coefficients.
  index:
    Index of the coefficient to inspect. The function assumes the
    coefficient dimension is the last axis of the parameter array.

  Returns
  -------
  numpy.ndarray
    1-D array of posterior draws for the requested coefficient.
  """
  _check_fitted(mmm)
  if parameter not in mmm.inference_data.posterior.data_vars:
    raise ValueError(f"Unknown parameter: {parameter}")

  values = mmm.inference_data.posterior[parameter].values
  if values.ndim == 0:
    raise IndexError("Parameter has no coefficient dimension")
  if index < 0 or index >= values.shape[-1]:
    raise IndexError("index out of bounds for coefficient dimension")
  return values[..., index].reshape(-1)


def fit_parameter_coef_distribution(
    mmm: meridian_model.Meridian, parameter: str, index: int
) -> Mapping[str, Any]:
  """Estimates distribution parameters for one coefficient of ``parameter``."""
  prior = getattr(mmm.prior_broadcast, parameter, None)
  if prior is None:
    raise ValueError(f"No prior information found for parameter: {parameter}")
  try:
    prior_slice = prior[index]
  except Exception:  # pragma: no cover - indexing may fail for scalar priors
    prior_slice = prior
  samples = get_posterior_coef_samples(mmm, parameter, index)
  return estimate_distribution(samples, prior_slice)


def inspect_t_stat(
    mmm: meridian_model.Meridian,
    channel_index: int,
    *,
    scale: str = "auto",
    geo_index: int | None = None,
) -> Mapping[str, float]:
  """Return t/z statistics for a beta coefficient.

  Parameters
  ----------
  mmm:
    Fitted :class:`Meridian` model instance.
  channel_index:
    Index of the media channel to inspect.
  scale:
    ``"auto"`` (default) chooses the coefficient scale based on the
    model's ``media_effects_dist``. Use ``"linear"`` to force computation on
    the linear coefficient (``beta_gm``) or ``"log"`` to operate on the
    log-scale mean parameter (``beta_m``).
  geo_index:
    Optional geo index to inspect. When ``None`` and the model has multiple
    geos, the coefficients are averaged over geos within each draw.

  Returns
  -------
  Mapping[str, float]
    A dictionary with mean, standard deviation (``sd``), standard error
    (``se``), z-score (``z``), t-statistic (``t``) and the probability that the
    coefficient is greater than zero.
  """
  _check_fitted(mmm)

  if scale not in {"auto", "linear", "log"}:
    raise ValueError("scale must be 'auto', 'linear' or 'log'")

  use_linear = scale in {"auto", "linear"}
  if mmm.media_effects_dist == c.MEDIA_EFFECTS_NORMAL:
    samples = get_posterior_coef_samples(mmm, c.BETA_M, channel_index)
    ess_source = c.BETA_M
  else:  # log-normal effects
    if use_linear:
      arr = mmm.inference_data.posterior[c.BETA_GM].values
      if arr.ndim == 4:  # chain, draw, geo, channel
        if geo_index is None:
          samples = arr.mean(axis=2)[..., channel_index].reshape(-1)
        else:
          samples = arr[..., geo_index, channel_index].reshape(-1)
      else:  # national model: chain, draw, channel
        samples = arr[..., channel_index].reshape(-1)
      ess_source = c.BETA_GM
    else:
      samples = get_posterior_coef_samples(mmm, c.BETA_M, channel_index)
      ess_source = c.BETA_M

  try:
    import arviz as az  # type: ignore
    da = mmm.inference_data.posterior[ess_source]
    if hasattr(da, "dims"):
      if ess_source == c.BETA_GM and "geo" in da.dims:
        if geo_index is None:
          da = da.mean("geo")
        else:
          da = da.isel(geo=geo_index)
      if da.dims:
        da = da.isel({da.dims[-1]: channel_index})
    ess = float(az.ess(da).values)  # type: ignore[arg-type]
  except Exception:  # pragma: no cover - arviz optional
    ess = samples.size

  mean = float(np.mean(samples))
  sd = float(np.std(samples, ddof=1))
  se = float(sd / np.sqrt(max(ess, 1)))
  z = float(mean / sd) if sd > 0 else np.nan
  t = float(mean / se) if se > 0 else np.nan
  p_gt0 = float((samples > 0).mean())
  return {
      "mean": mean,
      "sd": sd,
      "se": se,
      "z": z,
      "t": t,
      "p(beta>0)": p_gt0,
  }


def plot_posterior_coef(
    mmm: meridian_model.Meridian,
    parameter: str,
    index: int,
    *,
    maxbins: int = 50,
    width: int = 400,
    height: int = 200,
) -> alt.Chart:
  """Histogram of posterior draws for one coefficient.

  Uses a generic column name to avoid Vega-Lite field parsing issues when
  ``parameter`` contains characters like brackets. This ensures compatibility
  with environments that do not honour field quoting.
  """
  _check_fitted(mmm)
  samples = get_posterior_coef_samples(mmm, parameter, index)

  # Use a safe column name so Altair/Vega-Lite always recognise the field.
  samples = np.asarray(samples).astype(float)
  df = pd.DataFrame({"value": samples})

  alt.data_transformers.disable_max_rows()  # avoid silent truncation

  return (
      alt.Chart(df)
      .mark_bar(opacity=0.7)
      .encode(
          x=alt.X("value:Q", bin=alt.Bin(maxbins=maxbins), title="Value"),
          y=alt.Y("count()", title="Count"),
      )
      .properties(
          width=width,
          height=height,
          title=f"Posterior of {parameter}[{index}]",
      )
  )


def optimal_freq_safe(
    ana,
    *,
    selected_freqs: Sequence[float] | None = None,
    selected_channels: Sequence[str] | None = None,
    **kw,
) -> Any:
  """Wrapper for ``Analyzer.optimal_freq`` supporting simple filtering.

  Parameters
  ----------
  ana:
    :class:`Analyzer` instance.
  selected_freqs:
    Optional list of frequencies to keep from the result.
  selected_channels:
    Optional list of RF channels to keep from the result.
  **kw:
    Additional keyword arguments forwarded to ``ana.optimal_freq``.

  Returns
  -------
  xarray.Dataset
    The dataset returned by ``ana.optimal_freq`` filtered to the requested
    frequencies and channels.
  """
  out = ana.optimal_freq(**kw)
  isel_kw = {}

  if selected_freqs is not None:
    freq_mask = np.isin(out.frequency.values, selected_freqs)
    isel_kw['frequency'] = freq_mask

  if selected_channels is not None:
    channel_mask = np.isin(out.rf_channel.values, selected_channels)
    isel_kw['rf_channel'] = channel_mask

  if isel_kw:
    out = out.isel(**isel_kw)

  return out


def _posterior_mean_da(da: Any) -> np.ndarray:
  """Return mean of posterior draws across chain/draw/sample dims."""
  dims = list(getattr(da, 'dims', []))
  values = np.asarray(getattr(da, 'values', da))
  axes = [dims.index(d) for d in ('chain', 'draw', 'sample') if d in dims]
  if axes:
    values = values.mean(axis=tuple(axes))
  return values


def _tensor_from_da(da: Any) -> np.ndarray:
  """Convert a DataArray-like object to a float32 numpy array."""
  return np.asarray(getattr(da, 'values', da), dtype=np.float32)


def _broadcast_param(param: np.ndarray, target_rank: int) -> np.ndarray:
  """Ensure ``param`` has rank 1 (channel) and broadcast to ``target_rank``."""
  arr = np.asarray(param)
  if arr.ndim == 1:
    arr = arr.reshape((1, 1, -1))  # geo×time×channel broadcast
  while arr.ndim < target_rank:
    arr = np.expand_dims(arr, 0)
  return arr


def _transform_block(
    meridian: meridian_model.Meridian,
    media: np.ndarray | None,
    reach: np.ndarray | None,
    frequency: np.ndarray | None,
    alpha: np.ndarray,
    ec: np.ndarray,
    slope: np.ndarray,
    tag: str,
    geo_coords: Sequence[str],
    time_coords: Sequence[Any],
    channel_coords: Sequence[str],
) -> pd.DataFrame | None:
  """Apply adstock/hill combo and return tidy DataFrame."""
  if media is None and reach is None:
    return None

  if media is not None and reach is None:
    transformed = meridian.adstock_hill_media(
        media=media,
        alpha=_broadcast_param(alpha, media.ndim),
        ec=_broadcast_param(ec, media.ndim),
        slope=_broadcast_param(slope, media.ndim),
        n_times_output=meridian.n_times,
    )
  else:
    transformed = meridian.adstock_hill_rf(
        reach=reach,
        frequency=frequency,
        alpha=_broadcast_param(alpha, reach.ndim),
        ec=_broadcast_param(ec, reach.ndim),
        slope=_broadcast_param(slope, reach.ndim),
        n_times_output=meridian.n_times,
    )

  # ``transformed`` may be a TensorFlow Tensor which lacks ``reshape``.
  # Convert to a NumPy array before reshaping.
  vals = np.asarray(transformed)
  tidy = (
      pd.DataFrame(
          vals.reshape(-1, vals.shape[-1]),
          columns=channel_coords,
      )
      .assign(geo=np.repeat(geo_coords, meridian.n_times))
      .assign(time=np.tile(time_coords, len(geo_coords)))
      .melt(id_vars=['geo', 'time'], var_name='channel', value_name='value')
      .assign(block=tag)
  )
  return tidy


def _transformed_predictors(
    meridian: meridian_model.Meridian,
    use_posterior: bool = True,
    aggregate_geos: bool = True,
    selected_channels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, alt.Chart]:
  """Return DataFrame and Altair chart of transformed predictors."""
  idata = meridian.inference_data
  group = 'posterior' if (use_posterior and hasattr(idata, 'posterior')) else 'prior'
  posterior = getattr(idata, group)

  def _param(name: str) -> np.ndarray:
    return _tensor_from_da(_posterior_mean_da(getattr(posterior, name)))

  alpha_m = _param('alpha_m')
  ec_m = _param('ec_m')
  slope_m = _param('slope_m')

  if getattr(meridian, 'n_rf_channels', 0) > 0:
    alpha_rf = _param('alpha_rf')
    ec_rf = _param('ec_rf')
    slope_rf = _param('slope_rf')
  else:
    alpha_rf = ec_rf = slope_rf = None

  geo_coords = meridian.input_data.geo.values
  time_coords = meridian.input_data.time.values
  media_coords = meridian.input_data.media.coords[c.MEDIA_CHANNEL].values

  media_scaled = _tensor_from_da(meridian.media_tensors.media_scaled)
  df_parts: list[pd.DataFrame] = []

  df_media = _transform_block(
      meridian,
      media=media_scaled,
      reach=None,
      frequency=None,
      alpha=alpha_m,
      ec=ec_m,
      slope=slope_m,
      tag='MEDIA',
      geo_coords=geo_coords,
      time_coords=time_coords,
      channel_coords=media_coords,
  )
  df_parts.append(df_media)

  if getattr(meridian, 'n_rf_channels', 0) > 0:
    reach_scaled = _tensor_from_da(meridian.rf_tensors.reach_scaled)
    frequency = _tensor_from_da(meridian.rf_tensors.frequency)
    rf_coords = meridian.input_data.reach.coords[c.RF_CHANNEL].values
    df_rf = _transform_block(
        meridian,
        media=None,
        reach=reach_scaled,
        frequency=frequency,
        alpha=alpha_rf,
        ec=ec_rf,
        slope=slope_rf,
        tag='RF',
        geo_coords=geo_coords,
        time_coords=time_coords,
        channel_coords=rf_coords,
    )
    df_parts.append(df_rf)

  df = pd.concat([d for d in df_parts if d is not None], ignore_index=True)

  if selected_channels is not None:
    df = df[df['channel'].isin(selected_channels)]

  if aggregate_geos:
    df = (
        df.groupby(['time', 'channel', 'block'], as_index=False)
        .agg(value=('value', 'sum'))
    )

  base = alt.Chart(df).encode(
      x=alt.X('time:T', title='Date'),
      y=alt.Y('value:Q', title='Transformed value', stack=None),
      color=alt.Color('channel:N', title='Channel'),
      tooltip=['channel', 'value', 'time:T'],
  )
  chart = base.mark_line().properties(width=600, height=400)
  return df, chart


def view_transformed_variable(
    meridian: meridian_model.Meridian,
    channel: str,
    *,
    use_posterior: bool = True,
    aggregate_geos: bool = True,
    show_raw: bool = False,
) -> tuple[pd.DataFrame, alt.Chart]:
  """Return DataFrame and chart of the transformed predictor for ``channel``.

  If ``show_raw`` is ``True``, the untransformed predictor is overlaid on the
  same chart with an independent y-axis on the right-hand side.
  """

  df, chart = _transformed_predictors(
      meridian,
      use_posterior=use_posterior,
      aggregate_geos=aggregate_geos,
      selected_channels=[channel],
  )

  if not show_raw:
    return df, chart

  # Determine whether ``channel`` refers to media or RF reach/frequency and
  # extract the corresponding raw values prior to adstock/hill transformation.
  geo_coords = meridian.input_data.geo.values
  time_coords = meridian.input_data.time.values

  media_channels = meridian.input_data.media.coords[c.MEDIA_CHANNEL].values
  raw_array = None
  if channel in media_channels:
    media = _tensor_from_da(meridian.media_tensors.media_scaled)
    idx = int(np.where(media_channels == channel)[0][0])
    raw_array = media[:, :, idx]
  elif getattr(meridian, 'n_rf_channels', 0) > 0:
    rf_channels = meridian.input_data.reach.coords[c.RF_CHANNEL].values
    if channel in rf_channels:
      reach = _tensor_from_da(meridian.rf_tensors.reach_scaled)
      idx = int(np.where(rf_channels == channel)[0][0])
      raw_array = reach[:, :, idx]

  if raw_array is None:
    return df, chart

  raw_df = (
      pd.DataFrame(raw_array, index=geo_coords, columns=time_coords)
      .stack()
      .rename('raw_value')
      .reset_index()
      .rename(columns={'level_0': 'geo', 'level_1': 'time'})
  )
  if aggregate_geos:
    raw_df = raw_df.groupby('time', as_index=False).agg(raw_value=('raw_value', 'sum'))

  raw_chart = (
      alt.Chart(raw_df)
      .mark_line(color='black')
      .encode(
          x=alt.X('time:T', title='Date'),
          y=alt.Y('raw_value:Q', axis={'title': 'Raw value', 'orient': 'right'}),
          tooltip=['raw_value', 'time:T'],
      )
  )

  try:
    chart = (chart + raw_chart).resolve_scale(y='independent')
  except Exception:  # pragma: no cover - for altair stubs without layering
    layer_chart = alt.LayerChart(None)
    layer_chart.layer = [chart, raw_chart]
    chart = layer_chart

  return df, chart


def plot_posterior_coef_as_normal(
    mmm: meridian_model.Meridian,
    parameter: str,
    index: int,
    *,
    maxbins: int = 50,
    width: int = 400,
    height: int = 200,
    epsilon: float = 1e-9,
) -> alt.Chart:
  """Histogram after log-transforming positive posterior draws.

  The helper is meant for strictly-positive, right-skewed parameters
  (e.g. *log-normal* ``beta_gm``). It:

  1.  Retrieves the raw draws with :func:`get_posterior_coef_samples`.
  2.  Applies ``np.log`` (or ``np.log(value + ε)`` if any sample \u2264 0).
  3.  Returns an Altair bar chart whose shape should now be approximately
      Normal.

  Parameters
  ----------
  mmm, parameter, index
      Same meaning as :func:`plot_posterior_coef`.
  maxbins, width, height
      Passed straight to Altair for visual tweaking.
  epsilon
      Safety constant to avoid ``log(0)`` \u2192 ``-inf``.

  Notes
  -----
  \u2022 If the original parameter is already centred and symmetric
    (e.g. ``beta_m``), the chart will look essentially identical.
  """
  samples = get_posterior_coef_samples(mmm, parameter, index).astype(float)

  if np.any(samples <= 0):
    samples = np.log(samples + epsilon)
  else:
    samples = np.log(samples)

  df = pd.DataFrame({"value": samples})
  alt.data_transformers.disable_max_rows()

  return (
      alt.Chart(df)
      .mark_bar(opacity=0.7)
      .encode(
          x=alt.X("value:Q",
                  bin=alt.Bin(maxbins=maxbins),
                  title="log(value)"),
          y=alt.Y("count()", title="Count"),
      )
      .properties(
          width=width,
          height=height,
          title=f"Log-transformed posterior of {parameter}[{index}]",
      )
  )


