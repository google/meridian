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


def plot_posterior_coef(
    mmm: meridian_model.Meridian,
    parameter: str,
    index: int,
    *,
    maxbins: int = 50,
    width: int = 400,
    height: int = 200,
) -> alt.Chart:
  """Histogram of posterior draws for one coefficient on the log scale."""
  _check_fitted(mmm)
  samples = get_posterior_coef_samples(mmm, parameter, index)

  import pandas as pd, altair as alt
  alt.data_transformers.disable_max_rows()  # avoid silent truncation

  col = f"{parameter}[{index}]"
  df = pd.DataFrame({col: samples})

  return (
      alt.Chart(df)
      .mark_bar(opacity=0.7)
      .encode(
          x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=maxbins)),
          y="count()",
      )
      .properties(
          width=width,
          height=height,
          title=f"Posterior of {col} (log scale)",
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

  if selected_freqs is not None:
    freq_mask = np.isin(out.frequency.values, selected_freqs)
    out = out.isel(frequency=freq_mask)

  if selected_channels is not None:
    channel_mask = np.isin(out.rf_channel.values, selected_channels)
    out = out.isel(rf_channel=channel_mask)

  return out

