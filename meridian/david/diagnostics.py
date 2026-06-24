"""Posterior diagnostics utilities for Meridian.

This module provides simple helper functions to compute common
statistical diagnostics from the posterior draws of a fitted
``Meridian`` model.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import xarray as xr

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera as _jarque_bera
from scipy.stats import kurtosis as _kurtosis
try:  # pragma: no cover - optional dependency
    from statsmodels.stats.diagnostic import het_breuschpagan as _breusch_pagan
except Exception:  # pragma: no cover
    _breusch_pagan = None
try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.stattools import adfuller as _adfuller
except Exception:  # pragma: no cover
    _adfuller = None


# --------------------------------------------------------------------
# 1. Posterior mean, standard deviation and credible interval
# --------------------------------------------------------------------


def posterior_summary(
    idata: az.InferenceData,
    var_name: str,
    coord: Optional[Dict[str, Union[str, int]]] = None,
    cred_mass: float = 0.95,
) -> Dict[str, float]:
    """Summary statistics for a single scalar posterior.

    Parameters
    ----------
    idata:
      ArviZ ``InferenceData`` returned by ``Meridian.sample_posterior``.
    var_name:
      Name of the variable to summarize, e.g. ``"beta_m"``.
    coord:
      Optional mapping selecting a single coefficient, for example
      ``{"media": "media_channel_1"}``.
    cred_mass:
      Width of the central credible interval. ``0.95`` by default.

    Returns
    -------
    dict
      Keys ``mean``, ``sd``, ``t_stat`` and the bounds of the requested
      credible interval.
    """
    da = idata.posterior[var_name]
    if coord:
        da = da.sel(**coord)
    samples = da.values.reshape(-1)

    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    t = mean / sd

    alpha = 1.0 - cred_mass
    lo, hi = np.quantile(samples, [alpha / 2, 1.0 - alpha / 2])
    ci_key_lo = f"{cred_mass * 100:.1f}%_ci_lower"
    ci_key_hi = f"{cred_mass * 100:.1f}%_ci_upper"

    return {
        "mean": mean,
        "sd": sd,
        "t_stat": t,
        ci_key_lo: float(lo),
        ci_key_hi: float(hi),
    }


# --------------------------------------------------------------------
# 2. Variance inflation factors
# --------------------------------------------------------------------


def compute_vif(X: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
    """Computes OLS-style variance inflation factors for each column of ``X``."""
    if isinstance(X, np.ndarray):
        X_mat = X
        columns = [f"x{i}" for i in range(X.shape[1])]
    else:
        X_mat = X.values
        columns = list(X.columns)

    vifs = []
    for j in range(X_mat.shape[1]):
        y = X_mat[:, j]
        X_other = np.delete(X_mat, j, axis=1)
        coef, *_ = np.linalg.lstsq(X_other, y, rcond=None)
        y_hat = X_other @ coef
        ssr = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ssr / sst
        vifs.append(1.0 / (1.0 - r2 + 1e-12))

    return pd.Series(vifs, index=columns, name="VIF")


# --------------------------------------------------------------------
# 3. Durbin-Watson statistic
# --------------------------------------------------------------------


def durbin_watson(
    residuals,
    *,
    time_dim: str | None = None,
    by: tuple[str, ...] | None = None,
    reduce_over: tuple[str, ...] = ("chain", "draw"),
    skipna: bool = True,
):
    """Durbin–Watson statistic with xarray-aware conveniences.

    Parameters
    ----------
    residuals :
        - np.ndarray (as before): flattened DW over the vector.
        - xr.DataArray: DW computed along the time dimension, vectorized across
          other dims (e.g., per-geo). Chain/draw dims are averaged out by default.
        - tuple of (y, yhat) as np.ndarray or xr.DataArray: residuals computed
          internally as (y - yhat). If xarray, arrays are auto-aligned.
    time_dim : str, optional
        Name of the temporal dimension to difference along when ``residuals`` is
        an xr.DataArray. If ``None``, try to infer from ("time", "media_time", "geo_time").
        If no time-like dim is found, falls back to flattening behavior.
    by : tuple[str, ...], optional
        When residuals are an xr.DataArray, return DW per-group over these dims.
        If ``None`` (default), returns a scalar by averaging DW across all groups.
    reduce_over : tuple[str, ...]
        Dims to average out *before* computing DW (useful for ("chain", "draw")).
        Ignored for numpy input.
    skipna : bool
        If True, drop NaNs along the time dimension before computing DW.

    Returns
    -------
    float or xr.DataArray
        - float if numpy input, or if ``by is None`` (average across groups).
        - xr.DataArray with dims ``by`` if provided.

    Notes
    -----
    • For xr.DataArray input: DW is computed per-series along ``time_dim`` and
      then averaged across any remaining dims unless ``by`` is specified.
    • A small epsilon is added to the denominator to guard against division by zero.
    """
    import numpy as _np
    import xarray as _xr

    EPS = 1e-12

    # Allow passing (y, yhat)
    if isinstance(residuals, tuple) and len(residuals) == 2:
        y, yhat = residuals
        # xarray-aware alignment
        if isinstance(y, _xr.DataArray) and isinstance(yhat, _xr.DataArray):
            y, yhat = _xr.align(y, yhat, join="inner")
            residuals = y - yhat
        else:
            residuals = _np.asarray(y) - _np.asarray(yhat)

    # Numpy / list path: preserve legacy behavior
    if not hasattr(residuals, "dims"):  # not an xarray.DataArray
        r = _np.asarray(residuals).ravel()
        diff = _np.diff(r)
        denom = _np.sum(r**2) + EPS
        return float(_np.sum(diff**2) / denom)

    # Xarray path
    da = residuals  # type: ignore[assignment]
    # Reduce chain/draw if present
    for d in reduce_over:
        if d in da.dims:
            da = da.mean(d)

    # Find or infer time dimension
    if time_dim is None:
        try:
            time_dim = _infer_time_dim(da)  # uses helper already defined in this module
        except ValueError:
            # Fall back to legacy "flatten everything" behavior
            r = _np.asarray(da.values).ravel()
            diff = _np.diff(r)
            denom = _np.sum(r**2) + EPS
            return float(_np.sum(diff**2) / denom)

    # Optionally drop NaNs along time dimension
    if skipna:
        da = da.dropna(dim=time_dim, how="any")

    def _dw_1d(x):
        x = _np.asarray(x)
        diff = _np.diff(x)
        denom = _np.sum(x**2) + EPS
        return _np.sum(diff**2) / denom

    # Vectorize DW over all dims except time_dim
    result = _xr.apply_ufunc(
        _dw_1d,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
    ).rename("durbin_watson")

    if by is None:
        # Average DW across all remaining dims to return a scalar
        if result.dims:
            result = result.mean(result.dims)  # type: ignore[arg-type]
        return float(result.values)

    # Keep only the requested grouping dims; average the rest
    reduce_dims = tuple(d for d in result.dims if d not in by)
    if reduce_dims:
        result = result.mean(reduce_dims)
    # Ensure result has exactly the 'by' dims (may be scalar if by dims are missing)
    return result


def durbin_watson_from_idata(
    idata: az.InferenceData,
    *,
    target_priority: tuple[str, ...] = ("kpi", "revenue"),
    time_dim: str | None = None,
    by: tuple[str, ...] | None = None,
    reduce_over: tuple[str, ...] = ("chain", "draw"),
    observed: xr.DataArray | None = None,
):
    """DW from an InferenceData.

    Prefer ``posterior_predictive`` but fall back to ``predictions`` or
    deterministic posterior variables (e.g., ``kpi_hat``, ``mu_kpi``). When the
    InferenceData lacks an ``observed_data`` group, ``observed`` may be supplied
    directly.

    Example
    -------
    >>> durbin_watson_from_idata(mmm.inference_data, by=("geo",))
    """

    import xarray as _xr

    def _mean_over(da: _xr.DataArray, dims: tuple[str, ...]):
        keep = [d for d in dims if d in da.dims]
        return da.mean(keep) if keep else da

    def _pick_observed_var():
        if observed is not None:
            name = getattr(observed, "name", None) or target_priority[0]
            return name, observed
        if not hasattr(idata, "observed_data"):
            raise AttributeError("InferenceData has no observed_data group.")
        for v in target_priority:
            if v in idata.observed_data.data_vars:
                return v, idata.observed_data[v]
        raise KeyError(
            f"None of {target_priority} found in observed_data; "
            f"available: {list(idata.observed_data.data_vars)}"
        )

    def _pick_predicted_da(target_name: str) -> _xr.DataArray:
        # 1) posterior_predictive (ArviZ standard)
        if hasattr(idata, "posterior_predictive"):
            pp = idata.posterior_predictive
            if target_name in pp.data_vars:
                return _mean_over(pp[target_name], reduce_over)

        # 2) predictions group (some pipelines put yhat here)
        if hasattr(idata, "predictions"):
            pr = idata.predictions
            if target_name in pr.data_vars:
                return _mean_over(pr[target_name], reduce_over)

        # 3) deterministic variables in posterior with common naming
        candidates = [
            f"{target_name}_hat",
            f"{target_name}_pred",
            f"{target_name}_yhat",
            f"yhat_{target_name}",
            f"mu_{target_name}",
            f"{target_name}_mu",
            f"expected_{target_name}",
            f"{target_name}_expected",
        ]
        if hasattr(idata, "posterior"):
            post = idata.posterior
            for name in candidates:
                if name in post.data_vars:
                    da = post[name]
                    if any(d in da.dims for d in ("time", "media_time", "geo_time")):
                        return _mean_over(da, reduce_over)

        # 4) As a last resort, if the posterior itself has a time-like var with the same name
        if hasattr(idata, "posterior") and target_name in idata.posterior.data_vars:
            da = idata.posterior[target_name]
            if any(d in da.dims for d in ("time", "media_time", "geo_time")):
                return _mean_over(da, reduce_over)

        raise AttributeError(
            "Could not locate predictions for "
            f"{target_name!r}. Tried posterior_predictive, predictions, and common "
            "deterministic posterior names like '*_hat', 'mu_*'. "
            "Consider enabling posterior predictive sampling or storing yhat."
        )

    target_name, yobs = _pick_observed_var()
    yhat = _pick_predicted_da(target_name)

    yobs, yhat = _xr.align(yobs, yhat, join="inner")
    return durbin_watson(
        yobs - yhat,
        time_dim=time_dim,
        by=by,
        reduce_over=(),  # already reduced
        skipna=True,
    )


# --------------------------------------------------------------------
# 4. Jarque-Bera normality test
# --------------------------------------------------------------------


def jarque_bera(residuals: np.ndarray):
    """Returns the Jarque-Bera statistic and p-value."""
    return _jarque_bera(np.asarray(residuals).ravel())


# --------------------------------------------------------------------
# 5. Mass-above-threshold test
# --------------------------------------------------------------------


def mass_above_threshold(
    idata: az.InferenceData,
    var_name: str,
    coord: dict | None = None,
    cred_mass: float = 0.95,
    *,
    threshold: float | None = None,
    log_space: bool = False,
) -> dict[str, float | bool]:
    """Probability that a coefficient exceeds a threshold."""
    da = idata.posterior[var_name]
    if coord:
        try:
            da = da.sel(**coord)
        except KeyError as err:  # pragma: no cover - xarray raises KeyError
            dim, val = next(iter(coord.items()))
            if dim not in da.dims:
                raise KeyError(
                    f"Dimension {dim!r} not found; available dims: {list(da.dims)}"
                ) from err
            choices = list(da[dim].values)
            raise KeyError(
                f"Value {val!r} not found in coordinate {dim!r}; choices: {choices}"
            ) from err
    samples = da.values.reshape(-1)

    if log_space:
        samples = np.log(samples)

    if threshold is None:
        threshold = 0.0

    prob_above = np.mean(samples > threshold)

    return {
        "prob_above": float(prob_above),
        "meets_cred": bool(prob_above >= cred_mass),
        "cred_mass_req": float(cred_mass),
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------
# 6. Augmented Dickey-Fuller unit-root test
# --------------------------------------------------------------------


def augmented_dickey_fuller(series: np.ndarray):
    """Returns the ADF statistic and p-value."""
    if _adfuller is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("statsmodels is required for augmented_dickey_fuller")
    stat, pvalue, *_ = _adfuller(np.asarray(series).ravel())
    return float(stat), float(pvalue)


# --------------------------------------------------------------------
# 7. Design matrix construction
# --------------------------------------------------------------------


def _infer_time_dim(da: xr.DataArray) -> str:
  """Return the name of the temporal dimension in ``da``.

  Meridian tensors may expose one of ``"time"``, ``"media_time"`` or
  ``"geo_time"``.  This helper searches for the first matching dimension name
  and returns it.  A :class:`ValueError` is raised if none of the expected
  dimensions are present.
  """

  for cand in ("time", "media_time", "geo_time"):
    if cand in da.dims:
      return cand
  raise ValueError(
      f"Cannot locate temporal dimension in {list(da.dims)}; expected one of "
      "'time', 'media_time', 'geo_time'."
  )


def build_design_matrix(
    media_da: xr.DataArray,
    controls_da: xr.DataArray | None = None,
    add_constant: bool = True,
) -> pd.DataFrame:
  """Construct a long-format design matrix for regression.

  Parameters
  ----------
  media_da:
    Media predictors with dims ``('geo', <time_dim>, 'media_channel')``.
  controls_da:
    Optional controls with dims ``('geo', <time_dim>, <ctrl_dim>)``.
  add_constant:
    If ``True``, a column named ``"const"`` filled with ``1.0`` is
    prepended.

  Returns
  -------
  pandas.DataFrame
    Rows correspond to ``(#geo × #time)`` samples, columns to regressors.
  """

  tdim_media = _infer_time_dim(media_da)

  media_df = (
      media_da.stack(sample=("geo", tdim_media))
      .transpose("sample", "media_channel")
      .to_pandas()
  )

  if controls_da is not None:
    tdim_ctrl = _infer_time_dim(controls_da)
    ctrl_dim = next(d for d in controls_da.dims if d not in ("geo", tdim_ctrl))
    ctrl_df = (
        controls_da.stack(sample=("geo", tdim_ctrl))
        .transpose("sample", ctrl_dim)
        .to_pandas()
    )
    X = pd.concat([media_df, ctrl_df], axis=1)
  else:
    X = media_df

  if add_constant and "const" not in X.columns:
    X.insert(0, "const", 1.0)

  return X.astype("float64")


# --------------------------------------------------------------------
# 8. Breusch-Pagan-Godfrey heteroscedasticity test
# --------------------------------------------------------------------


def breusch_pagan_godfrey(
    residuals: np.ndarray, exog: Union[np.ndarray, pd.DataFrame]
) -> Dict[str, float]:
    """Tests residuals for heteroscedasticity with respect to ``exog``."""
    if _breusch_pagan is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "statsmodels is required for breusch_pagan_godfrey"
        )
    if isinstance(exog, pd.DataFrame):
        X = exog.values
    else:
        X = np.asarray(exog)
    lm_stat, lm_pvalue, f_stat, f_pvalue = _breusch_pagan(
        np.asarray(residuals).ravel(), X
    )
    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


# --------------------------------------------------------------------
# 9. Kurtosis of residuals
# --------------------------------------------------------------------


def residual_kurtosis(residuals: np.ndarray, *, fisher: bool = True) -> float:
    """Sample kurtosis of ``residuals``."""
    return float(_kurtosis(np.asarray(residuals).ravel(), fisher=fisher))


# --------------------------------------------------------------------
# 10. Correlation-matrix inspection
# --------------------------------------------------------------------


def inspect_correlation_matrix(
    X: Union[np.ndarray, pd.DataFrame], *, threshold: float = 0.8
) -> pd.DataFrame:
    """Returns pairs of variables with correlation exceeding ``threshold``."""
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X)
    else:
        df = X

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "corr"})
    )
    return pairs[pairs["corr"] >= threshold].reset_index(drop=True)
