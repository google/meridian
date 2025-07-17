"""Diagnostic utilities for the Meridian model.

This module primarily provides helper functions for inspecting a fitted
``Meridian`` instance.  In addition to the original plotting helper it now
exposes a small collection of statistical diagnostics that proved useful when
debugging models.  These utilities do not require TensorFlow and therefore can
be used independently of the rest of the library.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

try:  # pandas is optional
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import arviz as az

from meridian.model import Meridian
from meridian import constants as c


# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def plot_media_coef_lognormal(
    model: Meridian,
    channel_idx: int = 0,
    n_draws: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit a log-normal distribution to a media coefficient and plot it.

    This function optionally re-samples the posterior of ``model`` and then
    extracts the draws for a single media channel. The draws are assumed to
    follow a log-normal distribution. The corresponding normal parameters are
    estimated, a histogram of the draws with the fitted probability density
    function is plotted and the estimated parameters are returned.

    Parameters
    ----------
    model:
        A fitted :class:`Meridian` instance.
    channel_idx:
        Index of the media channel to inspect. Defaults to ``0``.
    n_draws:
        If provided, the model's posterior is re-sampled with this many draws.
    seed:
        Optional random seed used when re-sampling.

    Returns
    -------
    Tuple[float, float]
        ``(mu, sigma)`` estimates of the underlying normal distribution.
    """
    # Re-sample the posterior if requested.
    if n_draws is not None:
        model.sample_posterior(
            n_draws=n_draws,
            n_tune=int(n_draws * 0.5),
            n_chains=4,
            seed=seed,
        )

    # Extract posterior data
    idata = model.inference_data.posterior
    var_name = "beta_media"
    if var_name not in idata:
        raise KeyError(f"Variable '{var_name}' not found in inference data")

    arr = idata[var_name].stack(sample=("chain", "draw"))
    if "media_channel" not in arr.dims:
        raise KeyError("Dimension 'media_channel' not found in inference data")

    n_channels = arr.sizes["media_channel"]
    if channel_idx < 0 or channel_idx >= n_channels:
        raise IndexError(
            f"channel_idx {channel_idx} out of bounds for media_channel"
            f" dimension of size {n_channels}"
        )

    channel_samples = arr.isel(media_channel=channel_idx).values
    samples = channel_samples.reshape(-1)

    log_samps = np.log(samples)
    mu_hat = log_samps.mean()
    sigma_hat = log_samps.std(ddof=1)

    x = np.linspace(samples.min(), samples.max(), 500)
    pdf = (
        1.0
        / (x * sigma_hat * np.sqrt(2.0 * np.pi))
        * np.exp(-((np.log(x) - mu_hat) ** 2) / (2.0 * sigma_hat**2))
    )

    plt.figure(figsize=(8, 4))
    plt.hist(samples, bins=50, density=True, alpha=0.6, edgecolor="k")
    plt.plot(
        x,
        pdf,
        lw=2,
        label=f"Fitted LogN(\u03bc={mu_hat:.2f}, \u03c3={sigma_hat:.2f})",
    )
    plt.xlabel(f"\u03b2_media (channel {channel_idx})")
    plt.ylabel("Density")
    plt.title("Posterior samples vs. fitted log-normal PDF")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mu_hat, sigma_hat


def durbin_watson(residuals: np.ndarray) -> float:
    """Compute the Durbin-Watson statistic.

    Parameters
    ----------
    residuals:
        1-D array of model residuals in time order.

    Returns
    -------
    float
        The Durbin-Watson statistic.
    """
    residuals = np.asarray(residuals, dtype=float)
    diff = np.diff(residuals)
    return float(np.sum(diff**2) / np.sum(residuals**2))


def breusch_pagan(
    residuals: np.ndarray, exog: np.ndarray
) -> Tuple[float, float, float, float]:
    """Breusch-Pagan/White heteroscedasticity test.

    Parameters
    ----------
    residuals:
        1-D array of model residuals.
    exog:
        1-D or 2-D array of explanatory variables used in the regression
        (excluding the intercept). A column of ones is added automatically.

    Returns
    -------
    Tuple[float, float, float, float]
        ``(lm_stat, lm_pvalue, f_stat, f_pvalue)``.
    """
    residuals = np.asarray(residuals, dtype=float)
    exog = np.asarray(exog, dtype=float)

    if exog.ndim == 1:
        exog = exog[:, np.newaxis]

    n = residuals.shape[0]
    x = np.column_stack([np.ones(n), exog])
    y = residuals**2

    params, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ params
    sse = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - sse / tss

    k = x.shape[1]
    lm_stat = n * r2
    lm_pvalue = stats.chi2.sf(lm_stat, k - 1)
    f_stat = (n - k) * r2 / ((k - 1) * (1.0 - r2))
    f_pvalue = stats.f.sf(f_stat, k - 1, n - k)
    return float(lm_stat), float(lm_pvalue), float(f_stat), float(f_pvalue)


def jarque_bera(residuals: np.ndarray) -> Tuple[float, float]:
    """Jarque-Bera normality test for residuals.

    Parameters
    ----------
    residuals:
        1-D array of residuals.

    Returns
    -------
    Tuple[float, float]
        ``(jb_stat, jb_pvalue)``.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    mean = residuals.mean()
    std = residuals.std(ddof=1)
    if std == 0:
        return float("nan"), float("nan")
    skew = np.mean(((residuals - mean) / std) ** 3)
    kurt = np.mean(((residuals - mean) / std) ** 4)
    jb_stat = n / 6.0 * (skew**2 + 0.25 * (kurt - 3) ** 2)
    jb_pvalue = stats.chi2.sf(jb_stat, 2)
    return float(jb_stat), float(jb_pvalue)


def posterior_summary(
    idata: az.InferenceData,
    var_name: str,
    coord: Optional[Dict[str, Union[str, int]]] = None,
    cred_mass: float = 0.95,
) -> Dict[str, float]:
    """Compute summary statistics for a posterior variable."""
    da = idata.posterior[var_name]
    if coord:
        da = da.sel(**coord)
    samples = da.values.reshape(-1)

    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    t = mean / sd

    alpha = 1.0 - cred_mass
    lo, hi = np.quantile(samples, [alpha / 2, 1.0 - alpha / 2])
    return {
        "mean": mean,
        "sd": sd,
        "t_stat": t,
        f"{cred_mass * 100:.1f}%_ci_lower": float(lo),
        f"{cred_mass * 100:.1f}%_ci_upper": float(hi),
    }


def compute_vif(X: Union[np.ndarray, 'pd.DataFrame']) -> 'pd.Series':
    """Variance inflation factors for a design matrix."""
    if pd is not None and isinstance(X, pd.DataFrame):
        X_mat = X.values
        columns = list(X.columns)
    else:
        X_mat = X
        columns = [f"x{i}" for i in range(X.shape[1])]

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

    if pd is not None:
        return pd.Series(vifs, index=columns, name="VIF")
    return np.asarray(vifs)


def mass_above_threshold(
    idata: az.InferenceData,
    var_name: str,
    coord: dict | None = None,
    cred_mass: float = 0.95,
    *,
    threshold: float | None = None,
    log_space: bool = False,
) -> dict[str, float | bool]:
    """Probability mass of a variable lying above ``threshold``."""
    da = idata.posterior[var_name]
    if coord:
        da = da.sel(**coord)
    samples = da.values.reshape(-1)

    if log_space:
        samples = np.log(samples)

    threshold = 0.0 if threshold is None else threshold
    prob_above = np.mean(samples > threshold)

    return {
        "prob_above": float(prob_above),
        "meets_cred": bool(prob_above >= cred_mass),
        "cred_mass_req": float(cred_mass),
        "threshold": float(threshold),
    }


def series_diagnostics(
    mmm: Meridian,
    var_name: str,
    *,
    geo_index: int = 0,
    log_space: bool = False,
) -> Dict[str, float]:
    """Durbinâ€“Watson and Jarqueâ€“Bera for a single media regressor."""
    media_da = mmm.input_data.media
    if var_name not in media_da.coords[c.MEDIA_CHANNEL].values:
        raise ValueError(f"Unknown media channel: {var_name}")

    series = (
        media_da
        .sel(media_channel=var_name)
        .isel(geo=geo_index, drop=True)
        .values
        .astype(float)
        .ravel()
    )
    if log_space:
        series = np.log(series)

    dw = durbin_watson(series)
    jb_stat, jb_p = jarque_bera(series)
    return {"dw": float(dw), "jb_stat": float(jb_stat), "jb_p": float(jb_p)}


if __name__ == "__main__":
    # simple smoke test
    from meridian.model import Meridian

    # Users must supply ``input_data`` and ``spec`` below.
    model = Meridian(input_data, model_spec=spec)  # type: ignore[name-defined]
    mu, sigma = plot_media_coef_lognormal(
        model, channel_idx=0, n_draws=1000, seed=0
    )
    print(f"Estimated \u03bc={mu:.3f}, \u03c3={sigma:.3f}")
    # Consider adding a unit test in ``test/model_test.py`` to verify that this
    # function returns finite, positive values for a synthetic example.
