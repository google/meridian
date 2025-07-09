"""Diagnostic plotting utilities for the Meridian model.

This module provides helper functions to visualize posterior draws from a
``Meridian`` model. Currently it offers utilities for fitting a log-normal
distribution to media coefficients and plotting the result.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from meridian.model import Meridian


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
