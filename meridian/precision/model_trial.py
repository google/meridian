"""Bayesian hierarchical platform-level model for a single channel.

The model splits one advertising channel into multiple platforms whose weights
are constrained on a simplex. Adstock decay parameters are weakly pooled around
a channel-level value. The implementation relies on PyMC and Meridian's data
loading utilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:  # pragma: no cover - handled in tests with importorskip
    import pymc as pm
    import aesara
    import aesara.tensor as at
except Exception:  # pragma: no cover
    pm = None  # type: ignore
    aesara = None  # type: ignore
    at = None  # type: ignore

from meridian.data import load as data_load

__all__ = [
    "DataSpec",
    "load_dataframe_input",
    "build_platform_weight_model",
    "fit_platform_weight_model",
]


@dataclass
class DataSpec:
    """Configuration describing how to interpret a dataframe."""

    coord_to_columns: data_load.CoordToColumns
    media_to_channel: Mapping[str, str]
    media_spend_to_channel: Mapping[str, str]
    kpi_type: str = "non-revenue"


def load_dataframe_input(df: pd.DataFrame, spec: DataSpec):
    """Loads platform data using Meridian's :class:`DataFrameDataLoader`.

    Args:
      df: DataFrame containing the raw data.
      spec: A :class:`DataSpec` instance describing column mappings.

    Returns:
      An :class:`meridian.data.input_data.InputData` instance.
    """

    loader = data_load.DataFrameDataLoader(
        df=df,
        coord_to_columns=spec.coord_to_columns,
        kpi_type=spec.kpi_type,
        media_to_channel=spec.media_to_channel,
        media_spend_to_channel=spec.media_spend_to_channel,
    )
    return loader.load()


def _adstock_exponential(x: at.TensorVariable, lam: at.TensorVariable) -> at.TensorVariable:
    """Computes exponential adstock with decay ``lam`` for each platform."""

    def step(x_t, a_prev, lam_vec):
        return x_t + at.exp(-lam_vec) * a_prev

    init = at.zeros((x.shape[1],))
    adstock, _ = aesara.scan(step, sequences=x, outputs_info=init, non_sequences=lam)
    return adstock


def build_platform_weight_model(input_data, spend_share: np.ndarray | None = None):
    """Builds the hierarchical platform-weight model.

    Args:
      input_data: Input data returned by :func:`load_dataframe_input`.
      spend_share: Optional prior mean for platform weights. If ``None`` the
        shares are derived from total spend.

    Returns:
      A compiled :class:`pymc.Model`.
    """
    if pm is None:  # pragma: no cover - handled by importorskip in tests
        raise ImportError("pymc is required to build the model")

    y = np.asarray(input_data.kpi).squeeze()
    media = np.asarray(input_data.media).squeeze()
    controls = (
        np.asarray(input_data.controls).squeeze()
        if input_data.controls is not None
        else None
    )
    spend = np.asarray(input_data.media_spend).squeeze()
    costs = spend.sum(axis=0)

    n_time, n_platforms = media.shape
    if spend_share is None:
        total_cost = costs.sum()
        spend_share = costs / total_cost if total_cost else np.ones(n_platforms) / n_platforms

    with pm.Model() as model:
        media_data = pm.Data("media", media)
        control_data = pm.Data("controls", controls) if controls is not None else None

        lambda_social = pm.Beta("lambda_social", alpha=2, beta=2)
        tau_lambda = pm.HalfNormal("tau_lambda", sigma=0.05)
        logit_lambda_p = pm.Normal(
            "logit_lambda_p",
            mu=pm.math.logit(lambda_social),
            sigma=tau_lambda,
            shape=n_platforms,
        )
        lambda_p = pm.Deterministic("lambda_p", pm.math.sigmoid(logit_lambda_p))

        adstock = _adstock_exponential(media_data, lambda_p)

        theta = pm.Normal(
            "theta",
            mu=np.log(spend_share),
            sigma=0.3,
            shape=n_platforms,
        )
        w = pm.Deterministic("w", pm.math.softmax(theta))
        s = pm.math.dot(adstock, w)

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta_social = pm.StudentT(
            "beta_social", nu=3, mu=0.0, sigma=0.5 * y.std()
        )
        mu = alpha + beta_social * s
        if control_data is not None:
            gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=control_data.shape[1])
            mu = mu + pm.math.dot(control_data, gamma)

        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y)

        pm.Deterministic("roi", beta_social * w / costs)

    return model


def fit_platform_weight_model(input_data, **sample_kwargs):
    """Fits the platform-weight model and returns the trace.

    ``sample_kwargs`` are passed directly to :func:`pymc.sample`.
    """
    model = build_platform_weight_model(input_data)
    with model:
        trace = pm.sample(**sample_kwargs)
    return model, trace
