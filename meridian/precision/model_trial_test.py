import numpy as np
import pandas as pd
import pytest
import sys
import types

# Stub tensorflow so importing meridian does not require the heavy dependency.
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

from meridian.precision import model_trial
from meridian.data import load

pm = pytest.importorskip("pymc")


def _make_dataframe():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2021-01-01", periods=12, freq="W")
    df = pd.DataFrame(
        {
            "time": weeks.strftime("%Y-%m-%d"),
            "sales": rng.uniform(100, 200, size=len(weeks)),
            "facebook_impr": rng.uniform(0, 100, size=len(weeks)),
            "instagram_impr": rng.uniform(0, 120, size=len(weeks)),
            "tiktok_impr": rng.uniform(0, 80, size=len(weeks)),
            "facebook_spend": rng.uniform(1, 10, size=len(weeks)),
            "instagram_spend": rng.uniform(1, 10, size=len(weeks)),
            "tiktok_spend": rng.uniform(1, 10, size=len(weeks)),
            "price": rng.uniform(1, 5, size=len(weeks)),
        }
    )
    return df


def test_platform_weight_model_runs():
    df = _make_dataframe()
    coord_to_columns = load.CoordToColumns(
        time="time",
        kpi="sales",
        controls=["price"],
        media=["facebook_impr", "instagram_impr", "tiktok_impr"],
        media_spend=["facebook_spend", "instagram_spend", "tiktok_spend"],
    )
    spec = model_trial.DataSpec(
        coord_to_columns=coord_to_columns,
        media_to_channel={
            "facebook_impr": "facebook",
            "instagram_impr": "instagram",
            "tiktok_impr": "tiktok",
        },
        media_spend_to_channel={
            "facebook_spend": "facebook",
            "instagram_spend": "instagram",
            "tiktok_spend": "tiktok",
        },
    )
    input_data = model_trial.load_dataframe_input(df, spec)
    model = model_trial.build_platform_weight_model(input_data)
    with model:
        idata = pm.sample(
            draws=10,
            tune=10,
            chains=2,
            cores=1,
            random_seed=1,
            progressbar=False,
        )
    w = idata.posterior["w"].values
    assert np.allclose(w.sum(axis=-1), 1, atol=1e-6)
