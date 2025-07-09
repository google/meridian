import sys
import types

# Provide a minimal matplotlib stub if matplotlib is not installed.
try:
    import matplotlib.pyplot as _plt  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    plt_stub = types.ModuleType("matplotlib.pyplot")
    for name in [
        "figure",
        "hist",
        "plot",
        "xlabel",
        "ylabel",
        "title",
        "legend",
        "tight_layout",
        "show",
    ]:
        setattr(plt_stub, name, lambda *a, **k: None)
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.pyplot = plt_stub
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", plt_stub)

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from meridian import diagnostics


class FakeDataArray:
    def __init__(self, data, dims):
        self._data = np.asarray(data)
        self.dims = list(dims)
        self.sizes = {d: s for d, s in zip(self.dims, self._data.shape)}

    def stack(self, **mapping):
        assert list(mapping.keys()) == ["sample"]
        assert tuple(mapping["sample"]) == ("chain", "draw")
        new_data = self._data.reshape(
            self.sizes.get("chain", 1) * self.sizes.get("draw", 1),
            self.sizes.get("media_channel", 1),
        )
        return FakeDataArray(new_data, ["sample", "media_channel"])

    def isel(self, **indexers):
        idx = indexers.get("media_channel")
        new_data = self._data[..., idx]
        new_dims = [d for d in self.dims if d != "media_channel"]
        return FakeDataArray(new_data, new_dims)

    @property
    def values(self):
        return self._data


class FakeMeridian:
    def __init__(self, beta_media_array):
        self.inference_data = types.SimpleNamespace(
            posterior={"beta_media": beta_media_array} if beta_media_array else {}
        )
        self.sample_args = None

    def sample_posterior(self, **kwargs):
        self.sample_args = kwargs


class PlotMediaCoefLognormalTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        logs = np.array([
            [[0.0, 0.1], [0.5, 0.4]],
            [[0.8, 0.3], [1.0, 0.6]],
        ])
        data = np.exp(logs)
        self.beta_media = FakeDataArray(data, ["chain", "draw", "media_channel"])

    def test_returns_expected_parameters(self):
        model = FakeMeridian(self.beta_media)
        with absltest.mock.patch.object(diagnostics.plt, "show"):
            mu, sigma = diagnostics.plot_media_coef_lognormal(model, channel_idx=1)
        expected_mu = np.log(self.beta_media.values[..., 1].reshape(-1)).mean()
        expected_sigma = np.log(self.beta_media.values[..., 1].reshape(-1)).std(ddof=1)
        self.assertAlmostEqual(mu, expected_mu)
        self.assertAlmostEqual(sigma, expected_sigma)
        self.assertIsNone(model.sample_args)

    def test_samples_posterior_when_requested(self):
        model = FakeMeridian(self.beta_media)
        with absltest.mock.patch.object(diagnostics.plt, "show"):
            diagnostics.plot_media_coef_lognormal(
                model, channel_idx=0, n_draws=10, seed=42
            )
        self.assertIsNotNone(model.sample_args)
        self.assertEqual(
            model.sample_args,
            {
                "n_draws": 10,
                "n_tune": 5,
                "n_chains": 4,
                "seed": 42,
            },
        )

    def test_missing_variable_raises_key_error(self):
        model = FakeMeridian(None)
        with self.assertRaises(KeyError):
            diagnostics.plot_media_coef_lognormal(model)

    def test_missing_dimension_raises_key_error(self):
        data = np.ones((2, 2))
        arr = FakeDataArray(data, ["chain", "draw"])  # no media_channel
        model = FakeMeridian(arr)
        with self.assertRaises(KeyError):
            diagnostics.plot_media_coef_lognormal(model)

    def test_channel_index_out_of_bounds_raises_index_error(self):
        model = FakeMeridian(self.beta_media)
        with self.assertRaises(IndexError):
            diagnostics.plot_media_coef_lognormal(model, channel_idx=5)


class DiagnosticMetricsTest(absltest.TestCase):

    def test_durbin_watson(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        expected = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        self.assertAlmostEqual(diagnostics.durbin_watson(residuals), expected)

    def test_jarque_bera(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        n = residuals.size
        mean = residuals.mean()
        std = residuals.std(ddof=1)
        skew = np.mean(((residuals - mean) / std) ** 3)
        kurt = np.mean(((residuals - mean) / std) ** 4)
        jb_stat = n / 6.0 * (skew ** 2 + 0.25 * (kurt - 3) ** 2)
        jb_p = diagnostics.stats.chi2.sf(jb_stat, 2)
        res_stat, res_p = diagnostics.jarque_bera(residuals)
        self.assertAlmostEqual(res_stat, jb_stat)
        self.assertAlmostEqual(res_p, jb_p)

    def test_breusch_pagan(self):
        residuals = np.array([1.0, -1.0, 0.5, 0.5, -0.5])
        exog = np.arange(1, 6)
        x = np.column_stack([np.ones_like(exog), exog])
        y = residuals ** 2
        params, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ params
        sse = np.sum((y - y_hat) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - sse / tss
        n = residuals.size
        k = x.shape[1]
        lm = n * r2
        lm_p = diagnostics.stats.chi2.sf(lm, k - 1)
        f = (n - k) * r2 / ((k - 1) * (1.0 - r2))
        f_p = diagnostics.stats.f.sf(f, k - 1, n - k)
        res = diagnostics.breusch_pagan(residuals, exog)
        self.assertAlmostEqual(res[0], lm)
        self.assertAlmostEqual(res[1], lm_p)
        self.assertAlmostEqual(res[2], f)
        self.assertAlmostEqual(res[3], f_p)


if __name__ == "__main__":
    absltest.main()
