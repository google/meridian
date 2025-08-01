import arviz as az
import numpy as np
import pandas as pd
import pandas.testing as pdt
import scipy.stats
import xarray as xr
from absl.testing import absltest

from meridian.david import diagnostics


class PosteriorSummaryTest(absltest.TestCase):

    def test_basic_stats(self):
        arr = np.array([[0.5, 1.0, 1.5, 2.0]])
        idata = az.from_dict(posterior={"beta_m": arr[:, :, None]})
        result = diagnostics.posterior_summary(idata, "beta_m")
        samples = arr.reshape(-1)
        mean = samples.mean()
        sd = samples.std(ddof=1)
        t = mean / sd
        alpha = 0.05
        lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        self.assertAlmostEqual(result["mean"], mean)
        self.assertAlmostEqual(result["sd"], sd)
        self.assertAlmostEqual(result["t_stat"], t)
        self.assertAlmostEqual(result["95.0%_ci_lower"], lo)
        self.assertAlmostEqual(result["95.0%_ci_upper"], hi)


class ComputeVifTest(absltest.TestCase):

    def test_vif_values(self):
        X = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
        result = diagnostics.compute_vif(X)
        expected = []
        for j in range(X.shape[1]):
            y = X.iloc[:, j].values
            X_other = X.drop(X.columns[j], axis=1).values
            coef, *_ = np.linalg.lstsq(X_other, y, rcond=None)
            y_hat = X_other @ coef
            ssr = np.sum((y - y_hat) ** 2)
            sst = np.sum((y - y.mean()) ** 2)
            r2 = 1.0 - ssr / sst
            expected.append(1.0 / (1.0 - r2 + 1e-12))
        np.testing.assert_allclose(result.values, np.array(expected))


class DurbinWatsonTest(absltest.TestCase):

    def test_statistic(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        diff = np.diff(residuals)
        expected = np.sum(diff**2) / np.sum(residuals**2)
        self.assertAlmostEqual(diagnostics.durbin_watson(residuals), expected)


class JarqueBeraTest(absltest.TestCase):

    def test_statistic(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        stat, p = diagnostics.jarque_bera(residuals)
        self.assertGreaterEqual(stat, 0.0)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class MassAboveThresholdTest(absltest.TestCase):

    def test_prob_above(self):
        arr = np.array([[0.9, 1.1, 1.2, 0.8]])
        idata = az.from_dict(posterior={"beta_m": arr[:, :, None]})
        result = diagnostics.mass_above_threshold(
            idata, "beta_m", cred_mass=0.95, log_space=True
        )
        samples = np.log(arr.reshape(-1))
        prob = np.mean(samples > 0.0)
        self.assertAlmostEqual(result["prob_above"], prob)
        self.assertEqual(result["meets_cred"], prob >= 0.95)


class AugmentedDickeyFullerTest(absltest.TestCase):

    def test_statistic(self):
        series = np.array([0.0, 0.1, 0.05, 0.2, 0.15])
        stat, p = diagnostics.augmented_dickey_fuller(series)
        self.assertTrue(np.isfinite(stat))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class BuildDesignMatrixTest(absltest.TestCase):

    def test_matrix_construction(self):
        media = xr.DataArray(
            np.arange(8).reshape(1, 2, 4),
            dims=("geo", "media_time", "media_channel"),
        )
        controls = xr.DataArray(
            np.arange(2).reshape(1, 2, 1),
            dims=("geo", "media_time", "control"),
        )
        result = diagnostics.build_design_matrix(media, controls)
        media_df = (
            media.stack(sample=("geo", "media_time"))
            .transpose("sample", "media_channel")
            .to_pandas()
        )
        ctrl_df = (
            controls.stack(sample=("geo", "media_time"))
            .to_pandas()
        )
        expected = pd.concat([media_df, ctrl_df], axis=1)
        expected.insert(0, "const", 1.0)
        expected = expected.astype("float64")
        pdt.assert_frame_equal(result, expected)


class InferTimeDimTest(absltest.TestCase):

    def test_detect_known_dims(self):
        for dim in ("time", "media_time", "geo_time"):
            da = xr.DataArray(np.zeros((1, 2)), dims=("geo", dim))
            self.assertEqual(diagnostics._infer_time_dim(da), dim)

    def test_error_on_unknown_dim(self):
        da = xr.DataArray(np.zeros((1, 2)), dims=("geo", "not_time"))
        with self.assertRaises(ValueError):
            diagnostics._infer_time_dim(da)


class BreuschPaganGodfreyTest(absltest.TestCase):

    def test_statistic(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        exog = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [4, 3, 2, 1]})
        result = diagnostics.breusch_pagan_godfrey(residuals, exog)
        self.assertGreaterEqual(result["lm_stat"], 0.0)
        self.assertGreaterEqual(result["lm_pvalue"], 0.0)
        self.assertLessEqual(result["lm_pvalue"], 1.0)


class ResidualKurtosisTest(absltest.TestCase):

    def test_value(self):
        residuals = np.array([0.5, 0.1, -0.3, 0.2])
        expected = scipy.stats.kurtosis(residuals, fisher=True)
        result = diagnostics.residual_kurtosis(residuals)
        self.assertAlmostEqual(result, expected)


class InspectCorrelationMatrixTest(absltest.TestCase):

    def test_pairs(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 0, 1]})
        pairs = diagnostics.inspect_correlation_matrix(X, threshold=0.9)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(pairs.loc[0, ["var1", "var2"]]), {"a", "b"})


if __name__ == "__main__":
    absltest.main()
