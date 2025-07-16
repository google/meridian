import arviz as az
import numpy as np
import pandas as pd
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
    expected = np.sum(diff ** 2) / np.sum(residuals ** 2)
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


if __name__ == "__main__":
  absltest.main()
