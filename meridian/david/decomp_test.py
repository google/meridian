import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import xarray as xr
from absl.testing import absltest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from meridian.david import decomp


class _InferenceData(SimpleNamespace):
  pass


class ControlContributionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.controls_scaled = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    self.revenue_per_kpi = np.array(
        [
            [10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    posterior_gamma = np.array(
        [
            [
                [[0.1, 0.01], [0.2, 0.02]],
                [[0.15, 0.01], [0.25, 0.02]],
            ]
        ],
        dtype=np.float32,
    )
    prior_gamma = np.array(
        [
            [
                [[0.05, 0.0], [0.1, 0.0]],
                [[0.07, 0.0], [0.09, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    coords = {
        "chain": np.arange(posterior_gamma.shape[0]),
        "draw": np.arange(posterior_gamma.shape[1]),
        "geo": ["g0", "g1"],
        "control_variable": ["c0", "c1"],
    }
    posterior = xr.Dataset({
        "gamma_gc": (("chain", "draw", "geo", "control_variable"), posterior_gamma)
    }, coords=coords)
    prior = xr.Dataset({
        "gamma_gc": (("chain", "draw", "geo", "control_variable"), prior_gamma)
    }, coords=coords)

    self.base_meridian = SimpleNamespace(
        n_controls=2,
        controls_scaled=self.controls_scaled,
        n_geos=2,
        n_times=3,
        revenue_per_kpi=self.revenue_per_kpi,
        is_national=False,
        input_data=SimpleNamespace(
            control_variable=["c0", "c1"],
            geo=["g0", "g1"],
            time=["t0", "t1", "t2"],
            kpi_type="REVENUE",
        ),
        inference_data=_InferenceData(posterior=posterior, prior=prior),
    )

  def _build_analyzer(self, **overrides):
    meridian_copy = copy.deepcopy(self.base_meridian)
    input_data_override = overrides.pop("input_data", None)
    for name, value in overrides.items():
      setattr(meridian_copy, name, value)
    if input_data_override:
      for key, value in input_data_override.items():
        setattr(meridian_copy.input_data, key, value)
    return SimpleNamespace(_meridian=meridian_copy)

  def test_aggregate_posterior_revenue(self):
    analyzer = self._build_analyzer()
    result = decomp.control_contribution(analyzer, "c0", use_posterior=True, use_kpi=False)

    self.assertEqual(result.dims, ("metric",))
    self.assertSequenceEqual(
        result.coords["metric"].values.tolist(), ["mean", "median", "ci_low", "ci_high"]
    )
    self.assertEqual(result.attrs["distribution"], "posterior")
    self.assertEqual(result.attrs["scale"], "revenue")

    gamma = self.base_meridian.inference_data.posterior["gamma_gc"].values[:, :, :, 0]
    controls = self.controls_scaled[:, :, 0]
    revenue = self.revenue_per_kpi
    draws = (
        gamma[..., :, np.newaxis]
        * controls[np.newaxis, np.newaxis, :, :]
        * revenue[np.newaxis, np.newaxis, :, :]
    ).sum(axis=(-2, -1)).reshape(-1)
    expected = np.array([
        draws.mean(),
        np.median(draws),
        np.quantile(draws, 0.025),
        np.quantile(draws, 0.975),
    ])
    np.testing.assert_allclose(result.values, expected.astype(np.float32))

  def test_geo_time_breakdown_on_kpi_scale(self):
    analyzer = self._build_analyzer()
    result = decomp.control_contribution(
        analyzer,
        "c0",
        use_kpi=True,
        aggregate_geos=False,
        aggregate_times=False,
        selected_geos=["g0"],
        selected_times=[True, False, True],
        confidence_level=0.8,
    )

    self.assertEqual(result.dims, ("geo", "time", "metric"))
    np.testing.assert_array_equal(result.coords["geo"], np.array(["g0"]))
    np.testing.assert_array_equal(result.coords["time"], np.array(["t0", "t2"]))
    self.assertTrue(np.allclose(result.attrs["confidence_level"], 0.8))

    gamma = self.base_meridian.inference_data.posterior["gamma_gc"].values[:, :, 0, 0]
    controls = self.controls_scaled[0, [0, 2], 0]
    draws = gamma[..., np.newaxis] * controls[np.newaxis, np.newaxis, :]
    draws = draws.reshape(-1, controls.shape[0])
    expected = np.stack([
        draws.mean(axis=0),
        np.median(draws, axis=0),
        np.quantile(draws, 0.1, axis=0),
        np.quantile(draws, 0.9, axis=0),
    ], axis=-1)
    np.testing.assert_allclose(result.values[0], expected.astype(np.float32))

  def test_uses_prior_when_requested(self):
    analyzer = self._build_analyzer()
    result = decomp.control_contribution(
        analyzer,
        0,
        use_posterior=False,
        use_kpi=True,
        aggregate_geos=False,
        aggregate_times=True,
    )

    self.assertEqual(result.dims, ("geo", "metric"))
    prior_gamma = self.base_meridian.inference_data.prior["gamma_gc"].values[:, :, :, 0]
    controls = self.controls_scaled[:, :, 0]
    draws = (
        prior_gamma[..., np.newaxis]
        * controls[np.newaxis, np.newaxis, :, :]
    ).sum(axis=-1).reshape(-1, controls.shape[0])
    expected = np.stack([
        draws.mean(axis=0),
        np.median(draws, axis=0),
        np.quantile(draws, 0.025, axis=0),
        np.quantile(draws, 0.975, axis=0),
    ], axis=-1)
    np.testing.assert_allclose(result.values, expected.astype(np.float32))

  def test_missing_revenue_data_raises(self):
    analyzer = self._build_analyzer(
        revenue_per_kpi=None,
        input_data={"kpi_type": "KPI"},
    )
    with self.assertRaisesRegex(ValueError, "revenue_per_kpi is missing"):
      decomp.control_contribution(analyzer, "c0", use_kpi=False)

  def test_unknown_control_raises(self):
    analyzer = self._build_analyzer()
    with self.assertRaises(KeyError):
      decomp.control_contribution(analyzer, "unknown")


if __name__ == "__main__":
  absltest.main()
