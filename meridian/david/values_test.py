import numpy as np
import pandas as pd
import xarray as xr
from absl.testing import absltest

from meridian.david import values

mock = absltest.mock


class ExtractHillPosteriorTest(absltest.TestCase):

  def test_extracts_channel_parameters(self):
    ec = np.array([[1.0, 2.0], [3.0, 4.0]])
    slope = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = values.extract_hill_posterior(
        ec=ec,
        slope=slope,
        channel="b",
        channel_names=["a", "b"],
    )
    self.assertEqual(res["channel_index"], 1)
    self.assertEqual(res["channel_name"], "b")
    np.testing.assert_array_equal(res["samples"]["ec"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(res["samples"]["slope"], np.array([0.2, 0.4]))
    self.assertAlmostEqual(res["summary"]["ec"]["mean"], 3.0)
    self.assertAlmostEqual(res["summary"]["slope"]["mean"], 0.3)


class HillCurveQuantilesTest(absltest.TestCase):

  def test_evaluates_curve(self):
    media = np.linspace(0, 1.0, 3)
    ec_samples = np.array([1.0, 2.0])
    slope_samples = np.array([2.0, 2.0])
    res = values.hill_curve_quantiles(
        media, ec_samples, slope_samples, quantiles=(0.5,)
    )
    self.assertEqual(res["media"].shape, (3,))
    self.assertIn(0.5, res["quantiles"])
    self.assertEqual(res["mean"].shape, (3,))


class HillForChannelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    class DummyMMM:
      pass

    self.mmm = DummyMMM()
    self.mmm.ec = np.array([[1.0, 2.0], [3.0, 4.0]])
    self.mmm.slope = np.array([[0.1, 0.2], [0.3, 0.4]])
    self.mmm.media_channel_names = ["a", "b"]

  def test_returns_posterior_and_curve_df(self):
    res = values.hill_for_channel(
        self.mmm,
        "b",
        media_grid=np.array([0.0, 1.0]),
        quantiles=(0.5,),
    )
    self.assertEqual(res["channel_index"], 1)
    self.assertEqual(res["channel_name"], "b")
    self.assertIsInstance(res["curve_df"], pd.DataFrame)
    self.assertListEqual(list(res["curve_df"].columns), ["media", "response_mean", "response_q50"])
    self.assertEqual(len(res["curve_df"]), 2)


class HillCurveDFTest(absltest.TestCase):

  def test_returns_dataframe(self):
    class DummyMMM:
      pass

    mmm = DummyMMM()
    mmm.ec = np.array([[1.0, 2.0]])
    mmm.slope = np.array([[0.1, 0.2]])
    mmm.media_channel_names = ["a", "b"]

    df = values.hill_curve_df(mmm, "b", media_grid=np.array([0.0, 1.0]), quantiles=(0.5,))
    self.assertIsInstance(df, pd.DataFrame)
    self.assertListEqual(list(df.columns), ["media", "response_mean", "response_q50"])
    self.assertEqual(len(df), 2)


class GetCurveParameterDataTest(absltest.TestCase):

  def test_calls_analyzer_and_returns_dataframe(self):
    mmm = object()
    expected_df = pd.DataFrame({'a': [1]})
    with mock.patch.object(values.analyzer, 'Analyzer') as MockAnalyzer:
      MockAnalyzer.return_value.hill_curves.return_value = expected_df
      result = values.get_curve_parameter_data(mmm, confidence_level=0.5)
      MockAnalyzer.assert_called_once_with(mmm)
      MockAnalyzer.return_value.hill_curves.assert_called_once_with(
          confidence_level=0.5)
      self.assertIs(result, expected_df)


class GetBudgetOptimisationDataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mmm = object()
    self.rf_ds = xr.Dataset(
        coords={
            values.C.FREQUENCY: [1, 2],
            values.C.RF_CHANNEL: ['A', 'B'],
            values.C.METRIC: [values.C.MEAN],
        },
        data_vars={
            values.C.ROI: (
                [values.C.FREQUENCY, values.C.RF_CHANNEL, values.C.METRIC],
                [[[10], [20]], [[30], [40]]],
            ),
            values.C.OPTIMAL_FREQUENCY: ([values.C.RF_CHANNEL], [1, 2]),
        },
    )
    self.sum_ds = xr.Dataset(
        coords={
            values.C.CHANNEL: ['A', 'B', 'C', values.C.ALL_CHANNELS],
            values.C.METRIC: [values.C.MEAN],
        },
        data_vars={
            values.C.ROI: (
                [values.C.CHANNEL, values.C.METRIC],
                [[1], [2], [3], [4]],
            ),
        },
    )

  def test_returns_dataframe_with_rf_and_non_rf_channels(self):
    with mock.patch.object(values.analyzer, 'Analyzer') as MockAnalyzer:
      MockAnalyzer.return_value.optimal_freq.return_value = self.rf_ds
      MockAnalyzer.return_value.summary_metrics.return_value = self.sum_ds
      result = values.get_budget_optimisation_data(
          self.mmm,
          selected_channels=['B', 'C'],
          selected_times=['t'],
          use_kpi=True,
          confidence_level=0.9,
      )
      MockAnalyzer.assert_called_once_with(self.mmm)
      MockAnalyzer.return_value.optimal_freq.assert_called_once()
      MockAnalyzer.return_value.summary_metrics.assert_called_once()

      _, opt_kwargs = MockAnalyzer.return_value.optimal_freq.call_args
      self.assertEqual(opt_kwargs['selected_times'], ['t'])
      self.assertTrue(opt_kwargs['use_kpi'])
      self.assertEqual(opt_kwargs['confidence_level'], 0.9)

      _, sum_kwargs = MockAnalyzer.return_value.summary_metrics.call_args
      self.assertEqual(sum_kwargs['selected_times'], ['t'])
      self.assertTrue(sum_kwargs['use_kpi'])
      self.assertEqual(sum_kwargs['confidence_level'], 0.9)
      self.assertFalse(sum_kwargs['include_non_paid_channels'])
      self.assertNotIn('optimal_frequency', sum_kwargs)

    expected = pd.DataFrame({
        values.C.RF_CHANNEL: ['B', 'B', 'C'],
        values.C.FREQUENCY: [1, 2, np.nan],
        values.C.ROI: [20.0, 40.0, 3.0],
        values.C.OPTIMAL_FREQUENCY: [2, 2, np.nan],
    })
    expected = expected[[values.C.RF_CHANNEL, values.C.FREQUENCY,
                         values.C.ROI, values.C.OPTIMAL_FREQUENCY]]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

  def test_returns_non_rf_rows_when_no_rf_channels(self):
    sum_ds = xr.Dataset(
        coords={
            values.C.CHANNEL: ['C'],
            values.C.METRIC: [values.C.MEAN],
        },
        data_vars={
            values.C.ROI: ([values.C.CHANNEL, values.C.METRIC], [[5]]),
        },
    )

    with mock.patch.object(values.analyzer, 'Analyzer') as MockAnalyzer:
      MockAnalyzer.return_value.optimal_freq.side_effect = ValueError('no rf')
      MockAnalyzer.return_value.summary_metrics.return_value = sum_ds
      result = values.get_budget_optimisation_data(object())
      MockAnalyzer.assert_called_once()
      MockAnalyzer.return_value.optimal_freq.assert_called_once()
      MockAnalyzer.return_value.summary_metrics.assert_called_once()
      _, sum_kwargs = MockAnalyzer.return_value.summary_metrics.call_args
      self.assertFalse(sum_kwargs['include_non_paid_channels'])

    expected = pd.DataFrame({
        values.C.RF_CHANNEL: ['C'],
        values.C.FREQUENCY: [np.nan],
        values.C.ROI: [5.0],
        values.C.OPTIMAL_FREQUENCY: [np.nan],
    })
    expected = expected[[values.C.RF_CHANNEL, values.C.FREQUENCY,
                         values.C.ROI, values.C.OPTIMAL_FREQUENCY]]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)



class GetActualVsFittedDataFixedTest(absltest.TestCase):

  def test_calls_analyzer_and_returns_dataframe(self):
    mmm = object()
    ds = xr.Dataset({"x": (['time'], [1, 2])}, coords={"time": ['a', 'b']})
    with mock.patch.object(values.analyzer, 'Analyzer') as MockAnalyzer:
      MockAnalyzer.return_value.expected_vs_actual_data.return_value = ds
      result = values.get_actual_vs_fitted_data_fixed(
          mmm,
          confidence_level=0.8,
          aggregate_geos=True,
          aggregate_times=True,
          selected_times=['a'],
      )
      MockAnalyzer.assert_called_once_with(mmm)
      MockAnalyzer.return_value.expected_vs_actual_data.assert_called_once_with(
          confidence_level=0.8,
          aggregate_geos=True,
          aggregate_times=True,
      )
    expected = ds.sel(time=['a']).to_dataframe().reset_index()
    pd.testing.assert_frame_equal(result, expected)


if __name__ == '__main__':
  absltest.main()
