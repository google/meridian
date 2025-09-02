import pandas as pd
import tensorflow as tf
import xarray as xr
from absl.testing import absltest

from meridian.david import values

mock = absltest.mock


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
    class DummyMMM:
      pass

    self.mmm = DummyMMM()
    self.mmm.rf_tensors = mock.Mock(
        rf_impressions=tf.constant([[[1.0]]], dtype=tf.float64),
        rf_spend=tf.constant([[[2.0]]], dtype=tf.float64),
    )
    self.mmm.input_data = mock.Mock(
        revenue_per_kpi=tf.constant([[3.0]], dtype=tf.float64)
    )
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

  def test_returns_dataframe_with_selected_channel(self):
    with mock.patch.object(values.analyzer, 'Analyzer') as MockAnalyzer:
      MockAnalyzer.return_value.optimal_freq.return_value = self.rf_ds
      result = values.get_budget_optimisation_data(
          self.mmm,
          selected_channels=['B'],
          selected_times=['t'],
          use_kpi=True,
          confidence_level=0.9,
      )
      MockAnalyzer.assert_called_once_with(self.mmm)
      MockAnalyzer.return_value.optimal_freq.assert_called_once()
      _, kwargs = MockAnalyzer.return_value.optimal_freq.call_args
      self.assertEqual(kwargs['selected_times'], ['t'])
      self.assertTrue(kwargs['use_kpi'])
      self.assertEqual(kwargs['confidence_level'], 0.9)
      new_data = kwargs['new_data']
      self.assertEqual(new_data.rf_impressions.dtype, tf.float32)
      self.assertEqual(new_data.rf_spend.dtype, tf.float32)
      self.assertEqual(new_data.revenue_per_kpi.dtype, tf.float32)

    expected = pd.DataFrame({
        values.C.RF_CHANNEL: ['B', 'B'],
        values.C.FREQUENCY: [1, 2],
        values.C.ROI: [20, 40],
        values.C.OPTIMAL_FREQUENCY: [2, 2],
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
