import sys
import types

import sys
import types

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from absl.testing import absltest

# Provide a minimal altair stub for tests.
alt = types.ModuleType('altair')


class Chart:
  def __init__(self, df):
    self.data = df
    self.mark_bar_args = None
    self.mark_line_args = None
    self.encode_args = None
    self.properties_args = None

  def mark_bar(self, **kwargs):
    self.mark_bar_args = kwargs
    return self

  def mark_line(self, **kwargs):
    self.mark_line_args = kwargs
    return self

  def encode(self, *args, **kwargs):
    self.encode_args = (args, kwargs)
    return self

  def properties(self, **kwargs):
    self.properties_args = kwargs
    return self


def X(field, **kwargs):
  return {"field": field, **kwargs}


def Y(field, **kwargs):
  return {"field": field, **kwargs}


def Color(field, **kwargs):
  return {"field": field, **kwargs}


def Bin(**kwargs):
  return kwargs


class _DataTransformers:
  def __init__(self):
    self.disable_called = False

  def disable_max_rows(self):
    self.disable_called = True


alt.data_transformers = _DataTransformers()

alt.Chart = Chart
alt.X = X
alt.Y = Y
alt.Color = Color
alt.Bin = Bin
class TitleParams(dict):
  def __init__(self, **kw):
    super().__init__(**kw)

alt.TitleParams = TitleParams
class FacetChart(Chart):
  pass


class LayerChart(Chart):
  pass


class HConcatChart(Chart):
  pass


class VConcatChart(Chart):
  pass

alt.FacetChart = FacetChart
alt.LayerChart = LayerChart
alt.HConcatChart = HConcatChart
alt.VConcatChart = VConcatChart
sys.modules['altair'] = alt

from meridian.david import betas
from meridian.model import model as meridian_model
from meridian import constants as c


tfpd = tfp.distributions


class DummyDataArray:
  def __init__(self, values):
    self.values = np.array(values)


class DummyPosterior:
  def __init__(self, mapping):
    self._mapping = {k: DummyDataArray(v) for k, v in mapping.items()}

  @property
  def data_vars(self):
    return self._mapping

  def __getitem__(self, key):
    return self._mapping[key]


class DummyInferenceData:
  def __init__(self, mapping=None):
    if mapping is not None:
      self.posterior = DummyPosterior(mapping)


class DummyMeridian:
  def __init__(self, posterior=None, priors=None, channels=None):
    self.inference_data = DummyInferenceData(posterior)
    self.prior_broadcast = types.SimpleNamespace(**(priors or {}))
    if channels is not None:
      media_channel = types.SimpleNamespace(values=np.array(channels))
    else:
      media_channel = None
    self.input_data = types.SimpleNamespace(media_channel=media_channel)


class CheckFittedTest(absltest.TestCase):

  def test_not_fitted_raises_error(self):
    m = DummyMeridian()
    with self.assertRaises(meridian_model.NotFittedModelError):
      betas._check_fitted(m)

  def test_fitted_does_not_raise(self):
    m = DummyMeridian({'param': [1.0]})
    betas._check_fitted(m)  # should not raise


class GetPosteriorSamplesTest(absltest.TestCase):

  def test_unknown_parameter_raises(self):
    m = DummyMeridian({'a': [1.0]})
    with self.assertRaises(ValueError):
      betas.get_posterior_samples(m, 'b')

  def test_not_fitted_raises(self):
    m = DummyMeridian()
    with self.assertRaises(meridian_model.NotFittedModelError):
      betas.get_posterior_samples(m, 'a')

  def test_returns_flattened_values(self):
    m = DummyMeridian({'a': [[1, 2], [3, 4]]})
    result = betas.get_posterior_samples(m, 'a')
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))


class EstimateDistributionTest(absltest.TestCase):

  def test_normal(self):
    samples = np.array([1.0, 2.0, 3.0])
    dist = tfpd.Normal(loc=0.0, scale=1.0)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'Normal',
            'loc': float(np.mean(samples)),
            'scale': float(np.std(samples, ddof=1)),
        },
    )

  def test_log_normal(self):
    samples = np.exp([0.0, 1.0])
    dist = tfpd.LogNormal(loc=0.0, scale=1.0)
    logs = np.log(samples)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'LogNormal',
            'loc': float(np.mean(logs)),
            'scale': float(np.std(logs, ddof=1)),
        },
    )

  def test_half_normal(self):
    samples = np.array([1.0, 2.0, 3.0])
    dist = tfpd.HalfNormal(scale=1.0)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'HalfNormal',
            'scale': float(np.sqrt(np.mean(samples ** 2))),
        },
    )

  def test_uniform(self):
    samples = np.array([1.0, 2.0, 3.0])
    dist = tfpd.Uniform(low=0.0, high=1.0)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'Uniform',
            'low': float(samples.min()),
            'high': float(samples.max()),
        },
    )

  def test_beta(self):
    samples = np.array([0.2, 0.5, 0.8])
    dist = tfpd.Beta(concentration1=1.0, concentration0=1.0)
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta_v = (1 - mean) * (mean * (1 - mean) / var - 1)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'Beta',
            'concentration1': float(alpha),
            'concentration0': float(beta_v),
        },
    )

  def test_deterministic(self):
    samples = np.array([1.5, 1.5])
    dist = tfpd.Deterministic(loc=1.0)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'Deterministic',
            'loc': float(np.mean(samples)),
        },
    )

  def test_fallback(self):
    samples = np.array([1.0, 2.0])
    dist = tfpd.Gamma(concentration=1.0, rate=1.0)
    result = betas.estimate_distribution(samples, dist)
    self.assertEqual(
        result,
        {
            'distribution': 'Gamma',
            'mean': float(np.mean(samples)),
            'stddev': float(np.std(samples, ddof=1)),
        },
    )


class FitParameterDistributionTest(absltest.TestCase):

  def test_unknown_prior_raises(self):
    m = DummyMeridian({'a': [1.0]})
    with self.assertRaises(ValueError):
      betas.fit_parameter_distribution(m, 'b')

  def test_estimates_from_prior_and_samples(self):
    posterior = {'a': [1.0, 2.0, 3.0]}
    prior = {'a': tfpd.Normal(loc=0.0, scale=1.0)}
    m = DummyMeridian(posterior, prior)
    expected = betas.estimate_distribution(
        np.array(posterior['a']), prior['a']
    )
    result = betas.fit_parameter_distribution(m, 'a')
    self.assertEqual(result, expected)


class PlotPosteriorTest(absltest.TestCase):

  def test_returns_chart(self):
    posterior = {'a': [1.0, 2.0]}
    m = DummyMeridian(posterior)
    chart = betas.plot_posterior(m, 'a')
    self.assertIsInstance(chart, alt.Chart)
    self.assertEqual(chart.properties_args.get('title'), 'Posterior of a')
    self.assertEqual(list(chart.data.columns), ['a'])


class GetPosteriorCoefSamplesTest(absltest.TestCase):

  def test_unknown_parameter_raises(self):
    m = DummyMeridian({'a': [[[1.0]]]})
    with self.assertRaises(ValueError):
      betas.get_posterior_coef_samples(m, 'b', 0)

  def test_not_fitted_raises(self):
    m = DummyMeridian()
    with self.assertRaises(meridian_model.NotFittedModelError):
      betas.get_posterior_coef_samples(m, 'a', 0)

  def test_index_out_of_bounds_raises(self):
    m = DummyMeridian({'a': [[[1.0, 2.0]]]})
    with self.assertRaises(IndexError):
      betas.get_posterior_coef_samples(m, 'a', 5)

  def test_returns_flattened_values(self):
    m = DummyMeridian({'a': [[[1, 2], [3, 4]]]})
    result = betas.get_posterior_coef_samples(m, 'a', 1)
    np.testing.assert_array_equal(result, np.array([2, 4]))


class FitParameterCoefDistributionTest(absltest.TestCase):

  def test_unknown_prior_raises(self):
    m = DummyMeridian({'a': [[[1.0]]]})
    with self.assertRaises(ValueError):
      betas.fit_parameter_coef_distribution(m, 'b', 0)

  def test_estimates_from_prior_and_samples(self):
    posterior = {'a': [[[1.0, 2.0], [3.0, 4.0]]]}
    prior = {'a': tfpd.Normal(loc=0.0, scale=1.0)}
    m = DummyMeridian(posterior, prior)
    expected = betas.estimate_distribution(
        np.array([2.0, 4.0]), prior['a']
    )
    result = betas.fit_parameter_coef_distribution(m, 'a', 1)
    self.assertEqual(result, expected)


class GetScalarPriorNamesTest(absltest.TestCase):

  def test_not_fitted_raises(self):
    m = DummyMeridian()
    with self.assertRaises(meridian_model.NotFittedModelError):
      betas.get_scalar_prior_names(m)

  def test_returns_scalar_names(self):
    posterior = {
        'a': [[1.0]],  # scalar
        'b': [[[1.0]]],  # has extra dim
        'c': [[2.0]],
    }
    m = DummyMeridian(posterior)
    names = betas.get_scalar_prior_names(m)
    self.assertEqual(set(names), {'a', 'c'})


class GetBetaChannelNamesTest(absltest.TestCase):

  def test_returns_channels(self):
    channels = ['Ch0', 'Ch1', 'Ch2']
    m = DummyMeridian({'a': [[1.0]]}, channels=channels)
    self.assertEqual(betas.get_beta_channel_names(m), channels)

  def test_no_channels_returns_empty(self):
    m = DummyMeridian({'a': [[1.0]]})
    self.assertEqual(betas.get_beta_channel_names(m), [])


class PlotPosteriorCoefTest(absltest.TestCase):

  def test_returns_chart(self):
    posterior = {'a': [[[1.0, 2.0]]]}  # shape (1,1,2)
    m = DummyMeridian(posterior)
    chart = betas.plot_posterior_coef(m, 'a', 1, maxbins=20, width=300, height=100)
    self.assertIsInstance(chart, alt.Chart)
    self.assertEqual(chart.properties_args.get('title'), 'Posterior of a[1] (log scale)')
    self.assertEqual(chart.properties_args.get('width'), 300)
    self.assertEqual(chart.properties_args.get('height'), 100)
    self.assertEqual(list(chart.data.columns), ['a[1]'])
    # Ensure encoding used the provided maxbins and quoted field name.
    _, enc_kwargs = chart.encode_args
    self.assertEqual(enc_kwargs['x']['bin']['maxbins'], 20)
    self.assertEqual(enc_kwargs['x']['field'], '`a[1]`:Q')


class OptimalFreqSafeTest(absltest.TestCase):

  class DummyDataset:
    def __init__(self):
      self.frequency = types.SimpleNamespace(values=np.array([1.0, 2.0, 3.0]))
      self.rf_channel = types.SimpleNamespace(values=np.array(['A', 'B']))
      self.isel_args = None

    def isel(self, **kw):
      self.isel_args = kw
      return self

  class DummyAnalyzer:
    def __init__(self, ds):
      self.ds = ds
      self.called_with = None

    def optimal_freq(self, **kw):
      self.called_with = kw
      return self.ds

  def test_filters_dataset_and_passes_kwargs(self):
    ds = self.DummyDataset()
    ana = self.DummyAnalyzer(ds)
    result = betas.optimal_freq_safe(
        ana,
        selected_freqs=[2.0],
        selected_channels=['B'],
        foo='bar',
    )

    self.assertIs(result, ds)
    self.assertEqual(ana.called_with, {'foo': 'bar'})
    np.testing.assert_array_equal(
        ds.isel_args['frequency'], np.array([False, True, False])
    )
    np.testing.assert_array_equal(
        ds.isel_args['rf_channel'], np.array([False, True])
    )


class ViewTransformedVariableTest(absltest.TestCase):

  class DA:
    def __init__(self, values, dims):
      self.values = np.array(values)
      self.dims = dims

  class DummyMeridian:
    def __init__(self):
      posterior = types.SimpleNamespace(
          alpha_m=ViewTransformedVariableTest.DA([[[0.0]]], ('chain', 'draw', 'channel')),
          ec_m=ViewTransformedVariableTest.DA([[[1.0]]], ('chain', 'draw', 'channel')),
          slope_m=ViewTransformedVariableTest.DA([[[1.0]]], ('chain', 'draw', 'channel')),
      )
      self.inference_data = types.SimpleNamespace(posterior=posterior)
      self.n_times = 2
      self.n_rf_channels = 0
      self.input_data = types.SimpleNamespace(
          geo=types.SimpleNamespace(values=np.array(['G0'])),
          time=types.SimpleNamespace(values=np.array([0, 1])),
          media=types.SimpleNamespace(
              coords={
                  c.MEDIA_CHANNEL: types.SimpleNamespace(values=np.array(['Ch0']))
              }
          ),
      )
      media_scaled = ViewTransformedVariableTest.DA(
          np.array([[[1.0], [2.0]]]), ('geo', 'time', 'channel')
      )
      self.media_tensors = types.SimpleNamespace(media_scaled=media_scaled)

    def adstock_hill_media(self, media, alpha, ec, slope, n_times_output):
      return media

  def test_returns_dataframe_and_chart(self):
    m = self.DummyMeridian()
    df, chart = betas.view_transformed_variable(m, 'Ch0')
    self.assertIsInstance(df, pd.DataFrame)
    self.assertIsInstance(chart, alt.Chart)
    expected = pd.DataFrame({
        'time': [0, 1],
        'channel': ['Ch0', 'Ch0'],
        'block': ['MEDIA', 'MEDIA'],
        'value': np.array([1.0, 2.0], dtype=np.float32),
    })
    pd.testing.assert_frame_equal(df, expected)
    self.assertIs(chart.data, df)


if __name__ == '__main__':
  absltest.main()
