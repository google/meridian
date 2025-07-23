import sys
import types

import numpy as np
import tensorflow_probability as tfp
from absl.testing import absltest

# Provide a minimal altair stub if the real library isn't available.
try:
  import altair as alt  # noqa: F401
except ModuleNotFoundError:
  alt = types.ModuleType('altair')

  class Chart:
    def __init__(self, df):
      self.data = df
      self.mark_bar_args = None
      self.encode_args = None
      self.properties_args = None

    def mark_bar(self, **kwargs):
      self.mark_bar_args = kwargs
      return self

    def encode(self, *args, **kwargs):
      self.encode_args = (args, kwargs)
      return self

    def properties(self, **kwargs):
      self.properties_args = kwargs
      return self

  def X(field, **kwargs):
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
  alt.Bin = Bin
  sys.modules['altair'] = alt

from meridian.david import betas
from meridian.model import model as meridian_model


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
  def __init__(self, posterior=None, priors=None):
    self.inference_data = DummyInferenceData(posterior)
    self.prior_broadcast = types.SimpleNamespace(**(priors or {}))


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
    # Ensure encoding used the provided maxbins.
    _, enc_kwargs = chart.encode_args
    self.assertEqual(enc_kwargs['x']['bin']['maxbins'], 20)


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


if __name__ == '__main__':
  absltest.main()
