# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.model import equations
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import prior_sampler
from meridian.model import spec
import numpy as np


class PriorDistributionSamplerTest(
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):

  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def test_sample_prior_seed_same_seed(self):
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_and_rf
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=1)
    self.assertEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_sample_prior_different_seed(self):
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_and_rf
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    meridian2 = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian2.sample_prior(n_draws=self._N_DRAWS, seed=2)

    self.assertNotEqual(
        meridian.inference_data.prior, meridian2.inference_data.prior
    )

  def test_prior_distribution_sampler_uses_seed(self):
    model_spec = spec.ModelSpec()
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=model_spec,
    )
    sampler = prior_sampler.PriorDistributionSampler(meridian)
    with mock.patch.object(
        backend, "set_random_seed", autospec=True
    ) as mock_set_seed:
      sampler(n_draws=1, seed=123)
      mock_set_seed.assert_called_once_with(123)

  def test_sample_prior_media_and_rf_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_media_and_rf,
        )
    )

    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_and_rf
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)
    rf_parameters = list(constants.RF_PARAMETER_NAMES)
    rf_parameters.remove(constants.BETA_GRF)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in media_parameters
        ],
        rf_channel_shape: [getattr(prior, attr) for attr in rf_parameters],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_media_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_media_only,
        )
    )

    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in media_parameters
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_sample_prior_media_only_no_controls_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_media_only_no_controls,
        )
    )

    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_only_no_controls
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    media_channel_shape = (1, self._N_DRAWS, self._N_MEDIA_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        media_channel_shape: [
            getattr(prior, attr) for attr in media_parameters
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_media_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_MEDIA_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

    # Control parameters should not exist in the inference data priors.
    for attr in constants.CONTROL_PARAMETERS + constants.GEO_CONTROL_PARAMETERS:
      with self.assertRaises(AttributeError):
        getattr(meridian.inference_data.prior, attr)

  def test_sample_prior_rf_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_rf_only,
        )
    )

    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_rf_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    knots_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    control_shape = (1, self._N_DRAWS, self._N_CONTROLS)
    rf_channel_shape = (1, self._N_DRAWS, self._N_RF_CHANNELS)
    sigma_shape = (
        (1, self._N_DRAWS, self._N_GEOS)
        if meridian.unique_sigma_for_each_geo
        else (1, self._N_DRAWS, 1)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior
    shape_to_params = {
        knots_shape: [
            getattr(prior, attr) for attr in constants.KNOTS_PARAMETERS
        ],
        rf_channel_shape: [
            getattr(prior, attr) for attr in constants.RF_PARAMETER_NAMES
        ],
        control_shape: [
            getattr(prior, attr) for attr in constants.CONTROL_PARAMETERS
        ],
        sigma_shape: [
            getattr(prior, attr) for attr in constants.SIGMA_PARAMETERS
        ],
        geo_shape: [getattr(prior, attr) for attr in constants.GEO_PARAMETERS],
        time_shape: [
            getattr(prior, attr) for attr in constants.TIME_PARAMETERS
        ],
        geo_control_shape: [
            getattr(prior, attr) for attr in constants.GEO_CONTROL_PARAMETERS
        ],
        geo_rf_channel_shape: [
            getattr(prior, attr) for attr in constants.GEO_RF_PARAMETERS
        ],
    }
    for shape, params in shape_to_params.items():
      for param in params:
        self.assertEqual(param.shape, shape)

  def test_injected_sample_prior_media_and_rf_returns_correct_shape(self):
    """Checks validation passes with correct shapes."""
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_media_and_rf,
        )
    )
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_and_rf
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    inference_data = meridian.inference_data

    meridian_with_inference_data = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  def test_injected_sample_prior_media_only_returns_correct_shape(self):
    """Checks validation passes with correct shapes."""
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_media_only,
        )
    )
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    inference_data = meridian.inference_data

    meridian_with_inference_data = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  def test_injected_sample_prior_rf_only_returns_correct_shape(self):
    """Checks validation passes with correct shapes."""
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "_sample_prior",
            autospec=True,
            return_value=self.test_dist_rf_only,
        )
    )
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_rf_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS)
    inference_data = meridian.inference_data

    meridian_with_inference_data = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
        inference_data=inference_data,
    )

    self.assertEqual(
        meridian_with_inference_data.inference_data, inference_data
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="control_variables",
          coord=constants.CONTROL_VARIABLE,
          mismatched_priors={
              constants.GAMMA_C: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_CONTROLS + 1,
              ),
              constants.GAMMA_GC: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_CONTROLS + 1,
              ),
              constants.XI_C: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_CONTROLS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_CONTROLS + 1,
          expected_coord_size=input_data_samples._N_CONTROLS,
      ),
      dict(
          testcase_name="geos",
          coord=constants.GEO,
          mismatched_priors={
              constants.BETA_GM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_MEDIA_CHANNELS,
              ),
              constants.BETA_GRF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_RF_CHANNELS,
              ),
              constants.GAMMA_GC: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
                  input_data_samples._N_CONTROLS,
              ),
              constants.TAU_G: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_GEOS + 1,
          expected_coord_size=input_data_samples._N_GEOS,
      ),
      dict(
          testcase_name="knots",
          coord=constants.KNOTS,
          mismatched_priors={
              constants.KNOT_VALUES: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_TIMES_SHORT + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_TIMES_SHORT + 1,
          expected_coord_size=input_data_samples._N_TIMES_SHORT,
      ),
      dict(
          testcase_name="times",
          coord=constants.TIME,
          mismatched_priors={
              constants.MU_T: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_TIMES_SHORT + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_TIMES_SHORT + 1,
          expected_coord_size=input_data_samples._N_TIMES_SHORT,
      ),
      dict(
          testcase_name="media_channels",
          coord=constants.MEDIA_CHANNEL,
          mismatched_priors={
              constants.ALPHA_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.BETA_GM: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.BETA_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.EC_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.ETA_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.ROI_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
              constants.SLOPE_M: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_MEDIA_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_MEDIA_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_MEDIA_CHANNELS,
      ),
      dict(
          testcase_name="rf_channels",
          coord=constants.RF_CHANNEL,
          mismatched_priors={
              constants.ALPHA_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.BETA_GRF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_GEOS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.BETA_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.EC_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.ETA_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.ROI_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
              constants.SLOPE_RF: (
                  1,
                  input_data_samples._N_DRAWS,
                  input_data_samples._N_RF_CHANNELS + 1,
              ),
          },
          mismatched_coord_size=input_data_samples._N_RF_CHANNELS + 1,
          expected_coord_size=input_data_samples._N_RF_CHANNELS,
      ),
  )
  def test_validate_injected_inference_data_prior_incorrect_coordinates(
      self, coord, mismatched_priors, mismatched_coord_size, expected_coord_size
  ):
    """Checks prior validation fails with incorrect coordinates."""
    model_spec = spec.ModelSpec()
    input_data = self.short_input_data_with_media_and_rf
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    prior_samples = meridian.prior_sampler_callable._sample_prior(
        self._N_DRAWS, seed=0
    )
    prior_coords = meridian.create_inference_data_coords(1, self._N_DRAWS)
    prior_dims = meridian.create_inference_data_dims()

    prior_samples = dict(prior_samples)
    for param in mismatched_priors:
      prior_samples[param] = backend.zeros(mismatched_priors[param])
    prior_coords = dict(prior_coords)
    prior_coords[coord] = np.arange(mismatched_coord_size)

    inference_data = az.convert_to_inference_data(
        prior_samples,
        coords=prior_coords,
        dims=prior_dims,
        group=constants.PRIOR,
    )

    with self.assertRaisesRegex(
        ValueError,
        f"Injected inference data {constants.PRIOR} has incorrect coordinate"
        f" '{coord}': expected {expected_coord_size}, got"
        f" {mismatched_coord_size}",
    ):
      _ = model.Meridian(
          input_data=input_data,
          model_spec=model_spec,
          inference_data=inference_data,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="media_coefficient_prior",
          input_data_fixture_name="short_input_data_with_media_only",
          spec_updates={
              "media_prior_type": constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
          },
          expected_vars_and_shapes={
              constants.BETA_M: (
                  1,
                  model_test_data.WithInputDataSamples._N_DRAWS,
                  model_test_data.WithInputDataSamples._N_MEDIA_CHANNELS,
              )
          },
          unexpected_vars=[
              constants.ROI_M,
              constants.MROI_M,
              constants.CONTRIBUTION_M,
          ],
      ),
      dict(
          testcase_name="rf_coefficient_prior",
          input_data_fixture_name="short_input_data_with_rf_only",
          spec_updates={
              "rf_prior_type": constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
          },
          expected_vars_and_shapes={
              constants.BETA_RF: (
                  1,
                  model_test_data.WithInputDataSamples._N_DRAWS,
                  model_test_data.WithInputDataSamples._N_RF_CHANNELS,
              )
          },
          unexpected_vars=[
              constants.ROI_RF,
              constants.MROI_RF,
              constants.CONTRIBUTION_RF,
          ],
      ),
      dict(
          testcase_name="organic_media_coefficient_prior",
          input_data_fixture_name="short_input_data_non_media_and_organic",
          spec_updates={
              "organic_media_prior_type": (
                  constants.TREATMENT_PRIOR_TYPE_COEFFICIENT
              )
          },
          expected_vars_and_shapes={
              constants.BETA_OM: (
                  1,
                  model_test_data.WithInputDataSamples._N_DRAWS,
                  model_test_data.WithInputDataSamples._N_ORGANIC_MEDIA_CHANNELS,
              )
          },
          unexpected_vars=[constants.CONTRIBUTION_OM],
      ),
  )
  def test_sample_prior_with_coefficient_prior_type(
      self,
      input_data_fixture_name,
      spec_updates,
      expected_vars_and_shapes,
      unexpected_vars,
  ):
    """Checks that coefficient priors are sampled correctly."""
    model_spec = spec.ModelSpec(**spec_updates)

    input_data = getattr(self, input_data_fixture_name)
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)

    prior = meridian.inference_data.prior

    for var, shape in expected_vars_and_shapes.items():
      self.assertTrue(hasattr(prior, var))
      self.assertEqual(getattr(prior, var).shape, shape)

    for var in unexpected_vars:
      with self.assertRaises(AttributeError):
        getattr(prior, var)


class PriorDistributionSamplerInitTest(
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    self.meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=spec.ModelSpec(),
    )

  def test_init_with_meridian(self):
    sampler = prior_sampler.PriorDistributionSampler(self.meridian)
    self.assertIs(sampler._meridian, self.meridian)
    self.assertIs(sampler._model_context, self.meridian.model_context)
    self.assertIsInstance(sampler._model_equations, equations.ModelEquations)
    self.assertIs(
        sampler._model_equations._context, self.meridian.model_context
    )

  def test_init_with_model_context(self):
    sampler = prior_sampler.PriorDistributionSampler(
        model_context=self.meridian.model_context,
    )
    self.assertIsNone(sampler._meridian)
    self.assertIs(sampler._model_context, self.meridian.model_context)
    self.assertIsInstance(sampler._model_equations, equations.ModelEquations)
    self.assertIs(
        sampler._model_equations._context, self.meridian.model_context
    )

  def test_init_raises_error_if_meridian_and_context_are_none(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError,
        "Either `meridian` or `model_context` must be provided.",
    ):
      prior_sampler.PriorDistributionSampler()


if __name__ == "__main__":
  absltest.main()
