# Copyright 2026 The Meridian Authors.
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

from collections.abc import Mapping
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import arviz as az
from meridian import backend
from meridian import constants
from meridian.backend import test_utils
from meridian.model import changepoints
from meridian.model import context
from meridian.model import equations
from meridian.model import model
from meridian.model import model_test_data
from meridian.model import prior_distribution
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
        meridian.inference_data.prior, meridian2.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
        meridian.inference_data.prior, meridian2.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
            "__call__",
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
        else (1, self._N_DRAWS)
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

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
            "__call__",
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
        else (1, self._N_DRAWS)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
            "__call__",
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
        else (1, self._N_DRAWS)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_media_channel_shape = geo_shape + (self._N_MEDIA_CHANNELS,)

    media_parameters = list(constants.MEDIA_PARAMETER_NAMES)
    media_parameters.remove(constants.BETA_GM)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
        getattr(meridian.inference_data.prior, attr)  # pyrefly: ignore[missing-attribute]

  def test_sample_prior_rf_only_returns_correct_shape(self):
    self.enter_context(
        mock.patch.object(
            prior_sampler.PriorDistributionSampler,
            "__call__",
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
        else (1, self._N_DRAWS)
    )
    geo_shape = (1, self._N_DRAWS, self._N_GEOS)
    time_shape = (1, self._N_DRAWS, self._N_TIMES_SHORT)
    geo_control_shape = geo_shape + (self._N_CONTROLS,)
    geo_rf_channel_shape = geo_shape + (self._N_RF_CHANNELS,)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
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
            "__call__",
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
            "__call__",
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
            "__call__",
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
    prior_samples = meridian.prior_sampler_callable(
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

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]

    for var, shape in expected_vars_and_shapes.items():
      self.assertTrue(hasattr(prior, var))
      self.assertEqual(getattr(prior, var).shape, shape)

    for var in unexpected_vars:
      with self.assertRaises(AttributeError):
        getattr(prior, var)

  @parameterized.named_parameters(
      dict(testcase_name="batch_size_1", batch_size=1),
      dict(testcase_name="batch_size_2_uneven", batch_size=2),
      dict(testcase_name="batch_size_3_uneven", batch_size=3),
      dict(testcase_name="batch_size_equals_n_draws", batch_size=5),
  )
  def test_sample_prior_batch_size_matches_unbatched(self, batch_size: int):
    meridian_unbatched = model.Meridian(
        input_data=self.short_input_data_non_media_and_organic,
        model_spec=spec.ModelSpec(),
    )
    meridian_batched = model.Meridian(
        input_data=self.short_input_data_non_media_and_organic,
        model_spec=spec.ModelSpec(),
    )
    n_draws = 5
    meridian_unbatched.sample_prior(n_draws=n_draws, seed=42, batch_size=100)
    meridian_batched.sample_prior(
        n_draws=n_draws, seed=42, batch_size=batch_size
    )

    unbatched = meridian_unbatched.inference_data.prior  # pyrefly: ignore[missing-attribute]
    batched = meridian_batched.inference_data.prior  # pyrefly: ignore[missing-attribute]
    self.assertEqual(set(batched.data_vars), set(unbatched.data_vars))
    for key in unbatched.data_vars:
      self.assertEqual(batched[key].shape, unbatched[key].shape)
      np.testing.assert_allclose(
          np.asarray(batched[key]),
          np.asarray(unbatched[key]),
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"Mismatch in parameter {key} with batch_size={batch_size}",
      )

  def test_sample_prior_every_batch_has_the_same_size(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_non_media_and_organic,
        model_spec=spec.ModelSpec(),
    )
    batch_draw_counts = []

    def _record_batch_sizes(func):
      def _wrapped(*draw_args):
        batch_draw_counts.extend(arg.shape[1] for arg in draw_args)
        return func(*draw_args)

      return _wrapped

    with mock.patch.object(
        backend, "compile_temporary_function", side_effect=_record_batch_sizes
    ) as mock_compile:
      # 5 draws in batches of 2: the last batch covers draws 3-4, not just 4.
      meridian.sample_prior(n_draws=5, seed=42, batch_size=2)

    # One compiled function each for `beta_m`, `beta_rf`, `beta_om`,
    # `beta_orf`, and `gamma_n`.
    self.assertEqual(mock_compile.call_count, 5)
    self.assertNotEmpty(batch_draw_counts)
    self.assertEqual(set(batch_draw_counts), {2})

  def test_sample_prior_compiled_matches_uncompiled(self):
    meridian_uncompiled = model.Meridian(
        input_data=self.short_input_data_non_media_and_organic,
        model_spec=spec.ModelSpec(),
    )
    meridian_compiled = model.Meridian(
        input_data=self.short_input_data_non_media_and_organic,
        model_spec=spec.ModelSpec(),
    )
    with mock.patch.object(
        backend, "compile_temporary_function", side_effect=lambda func: func
    ):
      meridian_uncompiled.sample_prior(n_draws=5, seed=42, batch_size=2)
    meridian_compiled.sample_prior(n_draws=5, seed=42, batch_size=2)

    uncompiled = meridian_uncompiled.inference_data.prior  # pyrefly: ignore[missing-attribute]
    compiled = meridian_compiled.inference_data.prior  # pyrefly: ignore[missing-attribute]
    self.assertEqual(set(compiled.data_vars), set(uncompiled.data_vars))
    for key in uncompiled.data_vars:
      # Compilation can change the order of floating-point operations.
      np.testing.assert_allclose(
          np.asarray(compiled[key]),
          np.asarray(uncompiled[key]),
          rtol=1e-5,
          atol=1e-6,
          err_msg=f"Mismatch in parameter {key}",
      )

  @parameterized.named_parameters(
      dict(testcase_name="zero", batch_size=0),
      dict(testcase_name="negative", batch_size=-2),
  )
  def test_sample_prior_invalid_batch_size_raises_error(self, batch_size: int):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_only,
        model_spec=spec.ModelSpec(),
    )
    with self.assertRaisesRegex(
        ValueError, "`batch_size` must be at least 1"
    ):
      meridian.sample_prior(n_draws=5, seed=42, batch_size=batch_size)

  def _compute_aggregate_raw_baseline(
      self,
      meridian_instance: model.Meridian,
      prior_dataset: az.InferenceData,
  ) -> np.ndarray:
    """Computes aggregate raw baseline for each prior draw."""
    ctx = meridian_instance.model_context
    prior = prior_dataset.prior  # pyrefly: ignore[missing-attribute]
    mu_t = np.asarray(prior.mu_t.values[0])  # shape: (n_draws, n_times)

    if ctx.is_national:
      baseline_scaled = np.expand_dims(mu_t, 1)  # (n_draws, 1, n_times)
      if ctx.n_controls > 0 and ctx.controls_scaled is not None:
        gamma_c = np.asarray(prior.gamma_c.values[0])  # (n_draws, n_controls)
        z_controls = np.asarray(ctx.controls_scaled)  # (1, n_times, n_controls)
        baseline_scaled = baseline_scaled + np.einsum(
            "gtc,dc->dgt", z_controls, gamma_c
        )
    else:
      tau_g = np.asarray(prior.tau_g.values[0])  # (n_draws, n_geos)
      baseline_scaled = np.expand_dims(tau_g, -1) + np.expand_dims(mu_t, -2)
      if ctx.n_controls > 0 and ctx.controls_scaled is not None:
        gamma_gc = np.asarray(
            prior.gamma_gc.values[0]
        )  # (n_draws, n_geos, n_controls)
        z_controls = np.asarray(
            ctx.controls_scaled
        )  # (n_geos, n_times, n_controls)
        baseline_scaled = baseline_scaled + np.einsum(
            "gtc,dgc->dgt", z_controls, gamma_gc
        )

    # KpiTransformer.inverse convertsscaled baseline back to raw KPI
    baseline_raw = np.asarray(
        ctx.kpi_transformer.inverse(backend.to_tensor(baseline_scaled))
    )
    if ctx.revenue_per_kpi is not None:
      baseline_raw = baseline_raw * np.asarray(ctx.revenue_per_kpi)
    return np.sum(baseline_raw, axis=(-2, -1))

  def test_sample_prior_allows_negative_aggregate_baseline_false_strictly_positive(
      self,
  ):
    """Verifies positive aggregate baseline while allowing negative knots."""
    n_draws = 100
    model_spec = spec.ModelSpec(allows_negative_aggregate_baseline=False)
    input_data = self.short_input_data_with_media_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=n_draws, seed=42)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    total_raw_baseline = self._compute_aggregate_raw_baseline(
        meridian, meridian.inference_data
    )

    # 1. Total aggregate outcome baseline must be strictly positive across all
    # draws.
    self.assertTrue(np.all(total_raw_baseline > 0.0))

    # 2. Individual knots are not forced positive (can take negative values for
    # time effects).
    knot_values = prior.knot_values.values[0]
    self.assertTrue(np.any(knot_values < 0.0))

    # 3. Rotated parameters must not be present in prior InferenceData.
    self.assertFalse(hasattr(prior, constants.ROTATED_KNOT_0))
    self.assertFalse(hasattr(prior, constants.ROTATED_KNOT_REST))

  def test_sample_prior_allows_negative_aggregate_baseline_false_no_revenue_per_kpi(
      self,
  ):
    """Verifies positive aggregate KPI when revenue_per_kpi is None."""
    n_draws = 50
    model_spec = spec.ModelSpec(allows_negative_aggregate_baseline=False)
    input_data = self.input_data_non_revenue_no_revenue_per_kpi
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=n_draws, seed=42)

    total_raw_baseline = self._compute_aggregate_raw_baseline(
        meridian, meridian.inference_data
    )
    self.assertTrue(np.all(total_raw_baseline > 0.0))

  def test_sample_prior_allows_negative_aggregate_baseline_false_national_model(
      self,
  ):
    """Verifies positive aggregate baseline on a national model."""
    n_draws = 50
    model_spec = spec.ModelSpec(allows_negative_aggregate_baseline=False)
    input_data = self.national_input_data_media_only
    meridian = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )
    meridian.sample_prior(n_draws=n_draws, seed=42)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    total_raw_baseline = self._compute_aggregate_raw_baseline(
        meridian, meridian.inference_data
    )

    self.assertTrue(np.all(total_raw_baseline > 0.0))
    self.assertFalse(hasattr(prior, constants.ROTATED_KNOT_0))
    self.assertFalse(hasattr(prior, constants.ROTATED_KNOT_REST))

  def test_sample_prior_allows_negative_aggregate_baseline_false_population_scaled_non_media(
      self,
  ):
    """Verifies the truncation binds at zero with population-scaled non-media.

    Geo and control coefficients are fixed at zero, and `gamma_gn` is fixed so
    that the non-media treatments at their population-scaled baseline values
    offset the mean population-scaled KPI. The aggregate baseline is then
    proportional to the outcome-weighted sum of `mu_t`, which the constraint
    must truncate exactly at zero.
    """
    input_data = self.short_input_data_non_media_and_organic
    non_media_treatments = input_data.non_media_treatments
    revenue_per_kpi = input_data.revenue_per_kpi
    assert non_media_treatments is not None and revenue_per_kpi is not None
    population = np.asarray(input_data.population)[:, np.newaxis]
    kpi_scaled = np.asarray(input_data.kpi) / population
    non_media_scaled = np.asarray(non_media_treatments) / population[..., None]
    # Default (minimum) baseline value of each non-media channel, normalized.
    non_media_baseline = (
        non_media_scaled.min(axis=(0, 1)) - non_media_scaled.mean(axis=(0, 1))
    ) / non_media_scaled.std(axis=(0, 1))
    gamma_n = -kpi_scaled.mean() / kpi_scaled.std() / np.sum(non_media_baseline)
    zero = backend.np_float_dtype(0.0)
    model_spec = spec.ModelSpec(
        allows_negative_aggregate_baseline=False,
        population_scaled_non_media_channels=[
            str(channel)
            for channel in non_media_treatments[
                constants.NON_MEDIA_CHANNEL
            ].values
        ],
        non_media_treatments_prior_type=constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
        prior=prior_distribution.PriorDistribution(
            tau_g_excl_baseline=backend.tfd.Deterministic(zero),
            gamma_c=backend.tfd.Deterministic(zero),
            xi_c=backend.tfd.Deterministic(zero),
            gamma_n=backend.tfd.Deterministic(backend.np_float_dtype(gamma_n)),
            xi_n=backend.tfd.Deterministic(zero),
        ),
    )
    meridian = model.Meridian(input_data=input_data, model_spec=model_spec)
    meridian.sample_prior(n_draws=200, seed=42)

    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    mu_t = np.asarray(prior.mu_t.values[0])  # (n_draws, n_times)
    outcome_weights = population * np.asarray(revenue_per_kpi)
    aggregate_baseline = np.einsum("gt,dt->d", outcome_weights, mu_t)
    median = np.median(aggregate_baseline)
    self.assertGreaterEqual(np.min(aggregate_baseline), -1e-3 * median)
    self.assertLess(np.min(aggregate_baseline), 0.2 * median)


_ModelSpecWithChangepoints = spec._ModelSpecWithChangepoints  # pylint: disable=protected-access


class TimeVaryingPriorSamplingTest(
    test_utils.MeridianTestCase,
    parameterized.TestCase,
    model_test_data.WithInputDataSamples,
):
  """Tests prior sampling of models with changepoints."""

  # TODO: Add an end-to-end test that fits a model with
  # changepoints on simulated data with a known change in effect, and checks the
  # recovered per-interval ROI, once `Analyzer` supports changepoints.

  # pylint: disable=protected-access

  input_data_samples = model_test_data.WithInputDataSamples

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    model_test_data.WithInputDataSamples.setup()

  def setUp(self):
    super().setUp()
    data = self.short_input_data_with_media_and_rf
    assert data.media_channel is not None and data.rf_channel is not None
    self._media = [str(name) for name in data.media_channel.values]
    self._rf = [str(name) for name in data.rf_channel.values]
    dates = data.time_coordinates.all_dates
    self._changepoints = {
        self._media[1]: [dates[20]],
        self._rf[0]: [dates[10], dates[30]],
    }

  def _context(self, **spec_kwargs) -> context.ModelContext:
    return context.ModelContext(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=_ModelSpecWithChangepoints(
            changepoints=self._changepoints, **spec_kwargs
        ),
    )

  def test_stores_time_varying_parameters(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=_ModelSpecWithChangepoints(changepoints=self._changepoints),
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    self.assertEqual(
        prior[constants.ZETA_RF].dims,
        (constants.CHAIN, constants.DRAW, constants.CHANGEPOINT_RF_CHANNEL),
    )
    self.assertEqual(
        prior[constants.VARPHI_M].dims,
        (
            constants.CHAIN,
            constants.DRAW,
            constants.CHANGEPOINT_MEDIA_CHANNEL,
            constants.CHANGEPOINT_INTERVAL,
        ),
    )
    # The RF channel has 3 intervals, so both types share an axis of 3.
    self.assertEqual(prior[constants.VARPHI_M].shape, (1, self._N_DRAWS, 1, 3))
    self.assertEqual(
        list(prior[constants.CHANGEPOINT_MEDIA_CHANNEL].values),
        [self._media[1]],
    )

  def test_batched_prior_matches_unbatched(self):
    n_draws = 5
    priors = {}
    for batch_size in (100, 2):
      meridian = model.Meridian(
          input_data=self.short_input_data_with_media_and_rf,
          model_spec=_ModelSpecWithChangepoints(
              changepoints=self._changepoints
          ),
      )
      meridian.sample_prior(n_draws=n_draws, seed=42, batch_size=batch_size)
      priors[batch_size] = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    for key in (constants.BETA_M, constants.BETA_RF):
      test_utils.assert_allclose(
          np.asarray(priors[2][key]),
          np.asarray(priors[100][key]),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_stores_time_varying_parameters_of_media_only(self):
    meridian = model.Meridian(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=_ModelSpecWithChangepoints(
            changepoints={self._media[1]: self._changepoints[self._media[1]]}
        ),
    )
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    prior = meridian.inference_data.prior  # pyrefly: ignore[missing-attribute]
    self.assertEqual(prior[constants.VARPHI_M].shape, (1, self._N_DRAWS, 1, 2))
    self.assertNotIn(constants.ZETA_RF, prior)
    self.assertNotIn(constants.VARPHI_RF, prior)
    self.assertEqual(prior.sizes.get(constants.CHANGEPOINT_RF_CHANNEL, 0), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="other_changepoints",
          has_changepoints=True,
          coord=constants.CHANGEPOINT_RF_CHANNEL,
          expected=1,
          got=0,
      ),
      dict(
          testcase_name="no_changepoints",
          has_changepoints=False,
          coord=constants.CHANGEPOINT_MEDIA_CHANNEL,
          expected=0,
          got=1,
      ),
  )
  def test_injected_inference_data_must_match_changepoints(
      self, has_changepoints: bool, coord: str, expected: int, got: int
  ):
    data = self.short_input_data_with_media_and_rf
    # Only a media channel has changepoints, so the RF coordinate is empty.
    media_only_spec = _ModelSpecWithChangepoints(
        changepoints={self._media[1]: self._changepoints[self._media[1]]}
    )
    meridian = model.Meridian(input_data=data, model_spec=media_only_spec)
    meridian.sample_prior(n_draws=self._N_DRAWS, seed=1)
    inference_data = meridian.inference_data
    # The same changepoints accept it.
    model.Meridian(
        input_data=data,
        model_spec=media_only_spec,
        inference_data=inference_data,
    )

    with self.assertRaisesRegex(
        ValueError,
        f"Injected inference data {constants.PRIOR} has incorrect coordinate"
        f" '{coord}': expected {expected}, got {got}",
    ):
      model.Meridian(
          input_data=data,
          model_spec=(
              _ModelSpecWithChangepoints(changepoints=self._changepoints)
              if has_changepoints
              else spec.ModelSpec()
          ),
          inference_data=inference_data,
      )

  def test_model_without_changepoints_matches_model_spec(self):
    data = self.short_input_data_with_media_and_rf
    draws = prior_sampler.PriorDistributionSampler(
        model_context=context.ModelContext(
            input_data=data, model_spec=_ModelSpecWithChangepoints()
        )
    )(n_draws=self._N_DRAWS, seed=1)
    expected = prior_sampler.PriorDistributionSampler(
        model_context=context.ModelContext(
            input_data=data, model_spec=spec.ModelSpec()
        )
    )(n_draws=self._N_DRAWS, seed=1)
    self.assertEqual(set(draws), set(expected))
    for name, value in expected.items():
      test_utils.assert_allequal(draws[name], value, err_msg=name)

  @parameterized.product(
      prior_type=(
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ),
      changepoint_media_effects_dist=(
          constants.MEDIA_EFFECTS_LOG_NORMAL,
          constants.MEDIA_EFFECTS_NORMAL,
      ),
  )
  def test_prior_keeps_its_whole_history_meaning(
      self, prior_type: str, changepoint_media_effects_dist: str
  ):
    # The other channels use the other distribution.
    media_effects_dist = (
        constants.MEDIA_EFFECTS_NORMAL
        if changepoint_media_effects_dist == constants.MEDIA_EFFECTS_LOG_NORMAL
        else constants.MEDIA_EFFECTS_LOG_NORMAL
    )
    model_context = self._context(
        media_prior_type=prior_type,
        rf_prior_type=prior_type,
        changepoint_media_prior_type=prior_type,
        changepoint_rf_prior_type=prior_type,
        media_effects_dist=media_effects_dist,
        changepoint_media_effects_dist=changepoint_media_effects_dist,
    )
    draws = prior_sampler.PriorDistributionSampler(model_context=model_context)(
        n_draws=4, seed=1
    )
    for is_rf, time_varying in itertools.product((False, True), repeat=2):
      with self.subTest(is_rf=is_rf, time_varying=time_varying):
        self._assert_whole_history_meaning(
            model_context=model_context,
            draws=draws,
            prior_type=prior_type,
            media_effects_dist=(
                changepoint_media_effects_dist
                if time_varying
                else media_effects_dist
            ),
            is_rf=is_rf,
            time_varying=time_varying,
        )

  @parameterized.named_parameters(
      ("coefficient_and_roi", "coefficient", "roi"),
      ("roi_and_coefficient", "roi", "coefficient"),
      ("roi_and_mroi", "roi", "mroi"),
      ("contribution_and_roi", "contribution", "roi"),
  )
  def test_channels_with_changepoints_use_their_prior_type(
      self, prior_type: str, changepoint_prior_type: str
  ):
    model_context = self._context(
        media_prior_type=prior_type,
        rf_prior_type=prior_type,
        changepoint_media_prior_type=changepoint_prior_type,
        changepoint_rf_prior_type=changepoint_prior_type,
    )
    draws = prior_sampler.PriorDistributionSampler(model_context=model_context)(
        n_draws=4, seed=1
    )
    for is_rf, time_varying in itertools.product((False, True), repeat=2):
      channel_prior_type = (
          changepoint_prior_type if time_varying else prior_type
      )
      with self.subTest(is_rf=is_rf, time_varying=time_varying):
        if channel_prior_type != constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          self._assert_whole_history_meaning(
              model_context=model_context,
              draws=draws,
              prior_type=channel_prior_type,
              media_effects_dist=constants.MEDIA_EFFECTS_LOG_NORMAL,
              is_rf=is_rf,
              time_varying=time_varying,
          )
    # A treatment parameter is NaN for the channels that don't use it.
    for parameters, info in (
        (
            prior_sampler._MEDIA_TREATMENT_PARAMETERS,
            model_context._media_changepoint_info,
        ),
        (
            prior_sampler._RF_TREATMENT_PARAMETERS,
            model_context._rf_changepoint_info,
        ),
    ):
      assert info is not None
      for parameter_prior_type, name in parameters.items():
        if parameter_prior_type == constants.TREATMENT_PRIOR_TYPE_COEFFICIENT:
          continue
        with self.subTest(name):
          if parameter_prior_type not in (prior_type, changepoint_prior_type):
            self.assertNotIn(name, draws)
            continue
          uses_prior_type = info.is_changepoint_channel == (
              parameter_prior_type == changepoint_prior_type
          )
          np.testing.assert_array_equal(
              np.isnan(np.asarray(draws[name])),
              np.broadcast_to(~uses_prior_type, np.shape(draws[name])),
          )
    for beta in (constants.BETA_M, constants.BETA_RF):
      self.assertTrue(np.all(np.isfinite(np.asarray(draws[beta]))))

  @parameterized.named_parameters(("media", False), ("rf", True))
  def test_type_whose_channels_all_have_changepoints_uses_their_prior_type(
      self, is_rf: bool
  ):
    dates = self.short_input_data_with_media_and_rf.time_coordinates.all_dates
    channels = self._rf if is_rf else self._media
    # `rf_prior_type` or `media_prior_type` applies to no channel, since all
    # channels of the type have changepoints.
    model_context = context.ModelContext(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=_ModelSpecWithChangepoints(
            changepoints={channel: [dates[20]] for channel in channels},
            media_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
            rf_prior_type=constants.TREATMENT_PRIOR_TYPE_MROI,
        ),
    )
    draws = prior_sampler.PriorDistributionSampler(model_context=model_context)(
        n_draws=4, seed=1
    )
    if is_rf:
      mroi, roi = constants.MROI_RF, constants.ROI_RF
      counterfactual = (
          model_context.rf_tensors.prior_reach_scaled_counterfactual
      )
    else:
      mroi, roi = constants.MROI_M, constants.ROI_M
      counterfactual = (
          model_context.media_tensors.prior_media_scaled_counterfactual
      )
    self.assertNotIn(mroi, draws)
    self.assertTrue(np.all(np.isfinite(np.asarray(draws[roi]))))
    # ROI priors without calibration don't need the counterfactual.
    self.assertIsNone(counterfactual)

  def _assert_whole_history_meaning(
      self,
      *,
      model_context: context.ModelContext,
      draws: Mapping[str, backend.Tensor],
      prior_type: str,
      media_effects_dist: str,
      is_rf: bool,
      time_varying: bool,
  ):
    """Asserts that a channel's draws imply its drawn prior quantity."""
    eqn = equations.ModelEquations(model_context)
    if is_rf:
      changepoint_info = model_context._rf_changepoint_info
      tensors = model_context.rf_tensors
      assert tensors.reach_scaled is not None
      assert tensors.frequency is not None
      alpha, ec, slope = (
          draws[constants.ALPHA_RF],
          draws[constants.EC_RF],
          draws[constants.SLOPE_RF],
      )
      transformed = eqn.adstock_hill_rf(
          reach=tensors.reach_scaled,
          frequency=tensors.frequency,
          alpha=alpha,
          ec=ec,
          slope=slope,
          decay_functions=model_context.adstock_decay_spec.rf,
          saturation_spec=model_context.saturation_spec.rf,
      )
      counterfactual_difference = (
          eqn.linear_predictor_counterfactual_difference_rf(
              rf_transformed=transformed,
              alpha_rf=alpha,
              ec_rf=ec,
              slope_rf=slope,
          )
      )
      beta_gx, zeta, varphi = (
          draws[constants.BETA_GRF],
          draws[constants.ZETA_RF],
          draws[constants.VARPHI_RF],
      )
      treatment_parameter = {
          constants.TREATMENT_PRIOR_TYPE_ROI: constants.ROI_RF,
          constants.TREATMENT_PRIOR_TYPE_MROI: constants.MROI_RF,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION: (
              constants.CONTRIBUTION_RF
          ),
      }[prior_type]
    else:
      changepoint_info = model_context._media_changepoint_info
      tensors = model_context.media_tensors
      assert tensors.media_scaled is not None
      alpha, ec, slope = (
          draws[constants.ALPHA_M],
          draws[constants.EC_M],
          draws[constants.SLOPE_M],
      )
      transformed = eqn.adstock_hill_media(
          media=tensors.media_scaled,
          alpha=alpha,
          ec=ec,
          slope=slope,
          decay_functions=model_context.adstock_decay_spec.media,
          saturation_spec=model_context.saturation_spec.media,
      )
      counterfactual_difference = (
          eqn.linear_predictor_counterfactual_difference_media(
              media_transformed=transformed,
              alpha_m=alpha,
              ec_m=ec,
              slope_m=slope,
          )
      )
      beta_gx, zeta, varphi = (
          draws[constants.BETA_GM],
          draws[constants.ZETA_M],
          draws[constants.VARPHI_M],
      )
      treatment_parameter = {
          constants.TREATMENT_PRIOR_TYPE_ROI: constants.ROI_M,
          constants.TREATMENT_PRIOR_TYPE_MROI: constants.MROI_M,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION: constants.CONTRIBUTION_M,
      }[prior_type]
    assert changepoint_info is not None
    beta_gx = np.asarray(beta_gx)
    if time_varying:
      channel = int(changepoint_info.channel_indices[0])
      # The incremental outcome over all geos and time periods, using each
      # time period's coefficient, matches the prior draw.
      coefficients = changepoints.time_varying_coefficients(
          beta_gx=backend.to_tensor(
              beta_gx[..., channel : channel + 1], dtype=backend.float_dtype
          ),
          zeta=zeta,
          phi=changepoints.expand_intervals(varphi, changepoint_info.weights),
          media_effects_dist=media_effects_dist,
      )
    else:
      channel = min(
          set(range(beta_gx.shape[-1]))
          - set(changepoint_info.channel_indices.tolist())
      )
      coefficients = np.broadcast_to(
          beta_gx[..., np.newaxis, channel : channel + 1],
          beta_gx.shape[:-1] + (self._N_TIMES_SHORT, 1),
      )
    revenue_per_kpi = (
        np.ones((self._N_GEOS, self._N_TIMES_SHORT))
        if model_context.revenue_per_kpi is None
        else np.asarray(model_context.revenue_per_kpi)
    )
    incremental_outcome = np.einsum(
        "...gtx,gt,g,...gtx->...x",
        np.asarray(counterfactual_difference)[..., channel : channel + 1],
        revenue_per_kpi,
        np.asarray(model_context.population),
        np.asarray(coefficients),
    ) * float(model_context.kpi_transformer.population_scaled_stdev)
    expected = (
        np.asarray(draws[treatment_parameter])[..., channel : channel + 1]
        * np.asarray(tensors.prior_denominator)[channel]
    )
    test_utils.assert_allclose(incremental_outcome, expected, rtol=1e-3)

  @parameterized.product(
      data_name=(
          "short_input_data_with_media_only",
          "short_input_data_with_rf_only",
          "national_input_data_media_and_rf",
      ),
      prior_type=(
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_COEFFICIENT,
      ),
  )
  def test_samples_each_channel_type(self, data_name: str, prior_type: str):
    data = getattr(self, data_name)
    dates = data.time_coordinates.all_dates
    channels = {
        constants.MEDIA_CHANNEL: data.media_channel,
        constants.RF_CHANNEL: data.rf_channel,
    }
    changepoint_channels = {
        channel_type: str(names.values[0])
        for channel_type, names in channels.items()
        if names is not None
    }
    model_context = context.ModelContext(
        input_data=data,
        model_spec=_ModelSpecWithChangepoints(
            changepoints={
                channel: [dates[len(dates) // 2]]
                for channel in changepoint_channels.values()
            },
            media_prior_type=prior_type,
            rf_prior_type=prior_type,
            changepoint_media_prior_type=prior_type,
            changepoint_rf_prior_type=prior_type,
        ),
    )
    draws = prior_sampler.PriorDistributionSampler(model_context=model_context)(
        n_draws=self._N_DRAWS, seed=1
    )
    for channel_type, zeta, varphi, beta in (
        (
            constants.MEDIA_CHANNEL,
            constants.ZETA_M,
            constants.VARPHI_M,
            constants.BETA_GM,
        ),
        (
            constants.RF_CHANNEL,
            constants.ZETA_RF,
            constants.VARPHI_RF,
            constants.BETA_GRF,
        ),
    ):
      if channel_type not in changepoint_channels:
        self.assertNotIn(zeta, draws)
        self.assertNotIn(varphi, draws)
        continue
      self.assertEqual(draws[zeta].shape, (1, self._N_DRAWS, 1))
      self.assertEqual(draws[varphi].shape, (1, self._N_DRAWS, 1, 2))
      self.assertTrue(np.all(np.isfinite(np.asarray(draws[beta]))))

  def test_custom_zeta_prior_per_channel(self):
    model_context = context.ModelContext(
        input_data=self.short_input_data_with_media_and_rf,
        model_spec=_ModelSpecWithChangepoints(
            changepoints={
                self._media[0]: self._changepoints[self._media[1]],
                self._media[1]: self._changepoints[self._media[1]],
            },
            prior=prior_distribution._PriorDistributionWithChangepoints(
                zeta_m=backend.tfd.HalfNormal(
                    backend.to_tensor([0.01, 10.0], dtype=backend.float_dtype),
                    name=constants.ZETA_M,
                ),
            ),
        ),
    )
    draws = prior_sampler.PriorDistributionSampler(model_context=model_context)(
        n_draws=200, seed=1
    )
    zeta_m = np.asarray(draws[constants.ZETA_M])
    self.assertEqual(zeta_m.shape, (1, 200, 2))
    self.assertLess(np.max(zeta_m[..., 0]), 0.1)
    self.assertGreater(np.mean(zeta_m[..., 1]), 1.0)
    self.assertNotIn(constants.ZETA_RF, draws)


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
