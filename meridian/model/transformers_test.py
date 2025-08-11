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

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian.backend import test_utils as backend_test_utils
from meridian.model import transformers
import numpy as np

tfd = backend.tfd


class MediaTransformerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 4
    self._n_media_times = 10
    self._n_media_channels = 3

    # Generate random data based on dimensions specified above.
    backend.set_random_seed(1)
    self._media1 = tfd.HalfNormal(1).sample(
        [self._n_geos, self._n_media_times, self._n_media_channels]
    )
    self._media2 = tfd.HalfNormal(1).sample(
        [self._n_geos, self._n_media_times, self._n_media_channels]
    )
    self._population = tfd.Uniform(100, 1000).sample([self._n_geos])

  def test_output_shape_and_range(self):
    transformer = transformers.MediaTransformer(
        media=self._media1, population=self._population
    )
    transformed_media = transformer.forward(self._media2)
    self.assertEqual(
        transformed_media.shape,
        self._media2.shape,
        msg="Shape of `media` not preserved by `MediaTransform.forward()`.",
    )
    backend_test_utils.assert_all_finite(
        transformed_media, err_msg="Infinite values found in transformed media."
    )
    backend_test_utils.assert_all_non_negative(
        transformed_media, err_msg="Negative values found in transformed media."
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.MediaTransformer(
        media=self._media1, population=self._population
    )
    transformed_media = transformer.inverse(transformer.forward(self._media2))
    backend_test_utils.assert_allclose(
        transformed_media,
        self._media2,
        err_msg="`inverse(forward(media))` not equal to `media`.",
    )

  def test_median_of_transformed_media_is_one(self):
    transformer = transformers.MediaTransformer(
        media=self._media1, population=self._population
    )
    transformed_media = transformer.forward(self._media1)
    median = np.nanmedian(
        backend.where(transformed_media == 0, np.nan, transformed_media),
        axis=[0, 1],
    )
    backend_test_utils.assert_allclose(median, np.ones(self._n_media_channels))

  @parameterized.named_parameters(
      dict(testcase_name="all_zeros", channel_fill_value=0.0),
      dict(testcase_name="all_nans", channel_fill_value=np.nan),
  )
  def test_media_with_invalid_channel_raises_error(self, channel_fill_value):
    media_with_invalid = self._media1.numpy()
    media_with_invalid[..., 0] = channel_fill_value
    media_with_invalid = backend.to_tensor(media_with_invalid)
    with self.assertRaisesRegex(
        ValueError,
        "MediaTransformer has a NaN population-scaled non-zero median",
    ):
      transformers.MediaTransformer(
          media=media_with_invalid, population=self._population
      )

  def test_media_with_mixed_zeros_and_nans_raises_error(self):
    media_with_mixed = self._media1.numpy()
    # Set first half of times to 0 and second half to NaN for the first channel.
    media_with_mixed[:, : self._n_media_times // 2, 0] = 0.0
    media_with_mixed[:, self._n_media_times // 2 :, 0] = np.nan
    media_with_mixed = backend.to_tensor(media_with_mixed)
    with self.assertRaisesRegex(
        ValueError,
        "MediaTransformer has a NaN population-scaled non-zero median",
    ):
      transformers.MediaTransformer(
          media=media_with_mixed, population=self._population
      )


class CenteringAndScalingTransformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 10
    self._n_times = 1
    self._n_controls = 5

    # Generate random data based on dimensions specified above.
    backend.set_random_seed(1)
    self._controls1 = tfd.Normal(2, 1).sample(
        [self._n_geos, self._n_times, self._n_controls]
    )
    self._controls2 = tfd.Normal(2, 1).sample(
        [self._n_geos, self._n_times, self._n_controls]
    )
    self._controls3 = backend.to_tensor(
        np.ones((self._n_geos, 1, self._n_controls))
    )

    # Generate populations to test population scaling.
    self._population = tfd.Uniform().sample(self._n_geos)
    self._population_scaling_id = backend.to_tensor(
        [False, True, False, True, True]
    )
    self._controls4 = backend.broadcast_to(
        self._population[:, None, None],
        (self._n_geos, self._n_times, self._n_controls),
    )
    self._transformer = transformers.CenteringAndScalingTransformer(
        self._controls4, self._population, self._population_scaling_id
    )
    self._controls_transformed = self._transformer.forward(self._controls4)

  def test_output_shape_and_range(self):
    transformer = transformers.CenteringAndScalingTransformer(
        tensor=self._controls1, population=self._population
    )

    transformed_controls = transformer.forward(self._controls2)
    self.assertEqual(
        transformed_controls.shape,
        self._controls2.shape,
        msg=(
            "Shape of `controls` not preserved by"
            " `ControlsTransform.forward()`."
        ),
    )
    backend_test_utils.assert_all_finite(
        transformed_controls,
        err_msg="Infinite values found in transformed controls.",
    )

  def test_forward_no_variation(self):
    transformer = transformers.CenteringAndScalingTransformer(
        tensor=self._controls3, population=self._population
    )
    transformed_controls = transformer.forward(self._controls3)
    backend_test_utils.assert_allclose(
        transformed_controls,
        np.zeros_like(np.array(transformed_controls)),
        err_msg="`forward(controls)` not equal to `[[[0, 0, ..., 0]]]`.",
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.CenteringAndScalingTransformer(
        tensor=self._controls1, population=self._population
    )
    transformed_controls = transformer.inverse(
        transformer.forward(self._controls2)
    )
    backend_test_utils.assert_allclose(
        transformed_controls,
        self._controls2,
        err_msg="`inverse(forward(controls))` not equal to `controls`.",
    )

  def test_default_population_args(self):
    default_transformer = transformers.CenteringAndScalingTransformer(
        self._controls4, self._population
    )
    self.assertIsNone(default_transformer._population_scaling_factors)

  def test_inverse_population_scaled(self):
    backend_test_utils.assert_allclose(
        self._transformer.inverse(self._controls_transformed), self._controls4
    )

  def test_output_population_scaled(self):
    for c in [1, 3, 4]:
      population_scaled_control = (
          self._controls4[..., c] / self._population[:, None]
      )
      means = backend.reduce_mean(population_scaled_control, axis=(0, 1))
      stdevs = backend.reduce_std(population_scaled_control, axis=(0, 1))
      backend_test_utils.assert_allclose(
          self._population[:, None]
          * (self._controls_transformed[:, :, c] * stdevs + means),
          self._controls4[:, :, c],
      )

  def test_output_population_scaled_no_population_scaling(self):
    for c in [1, 3, 4]:
      population_scaled_controls = (
          self._controls4[..., c] / self._population[:, None]
      )
      # Reshape is needed because the two `forward` calls produce broadcast-
      # compatible but differently shaped arrays, and `assert_allclose`
      # requires an exact shape match.
      actual_result = self._transformer.forward(
          backend.to_tensor(
              np.reshape(np.array(population_scaled_controls), (-1, 1, 1))
          ),
          apply_population_scaling=False,
      )[..., c]
      expected_result = self._transformer.forward(
          self._controls4, apply_population_scaling=True
      )[..., c]
      backend_test_utils.assert_allclose(actual_result, expected_result)


class KpiTransformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 5
    self._n_times = 20

    # Generate random data based on dimensions specified above.
    backend.set_random_seed(1)
    self._kpi1 = tfd.HalfNormal(10).sample([self._n_geos, self._n_times])
    self._kpi2 = tfd.HalfNormal(10).sample([self._n_geos, self._n_times])
    self._population = tfd.Uniform(100, 1000).sample([self._n_geos])

  def test_population_scaled_mean(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    backend_test_utils.assert_allclose(
        transformer.population_scaled_mean,
        backend.reduce_mean(self._kpi1 / self._population[:, backend.newaxis]),
    )

  def test_population_scaled_stdev(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    backend_test_utils.assert_allclose(
        transformer.population_scaled_stdev,
        backend.reduce_std(self._kpi1 / self._population[:, backend.newaxis]),
    )

  def test_output_shape_and_range(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    transformed_kpi = transformer.forward(self._kpi2)
    self.assertEqual(
        transformed_kpi.shape,
        self._kpi2.shape,
        msg="Shape of `kpi` not preserved by `KpiTransform.forward()`.",
    )
    backend_test_utils.assert_all_finite(
        transformed_kpi, err_msg="Infinite values found in transformed kpi."
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    transformed_kpi = transformer.inverse(transformer.forward(self._kpi2))
    backend_test_utils.assert_allclose(
        transformed_kpi,
        self._kpi2,
        err_msg="`inverse(forward(kpi))` not equal to `kpi`.",
    )


if __name__ == "__main__":
  absltest.main()
