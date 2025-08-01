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
from meridian.model import transformers
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MediaTransformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 4
    self._n_media_times = 10
    self._n_media_channels = 3

    # Generate random data based on dimensions specified above.
    tf.random.set_seed(1)
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
    tf.debugging.assert_equal(
        transformed_media.shape,
        self._media2.shape,
        message="Shape of `media` not preserved by `MediaTransform.forward()`.",
    )
    tf.debugging.assert_all_finite(
        transformed_media, message="Infinite values found in transformed media."
    )
    tf.debugging.assert_non_negative(
        transformed_media, message="Negative values found in transformed media."
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.MediaTransformer(
        media=self._media1, population=self._population
    )
    transformed_media = transformer.inverse(transformer.forward(self._media2))
    tf.debugging.assert_near(
        transformed_media,
        self._media2,
        message="`inverse(forward(media))` not equal to `media`.",
    )

  def test_median_of_transformed_media_is_one(self):
    transformer = transformers.MediaTransformer(
        media=self._media1, population=self._population
    )
    transformed_media = transformer.forward(self._media1)
    median = np.nanmedian(
        tf.where(transformed_media == 0, np.nan, transformed_media), axis=[0, 1]
    )
    tf.debugging.assert_near(median, np.ones(self._n_media_channels))

  def test_all_zeros_media_raises_error(self):
    media_with_zeros = self._media1.numpy()
    media_with_zeros[..., 0] = 0
    media_with_zeros = tf.convert_to_tensor(media_with_zeros)
    with self.assertRaisesRegex(
        ValueError,
        "MediaTransformer has a NaN population-scaled non-zero median",
    ):
      transformers.MediaTransformer(
          media=media_with_zeros, population=self._population
      )

  def test_all_nan_media_raises_error(self):
    media_with_nan = self._media1.numpy()
    media_with_nan[..., 0] = np.nan
    media_with_nan = tf.convert_to_tensor(media_with_nan)
    with self.assertRaisesRegex(
        ValueError,
        "MediaTransformer has a NaN population-scaled non-zero median",
    ):
      transformers.MediaTransformer(
          media=media_with_nan, population=self._population
      )


class CenteringAndScalingTransformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 10
    self._n_times = 1
    self._n_controls = 5

    # Generate random data based on dimensions specified above.
    tf.random.set_seed(1)
    self._controls1 = tfd.Normal(2, 1).sample(
        [self._n_geos, self._n_times, self._n_controls]
    )
    self._controls2 = tfd.Normal(2, 1).sample(
        [self._n_geos, self._n_times, self._n_controls]
    )
    self._controls3 = np.ones((self._n_geos, 1, self._n_controls))

    # Generate populations to test population scaling.
    self._population = tfp.distributions.Uniform().sample(self._n_geos)
    self._population_scaling_id = tf.convert_to_tensor(
        [False, True, False, True, True]
    )
    self._controls4 = tf.tile(
        self._population[:, None, None], (1, self._n_times, self._n_controls)
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
    tf.debugging.assert_equal(
        transformed_controls.shape,
        self._controls2.shape,
        message=(
            "Shape of `controls` not preserved by"
            " `ControlsTransform.forward()`."
        ),
    )
    tf.debugging.assert_all_finite(
        transformed_controls,
        message="Infinite values found in transformed controls.",
    )

  def test_forward_no_variation(self):
    transformer = transformers.CenteringAndScalingTransformer(
        tensor=self._controls3, population=self._population
    )
    transformed_controls = transformer.forward(self._controls3)
    tf.debugging.assert_near(
        transformed_controls,
        np.zeros_like(transformed_controls),
        message="`forward(controls)` not equal to `[[[0, 0, ..., 0]]]`.",
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.CenteringAndScalingTransformer(
        tensor=self._controls1, population=self._population
    )
    transformed_controls = transformer.inverse(
        transformer.forward(self._controls2)
    )
    tf.debugging.assert_near(
        transformed_controls,
        self._controls2,
        message="`inverse(forward(controls))` not equal to `controls`.",
    )

  def test_default_population_args(self):
    default_transformer = transformers.CenteringAndScalingTransformer(
        self._controls4, self._population
    )
    self.assertIsNone(default_transformer._population_scaling_factors)

  def test_inverse_population_scaled(self):
    tf.debugging.assert_near(
        self._transformer.inverse(self._controls_transformed), self._controls4
    )

  def test_output_population_scaled(self):
    for c in [1, 3, 4]:
      population_scaled_control = (
          self._controls4[..., c] / self._population[:, None]
      )
      means = tf.reduce_mean(population_scaled_control, axis=(0, 1))
      stdevs = tf.math.reduce_std(population_scaled_control, axis=(0, 1))
      tf.debugging.assert_near(
          self._population[:, None]
          * (self._controls_transformed[:, :, c] * stdevs + means),
          self._controls4[:, :, c],
      )

  def test_output_population_scaled_no_population_scaling(self):
    for c in [1, 3, 4]:
      population_scaled_controls = (
          self._controls4[..., c] / self._population[:, None]
      )
      tf.debugging.assert_near(
          self._transformer.forward(
              population_scaled_controls, apply_population_scaling=False
          )[..., c],
          self._transformer.forward(
              self._controls4, apply_population_scaling=True
          )[..., c],
      )


class KpiTransformerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Data dimensions for default parameter values.
    self._n_geos = 5
    self._n_times = 20

    # Generate random data based on dimensions specified above.
    tf.random.set_seed(1)
    self._kpi1 = tfd.HalfNormal(10).sample([self._n_geos, self._n_times])
    self._kpi2 = tfd.HalfNormal(10).sample([self._n_geos, self._n_times])
    self._population = tfd.Uniform(100, 1000).sample([self._n_geos])

  def test_population_scaled_mean(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    tf.debugging.assert_near(
        transformer.population_scaled_mean,
        tf.reduce_mean(self._kpi1 / self._population[:, tf.newaxis]),
    )

  def test_population_scaled_stdev(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    tf.debugging.assert_near(
        transformer.population_scaled_stdev,
        tf.math.reduce_std(self._kpi1 / self._population[:, tf.newaxis]),
    )

  def test_output_shape_and_range(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    transformed_kpi = transformer.forward(self._kpi2)
    tf.debugging.assert_equal(
        transformed_kpi.shape,
        self._kpi2.shape,
        message="Shape of `kpi` not preserved by `KpiTransform.forward()`.",
    )
    tf.debugging.assert_all_finite(
        transformed_kpi,
        message="Infinite values found in transformed kpi.",
    )

  def test_forward_inverse_is_identity(self):
    transformer = transformers.KpiTransformer(
        kpi=self._kpi1, population=self._population
    )
    transformed_kpi = transformer.inverse(transformer.forward(self._kpi2))
    tf.debugging.assert_near(
        transformed_kpi,
        self._kpi2,
        message="`inverse(forward(kpi))` not equal to `kpi`.",
    )


if __name__ == "__main__":
  absltest.main()
