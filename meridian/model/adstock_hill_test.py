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

"""Unit tests for Adstock and Hill functions."""

from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.model import adstock_hill
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

_DECAY_FUNCTIONS = [
    dict(
        testcase_name=constants.GEOMETRIC_DECAY,
        decay_function=constants.GEOMETRIC_DECAY,
    ),
    dict(
        testcase_name=constants.BINOMIAL_DECAY,
        decay_function=constants.BINOMIAL_DECAY,
    ),
]

_DECAY_WEIGHTS = [
    # (function, alpha, expected_weights)
    (
        constants.GEOMETRIC_DECAY,
        0.0,
        (0.0, 0.0, 0.0, 0.0, 1.0),
    ),
    (constants.GEOMETRIC_DECAY, 0.5, (0.5**4, 0.5**3, 0.5**2, 0.5**1, 0.5**0)),
]

_BINOMIAL_0_0_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 1.0)
_BINOMIAL_0_25_WEIGHTS = (0.008, 0.064, 0.216, 0.512, 1.0)
_BINOMIAL_0_5_WEIGHTS = (0.2, 0.4, 0.6, 0.8, 1.0)
_BINOMIAL_0_6666_WEIGHTS = (
    np.sqrt(0.2),
    np.sqrt(0.4),
    np.sqrt(0.6),
    np.sqrt(0.8),
    1.0,
)
_BINOMIAL_1_0_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)

_GEOMETRIC_0_0_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 1.0)
_GEOMETRIC_0_5_WEIGHTS = (0.5**4, 0.5**3, 0.5**2, 0.5**1, 1.0)
_GEOMETRIC_1_0_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)

_MAX_LAG = 4


class TestAdstockDecayFunction(parameterized.TestCase):
  """Tests for adstock_hill.AdstockDecayFunction."""

  @parameterized.named_parameters(
      dict(
          testcase_name="geometric",
          decay_function=constants.GEOMETRIC_DECAY,
      ),
      dict(
          testcase_name="binomial",
          decay_function=constants.BINOMIAL_DECAY,
      ),
  )
  def test_from_parameterization(self, decay_function):
    adstock_decay_function = (
        adstock_hill.AdstockDecayFunction.from_parameterization(
            decay_function
        )
    )

    self.assertEqual(adstock_decay_function.media, decay_function)
    self.assertEqual(adstock_decay_function.rf, decay_function)
    self.assertEqual(adstock_decay_function.organic_media, decay_function)
    self.assertEqual(adstock_decay_function.organic_rf, decay_function)

  @parameterized.named_parameters(
      dict(
          testcase_name="all_defined_all_scalar",
          decay_function={
              constants.MEDIA: constants.GEOMETRIC_DECAY,
              constants.RF: constants.GEOMETRIC_DECAY,
              constants.ORGANIC_MEDIA: constants.BINOMIAL_DECAY,
              constants.ORGANIC_RF: constants.BINOMIAL_DECAY,
          },
      ),
      dict(
          testcase_name="all_defined_all_vector",
          decay_function={
              constants.MEDIA: [
                  constants.GEOMETRIC_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              constants.RF: [
                  constants.GEOMETRIC_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              constants.ORGANIC_MEDIA: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              constants.ORGANIC_RF: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              },
      ),
      dict(
          testcase_name="all_defined_mixed_scalar_vector",
          decay_function={
              constants.MEDIA: constants.GEOMETRIC_DECAY,
              constants.RF: constants.BINOMIAL_DECAY,
              constants.ORGANIC_MEDIA: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              constants.ORGANIC_RF: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              },
      ),
      dict(
          testcase_name="some_missing_all_scalar",
          decay_function={
              constants.MEDIA: constants.GEOMETRIC_DECAY,
              constants.RF: constants.BINOMIAL_DECAY,
              },
      ),
      dict(
          testcase_name="some_missing_all_vector",
          decay_function={
              constants.ORGANIC_MEDIA: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              constants.ORGANIC_RF: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              },
      ),
      dict(
          testcase_name="some_missing_mixed_scalar_vector",
          decay_function={
              constants.MEDIA: constants.GEOMETRIC_DECAY,
              constants.ORGANIC_RF: [
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY,
                  constants.GEOMETRIC_DECAY,
                  constants.BINOMIAL_DECAY
              ],
              },
      ),
  )
  def test_from_channels(self, decay_function):
    adstock_decay_function = adstock_hill.AdstockDecayFunction.from_channels(
        **decay_function
    )

    self.assertEqual(
        adstock_decay_function.media,
        decay_function.get(constants.MEDIA, constants.GEOMETRIC_DECAY),
    )
    self.assertEqual(
        adstock_decay_function.rf,
        decay_function.get(constants.RF, constants.GEOMETRIC_DECAY),
    )
    self.assertEqual(
        adstock_decay_function.organic_media,
        decay_function.get(constants.ORGANIC_MEDIA, constants.GEOMETRIC_DECAY),
    )
    self.assertEqual(
        adstock_decay_function.organic_rf,
        decay_function.get(constants.ORGANIC_RF, constants.GEOMETRIC_DECAY),
    )


class TestComputeDecayWeights(parameterized.TestCase):
  """Tests for adstock_hill.compute_decay_weights()."""

  @parameterized.named_parameters(
      dict(
          testcase_name="geometric_0.0",
          alpha=0.0,
          decay_function=constants.GEOMETRIC_DECAY,
          expected_weights=_GEOMETRIC_0_0_WEIGHTS
          ),
      dict(
          testcase_name="geometric_0.5",
          alpha=0.5,
          decay_function=constants.GEOMETRIC_DECAY,
          expected_weights=_GEOMETRIC_0_5_WEIGHTS
          ),
      dict(
          testcase_name="geometric_1.0",
          alpha=1.0,
          decay_function=constants.GEOMETRIC_DECAY,
          expected_weights=_GEOMETRIC_1_0_WEIGHTS
          ),
      dict(
          testcase_name="binomial_0.0",
          alpha=0.0,
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=_BINOMIAL_0_0_WEIGHTS
          ),
      dict(
          testcase_name="binomial_0.25",
          alpha=0.25,
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=_BINOMIAL_0_25_WEIGHTS
          ),
      dict(
          testcase_name="binomial_0.5",
          alpha=0.5,
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=_BINOMIAL_0_5_WEIGHTS
          ),
      dict(
          testcase_name="binomial_0.6666",
          alpha=2.0/3.0,
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=_BINOMIAL_0_6666_WEIGHTS
          ),
      dict(
          testcase_name="binomial_1.0",
          alpha=1.0,
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=_BINOMIAL_1_0_WEIGHTS
          ),
  )
  def test_compute_decay_weights_single_channel(
      self,
      alpha,
      decay_function,
      expected_weights
      ):

    l_range = tf.range(_MAX_LAG, -1, -1, dtype=tf.float32)

    with self.subTest("unnormalized"):
      weights = adstock_hill.compute_decay_weights(
          alpha,
          l_range,
          _MAX_LAG+1,
          decay_function,
          normalize=False
      )

      tf.debugging.assert_near(weights, expected_weights, rtol=1e-5)

    with self.subTest("normalized"):
      weights = adstock_hill.compute_decay_weights(
          alpha,
          l_range,
          _MAX_LAG+1,
          decay_function,
          normalize=True
      )
      tf.debugging.assert_near(
          tf.math.reduce_sum(weights),
          1.0,
          rtol=1e-5
          )
      tf.debugging.assert_near(
          weights / tf.math.reduce_max(weights),
          expected_weights,
          rtol=1e-5
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="all_geometric",
          alpha=(0.0, 0.5, 1.0),
          decay_function=constants.GEOMETRIC_DECAY,
          expected_weights=(
              _GEOMETRIC_0_0_WEIGHTS,
              _GEOMETRIC_0_5_WEIGHTS,
              _GEOMETRIC_1_0_WEIGHTS)
          ),
      dict(
          testcase_name="all_binomial",
          alpha=(0.0, 0.25, 0.5, 2.0/3.0, 1.0),
          decay_function=constants.BINOMIAL_DECAY,
          expected_weights=(
              _BINOMIAL_0_0_WEIGHTS,
              _BINOMIAL_0_25_WEIGHTS,
              _BINOMIAL_0_5_WEIGHTS,
              _BINOMIAL_0_6666_WEIGHTS,
              _BINOMIAL_1_0_WEIGHTS)
          ),
      dict(
          testcase_name="mixed_binomial_geometric",
          alpha=(0.0, 0.25, 0.5, 2.0/3.0, 1.0),
          decay_function=(
              constants.GEOMETRIC_DECAY,
              constants.BINOMIAL_DECAY,
              constants.GEOMETRIC_DECAY,
              constants.BINOMIAL_DECAY,
              constants.GEOMETRIC_DECAY,
              ),
          expected_weights=(
              _GEOMETRIC_0_0_WEIGHTS,
              _BINOMIAL_0_25_WEIGHTS,
              _GEOMETRIC_0_5_WEIGHTS,
              _BINOMIAL_0_6666_WEIGHTS,
              _GEOMETRIC_1_0_WEIGHTS)
          ),
  )
  def test_compute_decay_weights_multiple_channels(
      self,
      alpha,
      decay_function,
      expected_weights
      ):

    l_range = tf.range(_MAX_LAG, -1, -1, dtype=tf.float32)

    with self.subTest("unnormalized"):
      weights = adstock_hill.compute_decay_weights(
          alpha,
          l_range,
          _MAX_LAG+1,
          decay_function,
          normalize=False
      )

      tf.debugging.assert_near(weights, expected_weights, rtol=1e-5)

    with self.subTest("normalized"):
      weights = adstock_hill.compute_decay_weights(
          alpha,
          l_range,
          _MAX_LAG+1,
          decay_function,
          normalize=True
      )

      tf.debugging.assert_near(
          tf.math.reduce_sum(weights, axis=1),
          [1.0]*len(alpha),
          rtol=1e-5
          )
      tf.debugging.assert_near(
          weights / tf.math.reduce_max(weights, axis=1, keepdims=True),
          expected_weights,
          rtol=1e-5
      )

  def test_incompatible_alpha_decay_function_raises_error(self):
    alpha = tf.convert_to_tensor([0.5, 0.5])
    decay_function = [constants.GEOMETRIC_DECAY] * 3
    l_range = tf.range(_MAX_LAG, -1, -1, dtype=tf.float32)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The shape of alpha ((2,)) is incompatible with the length of "
        "decay_parameterization (3)"
    ):
      _ = adstock_hill.compute_decay_weights(
          alpha,
          l_range,
          _MAX_LAG+1,
          decay_function,
      )


class TestAdstock(parameterized.TestCase):
  """Tests for adstock()."""

  # Data dimensions for default parameter values.
  _N_CHAINS = 2
  _N_DRAWS = 5
  _N_GEOS = 4
  _N_MEDIA_TIMES = 10
  _N_MEDIA_CHANNELS = 3
  _MAX_LAG = 5

  # Generate random data based on dimensions specified above.
  tf.random.set_seed(1)
  _MEDIA = tfd.HalfNormal(1).sample(
      [_N_CHAINS, _N_DRAWS, _N_GEOS, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS]
  )
  _ALPHA = tfd.Uniform(0, 1).sample([_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS])

  def test_raises(self):
    """Test that exceptions are raised as expected."""
    with self.assertRaisesRegex(ValueError, "`n_times_output` cannot exceed"):
      adstock_hill.AdstockTransformer(
          alpha=self._ALPHA,
          max_lag=self._MAX_LAG,
          n_times_output=self._N_MEDIA_TIMES + 1,
      ).forward(self._MEDIA)
    with self.assertRaisesRegex(ValueError, "`media` batch dims do not"):
      adstock_hill.AdstockTransformer(
          alpha=self._ALPHA[1:, ...],
          max_lag=self._MAX_LAG,
          n_times_output=self._N_MEDIA_TIMES,
      ).forward(self._MEDIA)
    with self.assertRaisesRegex(ValueError, "`media` contains a different"):
      adstock_hill.AdstockTransformer(
          alpha=self._ALPHA,
          max_lag=self._MAX_LAG,
          n_times_output=self._N_MEDIA_TIMES,
      ).forward(self._MEDIA[..., 1:])
    with self.assertRaisesRegex(
        ValueError, "`n_times_output` must be positive"
    ):
      adstock_hill.AdstockTransformer(
          alpha=self._ALPHA, max_lag=self._MAX_LAG, n_times_output=0
      ).forward(self._MEDIA)
    with self.assertRaisesRegex(ValueError, "`max_lag` must be non-negative"):
      adstock_hill.AdstockTransformer(
          alpha=self._ALPHA, max_lag=-1, n_times_output=self._N_MEDIA_TIMES
      ).forward(self._MEDIA)

  @parameterized.named_parameters(
      dict(
          testcase_name="basic",
          media=_MEDIA,
          alpha=_ALPHA,
          n_time_output=_N_MEDIA_TIMES,
      ),
      dict(
          testcase_name="no media batch dims",
          media=_MEDIA[0, 0, ...],
          alpha=_ALPHA,
          n_time_output=_N_MEDIA_TIMES,
      ),
      dict(
          testcase_name="n_time_output < n_time",
          media=_MEDIA,
          alpha=_ALPHA,
          n_time_output=_N_MEDIA_TIMES - 1,
      ),
      dict(
          testcase_name="max_lag > n_media_times",
          media=_MEDIA[..., : (_MAX_LAG - 1)],
          alpha=_ALPHA,
          n_time_output=_N_MEDIA_TIMES,
      ),
      dict(
          testcase_name="excess lagged media history available",
          media=_MEDIA,
          alpha=_ALPHA,
          n_time_output=_N_MEDIA_TIMES - _MAX_LAG - 1,
      ),
  )
  def test_basic_output(self, media, alpha, n_time_output):
    """Basic test for valid output."""
    media_transformed = adstock_hill.AdstockTransformer(
        alpha, self._MAX_LAG, n_time_output
    ).forward(media)
    output_shape = tf.TensorShape(
        alpha.shape[:-1] + media.shape[-3] + [n_time_output] + alpha.shape[-1]
    )
    msg = f"{adstock_hill.AdstockTransformer.__name__}() failed."
    tf.debugging.assert_equal(
        media_transformed.shape, output_shape, message=msg
    )
    tf.debugging.assert_all_finite(media_transformed, message=msg)
    tf.debugging.assert_non_negative(media_transformed, message=msg)

  @parameterized.named_parameters(*_DECAY_FUNCTIONS)
  def test_max_lag_zero(self, decay_function: str):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=0,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=decay_function,
    ).forward(self._MEDIA)
    tf.debugging.assert_near(media_transformed, self._MEDIA)

  @parameterized.named_parameters(*_DECAY_FUNCTIONS)
  def test_alpha_zero(self, decay_function: str):
    """Alpha of zero is allowed, effectively no Adstock."""
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=tf.zeros_like(self._ALPHA),
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=decay_function,
    ).forward(self._MEDIA)
    tf.debugging.assert_near(media_transformed, self._MEDIA)

  @parameterized.named_parameters(*_DECAY_FUNCTIONS)
  def test_media_zero(self, decay_function: str):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=decay_function,
    ).forward(
        tf.zeros_like(self._MEDIA),
    )
    tf.debugging.assert_near(media_transformed, tf.zeros_like(self._MEDIA))

  @parameterized.named_parameters(*_DECAY_FUNCTIONS)
  def test_alpha_close_to_one(self, decay_function: str):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=0.99999 * tf.ones_like(self._ALPHA),
        max_lag=self._N_MEDIA_TIMES - 1,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=decay_function,
    ).forward(self._MEDIA)
    tf.debugging.assert_near(
        media_transformed,
        tf.cumsum(self._MEDIA, axis=-2) / self._N_MEDIA_TIMES,
        rtol=1e-4,
        atol=1e-4,
    )

  @parameterized.named_parameters(*_DECAY_FUNCTIONS)
  def test_alpha_one(self, decay_function: str):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=tf.ones_like(self._ALPHA),
        max_lag=self._N_MEDIA_TIMES - 1,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=decay_function,
    ).forward(self._MEDIA)
    tf.debugging.assert_near(
        media_transformed,
        tf.cumsum(self._MEDIA, axis=-2) / self._N_MEDIA_TIMES,
        rtol=1e-4,
        atol=1e-4,
    )

  def test_media_all_ones_geometric(self):
    # Calculate adstock on a media vector of all ones and no lag history.
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
        decay_function=constants.GEOMETRIC_DECAY,
    ).forward(tf.ones_like(self._MEDIA))
    # n_nonzero_terms is a tensor with length containing the number of nonzero
    # terms in the adstock for each output time period.
    n_nonzero_terms = np.minimum(
        np.arange(1, self._N_MEDIA_TIMES + 1), self._MAX_LAG + 1
    )
    # For each output time period and alpha value, the adstock is given by
    # adstock = series1 / series2, where:
    #   series1 = 1 + alpha + alpha^2 + ... + alpha^(n_nonzero_terms-1)
    #           = (1-alpha^n_nonzero_terms) / (1-alpha)
    #           := term1 / (1-alpha)
    #   series2 = 1 + alpha + alpha^2 + ... + alpha^max_lag
    #           = (1-alpha^(max_lag + 1)) / (1-alpha)
    #           := term2 / (1-alpha)
    # We can therefore write adstock = series1 / series2 = term1 / term2.

    # `term1` has dimensions (n_chains, n_draws, n_output_times, n_channels).
    term1 = 1 - self._ALPHA[:, :, None, :] ** n_nonzero_terms[:, None]
    # `term2` has dimensions (n_chains, n_draws, n_channels).
    term2 = 1 - self._ALPHA ** (self._MAX_LAG + 1)
    # `result` has dimensions (n_chains, n_draws, n_output_times, n_channels).
    result = term1 / term2[:, :, None, :]
    # Broadcast `result` across geos.
    result = tf.tile(
        result[:, :, None, :, :], multiples=[1, 1, self._N_GEOS, 1, 1]
    )
    tf.debugging.assert_near(media_transformed, result)

  @parameterized.named_parameters(
      dict(
          testcase_name=constants.GEOMETRIC_DECAY,
          decay_function=constants.GEOMETRIC_DECAY,
          expected_adstock=tf.constant([0.751, 0.435, 0.572]),
      ),
      dict(
          testcase_name=constants.BINOMIAL_DECAY,
          decay_function=constants.BINOMIAL_DECAY,
          expected_adstock=tf.constant([0.742, 0.463, 0.567]),
      ),
  )
  def test_output(self, decay_function: str, expected_adstock: tf.Tensor):
    """Test for valid adstock weights."""
    alpha = tf.constant([0.1, 0.5, 0.9])
    window_size = 5
    media = tf.constant([[
        [0.12, 0.55, 0.89],
        [0.34, 0.71, 0.23],
        [0.91, 0.08, 0.67],
        [0.45, 0.82, 0.11],
        [0.78, 0.29, 0.95],
    ]])
    adstock = adstock_hill.AdstockTransformer(
        alpha=alpha,
        max_lag=window_size - 1,
        n_times_output=1,
        decay_function=decay_function,
    ).forward(media)
    tf.debugging.assert_near(adstock, expected_adstock, rtol=1e-2)


class TestHill(parameterized.TestCase):
  """Tests for adstock_hill.hill()."""

  # Data dimensions for default parameter values.
  _N_CHAINS = 2
  _N_DRAWS = 5
  _N_GEOS = 4
  _N_MEDIA_TIMES = 10
  _N_MEDIA_CHANNELS = 3

  # Generate random data based on dimensions specified above.
  tf.random.set_seed(1)
  _MEDIA = tfd.HalfNormal(1).sample(
      [_N_CHAINS, _N_DRAWS, _N_GEOS, _N_MEDIA_TIMES, _N_MEDIA_CHANNELS]
  )
  _EC = tfd.Uniform(0, 1).sample([_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS])
  _SLOPE = tfd.HalfNormal(1).sample([_N_CHAINS, _N_DRAWS, _N_MEDIA_CHANNELS])

  def test_raises(self):
    """Test that exceptions are raised as expected."""
    with self.assertRaisesRegex(ValueError, "`slope` and `ec` dimensions"):
      adstock_hill.HillTransformer(
          ec=self._EC, slope=self._SLOPE[1:, ...]
      ).forward(self._MEDIA)
    with self.assertRaisesRegex(ValueError, "`media` batch dims do not"):
      adstock_hill.HillTransformer(ec=self._EC, slope=self._SLOPE).forward(
          self._MEDIA[1:, ...]
      )
    with self.assertRaisesRegex(ValueError, "`media` contains a different"):
      adstock_hill.HillTransformer(ec=self._EC, slope=self._SLOPE).forward(
          self._MEDIA[..., 1:]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="basic",
          media=_MEDIA,
      ),
      dict(
          testcase_name="no media batch dims",
          media=_MEDIA[0, 0, ...],
      ),
  )
  def test_basic_output(self, media):
    """Basic test for valid output."""
    media_transformed = adstock_hill.HillTransformer(
        ec=self._EC, slope=self._SLOPE
    ).forward(media)
    tf.debugging.assert_equal(media_transformed.shape, self._MEDIA.shape)
    tf.debugging.assert_all_finite(media_transformed, message="")
    tf.debugging.assert_non_negative(media_transformed)

  @parameterized.named_parameters(
      dict(
          testcase_name="media=0",
          media=tf.zeros_like(_MEDIA),
          ec=_EC,
          slope=_SLOPE,
          result=tf.zeros_like(_MEDIA),
      ),
      dict(
          testcase_name="slope=ec=1",
          media=_MEDIA,
          ec=tf.ones_like(_EC),
          slope=tf.ones_like(_SLOPE),
          result=_MEDIA / (1 + _MEDIA),
      ),
      dict(
          testcase_name="slope=0",
          media=_MEDIA,
          ec=_EC,
          slope=tf.zeros_like(_SLOPE),
          result=0.5 * tf.ones_like(_MEDIA),
      ),
  )
  def test_known_outputs(self, media, ec, slope, result):
    """Test special cases where expected output is known."""
    media_transformed = adstock_hill.HillTransformer(
        ec=ec, slope=slope
    ).forward(media)
    tf.debugging.assert_near(media_transformed, result)


if __name__ == "__main__":
  absltest.main()
