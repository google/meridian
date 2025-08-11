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
from meridian import backend
from meridian.backend import test_utils
from meridian.model import adstock_hill
import numpy as np

tfd = backend.tfd


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
  backend.set_random_seed(1)
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
    output_shape = (
        tuple(alpha.shape[:-1])
        + media.shape[-3:-2]
        + (n_time_output,)
        + alpha.shape[-1:]
    )
    msg = f"{adstock_hill.AdstockTransformer.__name__}() failed."
    self.assertEqual(media_transformed.shape, output_shape, msg=msg)
    test_utils.assert_all_finite(media_transformed, err_msg=msg)
    test_utils.assert_all_non_negative(media_transformed, err_msg=msg)

  def test_max_lag_zero(self):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=0,
        n_times_output=self._N_MEDIA_TIMES,
    ).forward(self._MEDIA)
    test_utils.assert_allclose(media_transformed, self._MEDIA)

  def test_alpha_zero(self):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=backend.zeros_like(self._ALPHA),
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
    ).forward(self._MEDIA)
    test_utils.assert_allclose(media_transformed, self._MEDIA)

  def test_media_zero(self):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
    ).forward(
        backend.zeros_like(self._MEDIA),
    )
    test_utils.assert_allclose(
        media_transformed, backend.zeros_like(self._MEDIA)
    )

  def test_alpha_close_to_one(self):
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=0.99999 * backend.ones_like(self._ALPHA),
        max_lag=self._N_MEDIA_TIMES - 1,
        n_times_output=self._N_MEDIA_TIMES,
    ).forward(self._MEDIA)
    test_utils.assert_allclose(
        media_transformed,
        backend.cumsum(self._MEDIA, axis=-2) / self._N_MEDIA_TIMES,
        rtol=1e-4,
        atol=1e-4,
    )

  def test_media_all_ones(self):
    # Calculate adstock on a media vector of all ones and no lag history.
    media_transformed = adstock_hill.AdstockTransformer(
        alpha=self._ALPHA,
        max_lag=self._MAX_LAG,
        n_times_output=self._N_MEDIA_TIMES,
    ).forward(backend.ones_like(self._MEDIA))
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
    result = backend.tile(
        result[:, :, None, :, :], multiples=[1, 1, self._N_GEOS, 1, 1]
    )
    test_utils.assert_allclose(media_transformed, result)

  def test_output(self):
    """Test for valid adstock weights."""
    alpha = backend.to_tensor([0.1, 0.5, 0.9])
    window_size = 5
    media = backend.to_tensor([[
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
    ).forward(media)
    expected_adstock = backend.to_tensor([[[0.751, 0.435, 0.572]]])
    test_utils.assert_allclose(adstock, expected_adstock, rtol=1e-2)


class TestHill(parameterized.TestCase):
  """Tests for adstock_hill.hill()."""

  # Data dimensions for default parameter values.
  _N_CHAINS = 2
  _N_DRAWS = 5
  _N_GEOS = 4
  _N_MEDIA_TIMES = 10
  _N_MEDIA_CHANNELS = 3

  # Generate random data based on dimensions specified above.
  backend.set_random_seed(1)
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
    self.assertEqual(media_transformed.shape, self._MEDIA.shape)
    test_utils.assert_all_finite(media_transformed, err_msg="")
    test_utils.assert_all_non_negative(media_transformed)

  @parameterized.named_parameters(
      dict(
          testcase_name="media=0",
          media=backend.zeros_like(_MEDIA),
          ec=_EC,
          slope=_SLOPE,
          result=backend.zeros_like(_MEDIA),
      ),
      dict(
          testcase_name="slope=ec=1",
          media=_MEDIA,
          ec=backend.ones_like(_EC),
          slope=backend.ones_like(_SLOPE),
          result=_MEDIA / (1 + _MEDIA),
      ),
      dict(
          testcase_name="slope=0",
          media=_MEDIA,
          ec=_EC,
          slope=backend.zeros_like(_SLOPE),
          result=0.5 * backend.ones_like(_MEDIA),
      ),
  )
  def test_known_outputs(self, media, ec, slope, result):
    """Test special cases where expected output is known."""
    media_transformed = adstock_hill.HillTransformer(
        ec=ec, slope=slope
    ).forward(media)
    test_utils.assert_allclose(media_transformed, result)


if __name__ == "__main__":
  absltest.main()
