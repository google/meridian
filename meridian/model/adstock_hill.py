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

"""Function definitions for Adstock and Hill calculations."""

import abc
from meridian import constants
import tensorflow as tf

__all__ = [
    'AdstockHillTransformer',
    'AdstockTransformer',
    'HillTransformer',
]


def _validate_arguments(
    media: tf.Tensor, alpha: tf.Tensor, max_lag: int, n_times_output: int
) -> None:
  batch_dims = alpha.shape[:-1]
  n_media_times = media.shape[-2]

  if n_times_output > n_media_times:
    raise ValueError(
        '`n_times_output` cannot exceed number of time periods in the media'
        ' data.'
    )
  if media.shape[:-3] not in [tf.TensorShape([]), tf.TensorShape(batch_dims)]:
    raise ValueError(
        '`media` batch dims do not match `alpha` batch dims. If `media` '
        'has batch dims, then they must match `alpha`.'
    )
  if media.shape[-1] != alpha.shape[-1]:
    raise ValueError(
        '`media` contains a different number of channels than `alpha`.'
    )
  if n_times_output <= 0:
    raise ValueError('`n_times_output` must be positive.')
  if max_lag < 0:
    raise ValueError('`max_lag` must be non-negative.')


def _adstock(
    media: tf.Tensor,
    alpha: tf.Tensor,
    max_lag: int,
    n_times_output: int,
    decay_function: str = constants.GEOMETRIC_DECAY,
) -> tf.Tensor:
  """Computes the Adstock function."""
  _validate_arguments(
      media=media, alpha=alpha, max_lag=max_lag, n_times_output=n_times_output
  )
  # alpha dims: batch_dims, n_media_channels.
  # media dims: batch_dims (optional), n_geos, n_media_times, n_channels.
  n_media_times = media.shape[-2]

  # The window size is the number of time periods that go into the adstock
  # weighted average for each output time period.
  window_size = min(max_lag + 1, n_media_times)

  # Drop any excess historical time periods that do not affect output.
  required_n_media_times = n_times_output + window_size - 1
  if n_media_times > required_n_media_times:
    # Note that ProductCoverage believes that unit tests should cover the case
    # that both conditions (1) `n_media_times > required_n_media_times` and
    # (2) `window_size = n_media_times`. However, this combination of conditions
    # is not possible.
    media = media[..., -required_n_media_times:, :]

  # If necessary, pad the media tensor with zeros. For each output time period,
  # we need a media history of at least `window_size` time periods from which to
  # calculate the adstock. The purpose of padding is to allow us to apply
  # a fixed window size calculation to all time periods. The purpose is NOT to
  # ensure that we have `max_lag` historical time periods for each output time
  # period. If `max_lag` is set to a huge value, it is not necessary to pad the
  # data with a huge number of zeros. The `normalization_factors` normalize
  # the weights to the correct values even if `window_size` < `max_lag`+1.
  if n_media_times < required_n_media_times:
    pad_shape = (
        media.shape[:-2]
        + (required_n_media_times - n_media_times,)
        + (media.shape[-1],)
    )
    media = tf.concat([tf.zeros(pad_shape), media], axis=-2)

  weights = _adstock_weights(alpha, window_size, decay_function)

  # Adstock calculation.
  window_list = [None] * window_size
  for i in range(window_size):
    window_list[i] = media[..., i : i + n_times_output, :]

  windowed = tf.stack(window_list)

  return tf.einsum('...mw,w...gtm->...gtm', weights, windowed)


def _adstock_weights(
    alpha: tf.Tensor,
    window_size: int,
    decay_function: str = constants.GEOMETRIC_DECAY,
):
  """Computes the weights for the Adstock function."""

  l_range = tf.range(window_size - 1, -1, -1, dtype=tf.float32)

  match decay_function:
    case constants.GEOMETRIC_DECAY:
      base = tf.expand_dims(alpha, -1)
      weights = base**l_range
    case constants.BINOMIAL_DECAY:
      alpha_binomial = tf.expand_dims(alpha, -1)
      mapped_alpha_binomial = _map_alpha_for_binomial_decay(alpha_binomial)
      weights = (1 - l_range / window_size) ** mapped_alpha_binomial
    case _:
      raise ValueError(f'Unsupported decay function: {decay_function}')

  normalization_factors = tf.reduce_sum(weights, axis=-1, keepdims=True)
  weights = tf.divide(weights, normalization_factors)

  return weights


def _map_alpha_for_binomial_decay(x: tf.Tensor):
  # Map x -> 1/x - 1 to map [0, 1] to [0, +inf].
  # 0 -> +inf is a valid mapping and reflects the "no adstock" case.

  return 1 / x - 1


def _hill(
    media: tf.Tensor,
    ec: tf.Tensor,
    slope: tf.Tensor,
) -> tf.Tensor:
  """Computes the Hill function."""
  batch_dims = slope.shape[:-1]

  # Argument checks.
  if slope.shape != ec.shape:
    raise ValueError('`slope` and `ec` dimensions do not match.')
  if media.shape[:-3] not in [tf.TensorShape([]), tf.TensorShape(batch_dims)]:
    raise ValueError(
        '`media` batch dims do not match `slope` and `ec` batch dims. '
        'If `media` has batch dims, then they must match `slope` and '
        '`ec`.'
    )
  if media.shape[-1] != slope.shape[-1]:
    raise ValueError(
        '`media` contains a different number of channels than `slope` and `ec`.'
    )

  t1 = media ** slope[..., tf.newaxis, tf.newaxis, :]
  t2 = (ec**slope)[..., tf.newaxis, tf.newaxis, :]
  return t1 / (t1 + t2)


class AdstockHillTransformer(metaclass=abc.ABCMeta):
  """Abstract class to compute the Adstock and Hill transformation of media."""

  @abc.abstractmethod
  def forward(self, media: tf.Tensor) -> tf.Tensor:
    """Computes the Adstock and Hill transformation of a given media tensor."""
    pass


class AdstockTransformer(AdstockHillTransformer):
  """Class to compute the Adstock transformation of media."""

  def __init__(
      self,
      alpha: tf.Tensor,
      max_lag: int,
      n_times_output: int,
      decay_function: str = constants.GEOMETRIC_DECAY,
  ):
    """Initializes this transformer based on Adstock function parameters.

    Args:
      alpha: Tensor of `alpha` parameters taking values in `[0, 1]` with
        dimensions `[..., n_media_channels]`. Batch dimensions `(...)` are
        optional. Note that `alpha = 0` is allowed, so it is possible to put a
        point mass prior at zero (effectively no Adstock).
      max_lag: Integer indicating the maximum number of lag periods (≥ `0`) to
        include in the Adstock calculation.
      n_times_output: Integer indicating the number of time periods to include
        in the output tensor. Cannot exceed the number of time periods of the
        media argument, for example, `media.shape[-2]`. The output time periods
        correspond to the most recent time periods of the media argument. For
        example, `media[..., -n_times_output:, :]` represents the media
        execution of the output weeks.
      decay_function: String indicating the decay function to use for the
        Adstock calculation. Allowed values are 'geometric' and 'binomial'.
        Default is 'geometric'.
    """
    self._alpha = alpha
    self._max_lag = max_lag
    self._n_times_output = n_times_output
    self._decay_function = decay_function

  def forward(self, media: tf.Tensor) -> tf.Tensor:
    """Computes the Adstock transformation of a given `media` tensor.

    For geo `g`, time period `t`, and media channel `m`, Adstock is calculated
    as `adstock_{g,t,m} = sum_{i=0}^max_lag media_{g,t-i,m} alpha^i`.

    Note: The Hill function can be applied before or after Adstock. If Hill is
    applied first, then the Adstock media input can contain batch dimensions
    because the transformed media tensor will be different for each posterior
    sample.

    Args:
      media: Tensor of media values with dimensions `[..., n_geos,
        n_media_times, n_media_channels]`. Batch dimensions `(...)` are
        optional, but if batch dimensions are included, they must match the
        batch dimensions of `alpha`. Media is not required to have batch
        dimensions even if `alpha` contains batch dimensions.

    Returns:
      Tensor with dimensions `[..., n_geos, n_times_output, n_media_channels]`
      representing Adstock transformed media.
    """
    return _adstock(
        media=media,
        alpha=self._alpha,
        max_lag=self._max_lag,
        n_times_output=self._n_times_output,
        decay_function=self._decay_function,
    )


class HillTransformer(AdstockHillTransformer):
  """Class to compute the Hill transformation of media."""

  def __init__(self, ec: tf.Tensor, slope: tf.Tensor):
    """Initializes the instance based on the Hill function parameters.

    Args:
      ec: Tensor with dimensions `[..., n_media_channels]`. Batch dimensions
        `(...)` are optional, but if batch dimensions are included, they must
        match the batch dimensions of `ec`.
      slope: Tensor with dimensions `[..., n_media_channels]`. Batch dimensions
        `(...)` are optional, but if batch dimensions are included, they must
        match the batch dimensions of `slope`.
    """
    self._ec = ec
    self._slope = slope

  def forward(self, media: tf.Tensor) -> tf.Tensor:
    """Computes the Hill transformation of a given `media` tensor.

    Calculates results for the Hill function, which accounts for the diminishing
    returns of media effects.

    Args:
      media: Tensor with dimensions `[..., n_geos, n_media_times,
        n_media_channels]`. Batch dimensions `(...)` are optional, but if batch
        dimensions are included, they must match the batch dimensions of `slope`
        and `ec`. Media is not required to have batch dimensions even if `slope`
        and `ec` contain batch dimensions.

    Returns:
      Tensor with dimensions `[..., n_geos, n_media_times, n_media_channels]`
      representing Hill-transformed media.
    """
    return _hill(media=media, ec=self._ec, slope=self._slope)
