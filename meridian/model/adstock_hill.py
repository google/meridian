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
from collections.abc import Sequence
import dataclasses
from meridian import backend
from meridian import constants


__all__ = [
    'AdstockDecaySpec',
    'AdstockHillTransformer',
    'AdstockTransformer',
    'HillTransformer',
    'transform_non_negative_reals_distribution',
    'compute_decay_weights',
]


@dataclasses.dataclass(frozen=True)
class AdstockDecaySpec:
  """Specification for each channel's adstock decay function.

  This class contains the adstock decay function(s) to use for each channel
  that the adstock transformation is applied to.

  Attributes:
    media: A string or sequence of strings specifying the adstock function(s)
      to use for media channels.
    rf: A string or sequence of strings specifying the adstock function(s)
      to use for reach and frequency channels.
    organic_media: A string or sequence of strings specifying the adstock
      function(s) to use for organic media channels.
    organic_rf: A string or sequence of strings specifying the adstock
      function(s) to use for organic reach and frequency channels.
  """

  media: str | Sequence[str] = constants.GEOMETRIC_DECAY
  rf: str | Sequence[str] = constants.GEOMETRIC_DECAY
  organic_media: str | Sequence[str] = constants.GEOMETRIC_DECAY
  organic_rf: str | Sequence[str] = constants.GEOMETRIC_DECAY

  @classmethod
  def from_consistent_type(
      cls,
      consistent_decay_function: str = constants.GEOMETRIC_DECAY,
  ) -> 'AdstockDecaySpec':
    """Create an `AdstockDecaySpec` with the same decay function for all channels.

    Arguments:
      consistent_decay_function: A string denoting the adstock decay function
      to use for all channels that the Adstock transformation is applied to.

    Raises:
      ValueError: If `consistent_decay_function` is not 'geometric' or
      'binomial'.
    """
    adstock_decay_functions = dict.fromkeys(
        constants.ADSTOCK_CHANNELS, consistent_decay_function
        )
    return cls(**adstock_decay_functions)

  def __post_init__(self):
    adstock_decay_functions = [
        self.media,
        self.rf,
        self.organic_media,
        self.organic_rf,
    ]

    for v in adstock_decay_functions:
      if isinstance(v, str):
        _validate_adstock_decay_function(v)
      else:
        for vi in v:
          _validate_adstock_decay_function(vi)


def _validate_adstock_decay_function(adstock_decay_func: str):
  if adstock_decay_func not in constants.ADSTOCK_DECAY_FUNCTIONS:
    raise ValueError(
        'Unrecognized adstock decay function value '
        f'("{adstock_decay_func}"), expected one of '
        f'{constants.ADSTOCK_DECAY_FUNCTIONS}.'
    )


def compute_decay_weights(
    alpha: backend.Tensor,
    l_range: backend.Tensor,
    window_size: int,
    decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
    normalize: bool = True,
) -> backend.Tensor:
  """Computes decay weights using geometric and/or binomial decay.

  This function always broadcasts the lag dimension (`l_range`) to the
  trailing axis of the output tensor.

  Args:
    alpha: The parameter for the adstock decay function.
    l_range: A 1D tensor representing the lag range, e.g., `[w-1, w-2, ...,
      0]`.
    window_size: The number of time periods that go into the adstock weighted
      average for each output time period.
    decay_functions: String or sequence of strings indicating the decay
      function(s) to use for the Adstock calculation. Allowed values
      are 'geometric' and 'binomial'.
    normalize: A boolean indicating whether to normalize the weights. Default:
      `True`.

  Returns:
    A tensor of weights with a shape of `(*alpha.shape, len(l_range))`.

  Raises:
    ValueError: If the shape of `decay_functions` is not broadcastable to
    the shape of `alpha`.

  """

  if isinstance(decay_functions, str):
    # Same decay function for all channels
    return _compute_single_decay_function_weights(
        alpha, l_range, window_size, decay_functions, normalize,
    )

  binomial_weights = _compute_single_decay_function_weights(
      alpha, l_range, window_size, constants.BINOMIAL_DECAY, normalize,
  )
  geometric_weights = _compute_single_decay_function_weights(
      alpha, l_range, window_size, constants.GEOMETRIC_DECAY, normalize,
  )

  is_binomial = [s == constants.BINOMIAL_DECAY for s in decay_functions]
  binomial_decay_mask = backend.reshape(
      backend.to_tensor(is_binomial, dtype=backend.bool_),
      (-1, 1),
  )

  try:
    # pytype: disable=bad-return-type
    return backend.where(
        binomial_decay_mask, binomial_weights, geometric_weights
    )
    # pytype: enable=bad-return-type
  except (backend.errors.InvalidArgumentError, ValueError) as e:
    raise ValueError(
        f'The shape of `alpha` ({alpha.shape}) is incompatible with the length'
        f' of `decay_functions` ({len(decay_functions)})'
    ) from e


def _compute_single_decay_function_weights(
    alpha: backend.Tensor,
    l_range: backend.Tensor,
    window_size: int,
    decay_function: str,
    normalize: bool,
    ) -> backend.Tensor:
  """Computes decay weights using geometric decay.

  This function always broadcasts the lag dimension (`l_range`) to the
  trailing axis of the output tensor.

  Args:
      alpha: The parameter for the adstock decay function.
      l_range: A 1D tensor representing the lag range, e.g., `[w-1, w-2, ...,
        0]`.
      window_size: The number of time periods that go into the adstock weighted
        average for each output time period.
      decay_function: String indicating the decay function to use for the
        Adstock calculation. Allowed values are 'geometric' and 'binomial'.
      normalize: A boolean indicating whether to normalize the weights.

  Returns:
      A tensor of weights with a shape of `(*alpha.shape, len(l_range))`.
  """
  expanded_alpha = backend.expand_dims(alpha, -1)

  if decay_function == constants.GEOMETRIC_DECAY:
    weights = expanded_alpha**l_range
  elif decay_function == constants.BINOMIAL_DECAY:
    mapped_alpha_binomial = _map_alpha_for_binomial_decay(expanded_alpha)
    weights = (1 - l_range / window_size) ** mapped_alpha_binomial
  else:
    raise ValueError(f'Unsupported decay function: {decay_function}')

  if normalize:
    normalization_factors = backend.reduce_sum(weights, axis=-1, keepdims=True)
    return backend.divide(weights, normalization_factors)
  return weights


def _validate_arguments(
    media: backend.Tensor,
    alpha: backend.Tensor,
    max_lag: int,
    n_times_output: int,
) -> None:
  batch_dims = alpha.shape[:-1]
  n_media_times = media.shape[-2]

  if n_times_output > n_media_times:
    raise ValueError(
        '`n_times_output` cannot exceed number of time periods in the media'
        ' data.'
    )
  if tuple(media.shape[:-3]) not in [(), tuple(batch_dims)]:
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
    media: backend.Tensor,
    alpha: backend.Tensor,
    max_lag: int,
    n_times_output: int,
    decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
) -> backend.Tensor:
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
    media = backend.concatenate([backend.zeros(pad_shape), media], axis=-2)

  # Adstock calculation.
  window_list = [None] * window_size
  for i in range(window_size):
    window_list[i] = media[..., i : i + n_times_output, :]
  windowed = backend.stack(window_list)
  l_range = backend.arange(window_size - 1, -1, -1, dtype=backend.float32)
  weights = compute_decay_weights(
      alpha=alpha,
      l_range=l_range,
      window_size=window_size,
      decay_functions=decay_functions,
      normalize=True,
  )
  return backend.einsum('...mw,w...gtm->...gtm', weights, windowed)


def _map_alpha_for_binomial_decay(x: backend.Tensor):
  # Map x -> 1/x - 1 to map [0, 1] to [0, +inf].
  # 0 -> +inf is a valid mapping and reflects the "no adstock" case.

  return 1 / x - 1


def _hill(
    media: backend.Tensor,
    ec: backend.Tensor,
    slope: backend.Tensor,
) -> backend.Tensor:
  """Computes the Hill function."""
  batch_dims = slope.shape[:-1]

  # Argument checks.
  if slope.shape != ec.shape:
    raise ValueError('`slope` and `ec` dimensions do not match.')
  if tuple(media.shape[:-3]) not in [(), tuple(batch_dims)]:
    raise ValueError(
        '`media` batch dims do not match `slope` and `ec` batch dims. '
        'If `media` has batch dims, then they must match `slope` and '
        '`ec`.'
    )
  if media.shape[-1] != slope.shape[-1]:
    raise ValueError(
        '`media` contains a different number of channels than `slope` and `ec`.'
    )

  t1 = media ** slope[..., backend.newaxis, backend.newaxis, :]
  t2 = (ec**slope)[..., backend.newaxis, backend.newaxis, :]
  return t1 / (t1 + t2)


class AdstockHillTransformer(metaclass=abc.ABCMeta):
  """Abstract class to compute the Adstock and Hill transformation of media."""

  @abc.abstractmethod
  def forward(self, media: backend.Tensor) -> backend.Tensor:
    """Computes the Adstock and Hill transformation of a given media tensor."""
    pass


class AdstockTransformer(AdstockHillTransformer):
  """Class to compute the Adstock transformation of media."""

  def __init__(
      self,
      alpha: backend.Tensor,
      max_lag: int,
      n_times_output: int,
      decay_functions: str | Sequence[str] = constants.GEOMETRIC_DECAY,
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
      decay_functions: String or list of strings indicating the decay
        function(s) to use for the Adstock calculation for each channel.
        Default is geometric decay for all channels.
    """
    self._alpha = alpha
    self._max_lag = max_lag
    self._n_times_output = n_times_output
    self._decay_functions = decay_functions

  def forward(self, media: backend.Tensor) -> backend.Tensor:
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
        decay_functions=self._decay_functions,
    )


class HillTransformer(AdstockHillTransformer):
  """Class to compute the Hill transformation of media."""

  def __init__(self, ec: backend.Tensor, slope: backend.Tensor):
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

  def forward(self, media: backend.Tensor) -> backend.Tensor:
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


def transform_non_negative_reals_distribution(
    distribution: backend.tfd.Distribution,
) -> backend.tfd.TransformedDistribution:
  """Transforms a distribution with support on `[0, infinity)` to `(0, 1]`.

  This allows for defining a prior on `alpha_*`, the exponent of the binomial
  Adstock decay function, directly, and then translating it to a distribution
  defined on the unit interval as Meridian expects. This transformation
  `(x -> 1 / (1 + x))` is the inverse of the interval mapping the Meridian
  performs `(x -> 1 / x - 1)` on alpha to define the binomial Adstock
  decay function's exponent.

  For example, to define a `LogNormal(0.2, 0.9)` prior on `alpha_*`:

  ```python
  from meridian import backend
  alpha_star_prior = backend.tfd.LogNormal(0.2, 0.9)
  alpha_prior = transform_non_negative_reals_distribution(alpha_star_prior)
  prior = prior_distribution.PriorDistribution(
      alpha_m=alpha_prior,
      ...
  )
  ```

  Args:
    distribution: A Tensorflow Probability distribution with support on `[0,
      infinity)`.

  Returns:
    A Tensorflow Probability `TransformedDistribution` with support on `(0, 1]`,
    such that the resultant prior on `alpha_*` is the input distribution.
  """

  bijector = backend.bijectors.Chain(
      [backend.bijectors.Reciprocal(), backend.bijectors.Shift(1)]
  )

  return backend.tfd.TransformedDistribution(
      distribution=distribution,
      bijector=bijector,
      name=f'{distribution.name}UnitIntervalMapped',
  )
