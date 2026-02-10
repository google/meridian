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

"""Data argument builders for various API surfaces in Meridian."""

from collections.abc import Sequence
from typing import Generic, TypeVar


__all__ = [
    'OrderedListArgumentBuilder',
]


T = TypeVar('T')


class OrderedListArgumentBuilder(Generic[T]):
  """A simple builder for an argument that expects an ordered list of values.

  For example, some Meridian function requires a list/array of some values that:

  - Must have the same length as some channel coordinates in the input data.
  - Must be in the same order as some channel coordinates in the input data.

  This argument builder can be bound to one such channel coordinates list, and
  provides a user-friendly and human-readable way to seed it with user given
  values.

  For example, to create an array (list) of values that must be indexed on the
  *paid media* channels in the input data:

  ```python
  paid_media_channels_arg_builder = (
      OrderedListArgumentBuilder[float](input_data.get_all_paid_channels())
  )
  # Note: rather than creating this builder directly, use methods like
  # `InputData.get_all_paid_channels_argument_builder()` where the container
  # determines which coordinates to bind the builder with.

  # Use `.with_default_value()` to set a default value for coordinates that are
  # not given in `__call__`.
  paid_media_channels_arg_builder = (
      paid_media_channels_arg_builder.with_default_value(0.3)
  )

  # Assuming we have paid media channels ['display', 'search', social'].
  some_arg = paid_media_channels_arg_builder(
      display=0.1,
      social=0.25,
  )
  # some_arg == [0.1, 0.3, 0.25]
  ```

  See: `InputData.get_paid_channels_argument_builder()`.
  """

  def __init__(self, ordered_coords: Sequence[str]):
    self._ordered_coords = list(ordered_coords)
    self._default_value = None  # Applied when a coordinate value is not given.

  def with_default_value(
      self, default_value: T
  ) -> 'OrderedListArgumentBuilder':
    """Sets the default value for coordinates that are not given in `__call__`.

    Args:
      default_value: The default value to use for coordinates that are not given
        in `__call__`. If unset (or set to `None`), then `__call__` will raise
        an error if any bound coordinate's value is not given.

    Returns:
      This builder itself for fluent chaining.
    """
    self._default_value = default_value
    return self

  def __call__(self, **kwargs) -> list[T]:
    """Builds an ordered argument values list, given the bound coordinates list.

    Args:
      **kwargs: The keys in `kwargs` are channel names. All channel names must
        be present in the `ordered_coords` bound to this builder.

    Returns:
      A list of values, in the same order as the `ordered_coords` bound to this
      builder.
    """
    if self._default_value is None and (
        set(kwargs.keys()) != set(self._ordered_coords)
    ):
      raise ValueError(
          'All coordinates must be present in the given keyword arguments: '
          f'Given: {kwargs.keys()} vs Expected: {self._ordered_coords}'
      )
    return [kwargs.get(c, self._default_value) for c in self._ordered_coords]
