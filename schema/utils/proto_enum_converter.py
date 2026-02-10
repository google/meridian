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

"""A generic class for converting between Protobuf enums and strings."""

from typing import Generic, Type, TypeVar
import warnings

import bidict


EnumType = TypeVar("EnumType")
DefaultType = TypeVar("DefaultType")


class ProtoEnumConverter(Generic[EnumType, DefaultType]):
  """Class for converting between proto enums and strings."""

  def __init__(
      self,
      enum_message: Type[EnumType],
      enum_display_name: str,
      mapping: bidict.bidict,
      enum_unspecified: EnumType,
      default_when_unspecified: DefaultType,
  ):
    """Initializes the ProtoEnumConverter.

    Arguments:
      enum_message: The proto enum message definition.
      enum_display_name: The loggable proto enum message name.
      mapping: The mapping between the proto enum name and the string
        representation.
      enum_unspecified: The enum value that corresponds to unspecified.
      default_when_unspecified: The default value that should be returned when
        the proto enum is unspecified.
    """
    self.enum_message = enum_message
    self.enum_display_name = enum_display_name
    self.mapping = mapping
    self.enum_unspecified = enum_unspecified
    self.default_when_unspecified = default_when_unspecified

  def to_proto(self, string_value: str | None) -> EnumType:
    """Converts a string to its corresponding proto enum.

    Args:
      string_value: The string to convert to a proto enum.

    Returns:
      The corresponding proto enum or enum_unspecified when the enum message
      doesn't exist.

    Raises:
      ValueError when given string is not found in the mapping.
    """
    if string_value is None:
      return self.enum_unspecified

    proto_name = self.mapping.get(string_value)
    if proto_name:
      try:
        return self.enum_message.Value(proto_name)
      except ValueError:
        warnings.warn(
            "Invalid %s value: %s. Resolving to %s."
            % (
                self.enum_message.DESCRIPTOR.name,
                string_value,
                self.enum_message.Name(self.enum_unspecified),
            )
        )
        return self.enum_unspecified
    else:
      raise ValueError(
          f"Unmatched {self.enum_message.DESCRIPTOR.name} value:"
          f" {string_value}."
      )

  def from_proto(self, proto_enum: EnumType) -> str | DefaultType:
    """Converts a proto enum to its string representation.

    Args:
      proto_enum: The enum value to convert to its string representation

    Returns:
      The string representation of the given proto_enum or the default value
      when the proto enum is unspecified.

    Raises:
      ValueError when given proto enum is not found in the mapping.
    """
    if proto_enum == self.enum_unspecified:
      warnings.warn(
          "%s is unspecified. Resolving to default: %s."
          % (
              self.enum_display_name,
              self.enum_message.Name(self.enum_unspecified),
          )
      )
      return self.default_when_unspecified

    try:
      proto_name = self.enum_message.Name(proto_enum)
    except ValueError as e:
      raise ValueError(
          f"Invalid {self.enum_message.DESCRIPTOR.name} proto enum value:"
          f" {proto_enum}."
      ) from e

    try:
      return self.mapping.inv[proto_name]
    except KeyError as e:
      raise KeyError(
          f"Protobuf enum name '{proto_name}' is not configured in the mapping."
      ) from e
