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

"""Serialization and deserialization of Meridian models."""

import abc
from typing import Generic, TypeVar


WireFormat = TypeVar("WireFormat")
PythonType = TypeVar("PythonType")


class Serde(Generic[WireFormat, PythonType], abc.ABC):
  """Serializes and deserializes a Python type into a wire format."""

  def serialize(self, obj: PythonType, **kwargs) -> WireFormat:
    """Serializes the given object into a wire format."""
    raise NotImplementedError()

  def deserialize(
      self, serialized: WireFormat, serialized_version: str = "", **kwargs
  ) -> PythonType:
    """Deserializes the given wire format into a Python object.

    Args:
      serialized: The serialized object.
      serialized_version: The version of the serialized object. This is used to
        handle changes in deserialization logic across different versions.
      **kwargs: Additional keyword arguments to pass to the deserialization
        function.

    Returns:
      The deserialized object.
    """
    raise NotImplementedError()
