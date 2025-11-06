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

"""Function registry for Serde."""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any, Callable
import warnings


class SourceCodeRetrievalError(Exception):
  """Raised when the source code of a function cannot be retrieved."""


class LambdaSourceCodeWarning(UserWarning):
  """Warning issued when trying to get source code of a lambda function."""


def _get_func_source(func: Callable[..., Any]) -> str:
  """Returns the source code of a function.

  Args:
    func: The function to get the source code for.

  Returns:
    The source code of the function.

  Raises:
    SourceCodeRetrievalError: If the source code of the function cannot be
      retrieved.
  """
  if hasattr(func, "__code__") and func.__code__.co_name == "<lambda>":
    warnings.warn(
        "Retrieving the source code of a lambda function might not work"
        " successfully. It's recommended to explicitly define a function.",
        LambdaSourceCodeWarning,
    )
  try:
    return inspect.getsource(func)
  except (OSError, TypeError) as e:
    raise SourceCodeRetrievalError(
        f"Source code of function {func} is not retrievable."
    ) from e


def _get_hash(value: str) -> str:
  """Returns a SHA-256 hash of the given value."""
  encoded_string = value.encode("utf-8")
  sha_256_hash = hashlib.sha256()
  sha_256_hash.update(encoded_string)
  return sha_256_hash.hexdigest()


class FunctionRegistry(dict[str, Callable[..., Any]]):
  """A dictionary-like container for custom functions used in serialization.

  This class extends dict and provides methods for hashing, validation,
  and key retrieval based on function identity, required for safe
  serialization and deserialization of models that use custom functions.
  """

  def __init__(self, *args, **kwargs):
    """Initializes the FunctionRegistry.

    Accepts the same arguments as a standard dictionary constructor.
    For example:
      reg = FunctionRegistry({'func1': my_func1, 'func2': my_func2})
      reg = FunctionRegistry(func1=my_func1, func2=my_func2)

    Args:
      *args: Positional arguments to pass to the dictionary constructor.
      **kwargs: Keyword arguments to pass to the dictionary constructor.
    """
    super().__init__(*args, **kwargs)

  @functools.cached_property
  def hashed_registry(self) -> dict[str, str]:
    """Returns hashed function registry with keys mapped to hashed function code."""
    return {
        key: _get_hash(_get_func_source(function))
        for key, function in self.items()
    }

  def validate(self, stored_hashed_function_registry: dict[str, str]):
    """Validates whether functions within the registry have changed.

    It checks that all functions listed in stored_hashed_function_registry
    are present in this registry, and that their source code hash matches
    the stored hash.

    Args:
      stored_hashed_function_registry: The hashed function registry from the
        serialized object.

    Raises:
      ValueError: If a function is missing or a hash mismatch is detected.
    """
    if not stored_hashed_function_registry and self:
      warnings.warn(
          "A function registry was provided during loading, but none was"
          " found on the serialized object. Custom functions will be"
          " ignored."
      )
      return

    for key, stored_hash in stored_hashed_function_registry.items():
      if key not in self:
        raise ValueError(
            f"Function '{key}' is required by the serialized object but"
            " is missing from the provided function registry."
        )
      func = self[key]
      try:
        source_code = _get_func_source(func)
      except SourceCodeRetrievalError as e:
        raise ValueError(
            f"Failed to retrieve source code of function {key}."
        ) from e
      evaluated_hash = _get_hash(source_code)
      if stored_hash != evaluated_hash:
        raise ValueError(f"Function registry hash mismatch for {key}.")

  def get_function_key(self, func: Callable[..., Any]) -> str | None:
    """Returns the function key for the given function from the registry."""
    for function_key, registry_func in self.items():
      if func is registry_func:
        return function_key
    return None
