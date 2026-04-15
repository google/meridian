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

"""Backend configuration for Meridian."""

import enum
import os
from typing import Union
import warnings


class Backend(enum.Enum):
  TENSORFLOW = "tensorflow"
  JAX = "jax"


class ComputationBackend(enum.IntEnum):
  """A computational backend for a Meridian model.

  This mirrors the `ComputationBackend` enum in
  `proto/mmm/v1/model/meridian/meridian_model.proto`.
  """

  COMPUTATION_BACKEND_UNSPECIFIED = 0
  TENSORFLOW = 1
  JAX = 2


class ComputationPrecision(enum.IntEnum):
  """A computational precision for a Meridian model.

  This mirrors the `ComputationPrecision` enum in
  `proto/mmm/v1/model/meridian/meridian_model.proto`.
  """

  COMPUTATION_PRECISION_UNSPECIFIED = 0
  FLOAT32 = 1
  FLOAT64 = 2


_DEFAULT_BACKEND = Backend.TENSORFLOW


def _initialize_backend() -> Backend:
  """Initializes the backend based on environment variables or defaults."""
  env_backend_str = os.environ.get("MERIDIAN_BACKEND")

  if not env_backend_str:
    return _DEFAULT_BACKEND

  try:
    backend = Backend(env_backend_str.lower())
    return backend
  except ValueError:
    warnings.warn(
        (
            "Invalid MERIDIAN_BACKEND environment variable:"
            f" '{env_backend_str}'. Supported values are 'tensorflow' and"
            f" 'jax'. Defaulting to {_DEFAULT_BACKEND.value}."
        ),
        RuntimeWarning,
    )
    return _DEFAULT_BACKEND


_TRUTHY_JAX_X64_VALUES = ("1", "true")


_BACKEND = _initialize_backend()

if _BACKEND == Backend.JAX:
  _enable_jax_x64_str = os.environ.get("MERIDIAN_ENABLE_JAX_X64", "false")
  if _enable_jax_x64_str.lower() in _TRUTHY_JAX_X64_VALUES:
    import jax  # pylint: disable=g-import-not-at-top,unused-import # pytype: disable=import-error

    jax.config.update("jax_enable_x64", True)


def set_backend(backend: Union[Backend, str]) -> None:
  """Sets the backend for Meridian.

  **Warning:** This function should ideally be called at the beginning of your
  program, before any other Meridian modules are imported or used.

  Changing the backend after Meridian's functions or classes have been
  imported can lead to unpredictable behavior. This is because already-imported
  modules will not reflect the backend change.

  Changing the backend at runtime requires reloading the `meridian.backend`
  module for the changes to take effect globally.

  Args:
    backend: The backend to use, must be a member of the `Backend` enum or a
      valid string ('tensorflow', 'jax').

  Raises:
    ValueError: If the provided backend is not valid.
  """
  global _BACKEND

  if isinstance(backend, str):
    try:
      backend_enum = Backend(backend.lower())
    except ValueError as exc:
      raise ValueError(
          f"Invalid backend string '{backend}'. Must be one of: "
          f"{[b.value for b in Backend]}"
      ) from exc
  elif isinstance(backend, Backend):
    backend_enum = backend
  else:
    raise ValueError("Backend must be a Backend enum member or a string.")

  _BACKEND = backend_enum


def get_backend() -> Backend:
  """Returns the current backend for Meridian."""
  return _BACKEND
