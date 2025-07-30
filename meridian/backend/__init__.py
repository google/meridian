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

"""Backend Abstraction Layer for Meridian."""

import os

from typing import Any, Optional

from meridian.backend import config
from typing_extensions import Literal


# The conditional imports in this module are a deliberate design choice for the
# backend abstraction layer. The TFP-on-JAX substrate provides a nearly
# identical API to the standard TFP library, making an alias-based approach more
# pragmatic than a full Abstract Base Class implementation, which would require
# extensive boilerplate.
# pylint: disable=g-import-not-at-top,g-bad-import-order


def standardize_dtype(dtype: Any) -> str:
  """Converts a backend-specific dtype to a standard string representation.

  Args:
    dtype: A backend-specific dtype object (e.g., tf.DType, np.dtype).

  Returns:
    A canonical string representation of the dtype (e.g., 'float32').
  """
  if hasattr(dtype, "name"):
    return dtype.name
  return str(dtype)


def result_type(*types: Any) -> str:
  """Infers the result dtype from a list of input types, backend-agnostically.

  This acts as the single source of truth for type promotion rules. The
  promotion logic is designed to be consistent across all backends.

  Rule: If any input is a float, the result is float32. Otherwise, the result
  is int64 to match NumPy/JAX's default behavior for precision.

  Args:
    *types: A variable number of type objects (e.g., `<class 'int'>`,
      np.dtype('float32')).

  Returns:
    A string representing the promoted dtype.
  """
  if any("float" in str(t) for t in types if t is not None):
    return "float32"
  return "int64"


def _resolve_dtype(dtype: Optional[Any], *args: Any) -> str:
  """Resolves the final dtype for an operation.

  If a dtype is explicitly provided, it's returned. Otherwise, it infers the
  dtype from the input arguments using the backend-agnostic `result_type`
  promotion rules.

  Args:
    dtype: The user-provided dtype, which may be None.
    *args: The input arguments to the operation, used for dtype inference.

  Returns:
    A string representing the resolved dtype.
  """
  if dtype is not None:
    return dtype

  input_types = [
      getattr(arg, "dtype", type(arg)) for arg in args if arg is not None
  ]
  return result_type(*input_types)


_BACKEND = config.get_backend()

if _BACKEND == config.Backend.JAX:
  import jax
  import jax.numpy as ops
  import tensorflow_probability.substrates.jax as tfp_jax

  Tensor = jax.Array
  tfd = tfp_jax.distributions
  _convert_to_tensor = ops.asarray

  # --- Explicit Function Implementations for JAX ---
  concatenate = ops.concatenate

  def arange(
      start: Any,
      stop: Optional[Any] = None,
      step: Any = 1,
      dtype: Optional[Any] = None,
  ) -> Tensor:
    """Creates a 1-D tensor of evenly spaced values within a given interval.

    This function provides a backend-agnostic interface that ensures consistent
    default dtype inference between JAX and TensorFlow.

    Args:
      start: The start of the interval.
      stop: The end of the interval. If None, the interval is `[0, start)`.
      step: The spacing between values.
      dtype: The desired data type of the resulting tensor. If None, the dtype
        is inferred from the input arguments to be consistent across backends.

    Returns:
      A 1-D tensor of evenly spaced values.
    """
    resolved_dtype = _resolve_dtype(dtype, start, stop, step)
    return ops.arange(start, stop, step=step, dtype=resolved_dtype)

elif _BACKEND == config.Backend.TENSORFLOW:
  import tensorflow as tf
  import tensorflow_probability as tfp

  ops = tf
  Tensor = tf.Tensor
  tfd = tfp.distributions
  _convert_to_tensor = tf.convert_to_tensor

  # --- Explicit Function Implementations for TensorFlow ---
  concatenate = ops.concat

  def arange(
      start: Any,
      stop: Optional[Any] = None,
      step: Any = 1,
      dtype: Optional[Any] = None,
  ) -> Tensor:
    """Creates a 1-D tensor of evenly spaced values within a given interval.

    This function provides a backend-agnostic interface that ensures consistent
    default dtype inference between JAX and TensorFlow. It includes a fallback
    mechanism for dtypes not natively supported by TensorFlow's `range` op.

    Args:
      start: The start of the interval.
      stop: The end of the interval. If None, the interval is `[0, start)`.
      step: The spacing between values.
      dtype: The desired data type of the resulting tensor. If None, the dtype
        is inferred from the input arguments to be consistent across backends.

    Returns:
      A 1-D tensor of evenly spaced values.
    """
    resolved_dtype = _resolve_dtype(dtype, start, stop, step)
    try:
      return ops.range(start, limit=stop, delta=step, dtype=resolved_dtype)
    except tf.errors.NotFoundError:
      result = ops.range(start, limit=stop, delta=step, dtype=tf.float32)
      return tf.cast(result, resolved_dtype)

else:
  raise ValueError(f"Unsupported backend: {_BACKEND}")
# pylint: enable=g-import-not-at-top,g-bad-import-order


def to_tensor(data: Any, dtype: Optional[Any] = None) -> Tensor:  # type: ignore
  """Converts input data to the currently active backend tensor type.

  Args:
    data: The data to convert.
    dtype: The desired data type of the resulting tensor. The accepted types
      depend on the active backend (e.g., jax.numpy.dtype or tf.DType).

  Returns:
    A tensor representation of the data for the active backend.
  """

  return _convert_to_tensor(data, dtype=dtype)
