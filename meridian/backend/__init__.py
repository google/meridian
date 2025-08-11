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
from typing import Any, Optional, TYPE_CHECKING

from meridian.backend import config
import numpy as np
from typing_extensions import Literal

# The conditional imports in this module are a deliberate design choice for the
# backend abstraction layer. The TFP-on-JAX substrate provides a nearly
# identical API to the standard TFP library, making an alias-based approach more
# pragmatic than a full Abstract Base Class implementation, which would require
# extensive boilerplate.
# pylint: disable=g-import-not-at-top,g-bad-import-order

if TYPE_CHECKING:
  import jax as _jax
  import tensorflow as _tf


def standardize_dtype(dtype: Any) -> str:
  """Converts a backend-specific dtype to a standard string representation.

  Args:
    dtype: A backend-specific dtype object (e.g., tf.DType, np.dtype).

  Returns:
    A canonical string representation of the dtype (e.g., 'float32').
  """

  # Handle None explicitly, as np.dtype(None) defaults to float64.

  if dtype is None:
    return str(None)

  if hasattr(dtype, "as_numpy_dtype"):
    dtype = dtype.as_numpy_dtype

  try:
    return np.dtype(dtype).name
  except TypeError:
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
  standardized_types = []
  for t in types:
    if t is None:
      continue
    try:
      # Standardize the input type before checking promotion rules.
      standardized_types.append(standardize_dtype(t))
    except Exception:  # pylint: disable=broad-except
      # Fallback if standardization fails for an unexpected type.
      standardized_types.append(str(t))

  if any("float" in t for t in standardized_types):
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
    return standardize_dtype(dtype)

  input_types = [
      getattr(arg, "dtype", type(arg)) for arg in args if arg is not None
  ]
  return result_type(*input_types)


# --- Private Backend-Specific Implementations ---


def _jax_arange(
    start: Any,
    stop: Optional[Any] = None,
    step: Any = 1,
    dtype: Optional[Any] = None,
) -> "_jax.Array":
  """JAX implementation for arange."""

  # Import locally to make the function self-contained.

  import jax.numpy as jnp

  resolved_dtype = _resolve_dtype(dtype, start, stop, step)
  return jnp.arange(start, stop, step=step, dtype=resolved_dtype)


def _tf_arange(
    start: Any,
    stop: Optional[Any] = None,
    step: Any = 1,
    dtype: Optional[Any] = None,
) -> "_tf.Tensor":
  """TensorFlow implementation for arange."""
  import tensorflow as tf

  resolved_dtype = _resolve_dtype(dtype, start, stop, step)
  try:
    return tf.range(start, limit=stop, delta=step, dtype=resolved_dtype)
  except tf.errors.NotFoundError:
    result = tf.range(start, limit=stop, delta=step, dtype=tf.float32)
    return tf.cast(result, resolved_dtype)


def _jax_cast(x: Any, dtype: Any) -> "_jax.Array":
  """JAX implementation for cast."""
  return x.astype(dtype)


def _jax_divide_no_nan(x, y):
  """JAX implementation for divide_no_nan."""
  import jax.numpy as jnp

  return jnp.where(y != 0, jnp.divide(x, y), 0.0)


def _jax_numpy_function(*args, **kwargs):  # pylint: disable=unused-argument
  raise NotImplementedError(
      "backend.numpy_function is not implemented for the JAX backend."
  )


# --- Backend Initialization ---
_BACKEND = config.get_backend()

# We expose standardized functions directly at the module level (backend.foo)
# to provide a consistent, NumPy-like API across backends. The 'ops' object
# remains available for accessing the full, raw backend library if necessary,
# but usage should prefer the top-level standardized functions.

if _BACKEND == config.Backend.JAX:
  import jax
  import jax.numpy as jax_ops
  import tensorflow_probability.substrates.jax as tfp_jax

  class _JaxErrors:
    ResourceExhaustedError = MemoryError  # pylint: disable=invalid-name

  ops = jax_ops
  errors = _JaxErrors()
  Tensor = jax.Array
  tfd = tfp_jax.distributions
  bijectors = tfp_jax.bijectors
  experimental = tfp_jax.experimental
  random = tfp_jax.random
  _convert_to_tensor = ops.asarray

  # Standardized Public API
  arange = _jax_arange
  concatenate = ops.concatenate
  stack = ops.stack
  zeros = ops.zeros
  zeros_like = ops.zeros_like
  ones = ops.ones
  ones_like = ops.ones_like
  repeat = ops.repeat
  tile = ops.tile
  where = ops.where
  transpose = ops.transpose
  broadcast_to = ops.broadcast_to
  cast = _jax_cast
  expand_dims = ops.expand_dims

  einsum = ops.einsum
  exp = ops.exp
  log = ops.log
  cumsum = ops.cumsum
  reduce_sum = ops.sum
  reduce_mean = ops.mean
  reduce_std = ops.std
  reduce_any = ops.any
  is_nan = ops.isnan
  divide = ops.divide
  divide_no_nan = _jax_divide_no_nan
  numpy_function = _jax_numpy_function

  float32 = ops.float32
  bool_ = ops.bool_
  newaxis = ops.newaxis

  function = jax.jit
  allclose = ops.allclose

  def set_random_seed(seed: int) -> None:  # pylint: disable=unused-argument
    raise NotImplementedError(
        "JAX does not support a global, stateful random seed. `set_random_seed`"
        " is not implemented. Instead, you must pass an explicit `seed`"
        " integer directly to the sampling methods (e.g., `sample_prior`),"
        " which will be used to create a JAX PRNGKey internally."
    )

elif _BACKEND == config.Backend.TENSORFLOW:
  import tensorflow as tf_backend
  import tensorflow_probability as tfp

  ops = tf_backend
  errors = ops.errors

  Tensor = tf_backend.Tensor

  tfd = tfp.distributions
  bijectors = tfp.bijectors
  experimental = tfp.experimental
  random = tfp.random

  _convert_to_tensor = tf_backend.convert_to_tensor
  arange = _tf_arange
  concatenate = ops.concat
  stack = ops.stack
  zeros = ops.zeros
  zeros_like = ops.zeros_like
  ones = ops.ones
  ones_like = ops.ones_like
  repeat = ops.repeat
  tile = ops.tile
  where = ops.where
  transpose = ops.transpose
  broadcast_to = ops.broadcast_to
  cast = ops.cast
  expand_dims = ops.expand_dims

  einsum = ops.einsum
  exp = ops.math.exp
  log = ops.math.log
  cumsum = ops.cumsum
  reduce_sum = ops.reduce_sum
  reduce_mean = ops.reduce_mean
  reduce_std = ops.math.reduce_std
  reduce_any = ops.reduce_any
  is_nan = ops.math.is_nan
  divide = ops.divide
  divide_no_nan = ops.math.divide_no_nan
  numpy_function = ops.numpy_function

  float32 = ops.float32
  bool_ = ops.bool
  newaxis = ops.newaxis

  function = ops.function
  allclose = ops.experimental.numpy.allclose

  set_random_seed = tf_backend.keras.utils.set_random_seed

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
