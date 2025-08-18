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


def _jax_get_indices_where(condition):
  """JAX implementation for get_indices_where."""
  import jax.numpy as jnp

  return jnp.stack(jnp.where(condition), axis=-1)


def _tf_get_indices_where(condition):
  """TensorFlow implementation for get_indices_where."""
  import tensorflow as tf

  return tf.where(condition)


def _jax_unique_with_counts(x):
  """JAX implementation for unique_with_counts."""
  import jax.numpy as jnp

  y, counts = jnp.unique(x, return_counts=True)
  # The TF version returns a tuple of (y, idx, count). The idx is not used in
  # the calling code, so we can return None for it to maintain tuple structure.
  return y, None, counts


def _tf_unique_with_counts(x):
  """TensorFlow implementation for unique_with_counts."""
  import tensorflow as tf

  return tf.unique_with_counts(x)


def _jax_boolean_mask(tensor, mask):
  """JAX implementation for boolean_mask."""
  # JAX uses standard array indexing for boolean masking.
  return tensor[mask]


def _tf_boolean_mask(tensor, mask):
  """TensorFlow implementation for boolean_mask."""
  import tensorflow as tf

  return tf.boolean_mask(tensor, mask)


def _jax_gather(params, indices):
  """JAX implementation for gather."""
  # JAX uses standard array indexing for gather operations.
  return params[indices]


def _tf_gather(params, indices):
  """TensorFlow implementation for gather."""
  import tensorflow as tf

  return tf.gather(params, indices)


def _jax_fill(dims, value):
  """JAX implementation for fill."""
  import jax.numpy as jnp

  return jnp.full(dims, value)


def _tf_fill(dims, value):
  """TensorFlow implementation for fill."""
  import tensorflow as tf

  return tf.fill(dims, value)


def _jax_argmax(tensor, axis=None):
  """JAX implementation for argmax, aligned with TensorFlow's default.

  This function finds the indices of the maximum values along a specified axis.
  Crucially, it mimics the default behavior of TensorFlow's `tf.argmax`, where
  if `axis` is `None`, the operation defaults to `axis=0`. This differs from
  NumPy's and JAX's native `argmax` behavior, which would flatten the array
  before finding the index.

  Args:
    tensor: The input JAX array.
    axis: An integer specifying the axis along which to find the index of the
      maximum value. If `None`, it defaults to `0` to match TensorFlow's
      behavior.

  Returns:
    A JAX array containing the indices of the maximum values.
  """
  import jax.numpy as jnp

  if axis is None:
    axis = 0
  return jnp.argmax(tensor, axis=axis)


def _tf_argmax(tensor, axis=None):
  """TensorFlow implementation for argmax."""
  import tensorflow as tf

  return tf.argmax(tensor, axis=axis)


def _jax_broadcast_dynamic_shape(shape_x, shape_y):
  """JAX implementation for broadcast_dynamic_shape."""
  import jax.numpy as jnp

  return jnp.broadcast_shapes(shape_x, shape_y)


def _jax_tensor_shape(dims):
  """JAX implementation for TensorShape."""
  if isinstance(dims, int):
    return (dims,)

  return tuple(dims)


# --- Backend Initialization ---
_BACKEND = config.get_backend()

# We expose standardized functions directly at the module level (backend.foo)
# to provide a consistent, NumPy-like API across backends. The '_ops' object
# is a private member for accessing the full, raw backend library if necessary,
# but usage should prefer the top-level standardized functions.

if _BACKEND == config.Backend.JAX:
  import jax
  import jax.numpy as jax_ops
  import tensorflow_probability.substrates.jax as tfp_jax

  class _JaxErrors:
    # pylint: disable=invalid-name
    ResourceExhaustedError = MemoryError
    InvalidArgumentError = ValueError
    # pylint: enable=invalid-name

  _ops = jax_ops
  errors = _JaxErrors()
  Tensor = jax.Array
  tfd = tfp_jax.distributions
  bijectors = tfp_jax.bijectors
  experimental = tfp_jax.experimental
  random = tfp_jax.random
  _convert_to_tensor = _ops.asarray

  # Standardized Public API
  arange = _jax_arange
  concatenate = _ops.concatenate
  stack = _ops.stack
  split = _ops.split
  zeros = _ops.zeros
  zeros_like = _ops.zeros_like
  ones = _ops.ones
  ones_like = _ops.ones_like
  repeat = _ops.repeat
  reshape = _ops.reshape
  tile = _ops.tile
  where = _ops.where
  transpose = _ops.transpose
  broadcast_to = _ops.broadcast_to
  broadcast_dynamic_shape = _jax_broadcast_dynamic_shape
  cast = _jax_cast
  expand_dims = _ops.expand_dims
  fill = _jax_fill
  gather = _jax_gather
  boolean_mask = _jax_boolean_mask
  equal = _ops.equal
  get_indices_where = _jax_get_indices_where

  einsum = _ops.einsum
  exp = _ops.exp
  log = _ops.log
  absolute = _ops.abs
  argmax = _jax_argmax
  cumsum = _ops.cumsum
  reduce_sum = _ops.sum
  reduce_mean = _ops.mean
  reduce_std = _ops.std
  reduce_any = _ops.any
  reduce_min = _ops.min
  reduce_max = _ops.max
  is_nan = _ops.isnan
  divide = _ops.divide
  divide_no_nan = _jax_divide_no_nan
  unique_with_counts = _jax_unique_with_counts
  numpy_function = _jax_numpy_function

  float32 = _ops.float32
  bool_ = _ops.bool_
  newaxis = _ops.newaxis
  TensorShape = _jax_tensor_shape

  function = jax.jit
  allclose = _ops.allclose

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

  _ops = tf_backend
  errors = _ops.errors

  Tensor = tf_backend.Tensor

  tfd = tfp.distributions
  bijectors = tfp.bijectors
  experimental = tfp.experimental
  random = tfp.random

  _convert_to_tensor = _ops.convert_to_tensor
  arange = _tf_arange
  concatenate = _ops.concat
  stack = _ops.stack
  split = _ops.split
  zeros = _ops.zeros
  zeros_like = _ops.zeros_like
  ones = _ops.ones
  ones_like = _ops.ones_like
  repeat = _ops.repeat
  reshape = _ops.reshape
  tile = _ops.tile
  where = _ops.where
  transpose = _ops.transpose
  broadcast_to = _ops.broadcast_to
  broadcast_dynamic_shape = _ops.broadcast_dynamic_shape
  cast = _ops.cast
  expand_dims = _ops.expand_dims
  fill = _tf_fill
  gather = _tf_gather
  boolean_mask = _tf_boolean_mask
  equal = _ops.equal
  get_indices_where = _tf_get_indices_where

  einsum = _ops.einsum
  exp = _ops.math.exp
  log = _ops.math.log
  absolute = _ops.math.abs
  argmax = _tf_argmax
  cumsum = _ops.cumsum
  reduce_sum = _ops.reduce_sum
  reduce_mean = _ops.reduce_mean
  reduce_std = _ops.math.reduce_std
  reduce_any = _ops.reduce_any
  reduce_min = _ops.reduce_min
  reduce_max = _ops.reduce_max
  is_nan = _ops.math.is_nan
  divide = _ops.divide
  divide_no_nan = _ops.math.divide_no_nan
  unique_with_counts = _tf_unique_with_counts
  numpy_function = _ops.numpy_function

  float32 = _ops.float32
  bool_ = _ops.bool
  newaxis = _ops.newaxis
  TensorShape = _ops.TensorShape

  function = _ops.function
  allclose = _ops.experimental.numpy.allclose

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
