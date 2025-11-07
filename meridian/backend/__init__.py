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

import abc
import functools
import os
from typing import Any, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import warnings
from meridian.backend import config
import numpy as np
from typing_extensions import Literal


# The conditional imports in this module are a deliberate design choice for the
# backend abstraction layer. The TFP-on-JAX substrate provides a nearly
# identical API to the standard TFP library, making an alias-based approach more
# pragmatic than a full Abstract Base Class implementation, which would require
# extensive boilerplate.
# pylint: disable=g-import-not-at-top,g-bad-import-order

_DEFAULT_FLOAT = "float32"
_DEFAULT_INT = "int64"

_TENSORFLOW_TILE_KEYWORD = "multiples"
_JAX_TILE_KEYWORD = "reps"

_ARG_AUTOGRAPH = "autograph"
_ARG_JIT_COMPILE = "jit_compile"
_ARG_STATIC_ARGNUMS = "static_argnums"
_ARG_STATIC_ARGNAMES = "static_argnames"

_DEFAULT_SEED_DTYPE = "int32"
_MAX_INT32 = np.iinfo(np.int32).max

if TYPE_CHECKING:
  import dataclasses
  import jax as _jax
  import tensorflow as _tf

  TensorShapeInstance = Union[_tf.TensorShape, Tuple[int, ...]]

SeedType = Any


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
    return _DEFAULT_FLOAT
  return _DEFAULT_INT


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
def _jax_stabilize_rf_roi_grid(
    spend_grid: np.ndarray,
    outcome_grid: np.ndarray,
    n_rf_channels: int,
) -> np.ndarray:
  """Stabilizes the RF ROI grid for JAX using a stable index lookup."""
  new_outcome_grid = outcome_grid.copy()
  rf_slice = slice(-n_rf_channels, None)
  rf_spend_grid = spend_grid[:, rf_slice]
  rf_outcome_grid = new_outcome_grid[:, rf_slice]

  last_valid_indices = np.sum(~np.isnan(rf_spend_grid), axis=0) - 1
  channel_indices = np.arange(n_rf_channels)

  ref_spend = rf_spend_grid[last_valid_indices, channel_indices]
  ref_outcome = rf_outcome_grid[last_valid_indices, channel_indices]

  rf_roi = np.divide(
      ref_outcome,
      ref_spend,
      out=np.zeros_like(ref_outcome, dtype=np.float64),
      where=(ref_spend != 0),
  )

  new_outcome_grid[:, rf_slice] = rf_roi * rf_spend_grid
  return new_outcome_grid


def _tf_stabilize_rf_roi_grid(
    spend_grid: np.ndarray,
    outcome_grid: np.ndarray,
    n_rf_channels: int,
) -> np.ndarray:
  """Stabilizes the RF ROI grid for TF using nanmax logic."""
  new_outcome_grid = outcome_grid.copy()
  rf_slice = slice(-n_rf_channels, None)
  rf_outcome_max = np.nanmax(new_outcome_grid[:, rf_slice], axis=0)
  rf_spend_max = np.nanmax(spend_grid[:, rf_slice], axis=0)

  rf_roi = np.divide(
      rf_outcome_max,
      rf_spend_max,
      out=np.zeros_like(rf_outcome_max, dtype=np.float64),
      where=(rf_spend_max != 0),
  )

  new_outcome_grid[:, rf_slice] = rf_roi * spend_grid[:, rf_slice]
  return new_outcome_grid


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


def _jax_function_wrapper(func=None, **kwargs):
  """A wrapper for jax.jit that handles TF-like args and static args.

  This wrapper provides compatibility with TensorFlow's `tf.function` arguments
  and improves ergonomics when decorating class methods in JAX.

  By default, if neither `static_argnums` nor `static_argnames` are provided, it
  defaults `static_argnums` to `(0,)`. This assumes the function is a method
  where the first argument (`self` or `cls`) should be treated as static.

  To disable this behavior for plain functions, explicitly provide an empty
  tuple: `@backend.function(static_argnums=())`.

  Args:
    func: The function to wrap.
    **kwargs: Keyword arguments passed to jax.jit. TF-specific arguments (like
      `jit_compile`) are ignored.

  Returns:
    The wrapped function or a decorator.
  """
  jit_kwargs = kwargs.copy()

  jit_kwargs.pop(_ARG_JIT_COMPILE, None)
  jit_kwargs.pop(_ARG_AUTOGRAPH, None)

  if _ARG_STATIC_ARGNUMS in jit_kwargs:
    if not jit_kwargs[_ARG_STATIC_ARGNUMS]:
      jit_kwargs.pop(_ARG_STATIC_ARGNUMS)
  else:
    jit_kwargs[_ARG_STATIC_ARGNUMS] = (0,)

  decorator = functools.partial(jax.jit, **jit_kwargs)

  if func:
    return decorator(func)
  return decorator


def _tf_function_wrapper(func=None, **kwargs):
  """A wrapper for tf.function that ignores JAX-specific arguments."""
  import tensorflow as tf

  kwargs.pop(_ARG_STATIC_ARGNAMES, None)
  kwargs.pop(_ARG_STATIC_ARGNUMS, None)

  decorator = tf.function(**kwargs)

  if func:
    return decorator(func)
  return decorator


def _jax_nanmedian(a, axis=None):
  """JAX implementation for nanmedian."""
  import jax.numpy as jnp

  return jnp.nanmedian(a, axis=axis)


def _tf_nanmedian(a, axis=None):
  """TensorFlow implementation for nanmedian using numpy_function."""
  import tensorflow as tf

  return tf.numpy_function(
      lambda x: np.nanmedian(x, axis=axis).astype(x.dtype), [a], a.dtype
  )


def _jax_numpy_function(*args, **kwargs):  # pylint: disable=unused-argument
  raise NotImplementedError(
      "backend.numpy_function is not implemented for the JAX backend."
  )


def _jax_make_tensor_proto(*args, **kwargs):  # pylint: disable=unused-argument
  raise NotImplementedError(
      "backend.make_tensor_proto is not implemented for the JAX backend."
  )


def _jax_make_ndarray(*args, **kwargs):  # pylint: disable=unused-argument
  raise NotImplementedError(
      "backend.make_ndarray is not implemented for the JAX backend."
  )


def _jax_get_indices_where(condition):
  """JAX implementation for get_indices_where."""
  import jax.numpy as jnp

  return jnp.stack(jnp.where(condition), axis=-1)


def _tf_get_indices_where(condition):
  """TensorFlow implementation for get_indices_where."""
  import tensorflow as tf

  return tf.where(condition)


def _jax_split(value, num_or_size_splits, axis=0):
  """JAX implementation for split that accepts size splits."""
  import jax.numpy as jnp

  if not isinstance(num_or_size_splits, int):
    indices = jnp.cumsum(jnp.array(num_or_size_splits))[:-1]
    return jnp.split(value, indices, axis=axis)

  return jnp.split(value, num_or_size_splits, axis=axis)


def _jax_tile(*args, **kwargs):
  """JAX wrapper for tile that supports the `multiples` keyword argument."""
  import jax.numpy as jnp

  if _TENSORFLOW_TILE_KEYWORD in kwargs:
    kwargs[_JAX_TILE_KEYWORD] = kwargs.pop(_TENSORFLOW_TILE_KEYWORD)
  return jnp.tile(*args, **kwargs)


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


def _jax_boolean_mask(tensor, mask, axis=None):
  """JAX implementation for boolean_mask that supports an axis argument."""
  import jax.numpy as jnp

  if axis is None:
    axis = 0
  mask = jnp.asarray(mask)
  tensor_swapped = jnp.moveaxis(tensor, axis, 0)
  masked = tensor_swapped[mask]
  return jnp.moveaxis(masked, 0, axis)


def _tf_boolean_mask(tensor, mask, axis=None):
  """TensorFlow implementation for boolean_mask."""
  import tensorflow as tf

  return tf.boolean_mask(tensor, mask, axis=axis)


def _jax_gather(params, indices, axis=0):
  """JAX implementation for gather with axis support."""
  import jax.numpy as jnp

  if isinstance(params, (list, tuple)):
    params = np.array(params)

  # JAX can't JIT-compile operations on string or object arrays. We detect
  # these types and fall back to standard NumPy operations.
  if isinstance(params, np.ndarray) and params.dtype.kind in ("S", "U", "O"):
    return np.take(params, np.asarray(indices), axis=axis)

  return jnp.take(params, indices, axis=axis)


def _tf_gather(params, indices, axis=0):
  """TensorFlow implementation for gather."""
  import tensorflow as tf

  return tf.gather(params, indices, axis=axis)


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


def _jax_transpose(a, perm=None):
  """JAX wrapper for transpose to support the 'perm' keyword argument."""
  import jax.numpy as jnp

  return jnp.transpose(a, axes=perm)


def _jax_get_seed_data(seed: Any) -> Optional[np.ndarray]:
  """Extracts the underlying numerical data from a JAX PRNGKey."""
  if seed is None:
    return None

  return np.array(jax.random.key_data(seed))


def _tf_get_seed_data(seed: Any) -> Optional[np.ndarray]:
  """Converts a TensorFlow-style seed into a NumPy array."""
  if seed is None:
    return None
  return np.array(seed)


def _jax_convert_to_tensor(data, dtype=None):
  """Converts data to a JAX array, handling strings as NumPy arrays."""
  # JAX does not natively support string tensors in the same way TF does.
  # If a string dtype is requested, or if the data is inherently strings,
  # we fall back to a standard NumPy array.
  if dtype == np.str_ or (
      dtype is None
      and isinstance(data, (list, np.ndarray))
      and np.array(data).dtype.kind in ("S", "U")
  ):
    return np.array(data, dtype=np.str_)
  return jax_ops.asarray(data, dtype=dtype)


def _tf_nanmean(a, axis=None, keepdims=False):
  import tensorflow.experimental.numpy as tnp

  return tf_backend.convert_to_tensor(
      tnp.nanmean(a, axis=axis, keepdims=keepdims)
  )


def _tf_nansum(a, axis=None, keepdims=False):
  import tensorflow.experimental.numpy as tnp

  return tf_backend.convert_to_tensor(
      tnp.nansum(a, axis=axis, keepdims=keepdims)
  )


def _tf_nanvar(a, axis=None, keepdims=False):
  """Calculates variance ignoring NaNs, strictly returning a Tensor."""
  import tensorflow as tf
  import tensorflow.experimental.numpy as tnp
  # We implement two-pass variance to correctly handle NaNs and ensure
  # all operations remain within the TF graph (maintaining differentiability).
  a_tensor = tf.convert_to_tensor(a)
  mean = tnp.nanmean(a_tensor, axis=axis, keepdims=True)
  sq_diff = tf.math.squared_difference(a_tensor, mean)
  var = tnp.nanmean(sq_diff, axis=axis, keepdims=keepdims)
  return tf.convert_to_tensor(var)


def _jax_one_hot(*args, **kwargs):  # pylint: disable=unused-argument
  """JAX implementation for one_hot."""
  raise NotImplementedError(
      "backend.one_hot is not implemented for the JAX backend."
  )


def _jax_roll(*args, **kwargs):  # pylint: disable=unused-argument
  """JAX implementation for roll."""
  raise NotImplementedError(
      "backend.roll is not implemented for the JAX backend."
  )


def _jax_enable_op_determinism():
  """No-op for JAX. Determinism is handled via stateless PRNGKeys."""
  warnings.warn(
      "op determinism is a TensorFlow-specific concept and has no effect when"
      " using the JAX backend."
  )


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
  from jax import tree_util

  class ExtensionType:
    """A JAX-compatible stand-in for tf.experimental.ExtensionType.

    This class registers itself as a JAX Pytree node, allowing it to be passed
    through JIT-compiled functions.
    """

    def __init_subclass__(cls, **kwargs):
      super().__init_subclass__(**kwargs)
      tree_util.register_pytree_node(
          cls,
          cls._tree_flatten,
          cls._tree_unflatten,
      )

    def _tree_flatten(self):
      """Flattens the object for JAX tracing.

      Fields containing JAX arrays (or convertibles) are treated as children.
      Fields containing strings or NumPy string arrays must be treated as
      auxiliary data because JAX cannot trace non-numeric/non-boolean data.

      Returns:
        A tuple of (children, aux_data), where children are the
        tracer-compatible parts of the object, and aux_data contains auxiliary
        information needed for unflattening.
      """
      d = vars(self)
      all_keys = sorted(d.keys())
      children = []
      aux = {}
      children_keys = []

      for k in all_keys:
        v = d[k]
        # Identify string data to prevent JAX tracing errors.
        # 'S' is zero-terminated bytes (fixed-width), 'U' is unicode string.
        is_numpy_string = isinstance(v, np.ndarray) and v.dtype.kind in (
            "S",
            "U",
        )
        is_plain_string = isinstance(v, str)

        if is_numpy_string or is_plain_string:
          aux[k] = v
        else:
          children.append(v)
          children_keys.append(k)
      return children, (aux, children_keys)

    @classmethod
    def _tree_unflatten(cls, aux_and_keys, children):
      aux, children_keys = aux_and_keys
      obj = cls.__new__(cls)
      vars(obj).update(aux)
      for k, v in zip(children_keys, children):
        setattr(obj, k, v)
      return obj

  class _JaxErrors:
    # pylint: disable=invalid-name
    ResourceExhaustedError = MemoryError
    InvalidArgumentError = ValueError
    # pylint: enable=invalid-name

  class _JaxRandom:
    """Provides JAX-based random number generation utilities.

    This class mirrors the structure needed by `RNGHandler` for JAX.
    """

    @staticmethod
    def prng_key(seed):
      return jax.random.PRNGKey(seed)

    @staticmethod
    def split(key):
      return jax.random.split(key)

    @staticmethod
    def generator_from_seed(seed):
      raise NotImplementedError("JAX backend does not use Generators.")

    @staticmethod
    def stateless_split(seed: Any, num: int = 2):
      raise NotImplementedError(
          "Direct stateless splitting from an integer seed is not the primary"
          " pattern used in the JAX backend. Use `backend.random.split(key)`"
          " instead."
      )

    @staticmethod
    def stateless_randint(key, shape, minval, maxval, dtype=jax_ops.int32):
      """Wrapper for jax.random.randint."""
      return jax.random.randint(
          key, shape, minval=minval, maxval=maxval, dtype=dtype
      )

    @staticmethod
    def stateless_uniform(
        key, shape, dtype=jax_ops.float32, minval=0.0, maxval=1.0
    ):
      """Replacement for tfp_jax.random.stateless_uniform using jax.random.uniform."""
      return jax.random.uniform(
          key, shape=shape, dtype=dtype, minval=minval, maxval=maxval
      )

    @staticmethod
    def sanitize_seed(seed):
      return tfp_jax.random.sanitize_seed(seed)

  random = _JaxRandom()

  @functools.partial(
      jax.jit,
      static_argnames=[
          "joint_dist",
          "n_chains",
          "n_draws",
          "num_adaptation_steps",
          "dual_averaging_kwargs",
          "max_tree_depth",
          "unrolled_leapfrog_steps",
          "parallel_iterations",
      ],
  )
  def _jax_xla_windowed_adaptive_nuts(**kwargs):
    """JAX-specific JIT wrapper for the NUTS sampler."""
    kwargs["seed"] = random.prng_key(kwargs["seed"])
    return experimental.mcmc.windowed_adaptive_nuts(**kwargs)

  xla_windowed_adaptive_nuts = _jax_xla_windowed_adaptive_nuts

  _ops = jax_ops
  errors = _JaxErrors()
  Tensor = jax.Array
  tfd = tfp_jax.distributions
  bijectors = tfp_jax.bijectors
  experimental = tfp_jax.experimental
  mcmc = tfp_jax.mcmc
  _convert_to_tensor = _jax_convert_to_tensor

  # Standardized Public API
  absolute = _ops.abs
  allclose = _ops.allclose
  arange = _jax_arange
  argmax = _jax_argmax
  boolean_mask = _jax_boolean_mask
  broadcast_dynamic_shape = _jax_broadcast_dynamic_shape
  broadcast_to = _ops.broadcast_to
  cast = _jax_cast
  concatenate = _ops.concatenate
  cumsum = _ops.cumsum
  divide = _ops.divide
  divide_no_nan = _jax_divide_no_nan
  einsum = _ops.einsum
  enable_op_determinism = _jax_enable_op_determinism
  equal = _ops.equal
  exp = _ops.exp
  expand_dims = _ops.expand_dims
  fill = _jax_fill
  function = _jax_function_wrapper
  gather = _jax_gather
  get_indices_where = _jax_get_indices_where
  get_seed_data = _jax_get_seed_data
  is_nan = _ops.isnan
  log = _ops.log
  make_ndarray = _jax_make_ndarray
  make_tensor_proto = _jax_make_tensor_proto
  nanmean = jax_ops.nanmean
  nanmedian = _jax_nanmedian
  nansum = jax_ops.nansum
  nanvar = jax_ops.nanvar
  numpy_function = _jax_numpy_function
  one_hot = _jax_one_hot
  ones = _ops.ones
  ones_like = _ops.ones_like
  rank = _ops.ndim
  reduce_any = _ops.any
  reduce_max = _ops.max
  reduce_mean = _ops.mean
  reduce_min = _ops.min
  reduce_std = _ops.std
  reduce_sum = _ops.sum
  repeat = _ops.repeat
  reshape = _ops.reshape
  roll = _jax_roll
  split = _jax_split
  stack = _ops.stack
  tile = _jax_tile
  transpose = _jax_transpose
  unique_with_counts = _jax_unique_with_counts
  where = _ops.where
  zeros = _ops.zeros
  zeros_like = _ops.zeros_like

  float32 = _ops.float32
  bool_ = _ops.bool_
  newaxis = _ops.newaxis
  TensorShape = _jax_tensor_shape
  int32 = _ops.int32
  string = np.str_

  stabilize_rf_roi_grid = _jax_stabilize_rf_roi_grid

  def set_random_seed(seed: int) -> None:  # pylint: disable=unused-argument
    warnings.warn(
        "backend.set_random_seed is a no-op in JAX. Randomness is managed "
        "statelessly via PRNGKeys passed to sampling methods."
    )

elif _BACKEND == config.Backend.TENSORFLOW:
  import tensorflow as tf_backend
  import tensorflow_probability as tfp

  _ops = tf_backend
  errors = _ops.errors

  Tensor = tf_backend.Tensor
  ExtensionType = _ops.experimental.ExtensionType

  class _TfRandom:
    """Provides TensorFlow-based random number generation utilities.

    This class mirrors the structure needed by `RNGHandler` for TensorFlow.
    """

    @staticmethod
    def prng_key(seed):
      raise NotImplementedError(
          "TensorFlow backend does not use explicit PRNG keys for"
          " standard sampling."
      )

    @staticmethod
    def split(key):
      raise NotImplementedError(
          "TensorFlow backend does not implement explicit key splitting."
      )

    @staticmethod
    def generator_from_seed(seed):
      return tf_backend.random.Generator.from_seed(seed)

    @staticmethod
    def stateless_split(
        seed: "tf_backend.Tensor", num: int = 2
    ) -> "tf_backend.Tensor":
      return tf_backend.random.experimental.stateless_split(seed, num=num)

    @staticmethod
    def stateless_randint(
        seed: "tf_backend.Tensor",
        shape: "TensorShapeInstance",
        minval: int,
        maxval: int,
        dtype: Any = _DEFAULT_SEED_DTYPE,
    ) -> "tf_backend.Tensor":
      return tf_backend.random.stateless_uniform(
          shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype
      )

    @staticmethod
    def stateless_uniform(
        seed, shape, minval=0, maxval=None, dtype=tf_backend.float32
    ):
      return tf_backend.random.stateless_uniform(
          shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype
      )

    @staticmethod
    def sanitize_seed(seed):
      return tfp.random.sanitize_seed(seed)

  random = _TfRandom()

  @tf_backend.function(autograph=False, jit_compile=True)
  def _tf_xla_windowed_adaptive_nuts(**kwargs):
    """TensorFlow-specific XLA wrapper for the NUTS sampler."""
    return experimental.mcmc.windowed_adaptive_nuts(**kwargs)

  xla_windowed_adaptive_nuts = _tf_xla_windowed_adaptive_nuts

  tfd = tfp.distributions
  bijectors = tfp.bijectors
  experimental = tfp.experimental
  mcmc = tfp.mcmc
  _convert_to_tensor = _ops.convert_to_tensor

  # Standardized Public API
  absolute = _ops.math.abs
  allclose = _ops.experimental.numpy.allclose
  arange = _tf_arange
  argmax = _tf_argmax
  boolean_mask = _tf_boolean_mask
  broadcast_dynamic_shape = _ops.broadcast_dynamic_shape
  broadcast_to = _ops.broadcast_to
  cast = _ops.cast
  concatenate = _ops.concat
  cumsum = _ops.cumsum
  divide = _ops.divide
  divide_no_nan = _ops.math.divide_no_nan
  einsum = _ops.einsum
  enable_op_determinism = _ops.config.experimental.enable_op_determinism
  equal = _ops.equal
  exp = _ops.math.exp
  expand_dims = _ops.expand_dims
  fill = _tf_fill
  function = _tf_function_wrapper
  gather = _tf_gather
  get_indices_where = _tf_get_indices_where
  get_seed_data = _tf_get_seed_data
  is_nan = _ops.math.is_nan
  log = _ops.math.log
  make_ndarray = _ops.make_ndarray
  make_tensor_proto = _ops.make_tensor_proto
  nanmean = _tf_nanmean
  nanmedian = _tf_nanmedian
  nansum = _tf_nansum
  nanvar = _tf_nanvar
  numpy_function = _ops.numpy_function
  one_hot = _ops.one_hot
  ones = _ops.ones
  ones_like = _ops.ones_like
  rank = _ops.rank
  reduce_any = _ops.reduce_any
  reduce_max = _ops.reduce_max
  reduce_mean = _ops.reduce_mean
  reduce_min = _ops.reduce_min
  reduce_std = _ops.math.reduce_std
  reduce_sum = _ops.reduce_sum
  repeat = _ops.repeat
  reshape = _ops.reshape
  roll = _ops.roll
  set_random_seed = tf_backend.keras.utils.set_random_seed
  split = _ops.split
  stack = _ops.stack
  tile = _ops.tile
  transpose = _ops.transpose
  unique_with_counts = _tf_unique_with_counts
  where = _ops.where
  zeros = _ops.zeros
  zeros_like = _ops.zeros_like

  stabilize_rf_roi_grid = _tf_stabilize_rf_roi_grid

  float32 = _ops.float32
  bool_ = _ops.bool
  newaxis = _ops.newaxis
  TensorShape = _ops.TensorShape
  int32 = _ops.int32
  string = _ops.string

else:
  raise ValueError(f"Unsupported backend: {_BACKEND}")
# pylint: enable=g-import-not-at-top,g-bad-import-order


def _extract_int_seed(s: Any) -> Optional[int]:
  """Attempts to extract a scalar Python integer from various input types.

  Args:
    s: The input seed, which can be an int, Tensor, or array-like object.

  Returns:
    A Python integer if a scalar integer can be extracted, otherwise None.
  """
  try:
    if isinstance(s, int):
      return s

    value = np.asarray(s)

    if value.ndim == 0 and np.issubdtype(value.dtype, np.integer):
      return int(value)
  # A broad exception is used here because the input `s` can be of many types
  # (e.g., JAX PRNGKey) which may cause np.asarray or other operations to fail
  # in unpredictable ways. The goal is to safely attempt extraction and fail
  # gracefully.
  except Exception:  # pylint: disable=broad-except
    return None
  return None


class _BaseRNGHandler(abc.ABC):
  """A backend-agnostic abstract base class for random number generation state.

  This handler provides a stateful-style interface for consuming randomness,
  abstracting away the differences between JAX's stateless PRNG keys and
  TensorFlow's paradigms.

  Attributes:
    _seed_input: The original seed object provided during initialization.
    _int_seed: A Python integer extracted from the seed, if possible.
  """

  def __init__(self, seed: SeedType):
    """Initializes the RNG handler.

    Args:
      seed: The initial seed. The accepted type depends on the backend. For JAX,
        this must be an integer. For TensorFlow, this can be an integer, a
        sequence of two integers, or a Tensor. If None, the handler becomes a
        no-op, returning None for all seed requests.
    """
    self._seed_input = seed
    self._int_seed: Optional[int] = _extract_int_seed(seed)

  @abc.abstractmethod
  def get_kernel_seed(self) -> Any:
    """Provides a backend-appropriate sanitized seed/key for an MCMC kernel.

    This method exposes the current state of the handler in the format expected
    by the backend's MCMC machinery. It does not advance the internal state.

    Returns:
      A backend-specific seed object (e.g., a JAX PRNGKey or a TF Tensor).
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_next_seed(self) -> Any:
    """Provides the appropriate seed object for the next sequential operation.

    This is primarily used for prior sampling and typically advances the
    internal state.

    Returns:
      A backend-specific seed object for a single random operation.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def advance_handler(self) -> "_BaseRNGHandler":
    """Creates a new, independent RNGHandler for a subsequent operation.

    This method is used to generate a new handler derived deterministically
    from the current handler's state.

    Returns:
      A new, independent `RNGHandler` instance.
    """
    raise NotImplementedError


class _JaxRNGHandler(_BaseRNGHandler):
  """JAX implementation of the RNGHandler using explicit key splitting."""

  def __init__(self, seed: SeedType):
    """Initializes the JAX RNG handler.

    Args:
      seed: The initial seed, which must be a Python integer, a scalar integer
        Tensor/array, or None.

    Raises:
      ValueError: If the provided seed is not a scalar integer or None.
    """
    super().__init__(seed)
    self._key: Optional["_jax.Array"] = None

    if seed is None:
      return

    if (
        isinstance(seed, jax.Array)  # pylint: disable=undefined-variable
        and seed.shape == (2,)
        and seed.dtype == jax_ops.uint32  # pylint: disable=undefined-variable
    ):
      self._key = seed
      return

    if self._int_seed is None:
      raise ValueError(
          "JAX backend requires a seed that is an integer or a scalar array,"
          f" but got: {type(seed)} with value {seed!r}"
      )

    self._key = random.prng_key(self._int_seed)

  def get_kernel_seed(self) -> Any:
    if self._key is None:
      return None
    _, subkey = random.split(self._key)
    return int(jax.random.randint(subkey, (), 0, 2**31 - 1))  # pylint: disable=undefined-variable

  def get_next_seed(self) -> Any:
    if self._key is None:
      return None
    self._key, subkey = random.split(self._key)
    return subkey

  def advance_handler(self) -> "_JaxRNGHandler":
    if self._key is None:
      return _JaxRNGHandler(None)

    self._key, subkey_for_new_handler = random.split(self._key)
    new_seed_tensor = random.stateless_randint(
        key=subkey_for_new_handler,
        shape=(),
        minval=0,
        maxval=_MAX_INT32,
        dtype=_DEFAULT_SEED_DTYPE,
    )
    return _JaxRNGHandler(np.asarray(new_seed_tensor).item())


class _TFRNGHandler(_BaseRNGHandler):
  """A stateless-style RNG handler for TensorFlow.

  This handler canonicalizes any seed input into a single stateless seed Tensor.
  """

  def __init__(self, seed: SeedType):
    """Initializes the TensorFlow RNG handler.

    Args:
      seed: The initial seed. Can be an integer, a sequence of two integers, a
        corresponding Tensor, or None. It will be sanitized and stored
        internally as a single stateless seed Tensor.
    """
    super().__init__(seed)
    self._seed_state: Optional["_tf.Tensor"] = None

    if seed is None:
      return

    if isinstance(seed, Sequence) and len(seed) != 2:
      raise ValueError(
          "Invalid seed: Must be either a single integer (stateful seed) or a"
          " pair of two integers (stateless seed). See"
          " [tfp.random.sanitize_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed)"
          " for details."
      )

    if isinstance(seed, int):
      seed_to_sanitize = (seed, seed)
    else:
      seed_to_sanitize = seed

    self._seed_state = random.sanitize_seed(seed_to_sanitize)

  def get_kernel_seed(self) -> Any:
    """Provides the current stateless seed of the handler without advancing it."""
    return self._seed_state

  def get_next_seed(self) -> Any:
    """Returns a new unique seed and advances the handler's internal state."""
    if self._seed_state is None:
      return None
    new_seed, next_state = random.stateless_split(self._seed_state)
    self._seed_state = next_state
    return new_seed

  def advance_handler(self) -> "_TFRNGHandler":
    """Creates a new handler from a new seed and advances the current handler."""
    if self._seed_state is None:
      return _TFRNGHandler(None)
    seed_for_new_handler, next_state = random.stateless_split(self._seed_state)
    self._seed_state = next_state
    return _TFRNGHandler(seed_for_new_handler)


if _BACKEND == config.Backend.JAX:
  RNGHandler = _JaxRNGHandler
elif _BACKEND == config.Backend.TENSORFLOW:
  RNGHandler = _TFRNGHandler
else:
  raise ImportError(f"RNGHandler not implemented for backend: {_BACKEND}")


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
