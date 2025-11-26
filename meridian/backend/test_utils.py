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

"""Common testing utilities for Meridian, designed to be backend-agnostic."""

import dataclasses
from typing import Any, Optional

from absl.testing import parameterized
from google.protobuf import descriptor
from google.protobuf import message
from meridian import backend
from meridian.backend import config
import numpy as np

from tensorflow.python.util.protobuf import compare
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import tensor_pb2


# pylint: enable=g-direct-tensorflow-import

FieldDescriptor = descriptor.FieldDescriptor

# A type alias for backend-agnostic array-like objects.
# We use `Any` here to avoid circular dependencies with the backend module
# while still allowing the function to accept backend-specific tensor types.
ArrayLike = Any


def assert_allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    err_msg: str = "",
):
  """Backend-agnostic assertion to check if two array-like objects are close.

  This function converts both inputs to NumPy arrays before comparing them,
  making it compatible with TensorFlow Tensors, JAX Arrays, and standard
  Python lists or NumPy arrays.

  Args:
    a: The first array-like object to compare.
    b: The second array-like object to compare.
    rtol: The relative tolerance parameter.
    atol: The absolute tolerance parameter.
    err_msg: The error message to be printed in case of failure.

  Raises:
    AssertionError: If the two arrays are not equal within the given tolerance.
  """
  np.testing.assert_allclose(
      np.array(a), np.array(b), rtol=rtol, atol=atol, err_msg=err_msg
  )


def assert_allequal(a: ArrayLike, b: ArrayLike, err_msg: str = ""):
  """Backend-agnostic assertion to check if two array-like objects are equal.

  This function converts both inputs to NumPy arrays before comparing them.

  Args:
    a: The first array-like object to compare.
    b: The second array-like object to compare.
    err_msg: The error message to be printed in case of failure.

  Raises:
    AssertionError: If the two arrays are not equal.
  """
  np.testing.assert_array_equal(np.array(a), np.array(b), err_msg=err_msg)


def assert_deep_equals(
    test_case,
    obj1: Any,
    obj2: Any,
    msg: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
  """Recursive equality check handling Dataclasses, Lists, and Backend Tensors.

  Args:
    test_case: The unittest.TestCase instance (self) to use for assertions.
    obj1: The first object to compare.
    obj2: The second object to compare.
    msg: Optional error message prefix.
    rtol: Relative tolerance for float comparison.
    atol: Absolute tolerance for float comparison.
  """
  if obj1 is None or obj2 is None:
    test_case.assertEqual(obj1, obj2, msg=msg)
    return

  if (
      hasattr(obj1, "__array__")
      or hasattr(obj1, "numpy")
      or isinstance(obj1, (np.ndarray, backend.Tensor))
  ):
    arr1 = np.array(obj1)
    arr2 = np.array(obj2)

    # Check for non-numeric types where atol/rtol don't apply
    if arr1.dtype.kind in ("U", "S", "O", "b"):
      np.testing.assert_array_equal(arr1, arr2, err_msg=msg)
    else:
      np.testing.assert_allclose(arr1, arr2, err_msg=msg, rtol=rtol, atol=atol)
    return

  if dataclasses.is_dataclass(obj1):
    test_case.assertIs(
        type(obj1),
        type(obj2),
        msg=f"{msg} Type mismatch: {type(obj1)} vs {type(obj2)}",
    )
    for field in dataclasses.fields(obj1):
      val1 = getattr(obj1, field.name)
      val2 = getattr(obj2, field.name)
      assert_deep_equals(
          test_case,
          val1,
          val2,
          msg=f"{msg}.{field.name}",
          rtol=rtol,
          atol=atol,
      )
    return

  if isinstance(obj1, (list, tuple)):
    test_case.assertIsInstance(obj2, (list, tuple), msg=f"{msg} Type mismatch")
    test_case.assertEqual(len(obj1), len(obj2), msg=f"{msg} Length mismatch")
    for i, (item1, item2) in enumerate(zip(obj1, obj2)):
      assert_deep_equals(
          test_case, item1, item2, msg=f"{msg}[{i}]", rtol=rtol, atol=atol
      )
    return

  # Fallback to standard equality for primitives (int, str, float, etc.)
  test_case.assertEqual(obj1, obj2, msg=msg)


def assert_seed_allequal(a: Any, b: Any, err_msg: str = ""):
  """Backend-agnostic assertion to check if two seed objects are equal."""
  data_a = backend.get_seed_data(a)
  data_b = backend.get_seed_data(b)
  if data_a is None and data_b is None:
    return
  np.testing.assert_array_equal(data_a, data_b, err_msg=err_msg)


def assert_not_allequal(a: ArrayLike, b: ArrayLike, err_msg: str = ""):
  """Asserts that two objects are not element-wise equal."""
  np.testing.assert_(
      not np.array_equal(np.array(a), np.array(b)),
      msg=f"Arrays are unexpectedly equal.\n{err_msg}",
  )


def assert_seed_not_allequal(a: Any, b: Any, err_msg: str = ""):
  """Asserts that two seed objects are not element-wise equal."""
  data_a = backend.get_seed_data(a)
  data_b = backend.get_seed_data(b)
  if data_a is None and data_b is None:
    raise AssertionError(
        f"Seeds are unexpectedly equal (both are None). {err_msg}"
    )
  if data_a is None or data_b is None:
    return
  np.testing.assert_(
      not np.array_equal(data_a, data_b),
      msg=f"Seeds are unexpectedly equal.\n{err_msg}",
  )


def assert_all_finite(a: ArrayLike, err_msg: str = ""):
  """Backend-agnostic assertion to check if all elements in an array are finite.

  Args:
    a: The array-like object to check.
    err_msg: The error message to be printed in case of failure.

  Raises:
    AssertionError: If the array contains non-finite values.
  """
  if not np.all(np.isfinite(np.array(a))):
    raise AssertionError(err_msg or "Array contains non-finite values.")


def assert_all_non_negative(a: ArrayLike, err_msg: str = ""):
  """Backend-agnostic assertion to check if all elements are non-negative.

  Args:
    a: The array-like object to check.
    err_msg: The error message to be printed in case of failure.

  Raises:
    AssertionError: If the array contains negative values.
  """
  if not np.all(np.array(a) >= 0):
    raise AssertionError(err_msg or "Array contains negative values.")


# --- Proto Utilities ---
def normalize_tensor_protos(proto: message.Message):
  """Recursively normalizes TensorProto messages within a proto (In-place).

  This ensures a consistent serialization format across different backends
  (e.g., JAX vs TF) by repacking TensorProtos using the current backend's
  canonical method (backend.make_tensor_proto). This handles differences
  like using `bool_val` versus `tensor_content` for boolean tensors.

  Args:
    proto: The protobuf message object to normalize. This object is modified in
      place.
  """
  if not isinstance(proto, message.Message):
    return

  for desc, value in proto.ListFields():
    if desc.type != FieldDescriptor.TYPE_MESSAGE:
      continue

    # A map is defined as a repeated field whose message type has the
    # map_entry option set.
    is_map = (
        desc.label == FieldDescriptor.LABEL_REPEATED
        and desc.message_type.has_options
        and desc.message_type.GetOptions().map_entry
    )

    if is_map:
      for item in value.values():
        # Helper checks if values are scalars or messages.
        _process_message_for_normalization(item)

    elif desc.label == FieldDescriptor.LABEL_REPEATED:
      # Handle standard repeated message fields.
      for item in value:
        _process_message_for_normalization(item)
    else:
      # Handle singular message fields.
      _process_message_for_normalization(value)


def _process_message_for_normalization(msg: Any):
  """Helper to process a potential message during normalization traversal."""
  # Ensure we only process message objects.
  # If msg is a scalar (e.g., string from map<string, string>), stop recursion.
  if not isinstance(msg, message.Message):
    return

  if isinstance(msg, tensor_pb2.TensorProto):
    _repack_tensor_proto(msg)
  else:
    # If it's another message type, recurse into its fields.
    normalize_tensor_protos(msg)


def _repack_tensor_proto(tensor_proto: "tensor_pb2.TensorProto"):
  """Repacks a TensorProto in place to use a consistent serialization format."""
  if not tensor_proto.ByteSize():
    return

  try:
    data_array = backend.make_ndarray(tensor_proto)
  except Exception as e:
    raise ValueError(
        "Failed to deserialize TensorProto during normalization:"
        f" {e}\nProto content:\n{tensor_proto}"
    ) from e

  new_tensor_proto = backend.make_tensor_proto(data_array)

  tensor_proto.Clear()
  tensor_proto.CopyFrom(new_tensor_proto)


def assert_normalized_proto_equal(
    test_case: parameterized.TestCase,
    expected: message.Message,
    actual: message.Message,
    msg: Optional[str] = None,
    **kwargs: Any,
):
  """Compares two protos after normalizing TensorProto fields.

  Use this instead of compare.assertProtoEqual when protos contain tensors
  to ensure backend-agnostic comparison.

  Args:
    test_case: The TestCase instance (self).
    expected: The expected protobuf message.
    actual: The actual protobuf message.
    msg: An optional message to display on failure.
    **kwargs: Additional keyword arguments passed to assertProto2Equal (e.g.,
      precision).
  """
  # Work on copies to avoid mutating the original objects
  expected_copy = expected.__class__()
  expected_copy.CopyFrom(expected)
  actual_copy = actual.__class__()
  actual_copy.CopyFrom(actual)

  try:
    normalize_tensor_protos(expected_copy)
    normalize_tensor_protos(actual_copy)
  except ValueError as e:
    test_case.fail(f"Proto normalization failed: {e}. {msg}")

  compare.assertProtoEqual(
      test_case, expected_copy, actual_copy, msg=msg, **kwargs
  )


class MeridianTestCase(parameterized.TestCase):
  """Base test class for Meridian providing backend-aware utilities.

  This class handles initialization timing issues (crucial for JAX by forcing
  tensor operations into setUp) and provides a unified way to handle random
  number generation across backends (Stateful TF vs Stateless JAX).
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Enforce determinism for TensorFlow tests before any tests are run.
    # This is a no-op with a warning for the JAX backend.
    backend.enable_op_determinism()

  def setUp(self):
    super().setUp()
    # Default seed, can be overridden by subclasses before calling
    # _initialize_rng().
    self.seed = 42
    self._jax_key = None
    self._initialize_rng()

  def _initialize_rng(self):
    """Initializes the RNG state or key based on self.seed."""
    current_backend = config.get_backend()

    if current_backend == config.Backend.TENSORFLOW:
      # In TF, we use the global stateful seed for test reproducibility.
      try:
        backend.set_random_seed(self.seed)
      except NotImplementedError:
        # Handle cases where backend might be misconfigured during transition.
        pass
    elif current_backend == config.Backend.JAX:
      # In JAX, we must manage PRNGKeys explicitly.
      # Import JAX locally to avoid hard dependency if TF is the active backend,
      # and to ensure initialization happens after absltest.main() starts.
      # pylint: disable=g-import-not-at-top
      import jax
      # pylint: enable=g-import-not-at-top
      self._jax_key = jax.random.PRNGKey(self.seed)
    else:
      raise ValueError(f"Unknown backend: {current_backend}")

  def get_next_rng_seed_or_key(self) -> Optional[Any]:
    """Gets the next available seed or key for backend operations.

    This should be passed to the `seed` argument of TFP sampling methods.

    Returns:
      A JAX PRNGKey if the backend is JAX (splitting the internal key).
      None if the backend is TensorFlow (relying on the global state).
    """
    if self._jax_key is not None:
      # JAX requires splitting the key for each use.
      # pylint: disable=g-import-not-at-top
      import jax
      # pylint: enable=g-import-not-at-top
      self._jax_key, subkey = jax.random.split(self._jax_key)
      return subkey
    else:
      # For stateful TF, returning None allows TFP/TF to use the global seed.
      return None

  def sample(
      self,
      distribution: backend.tfd.Distribution,
      sample_shape: Any = (),
      **kwargs: Any,
  ) -> backend.Tensor:
    """Performs a backend-agnostic sample from a distribution.

    This method abstracts away the need for explicit seed management in JAX.
    When the JAX backend is active, it automatically provides a PRNGKey from
    the test's managed key state. In TensorFlow, it performs a standard sample.

    Args:
      distribution: The TFP distribution object to sample from.
      sample_shape: The shape of the desired sample.
      **kwargs: Additional keyword arguments to pass to the underlying `sample`
        method (e.g., `name`).

    Returns:
      A tensor containing the sampled values.
    """
    seed = self.get_next_rng_seed_or_key()
    return distribution.sample(sample_shape=sample_shape, seed=seed, **kwargs)
