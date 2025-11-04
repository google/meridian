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

from typing import Any, Optional
from absl.testing import parameterized
from meridian import backend
from meridian.backend import config
import numpy as np

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
