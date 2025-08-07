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

from typing import Any
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
