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

"""Tests for the backend abstraction layer."""

# pylint: disable=g-import-not-at-top

import importlib

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from meridian import backend
from meridian.backend import config
from meridian.backend import test_utils
import numpy as np
import tensorflow as tf


class BackendTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._original_backend = config.get_backend()

  def tearDown(self):
    super().tearDown()
    config.set_backend(self._original_backend)
    importlib.reload(backend)

  @parameterized.named_parameters(
      ("tensorflow", config.Backend.TENSORFLOW),
      ("jax", config.Backend.JAX),
  )
  def test_set_backend(self, backend_name):
    config.set_backend(backend_name)
    importlib.reload(backend)

    self.assertEqual(config.get_backend(), backend_name)

    ops_name = getattr(backend.ops, "__name__", "")
    if backend_name == config.Backend.JAX:
      self.assertIn("jax", ops_name)
    else:
      self.assertIn("tensorflow", ops_name)

  def test_invalid_backend(self):
    with self.assertRaises(ValueError):
      config.set_backend("invalid_backend")

  def test_set_backend_to_jax_raises_warning(self):
    with self.assertWarns(UserWarning) as cm:
      config.set_backend(config.Backend.JAX)
    self.assertIn(
        "under development and is not yet functional", str(cm.warning)
    )

  def test_set_random_seed_raises_for_jax(self):
    config.set_backend(config.Backend.JAX)
    importlib.reload(backend)
    with self.assertRaises(NotImplementedError):
      backend.set_random_seed(0)

  @parameterized.named_parameters(
      ("numpy_int32", np.int32, "int32"),
      ("tf_float64", tf.float64, "float64"),
      ("jax_bfloat16", jnp.bfloat16, "bfloat16"),
      # We use np.dtype().name to ensure the test is platform-agnostic.
      ("python_int", int, np.dtype(int).name),
      ("python_float", float, np.dtype(float).name),
      ("string", "float32", "float32"),
      ("none_type", None, "None"),
  )
  def test_standardize_dtype(self, dtype_in, expected_str):
    self.assertEqual(backend.standardize_dtype(dtype_in), expected_str)

  @parameterized.named_parameters(
      dict(testcase_name="no_args", types=[], expected="int64"),
      dict(testcase_name="only_int", types=[int, np.int32], expected="int64"),
      dict(
          testcase_name="only_float",
          types=[float, np.float64],
          expected="float32",
      ),
      dict(
          testcase_name="mixed_int_float",
          types=[int, float],
          expected="float32",
      ),
      dict(
          testcase_name="mixed_np_int_float",
          types=[np.int32, np.float32],
          expected="float32",
      ),
      dict(
          testcase_name="mixed_tf_int_float",
          types=[tf.int32, tf.float64],
          expected="float32",
      ),
      dict(
          testcase_name="mixed_jax_int_float",
          types=[jnp.int64, jnp.float32],
          expected="float32",
      ),
      dict(
          testcase_name="with_none",
          types=[int, None, float],
          expected="float32",
      ),
      dict(testcase_name="only_none", types=[None, None], expected="int64"),
  )
  def test_result_type(self, types, expected):
    self.assertEqual(backend.result_type(*types), expected)

  @parameterized.named_parameters(
      ("tensorflow", config.Backend.TENSORFLOW),
      ("jax", config.Backend.JAX),
  )
  def test_to_tensor_from_list(self, backend_name):
    config.set_backend(backend_name)
    importlib.reload(backend)

    py_list = [1.0, 2.0, 3.0]
    list_tensor = backend.to_tensor(py_list)

    if backend_name == config.Backend.JAX:
      import jax

      self.assertIsInstance(list_tensor, jax.Array)
      self.assertEqual(list_tensor.dtype, jnp.float32)

      tensor_f64 = backend.to_tensor(py_list, dtype=jnp.float64)
      # JAX will downcast to float32 by default.
      self.assertEqual(tensor_f64.dtype, jnp.float32)
    else:
      self.assertIsInstance(list_tensor, tf.Tensor)
      self.assertEqual(list_tensor.dtype, tf.float32)

      tensor_f64 = backend.to_tensor(py_list, dtype=tf.float64)
      self.assertEqual(tensor_f64.dtype, tf.float64)

  @parameterized.named_parameters(
      ("tensorflow", config.Backend.TENSORFLOW),
      ("jax", config.Backend.JAX),
  )
  def test_to_tensor_from_numpy(self, backend_name):
    config.set_backend(backend_name)
    importlib.reload(backend)

    np_array = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    np_tensor = backend.to_tensor(np_array)

    if backend_name == config.Backend.JAX:
      import jax

      self.assertIsInstance(np_tensor, jax.Array)
      # JAX downcasts float64 NumPy arrays to float32 by default
      self.assertEqual(np_tensor.dtype, jnp.float32)
    else:
      self.assertIsInstance(np_tensor, tf.Tensor)
      self.assertEqual(np_tensor.dtype, tf.float64)

  _concatenate_test_cases = [
      dict(
          testcase_name="axis_0",
          tensors_in=[[[1, 2], [3, 4]], [[5, 6]]],
          kwargs={"axis": 0},
          expected=np.array([[1, 2], [3, 4], [5, 6]]),
      ),
      dict(
          testcase_name="axis_1",
          tensors_in=[[[1, 2], [3, 4]], [[5], [7]]],
          kwargs={"axis": 1},
          expected=np.array([[1, 2, 5], [3, 4, 7]]),
      ),
      dict(
          testcase_name="1d_tensors",
          tensors_in=[[1, 2], [3, 4]],
          kwargs={"axis": 0},
          expected=np.array([1, 2, 3, 4]),
      ),
  ]

  @parameterized.product(
      backend_name=[config.Backend.TENSORFLOW, config.Backend.JAX],
      test_case=_concatenate_test_cases,
  )
  def test_concatenate(self, backend_name, test_case):
    config.set_backend(backend_name)
    importlib.reload(backend)
    tensors = [backend.to_tensor(t) for t in test_case["tensors_in"]]
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.concatenate(tensors, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _arange_test_cases = [
      dict(
          testcase_name="stop_only_defaults_to_int64",
          args=[5],
          kwargs={},
          expected=np.array([0, 1, 2, 3, 4], dtype=np.int64),
      ),
      dict(
          testcase_name="start_and_stop_defaults_to_int64",
          args=[2, 6],
          kwargs={},
          expected=np.array([2, 3, 4, 5], dtype=np.int64),
      ),
      dict(
          testcase_name="start_stop_and_step_defaults_to_int64",
          args=[1, 10, 2],
          kwargs={},
          expected=np.array([1, 3, 5, 7, 9], dtype=np.int64),
      ),
      dict(
          testcase_name="with_dtype_int16",
          args=[3],
          kwargs={"dtype": np.int16},
          expected=np.array([0, 1, 2], dtype=np.int16),
      ),
      dict(
          testcase_name="with_float_input_defaults_to_float32",
          args=[5.0],
          kwargs={},
          expected=np.arange(5.0, dtype=np.float32),
      ),
      dict(
          testcase_name="explicit_dtype_tf",
          args=[3],
          kwargs={"dtype": tf.float32},
          expected=np.array([0.0, 1.0, 2.0], dtype=np.float32),
      ),
  ]

  @parameterized.product(
      backend_name=[config.Backend.TENSORFLOW, config.Backend.JAX],
      test_case=_arange_test_cases,
  )
  def test_arange(self, backend_name, test_case):
    config.set_backend(backend_name)
    importlib.reload(backend)

    args = test_case["args"]
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    # JAX disables 64-bit precision by default and will silently downcast.
    if backend_name == config.Backend.JAX:
      if expected.dtype == np.int64:
        expected = expected.astype(np.int32)
      elif expected.dtype == np.float64:
        expected = expected.astype(np.float32)

    result = backend.arange(*args, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

    self.assertEqual(
        backend.standardize_dtype(result.dtype),
        backend.standardize_dtype(expected.dtype),
    )


if __name__ == "__main__":
  absltest.main()
