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

import dataclasses
import importlib
import os
import sys
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from meridian import backend
from meridian.backend import config
from meridian.backend import test_utils
import numpy as np
import tensorflow as tf


class BackendInitializationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._original_environ = os.environ.copy()
    self._original_modules = sys.modules.copy()

    # Unload backend modules from Python's cache. This is to ensure that the
    # module's initialization logic is re-run for each test under different
    # environment variable conditions.
    modules_to_unload = ["meridian.backend.config", "meridian.backend"]
    for mod_name in modules_to_unload:
      if mod_name in sys.modules:
        del sys.modules[mod_name]

  def tearDown(self):
    super().tearDown()
    # Restore the original environment variables.
    os.environ.clear()
    os.environ.update(self._original_environ)

    # Restore the original modules.
    sys.modules.update(self._original_modules)

    # Reload the backend modules to a clean, default state for subsequent tests.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      importlib.reload(config)
      importlib.reload(backend)

  def _import_backend_modules(self):
    from meridian.backend import config as config_mod  # pylint: disable=reimported
    from meridian import backend as backend_mod  # pylint: disable=reimported

    return config_mod, backend_mod

  def test_default_backend(self):
    if "MERIDIAN_BACKEND" in os.environ:
      del os.environ["MERIDIAN_BACKEND"]

    config_mod, backend_mod = self._import_backend_modules()

    self.assertEqual(config_mod.get_backend(), config_mod.Backend.TENSORFLOW)
    self.assertIs(backend_mod.Tensor, tf.Tensor)

  @parameterized.named_parameters(
      ("lowercase", "tensorflow"),
      ("uppercase", "TENSORFLOW"),
  )
  def test_env_var_tensorflow(self, env_value):
    os.environ["MERIDIAN_BACKEND"] = env_value

    config_mod, backend_mod = self._import_backend_modules()

    self.assertEqual(config_mod.get_backend(), config_mod.Backend.TENSORFLOW)
    self.assertIs(backend_mod.Tensor, tf.Tensor)

  @parameterized.named_parameters(
      ("lowercase", "jax"),
      ("uppercase", "JAX"),
  )
  def test_env_var_jax(self, env_value):
    os.environ["MERIDIAN_BACKEND"] = env_value

    # We expect the UserWarning during import because JAX is selected
    with self.assertWarns(UserWarning) as cm:
      config_mod, backend_mod = self._import_backend_modules()

    self.assertIn("under development", str(cm.warning))
    self.assertEqual(config_mod.get_backend(), config_mod.Backend.JAX)
    self.assertIs(backend_mod.Tensor, jax.Array)

  def test_env_var_invalid(self):
    os.environ["MERIDIAN_BACKEND"] = "pytorch"

    with self.assertWarns(RuntimeWarning) as cm:
      config_mod, backend_mod = self._import_backend_modules()

    self.assertIn(
        "Invalid MERIDIAN_BACKEND environment variable: 'pytorch'",
        str(cm.warning),
    )
    self.assertEqual(config_mod.get_backend(), config_mod.Backend.TENSORFLOW)
    self.assertIs(backend_mod.Tensor, tf.Tensor)


_TF = config.Backend.TENSORFLOW.value
_JAX = config.Backend.JAX.value
_ALL_BACKENDS = [_TF, _JAX]


class BackendTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._original_backend = config.get_backend()

  def tearDown(self):
    super().tearDown()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      config.set_backend(self._original_backend)

    importlib.reload(backend)

  def _set_backend_for_test(self, backend_name: str):
    expected_backend = config.Backend(backend_name)

    if (
        expected_backend == config.Backend.JAX
        and config.get_backend() != config.Backend.JAX
    ):
      with self.assertWarns(UserWarning):
        config.set_backend(backend_name)
    else:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config.set_backend(backend_name)

    importlib.reload(backend)

  @parameterized.named_parameters(
      ("tensorflow_enum", _TF, _TF, True),
      ("jax_enum", _JAX, _JAX, True),
      ("tensorflow_str", _TF, _TF, False),
      ("jax_str_caps", "JAX", _JAX, False),
  )
  def test_set_backend(self, input_value, expected_str, is_enum_input):
    expected_backend = config.Backend(expected_str)

    if is_enum_input:
      backend_selection = config.Backend(input_value)
    else:
      backend_selection = input_value

    if (
        expected_backend == config.Backend.JAX
        and config.get_backend() != config.Backend.JAX
    ):
      with self.assertWarns(UserWarning):
        config.set_backend(backend_selection)
    else:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config.set_backend(backend_selection)

    importlib.reload(backend)

    self.assertEqual(config.get_backend(), expected_backend)

    if expected_backend == config.Backend.JAX:
      self.assertIs(backend.Tensor, jax.Array)
    else:
      self.assertIs(backend.Tensor, tf.Tensor)

  def test_invalid_backend_string(self):
    with self.assertRaisesRegex(ValueError, "Invalid backend string"):
      config.set_backend("invalid_backend")

  def test_invalid_backend_type(self):
    with self.assertRaisesRegex(
        ValueError, "Backend must be a Backend enum member or a string."
    ):
      config.set_backend(123)

  def test_set_backend_to_jax_raises_warning(self):
    self._set_backend_for_test(_TF)
    self._set_backend_for_test(_JAX)

  def test_set_backend_to_jax_idempotent_warning(self):
    self._set_backend_for_test(_JAX)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      config.set_backend(config.Backend.JAX)
      self.assertEmpty(w)

  def test_set_random_seed_warns_for_jax(self):
    self._set_backend_for_test(_JAX)
    with self.assertWarns(UserWarning) as cm:
      backend.set_random_seed(0)
    self.assertIn("is a no-op in JAX", str(cm.warning))

  @parameterized.named_parameters(
      ("numpy_int32", np.int32, "int32"),
      ("tf_float64", tf.float64, "float64"),
      ("jax_bfloat16", jnp.bfloat16, "bfloat16"),
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
      ("tensorflow", _TF),
      ("jax", _JAX),
  )
  def test_to_tensor_from_list(self, backend_name):
    self._set_backend_for_test(backend_name)

    py_list = [1.0, 2.0, 3.0]
    list_tensor = backend.to_tensor(py_list)

    if backend_name == _JAX:
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
      ("tensorflow", _TF),
      ("jax", _JAX),
  )
  def test_to_tensor_from_numpy(self, backend_name):
    self._set_backend_for_test(backend_name)

    np_array = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    np_tensor = backend.to_tensor(np_array)

    if backend_name == _JAX:
      self.assertIsInstance(np_tensor, jax.Array)
      # JAX downcasts float64 NumPy arrays to float32 by default
      self.assertEqual(np_tensor.dtype, jnp.float32)
    else:
      self.assertIsInstance(np_tensor, tf.Tensor)
      self.assertEqual(np_tensor.dtype, tf.float64)

  @parameterized.named_parameters(
      ("tensorflow", _TF),
      ("jax", _JAX),
  )
  def test_to_tensor_strings(self, backend_name):
    self._set_backend_for_test(backend_name)
    data = ["a", "b", "c"]
    t = backend.to_tensor(data, dtype=backend.string)

    if backend_name == _JAX:
      # JAX backend uses numpy unicode strings
      self.assertIsInstance(t, np.ndarray)
      self.assertEqual(t.dtype.kind, "U")
      test_utils.assert_allequal(t, np.array(data))
    else:
      # TensorFlow natively supports string tensors (bytes).
      self.assertIsInstance(t, tf.Tensor)
      self.assertEqual(t.dtype, tf.string)
      expected = np.array([b"a", b"b", b"c"], dtype=object)
      test_utils.assert_allequal(np.array(t).astype(object), expected)

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
      backend_name=_ALL_BACKENDS,
      test_case=_concatenate_test_cases,
  )
  def test_concatenate(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensors = [backend.to_tensor(t) for t in test_case["tensors_in"]]
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.concatenate(tensors, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _transpose_test_cases = [
      dict(
          testcase_name="2d_default_no_perm",
          tensor_in=np.array([[1, 2, 3], [4, 5, 6]]),
          kwargs={},
          expected=np.array([[1, 4], [2, 5], [3, 6]]),
      ),
      dict(
          testcase_name="3d_with_perm_keyword",
          tensor_in=np.arange(24).reshape((2, 3, 4)),
          kwargs={"perm": [2, 0, 1]},
          expected=np.transpose(
              np.arange(24).reshape((2, 3, 4)), axes=(2, 0, 1)
          ),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS, test_case=_transpose_test_cases
  )
  def test_transpose(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.transpose(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _broadcast_dynamic_shape_test_cases = [
      dict(
          testcase_name="broadcast_scalar_to_vector",
          shape_x=(1,),
          shape_y=(3,),
          expected=(3,),
      ),
      dict(
          testcase_name="broadcast_vector_to_matrix",
          shape_x=(3,),
          shape_y=(2, 3),
          expected=(2, 3),
      ),
      dict(
          testcase_name="broadcast_different_shapes",
          shape_x=(2, 1),
          shape_y=(1, 3),
          expected=(2, 3),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_broadcast_dynamic_shape_test_cases,
  )
  def test_broadcast_dynamic_shape(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    shape_x = test_case["shape_x"]
    shape_y = test_case["shape_y"]
    expected = test_case["expected"]

    result = backend.broadcast_dynamic_shape(shape_x, shape_y)

    test_utils.assert_allclose(result, expected)

  _tensor_shape_test_cases = [
      dict(testcase_name="from_int", dims=5, expected=(5,)),
      dict(testcase_name="from_list", dims=[2, 3], expected=(2, 3)),
      dict(testcase_name="from_tuple", dims=(4, 1), expected=(4, 1)),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_tensor_shape_test_cases,
  )
  def test_tensor_shape(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    dims = test_case["dims"]
    expected = test_case["expected"]

    result = backend.TensorShape(dims)

    test_utils.assert_allequal(result, expected)

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
      backend_name=_ALL_BACKENDS,
      test_case=_arange_test_cases,
  )
  def test_arange(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)

    args = test_case["args"]
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    # JAX disables 64-bit precision by default and will silently downcast.
    if backend_name == _JAX:
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

  _argmax_test_cases = [
      dict(
          testcase_name="1d",
          tensor_in=[1, 5, 2],
          kwargs={},
          expected=np.array(1),
      ),
      dict(
          testcase_name="2d_no_axis",
          tensor_in=[[1, 5, 2], [8, 3, 4]],
          kwargs={},
          expected=np.array([1, 0, 1]),
      ),
      dict(
          testcase_name="2d_axis_1",
          tensor_in=[[1, 5, 2], [8, 3, 4]],
          kwargs={"axis": 1},
          expected=np.array([1, 0]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_argmax_test_cases,
  )
  def test_argmax(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.argmax(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _fill_test_cases = [
      dict(
          testcase_name="simple_shape",
          kwargs={"dims": (2, 3), "value": 5.0},
          expected=np.full((2, 3), 5.0),
      ),
      dict(
          testcase_name="int_value",
          kwargs={"dims": [4], "value": 1},
          expected=np.full([4], 1),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_fill_test_cases,
  )
  def test_fill(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.fill(**kwargs)
    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _gather_test_cases = [
      dict(
          testcase_name="1d_tensor",
          tensor_in=[10, 20, 30, 40],
          indices=[0, 3, 1],
          kwargs={},
          expected=np.array([10, 40, 20]),
      ),
      dict(
          testcase_name="2d_tensor",
          tensor_in=[[1, 2], [3, 4], [5, 6]],
          indices=[2, 0],
          kwargs={},
          expected=np.array([[5, 6], [1, 2]]),
      ),
      dict(
          testcase_name="2d_tensor_axis_1",
          tensor_in=[[1, 2], [3, 4], [5, 6]],
          indices=[1, 0],
          kwargs={"axis": 1},
          expected=np.array([[2, 1], [4, 3], [6, 5]]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_gather_test_cases,
  )
  def test_gather(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    indices = backend.to_tensor(test_case["indices"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.gather(tensor, indices, **kwargs)
    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _boolean_mask_test_cases = [
      dict(
          testcase_name="1d",
          tensor_in=[1, 2, 3, 4],
          mask=[True, False, True, False],
          expected=np.array([1, 3]),
      ),
      dict(
          testcase_name="2d",
          tensor_in=[[1, 2], [3, 4]],
          mask=[[True, False], [False, True]],
          expected=np.array([1, 4]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_boolean_mask_test_cases,
  )
  def test_boolean_mask(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    mask = backend.to_tensor(test_case["mask"])
    expected = test_case["expected"]

    result = backend.boolean_mask(tensor, mask)
    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _boolean_mask_with_axis_test_cases = [
      dict(
          testcase_name="axis_0",
          tensor_in=[[1, 2], [3, 4], [5, 6]],
          mask=[True, False, True],
          axis=0,
          expected=np.array([[1, 2], [5, 6]]),
      ),
      dict(
          testcase_name="axis_1",
          tensor_in=[[1, 2, 3], [4, 5, 6]],
          mask=[False, True, True],
          axis=1,
          expected=np.array([[2, 3], [5, 6]]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_boolean_mask_with_axis_test_cases,
  )
  def test_boolean_mask_with_axis(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    mask = backend.to_tensor(test_case["mask"])
    axis = test_case["axis"]
    expected = test_case["expected"]

    result = backend.boolean_mask(tensor, mask, axis=axis)
    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="positional_arg",
          call=lambda t: backend.tile(t, [2]),
          expected=np.array([4, 5, 4, 5]),
      ),
      dict(
          testcase_name="keyword_arg_multiples",
          call=lambda t: backend.tile(t, multiples=[2]),
          expected=np.array([4, 5, 4, 5]),
      ),
  )
  def test_tile_signature_compatibility(self, call, expected):
    tiled_tensor = call(backend.to_tensor([4, 5]))
    test_utils.assert_allequal(tiled_tensor, expected)

  @parameterized.named_parameters(
      ("tensorflow", _TF),
      ("jax", _JAX),
  )
  def test_unique_with_counts(self, backend_name):
    self._set_backend_for_test(backend_name)
    tensor_in = backend.to_tensor([1, 2, 1, 3, 2, 1])
    expected_y = np.array([1, 2, 3])
    expected_counts = np.array([3, 2, 1])

    y, _, counts = backend.unique_with_counts(tensor_in)
    self.assertIsInstance(y, backend.Tensor)
    self.assertIsInstance(counts, backend.Tensor)
    test_utils.assert_allclose(y, expected_y)
    test_utils.assert_allclose(counts, expected_counts)

  _get_indices_where_test_cases = [
      dict(
          testcase_name="1d",
          condition=[True, False, True],
          expected=np.array([[0], [2]]),
      ),
      dict(
          testcase_name="2d",
          condition=[[True, False], [False, True]],
          expected=np.array([[0, 0], [1, 1]]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_get_indices_where_test_cases,
  )
  def test_get_indices_where(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    condition = backend.to_tensor(test_case["condition"])
    expected = test_case["expected"]

    result = backend.get_indices_where(condition)
    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _split_test_cases = [
      dict(
          testcase_name="by_num_sections_even",
          tensor_in=np.arange(6),
          split_arg=3,
          kwargs={"axis": 0},
          expected=[np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
      ),
      dict(
          testcase_name="by_sizes",
          tensor_in=np.arange(7),
          split_arg=[1, 3, 3],
          kwargs={"axis": 0},
          expected=[np.array([0]), np.array([1, 2, 3]), np.array([4, 5, 6])],
      ),
      dict(
          testcase_name="by_sizes_2d_axis1",
          tensor_in=np.arange(12).reshape(2, 6),
          split_arg=[2, 1, 3],
          kwargs={"axis": 1},
          expected=[
              np.array([[0, 1], [6, 7]]),
              np.array([[2], [8]]),
              np.array([[3, 4, 5], [9, 10, 11]]),
          ],
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_split_test_cases,
  )
  def test_split(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    split_arg = test_case["split_arg"]
    kwargs = test_case["kwargs"]
    expected_list = test_case["expected"]

    result_list = backend.split(tensor, split_arg, **kwargs)

    self.assertLen(result_list, len(expected_list))
    for result, expected in zip(result_list, expected_list):
      self.assertIsInstance(result, backend.Tensor)
      test_utils.assert_allclose(result, expected)

  def test_jax_extension_type_is_pytree(self):
    self._set_backend_for_test(_JAX)

    @dataclasses.dataclass
    class MyType(backend.ExtensionType):
      x: backend.Tensor
      y: int
      z: str

    obj = MyType(x=jnp.ones(3), y=5, z="hello")

    # Test flattening/unflattening implicitly via jit
    @jax.jit
    def f(input_obj):
      return input_obj.x * input_obj.y

    res = f(obj)
    test_utils.assert_allclose(res, jnp.ones(3) * 5)

  _nanmedian_test_cases = [
      dict(
          testcase_name="1d_with_nan",
          tensor_in=np.array([1.0, np.nan, 3.0, 5.0]),
          kwargs={},
          expected=np.array(3.0),
      ),
      dict(
          testcase_name="2d_axis_0",
          tensor_in=np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]]),
          kwargs={"axis": 0},
          expected=np.array([2.0, 15.0]),
      ),
      dict(
          testcase_name="2d_axis_1",
          tensor_in=np.array([[1.0, 10.0, np.nan], [np.nan, 20.0, 30.0]]),
          kwargs={"axis": 1},
          expected=np.array([5.5, 25.0]),
      ),
      dict(
          testcase_name="all_nan",
          tensor_in=np.array([np.nan, np.nan]),
          kwargs={},
          expected=np.array(np.nan),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_nanmedian_test_cases,
  )
  def test_nanmedian(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.nanmedian(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _nanmean_test_cases = [
      dict(
          testcase_name="1d_with_nan",
          tensor_in=np.array([1.0, np.nan, 3.0, 5.0]),
          kwargs={},
          expected=np.array(3.0),
      ),
      dict(
          testcase_name="2d_axis_0",
          tensor_in=np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]]),
          kwargs={"axis": 0},
          expected=np.array([2.0, 15.0]),
      ),
      dict(
          testcase_name="2d_axis_1_keepdims",
          tensor_in=np.array([[1.0, 10.0, np.nan], [np.nan, 20.0, 30.0]]),
          kwargs={"axis": 1, "keepdims": True},
          expected=np.array([[5.5], [25.0]]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_nanmean_test_cases,
  )
  def test_nanmean(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.nanmean(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _nansum_test_cases = [
      dict(
          testcase_name="1d_with_nan",
          tensor_in=np.array([1.0, np.nan, 3.0, 5.0]),
          kwargs={},
          expected=np.array(9.0),
      ),
      dict(
          testcase_name="2d_axis_0",
          tensor_in=np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]]),
          kwargs={"axis": 0},
          expected=np.array([4.0, 30.0]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_nansum_test_cases,
  )
  def test_nansum(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.nansum(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _nanvar_test_cases = [
      dict(
          testcase_name="1d_with_nan",
          tensor_in=np.array([1.0, np.nan, 3.0, 5.0]),
          kwargs={},
          expected=np.var([1.0, 3.0, 5.0]),
      ),
      dict(
          testcase_name="2d_axis_0",
          tensor_in=np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]]),
          kwargs={"axis": 0},
          expected=np.array([np.var([1.0, 3.0]), np.var([10.0, 20.0])]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_nanvar_test_cases,
  )
  def test_nanvar(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    tensor = backend.to_tensor(test_case["tensor_in"])
    kwargs = test_case["kwargs"]
    expected = test_case["expected"]

    result = backend.nanvar(tensor, **kwargs)

    self.assertIsInstance(result, backend.Tensor)
    test_utils.assert_allclose(result, expected)

  _stabilize_rf_roi_grid_test_cases = [
      dict(
          testcase_name="equivalent_case_max_outcome_on_last_row",
          spend_grid=np.array([
              [10.0, 100.0],
              [20.0, 200.0],
              [30.0, 300.0],
              [np.nan, 400.0],
          ]),
          outcome_grid=np.array([
              [1.0, 15.0],
              [2.0, 30.0],
              [2.7, 45.0],
              [np.nan, 60.0],
          ]),
          n_rf_channels=2,
          expected_tf=np.array([
              [0.9, 15.0],
              [1.8, 30.0],
              [2.7, 45.0],
              [np.nan, 60.0],
          ]),
          expected_jax=np.array([
              [0.9, 15.0],
              [1.8, 30.0],
              [2.7, 45.0],
              [np.nan, 60.0],
          ]),
      ),
      dict(
          testcase_name="divergent_case_max_outcome_not_on_last_row",
          #  The maximum outcome for the first channel (2.5) is
          #  on a different row than the maximum spend (30.0). This causes the
          #  TF logic to calculate an incorrect reference ROI, while the
          #  index-based JAX logic finds the ROI at the highest spend point.
          spend_grid=np.array([
              [10.0, 100.0],
              [20.0, 200.0],
              [30.0, 300.0],
              [np.nan, 400.0],
          ]),
          outcome_grid=np.array([
              [1.0, 15.0],
              [2.5, 30.0],
              [2.1, 45.0],
              [np.nan, 60.0],
          ]),
          n_rf_channels=2,
          expected_tf=np.array([
              [0.833333, 15.0],
              [1.666666, 30.0],
              [2.5, 45.0],
              [np.nan, 60.0],
          ]),
          expected_jax=np.array([
              [0.7, 15.0],
              [1.4, 30.0],
              [2.1, 45.0],
              [np.nan, 60.0],
          ]),
      ),
  ]

  @parameterized.product(
      backend_name=_ALL_BACKENDS,
      test_case=_stabilize_rf_roi_grid_test_cases,
  )
  def test_stabilize_rf_roi_grid(self, backend_name, test_case):
    self._set_backend_for_test(backend_name)
    spend_grid = test_case["spend_grid"]
    outcome_grid = test_case["outcome_grid"]
    n_rf_channels = test_case["n_rf_channels"]

    if backend_name == _JAX:
      expected = test_case["expected_jax"]
    else:
      expected = test_case["expected_tf"]

    result = backend.stabilize_rf_roi_grid(
        spend_grid, outcome_grid, n_rf_channels
    )

    # The function should return a new grid, not modify in-place.
    self.assertFalse(np.all(result == outcome_grid))
    test_utils.assert_allclose(result, expected, atol=1e-6)

  def test_jax_one_hot_raises_not_implemented(self):
    self._set_backend_for_test(_JAX)
    with self.assertRaises(NotImplementedError):
      backend.one_hot(indices=[0, 1], depth=2)

  def test_jax_roll_raises_not_implemented(self):
    self._set_backend_for_test(_JAX)
    with self.assertRaises(NotImplementedError):
      backend.roll(backend.to_tensor([1, 2, 3]), shift=1, axis=0)


class BackendFunctionWrappersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._original_backend = config.get_backend()

  def tearDown(self):
    super().tearDown()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      config.set_backend(self._original_backend)
    importlib.reload(backend)

  def _set_backend_for_test(self, backend_name: str):
    if config.get_backend().value != backend_name:
      if backend_name == "jax":
        with self.assertWarns(UserWarning):
          config.set_backend(backend_name)
      else:
        config.set_backend(backend_name)
      importlib.reload(backend)

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_function_wrapper_simple_decorator(self, backend_name):
    self._set_backend_for_test(backend_name)

    # Explicitly setting static_argnums=() ensures that JAX does not default
    # to making the first argument static, which is correct for this plain
    # function.
    @backend.function(static_argnums=())
    def add_one(x):
      return x + 1

    result = add_one(backend.to_tensor(5))
    test_utils.assert_allclose(result, 6)

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_function_wrapper_ignores_cross_backend_args(self, backend_name):
    self._set_backend_for_test(backend_name)

    # This tests that TF ignores JAX args (static_arg*) and JAX ignores TF args
    # (jit_compile). We must set static_argnums=() because this is a standalone
    # function.
    @backend.function(
        autograph=False,
        jit_compile=True,
        static_argnames="should_be_ignored_by_tf",
        static_argnums=(),
    )
    def add_one(
        x,
        should_be_ignored_by_tf=False,
    ):
      if should_be_ignored_by_tf:
        return x + 2
      return x + 1

    result = add_one(backend.to_tensor(5))
    test_utils.assert_allclose(result, 6)

  def test_jax_function_handles_static_args_on_methods(self):
    self._set_backend_for_test(_JAX)

    class MyProcessor:

      def __init__(self, increment):
        self._increment = increment

      @backend.function(static_argnames="as_static")
      def process(self, x, as_static: bool):
        if as_static:
          return x + self._increment
        else:
          return x - self._increment

    processor = MyProcessor(increment=10)
    result = processor.process(backend.to_tensor(5), as_static=True)
    test_utils.assert_allclose(result, 15)

  def test_jax_function_implicit_static_self(self):
    self._set_backend_for_test(_JAX)

    class SimpleAdder:

      def __init__(self, bias):
        self.bias = bias

      @backend.function
      def add_bias(self, x):
        return x + self.bias

    adder = SimpleAdder(bias=5.0)
    result = adder.add_bias(backend.to_tensor(10.0))
    test_utils.assert_allclose(result, 15.0)

    # Since self is static, changing an attribute is ignored by the compiled
    # function.
    adder.bias = 100.0
    result2 = adder.add_bias(backend.to_tensor(10.0))
    test_utils.assert_allclose(result2, 15.0)


class RNGHandlerTest(BackendTest):

  # Helper to compare JAX keys, as they cannot be converted to NumPy directly.
  def _assert_key_equal(self, key1, key2):
    if key1 is None and key2 is None:
      return
    self.assertIsNotNone(key1)
    self.assertIsNotNone(key2)
    data1 = jax.random.key_data(key1)
    data2 = jax.random.key_data(key2)
    test_utils.assert_allequal(data1, data2)

  def _assert_key_not_equal(self, key1, key2):
    if key1 is None or key2 is None:
      self.assertNotEqual(key1, key2)
      return
    data1 = jax.random.key_data(key1)
    data2 = jax.random.key_data(key2)
    self.assertFalse(np.array_equal(data1, data2))

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_correct_class_is_exposed(self, backend_name):
    """Verifies that the correct concrete RNGHandler class is exposed."""
    self._set_backend_for_test(backend_name)
    # We need to access the private classes for this check.
    # pylint: disable=protected-access
    if backend_name == _JAX:
      self.assertIs(backend.RNGHandler, backend._JaxRNGHandler)
    else:
      self.assertIs(backend.RNGHandler, backend._TFRNGHandler)
    # pylint: enable=protected-access

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_initialization_with_none_seed_is_noop(self, backend_name):
    """Verifies that a None seed creates a handler that returns None."""
    self._set_backend_for_test(backend_name)
    handler = backend.RNGHandler(None)

    self.assertIsNone(handler._seed_input)
    self.assertIsNone(handler.get_next_seed())
    self.assertIsNone(handler.get_kernel_seed())

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_initialization_with_integer_seed(self, backend_name):
    """Tests that the handler initializes its internal state correctly."""
    self._set_backend_for_test(backend_name)
    seed = 12345
    handler = backend.RNGHandler(seed)

    self.assertEqual(handler._seed_input, seed)
    self.assertEqual(handler._int_seed, seed)

    if backend_name == _JAX:
      expected_key = jax.random.PRNGKey(seed)
      # Kernel seed should be the sanitized version of the key
      expected_kernel_seed = backend.random.sanitize_seed(expected_key)
      self._assert_key_equal(handler.get_kernel_seed(), expected_kernel_seed)
    else:
      # TF Regression Safety: Must match (s, s) behavior.
      # The legacy handler explicitly converts int -> (int, int) before
      # sanitizing.
      expected_kernel_seed = backend.random.sanitize_seed((seed, seed))
      test_utils.assert_allequal(
          handler.get_kernel_seed(), expected_kernel_seed
      )

  def test_tf_initialization_with_sequence_seed(self):
    """Tests TF initialization with a stateless sequence seed."""
    self._set_backend_for_test(_TF)
    seed_seq = [42, 99]
    handler = backend.RNGHandler(seed_seq)

    self.assertEqual(handler._seed_input, seed_seq)
    self.assertIsNone(handler._int_seed)

    expected_kernel_seed = backend.random.sanitize_seed(seed_seq)
    test_utils.assert_allequal(handler.get_kernel_seed(), expected_kernel_seed)

  def test_tf_initialization_with_tensor_int_seed(self):
    """Tests TF initialization with a scalar EagerTensor containing an int."""
    self._set_backend_for_test(_TF)
    seed_val = 777
    seed_tensor = tf.constant(seed_val, dtype=tf.int32)
    handler = backend.RNGHandler(seed_tensor)

    self.assertEqual(handler._int_seed, seed_val)

    expected_kernel_seed = backend.random.sanitize_seed(seed_tensor)
    test_utils.assert_allequal(handler.get_kernel_seed(), expected_kernel_seed)

  def test_tf_initialization_with_tensor_stateless_seed(self):
    """Tests TF initialization with an EagerTensor stateless seed."""
    self._set_backend_for_test(_TF)
    seed_tensor = tf.constant([42, 99], dtype=tf.int32)
    handler = backend.RNGHandler(seed_tensor)

    # Cannot extract a single Python integer from a non-scalar tensor.
    self.assertIsNone(handler._int_seed)

    expected_kernel_seed = backend.random.sanitize_seed(seed_tensor)
    test_utils.assert_allequal(handler.get_kernel_seed(), expected_kernel_seed)

  def test_jax_initialization_with_scalar_array_seed(self):
    """JAX should accept scalar arrays."""
    self._set_backend_for_test(_JAX)
    seed_val = 555
    seed_array = jnp.array(seed_val, dtype=jnp.int32)
    handler = backend.RNGHandler(seed_array)

    self.assertEqual(handler._int_seed, seed_val)

  def test_jax_initialization_with_prng_key(self):
    """JAX should accept an existing PRNGKey."""
    self._set_backend_for_test(_JAX)
    seed_key = jax.random.PRNGKey(42)
    handler = backend.RNGHandler(seed_key)

    self._assert_key_equal(handler._key, seed_key)

  def test_jax_initialization_with_sequence_seed_raises(self):
    """JAX must not be initialized with a sequence."""
    self._set_backend_for_test(_JAX)
    with self.assertRaisesRegex(
        ValueError, "JAX backend requires a seed that is an integer"
    ):
      backend.RNGHandler([42, 99])

  def test_jax_initialization_with_non_scalar_array_raises(self):
    """JAX must not be initialized with a non-scalar array."""
    self._set_backend_for_test(_JAX)
    seed_array = jnp.array([42, 99])
    with self.assertRaisesRegex(
        ValueError, "JAX backend requires a seed that is an integer"
    ):
      backend.RNGHandler(seed_array)

  def test_tf_initialization_with_invalid_sequence_length_raises(self):
    """Tests that TF initialization validates sequence length (Req 2)."""
    self._set_backend_for_test(_TF)
    with self.assertRaisesRegex(
        ValueError, r"Invalid seed: Must be either.*or a pair"
    ):
      backend.RNGHandler([1, 2, 3])

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_get_next_seed_backend_specific_behavior(self, backend_name):
    """Validates the distinct seed generation logic for each backend."""
    self._set_backend_for_test(backend_name)
    seed = 42
    handler = backend.RNGHandler(seed)

    initial_kernel_seed = handler.get_kernel_seed()

    seed1 = handler.get_next_seed()
    seed2 = handler.get_next_seed()

    if backend_name == _TF:
      self.assertIsInstance(seed1, backend.Tensor)
      self.assertIsInstance(seed2, backend.Tensor)
      self.assertFalse(np.array_equal(seed1.numpy(), seed2.numpy()))
      test_utils.assert_not_allequal(
          handler.get_kernel_seed(), initial_kernel_seed
      )
    elif backend_name == _JAX:
      self.assertIsInstance(seed1, jax.Array)
      self.assertIsInstance(seed2, jax.Array)
      self._assert_key_not_equal(seed1, seed2)
      self._assert_key_not_equal(handler.get_kernel_seed(), initial_kernel_seed)

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_get_next_seed_is_reproducible(self, backend_name):
    """Ensures two handlers with the same seed produce the same sequence."""
    self._set_backend_for_test(backend_name)
    seed = 99
    handler1 = backend.RNGHandler(seed)
    handler2 = backend.RNGHandler(seed)

    seq1 = [handler1.get_next_seed() for _ in range(3)]
    seq2 = [handler2.get_next_seed() for _ in range(3)]

    for s1, s2 in zip(seq1, seq2):
      if backend_name == _JAX:
        self._assert_key_equal(s1, s2)
      else:
        test_utils.assert_allequal(s1, s2)

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_advance_handler_with_none_seed(self, backend_name):
    """Tests that advancing a no-op handler produces another no-op handler."""
    self._set_backend_for_test(backend_name)
    handler = backend.RNGHandler(None)
    new_handler = handler.advance_handler()

    self.assertIsNot(handler, new_handler)
    self.assertIsNone(new_handler._seed_input)
    self.assertIsNone(handler.get_next_seed())
    self.assertIsNone(new_handler.get_kernel_seed())

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_advance_handler_provides_independent_handlers(self, backend_name):
    """Ensures advance_handler generates new handlers with different states."""
    self._set_backend_for_test(backend_name)
    seed = 101
    handler = backend.RNGHandler(seed)

    initial_kernel_seed = handler.get_kernel_seed()

    new_handler1 = handler.advance_handler()

    if backend_name == _JAX:
      self._assert_key_not_equal(handler.get_kernel_seed(), initial_kernel_seed)
    else:
      test_utils.assert_not_allequal(
          handler.get_kernel_seed(), initial_kernel_seed
      )

    if backend_name == _JAX:
      new_handler2 = handler.advance_handler()
    else:
      new_handler2 = new_handler1.advance_handler()

    self.assertIsNot(handler, new_handler1)
    self.assertIsNot(new_handler1, new_handler2)

    kernel_seed1 = new_handler1.get_kernel_seed()
    kernel_seed2 = new_handler2.get_kernel_seed()

    if backend_name == _JAX:
      self._assert_key_not_equal(kernel_seed1, initial_kernel_seed)
      self._assert_key_not_equal(kernel_seed1, kernel_seed2)
    else:
      self.assertFalse(np.array_equal(kernel_seed1, initial_kernel_seed))
      self.assertFalse(np.array_equal(kernel_seed1, kernel_seed2))

  @parameterized.named_parameters(("tensorflow", _TF), ("jax", _JAX))
  def test_advance_handler_is_reproducible(self, backend_name):
    """Tests that the sequence of advanced handlers is deterministic."""
    self._set_backend_for_test(backend_name)
    seed = 202
    handler1_start = backend.RNGHandler(seed)
    handler2_start = backend.RNGHandler(seed)

    # Simulate an MCMC loop by iteratively advancing the handlers.
    seq1 = []
    h1 = handler1_start
    for _ in range(3):
      if backend_name == _TF:
        h1 = h1.advance_handler()
        seq1.append(h1.get_kernel_seed())
      else:
        new_h1 = h1.advance_handler()
        seq1.append(new_h1.get_kernel_seed())

    seq2 = []
    h2 = handler2_start
    for _ in range(3):
      if backend_name == _TF:
        h2 = h2.advance_handler()
        seq2.append(h2.get_kernel_seed())
      else:
        new_h2 = h2.advance_handler()
        seq2.append(new_h2.get_kernel_seed())

    for s1, s2 in zip(seq1, seq2):
      if backend_name == _JAX:
        self._assert_key_equal(s1, s2)
      else:
        test_utils.assert_allequal(s1, s2)

    if backend_name == _JAX:
      self._assert_key_not_equal(seq1[0], seq1[1])
      self._assert_key_not_equal(seq1[1], seq1[2])
    else:
      self.assertFalse(np.array_equal(seq1[0], seq1[1]))
      self.assertFalse(np.array_equal(seq1[1], seq1[2]))


if __name__ == "__main__":
  absltest.main()
