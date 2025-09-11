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

  def test_set_random_seed_raises_for_jax(self):
    self._set_backend_for_test(_JAX)
    with self.assertRaises(NotImplementedError):
      backend.set_random_seed(0)

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
          expected=np.array([10, 40, 20]),
      ),
      dict(
          testcase_name="2d_tensor",
          tensor_in=[[1, 2], [3, 4], [5, 6]],
          indices=[2, 0],
          expected=np.array([[5, 6], [1, 2]]),
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
    expected = test_case["expected"]

    result = backend.gather(tensor, indices)
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

  def test_extension_type_raises_for_jax(self):
    self._set_backend_for_test(_JAX)

    class MyExtension(backend.ExtensionType):
      foo: int
      bar: str

    with self.assertRaises(NotImplementedError):
      MyExtension()


if __name__ == "__main__":
  absltest.main()
