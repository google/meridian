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
from meridian import backend
from meridian.backend import config
import numpy as np


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

    if backend_name == config.Backend.JAX:
      self.assertIn("jax", backend.ops.__name__)
    else:
      self.assertIn("tensorflow", backend.ops.__name__)

  def test_invalid_backend(self):
    with self.assertRaises(ValueError):
      config.set_backend("invalid_backend")

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
      import jax.numpy as jnp

      self.assertIsInstance(list_tensor, jax.Array)
      self.assertEqual(list_tensor.dtype, jnp.float32)

      tensor_f64 = backend.to_tensor(py_list, dtype=jnp.float64)
      # JAX will downcast to float32 by default.
      self.assertEqual(tensor_f64.dtype, jnp.float32)
    else:
      import tensorflow as tf

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
      import jax.numpy as jnp

      self.assertIsInstance(np_tensor, jax.Array)
      # JAX downcasts float64 NumPy arrays to float32 by default
      self.assertEqual(np_tensor.dtype, jnp.float32)
    else:
      import tensorflow as tf

      self.assertIsInstance(np_tensor, tf.Tensor)
      self.assertEqual(np_tensor.dtype, tf.float64)


if __name__ == "__main__":
  absltest.main()
