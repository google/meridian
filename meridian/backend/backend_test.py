# Copyright 2024 The Meridian Authors.
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

"""Tests for the Meridian backend abstraction layer."""

import importlib
import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

# Need to import backend dynamically within tests based on env var setting
# from meridian import backend # Cannot import directly here


class BackendTest(parameterized.TestCase):

  def tearDown(self):
    # Clean up environment variable and remove module from cache
    if 'MERIDIAN_BACKEND' in os.environ:
      del os.environ['MERIDIAN_BACKEND']
    if 'meridian.backend' in sys.modules:
      del sys.modules['meridian.backend']
    if 'meridian.backend.tensorflow_backend' in sys.modules:
      del sys.modules['meridian.backend.tensorflow_backend']
    if 'meridian.backend.jax_backend' in sys.modules:
      del sys.modules['meridian.backend.jax_backend']
    super().tearDown()

  @parameterized.named_parameters(
      ('TensorFlow_Default', None, 'tensorflow'),
      ('TensorFlow_Explicit', 'tensorflow', 'tensorflow'),
      ('JAX_Explicit', 'jax', 'jax'),
  )
  def test_backend_selection(self, env_value, expected_backend_name):
    """Tests that the correct backend module is loaded."""
    if env_value:
      os.environ['MERIDIAN_BACKEND'] = env_value
    else:
      # Ensure it's not set if testing default
      if 'MERIDIAN_BACKEND' in os.environ:
        del os.environ['MERIDIAN_BACKEND']

    # Import backend *after* setting env var
    backend = importlib.import_module('meridian.backend')
    self.assertEqual(backend._BACKEND, expected_backend_name) # pylint: disable=protected-access

    # Check if a known function from the expected backend exists
    if expected_backend_name == 'tensorflow':
       self.assertTrue(hasattr(backend, 'convert_to_tensor'))
       # Check that a known JAX function isn't there or raises error
       # (depending on how __getattr__ handles it finally)
       with self.assertRaises(AttributeError):
            _ = backend.some_nonexistent_jax_function
    elif expected_backend_name == 'jax':
       # JAX backend should have its own convert_to_tensor (jnp.asarray)
       self.assertTrue(hasattr(backend, 'convert_to_tensor'))
       # Test a function known to be unimplemented in the scaffold
       with self.assertRaises(NotImplementedError):
            backend.xla_windowed_adaptive_nuts()


  def test_unknown_backend(self):
    """Tests that an unknown backend raises ValueError."""
    os.environ['MERIDIAN_BACKEND'] = 'invalid_backend'
    with self.assertRaises(ValueError):
      importlib.import_module('meridian.backend')

  # Add more tests below to check specific function calls via the backend #

  def test_tensorflow_convert_to_tensor(self):
    """Tests a basic TF function call via the backend."""
    os.environ['MERIDIAN_BACKEND'] = 'tensorflow'
    backend = importlib.import_module('meridian.backend')
    tensor = backend.convert_to_tensor([1, 2, 3], dtype=backend.float32)
    # Basic check if it's a tensor-like object (cannot use isinstance easily)
    self.assertTrue(hasattr(tensor, 'numpy'))
    self.assertEqual(tensor.shape, (3,))

  @mock.patch.dict(os.environ, {"MERIDIAN_BACKEND": "tensorflow"})
  def test_tensorflow_normal_distribution(self):
      """Tests accessing a TFP distribution via the backend."""
      # Reload backend after setting env var via mock
      if 'meridian.backend' in sys.modules:
          del sys.modules['meridian.backend']
      backend = importlib.import_module('meridian.backend')

      dist = backend.tfp_distributions.Normal(loc=0., scale=1.)
      self.assertIsInstance(dist, backend.tfp_distributions.Distribution)
      self.assertEqual(dist.loc, 0.)


  def test_jax_convert_to_tensor_works(self):
      """Tests that a JAX function call works if implemented."""
      # Test a function that is implemented in the JAX scaffold
      os.environ['MERIDIAN_BACKEND'] = 'jax'
      backend = importlib.import_module('meridian.backend')
      # jnp.asarray should work
      array = backend.convert_to_tensor([1, 2, 3], dtype=backend.float32)
      self.assertTrue(hasattr(array, 'dtype')) # Basic check for JAX array
      self.assertEqual(array.shape, (3,))


  def test_jax_notimplemented_raises(self):
      """Tests that a JAX function call raises NotImplemented if needed."""
      # Test a function that raises NotImplementedError
      os.environ['MERIDIAN_BACKEND'] = 'jax'
      backend = importlib.import_module('meridian.backend')
      with self.assertRaises(NotImplementedError):
        backend.xla_windowed_adaptive_nuts() # This was set to _not_implemented


if __name__ == '__main__':
  absltest.main()
