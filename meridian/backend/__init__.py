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

"""Backend abstraction layer for Meridian."""

import os
import importlib

_BACKEND = os.environ.get('MERIDIAN_BACKEND', 'tensorflow').lower()

if _BACKEND == 'tensorflow':
  print('Using TensorFlow backend.')
  _backend_module = importlib.import_module('.tensorflow_backend', package='meridian.backend')
elif _BACKEND == 'jax':
  print('Using JAX backend.')
  _backend_module = importlib.import_module('.jax_backend', package='meridian.backend')
else:
  raise ValueError(f'Unknown backend: {_BACKEND}. Must be "tensorflow" or "jax".')

# Dynamically expose functions/classes from the selected backend module
# Example: get function 'some_function' from the backend
# some_function = getattr(_backend_module, 'some_function')

# TODO: Add getattr calls for all abstracted functions/classes once defined.

# Placeholder for exposing attributes - will be populated later
def __getattr__(name):
  # This function allows dynamic attribute access on the module.
  # It will attempt to get the attribute `name` from the loaded `_backend_module`.
  try:
    return getattr(_backend_module, name)
  except AttributeError:
    raise AttributeError(f"module 'meridian.backend.{_BACKEND}_backend' has no attribute '{name}'")
