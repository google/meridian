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

"""Backend configuration for Meridian."""

import enum


class Backend(enum.Enum):
  TENSORFLOW = "tensorflow"
  JAX = "jax"


_BACKEND = Backend.TENSORFLOW


def set_backend(backend: Backend) -> None:
  """Sets the backend for Meridian."""
  global _BACKEND
  if not isinstance(backend, Backend):
    raise ValueError("Backend must be a member of the Backend enum.")
  _BACKEND = backend


def get_backend() -> Backend:
  """Returns the current backend for Meridian."""
  return _BACKEND

