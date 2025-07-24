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

"""Backend Abstraction Layer for Meridian."""

import os

from typing import Any, Optional

from meridian.backend import config
from typing_extensions import Literal


# The conditional imports in this module are a deliberate design choice for the
# backend abstraction layer. The TFP-on-JAX substrate provides a nearly
# identical API to the standard TFP library, making an alias-based approach more
# pragmatic than a full Abstract Base Class implementation, which would require
# extensive boilerplate.
# pylint: disable=g-import-not-at-top,g-bad-import-order

_BACKEND = config.get_backend()

if _BACKEND == config.Backend.JAX:
  import jax
  import jax.numpy as ops
  import tensorflow_probability.substrates.jax as tfp_jax

  Tensor = jax.Array
  tfd = tfp_jax.distributions
  _convert_to_tensor = ops.asarray
elif _BACKEND == config.Backend.TENSORFLOW:
  import tensorflow as tf
  import tensorflow_probability as tfp

  ops = tf
  Tensor = tf.Tensor
  tfd = tfp.distributions
  _convert_to_tensor = tf.convert_to_tensor
else:
  raise ValueError(f"Unsupported backend: {_BACKEND}")
# pylint: enable=g-import-not-at-top,g-bad-import-order


def to_tensor(data: Any, dtype: Optional[Any] = None) -> Tensor:  # type: ignore
  """Converts input data to the currently active backend tensor type.

  Args:
    data: The data to convert.
    dtype: The desired data type of the resulting tensor. The accepted types
      depend on the active backend (e.g., jax.numpy.dtype or tf.DType).

  Returns:
    A tensor representation of the data for the active backend.
  """

  return _convert_to_tensor(data, dtype=dtype)
