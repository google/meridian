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

"""TensorFlow backend implementation for Meridian."""

import tensorflow as tf
import tensorflow_probability as tfp

tfp_distributions = tfp.distributions
tfp_experimental_mcmc = tfp.experimental.mcmc
tfp_mcmc = tfp.mcmc
tfp_random = tfp.random


# --- Tensor Operations ---
# Expose common TensorFlow functions needed by Meridian
convert_to_tensor = tf.convert_to_tensor
Tensor = tf.Tensor
float32 = tf.float32
int32 = tf.int32
bool = tf.bool # tf.bool is actually a dtype like tf.float32
zeros = tf.zeros
ones = tf.ones
shape = tf.shape
concat = tf.concat
einsum = tf.einsum
transpose = tf.transpose
broadcast_to = tf.broadcast_to
where = tf.where
abs = tf.abs # Use tf.abs for tensors
exp = tf.math.exp
log = tf.math.log
sqrt = tf.math.sqrt
argmax = tf.argmax
reduce_sum = tf.reduce_sum
reduce_std = tf.math.reduce_std
unique_with_counts = tf.unique_with_counts
boolean_mask = tf.boolean_mask
gather = tf.gather
equal = tf.equal
less = tf.less
greater = tf.greater
logical_and = tf.logical_and
logical_or = tf.logical_or
logical_not = tf.logical_not
newaxis = tf.newaxis
function = tf.function # For tf.function decorator
keras_utils = tf.keras.utils # For set_random_seed

# --- Error Types ---
ResourceExhaustedError = tf.errors.ResourceExhaustedError

# --- Helper Functions/Classes (to be potentially moved/wrapped) ---

# Example: If we decide to wrap a distribution
# def Normal(loc, scale, name=None):
#   return tfp_distributions.Normal(loc=loc, scale=scale, name=name)

# Example: Function moved from posterior_sampler.py
@tf.function(autograph=False, jit_compile=True)
def xla_windowed_adaptive_nuts(**kwargs):
  """XLA wrapper for windowed_adaptive_nuts."""
  return tfp.experimental.mcmc.windowed_adaptive_nuts(**kwargs)


# TODO: Add wrappers for specific distributions (Normal, Sample, Deterministic, etc.)
# TODO: Add wrappers/exports for MCMC kernels (windowed_adaptive_nuts)
# TODO: Add wrappers/exports for other TF/TFP functions as needed during refactoring.
