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

"""JAX backend implementation scaffold for Meridian."""

# It's okay if these imports fail initially if TFP-on-JAX isn't installed.
# The backend selection logic prevents this file from being imported unless requested.
try:
  import jax
  import jax.numpy as jnp
  import tensorflow_probability.substrates.jax as tfp
except ImportError:
  raise ImportError(
      'JAX backend requires jax, jaxlib, and tensorflow-probability[jax] to be'
      ' installed.'
  )

tfp_distributions = tfp.distributions
tfp_experimental_mcmc = tfp.experimental.mcmc
tfp_mcmc = tfp.mcmc
tfp_random = tfp.random


# --- Helper Function ---
def _not_implemented(*args, **kwargs):
  raise NotImplementedError(
      'This function is not yet implemented for the JAX backend.'
  )


# --- Tensor Operations ---
convert_to_tensor = jnp.asarray # JAX uses arrays
Tensor = jax.Array # JAX array type
float32 = jnp.float32
int32 = jnp.int32
bool = jnp.bool_ # JAX uses bool_
zeros = jnp.zeros
ones = jnp.ones
shape = jnp.shape
concat = jnp.concatenate # JAX uses concatenate
einsum = jnp.einsum
transpose = jnp.transpose
# broadcast_to needs careful handling depending on context in JAX
broadcast_to = lambda tensor, shape: jnp.broadcast_to(tensor, tuple(shape)) # Basic wrapper
where = jnp.where
abs = jnp.abs
exp = jnp.exp
log = jnp.log
sqrt = jnp.sqrt
argmax = jnp.argmax
reduce_sum = jnp.sum
reduce_std = jnp.std
# unique_with_counts might need a custom implementation or jax.lax.unique
unique_with_counts = _not_implemented
# boolean_mask needs careful conversion
boolean_mask = lambda tensor, mask: tensor[mask] # Basic JAX indexing
gather = _not_implemented # gather often maps to specific indexing patterns
equal = jnp.equal
less = jnp.less
greater = jnp.greater
logical_and = jnp.logical_and
logical_or = jnp.logical_or
logical_not = jnp.logical_not
newaxis = jnp.newaxis
function = jax.jit # JAX equivalent for optimization/compilation
# keras_utils might not have a direct JAX equivalent, depends on usage
keras_utils = _not_implemented

# --- Error Types ---
# JAX might have different error types or mechanisms
ResourceExhaustedError = _not_implemented


# --- Helper Functions/Classes (Scaffolds) ---
xla_windowed_adaptive_nuts = _not_implemented

# TODO: Add scaffolds for specific distributions (Normal, Sample, Deterministic, etc.)
# TODO: Add scaffolds/exports for MCMC kernels (windowed_adaptive_nuts)
# TODO: Add scaffolds/exports for other TF/TFP functions as needed during refactoring.
