# Meridian Backend Abstraction Layer

This document is a developer's cookbook for working with Meridian's dual-backend
system. It outlines the essential patterns and principles for writing code that
can run on both TensorFlow and JAX.

For a comprehensive understanding of JAX, please refer to the [official JAX
documentation](https://jax.readthedocs.io/en/latest/).

## Core Principle: No Direct Backend Imports

To maintain backend agnosticism, **you must not directly import `tensorflow` or
`jax`** in the core Meridian library code (e.g., inside `meridian/model/` or
`meridian/analysis/`).

Instead, all numerical, probabilistic, and tensor operations must be accessed
through the `meridian.backend` module.

## The `meridian.backend` Cookbook

The `meridian.backend` module provides a unified API for backend operations.

### 1. Importing the Backend

Always start by importing the backend module:

```python
from meridian import backend
```

### 2. Numerical Operations (`backend.ops`)

For any numerical operations that would typically use `tensorflow` or
`jax.numpy`, use the `backend.ops` alias. This object will point to the correct
library based on the active backend.

**Before (TensorFlow-specific):**

```python
import tensorflow as tf
result = tf.einsum('gc,gtc->gt', gamma_gc, controls_scaled)
concatenated = tf.concat([part1, part2], axis=-1)
```

**After (Backend-agnostic):**

```python
from meridian import backend
result = backend.ops.einsum('gc,gtc->gt', gamma_gc, controls_scaled)
concatenated = backend.ops.concatenate([part1, part2], axis=-1)
```

### 3. Probabilistic Distributions (`backend.tfd`)

For all TensorFlow Probability distributions, use the `backend.tfd` alias. This
ensures that you are using the correct TFP substrate (either TFP-on-TF or
TFP-on-JAX).

**Before (TensorFlow-specific):**

```python
import tensorflow_probability as tfp
tfd = tfp.distributions
my_dist = tfd.LogNormal(loc=0.2, scale=0.9)
```

**After (Backend-agnostic):**

```python
from meridian import backend
my_dist = backend.tfd.LogNormal(loc=0.2, scale=0.9)
```

### 4. Tensor Type Hinting (`backend.Tensor`)

When type-hinting tensors or arrays, use the `backend.Tensor` alias. This will
correctly resolve to `tf.Tensor` or `jax.Array` depending on the active backend.

**Before (TensorFlow-specific):**

```python
import tensorflow as tf
def my_function(x: tf.Tensor) -> tf.Tensor:
  # ...
```

**After (Backend-agnostic):**

```python
from meridian import backend
def my_function(x: backend.Tensor) -> backend.Tensor:
  # ...
```

### 5. Converting Data to Tensors (`backend.to_tensor`)

Use the `backend.to_tensor()` utility to convert Python lists or NumPy arrays
into the tensor type of the currently active backend. This is especially useful
in test setups.

```python
from meridian import backend

# This will create a tf.Tensor in 'tensorflow' mode
# and a jax.Array in 'jax' mode.
my_tensor = backend.to_tensor([1.0, 2.0, 3.0])
```