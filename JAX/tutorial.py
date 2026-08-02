"""
JAX Quickstart Tutorial
========================
Based on: https://docs.jax.dev/en/latest/quickstart.html

Covers:
  1. JAX vs NumPy
  2. JAX arrays (jax.Array) — creation, devices, sharding
  3. Just-in-time compilation with jax.jit
  4. Automatic differentiation with jax.grad
  5. Auto-vectorization with jax.vmap
  6. Pseudorandom numbers
  7. Debugging (jax.debug.print, flags)
"""

# ──────────────────────────────────────────────
# 1. JAX vs NumPy
# ──────────────────────────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# JAX numpy mirrors the NumPy API — just swap np → jnp
x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)
plt.title("JAX: 2·sin(x)·cos(x)")
plt.show()

# Arrays are different Python types
x_np = np.linspace(0, 10, 1000)
print(type(x_np))    # numpy.ndarray
print(type(x_jnp))   # jaxlib._jax.ArrayImpl

# NumPy arrays are mutable …
x_mut = np.arange(10)
x_mut[0] = 10
print("NumPy (mutable):", x_mut)

# … JAX arrays are IMMUTABLE — use .at[].set() instead
x_immut = jnp.arange(10)
y_immut = x_immut.at[0].set(10)   # returns a new array
print("JAX original:  ", x_immut)  # unchanged
print("JAX updated:   ", y_immut)

# ──────────────────────────────────────────────
# 2. JAX Arrays — devices & sharding
# ──────────────────────────────────────────────
import jax

x = jnp.arange(5)
print("Is jax.Array?", isinstance(x, jax.Array))   # True
print("Devices:      ", x.devices())               # {CpuDevice(id=0)}
print("Sharding:     ", x.sharding)                # SingleDeviceSharding(...)

# ──────────────────────────────────────────────
# 3. Just-in-time compilation with jax.jit
# ──────────────────────────────────────────────
from jax import jit

def norm(X):
    """Row-wise normalisation: subtract column mean, divide by column std."""
    X = X - X.mean(0)
    return X / X.std(0)

norm_compiled = jit(norm)   # JIT-compiled version

np.random.seed(1701)
X = jnp.array(np.random.rand(10_000, 10))

# Both produce the same result …
assert np.allclose(norm(X), norm_compiled(X), atol=1e-6)
print("norm vs norm_compiled match: True")

# … but the compiled version is faster on repeated calls.
# (Use block_until_ready() to account for JAX's async dispatch.)
import timeit
t1 = timeit.timeit(lambda: norm(X).block_until_ready(), number=100)
t2 = timeit.timeit(lambda: norm_compiled(X).block_until_ready(), number=100)
print(f"Eager  (100 runs): {t1*1000:.1f} ms")
print(f"JIT    (100 runs): {t2*1000:.1f} ms")

# ⚠ JIT requires static shapes — dynamic shapes won't compile:
def get_negatives(x):
    return x[x < 0]          # output shape depends on runtime values

x_rand = jnp.array(np.random.randn(10))
print("Eager get_negatives:", get_negatives(x_rand))

# The line below would raise NonConcreteBooleanIndexError — uncomment to test:
# jit(get_negatives)(x_rand)

# ──────────────────────────────────────────────
# 4. Automatic differentiation with jax.grad
# ──────────────────────────────────────────────
from jax import grad

def sum_logistic(x):
    """Sum of sigmoid(x_i)."""
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print("grad(sum_logistic):", derivative_fn(x_small))

# Verify with finite differences
def first_finite_differences(f, x, eps=1e-3):
    return jnp.array([
        (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
        for v in jnp.eye(len(x))
    ])

print("finite diff approx:", first_finite_differences(sum_logistic, x_small))

# grad and jit compose arbitrarily
print("grad(jit(grad(jit(grad(...))))):", grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

# Full Jacobian for vector-valued functions
from jax import jacobian
print("Jacobian of exp:\n", jacobian(jnp.exp)(x_small))

# Hessian via forward-over-reverse mode AD
from jax import jacfwd, jacrev

def hessian(fun):
    return jit(jacfwd(jacrev(fun)))

print("Hessian of sum_logistic:\n", hessian(sum_logistic)(x_small))

# ──────────────────────────────────────────────
# 5. Auto-vectorization with jax.vmap
# ──────────────────────────────────────────────
from jax import random, vmap

key = random.key(1701)
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100))

def apply_matrix(x):
    """Matrix-vector product: mat @ x."""
    return jnp.dot(mat, x)

# Naive Python loop — slow
def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])

# Manual batching — fast but tedious for complex functions
@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, mat.T)

# vmap — automatic and composable
@jit
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)

# All three should agree
np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    batched_apply_matrix(batched_x), atol=1e-4, rtol=1e-4)
np.testing.assert_allclose(
    naively_batched_apply_matrix(batched_x),
    vmap_batched_apply_matrix(batched_x), atol=1e-4, rtol=1e-4)
print("All batching approaches match: True")

t_naive  = timeit.timeit(lambda: naively_batched_apply_matrix(batched_x).block_until_ready(), number=100)
t_manual = timeit.timeit(lambda: batched_apply_matrix(batched_x).block_until_ready(), number=100)
t_vmap   = timeit.timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(), number=100)
print(f"Naive   (100 runs): {t_naive*1000:.1f} ms")
print(f"Manual  (100 runs): {t_manual*1000:.1f} ms")
print(f"vmap    (100 runs): {t_vmap*1000:.1f} ms")

# ──────────────────────────────────────────────
# 6. Pseudorandom numbers
# ──────────────────────────────────────────────
# JAX uses *explicit* keys — no hidden global state.

key = random.key(43)
print("Initial key:", key)

# Same key → same sample (deterministic)
print("Same key, two draws:", random.normal(key), random.normal(key))

# Different samples → split the key first
for i in range(3):
    new_key, subkey = random.split(key)
    val = random.normal(subkey)
    print(f"draw {i}: {val}")
    key = new_key   # advance to the new key for next iteration

# ──────────────────────────────────────────────
# 7. Debugging
# ──────────────────────────────────────────────

# 7a. Python print() inside jit shows tracer objects (trace-time), not values:
@jax.jit
def f_print(x):
    print("print(x) ->", x)        # trace-time — shows JitTracer
    y = jnp.sin(x)
    print("print(y) ->", y)        # trace-time
    return y

_ = f_print(2.)

# 7b. jax.debug.print() runs at runtime and shows actual values:
@jax.jit
def f_debug(x):
    jax.debug.print("jax.debug.print(x) -> {x}", x=x)
    y = jnp.sin(x)
    jax.debug.print("jax.debug.print(y) -> {y}", y=y)
    return y

_ = f_debug(2.)

# 7c. Disable JIT globally for step-by-step Python debugging:
with jax.disable_jit():
    result = f_debug(2.)   # runs eagerly — compatible with pdb, print, etc.
    print("Eager result:", result)

# 7d. Detect NaNs automatically (useful during development):
# jax.config.update("jax_debug_nans", True)
# _ = jnp.log(jnp.array(-1.0))   # would raise FloatingPointError with flag on
