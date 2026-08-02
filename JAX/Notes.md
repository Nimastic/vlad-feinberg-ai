Here's a full explanation of every code cell in `tutorial.ipynb`, section by section.

- * *

## Cell 1 — Setup & Install

python

import sys

print("Kernel Python:", sys.executable)

!{sys.executable} \-m pip install jax matplotlib \-q

print("Done! Now run the cells below.")

**What it does:** Identifies which Python interpreter the notebook is using, then installs `jax` and `matplotlib` into that exact Python environment using pip. The `!{}` syntax runs a shell command from inside the notebook.

- * *

## Section 1 — JAX vs NumPy

### Cell 2 — Plotting with JAX

python

import numpy as np

import jax.numpy as jnp

import matplotlib.pyplot as plt

x_jnp \= jnp.linspace(0, 10, 1000)

y_jnp \= 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)

plt.plot(x_jnp, y_jnp)

plt.title("JAX: 2·sin(x)·cos(x)")

plt.xlabel("x")

plt.ylabel("y")

plt.show()

**What it does:** Creates 1000 evenly spaced x-values from 0 to 10 using `jnp.linspace`, computes y = 2·sin(x)·cos(x), and plots the result. This demonstrates that `jax.numpy` works like `numpy` — you can just swap `np` → `jnp`.

- * *

### Cell 3 — Array Types

python

x_np \= np.linspace(0, 10, 1000)

print("NumPy type:", type(x_np))    \# numpy.ndarray

print("JAX type:  ", type(x_jnp))   \# jaxlib._jax.ArrayImpl

**What it does:** Shows that NumPy and JAX arrays are **different Python types** even though they behave similarly. NumPy produces `numpy.ndarray`; JAX produces `jaxlib._jax.ArrayImpl`.

- * *

### Cell 4 — NumPy Arrays Are Mutable

python

x_mut \= np.arange(10)

x_mut[0] \= 10

print("NumPy (mutable):", x_mut)

**What it does:** Creates a NumPy array `[0, 1, 2, ..., 9]` and directly changes the first element to `10` **in-place**. This works because NumPy arrays are **mutable**.

- * *

### Cell 5 — JAX Arrays Are Immutable

python

x_immut \= jnp.arange(10)

y_immut \= x_immut.at[0].set(10)   \# returns a NEW array

print("JAX original: ", x_immut)   \# unchanged

print("JAX updated:  ", y_immut)

# This would raise TypeError: JAX arrays are immutable

# x_immut[0] = 10

**What it does:** JAX arrays **cannot be changed in-place**. Instead, `.at[0].set(10)` returns a **new array** with that element changed. The original array `x_immut` remains untouched. Attempting `x_immut[0] = 10` would crash with a `TypeError`.

- * *

## Section 2 — JAX Arrays: Devices & Sharding

### Cell 6 — Device Awareness

python

import jax

x \= jnp.arange(5)

print("Is jax.Array?", isinstance(x, jax.Array))   \# True

print("Devices:      ", x.devices())               \# {CpuDevice(id=0)}

print("Sharding:     ", x.sharding)                \# SingleDeviceSharding(...)

**What it does:** Shows that every JAX array is a `jax.Array` and knows which device it lives on (CPU, GPU, or TPU). `.devices()` returns the set of devices; `.sharding` describes how the array is distributed. On a regular laptop, you'll see CPU only.

- * *

## Section 3 — Just-in-Time Compilation (jax.jit)

### Cell 7 — JIT Basics

python

from jax import jit

def norm(X):

"""Row-wise normalisation: subtract column mean, divide by column std."""

X \= X \- X.mean(0)

return X / X.std(0)

norm_compiled \= jit(norm)   \# JIT-compiled version

np.random.seed(1701)

X \= jnp.array(np.random.rand(10_000, 10))

print("Results match:", np.allclose(norm(X), norm_compiled(X), atol\=1e-6))

**What it does:** Defines a function `norm` that normalises each column of a matrix. `jit(norm)` creates a compiled version using XLA (a compiler used by JAX/TensorFlow). Both give the same result — `jit` only changes **performance**, not behavior. The first call compiles; subsequent calls use the cached compilation.

- * *

### Cell 8 — JIT Benchmark

python

import timeit

_ \= norm_compiled(X).block_until_ready()  \# warm up

t1 \= timeit.timeit(lambda: norm(X).block_until_ready(), number\=100)

t2 \= timeit.timeit(lambda: norm_compiled(X).block_until_ready(), number\=100)

print(f"Eager  (100 runs): {t1*1000:.1f} ms")

print(f"JIT    (100 runs): {t2*1000:.1f} ms")

**What it does:** Benchmarks the eager (non-JIT) vs. JIT-compiled version over 100 runs. `.block_until_ready()` is crucial — JAX uses **asynchronous dispatch**, so without it the timer would return before the computation actually finishes. JIT is generally faster after the initial compile.

- * *

### Cell 9 — JIT Limitation (Dynamic Shapes)

python

def get_negatives(x):

return x[x < 0]   \# output shape depends on runtime values

x_rand \= jnp.array(np.random.randn(10))

print("Eager (works):", get_negatives(x_rand))

# Uncomment to see NonConcreteBooleanIndexError:

# jit(get_negatives)(x_rand)

**What it does:** Shows a key **limitation** of JIT — it requires all array shapes to be known at **compile time**. `x[x < 0]` produces a result whose size depends on the actual data values, so JIT can't compile it. Eagerly it works fine; under JIT it would crash.

- * *

## Section 4 — Automatic Differentiation (jax.grad)

### Cell 10 — Computing a Gradient

python

from jax import grad

def sum_logistic(x):

"""Sum of sigmoid(x_i) — a scalar output."""

return jnp.sum(1.0 / (1.0 + jnp.exp(\-x)))

x_small \= jnp.arange(3.)

derivative_fn \= grad(sum_logistic)

print("grad(sum_logistic):", derivative_fn(x_small))

**What it does:** `grad` transforms `sum_logistic` into a function that computes its **gradient** (derivative with respect to each input element). The sigmoid function σ(x) = 1/(1+e⁻ˣ) has derivative σ(x)·(1-σ(x)). `grad` requires the function to return a **scalar** (a single number).

- * *

### Cell 11 — Finite Difference Verification

python

def first_finite_differences(f, x, eps\=1e-3):

return jnp.array([

(f(x + eps * v) \- f(x \- eps * v)) / (2 * eps)

for v in jnp.eye(len(x))

])

print("Finite diff approx:", first_finite_differences(sum_logistic, x_small))

**What it does:** Manually approximates the gradient using **finite differences** — the classic numerical differentiation method: (f(x+ε) - f(x-ε)) / (2ε). `jnp.eye(len(x))` creates unit vectors, one for each dimension. This verifies that `jax.grad` gives the mathematically correct answer.

- * *

### Cell 12 — Higher-Order Derivatives

python

print("3rd-order derivative at 1.0:", grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

**What it does:** Chains `grad` three times to compute the **3rd-order derivative**. Also mixes in `jit` to show they compose freely. JAX handles any depth of differentiation automatically.

- * *

### Cell 13 — Jacobian Matrix

python

from jax import jacobian

print("Jacobian of exp(x_small):\n", jacobian(jnp.exp)(x_small))

**What it does:** For **vector-valued** functions, `grad` isn't enough — you need a full **Jacobian matrix** (matrix of all partial derivatives). `jacobian(jnp.exp)` produces a diagonal matrix because `exp` acts element-wise, so each output only depends on one input.

- * *

### Cell 14 — Hessian Matrix

python

from jax import jacfwd, jacrev

def hessian(fun):

return jit(jacfwd(jacrev(fun)))

print("Hessian of sum_logistic:\n", hessian(sum_logistic)(x_small))

**What it does:** Computes the **Hessian** (matrix of second derivatives). The trick is to apply `jacrev` (reverse-mode Jacobian) first, then `jacfwd` (forward-mode Jacobian) on top — this is the most efficient combination. JIT is applied around the whole thing for speed.

- * *

## Section 5 — Auto-vectorization (jax.vmap)

### Cell 15 — Setup

python

from jax import random, vmap

key \= random.key(1701)

key1, key2 \= random.split(key)

mat \= random.normal(key1, (150, 100))

batched_x \= random.normal(key2, (10, 100))

def apply_matrix(x):

"""Matrix-vector product: mat @ x  (single vector)."""

return jnp.dot(mat, x)

**What it does:** Creates a random 150×100 matrix `mat` and a batch of 10 random vectors (each of size 100). `apply_matrix` is written for a **single** vector and does a matrix-vector product.

- * *

### Cell 16 — Naive Python Loop

python

def naively_batched_apply_matrix(v_batched):

return jnp.stack([apply_matrix(v) for v in v_batched])

print("Naive output shape:", naively_batched_apply_matrix(batched_x).shape)

**What it does:** Handles the batch with a plain Python `for` loop — applying `apply_matrix` one vector at a time and stacking results. Output shape is `(10, 150)`. This works but is slow.

- * *

### Cell 17 — Manual Batching

python

@jit

def batched_apply_matrix(batched_x):

return jnp.dot(batched_x, mat.T)

np.testing.assert_allclose(

naively_batched_apply_matrix(batched_x),

batched_apply_matrix(batched_x), atol\=1e-4, rtol\=1e-4)

print("Manual batching matches naive: True")

**What it does:** Manually rewrites the function to handle a full batch at once using matrix multiplication (`batched_x @ mat.T`). Then verifies the result matches the naive loop. This is fast but requires you to **rewrite** the function.

- * *

### Cell 18 — vmap Auto-vectorization

python

@jit

def vmap_batched_apply_matrix(batched_x):

return vmap(apply_matrix)(batched_x)

np.testing.assert_allclose(

naively_batched_apply_matrix(batched_x),

vmap_batched_apply_matrix(batched_x), atol\=1e-4, rtol\=1e-4)

print("vmap batching matches naive: True")

**What it does:** `vmap(apply_matrix)` automatically transforms the single-vector function into a batched one — **no rewriting needed**. JAX generates the efficient batched implementation internally. Result is the same as the other two approaches.

- * *

### Cell 19 — Benchmark All Three

python

_ \= batched_apply_matrix(batched_x).block_until_ready()

_ \= vmap_batched_apply_matrix(batched_x).block_until_ready()

t_naive  \= timeit.timeit(lambda: naively_batched_apply_matrix(batched_x).block_until_ready(), number\=100)

t_manual \= timeit.timeit(lambda: batched_apply_matrix(batched_x).block_until_ready(), number\=100)

t_vmap   \= timeit.timeit(lambda: vmap_batched_apply_matrix(batched_x).block_until_ready(), number\=100)

print(f"Naive   (100 runs): {t_naive*1000:.1f} ms")

print(f"Manual  (100 runs): {t_manual*1000:.1f} ms")

print(f"vmap    (100 runs): {t_vmap*1000:.1f} ms")

**What it does:** Benchmarks all three batching approaches. Naive loop is ~30ms; manual and vmap are both ~1ms. This shows `vmap` is just as fast as the hand-optimised version, but without any manual rewriting.

- * *

## Section 6 — Pseudorandom Numbers

### Cell 20 — Deterministic Keys

python

from jax import random

key \= random.key(43)

print("Initial key:", key)

print("Draw 1 (same key):", random.normal(key))

print("Draw 2 (same key):", random.normal(key))   \# identical!

**What it does:** Creates a **PRNG key** seeded with `43`. JAX's random system is fully **explicit** — every call uses the key you pass. Using the **same key twice produces identical numbers**. This is unlike NumPy's global hidden state.

- * *

### Cell 21 — Splitting Keys for Different Samples

python

for i in range(3):

new_key, subkey \= random.split(key)

val \= random.normal(subkey)

print(f"draw {i}: {val:.6f}")

key \= new_key   \# advance to the next key

**What it does:** To get **different** random numbers, you must `split` the key first. `random.split(key)` produces two new independent keys: `new_key` (used to continue the sequence) and `subkey` (used for this draw). Then `key` is updated so the next iteration gets a fresh split.

- * *

## Section 7 — Debugging

### Cell 22 — Python print() Inside JIT (Trace Time)

python

import jax

@jax.jit

def f_print(x):

print("print(x) ->", x)   \# trace-time: shows JitTracer

y \= jnp.sin(x)

print("print(y) ->", y)   \# trace-time

return y

_ \= f_print(2.)

**What it does:** Inside a JIT-compiled function, regular `print()` runs only at **trace time** (when JAX analyzes your code), not at execution time. It shows abstract **tracer objects** (e.g., `Traced<~float32[]>`), not the actual values. This is confusing but expected.

- * *

### Cell 23 — jax.debug.print() (Runtime)

python

@jax.jit

def f_debug(x):

jax.debug.print("jax.debug.print(x) -> {x}", x\=x)

y \= jnp.sin(x)

jax.debug.print("jax.debug.print(y) -> {y}", y\=y)

return y

_ \= f_debug(2.)

**What it does:** `jax.debug.print()` is specifically designed to work inside JIT — it runs at **actual execution time** and prints the **real values**. Use this when you need to inspect values inside compiled functions.

- * *

### Cell 24 — jax.disable_jit()

python

with jax.disable_jit():

result \= f_debug(2.)

print("Eager result:", result)

**What it does:** `jax.disable_jit()` is a context manager that **turns off JIT globally** within the `with` block. All functions run eagerly (step by step), making them compatible with standard Python `print()` and debuggers like `pdb`. Useful for debugging.

- * *

### Cell 25 — NaN Detection (Commented)

python

# Detect NaNs automatically during development:

# jax.config.update("jax_debug_nans", True)

# _ = jnp.log(jnp.array(-1.0))   # would raise FloatingPointError

print("Tutorial complete! 🎉")

**What it does:** Shows how to enable `jax_debug_nans` — a config flag that makes JAX **raise an error immediately** when a NaN appears, instead of silently propagating it. The commented code would compute `log(-1) = NaN` and crash with a `FloatingPointError`. Left commented to avoid breaking the notebook. Prints the completion message.

- * *

## Summary Table

| Section | Key Concept | What You Learn |
| --- | --- | --- |
| 1 | JAX vs NumPy | `jnp` mirrors `np`; JAX arrays are immutable |
| 2 | Devices & Sharding | Arrays know where they live (CPU/GPU/TPU) |
| 3 | `jax.jit` | Compile functions for speed; shapes must be static |
| 4 | `jax.grad` | Auto-differentiation, Jacobians, Hessians |
| 5 | `jax.vmap` | Auto-vectorize single-example functions to batch |
| 6 | Random keys | Explicit, reproducible, stateless PRNG |
| 7 | Debugging | Trace-time vs. runtime printing, `disable_jit` |
