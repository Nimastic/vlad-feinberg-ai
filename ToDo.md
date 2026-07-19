# Frontier Lab Path — ToDo

Source: [How to Land a Frontier Lab Job](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html) (Vlad Feinberg, 10 May 2026)

Strategy: work at the **edges** of the LLM stack — **kernels** (below) and **agentic loops** (above) — then demonstrate skill with public artifacts.

---

## Phase 0 — Orientation

- [ ] Skim the full post and lock the goal: hireable skill signal, not vibes
  - Post: https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html
  - Your notes: [`Notes.md`](./Notes.md)
- [ ] Internalize the two wedges
  - Below stack: kernels / inference / systems (comm, FLOPs, HBM, latch bounds)
  - Above stack: rigorous agent experiments (not just CLAUDE.md)
- [ ] Optional context interview: [Developing.dev — Feinberg on frontier-lab hiring](https://www.developing.dev/p/google-deepmind-pre-training-lead)

---

## Phase 1 — Roofline / Reiner analysis (do this in your sleep)

- [ ] Watch / study Dwarkesh × Reiner Pope and redo the analyses by hand
  - https://www.dwarkesh.com/p/reiner-pope
  - Local scratch: [`1. Reiner Analysis.md`](./1.%20Reiner%20Analysis.md)
- [ ] Practice until automatic:
  - [ ] `t >= max(t_memoryfetch, t_compute)`
  - [ ] weight memory vs KV-cache memory bottlenecks
  - [ ] batch size, speculative decoding, multi-token prediction effects on the roofline
- [ ] Drill: pick a model shape + GPU/TPU spec and estimate decode time without looking anything up

---

## Phase 2 — JAX fluency

- [ ] Work through JAX “Thinking in JAX”
  - https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
  - Local scratch: [`2. JAX Tutorial.md`](./2.%20JAX%20Tutorial.md)
- [ ] Cover the rest of the official tutorial path as needed
  - Tutorial index: https://docs.jax.dev/en/latest/tutorials.html
- [ ] Get comfortable with: `jit`, `vmap`, `pmap`/`shard_map`, `grad`, pure functions, arrays-as-state
- [ ] Skim Flax + Optax enough to train a small model
  - Flax: https://flax.readthedocs.io/
  - Optax: https://optax.readthedocs.io/

---

## Phase 3 — Scaling Book (every exercise)

- [ ] Read **and do every exercise** in the How to Scale Your Model book
  - https://jax-ml.github.io/scaling-book/
  - Local scratch: [`3. Scaling Book Exercise .md`](./3.%20Scaling%20Book%20Exercise%20.md)
- [ ] Prefer paper + pencil first (Vlad’s preferred evidence trail)
- [ ] After each chapter: restate the bottleneck you modeled (comms / FLOPs / HBM / something else)

---

## Phase 4 — Capstone exercise (Vlad’s hiring bar)

Do this from scratch in JAX (Colab free GPU/TPU is fine).

### 4a — Tiny addition transformer

- [ ] Code a ~10M transformer with **only** JAX + Flax + Optax
  - Colab: https://colab.research.google.com/
- [ ] Hard-code vocab: digits `0-9`, space, `+`, `=`
- [ ] Generate a dataset of up-to-3-digit addition problems
- [ ] Pad to fixed length; confirm it trains quickly on a T4 (or Colab TPU)
- [ ] Screen-record yourself writing this by hand (no paste-from-LLM)

### 4b — Chinchilla laws (dense vs MoE)

- [ ] Read Chinchilla for the original framing
  - https://arxiv.org/abs/2203.15556
- [ ] Derive Chinchilla-style scaling laws for **your** addition setup
- [ ] Compare **dense** vs **MoE** architectures (implement MoE yourself)
- [ ] Prefer `jax.lax.ragged_dot` for the MoE layer if applicable
  - Docs: https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html
- [ ] Write up: what differs, and why you think it differs

### 4c — Pallas kernel that beats `ragged_dot`

- [ ] Learn enough Pallas to write a fused kernel
  - https://docs.jax.dev/en/latest/pallas/index.html
- [ ] For `F > D`, write a Pallas kernel that fuses MoE up/down projections and **beats** `ragged_dot`
- [ ] Find a measurable forward-pass speedup; explain the unmodeled / newly modeled constraint that creates it
- [ ] Screen-record the kernel work and the benchmarking writeup

---

## Phase 5 — Evidence pack + send to Vlad

- [ ] Paper/pencil: all Scaling Book exercises → scan → chatbot → LaTeX
- [ ] Keep video of a random subset of those problem sessions (he may ask)
- [ ] Screen-record: transformer-from-scratch + Chinchilla derivation
- [ ] Package:
  - [ ] Scaling-law report (dense vs MoE)
  - [ ] Exercise writeup (transformer + kernel + speedup explanation)
  - [ ] Links to recordings / GitHub repo
- [ ] Email Vlad with the scaling-law report + exercise writeup
  - Site / contact: https://vladfeinberg.com/
  - He notes he is hiring in NYC and wants this as a self-consistency check

---

## Phase 6 — Kernel research reading (learn what “contribution” looks like)

Goal: see how people find **unmodeled constraints** (FlashAttention lesson), not just memorize FA.

### FlashAttention lineage + DSLs

- [ ] FlashAttention-1 (original trick / IO-aware attention)
  - https://arxiv.org/abs/2205.14135
- [ ] FlashAttention-2
  - https://arxiv.org/abs/2307.08691
- [ ] FlashAttention-3
  - https://arxiv.org/abs/2407.08608
- [ ] FlashAttention-4 (B200 / CuTe DSL) — start here if short on time
  - PDF: https://arxiv.org/pdf/2603.05451
  - abs: https://arxiv.org/abs/2603.05451
- [ ] ThunderKittens (DSL / kernel abstraction Vlad calls out)
  - https://arxiv.org/abs/2410.20399
  - GitHub: https://github.com/HazyResearch/ThunderKittens
- [ ] CuTe / CUTLASS (context for FA4)
  - https://github.com/NVIDIA/cutlass

### Quantization (quality–performance tradeoff, GPU-cheap)

- [ ] LLM.int8()
  - https://arxiv.org/abs/2208.07339
- [ ] QuIP
  - https://arxiv.org/abs/2307.13304
- [ ] QuIP#
  - https://arxiv.org/abs/2402.04396
- [ ] QTIP
  - https://arxiv.org/abs/2406.11235
- [ ] AQLM
  - https://arxiv.org/abs/2401.06118

### Further systems examples

- [ ] SnapKV (KV HBM bandwidth on decode)
  - https://arxiv.org/abs/2404.14469
- [ ] Barbarians at the Gate (meta: AI upending systems research)
  - https://arxiv.org/abs/2510.06189

### Practice after reading

- [ ] Take a slow open model and make a forward pass faster; treat the benchmark as the grade
- [ ] For each win, write one paragraph: which constraint was previously unmodeled?

---

## Phase 7 — Agent work (above the stack)

Not “use agents” — **controlled experiments** on agent behavior / search loops.

- [ ] Study Karpathy-style autoresearch setups (facilitate LLMs creating useful outputs under evaluation)
  - Search starting point: https://github.com/karpathy (look for autoresearch / related experiment harnesses)
- [ ] Read FunSearch / AlphaEvolve-style papers for the research pattern
  - FunSearch: https://arxiv.org/abs/2309.13007
  - AlphaEvolve (DeepMind): https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- [ ] Build a tiny closed-loop experiment: generate → verify → select, with a hard evaluator (tests, benchmarks, or math checks)

---

## Phase 8 — Subject matter (speak the language)

- [ ] Work through a serious LM / DL paper lineage (highly cited classics + recent pretraining theory)
  - One starter list: https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap (or any strong curated LM list)
- [ ] Revisit Vlad’s pretraining lecture materials when you can find the slides/recording from the post
  - Linked from the May 2026 post above
- [ ] Goal: converse fluently about pretraining concepts, scaling, and systems tradeoffs

---

## Phase 9 — Public signal (what actually gets you hired)

- [ ] Ship a public GitHub artifact people might actually use
  - Faster kernel / inference trick, quantization tool, agent eval harness, etc.
- [ ] Write a crisp README: problem → unmodeled constraint → result → repro
- [ ] Optionally: quant research side project (GPU-cheap; trains lateral systems thinking)
- [ ] Re-evaluate every ~6 months: skills, artifacts, and whether you’re digging in the right place

---

## Suggested order (if you want one linear path)

1. Reiner analysis → 2. JAX tutorials → 3. Scaling Book exercises → 4. Capstone (transformer → Chinchilla → Pallas) → 5. Evidence pack / email → 6. Kernel + quant paper series → 7. One public repo → 8. Agent experiment track in parallel once Phase 4 is solid.
