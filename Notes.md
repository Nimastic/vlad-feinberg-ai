# Questions I Have:
- What is RAG
- What is Vector Database

# Get off your soapbox
work at the edges of where frontier labs operate

they spend their time creating LLMs

## What do LLMs require to run, and what are the touchpoints for their outputs?
1. Below the LLM stack: Kernel work. Kernels are small highly optimised GPU programmes used to execute fundamental math operations
2. Above the LLM layer: Agentic loops. Leverage LLM as a grey box, you can harness it to produce useful results
3. Extra: Quant Research is GPU Cheap and great example of how lateral thinking can make an impact

# Kernel Work
Every project needs people who can tune the LLMs at the kernel level.

a little bit of algebra will tell you if your bottleneck is communication, or flops, or memory

Unless you tell your coding agent to watch out for latch boundedness, it will happily analyze just comms/flops/HBM rooflines, for instance. And whatever the next physical constraint is that didn’t get modeled after that.

## Latch boundedness
Latch boundedness is a physical hardware constraint in AI accelerator systems—such as GPUs and TPUs—where data transfers are limited by the physical propagation delays and synchronization of signal latches. In highly optimized deep learning workloads, it means overall execution speed is bottlenecked by the time it takes for data to pass through and settle at register stages rather than raw math computing power (FLOPs) or memory bandwidth (HBM)

## How do you get good at kernel work?
```
How do you get good at kernel work?
You just do it. And it’s just a combination of coding and reasoning about systems. There are plenty of LLMs out there that run slowly on GPUs and TPUs. Just make them run fast. You only need enough accelerators to run a forward pass of the models. You get feedback instantly; it’s the benchmark. Low level coding isn’t the obstacle anymore, it’s awareness and integration of details about physical accelerator devices, as well as a scientific approach to spending time where it matters.
I call this “kernel work” but really it includes a lot more than that:
Actual device kernel development.
Inference stack innovations and CPU optimizations.
Tooling that facilitates R&D around this ecosystem.
```

# Research
- review how researchers make progress in the area
- DSL/Programming Language design comes up to accelerate kernel developmenet example of “working at the edges” of the LLM that has very high impac

## Flash Attention Series of papers
- https://arxiv.org/abs/2603.05451

```
In what now seems like a trite pedagogical example, but was worth detailing in a full paper at the time, the “Flash Attention trick” demonstrated how modelling flops alone might steer one to believe that the unfused attention implementation that is a direct transcription of attention math might come about.

flash unfused

If your model of accelerator runtime is only counting flops, you might spend years trying to change the math of the attention operation itself (at potential quality loss, spending lots of GPU-time validating quality experiments about sparse attention whose results can’t be predicted well ahead of time). By considering a new variable, one that’s obvious now in hindsight, such as memory bandwidth, we realize that the operation can be restructured to avoid materializing intermediate values in slow HBM memory.

flash fused

For folks already familiar with the flash attention trick, I urge you not to be dismissive just because you’re already aware of the memory bottleneck: the important part is identifying unmodelled constraints on system performance, which can always be done by zooming out and viewing a system for generating responses and interacting with the user more holistically.
```

## Quantisation
```
starting with LLM.int8(), a classic which exposes you to a strong set of tricks to walk the quality-performance tradeoff. There’s then another good series of papers from Chris De Sa’s group:
QuiP
QuiP#
QTIP
A slightly different but still powerful leaf in this tech tree is AQLM.
```

# Others:
- On the decoding side, you might look at approaches like SnapKV to reduce the KV HBM BW bottleneck
- The most meta take, assessing systems research itself, is the Barbarians at the Gate paper.

# High Bandwidth Memory (HBM)
High Bandwidth Memory (HBM) is a specialized, ultra-high-performance type of RAM used almost exclusively in high-end graphics cards (GPUs) and AI accelerators (like Google's TPUs).

Instead of laying memory chips flat on a motherboard like traditional RAM, HBM vertically stacks the memory chips and places them directly on the same physical package as the processor. This allows for a massive "highway" of data to travel between the memory and the computing cores all at once, providing the extreme bandwidth required for AI and machine learning workloads.

# Practical next steps:
1. get familiar with this style of Reiner analysis (Dwarkesh lecture with Reiner Pope). You should be able to do this stuff in your sleep https://www.dwarkesh.com/p/reiner-pope
2. Go through Jax tutorials https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
3. Read and do every single exercise in the scaling book https://jax-ml.github.io/scaling-book/

## Exercise:
- Code a ~10M transformer using only jax, flax, optax in free colab using tpu.
- Hard code it to accept digits 0-9, space, +, =.
- Generate a dataset of simple up-to-3-digit numbers, have it learn addition.
- Should train quickly on T4 GPU (pad examples to fixed length)
- Derive Chinchilla laws for this
- see how they differ for dense vs MoE architectures
- Code your solution from scratch in jax by hand if you actually want the learning experience.
- assuming you used jax.lax.ragged_dot for the MoE layer; write a pallas kernel that beats ragged dot for F > D by fusing the up/down projections
- Find a setting where you notice a measurable forward pass speedup and explain why it’s there.
- record yourself doing the jax scaling book exercises above with paper and pencil (all of them)

chatbot convert scanned versions of those results to latex. Send it to me (be ready for me to ask for a random subset of videos of you doing the problems!). Similarly screen-record yourself manually writing the code for implementing the transformer from scratch and deriving the Chinchilla laws.

Send me an email with the scaling law report and the exercise writeup.
