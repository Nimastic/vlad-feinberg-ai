# Notes on KIMIK3

## Industry Context & Significance

- **The DeepSeek & Kimi Catalyst**: Following DeepSeek's January 2025 breakthrough in cost-efficient benchmarking, Kimi K3 has nearly caught up to Anthropic and OpenAI frontier models.
- **Closing the Closed-Source Gap**: Open-source models were widely estimated ~3–6 months behind closed models (DeepSeek's own framing around V4). K3 largely invalidated that assumption—near-equivalent to Claude Fable 5 on many fronts, beating it on some benchmarks, and placing as roughly the third-best model overall while remaining open weights.
- **Scale Milestone**: At 2.8T parameters, ~75% larger than the previous largest open-weight model (DeepSeek V4). Before K3, only a handful of trillion-class models existed (~1T–1.6T); this is the first ~2.8T open-weights release.
- **Model Layer Innovation**: Shifts the broader debate around who leads AI innovation at the model layer and whether Chinese open models will define the next frontier.
- **Inference Economics & Hardware Impact**: Demonstrates how model-layer architectural efficiency fundamentally alters the semiconductor and application layers below, redefining inference cost structures. Reported ~3× cheaper than Fable 5 on long-context reasoning while topping that benchmark class; ~2.5× more scaling efficiency vs. the K2 series from architectural + data + recipe advances.
- **Release**: Weights + a ~47-page technical report open-sourced (as of ~July 27). Moonshot also open-sourced infrastructure pieces (Moon EP, agentic RL runtime)—mirroring DeepSeek's Deep EP open-sourcing pattern, with claimed improvements.

---

## Model Specifications

### Kimi K3
- **Parameters**: 2.8T Parameters
- **Experts**: 896 experts
- **Activated**: 16 activated
- **Activation Rate**: 1.8% Activation
- **Throughput**: Maintains 6x decoding throughput while outperforming status quo models in benchmark scores without sacrificing speed.
- **Positioning**: Built heavily for coding agents and long-horizon / autonomous tasks; strong multimodal (native vision) and visual generation (HTML/landing pages, 3D games, slide decks, Manim-style viz).

### Reported Performance Snapshot
- Coding agents: roughly on par with GPT-5.x "X High" class; #1 front-end coding; strong on private coding evals (e.g. DeepSeek-style private bench); on par with Claude Fable on Terminal Bench 2.1.
- Token efficiency: large jump vs. prior Kimi releases—beats Claude models by ~≥1× on efficiency framing, sits just behind GPT-5.6-class; ~2nd on realistic business workflows.
- Long-context reasoning: topped the relevant benchmark class despite using in-house KDA; still relatively cheap (~≥3× cheaper than Fable 5 in the reported framing).

### Kimi K2
- **Parameters**: 1T Parameters
- **Experts**: 384 experts
- **Activated**: 8 activated
- **Activation Rate**: 2% Activation

### Industry Sparsity Comparisons
Since ~2024, Mixture of Experts (MoE) became the industry standard to reduce compute overhead per token by activating only a subset of the model:
- **Minimax M3**: ~3.1% experts activated
- **Inkling (Thinking Machine)**: ~3.1% experts activated
- **Nemotron 3 Ultra**: ~4.3% experts activated
- **Kimi K2**: 1T parameters, 384 experts, 8 activated (~2.0% activation)
- **Kimi K3**: 2.8T parameters, 896 experts, 16 activated (~1.8% activation) — scaled size more than double over K2 while decreasing activation ratio, making it one of the largest open models with the highest sparsity.

---

## Infrastructure & Architecture

### GPU Cluster Setup & Inter-GPU Communication Overhead
- K3's weights are stored acrossed 6 NVIDIA GPUs, in a B300 DGX setup, leaving very little room for KV cache, activations, runtime buffers
- **Realistic Deployment Setup**: Experts are typically spread across a supernode of 64 GPUs (recommended) or an NVIDIA NVL72 rack setup containing 72 chips in a single rack.
- **Communication Overhead**: Because 896 experts are scattered across a wide array of GPUs, every token passing through the model makes multiple trips between GPUs inside the rack, adding significant communication overhead every time data moves between chips.

### Stable Latent MoE (SLMoE)
- Kimi K3 uses stable latent moe that reduces the dimensions of tokens (building on Latent MoE concepts like Nemotron). Extremely sparse MoE (16/896) is only practical at this scale with latent compression + routing stability.
- **Mechanism**: Token embeddings are down-projected / compressed into a lower-dimensional latent representation before routing.
- **Efficiency Gain**: Passing compressed tokens between GPUs drastically reduces inter-GPU communication overhead and lowers matrix computation overhead—critical when 896 experts are scattered and every token otherwise ships a full hidden state across GPUs.
- **Dataflow**: Token down-projected -> routed to experts (with 2 shared experts always activated by default) -> flows through the expert pool spread across GPUs -> projected back up to the original dimension for Softmax.
- **Router Stability**: The "stable" in Stable Latent MoE refers to stabilizing the router during training when routing tokens across 896 experts—keeping load balanced so a few experts aren't overloaded while others sit idle.

### Open-Sourced Serving / RL Infra (Moon EP & Agentic RL)
- **Moon EP**: Dynamically replicates overloaded experts so every accelerator gets more even work—positioned as an improvement over DeepSeek's Deep EP (already widely adopted).
- **Agentic RL system**: Can suspend and resume long-running micro-VM environments without losing state—important for multi-step agent training loops.

### Quantile Balancing & Router Selection
- Selecting only 16 out of 896 experts per token is a non-trivial task where slight score variations can throw off training equilibrium.
- Quantile Balancing and LSAT score helps select which expert to use
- **Comparison with Status Quo**: Unlike DeepSeek V3 or Nemotron 3 Ultra (which use auxiliary loss / bias terms to nudge the router by penalizing overused experts and promoting underused ones), K3 uses **Quantile Balancing**.
- **The LSAT Analogy**: Quantile Balancing allocates experts directly from the distribution of router scores—similar to grading candidates on an LSAT percentile curve rather than raw fixed scores—allowing relative selection across the expert pool.

- An expert in a Mixture of Experts (MoE) large language model is an individual, smaller feed-forward sub-network that processes specific tokens, replacing a single dense layer to make training and running the model faster.

---

## Attention Mechanisms & Sequence Modeling

### Kimi Delta Attention (KDA) & Linear Attention
- How to make growing compute demand and high context window more manageable as model scales?
- Standard Transformer attention has quadratic complexity O(N²), making 1M context windows computationally prohibitive without architectural changes. Linear attention transforms quadratic complexity into subquadratic/linear O(N).
- **Why it matters for agents**: Coding agents with huge repos, repeated tool calls, and long reasoning trajectories make KV-cache / attention cost a primary bottleneck—pushing labs toward sparse attention (e.g. DeepSeek V4) or linear attention (KDA).
- **KDA Origin**: Kimi Delta Attention (KDA) was released around October/November 2025 (Chimera Linear paper era; ~9 months prior to K3's integration at scale). Matured from research variant into a reliable frontier attention design.
- **Core idea (intuition)**: Softmax attention lets every token directly retrieve from any prior token (expressive, but quadratic). Linear attention compresses history into a fixed-size recurrent state—update on each new token, then read—so per-layer memory does not grow like a full KV cache. Historical weakness: associations interfere inside the finite state, and precise per-token retrieval is lost; early linear attention often underperformed full attention even on short LM.
- **Mamba 2 Comparison**: Models like Nemotron 3 use Mamba 2, which replaces some attention layers with recurrent state space memory (predefined memory cell where new info updates memory and old memory decays according to a schedule).
  - Predefined memory cell, new info comes in, old memory decays
  - Sounds like recurrent neural network

#### Linear Attention Formula
Formula: S_t = α_t S_(t-1) + V_t K_tᵀ

##### Variable Breakdown
- **S_t**: The new hidden memory state (typically a matrix) at the current time step t.
- **α_t (circled)**: The decay factor or forget gate. It dictates how much of the previous memory should be retained. This mechanism prevents the memory matrix from growing infinitely and helps the model "forget" older, less relevant data to prioritize recent context.
- **S_(t-1) (yellow underline)**: The previous memory state from time step t - 1.
- **V_t K_tᵀ (blue underline)**: The new information being written into memory at the current step. This is the mathematical outer product of the Value (V_t) and Key (K_t) vectors for the current token.

##### Why It Might Look Familiar
If you have been studying deep learning, large language models (LLMs), or sequential modeling, this is a core concept for making attention mechanisms highly efficient.

In a standard Transformer model, the attention mechanism has a quadratic computational complexity—O(N²)—meaning it gets exponentially slower and more memory-hungry as the context window grows.

By reorganizing the attention mechanism into a recurrent formula (as seen in your image), models can compress past information into the fixed-size state S_t. This allows them to process long sequences with linear complexity—O(N)—which drastically speeds up both training and inference. When a new Query vector (Q_t) arrives, the model simply multiplies it by this memory matrix to retrieve the relevant information, mimicking the behavior of standard attention at a fraction of the computational cost.

*recurrent state update equation used in Gated Linear Attention models, as well as similar efficient sequence-modeling architectures like Fast Weight Programmers, RetNet, or RWKV.*

---

### Gated Delta Net (GDN) & Kimi Delta Attention (KDA)

From new info, find error, correct memory cell: Gated Delta Net (GDN)

New Formula:
- improve efficiency and memory

#### 1. The Delta Rule Update (`image_b1f44c.png`)
This image shows the fundamental logic of an error-driven memory update.

S_t = α_t S_(t-1) + β_t K_t (V_t - S_(t-1)ᵀ K_t)ᵀ

##### Variable Breakdown
- **α_t S_(t-1)**: The decayed previous memory state, same as before.
- **β_t**: A new scalar (often a learning rate or learned gate) that controls the strength of the new update.
- **S_(t-1)ᵀ K_t**: This acts as a prediction. The model multiplies the previous memory (S_(t-1)) by the current Key (K_t) to see what value the memory currently associates with that key.
- **(V_t - S_(t-1)ᵀ K_t)ᵀ (labeled "error")**: This is the critical difference. It subtracts the model's "prediction" from the actual true Value (V_t).
- **K_t × error**: The memory is updated by taking the outer product of the Key and the error, rather than the Key and the raw Value.

**Why this matters**: In the simpler formula from your previous image, the memory matrix S_t can grow infinitely large and noisy because it just keeps adding V_t K_tᵀ at every step. The Delta Rule prevents this. If the memory already perfectly predicts the Value for a given Key, the error is 0, and nothing new is added, keeping the memory clean and efficient.

#### 2. Kimi Delta Attention (`image_b1f3d1.png`)
This image shows Kimi Delta Attention (KDA), which is an algebraic rearrangement and optimization of the Delta Rule formula shown above, designed specifically for efficient computation in modern Large Language Models (like those developed by Moonshot AI).

S_t = (I - β_t K_t K_tᵀ) Diag(α_t) S_(t-1) + β_t K_t V_tᵀ

##### Variable Breakdown
- **I**: The Identity matrix.
- **K_t K_tᵀ**: The outer product of the Key with itself.
- **Diag(α_t) S_(t-1)**: The previous state, scaled by the forget gate α_t (formatted here as a diagonal matrix operation).
- **(I - β_t K_t K_tᵀ)**: This acts as a dynamic "forgetting" matrix. By subtracting the Key's projection from the Identity matrix, the model actively erases information in the direction of the current Key before writing new information.
- **β_t K_t V_tᵀ**: The standard new information being written to the state.

**Why this matters**: If you expand the math in the first formula, it mathematically transforms into this KDA formula. Writing it in this specific matrix format (separating the decay/erasure term from the new write term) makes it much faster to compute on GPUs. It allows linear attention models to handle massive context windows without suffering from the "memory overflow" or hallucination issues that plague simpler recurrent models.

**GDN vs. KDA Distinction**: Gated Delta Net uses one forgetting rate per attention head—every feature channel in that head decays together even if they store information with different useful lifespans. KDA's main change is straightforward but high-leverage: **per-channel independent decay** instead of a head-level scalar. One channel can rapidly drop temporary info while another preserves state over a much longer sequence—controlling memory lifetime at feature level without enlarging the recurrent state. Same compression mechanism as linear attention; better organization of what gets stored/forgotten.

**What KDA still cannot do**: It does not restore exact full-attention retrieval (original keys/values are not all retained). Hence the hybrid stack below—same pattern as other modern linear-attention frontiers.

Linearity helps with inference and training

#### 3. Hybrid Layer Architecture
- K3 interleaves Kimi Delta Attention at a **3:1 ratio** based on ablation studies: 3 KDA layers to 1 global attention layer (Gated Multi-head Latent Attention / Gated MLA).
- **Efficiency claim**: Only 1/4 layers need a conventional full-attention KV cache → up to ~**75% KV-cache reduction**; up to ~**6× decoding throughput** at 1M context vs. standard-attention baseline.

#### 3b. Chimera Linear Precursor (ablations that justified scaling KDA)
Early KDA paper / Chimera Linear setup: ~48B MoE with ~3B active params, trained on ~1.4T tokens. Chimera Linear did not merely match full attention—it beat it in their reported runs:
- MMLU Pro: ~+4 points vs. MLA baseline
- RULER (128K): ~+3 points, with ~4× decoding acceleration
- 1M context TPOT: **11.48 ms → 1.84 ms** (~**6.3×** decode speedup)
- Quality gains also showed up under RL2—making KDA a strong candidate to scale into K3

#### 4. Hardware-Efficient Chunkwise Parallel Training
- Standard RNNs suffered from training inefficiency due to sequential temporal dependence.
- K3 adopts a **hardware-efficient chunkwise parallel algorithm** (building on GDN chunkwise training) that groups multiple sequence steps together, allowing training to be parallelized across GPU hardware despite temporal dependencies.

---

## Attention Residuals ("Plumbing" & Interconnect Optimization)

- **Origin**: Attention Residual research was published around March 2025 (4 months prior to K3).
- **The Plumbing Analogy**: Residual networks act like unsexy "plumbing" in model architecture, creating communication streams so information passes through deep layers without pressure or distortion building up.
- **Standard Residual Limit**: In standard Transformers, deeper layers become diluted and lose touch with earlier layers—each layer typically receives a fixed sum of prior computations.
- **Attention Residual Innovation**: Lets each layer use softmax attention over earlier layer outputs—selectively weighting whichever earlier representations are useful for the current token. Intuition: attention across the *model depth / stack*, not only across the token sequence; stronger learning dynamics.
- **Logical Block Grouping in K3**: K3 groups attention residuals into larger logical blocks, preventing inter-layer connection overhead from becoming too expensive and drastically reducing GPU interconnect strain while increasing model expressiveness.

### Autonomous Kernel Optimization Demo
- Researchers gave K3 the production-sized attention-residuals kernel and let it iterate autonomously for up to ~24 hours (profile → edit → benchmark → continue).
- After ~15 hours: designed a new two-phase algorithm, fused ops without changing numerics, cut runtime **283.6 ms → 114.4 ms** (~**60%** speedup). Final kernel performance trends reportedly close to Claude Fable 5 across related kernel benches.

---

## Mixture of Experts (MoE) Deep Dive

### What an Expert Does
- **Sub-network structure**: An expert is a standard feed-forward neural network block inside the transformer layers.
- **Token-level routing**: A gate network (or router) directs individual words or tokens to only two or a few relevant experts instead of using the whole model.
- **Granular specialization**: Rather than separating by human topics like "coding" or "math," experts naturally specialize in low-level grammar, syntax, or punctuation patterns

all experts are unique from each other.

### Why Every Expert is Unique
- **Random initialisation**: Each expert starts with slightly different, randomized weights before training begins.
- **Specialised training paths**: The router learns to send different types of data to different experts, causing them to develop unique parameter

weights are parameters configured through training that directly define an expert's capabilities and characteristics, but they are adjusted mathematically by algorithms, not manually by engineers.

### How Model Weights Work in MoE
- **Mathematical configurations**: Weights are numerical values (fractions and decimals) inside the expert's neural network that determine how strongly a signal is passed from one neuron to the next.
- **The root of intelligence**: Everything an expert "knows"—its unique style, its ability to parse code, or its grasp of grammar—is entirely stored within these weights.
- **The routing factor**: The gating network (router) also has its own learned weights, which it uses to judge which expert's characteristics are best suited for an incoming token.

---

## Training vs. Inference & Hyperparameters

### Manual Settings vs. Automated Training
To understand how weights get their characteristics, it helps to separate what humans do from what the machine does:

| Feature | Human Engineers (Manual Settings) | The Training Process (Automated) |
| :--- | :--- | :--- |
| **What it modifies** | Hyperparameters (e.g., number of experts, size of layers, learning rate). | Weights and Biases (the trillions of specific numerical connections). |
| **How it works** | Set before training starts; these define the architecture and layout of the "factory." | Adjusted automatically using calculus (gradient descent) during training. |
| **The Result** | Dictates how many experts exist (e.g., 300 experts) and how fast they are allowed to learn. | Dictates what each expert actually specializes in based on millions of text examples. |

### Temperature & Generation Controls
Temperature is a manual hyperparameter that controls the randomness and creativity of an LLM's output by scaling the mathematical confidence of its predictions before picking a word.

The training builds a fixed "map" of mathematical possibilities, while parameters like temperature act as a lens that changes how you look at that map.

Training and generation (inference) are two completely separate stages. Training creates the model's brain, while settings like temperature simply control how that brain makes decisions in real time.

---

## Context Length, Vision & Post-Training

### Million-Token Context (not just RoPE tweaks)
- **No explicit positional encoding**; context was scaled progressively: **8K → 64K → 256K → 1M**.
- Training included synthetic long-context tasks where required information was intentionally scattered across the full context—forcing genuine long-range retrieval rather than local shortcuts.

### Native Multimodal Training (Moon ViT V2)
- Vision trained **natively from the start**, not bolted on via a pretrained SigLIP-style encoder after the fact.
- **Moon ViT V2** trained from scratch alongside the LM with next-token prediction; reported as more stable while still matching pretrained-vision baselines.
- Strong visual/generation demos: HTML sites & landing pages, 3D games/engines, research slide decks / infographics, Manim / 3Blue1Brown-style viz.

### Post-Training Recipe Highlights
- **Multi-teacher on-policy distillation** (common among latest open weights): trained **nine specialized teachers** across coding, general agents, and reasoning effort tiers (low / high / max), then merged back into one model so K3 can **dynamically scale reasoning effort**.
- **Multi-harness agent training**: trained across environments imitating Claude Code, Codex, OpenClaw, Hermes, etc.—reduces overfitting to one tool format / agent scaffold.
- **Deployment-aware post-training** (similar spirit to DeepSeek V4):
  - Quantization-aware training with **MXFP4** expert weights and **MXFP8** activations
  - Pretraining prediction layer converted into an **Eagle-3** speculative decoding draft model, trained to maximize drafted-token acceptance

### Autonomy Showcase: Chip Design
Past "AI writes its own training code/kernels" demos—K3 reportedly designed hardware for itself in a single **~48-hour** autonomous run using open-source tools on a **9-gate 45nm** library:
- Within **~4 mm²**: packed **1.46M** standard cells, **0.277 MB** SRAM, and an **8-bit 4-MAC** array with fused dequantization
- Simulated decode throughput serving a nano model: **~8,700 tokens/sec**

---

## Known Limitations (from Moonshot's own notes)

Moonshot published transparent self-critiques—useful signal on where evals diverge from product feel:
1. **Thinking history must stay intact** — performance degrades if the reasoning/thinking trajectory is stripped or truncated (likely tied to how heavily RL shaped the thinking process).
2. **Assumes intent when unclear** — trained for long-horizon / autonomous work, so it may fill in goals rather than ask; autonomy training can backfire in ambiguous interactive UX.
3. **UX / "vibes" gap** — competitive on many evals, but they acknowledge a noticeable user-experience gap vs. Claude Fable 5 and GPT-5.6 Soul—admitting evals ≠ product feel.

---

## Pricing & Inference Economics

- **API Pricing**: $3.00 per 1 million input tokens / $15.00 per 1 million output tokens.
- **Infrastructure Impact**: Highly optimized open models like Kimi K3, designed for extreme efficiency, can potentially run even faster and more cost-effectively when deployed on top-tier US hardware and cloud infrastructure.
- **Demand / capacity pressure**: Around launch, Moonshot paused new subscription sign-ups to concentrate compute on existing users, and introduced usage-type subscriptions to allocate GPUs between power users and casual usage—another signal that frontier open weights are shifting inference demand curves.
