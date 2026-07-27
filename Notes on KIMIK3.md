# Notes on KIMIK3

## Industry Context & Significance

- **The DeepSeek & Kimi Catalyst**: Following DeepSeek's January 2025 breakthrough in cost-efficient benchmarking, Kimi K3 has nearly caught up to Anthropic and OpenAI frontier models.
- **Model Layer Innovation**: Shifts the broader debate around who leads AI innovation at the model layer and whether Chinese open models will define the next frontier.
- **Inference Economics & Hardware Impact**: Demonstrates how model-layer architectural efficiency fundamentally alters the semiconductor and application layers below, redefining inference cost structures.

---

## Model Specifications

### Kimi K3
- **Parameters**: 2.8T Parameters
- **Experts**: 896 experts
- **Activated**: 16 activated
- **Activation Rate**: 1.8% Activation
- **Throughput**: Maintains 6x decoding throughput while outperforming status quo models in benchmark scores without sacrificing speed.

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
- Kimi K3 uses stable latent moe that reduces the dimensions of tokens (building on Latent MoE concepts like Nemotron).
- **Mechanism**: Token embeddings are down-projected / compressed into a lower-dimensional latent representation before routing.
- **Efficiency Gain**: Passing compressed tokens between GPUs drastically reduces inter-GPU communication overhead and lowers matrix computation overhead.
- **Dataflow**: Token down-projected -> routed to experts (with 2 shared experts always activated by default) -> flows through the expert pool spread across GPUs -> projected back up to the original dimension for Softmax.
- **Router Stability**: The "stable" in Stable Latent MoE refers to stabilizing the router during training when routing tokens across 896 experts.

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
- **KDA Origin**: Kimi Delta Attention (KDA) was released around October 2025 (9 months prior to K3's integration at scale).
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

**GDN vs. KDA Distinction**: Gated Delta Net could not independently control how information is retained and forgotten. KDA replaces the decay control in GDN with a fine-grained function allowing channel-by-channel control over retention and erasure rates.

Linearity helps with inference and training

#### 3. Hybrid Layer Architecture
- K3 interleaves Kimi Delta Attention at a **3:1 ratio** based on ablation studies: 3 KDA layers to 1 global attention layer (Gated Multi-head Latent Attention / Gated MLA).

#### 4. Hardware-Efficient Chunkwise Parallel Training
- Standard RNNs suffered from training inefficiency due to sequential temporal dependence.
- K3 adopts a **hardware-efficient chunkwise parallel algorithm** (building on GDN chunkwise training) that groups multiple sequence steps together, allowing training to be parallelized across GPU hardware despite temporal dependencies.

---

## Attention Residuals ("Plumbing" & Interconnect Optimization)

- **Origin**: Attention Residual research was published around March 2025 (4 months prior to K3).
- **The Plumbing Analogy**: Residual networks act like unsexy "plumbing" in model architecture, creating communication streams so information passes through deep layers without pressure or distortion building up.
- **Standard Residual Limit**: In standard Transformers, deeper layers become diluted and lose touch with earlier layers.
- **Attention Residual Innovation**: Allows the current layer to selectively pull information from earlier residual states.
- **Logical Block Grouping in K3**: K3 groups attention residuals into larger logical blocks, preventing inter-layer connection overhead from becoming too expensive and drastically reducing GPU interconnect strain while increasing model expressiveness.

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

## Pricing & Inference Economics

- **API Pricing**: $3.00 per 1 million input tokens / $15.00 per 1 million output tokens.
- **Infrastructure Impact**: Highly optimized open models like Kimi K3, designed for extreme efficiency, can potentially run even faster and more cost-effectively when deployed on top-tier US hardware and cloud infrastructure.
