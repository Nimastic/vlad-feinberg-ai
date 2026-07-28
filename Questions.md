# Questions & Answers

## 1. What is Quantization, how does it work, and why is it unique?

**Answer:**
Quantization is a model compression technique that reduces the numerical precision of weights and activations (e.g., from 16-bit floating-point `FP16` down to 8-bit `INT8` or 4-bit `INT4` integers) rather than deleting parameters.

- **How It Works:**
  - **Lower Precision:** Converts high-bit formats (`FP16`/`BF16`) to lower-bit formats (`INT8`, `INT4`, or sub-byte).
  - **Smaller Footprint:** Reduces memory (VRAM) requirements by 2x to 4x or more.
  - **Faster Speed:** Accelerates inference because transferring smaller data types through GPU/CPU memory bandwidth takes less time.

- **Why It Is Unique:**
  - Unlike **pruning** (which deletes weights) or **distillation** (which trains a smaller student model), quantization keeps the exact original network architecture intact.
  - It trades a tiny, often negligible amount of mathematical precision for massive hardware efficiency.

---

## 2. What is the difference between 4-bit (e.g., Q4_K_M) and 8-bit (e.g., Q8_0) quantization formats?

**Answer:**
- **8-bit Quantization (**`Q8_0`**):**
  - **Precision:** High retention of original `FP16` model capability and accuracy (~99%+ of baseline).
  - **VRAM Savings:** ~50% reduction in VRAM compared to `FP16`.
  - **Use Case:** Best when VRAM is sufficient and zero loss in reasoning quality is required.

- **4-bit Quantization (**`Q4_K_M`**):**
  - **Precision:** Slight degradation in precision, but `k-quant` variants (like `Q4_K_M` - medium mixture) strategically quantize critical layers at higher bits to preserve quality.
  - **VRAM Savings:** ~75% reduction in VRAM compared to `FP16` (e.g., a 70B model fits into ~40 GB VRAM instead of 140 GB).
  - **Use Case:** Standard sweet spot for running larger models locally on consumer hardware.

---

## 3. What is Retrieval-Augmented Generation (RAG)?

**Answer:**
Retrieval-Augmented Generation (RAG) is an architecture pattern that enhances an LLM's output by retrieving relevant factual context from an external database or document repository before generating a response.

- **How It Works:**
  1. **Query & Retrieval:** When a user asks a question, the system searches a knowledge base (often using vector embeddings) to find relevant snippets/documents.
  2. **Prompt Augmentation:** The retrieved snippets are injected into the LLM's prompt context window alongside the user's original query.
  3. **Generation:** The LLM generates an accurate answer grounded directly in the provided reference materials.
- **Benefits:** Prevents hallucinations, allows access to private/up-to-date data, and eliminates the need for expensive model retraining or fine-tuning.

---

## 4. What is a Vector Database?

**Answer:**
A Vector Database is a specialized database designed to store, index, and query high-dimensional vector representations (embeddings) of data (text, images, audio, etc.).

- **How It Works:**
  - Text or media is converted into numerical vectors (dense arrays of floats) by an embedding model.
  - The vector database indexes these vectors to perform fast Approximate Nearest Neighbor (ANN) searches (e.g., using Cosine Similarity or Euclidean Distance).
- **Role in AI Systems:**
  - Serves as the core retrieval engine for **RAG** systems to perform semantic search (matching content by meaning rather than exact keywords).
  - Provides long-term memory for AI agents.

---

## 5. What is a KV Cache, how does it work, and why is it an inference bottleneck?

**Answer:**
A **KV (Key-Value) Cache** is a memory optimization technique used during autoregressive LLM decoding to store the Key ($K$) and Value ($V$) projection tensors calculated for all preceding tokens in a sequence.

- **How It Works:**
  - LLMs generate text token-by-token (autoregressively).
  - To produce each new token, the current Query token must attend to all previous tokens in the context window.
  - Instead of recomputing $K$ and $V$ tensors for past tokens at every generation step (which would require $O(N^2)$ redundant matrix multiplications), the model computes $K$ and $V$ once for each token and caches them in GPU memory (HBM).
  - On step $N+1$, the model only computes $K_{N+1}$ and $V_{N+1}$ and appends them to the cache.

- **Why It Becomes an Inference Bottleneck:**
  1. **VRAM Memory Footprint:** The size of the KV cache grows linearly with context length ($\text{len\_ctx}$) and batch size ($B$). At long contexts (e.g., 100k+ tokens), the KV cache size can surpass the memory needed to store the model weights themselves.
  2. **Memory Bandwidth (HBM) Limit:** On every single decode step, the GPU must fetch the entire accumulated KV cache from HBM. As context grows, loading the KV cache turns the system from compute-bound to memory-bandwidth-bound, imposing a speed ceiling and causing price tier jumps (e.g., Gemini's +200k context pricing kink).
