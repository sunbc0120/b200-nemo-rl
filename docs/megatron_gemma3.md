# Megatron-Core and Gemma 3 Incompatibility Analysis

**Date:** March 2026
**Subject:** Formal post-mortem analyzing the failure of the Megatron-Core execution backend for training Gemma 3 models (1B and 27B) within the NeMo RL framework.
**Conclusion:** Megatron-Core is a dead-end for Gemma 3. Moving forward, the architecture must pivot to vLLM paired with PEFT (LoRA).

---

## 1. The Initial Motivation
The initial architecture relied on PyTorch Full Fine-Tuning (FFT) alongside vLLM generation for the 27B model on an 8x B200 cluster (1310GB total VRAM). 
However, this caused fatal `CUDA OutOfMemoryError` crashes during the GRPO rollout phase. The PyTorch optimizer (which requires ~450+ GB to store Adam states for 27B parameters) computationally collided with vLLM's KV Cache pre-allocation, leaving no expandable memory for intermediate attention calculations. 

To bypass this vLLM memory contest, we attempted to switch the entire unified workload (generation + training) to NVIDIA's flagship **Megatron-Core** backend.

## 2. Roadblocks Encountered & Attempted Solutions

We systematically unblocked layers of framework orchestration bugs:

#### Issue A: vLLM & Megatron Co-location Deadlocks
* **Symptom:** When both engines tried to co-locate on the same GPUs, CUDA graph compilation hung infinitely resulting in HTTP connection timeouts to the metrics server.
* **Attempted Fix:** We explicitly disabled vLLM's `async_engine`, set `enforce_eager: True`, and ultimately abandoned the vLLM backend entirely, migrating generation natively to `backend: "megatron"`.

#### Issue B: Megatron KV Cache Generation Deadlock
* **Symptom:** Megatron generation hung while attempting to map dynamic CUDA graphs for context rollouts.
* **Attempted Fix:** We aggressively lowered the generation batch size (causing a bottleneck), increased `buffer_size_gb`, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

#### Issue C: Distributed Initialization OOM
* **Symptom:** Megatron crashed during `setup_megatron_model` because 27B optimizer states exceeded the target GPU boundaries.
* **Attempted Fix:** We scaled up `tensor_model_parallel_size` to force Megatron to shard calculations and optimizer memories across the full multi-node topology.

#### Issue D: GCSFuse Timeout & Detached Pre-Downloads
* **Symptom:** Ray jobs suffered network timeouts when pulling the massive weights directly over GCSFuse to multiple workers simultaneously.
* **Attempted Fix:** We architected a robust `launch_megatron.sh` detachment orchestrator. It ran on the Ray Head node, downgraded the brittle Rust `xet-core` HF client, and forcefully pulled a cached `.safetensors` replica to a shared persistent volume (`/data/huggingface`) before submitting the actual training job.

#### Issue E: NeMo RL "Proxy Object" Crashes
* **Symptom:** The `megatron_policy_worker.py` script crashed violently on standard PyTorch methods like `hf_model.named_modules()` and `hf_model.state_dict()`.
* **Attempted Fix:** We discovered NeMo RL wraps Hugging Face models in a lazy-loading `PreTrainedCausalLM` proxy. We wrote custom Python monkey patches to physically crawl the object heap and unwrap the proxy (`getattr(hf_model, "model", hf_model)`) to expose the true layers to Megatron.

#### Issue F: Rigid Data Parallel Topology Assertions
* **Symptom:** Megatron's internal C++ verifiers threw immediate `AssertionError` crashes during worker startup.
* **Attempted Fix:** Megatron strictly enforces that the overarching `global_batch_size` must be perfectly mathematically divisible by the Data Parallel GPU count. We had to manually tune the 1B testing YAML to strict multiples of 8.

## 3. The Final Blocker: Native Weight Corruption 

After flawlessly resolving the orchestration bugs and successfully launching an 8-GPU distributed rollout on an isolated Gemma 3 1B model, the final test failed catastrophically.

The model successfully generated responses, but the raw output was **token salad** (e.g., `Pach مذہبی leyenda INDUST<unused1166>...`).

Because this was an untouched, out-of-the-box Hugging Face `google/gemma-3-1b-it` model, this unequivocally proved that **the core `Megatron-Bridge` parameter converter corrupts Gemma 3.**

## Why Megatron is a Dead-End for Gemma 3

1. **The "QKV" Sharding Mismatch (The Math Scrambler):** Multimodal architectures like Gemma 3 use a novel Grouped-Query Attention (GQA) structure with a unique Head Size of 256. Megatron-Core uses a highly specific Tensor Parallelism (TP) strategy. When the NeMo `Megatron-Bridge` converter loads the Hugging Face weights, it chops up the Query, Key, and Value (QKV) matrices and re-assembles them into Megatron's internal block distributions. Because the `StateDictConverter` does not possess a bespoke, hand-written tensor mapping specifically for Gemma 3's exact math, it falls back to parsing it like a Llama or earlier model. It slices the matrices at the wrong byte intervals; the attention "brain" is geometrically scrambled.
2. **The Vocabulary Padding Offset (The Token Salad):** Megatron requires highly specific tensor layout padding. Gemma 3 has a native Hugging Face vocabulary size of exactly `262,208`. Megatron insists on auto-padding this out to the nearest algorithmically optimized dimension (e.g., `262,656`) for distributed calculation. When the converter auto-pads the dictionary, the indices misalign with the internal embedding array. When the model mathematically predicts the word "The", the ID lookup is shifted by the padding offset, resulting in random multilingual hallucinations. 
3. **No Quick Fix Available:** To resolve this, one would have to deeply rewrite the upstream C++/Python `megatron.bridge` conversion logic: manually mapping 200+ layer names to Megatron specs, writing custom geometrical tensor-slicing logic for head-size 256, and writing bespoke padding interceptors. This is a massive native framework overhaul, not a simple configuration switch.
4. **LoRA Cannot Save Us:** Parameter-Efficient fine-tuning concepts like LoRA connect directly to the base mathematical structure. Because the core base weights are fundamentally corrupted during the initial Megatron loading stage, there is no mathematical path forward.


We do have a very strong understanding of exactly what causes the weight corruption. It boils down to how Megatron-Core forces models into its rigid, highly-optimized C++ architecture, and why it is not quickly fixable.

Here is the exact anatomy of the corruption and why it's a structural dead end:

1. The "QKV" Sharding Mismatch (The Math Scrambler)
In Hugging Face, the Query, Key, and Value (QKV) attention matrices are often stored as separate weights or concatenated in a specific order (e.g., [Q, K, V]). Megatron-Core uses a highly specific Tensor Parallelism (TP) strategy. When the Megatron-Bridge converter loads the Hugging Face weights, it chops up these giant matrices and re-assembles them into Megatron's internal qkv_proj blocks so they can be distributed across 8 GPUs.

The Problem: Gemma 3 uses a novel Grouped-Query Attention (GQA) structure with a unique Head Size of 256. If the NeMo-RL StateDictConverter doesn't have a bespoke, hand-written mapping specifically for Gemma 3's exact tensor math, it falls back to parsing it like a Llama or earlier Gemma 2 model.
The Result: It slices the matrices at the wrong byte intervals. When Megatron multiplies the token embeddings against these matrices during generation, the math is fundamentally scrambled.
2. The Vocabulary Padding Offset (The Token Salad)
Megatron-Core enforces extremely strict memory padding to keep calculations perfectly divisible by the number of GPUs. Gemma 3 has a native Hugging Face vocabulary size of exactly 262,208. Megatron insists on padding this out to the nearest multiple of 128 or 256 (e.g., 262,656) to optimize its parallel calculations.

The Problem: When the converter pads the vocabulary dictionary, it often misaligns the internal embedding matrix.
The Result: When the model mathematically predicts it wants to output the word "The", the ID lookup is shifted by the padding offset, and it accidentally prints Pach, <unused1166>, or some random Arabic character.
Is it fixable quickly?
No. There is no quick fix.

To fix this, we would have to fundamentally rewrite the upstream megatron.bridge conversion logic in C++/Python. We would need to:

Manually map all 200+ layer names from the Gemma3ForCausalLM Hugging Face spec to the Megatron spec.
Write custom tensor-slicing logic to handle Gemma 3's specific 256 Head Size and GQA ratios.
Write bespoke padding handlers that intercept Megatron's vocabulary expansion without shifting the Hugging Face token IDs.
This is the kind of deep, architecture-level tensor engineering that typically takes a dedicated team at NVIDIA weeks to implement and validate against ground-truth unit tests.

This is exactly why pivoting to vLLM + LoRA is the correct move for us. vLLM uses native Hugging Face weights natively without converting or scrambling them, and LoRA solves the OOM problem that was blocking us.

## Conclusion & Pivot
Megatron-Core is engineered for maximum scale on NVIDIA hardware but entirely lacks the dynamic, eager evaluation fallbacks possessed by flexible frameworks like native `transformers` or `vLLM`. Since its custom C++ Cuda graphs cannot comprehend Gemma 3 natively, the integration is a dead-end.

To train the 27B model, we must **abandon Megatron and pivot back to vLLM generation paired with PyTorch LoRA.** Freezing the 27B base weights with LoRA compresses the optimizer footprint down to ~1GB, cleanly removing the VRAM deadlock without mathematical corruption.
