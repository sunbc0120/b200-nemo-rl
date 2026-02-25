# Scientific Notes: Architectural Decisions

## Dataparallel (FSDP2) vs 3D Parallelism (Megatron-Core)

When deploying large language models for Reinforcement Learning (RLHF/GRPO), the choice between native PyTorch FSDP (Fully Sharded Data Parallel) and NVIDIA's Megatron-Core is extremely important. 

### Why FSDP + vLLM is the Default Standard
For the majority of RL tasks utilizing models under **~30 to 70 Billion Parameters** (e.g. Gemma 3 1B/4B/8B, Llama 3 8B), the default combination of PyTorch FSDP2 for training and vLLM for generation is optimal:

1. **Simplicity and Ecosystem Compatibility:** FSDP natively shards parameters, gradients, and optimizer states across your cluster's GPUs. It interfaces seamlessly with HuggingFace `.safetensors`, eliminating the need to convert models into proprietary formats.
2. **Speed of Execution:** vLLM has arguably the fastest generalized inference engine with PagedAttention. Handing off generation tasks to an explicitly co-located (or separately allocated) vLLM backend provides extreme rollout speed.
3. **No 3D Topology Required:** Unlike Megatron, FSDP handles 1D sharding automatically without requiring the user to mathematically balance Pipeline Parallel (PP) and Tensor Parallel (TP) geometries across node boundaries.

*In `grpo_math_1b.yaml`, `megatron_cfg.enabled` is `false` because a 1B model easily fits into VRAM when sharded via FSDP2 (`dtensor_cfg`), and the pipeline safely routes the generations via the `vllm_cfg` backend block.*

### When to Graduate to Megatron-Core
Megatron-Core introduces extreme engineering complexity but unlocks capabilities that native PyTorch simply cannot achieve at unprecedented scales.

1. **Massive Scale (100B+ Parameters):** When a model (or the combined footprint of the Policy, Reference, Value, and Reward models in RLHF) exceeds the collective VRAM of your cluster even when aggressively sharded via FSDP, Megatron becomes mandatory.
2. **Tensor Parallelism (TP):** Megatron physically slices individual matrix multiplications (e.g., Attention Heads) across multiple GPUs located *within the same node* synchronously. This is critical if a single un-sharded layer is larger than a single GPU's 80GB VRAM footprint.
3. **Pipeline Parallelism (PP):** Megatron distributes the model's sequential layers across *different nodes* (e.g., Node A holds layers 1-10, Node B holds 11-20), passing continuous activations between them.
4. **Sequence / Context Parallelism:** For training on massive context windows (128K+ tokens), Megatron can parallelize the context window itself across GPUs, preventing KV-cache OOMs.
5. **Model Flops Utilization (MFU):** Megatron utilizes highly specialized, custom CUDA kernels (Fused Rotary Embeddings, Fused SwiGLU) that can squeeze out a higher percentage of raw TFLOPs from H100s/B200s than generic PyTorch compilers.

**Summary:** Stick to PyTorch FSDP + HuggingFace until you encounter insurmountable Out-Of-Memory errors despite having 8x GPUs. Only when the mathematical boundaries of 1D Data Parallelism are exceeded should you inherit the burden of `.nemo` model conversions and Megatron 3D Parallelism topologies.
