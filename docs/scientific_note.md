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

## GPU Utilization: Synchronous vs. Asynchronous GRPO

When observing systemic GPU utilization via NVML or Ray Dashboard during a default NeMo RL baseline, users will frequently notice extreme fluctuations: utilization bounding between 0% and 100% rather than remaining constantly saturated. 

This oscillating behavior is **completely normal and mathematically expected** for standard Reinforcement Learning algorithms (PPO/GRPO) operating in a multi-engine cluster.

### 1. The Synchronous "Rollout -> Train" Pipeline
By default, the framework operates in a strict, sequential pipeline to ensure mathematical correctness (ensuring the model generating the data is identically synchronized with the model being trained). The distinct phases are:

1. **Rollout (Generation)**: The PyTorch training environment pauses. The `vLLM` workers take control of the GPUs, heavily utilizing them to execute lightning-fast inference and generate thousands of response trajectories to the prompts.
2. **Reward Calculation**: The generated texts are evaluated by reward models or rule-based scoring functions. 
3. **Training (Optimizer Step)**: The policy model switches to training mode. It calculates log-probabilities, forces a massive `loss.backward()` across the trajectory batch, and executes an optimizer step across all FSDP GPUs. During this phase, the vLLM engine is idle.
4. **Weight Sync**: The newly updated weights from the PyTorch training model are synchronously broadcasted over to the vLLM inference engine (via `update_weights_from_collective` RPCs) so vLLM can accurately generate the next loop's trajectories.

Because these massive software engines hand off control of the GPU memory successively, the active utilization chart naturally spikes up and down.

### 2. Optimizing Throughput: Asynchronous Engine
To smooth out GPU utilization and maximize raw cluster throughput, the framework supports **Asynchronous Engine** execution.

If `vllm_cfg.async_engine: True` is enabled in the configuration YAML, the framework violently decouples the generation and training loops:
* The vLLM workers continuously pump out generated data in the background using slightly "older" weights (Stale Policy).
* The PyTorch training workers continuously ingest that asynchronous buffer and execute optimizer steps simultaneously.

This architecture drastically improves Wall-Clock time and keeps GPUs pegged near 100% saturation. However, it introduces **Policy Staleness** (Off-Policy corrections), which is mathematically complex and can sometimes destabilize or completely diverge training, especially on sensitive reasoning tasks. It is highly recommended to stick to `async_engine: False` to establish a healthy, converging baseline before attempting to optimize via asynchronous decoupled mechanics.

## Logging & Observability

### Why do we need W&B / TensorBoard when we have the Ray Dashboard?
They serve completely different, but equally critical, purposes:

1. **Ray Dashboard = Infrastructure & System Health**
   * It tells you **HOW** the hardware is executing. Is the cluster OOMing? Are GPUs at 100% utilization? Which Python worker crashed and what is the stack trace?
   * **The Catch:** As soon as the Ray Cluster shuts down or the Pod is deleted, all of this data is completely erased.

2. **TensorBoard / W&B = Machine Learning Metrics**
   * They tell you **WHAT** the math is doing. Is the Reward actually going up? Is the Policy Mode-Collapsing (KL Divergence exploding)?
   * **The Catch:** These loggers natively track historical data over months. You can overlay the graph from *Experiment #1* (Learning Rate 1e-4) directly on top of *Experiment #2* (Learning Rate 5e-5) to see which one mathematically converges faster. The Ray dashboard cannot compare historical training runs.

## Training Dynamics: The RLHF Alignment Tax (KL Rebound)

During early GRPO training (Steps 1 to 50), it is common to observe sharp accuracy drops that look like catastrophic "crashes" or "overfitting." However, this is usually a mathematically expected behavioral correction triggered by the KL Penalty.

Let's dissect exactly what the graphs are showing around Step 30 to Step 50:

### 1. The KL Penalty "Rubber Band" Snaps
Look at the `train/kl_penalty` graph. From Step 0 to 30, it rises gently, which allows the model freedom to explore new mathematical reasoning paths. But around Step 35, the KL Penalty accelerates steeply, hitting `~0.09`.
* **What this means:** The algorithm detected that the model's new behavior was straying *too far* from the original, coherent English of the base Gemma 3 model. The RL framework violently yanks the "leash" to pull the model back to safety.

### 2. The Model Panics and Ramble-Chops
Look at `train/mean_gen_tokens` and `validation/avg_length`. Right around Step 30, the model's generation length plummets from ~700 tokens down to ~550 tokens.
* **What this means:** Because the KL Penalty suddenly started hurting its "score," the model panicked. It realized that generating long, elaborate `<think>` chains was accumulating too much KL divergence. Its optimal strategy to avoid the penalty was to simply *stop thinking* and chop its answers short.

### 3. Accuracy Plummets
Because the model stopped thinking deeply (due to the KL penalty fear), its `validation/accuracy` and `train/reward` naturally crashed from the high 40%s back down below baseline. A model that doesn't think, doesn't solve math.

### The Conclusion -> Are you ruined?
Absolutely not. This is famously known as the **"RLHF Alignment Tax" or "KL Rebound"**.
1. The model explores and finds high reward (Steps 1-30).
2. It drifts too far linguistically, triggering a massive KL penalty (Step 35).
3. It over-corrects, shortening its thoughts and losing accuracy (Steps 40-50).
4. **The Next Step:** If you let training continue to Step 100+, the model will eventually learn the *balance*. It will figure out how to maintain the deep mathematical reasoning of Step 30 *without* triggering the linguistic KL penalty of Step 35.

**Recommendation:** Do not use the post-crash `step_50` checkpoint for inference. If you want the smartest short-term model, `step_30` is currently your golden checkpoint before the crash. Otherwise, you need to spin the cluster back up and let it train to `step_150+` so it can stabilize past the KL rebound!

## Implicit vs Explicit Reasoning (`<think>` Tags)

When reviewing evaluation generation logs (e.g., `step_30`), users might notice that the model outputs massive blocks of rambling mathematical logic, but **never explicitly wraps its thoughts in `<think>...</think>` XML tags**.

This is a direct result of the Reward Function design in the NeMo-RL Baseline:

### Why is there no `<think>` tag?
In many popular GRPO tutorials (like DeepSeek-R1 open-source replications), engineers manually parse the reasoning output by creating a reward function that explicitly enforces XML tags (penalizing the model if it fails to use them).

However, the `grpo_math_1b.yaml` baseline uses `math_verify_impl: "hf_math_verify"`. This reward model does not care at all about the formatting of the reasoning process. It simply uses advanced regex to scour the entire generation block for a final numerical conclusion (e.g., `The answer is \boxed{X}`) and checks if `X` matches the ground truth.

Because the model is **never explicitly punished for omitting XML tags**, it learns a more "organic" reasoning process:
1. It reads the prompt.
2. It immediately begins talking to itself (the implicit "thinking" phase).
3. Once it calculates the final logic, it prints `The answer is \boxed{X}`.

So at Step 30, the model is absolutely "thinking" (generating massive, multi-hundred token internal monologues where it calculates the math step-by-step), but it doesn't wrap that monologue in `<think>...</think>` tags simply because the reward pipeline never commanded it to.
