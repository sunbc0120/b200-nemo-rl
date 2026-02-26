# NeMo RL Cluster Deployment Guide

This guide details how to seamlessly deploy a Ray cluster on Google Kubernetes Engine (GKE) configured explicitly for training models like Gemma 3 using the NVIDIA NeMo-RL framework.

The cluster provisions **B200** Spot instances (8 GPUs per node) and mounts a high-performance Google Cloud Storage (GCS) bucket natively using the Fuse CSI driver.

## 📋 Prerequisites
- Authenticated `kubectl` context pointing to your target GKE cluster.
- The `ray-cluster-b200-nemo.yaml` manifest.
- A valid HuggingFace Access Token exported into your terminal (`export HF_TOKEN="hf_..."`) to download gated Gemma 3 weights.

## 🚀 Deployment Instructions

### 1. 🧹 Ensure a Clean State
If you have an existing or stale Ray cluster running on the target node pools, you must tear it down first to free up the expensive GPU resources and avoid scheduling deadlocks.

```bash
# Delete any existing testing cluster
kubectl delete raycluster ray-cluster-b200

# Delete the target NeMo cluster if explicitly restarting
kubectl delete raycluster ray-cluster-b200-nemo
```

### 2. 📄 Apply the Manifest
The manifest defines a RayCluster Custom Resource (CRD). The KubeRay operator will intercept this and begin provisioning the Head and Worker pods.

```bash
kubectl apply -f manifests/ray-cluster-b200-nemo.yaml
```

*Note: The YAML explicitly configures `replicas: 2` under `workerGroupSpecs`, requesting 16 B200 GPUs total. Adjust this value in the YAML if you need a different scale.*

### 3. ⏳ Monitor the Cold Boot
Ray pods initialize via an `initContainer` (`install-ray`) which downloads the raw python orchestrator binaries into a temporary volume before the main container starts. This takes roughly 45-60 seconds.

Watch the pod spin-up process:
```bash
kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -w
```
Wait until both the `ray-head` and `ray-worker` pods reach the `Running` state and register `Ready` (e.g., `4/4` and `3/3`).

### 4. 🩺 Verify Cluster Health
Once the pods are running, connect to the Head node and query the native Ray daemon to ensure all worker GPUs are actively registered in the pool.

```bash
# Fetch the head pod ID
HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head --no-headers | awk '{print $1}' | head -n 1)

# Execute 'ray status' (ensuring the pip-installed PATH is targeted)
kubectl exec $HEAD_POD -c ray-head -- /bin/bash -c "export PATH=/tmp/ray/packages/bin:\$PATH; ray status"
```

If successful, you will see a console output reporting **232 active CPUs** and **16.0 active GPUs** (assuming 2 replicas).

## 5. 🖥️ Connect to the Ray Dashboard
You can monitor the cluster's GPU utilization, view logs, and track the real-time progress of jobs via the Ray Dashboard. Set up a local port-forward to the Ray Head service using `kubectl`:

```bash
kubectl port-forward svc/ray-cluster-b200-nemo-head-svc 8265:8265
```

Once that is running, open your web browser and navigate to: **[http://localhost:8265](http://localhost:8265)**

## 6. 📈 Real-time TensorBoard Analytics (GCS Backed)
Because the `grpo-*.yaml` is configured to write logs to `/data/nemo-rl-logs`, we are streaming metrics directly into Google Cloud Storage via the Kubernetes FUSE mount. You can natively run TensorBoard on the cluster to track these logs!

Spin up the TensorBoard daemon in the background of the Head Pod:
```bash
RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o 'jsonpath={.items[0].metadata.name}')
kubectl exec $RAY_HEAD_POD -- bash -c "nohup tensorboard --logdir=/data/nemo-rl-logs/ --host=0.0.0.0 --port=6006 > /tmp/tensorboard.log 2>&1 &"
```

Then, execute a local port-forward to your machine:
```bash
kubectl port-forward $RAY_HEAD_POD 6006:6006
```
Open **[http://localhost:6006](http://localhost:6006)** in your browser. As VLLM processes chunks of logic and the rewards calculate, this dashboard will update natively from the dataset in your Cloud Storage Bucket!

### Navigating the TensorBoard UI

**Step 1: Clean Up the View**
On the left side of your screen, under the **Run** list, you see `exp_001`, `exp_002`, etc. Uncheck all the boxes except for your current active run. This will hide all the old, aborted test runs and leave behind one clean line on the graphs.

**Step 2: The Only 3 Graphs You Need to Care About**
At the very top of the page, click the **SCALARS** tab. Search for and pin these three metrics:

1. 🏆 **Average Reward (or Accuracy)** -> `metrics/reward` or `metrics/avg_reward`
   * **What it means:** This is the ultimate "is it working" line. It combines how well your model is formatting the `<think>` tags and whether it actually got the right math answer. 
   * **What you want:** A steady climb up and to the right from your baseline (e.g., 30%).

2. 🧠 **Mean Generation Length** -> `metrics/mean_generation_length` (or `generation_tokens`)
   * **What it means:** How long Gemma 3 is thinking before it spits out a final answer. 
   * **What you want:** In reasoning tasks, you *want* this line to go up over time! It proves the model is learning to spend more tokens exploring the problem logically before rushing an answer.

3. 📉 **KL Divergence Error** -> `metrics/kl_error` (or `policy_kl`)
   * **What it means:** This measures how much the model's "brain structure" has drifted from the original, un-trained state.
   * **What you want:** A low, stable line! If this spikes wildly and stays high, it means the model has "mode-collapsed" (it found a hack to cheat the reward system by forgetting English and just spamming XML tags infinitely). If the KL penalty is working, this will flatline ideally.

*(Everything else—like `samples_per_sec` or `policy_training_lck`—are just hardware diagnostics. Keep your eyes on the Reward and Generation Length!)*

## 7. 🚀 Job Execution (GRPO Gemma 3)
Your infrastructure is now fully stabilized and distributed! You do **not** need a local Python environment or a local Ray installation.

To initiate the training pipeline, we leverage a native bash wrapper (`launch_grpo.sh`) that uses `kubectl exec` to proxy into the Head pod and trigger Ray's native JobSubmissionClient from *inside* the cluster:

```bash
# Ensure your token is exported locally!
export HF_TOKEN="your_huggingface_token"

# Execute the launcher
./scripts/launch_grpo.sh
```

This script will automatically:
1. Find your Ray Head Pod.
2. Inject your `$HF_TOKEN`, `setup_nemo_rl.sh`, and custom `grpo-gemma*.yaml` manifests directly into the Head node's `/workspace`.
3. Broadcast the codebase to all Workers via `ray job submit` and commence OpenMathInstruct-2 GRPO modeling!

## 8. 🔍 Inspecting Raw Generation Outputs

The framework aggregates all generation trajectories, prompt inputs, and computed rewards at every step. Because they are not currently synced to an external dashboard, they reside natively inside the Ray Head Pod.

You can view the raw generated strings and exactly how the LLM reasoned its answers by executing this command locally in your terminal. It targets the `train_data_step[X].jsonl` file and uses `jq` to nicely format the 'prompt' and the generated 'text' columns:

```bash
RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o 'jsonpath={.items[0].metadata.name}')

kubectl exec -it $RAY_HEAD_POD -- bash -c "tail -n 1 /opt/nemo-rl/logs/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/exp_008/train_data_step1.jsonl | jq '. | {data_source, prompt: .prompts[0], generated_text: .texts[0], rewards: .rewards}'"
```

Running that will output a clean JSON block showing you the exact math problem, exactly what Gemma printed inside its `<reasoning>` and `<answer>` tags, and the reward score it received for that specific trajectory!

To view different steps as training progresses, you can simply change `train_data_step1.jsonl` to `train_data_step60.jsonl` (or whichever step you want to inspect). *Note: Ensure you update `exp_008` to match your actual experiment run directory.*

## 9. 🪙 Gemma 3 Special Tokens Handling

Gemma 3 introduces several new special control tokens. NeMo-RL's `math_hf_data_processor` handles them natively via HuggingFace's `apply_chat_template`:

1. **`<start_of_turn>` and `<end_of_turn>`**: Injected automatically around user/model prompts when `add_generation_prompt=True` is executed.
2. **`<|thought|>` (Reasoning Mode)**: This is turned off by default in the chat template. By modifying `policy.tokenizer.chat_template_kwargs: {enable_thinking: true}` in your YAML config, this token is injected to force the model into its internal reasoning phase.
3. **`<|file_separator|>` and `<|n_th_step|>`**: Strictly used for multimodal layout or tool-calling steps. NeMo-RL safely ignores these during text-only generation.


## 10. ♻️ Restarting a Run from Scratch (Disabling Auto-Resume)

By default, NeMo-RL will **implicitly resume** from the last available PyTorch checkpoint if it detects an existing `results/<experiment_name>` folder on the local disk. There is no explicit `--no-resume` flag in the YAML config.

If you abort a training job and want to start a brand new run with the exact same experiment name, you **must delete the old results folders** from both the Head Pod AND all Worker Pods (because PyTorch FSDP saves the fragmented model shards onto the local disks of the worker pods).

Run these commands locally to completely flush the cluster's local disks before launching a fresh job:

```bash
EXPERIMENT="grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200"

# 1. Wipe the Head Pod's configuration checkpoint
RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec $RAY_HEAD_POD -- bash -c "rm -rf /opt/nemo-rl/results/$EXPERIMENT /opt/nemo-rl/logs/$EXPERIMENT"

# 2. Wipe every Worker Pod's FSDP Shard weights
WORKER_PODS=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}')
for POD in $WORKER_PODS; do
    kubectl exec $POD -- bash -c "rm -rf /opt/nemo-rl/results/$EXPERIMENT /opt/nemo-rl/logs/$EXPERIMENT"
done

echo "Cluster wiped. Safe to launch!"
```


## 11. 💾 Converting FSDP Checkpoints (PyTorch DCP to HuggingFace)

Because this pipeline uses Fully Sharded Data Parallel (FSDP) to train across all 8 GPUs, PyTorch natively saves the model checkpoints in **Distributed Checkpoint (DCP)** format.

### What is DCP?
Instead of a single massive `model.safetensors` file, the DCP format shards the model weights and optimizer states into dozens of smaller `.pt` files. This allows PyTorch FSDP to asynchronously write checkpoints from all 8 GPUs simultaneously without choking the node's memory or encountering immediate lock-contention.

However, VLLM and standard HuggingFace inference pipelines cannot natively read DCP shards. You must merge them back into a unified HuggingFace model.

### How to Merge
NeMo-RL ships with a native script to perform this merge on the cluster. After the training completes, run the following command directly on the Ray Head node to compile a specific `step_X` checkpoint into a `.safetensors` model ready for serving:

```bash
kubectl exec -it $RAY_HEAD_POD -- bash -c "python /opt/nemo-rl/examples/converters/convert_dcp_to_hf.py \
    --config /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_100/policy/config.yaml \
    --dcp-ckpt-path /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_100/policy/weights \
    --hf-ckpt-path /data/nemo-rl-results/hf_merged_gemma3_step100"
```

## 12. 🎯 Adding Custom Reward Functions

NeMo-RL is natively designed for modular reward engineering. Rather than strictly entangling rewards with the PPO/GRPO core loops, the framework executes simple Python functions mapped via YAML configs.

To add a completely new reward (e.g., regex penalties, semantic formatting, custom logical rules):

### 1. Write the Python Reward Function
Open `/opt/nemo-rl/nemo_rl/environments/rewards.py` (either on your active cluster or in your fork) and define your function. It must accept the ground truth and model response, returning a `(float_reward, boolean_passed)` tuple.

```python
def no_swearing_penalty(ground_truth: str, response: str) -> tuple[float, bool]:
    if "darn" in response.lower():
        return -1.0, False  # Apply heavy penalty
    return 0.0, True      # Neutral/Pass
```

### 2. Register the Function in the Environment
Open `/opt/nemo-rl/nemo_rl/environments/vlm_environment.py`. Locate the `_instantiate_reward_functions` method and map a YAML string identifier to your new python function:

```python
elif reward_func_name == "no_swearing":
    reward_func = no_swearing_penalty
```

### 3. Apply it in your YAML Config
You can now construct massive multi-objective reward pipelines natively from your experiment's configuration file. The framework will automatically invoke `combine_reward_functions()` to calculate the weighted sum of all functions listed!

```yaml
env:
  reward_functions:
    - name: "format"
      weight: 0.2
      kwargs:
        think_tag: "think"
        answer_tag: "answer"
    - name: "exact_alnum"
      weight: 0.8
    - name: "no_swearing"
      weight: 1.0  # Applied as a heavy penalty multiplier
```

## 13. 🛠️ Troubleshooting

### "Bug in FlashInfer block_size 16 head size 256 support"
If your Gemma 3 training job crashes immediately during `vllmWorker` initialization with the FlashInfer assertion error, this is because Gemma 3 has a head size of 256, which breaks FlashInfer's default block size of 16.

To fix this, ensure your configuration YAML (e.g., `grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200.yaml`) explicitly defines the `block_size: 32` override specifically inside the **`vllm_kwargs`** dictionary (NOT `vllm_cfg`), as NeMo RL requires explicit kwargs passing for the vLLM engine constructor:
```yaml
  policy:
    generation:
      vllm_kwargs:
        block_size: 32
```
