#!/bin/bash
set -ex

# This script is designed to be run from your local laptop/environment.
# It uses `kubectl exec` to bridge into the Ray Head pod and use the native `ray` CLI.
# This completely avoids the need for a local python/ray environment!

RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$RAY_HEAD_POD" ]; then
    echo "Error: Could not find the ray-head pod for ray-cluster-b200-nemo."
    exit 1
fi

echo "Found Ray Head Pod: $RAY_HEAD_POD"

if [ -z "$HF_TOKEN" ]; then
    echo "Error: $HF_TOKEN is not set in your environment. You must 'export HF_TOKEN=...' to download Gemma 3."
    exit 1
fi

# Define a retry wrapper for kubectl cp to handle flaky GKE control plane connections
retry_cp() {
    local max_attempts=5
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if kubectl cp "$@"; then
            return 0
        else
            echo "Attempt $attempt failed. Retrying in 2 seconds..."
            sleep 2
            attempt=$((attempt + 1))
        fi
    done
    echo "Failed to copy after $max_attempts attempts."
    return 1
}

# 1. Copy the custom config straight into the Head Pod's ephemeral /workspace
echo "Transferring custom manifests to the Head Pod..."
retry_cp manifests/02b_megatron/grpo-gemma3-27b-it-megatron.yaml $RAY_HEAD_POD:/workspace/grpo-gemma3-27b-it-megatron.yaml
retry_cp scripts/megatron_job/download_model.py $RAY_HEAD_POD:/workspace/download_model.py

# 1.5. Patch Megatron Policy Worker to disable hardcoded CUDA graphs for Gemma 3
echo "Patching Megatron Policy Worker across the cluster..."
retry_cp scripts/megatron_job/megatron_policy_worker.py $RAY_HEAD_POD:/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py -c ray-head
retry_cp scripts/megatron_job/abstract_model_inference_wrapper.py $RAY_HEAD_POD:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py -c ray-head

WORKER_PODS=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}')
for pod in $WORKER_PODS; do
    echo "Patching $pod..."
    retry_cp scripts/megatron_job/megatron_policy_worker.py $pod:/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py -c ray-worker
    retry_cp scripts/megatron_job/abstract_model_inference_wrapper.py $pod:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py -c ray-worker
done

# 2. Execute the Download and Job Submission completely in the background on the cluster
# This prevents WiFi reconnects or Laptop Sleep from killing the multi-hour 54GB FUSE download.
echo "Initializing Detached Download & Launch Sequence on Ray Head Node..."

DETACHED_CMD="nohup bash -c '
    set -e
    echo \"Starting pre-download of Gemma 3 27B...\"
    export HF_TOKEN=$HF_TOKEN
    export HF_HOME=/data/huggingface
    export PYTHONUNBUFFERED=1
    
    # Force =0.23.0 to bypass the brittle experimental Rust xet-core client
    # and force HF to use the immortal, auto-retrying Python requests library.
    /opt/nemo_rl_venv/bin/pip install -q huggingface_hub==0.23.0
    /opt/nemo_rl_venv/bin/python /workspace/download_model.py
    
    # Restore the latest compatible huggingface_hub so the training job dependencies (transformers/datasets) do not break
    /opt/nemo_rl_venv/bin/pip install -q \"huggingface_hub>=0.34.0,<1.0\" hf_transfer
    
    echo \"Download complete. Submitting Ray Job...\"
    ray job submit --working-dir /workspace --no-wait -- bash -c \"export HF_TOKEN=$HF_TOKEN && export HF_HOME=/data/huggingface && cd /opt/nemo-rl && /opt/nemo_rl_venv/bin/python examples/run_grpo_math.py --config /workspace/grpo-gemma3-27b-it-megatron.yaml\"
' > /data/nemo-rl-logs/detached_launch_27b.log 2>&1 &"

kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "$DETACHED_CMD"

echo "========================================="
echo "Sequence Successfully Detached!"
echo "Your laptop can now safely go to sleep or lose WiFi. The cluster is handling everything."
echo "To tail the live progress of the download (and subsequent training), run:"
echo "kubectl exec -it $RAY_HEAD_POD -- tail -f /data/nemo-rl-logs/detached_launch_27b.log"



