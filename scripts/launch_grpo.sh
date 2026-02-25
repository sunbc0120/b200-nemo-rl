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

# 1. Copy the custom config straight into the Head Pod's ephemeral /workspace
echo "Transferring custom manifests to the Head Pod..."
kubectl cp manifests/grpo_math_1b.yaml $RAY_HEAD_POD:/workspace/grpo_math_1b.yaml
kubectl cp manifests/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200.yaml $RAY_HEAD_POD:/workspace/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200.yaml

# 2. Use 'ray job submit' natively from INSIDE the Head Pod!
# This leverages the completely pre-installed NeMo-RL codebase natively present at /opt/nemo-rl!
echo "Submitting Ray Job from inside the cluster..."

CMD="ray job submit --working-dir /workspace -- bash -c \"export HF_TOKEN=$HF_TOKEN && export HF_HOME=/data/huggingface && cd /opt/nemo-rl && /opt/nemo_rl_venv/bin/python examples/run_grpo_math.py --config /workspace/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200.yaml cluster.num_nodes=2\""

kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "$CMD"

echo "========================================="
echo "Job Successfully Submitted to Ray Daemon!"
echo "To tail the live logs, run:"
echo "kubectl exec -it $RAY_HEAD_POD -- ray job logs <JOB_ID> -f"
