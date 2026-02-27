#!/bin/bash
set -e

RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

CHECKPOINT_DIR="/data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model"

echo "Copying manual merge script to head pod..."
kubectl cp scripts/merge_benchmarking/manual_merge_fsdp.py $RAY_HEAD_POD:/tmp/manual_merge_fsdp.py -c ray-head

echo "Running manual merge script natively to consolidate PyTorch DCP safetensors into HuggingFace format..."
kubectl exec -it $RAY_HEAD_POD -c ray-head -- bash -c "python /tmp/manual_merge_fsdp.py --checkpoint-dir $CHECKPOINT_DIR"

echo 'Copying HF configuration files from checkpoint root into the consolidated folder...'
kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "cp -v /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/.hf_metadata/config.json /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/consolidated_physical/"
kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "cp -v /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/.hf_metadata/generation_config.json /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/consolidated_physical/"

echo 'Copying merged model to local SSD on ALL worker pods to avoid mmap GCS Fuse errors...'
WORKER_PODS=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}')
for pod in $WORKER_PODS; do 
  echo "Syncing weights to worker node: $pod..."
  kubectl exec $pod -c ray-worker -- bash -c "mkdir -p /tmp/hf_merged_gemma3_step30 && cp -rv /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/consolidated_physical/* /tmp/hf_merged_gemma3_step30/ && cp -v /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/.hf_metadata/* /tmp/hf_merged_gemma3_step30/ && cp -v /data/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/tokenizer* /tmp/hf_merged_gemma3_step30/"
done

echo "Syncing local copy for the Ray Head Pod launcher process..."
kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "mkdir -p /tmp/hf_merged_gemma3_step30 && cp -rv /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/consolidated_physical/* /tmp/hf_merged_gemma3_step30/ && cp -v /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model/.hf_metadata/* /tmp/hf_merged_gemma3_step30/ && cp -v /data/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/tokenizer* /tmp/hf_merged_gemma3_step30/"

echo "Submitting MATH-500 Eval Ray Job..."
CMD="ray job submit --working-dir /workspace -- bash -c \"
export HF_HOME=/data/huggingface

echo 'Evaluating the model on MATH-500 using native NeMo-RL...'
cd /opt/nemo-rl
/opt/nemo_rl_venv/bin/python examples/run_eval.py \\
    --config examples/configs/evals/math_eval.yaml \\
    generation.model_name=/tmp/hf_merged_gemma3_step30 \\
    generation.temperature=0.0 \\
    generation.vllm_cfg.max_model_len=8192 \\
    data.dataset_name=math500 \\
    eval.num_tests_per_prompt=1 \\
    cluster.num_nodes=1 \\
    cluster.gpus_per_node=8 \\
    generation.vllm_kwargs.block_size=32
\""

kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "$CMD"
