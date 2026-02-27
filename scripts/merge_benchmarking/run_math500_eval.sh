#!/bin/bash
set -e

# ==============================================================================
# 🚀 NeMo-RL FSDP Merge & Evaluate Pipeline
# ==============================================================================
#
# This script robustly offline-merges a raw PyTorch FSDP Distributed Checkpoint
# into a HuggingFace safetensors model, stages it locally on node SSDs, and
# triggers a MATH-500 evaluation job across the Ray cluster.
#
# CONFIGURATION INSTRUCTIONS:
# ---------------------------
# 1. CHECKPOINT_DIR:
#    This variable MUST point to the root directory where your step checkpoint
#    is saved. Due to our cluster architecture, this is typically on the
#    GCS-Fuse mount (starting with `/data/...`).
#
#    Example from a training run:
#    /data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model
#
# FLAGS:
# ------
# --checkpoint-dir <path> : Override the default CHECKPOINT_DIR below.
# --skip-merge            : If the model has already been successfully merged mapped
#                           in CHECKPOINT_DIR/consolidated_physical, skip the heavy 
#                           python concatenation phase and proceed directly to eval.
# --skip-sync             : Skip copying the merged model to the Worker Pod SSDs.
#                           Useful if you are just re-running an eval on a model
#                           that is already staged locally on the cluster.
# --local-dir <path>      : The ephemeral path on the Ray worker pods where the
#                           model will be statically staged to avoid GCS mmap crashes.
# --vanilla-model <id>    : Skip merging and syncing entirely, and just evaluate a
#                           base HuggingFace model (e.g., google/gemma-3-1b-it).
#                           This is used for generating zero-shot baseline metrics.
#
# ==============================================================================

# Default Arguments (Change this to your active experiment path!)
CHECKPOINT_DIR="/data/nemo-rl-results/grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200/step_30/policy/weights/model"
SKIP_MERGE=false
SKIP_SYNC=false
LOCAL_MERGE_DIR="/tmp/hf_merged_model_eval"
VANILLA_MODEL=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift ;;
        
        --skip-merge) 
            if [[ "$2" == "true" || "$2" == "false" ]]; then
                SKIP_MERGE="$2"
                shift # Consume the explicit boolean argument
            else
                SKIP_MERGE=true
            fi
            ;;
        --skip-merge=*) 
            SKIP_MERGE="${1#*=}"
            ;;
            
        --skip-sync) 
            if [[ "$2" == "true" || "$2" == "false" ]]; then
                SKIP_SYNC="$2"
                shift # Consume the explicit boolean argument
            else
                SKIP_SYNC=true
            fi
            ;;
        --skip-sync=*) 
            SKIP_SYNC="${1#*=}"
            ;;
            
        --local-dir) LOCAL_MERGE_DIR="$2"; shift ;;
        --vanilla-model) VANILLA_MODEL="$2"; shift ;;
        
        # Catch-all for KEY=VALUE pairs passed conventionally
        SKIP_MERGE=*) SKIP_MERGE="${1#*=}" ;;
        SKIP_SYNC=*) SKIP_SYNC="${1#*=}" ;;
        
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# If evaluating a vanilla model, force skip the merge and sync steps
if [ -n "$VANILLA_MODEL" ]; then
    echo "Vanilla model requested: $VANILLA_MODEL. Forcing skip-merge and skip-sync."
    SKIP_MERGE=true
    SKIP_SYNC=true
    MODEL_NAME="$VANILLA_MODEL"
else
    MODEL_NAME="$LOCAL_MERGE_DIR"
fi

RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

echo "Target Checkpoint: $CHECKPOINT_DIR"

if [ "$SKIP_MERGE" = false ]; then
    echo "============================================="
    echo "1. Merging PyTorch DCP Checkpoint to HuggingFace"
    echo "============================================="
    echo "Copying manual merge script to head pod..."
    kubectl cp scripts/merge_benchmarking/manual_merge_fsdp.py $RAY_HEAD_POD:/tmp/manual_merge_fsdp.py -c ray-head

    echo "Running manual merge script natively against GCS-Fuse..."
    # The script reads directly from GCS-Fuse and writes to $CHECKPOINT_DIR/consolidated_physical
    kubectl exec -it $RAY_HEAD_POD -c ray-head -- bash -c "python /tmp/manual_merge_fsdp.py --checkpoint-dir $CHECKPOINT_DIR"
    
    echo "Copying HF metadata to the consolidated folder..."
    kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "mkdir -p $CHECKPOINT_DIR/consolidated_physical && cp -v $CHECKPOINT_DIR/.hf_metadata/config.json $CHECKPOINT_DIR/consolidated_physical/"
    kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "cp -v $CHECKPOINT_DIR/.hf_metadata/generation_config.json $CHECKPOINT_DIR/consolidated_physical/ 2>/dev/null || true"
else
    echo "============================================="
    echo "1. Skipping Merge (Using existing consolidated_physical model)"
    echo "============================================="
fi

echo ""
echo "============================================="
echo "2. Broadcasting Merged Model to All Worker SSDs"
echo "============================================="
if [ "$SKIP_SYNC" = false ]; then
    echo 'Copying merged model to local SSD on ALL worker pods to avoid mmap GCS Fuse errors...'
    WORKER_PODS=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}')

    for pod in $WORKER_PODS; do 
      echo "Syncing weights to worker node: $pod..."
      kubectl exec $pod -c ray-worker -- bash -c "mkdir -p $LOCAL_MERGE_DIR && cp -rv $CHECKPOINT_DIR/consolidated_physical/* $LOCAL_MERGE_DIR/ && cp -v $CHECKPOINT_DIR/.hf_metadata/* $LOCAL_MERGE_DIR/ 2>/dev/null || true && cp -v /data/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/tokenizer* $LOCAL_MERGE_DIR/ 2>/dev/null || true"
    done

    echo "Syncing local copy for the Ray Head Pod launcher process..."
    kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "mkdir -p $LOCAL_MERGE_DIR && cp -rv $CHECKPOINT_DIR/consolidated_physical/* $LOCAL_MERGE_DIR/ && cp -v $CHECKPOINT_DIR/.hf_metadata/* $LOCAL_MERGE_DIR/ 2>/dev/null || true && cp -v /data/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/tokenizer* $LOCAL_MERGE_DIR/ 2>/dev/null || true"
else
    echo "Skipping SSD broadcast..."
fi

echo ""
echo "============================================="
echo "3. Submitting MATH-500 Eval Ray Job"
echo "============================================="
CMD="ray job submit --working-dir /workspace -- bash -c \"
export HF_HOME=/data/huggingface
if [ -n '${HF_TOKEN}' ]; then
    export HF_TOKEN='${HF_TOKEN}'
fi

echo 'Evaluating the model on MATH-500 using native NeMo-RL...'
cd /opt/nemo-rl
/opt/nemo_rl_venv/bin/python examples/run_eval.py \\
    --config examples/configs/evals/math_eval.yaml \\
    generation.model_name=$MODEL_NAME \\
    generation.temperature=0.0 \\
    generation.vllm_cfg.max_model_len=8192 \\
    data.dataset_name=math500 \\
    eval.num_tests_per_prompt=1 \\
    cluster.num_nodes=1 \\
    cluster.gpus_per_node=8 \\
    generation.vllm_kwargs.block_size=32
\""

kubectl exec $RAY_HEAD_POD -c ray-head -- bash -c "$CMD"
