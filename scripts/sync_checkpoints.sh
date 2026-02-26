#!/bin/bash
set -e

EXPERIMENT_NAME="grpo-gemma3-1b-it-1n8g-fsdp2tp1-b200"

echo "Starting FSDP Checkpoint Sync for experiment: $EXPERIMENT_NAME..."

# Find the head pod
RAY_HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

if [ -z "$RAY_HEAD_POD" ]; then
    echo "Error: Could not find the ray-head pod."
    exit 1
fi

# Sync Head Pod config files to GCS
echo "Syncing Head Pod configuration files to GCS..."
kubectl exec $RAY_HEAD_POD -- bash -c "mkdir -p /data/nemo-rl-results/ && cp -r /opt/nemo-rl/results/$EXPERIMENT_NAME /data/nemo-rl-results/ || true"

# Iterate over all worker pods
WORKER_PODS=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=worker -o jsonpath='{.items[*].metadata.name}')

for POD in $WORKER_PODS; do
    echo "Syncing FSDP shard weights from Worker Pod: $POD..."
    kubectl exec $POD -- bash -c "
        cd /opt/nemo-rl/results/$EXPERIMENT_NAME || exit 0
        for step_dir in tmp_step_*; do
            if [ -d \"\$step_dir\" ]; then
                # Remove the 'tmp_' prefix to match the Head Node's folder structure
                step_num=\${step_dir#tmp_}
                
                # ONLY sync if the Head Node has already validated this step and pushed config.yaml to GCS!
                if [ -f \"/data/nemo-rl-results/$EXPERIMENT_NAME/\$step_num/config.yaml\" ]; then
                    echo \"Syncing \$step_dir to \$step_num...\"
                    mkdir -p \"/data/nemo-rl-results/$EXPERIMENT_NAME/\$step_num\"
                    # Merge the worker's policy shards into the corresponding step directory
                    cp -a \"\$step_dir/policy\" \"/data/nemo-rl-results/$EXPERIMENT_NAME/\$step_num/\" 2>/dev/null || true
                else
                    echo \"Skipping obsolete/aborted local shard: \$step_dir\"
                fi
            fi
        done
    "
done

echo "========================================="
echo "Checkpoint Sync Complete! You can now merge the FSDP PyTorch checkpoints into SafeTensors."
