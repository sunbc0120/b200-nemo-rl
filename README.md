# NeMo RL Cluster Deployment Guide

This guide details how to seamlessly deploy a Ray cluster on Google Kubernetes Engine (GKE) configured explicitly for training models like Gemma 3 using the NVIDIA NeMo-RL framework.

The cluster provisions **B200** Spot instances (8 GPUs per node) and mounts a high-performance Google Cloud Storage (GCS) bucket natively using the Fuse CSI driver.

## Prerequisites
- Authenticated `kubectl` context pointing to your target GKE cluster.
- The `ray-cluster-b200-nemo.yaml` manifest.
- A valid HuggingFace Access Token exported into your terminal (`export HF_TOKEN="hf_..."`) to download gated Gemma 3 weights.

## Deployment Instructions

### 1. Ensure a Clean State
If you have an existing or stale Ray cluster running on the target node pools, you must tear it down first to free up the expensive GPU resources and avoid scheduling deadlocks.

```bash
# Delete any existing testing cluster
kubectl delete raycluster ray-cluster-b200

# Delete the target NeMo cluster if explicitly restarting
kubectl delete raycluster ray-cluster-b200-nemo
```

### 2. Apply the Manifest
The manifest defines a RayCluster Custom Resource (CRD). The KubeRay operator will intercept this and begin provisioning the Head and Worker pods.

```bash
kubectl apply -f manifests/ray-cluster-b200-nemo.yaml
```

*Note: The YAML explicitly configures `replicas: 2` under `workerGroupSpecs`, requesting 16 B200 GPUs total. Adjust this value in the YAML if you need a different scale.*

### 3. Monitor the Cold Boot
Ray pods initialize via an `initContainer` (`install-ray`) which downloads the raw python orchestrator binaries into a temporary volume before the main container starts. This takes roughly 45-60 seconds.

Watch the pod spin-up process:
```bash
kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -w
```
Wait until both the `ray-head` and `ray-worker` pods reach the `Running` state and register `Ready` (e.g., `4/4` and `3/3`).

### 4. Verify Cluster Health
Once the pods are running, connect to the Head node and query the native Ray daemon to ensure all worker GPUs are actively registered in the pool.

```bash
# Fetch the head pod ID
HEAD_POD=$(kubectl get pods -l ray.io/cluster=ray-cluster-b200-nemo -l ray.io/node-type=head --no-headers | awk '{print $1}' | head -n 1)

# Execute 'ray status' (ensuring the pip-installed PATH is targeted)
kubectl exec $HEAD_POD -c ray-head -- /bin/bash -c "export PATH=/tmp/ray/packages/bin:\$PATH; ray status"
```

If successful, you will see a console output reporting **232 active CPUs** and **16.0 active GPUs** (assuming 2 replicas).

## 5. Connect to the Ray Dashboard
You can monitor the cluster's GPU utilization, view logs, and track the real-time progress of jobs via the Ray Dashboard. Set up a local port-forward to the Ray Head service using `kubectl`:

```bash
kubectl port-forward svc/ray-cluster-b200-nemo-head-svc 8265:8265
```

Once that is running, open your web browser and navigate to: **[http://localhost:8265](http://localhost:8265)**

## 6. Job Execution (GRPO Gemma 3)
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
