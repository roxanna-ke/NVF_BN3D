#!/usr/bin/env bash
# Submit from a Run:ai login node; training executes in the cluster container.
set -euo pipefail

CONTAINER_REPO=/mnt/scratch/imos-students/ke/repos/NVF_BN3D
CONTAINER_SCRIPT="$CONTAINER_REPO/hotdog_random.sh"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"
mode="${1:-submit}"

if [[ "$mode" == worker ]]; then
  job_name="${2:?Missing job name}"
  [[ "$job_name" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid job name" >&2; exit 2; }
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
  for asset in blend_files/hotdog.blend blend_files/hotdog.ply; do
    [[ -f "$CONTAINER_REPO/data/assets/$asset" ]] || { echo "Missing asset: $asset" >&2; exit 1; }
  done
  # Keep the same relative paths as an interactive eval.py run from the repo.
  cd "$CONTAINER_REPO"
  "$PYTHON_BIN" -u -B -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
  "$PYTHON_BIN" -u -B "$CONTAINER_REPO/eval.py" \
    --scene hotdog --method nvf --agent random
  echo "Complete. Native results: $CONTAINER_REPO/results/"
  exit 0
fi

[[ "$mode" == submit ]] || { echo "Usage: bash $0 [submit]" >&2; exit 2; }
command -v runai >/dev/null 2>&1 || { echo "Submit from a cluster login node with Run:ai" >&2; exit 1; }
project="${RUNAI_PROJECT:-imos-ke}"
image="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
job_name="${CLUSTER_USER:-ke}-hotdog-random-$(date +%m%d-%H%M%S)"
runai submit --name "$job_name" \
  -p "$project" --image "$image" --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm \
  --node-pools "${NODE:-default}" --working-dir "/mnt/scratch/imos-students/ke" \
  --environment CLUSTER_USER="${CLUSTER_USER:-ke}" \
  --environment CLUSTER_USER_ID="${CLUSTER_USER_ID:-308061}" \
  --environment CLUSTER_GROUP_NAME="${CLUSTER_GROUP_NAME:-imos}" \
  --environment CLUSTER_GROUP_ID="${CLUSTER_GROUP_ID:-79821}" \
  --environment PYTHON_BIN="$PYTHON_BIN" \
  --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
  --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
  --command -- /bin/bash "$CONTAINER_SCRIPT" worker "$job_name"
cat <<EOF_STATUS
Submitted: $job_name
Logs:      runai logs -f $job_name -p $project
Status:    runai describe job $job_name -p $project
Results:   $CONTAINER_REPO/results/
Data:      $CONTAINER_REPO/data/hotdog/
EOF_STATUS
