#!/usr/bin/env bash
# Native Hotdog evaluation. Run submit on the cluster login node.
set -euo pipefail

CONTAINER_REPO=/mnt/scratch/imos-students/ke/repos/NVF_BN3D
CONTAINER_SCRIPT="$CONTAINER_REPO/scripts/hotdog_native_job.sh"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"

mode="${1:-}"
policy="${2:-}"
profile="${3:-smoke}"
[[ "$policy" == nvf || "$policy" == random ]] || { echo "Policy must be nvf or random" >&2; exit 2; }
[[ "$profile" == smoke || "$profile" == full ]] || { echo "Profile must be smoke or full" >&2; exit 2; }

if [[ "$mode" == worker ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
  cd "$CONTAINER_REPO"
  for asset in hotdog.blend hotdog.ply; do
    [[ -f "data/assets/blend_files/$asset" ]] || { echo "Missing Hotdog asset: $asset" >&2; exit 1; }
  done
  "$PYTHON_BIN" -B -c 'import torch, bpy, bpycv, gtsam, pytorch3d, tinycudann, nerfacc, viser; from eval import multi_evaluation; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("NVF environment OK", torch.__version__, bpy.app.version_string, flush=True)'
  "$PYTHON_BIN" -u -B tools/hotdog_native.py --policy "$policy" --profile "$profile"
  exit 0
fi

[[ "$mode" == submit ]] || { echo "Usage: $0 {submit|worker} {nvf|random} [smoke|full]" >&2; exit 2; }
command -v runai >/dev/null 2>&1 || {
  echo "Run:ai CLI unavailable; submit from a login node with runai." >&2; exit 1;
}
project="${RUNAI_PROJECT:-imos-ke}"
image="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
job_name="${CLUSTER_USER:-ke}-hotdog-native-${policy}-${profile}-$(date +%H%M%S)"
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
  --command -- /bin/bash "$CONTAINER_SCRIPT" worker "$policy" "$profile"
cat <<EOF
Submitted: $job_name
Logs:      runai logs -f $job_name -p $project
Status:    runai describe job $job_name -p $project
Output:    /mnt/scratch/imos-students/ke/outputs/hotdog_native/$profile/$policy
EOF
