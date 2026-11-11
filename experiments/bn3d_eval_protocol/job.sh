#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_repo="$(cd "$script_dir/../.." && pwd)"
cluster_repo="${CONTAINER_REPO:-/mnt/scratch/imos-students/ke/repos/NVF_BN3D}"
python_bin="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"
mode="${1:-}"
config="${2:-$script_dir/config.json}"
comparison="${3:-}"
case "$mode" in submit|worker) ;; *)
  echo "Usage: bash $0 {submit|worker} [config.json] [comparison_id]" >&2; exit 2 ;;
esac
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
if [[ "$mode" == worker ]]; then
  cd "$cluster_repo"
  [[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  options=(--config "$config")
  if [[ -n "$comparison" ]]; then options+=(--comparison "$comparison"); fi
  "$python_bin" -B -m experiments.bn3d_eval_protocol.run plan "${options[@]}" --check-paths
  "$python_bin" -B -c 'import torch, bpy, bpycv, gtsam, pytorch3d, tinycudann, nerfacc, yaml, torchmetrics; assert torch.cuda.is_available(), "CUDA unavailable"; print("Environment OK", torch.__version__, bpy.app.version_string, flush=True)'
  exec "$python_bin" -B -m experiments.bn3d_eval_protocol.run all "${options[@]}"
fi
command -v runai >/dev/null 2>&1 || { echo "Submit on a login node with Run:ai CLI." >&2; exit 1; }
# Config must travel with the repository, not point to a workstation-only path.
case "$config" in
  "$local_repo"/*) cluster_config="$cluster_repo/${config#"$local_repo/"}" ;;
  *) echo "Place config inside this repository before submitting." >&2; exit 2 ;;
esac
project="${RUNAI_PROJECT:-imos-ke}"
job_name="${CLUSTER_USER:-ke}-bn3d-eval-$(date +%m%d-%H%M%S)"
runai submit --name "$job_name" \
  -p "$project" --image "${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}" \
  --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm --node-pools "${NODE:-default}" \
  --working-dir "$cluster_repo" \
  --environment CLUSTER_USER="${CLUSTER_USER:-ke}" \
  --environment CLUSTER_USER_ID="${CLUSTER_USER_ID:-308061}" \
  --environment CLUSTER_GROUP_NAME="${CLUSTER_GROUP_NAME:-imos}" \
  --environment CLUSTER_GROUP_ID="${CLUSTER_GROUP_ID:-79821}" \
  --environment PYTHON_BIN="$python_bin" --environment CONTAINER_REPO="$cluster_repo" \
  --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
  --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
  --command -- /bin/bash "$cluster_repo/experiments/bn3d_eval_protocol/job.sh" \
  worker "$cluster_config" "$comparison"
echo "Submitted: $job_name"
echo "Logs: runai logs -f $job_name -p $project"
echo "Results: output_root in $config"
