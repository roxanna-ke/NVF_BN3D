#!/usr/bin/env bash
# No automatic submission chaining: the user submits analyze, then branches.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_repo="$(cd "$script_dir/../.." && pwd)"
cluster_repo="${CONTAINER_REPO:-/mnt/scratch/imos-students/ke/repos/NVF_BN3D}"
python_bin="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"
mode="${1:-}"
task="${2:-}"
config="${3:-$script_dir/config.json}"
case "$mode" in submit|worker) ;; *) echo "Usage: bash $0 {submit|worker} {analyze|branches} [config.json]" >&2; exit 2 ;; esac
case "$task" in analyze|branches) ;; *) echo 'Task must be analyze or branches' >&2; exit 2 ;; esac
if [[ "$config" != /* ]]; then config="$local_repo/$config"; fi
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
if [[ "$mode" == worker ]]; then
  [[ -d "$cluster_repo" && -x "$python_bin" ]] || { echo 'Missing container repository or Python' >&2; exit 1; }
  cd "$cluster_repo"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  exec "$python_bin" -B -m experiments.bn3d_diagnostics.run "$task" --config "$config"
fi
case "$config" in
  "$local_repo"/*) cluster_config="$cluster_repo/${config#"$local_repo/"}" ;;
  *) echo 'Submission config must be inside this repository' >&2; exit 2 ;;
esac
command -v runai >/dev/null 2>&1 || { echo 'Submit from a login node with Run:ai CLI' >&2; exit 1; }
project="${RUNAI_PROJECT:-imos-ke}"
job_name="${CLUSTER_USER:-ke}-bn3d-b1-${task}-$(date +%m%d-%H%M%S)"
runai submit --name "$job_name" \
  -p "$project" --image "${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}" \
  --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm --node-pools "${NODE:-default}" \
  --working-dir /tmp \
  --environment CLUSTER_USER="${CLUSTER_USER:-ke}" \
  --environment CLUSTER_USER_ID="${CLUSTER_USER_ID:-308061}" \
  --environment CLUSTER_GROUP_NAME="${CLUSTER_GROUP_NAME:-imos}" \
  --environment CLUSTER_GROUP_ID="${CLUSTER_GROUP_ID:-79821}" \
  --environment PYTHON_BIN="$python_bin" --environment CONTAINER_REPO="$cluster_repo" \
  --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
  --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
  --command -- /bin/bash "$cluster_repo/experiments/bn3d_diagnostics/job.sh" worker "$task" "$cluster_config"
echo "Submitted: $job_name"
echo "runai logs -f $job_name -p $project"
