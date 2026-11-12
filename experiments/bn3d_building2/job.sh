#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_repo="$(cd "$script_dir/../.." && pwd)"
cluster_repo="${CONTAINER_REPO:-/mnt/scratch/imos-students/ke/repos/NVF_BN3D}"
python_bin="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"
mode="${1:-}"
task="${2:-}"
config="${3:-$script_dir/config.json}"
case "$mode" in submit|worker) ;; *) echo "Usage: $0 {submit|worker} {freeze|nvf|random|evaluate} [config.json]" >&2; exit 2 ;; esac
case "$task" in freeze|nvf|random|evaluate) ;; *) echo 'Task must be freeze, nvf, random, or evaluate' >&2; exit 2 ;; esac
# Relative config paths are repository-relative, independent of the caller's cwd.
if [[ "$config" != /* ]]; then config="$local_repo/$config"; fi
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
config="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"
if [[ "$mode" == worker ]]; then
  [[ -x "$python_bin" && -d "$cluster_repo" ]] || { echo 'Missing Python or repository' >&2; exit 1; }
  export PYTHONPATH="$cluster_repo${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  cd "$cluster_repo"
  # Resolve paths with stdlib only, so failed training imports are logged too.
  log_dir="$("$python_bin" -B -c '
import json, sys
from pathlib import Path
bundle_path = Path(sys.argv[1])
bundle = json.loads(bundle_path.read_text())
key = "eval_config" if sys.argv[2] == "evaluate" else "train_config"
config = json.loads((bundle_path.parent / bundle[key]).read_text())
print(Path(config["output_root"]) / "_logs")
' "$config" "$task")"
  mkdir -p "$log_dir"
  attempt_log="$(mktemp "$log_dir/${task}.log.XXXXXX")"
  echo "Attempt log: $attempt_log"
  # Each replacement Pod gets a fresh log, including native subprocess output.
  # pipefail preserves Python's failure status; existing logs are never replaced.
  {
    echo "Task: $task; host: ${HOSTNAME:-unknown}; started: $(date -u +%FT%TZ)"
    "$python_bin" -B -m experiments.bn3d_building2.run "$task" --config "$config"
  } 2>&1 | tee "$attempt_log"
  exit 0
fi
case "$config" in
  "$local_repo"/*) cluster_config="$cluster_repo/${config#"$local_repo/"}" ;;
  *) echo 'Submission config must be inside this repository' >&2; exit 2 ;;
esac
command -v runai >/dev/null 2>&1 || { echo 'Submit from a login node with Run:ai CLI' >&2; exit 1; }
project="${RUNAI_PROJECT:-imos-ke}"
job_name="${CLUSTER_USER:-ke}-bn3d-building2-${task}-$(date +%m%d-%H%M%S)"
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
  --command -- /bin/bash "$cluster_repo/experiments/bn3d_building2/job.sh" worker "$task" "$cluster_config"
echo "Submitted: $job_name"
echo "runai logs -f $job_name -p $project"
