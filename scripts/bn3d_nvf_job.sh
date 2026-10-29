#!/usr/bin/env bash
# Shared Run:ai submit/worker logic for one BN3D policy, seed 42.
set -euo pipefail

CONTAINER_REPO=/mnt/scratch/imos-students/ke/repos/NVF_BN3D
CONTAINER_SCRIPT="$CONTAINER_REPO/scripts/bn3d_nvf_job.sh"
CONFIG="$CONTAINER_REPO/configs/bn3d_nvf.json"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/conda-envs/pun/bin/python}"

mode="${1:-}"
policy="${2:-}"
[[ "$policy" == nvf || "$policy" == random ]] || {
  echo "Usage: $0 {submit|worker} {nvf|random}" >&2
  exit 2
}

if [[ "$mode" == worker ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
  [[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
  cd "$CONTAINER_REPO"
  "$PYTHON_BIN" -B -c 'import torch, bpy, bpycv, gtsam, pytorch3d, tinycudann, nerfacc, viser; from eval import set_params; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("NVF environment OK:", torch.__version__, bpy.app.version_string, flush=True)'
  config_values="$("$PYTHON_BIN" -B -c 'import json; c=json.load(open("configs/bn3d_nvf.json")); assert c["seeds"]==[42], "expected seed 42 only"; print(c["mesh_path"]); print(c["data_root"]); print(c["output_root"])')"
  readarray -t paths <<< "$config_values"
  [[ -f "${paths[0]}" && -f "${paths[1]}/meta_data.json" ]] || {
    echo "Missing BN3D OBJ or metadata; check $CONFIG" >&2; exit 1;
  }
  output="${paths[2]}"
  run_dir="$output/$policy/seed42"
  if [[ -f "$run_dir/complete.json" ]]; then
    echo "Already complete: $run_dir"
  else
    if [[ -e "$run_dir" ]]; then
      echo "Incomplete run exists: $run_dir. Move it aside before retrying." >&2
      exit 1
    fi
    "$PYTHON_BIN" -B tools/bn3d_nvf.py preflight --config "$CONFIG" --policy "$policy"
    echo "Review preflight image: $output/prepared/preflight_${policy}.png"
    "$PYTHON_BIN" -B tools/bn3d_nvf.py run --config "$CONFIG" --policy "$policy" --seed 42
  fi
  if [[ -f "$output/nvf/seed42/complete.json" && -f "$output/random/seed42/complete.json" ]]; then
    "$PYTHON_BIN" -B tools/bn3d_nvf.py summarize --config "$CONFIG"
    echo "Summary: $output/summary/final.json"
  else
    echo "Summary pending until both seed 42 policies complete."
  fi
  exit 0
fi

[[ "$mode" == submit ]] || { echo "Usage: $0 {submit|worker} {nvf|random}" >&2; exit 2; }
command -v runai >/dev/null 2>&1 || {
  echo "Run:ai CLI unavailable; submit from a login node with runai." >&2; exit 1;
}
[[ -f "$(dirname "${BASH_SOURCE[0]}")/../configs/bn3d_nvf.json" ]] || {
  echo "NVF_BN3D config missing beside submit script" >&2; exit 1;
}
project="${RUNAI_PROJECT:-imos-ke}"
image="${MY_IMAGE:-registry.rcp.epfl.ch/imos-ke/avs:v0.0.1}"
job_name="${CLUSTER_USER:-ke}-bn3d-${policy}-42-$(date +%H%M%S)"
runai submit --name "$job_name" \
  -p "$project" --image "$image" --gpu "${MY_GPU_AMOUNT:-1.0}" --large-shm \
  --node-pools "${NODE:-default}" --working-dir "$CONTAINER_REPO" \
  --environment CLUSTER_USER="${CLUSTER_USER:-ke}" \
  --environment CLUSTER_USER_ID="${CLUSTER_USER_ID:-308061}" \
  --environment CLUSTER_GROUP_NAME="${CLUSTER_GROUP_NAME:-imos}" \
  --environment CLUSTER_GROUP_ID="${CLUSTER_GROUP_ID:-79821}" \
  --environment PYTHON_BIN="$PYTHON_BIN" \
  --existing-pvc "claimname=${SCRATCH_PVC:-imos-scratch},path=/mnt/scratch" \
  --existing-pvc "claimname=${HOME_PVC:-home},path=/mnt/home" \
  --command -- /bin/bash "$CONTAINER_SCRIPT" worker "$policy"
cat <<EOF
Submitted: $job_name
Logs:      runai logs -f $job_name -p $project
Status:    runai describe job $job_name -p $project
Output:    /mnt/scratch/imos-students/ke/outputs/bn3d_nvf/building1/$policy/seed42
EOF
