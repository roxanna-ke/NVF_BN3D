#!/usr/bin/env bash
# Submit the BN3D seed-42 NVF experiment from a Run:ai login node.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /bin/bash "$repo/scripts/bn3d_nvf_job.sh" submit nvf
