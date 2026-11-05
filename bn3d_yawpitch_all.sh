#!/usr/bin/env bash

set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for total in 20 30 50; do
  /bin/bash "$repo/scripts/bn3d_yawpitch_job.sh" submit nvf "$total"
done
for total in 20 50; do
  /bin/bash "$repo/scripts/bn3d_yawpitch_job.sh" submit random "$total"
done
