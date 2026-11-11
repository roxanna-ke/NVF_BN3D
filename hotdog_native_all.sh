#!/usr/bin/env bash
# Submit two native agents. Default is the short validation profile.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-smoke}"
[[ "$profile" == smoke || "$profile" == full ]] || { echo "Usage: $0 [smoke|full]" >&2; exit 2; }
for policy in nvf random; do
  /bin/bash "$repo/scripts/hotdog_native_job.sh" submit "$policy" "$profile"
done
