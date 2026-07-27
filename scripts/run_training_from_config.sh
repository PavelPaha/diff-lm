#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.json" >&2
  exit 2
fi

config_path="$(realpath "$1")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DLM_PYTHON:-/home/vasilievpavel/.venv/bin/python}"

readarray -t config_values < <("$python_bin" - "$config_path" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)

for key in ("git_commit", "run_name"):
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"missing non-empty {key!r} in config")
    print(value)
print(config["distributed"]["cuda_visible_devices"])
print(config["distributed"]["world_size"])
PY
)
commit="${config_values[0]}"
run_name="${config_values[1]}"
cuda_visible_devices="${config_values[2]}"
world_size="${config_values[3]}"

if ! git -C "$repo_root" cat-file -e "${commit}^{commit}"; then
  echo "Config commit does not exist locally: $commit" >&2
  exit 1
fi

if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "git_commit must be a full 40-character SHA, not a branch or HEAD" >&2
  exit 1
fi

snapshot_dir="$(mktemp -d "/tmp/diff-lm-${run_name}-XXXXXX")"
cleanup() { rm -rf "$snapshot_dir"; }
trap cleanup EXIT

git -C "$repo_root" archive --format=tar "$commit" | tar -xf - -C "$snapshot_dir"
cp "$config_path" "$snapshot_dir/run_config.json"

echo "Training snapshot: $snapshot_dir"
echo "Commit: $commit"
cd "$snapshot_dir"
export PYTHONPATH="$snapshot_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$world_size" \
  -m dlm_attn_res.training.train --config "$snapshot_dir/run_config.json"
