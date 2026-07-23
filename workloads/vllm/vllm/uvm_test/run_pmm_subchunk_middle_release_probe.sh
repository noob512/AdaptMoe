#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/pmm_subchunk_middle_release_probe_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "device=${DEVICE}"
  if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
    echo "uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
  fi
} | tee "${RUN_ROOT}/env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" || true
fi

make -C "${SCRIPT_DIR}" pmm_subchunk_middle_release_probe \
  2>&1 | tee "${RUN_ROOT}/build.log"

run_case() {
  local name="$1"
  local mode="$2"
  local log="${RUN_ROOT}/${name}.log"

  set +e
  "${SCRIPT_DIR}/pmm_subchunk_middle_release_probe" \
    --device "${DEVICE}" "${mode}" \
    2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e

  echo "${status}" > "${RUN_ROOT}/${name}.status"
  return 0
}

run_case "envelope" "--envelope"
run_case "legacy" "--legacy"

python3 - "${RUN_ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def parse_case(name: str) -> dict:
    text = (root / f"{name}.log").read_text(errors="replace")
    status = int((root / f"{name}.status").read_text().strip())
    final = {}
    for line in text.splitlines():
        if line.startswith("pmm_subchunk_middle_release_probe_final"):
            for key, value in re.findall(r"(\w+)=([^\s]+)", line):
                final[key] = value
    queries = {}
    for line in text.splitlines():
        if line.startswith("query_region_blocks"):
            fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
            label = fields.get("label")
            if label:
                queries[label] = fields
    releases = []
    for line in text.splitlines():
        if line.startswith("expert_offload_release"):
            releases.append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
    return {
        "status": status,
        "final": final,
        "queries": queries,
        "releases": releases,
    }

summary = {
    "run_root": str(root),
    "envelope": parse_case("envelope"),
    "legacy": parse_case("legacy"),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_after.txt" || true
fi
dmesg -T | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log" || true

echo "run_root=${RUN_ROOT}"
echo "summary=${RUN_ROOT}/summary.json"
