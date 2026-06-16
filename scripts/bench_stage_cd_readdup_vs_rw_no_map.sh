#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PAGES_VALUES="${PAGES_VALUES:-1024 8192 65536}"
ITERATIONS="${ITERATIONS:-10}"
DEVICE="${DEVICE:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/stage_cd_readdup_vs_rw_no_map_bench_$(date -u +%Y%m%d_%H%M%S)}"
RUN_STAGE_C="${RUN_STAGE_C:-1}"
RUN_STAGE_D="${RUN_STAGE_D:-1}"
VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS:-1}"

STAGE_C_RUN_ROOT="${RUN_ROOT}/stage_c_readdup"
STAGE_D_RUN_ROOT="${RUN_ROOT}/stage_d_rw_no_map"
COMPARISON="${RUN_ROOT}/comparison.csv"
META="${RUN_ROOT}/metadata.txt"

mkdir -p "${RUN_ROOT}"

summary_value() {
  local summary="$1"
  local case_name="$2"
  local column="$3"

  awk -F, -v case_name="${case_name}" -v column="${column}" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) {
        if ($i == column)
          column_index = i
      }
      next
    }
    $1 == case_name {
      found = 1
      if (column_index > 0)
        print $column_index
      else
        print "NA"
      exit
    }
    END {
      if (!found)
        print "NA"
    }
  ' "${summary}"
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "pages_values=${PAGES_VALUES}"
    echo "iterations=${ITERATIONS}"
    echo "device=${DEVICE}"
    echo "stage_c_run_root=${STAGE_C_RUN_ROOT}"
    echo "stage_d_run_root=${STAGE_D_RUN_ROOT}"
    echo "run_stage_c=${RUN_STAGE_C}"
    echo "run_stage_d=${RUN_STAGE_D}"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
  } | tee "${META}"
}

run_stage_c() {
  echo
  echo "Running Stage C read-dup benchmark"
  RUN_ROOT="${STAGE_C_RUN_ROOT}" \
  PAGES_VALUES="${PAGES_VALUES}" \
  ITERATIONS="${ITERATIONS}" \
  DEVICE="${DEVICE}" \
  MODES="readdup_soft" \
  VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS}" \
  "${ROOT_DIR}/scripts/bench_stage_c_weight_readdup_unmap.sh"
}

run_stage_d() {
  echo
  echo "Running Stage D RW CPU-storage no-map benchmark"
  RUN_ROOT="${STAGE_D_RUN_ROOT}" \
  PAGES_VALUES="${PAGES_VALUES}" \
  ITERATIONS="${ITERATIONS}" \
  DEVICE="${DEVICE}" \
  MODES="cpu_no_map" \
  VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS}" \
  "${ROOT_DIR}/scripts/bench_stage_d_rw_cpu_storage_no_map.sh"
}

write_comparison() {
  local stage_c_summary="${STAGE_C_RUN_ROOT}/summary.csv"
  local stage_d_summary="${STAGE_D_RUN_ROOT}/summary.csv"

  if [[ ! -f "${stage_c_summary}" ]]; then
    echo "ERROR: missing ${stage_c_summary}" >&2
    exit 1
  fi

  if [[ ! -f "${stage_d_summary}" ]]; then
    echo "ERROR: missing ${stage_d_summary}" >&2
    exit 1
  fi

  {
    echo "pages,bytes,readdup_apply_ms,readdup_avg_gpu_read_ms,readdup_query_on_cpu,readdup_query_on_gpu,readdup_cpu_unmap_pages_delta,readdup_cpu_preserved_pages_delta,rw_setup_apply_ms,rw_setup_cpu_migrate_ms,rw_avg_apply_ms,rw_avg_cpu_migrate_ms,rw_avg_gpu_write_ms,rw_avg_cycle_ms,rw_total_apply_cpu_unmap_bytes,rw_cpu_unmap_pages_delta,rw_final_query_on_cpu,readdup_apply_over_rw_setup_apply,rw_cycle_over_readdup_apply,rw_cycle_over_readdup_gpu_read"

    for pages in ${PAGES_VALUES}; do
      local readdup_case="readdup_soft_p${pages}"
      local rw_case="cpu_no_map_p${pages}"
      local bytes
      local readdup_apply readdup_read readdup_on_cpu readdup_on_gpu
      local readdup_unmap_pages readdup_preserved_pages
      local rw_setup_apply rw_setup_migrate rw_apply rw_migrate rw_write rw_cycle
      local rw_unmap_bytes rw_unmap_pages rw_final_on_cpu

      bytes="$(summary_value "${stage_c_summary}" "${readdup_case}" bytes)"
      readdup_apply="$(summary_value "${stage_c_summary}" "${readdup_case}" apply_ms)"
      readdup_read="$(summary_value "${stage_c_summary}" "${readdup_case}" avg_gpu_read_ms)"
      readdup_on_cpu="$(summary_value "${stage_c_summary}" "${readdup_case}" query_on_cpu)"
      readdup_on_gpu="$(summary_value "${stage_c_summary}" "${readdup_case}" query_on_gpu)"
      readdup_unmap_pages="$(summary_value "${stage_c_summary}" "${readdup_case}" gpu_only_cpu_unmap_pages_delta)"
      readdup_preserved_pages="$(summary_value "${stage_c_summary}" "${readdup_case}" gpu_only_cpu_preserved_pages_delta)"

      rw_setup_apply="$(summary_value "${stage_d_summary}" "${rw_case}" setup_apply_ms)"
      rw_setup_migrate="$(summary_value "${stage_d_summary}" "${rw_case}" setup_cpu_migrate_ms)"
      rw_apply="$(summary_value "${stage_d_summary}" "${rw_case}" avg_apply_ms)"
      rw_migrate="$(summary_value "${stage_d_summary}" "${rw_case}" avg_cpu_migrate_ms)"
      rw_write="$(summary_value "${stage_d_summary}" "${rw_case}" avg_gpu_write_ms)"
      rw_unmap_bytes="$(summary_value "${stage_d_summary}" "${rw_case}" total_apply_cpu_unmap_bytes)"
      rw_unmap_pages="$(summary_value "${stage_d_summary}" "${rw_case}" gpu_only_cpu_unmap_pages_delta)"
      rw_final_on_cpu="$(summary_value "${stage_d_summary}" "${rw_case}" final_query_on_cpu)"

      awk -v pages="${pages}" \
          -v bytes="${bytes}" \
          -v ra="${readdup_apply}" \
          -v rr="${readdup_read}" \
          -v roc="${readdup_on_cpu}" \
          -v rog="${readdup_on_gpu}" \
          -v rup="${readdup_unmap_pages}" \
          -v rpp="${readdup_preserved_pages}" \
          -v sa="${rw_setup_apply}" \
          -v sm="${rw_setup_migrate}" \
          -v wa="${rw_apply}" \
          -v wm="${rw_migrate}" \
          -v ww="${rw_write}" \
          -v wub="${rw_unmap_bytes}" \
          -v wup="${rw_unmap_pages}" \
          -v wfoc="${rw_final_on_cpu}" '
        BEGIN {
          rw_cycle = wa + wm + ww;
          apply_ratio = (sa > 0 ? ra / sa : 0);
          cycle_apply_ratio = (ra > 0 ? rw_cycle / ra : 0);
          cycle_read_ratio = (rr > 0 ? rw_cycle / rr : 0);
          printf "%s,%s,%.6f,%.6f,%s,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%s,%.6f,%.6f,%.6f\n",
                 pages, bytes, ra, rr, roc, rog, rup, rpp,
                 sa, sm, wa, wm, ww, rw_cycle, wub, wup, wfoc,
                 apply_ratio, cycle_apply_ratio, cycle_read_ratio;
        }'
    done
  } > "${COMPARISON}"
}

main() {
  write_metadata

  if [[ "${RUN_STAGE_C}" == "1" ]]; then
    run_stage_c
  fi

  if [[ "${RUN_STAGE_D}" == "1" ]]; then
    run_stage_d
  fi

  write_comparison

  echo
  echo "Done."
  echo "Run root: ${RUN_ROOT}"
  echo "Stage C summary: ${STAGE_C_RUN_ROOT}/summary.csv"
  echo "Stage D summary: ${STAGE_D_RUN_ROOT}/summary.csv"
  echo "Comparison: ${COMPARISON}"
  column -s, -t "${COMPARISON}" || cat "${COMPARISON}"
}

main "$@"
