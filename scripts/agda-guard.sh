#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 64
fi

detect_mem_total_kb() {
  if [[ -r /proc/meminfo ]]; then
    local kb
    kb=$(awk '/MemTotal:/ { print $2; exit }' /proc/meminfo)
    if [[ -n "${kb}" ]]; then
      printf '%s\n' "${kb}"
      return 0
    fi
  fi

  if command -v sysctl >/dev/null 2>&1; then
    local bytes
    bytes=$(sysctl -n hw.memsize 2>/dev/null || true)
    if [[ -n "${bytes}" ]]; then
      printf '%s\n' "$((bytes / 1024))"
      return 0
    fi
  fi

  printf '%s\n' 4194304
}

# Virtual-memory cap is OPT-IN.  Set AGDA_GUARD_VMEM_PCT to a percentage (e.g. 80)
# to cap virtual memory at that fraction of RAM.  Default (unset/0) = no cap:
# `ulimit -Sv` throttles GHC/Agda and was the cause of stalled builds.
vmem_pct="${AGDA_GUARD_VMEM_PCT:-0}"
mem_total_kb=$(detect_mem_total_kb)
limit_kb=$(( mem_total_kb * vmem_pct / 100 ))
if ((vmem_pct > 0)) && ((limit_kb < 262144)); then
  limit_kb=262144   # floor at 256 MB when a cap is requested
fi

run_with_limit() {
  if ((vmem_pct > 0)); then
    (
      ulimit -Sv "${limit_kb}"
      "$@"
    )
  else
    "$@"
  fi
}

is_oom_failure() {
  local status="$1"
  local log_file="$2"

  if [[ "${status}" -eq 137 || "${status}" -eq 134 ]]; then
    return 0
  fi

  local log_text
  log_text=$(<"${log_file}")
  case "${log_text}" in
    *"heap exhausted"*|*"out of memory"*|*"Out of memory"*|*"cannot allocate memory"*|*"Memory exhausted"*|*"allocation limit"*|*"Heap exhausted"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

tmp_log=$(mktemp)
trap 'rm -f "${tmp_log}"' EXIT

for attempt in 1 2; do
  set +e
  run_with_limit "$@" 2>&1 | tee "${tmp_log}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -eq 0 ]]; then
    exit 0
  fi

  if [[ "${attempt}" -eq 1 ]] && is_oom_failure "${status}" "${tmp_log}"; then
    echo "Memory cap hit (80% RAM). Releasing resources and retrying once..." >&2
    sync || true
    sleep 2
    : > "${tmp_log}"
    continue
  fi

  exit "${status}"
done
