#!/usr/bin/env bash
# Run one benchmark with exclusive GPU/VRAM, then restore the resident server.
#
# The host keeps a `gufo serve` instance resident, holding tens of GB of VRAM
# and contending for the GPU. Kernel and model benchmarks need that VRAM and an
# uncontended device, so this wrapper stops the server, runs one command, and
# ALWAYS brings the server back before returning -- even if the benchmark fails
# or the wrapper is interrupted. The benchmark's exit status is preserved.
#
# Usage:
#   tools/bench/gpu_exclusive.sh [options] -- <command> [args...]
#
# Options:
#   --launch-script PATH   Fallback launch script used only if no live server
#                          command was captured. Respawn normally replays the
#                          captured command line with an absolute binary path,
#                          so it works from any directory.
#                          [./run_rocm_qwen38_flash_next_mmproj.sh]
#   --host HOST            Host polled for readiness                       [127.0.0.1]
#   --port PORT            Port polled for readiness                       [8083]
#   --expect-model SUBSTR  Require this substring in /v1/models before ready [any model]
#   --ready-timeout SEC    Max wait for the server to serve again          [900]
#   --settle SEC           Extra wait after readiness before returning     [180]
#   --no-kill              Skip stopping a running server (assume VRAM free)
#   --no-respawn           Stop + run only; do NOT respawn (the foreground
#                          harness, e.g. bench_wait.sh, respawns instead)
#   -h, --help             Show this help
#
# Environment:
#   GUFO_BENCH_MIN_AVAIL_MIB  Refuse to start the benchmark unless MemAvailable
#                          is at least this many MiB (Strix Halo UMA: GPU
#                          allocations leave host RAM, so MemAvailable is the
#                          only global headroom signal; rocm-smi only sees the
#                          small visible window). Wait up to
#                          300 s for a stopped server's memory to return,
#                          then exit 75 instead of OOM-killing the host.
#
# Examples:
#   tools/bench/gpu_exclusive.sh -- /tmp/attn_causal_bench
#   tools/bench/gpu_exclusive.sh --expect-model Qwen3.8 -- \
#       env GUFO_QWEN36_A3B_BENCH=2048,4096,100000 \
#       build/gpu-test/staging/tests/models/qwen36_a3b/qwen36_a3b_rocm_forward_test
#   tools/bench/gpu_exclusive.sh -- bash -c '/tmp/bench_a && /tmp/bench_b'
set -euo pipefail

cd "$(dirname "$0")/../.."

# The ROCm environment is mandatory before running ANY gufo process (the
# benchmark and the respawned server alike). Source it once so both inherit it.
# shellcheck disable=SC1091
source ./rocm_env.sh

launch_script="./run_rocm_qwen38_flash_next_mmproj.sh"
host="127.0.0.1"
port="8083"
expect_model=""
ready_timeout=900
settle=60
do_kill=1
no_respawn=0
cmdline_file="${GUFO_SERVE_CMDLINE:-/tmp/gufo_serve_cmdline}"

while [ $# -gt 0 ]; do
  case "$1" in
    --launch-script) launch_script="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --expect-model) expect_model="$2"; shift 2 ;;
    --ready-timeout) ready_timeout="$2"; shift 2 ;;
    --settle) settle="$2"; shift 2 ;;
    --no-kill) do_kill=0; shift ;;
    --no-respawn) no_respawn=1; shift ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "$0" | sed '$d'
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "gpu_exclusive: unknown option '$1' (put the command after --)" >&2
      exit 2
      ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "gpu_exclusive: no benchmark command given after --" >&2
  exit 2
fi

bench_cmd=("$@")
respawned=0

log() { printf '\n=== [gpu_exclusive] %s ===\n' "$*"; }

# Only real `gufo serve` processes: the executable's comm must be `gufo` or a
# versioned archive name `gufo-<digits>` (respawn replays the captured
# /proc/PID/exe path, e.g. /opt/llm/gufo/bin/gufo-20261006, whose comm then
# stops matching a bare `gufo`), AND its args must contain the `serve`
# subcommand. Matching the bare string "gufo serve" with pgrep -f also hits
# wrappers/benchmarks whose command line merely mentions it, which would make
# stop_server kill the benchmark itself.
server_pids() {
  local p comm
  for p in $(pgrep -f gufo 2>/dev/null || true); do
    comm="$(cat "/proc/$p/comm" 2>/dev/null)" || continue
    case "$comm" in
      gufo | gufo-[0-9]*) ;;
      *) continue ;;
    esac
    if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q ' serve '; then
      printf '%s\n' "$p"
    fi
  done
  return 0
}

# Record the exact serve command line (argv[0] resolved to an absolute path via
# /proc/PID/exe) so respawn replays it from any directory, independent of the
# launch script's relative `build/.../gufo` path. Stored NUL-separated.
capture_cmdline() {
  local pid="$1" out="$2" exe a
  exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
  [ -n "$exe" ] || return 1
  # If the binary was replaced while the server ran, /proc may report the old
  # path with a " (deleted)" suffix. Store the normal path so respawn can wait
  # for the recreated executable instead of trying to exec the suffix literally.
  if [[ "$exe" == *" (deleted)" && ! -x "$exe" ]]; then
    exe="${exe% (deleted)}"
  fi
  local -a argv=()
  mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
  [ "${#argv[@]}" -gt 0 ] || return 1
  argv[0]="$exe"
  : > "$out"
  for a in "${argv[@]}"; do printf '%s\0' "$a" >> "$out"; done
  return 0
}

stop_server() {
  local pids first
  pids="$(server_pids)"
  if [ -z "$pids" ]; then
    log "no running 'gufo serve' to stop"
    return 0
  fi
  first="$(printf '%s\n' "$pids" | head -1)"
  if capture_cmdline "$first" "$cmdline_file"; then
    log "captured serve command for respawn -> $cmdline_file"
  fi
  log "stopping gufo serve (pid $pids)"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    [ -z "$(server_pids)" ] && break
    sleep 1
  done
  pids="$(server_pids)"
  if [ -n "$pids" ]; then
    log "force killing gufo serve (pid $pids)"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi
  # A respawner (fallback respawn, launch script) can start a NEW serve while
  # this wrapper is between stop and benchmark. Two serves exceed UMA and the
  # kernel OOM-kills the host, so keep stopping any reappearing serve.
  local round
  for round in 1 2 3; do
    sleep 5
    pids="$(server_pids)"
    [ -z "$pids" ] && break
    log "serve reappeared during stop (round $round/3, pid $pids); stopping again"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  done
  pids="$(server_pids)"
  if [ -n "$pids" ]; then
    echo "gpu_exclusive: ERROR: gufo serve keeps restarting (pid $pids);" >&2
    echo "  refusing to run the benchmark against a contended GPU" >&2
    exit 74
  fi
  log "server stopped"
}

# Strix Halo puts every GPU allocation in host RAM, so a *stopped* server can
# still hold ~100 GiB until the driver reclaims it and free/rocm-smi show no
# per-process GPU memory. MemAvailable is the only global signal: wait for it
# to recover before the benchmark allocates, or fail instead of OOM-killing
# the whole host. Skipped entirely when GUFO_BENCH_MIN_AVAIL_MIB is unset.
require_headroom() {
  [ -n "${GUFO_BENCH_MIN_AVAIL_MIB:-}" ] || return 0
  local need="${GUFO_BENCH_MIN_AVAIL_MIB}" waited=0 avail
  while :; do
    avail="$(awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo)"
    avail=$((avail / 1024))
    if [ "$avail" -ge "$need" ]; then
      log "memory headroom ok (MemAvailable ${avail}MiB >= ${need}MiB)"
      return 0
    fi
    if [ "$waited" -ge 300 ]; then
      echo "gpu_exclusive: ERROR: only ${avail}MiB available (< ${need}MiB" >&2
      echo "  required); GPU memory was not reclaimed - not starting the" >&2
      echo "  benchmark to avoid a host OOM" >&2
      exit 75
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

respawn_server() {
  [ "$respawned" = 1 ] && return 0
  respawned=1
  # Canonical foreground respawn lives in gpu_respawn_server.sh (shared with
  # bench_wait.sh). It replays the captured command line (absolute binary path)
  # and polls /v1/models until ready; --launch-script is only a fallback.
  GUFO_SERVE_CMDLINE="$cmdline_file" tools/bench/gpu_respawn_server.sh \
    --launch-script "$launch_script" --host "$host" --port "$port" \
    --expect-model "$expect_model" --ready-timeout "$ready_timeout"
}

restore() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$no_respawn" != 1 ]; then
    respawn_server || true
    if [ "$settle" -gt 0 ]; then
      log "settling ${settle}s before returning"
      sleep "$settle"
    fi
  fi
  log "done (benchmark exit ${rc})"
  exit "$rc"
}
trap restore EXIT INT TERM

if [ "$do_kill" = 1 ]; then stop_server; fi
require_headroom

log "running benchmark: ${bench_cmd[*]}"
"${bench_cmd[@]}"
# Reaching the end (or any failure) fires the EXIT trap -> restore().
