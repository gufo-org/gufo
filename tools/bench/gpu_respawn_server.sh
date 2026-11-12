#!/usr/bin/env bash
# Foreground: (re)start the resident `gufo serve` and wait until it serves again.
#
# This is the canonical respawn used by gpu_exclusive.sh and bench_wait.sh. It
# replays the exact command line captured before the server was stopped (stored
# NUL-separated in $GUFO_SERVE_CMDLINE, with argv[0] resolved to an absolute path
# via /proc/PID/exe), detached with nohup so it outlives this call, then polls
# /v1/models until it answers -- optionally requiring a model substring -- then
# optionally settles. Replaying the captured command makes respawn work from any
# directory and independent of the launch script's relative binary path; the
# launch script is used only as a fallback when nothing was captured. Run it in
# the FOREGROUND so the caller only returns once the resident server is back.
# Keeping the respawn here (rather than inside a backgrounded benchmark) means a
# killed benchmark can never take an in-flight respawn down with it.
#
# Usage:
#   tools/bench/gpu_respawn_server.sh [options]
#
# Options:
#   --launch-script PATH   Fallback launch script when no command was captured
#                          [./run_rocm_qwen38_flash_next_mmproj.sh]
#   --host HOST            Host polled for readiness                    [127.0.0.1]
#   --port PORT            Port polled for readiness                    [8083]
#   --expect-model SUBSTR  Require this substring in /v1/models         [any model]
#   --ready-timeout SEC    Max wait for the server to serve             [900]
#   --settle SEC           Extra wait after readiness before returning  [0]
#   -h, --help             Show this help
set -uo pipefail

cd "$(dirname "$0")/../.."

# The ROCm environment is mandatory before running gufo (LD_LIBRARY_PATH etc.);
# without it the respawned server dies loading libhipblas. Source it here so the
# captured command (or fallback launch script) starts with the correct env.
# shellcheck disable=SC1091
source ./rocm_env.sh

launch_script="./run_rocm_qwen38_flash_next_mmproj.sh"
host="127.0.0.1"
port="8083"
expect_model=""
ready_timeout=900
settle=0
launch_attempts=10
launch_retry_sleep=30
respawn_log="/tmp/gufo_serve_respawn.log"
cmdline_file="${GUFO_SERVE_CMDLINE:-/tmp/gufo_serve_cmdline}"

while [ $# -gt 0 ]; do
  case "$1" in
    --launch-script) launch_script="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --expect-model) expect_model="$2"; shift 2 ;;
    --ready-timeout) ready_timeout="$2"; shift 2 ;;
    --settle) settle="$2"; shift 2 ;;
    -h | --help)
      sed -n '2,/^set -uo/p' "$0" | sed '$d'
      exit 0
      ;;
    *)
      echo "gpu_respawn_server: unknown option '$1'" >&2
      exit 2
      ;;
  esac
done

url="http://${host}:${port}/v1/models"
log() { printf '\n=== [gpu_respawn] %s ===\n' "$*"; }

# Refuse to launch while any gufo serve is resident or still loading (same
# matcher as gpu_exclusive.sh). Two serve processes exceed Strix Halo UMA and
# the kernel OOM-kills the whole host; a respawner woken during another run
# must defer to the serve already on the GPU.
# Even with no serve visible, the UMA may not be reclaimed yet (driver lag
# after a kill, or a benchmark still holding it). MemAvailable is the only
# global signal on Strix Halo; refuse to launch below this floor (46 GiB:
# healthy idle hosts sit far above, any in-flight big-model run is far below).
respawn_headroom_ok() {
  local floor="${GUFO_RESPAWN_MIN_AVAIL_MIB:-47104}" avail
  avail="$(awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo)"
  avail=$((avail / 1024))
  if [ "$avail" -lt "$floor" ]; then
    log "only ${avail}MiB MemAvailable (< ${floor}MiB floor); another GPU job still holds UMA - not launching"
    exit 0
  fi
}

defer_to_existing_serve() {
  local p comm
  for p in $(pgrep -f gufo 2>/dev/null || true); do
    comm="$(cat "/proc/$p/comm" 2>/dev/null)" || continue
    case "$comm" in
      gufo | gufo-[0-9]*) ;;
      *) continue ;;
    esac
    if tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null | grep -q ' serve '; then
      log "another gufo serve is resident (pid $p); not launching to avoid UMA OOM"
      exit 0
    fi
  done
}

# Idempotent: a caller racing another respawner across the respawn flock must
# not start a second server on the port. If something already serves (and
# matches --expect-model), return success without launching.
body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
if [ -n "$body" ] && { [ -z "$expect_model" ] || printf '%s' "$body" | grep -q "$expect_model"; }; then
  log "already serving at $url; nothing to do"
  exit 0
fi

# Primary: replay the exact command line captured before the server was stopped
# (absolute binary path), so respawn works from any directory. Fallback: the
# launch script, only when nothing was captured (e.g. no server was running).
launch_captured() {
  local -a argv=()
  mapfile -d '' -t argv < "$cmdline_file" 2>/dev/null || return 1
  [ "${#argv[@]}" -gt 0 ] || return 1

  # A server stopped while its binary was being rebuilt can expose its exe link
  # as "/path/to/gufo (deleted)". The replacement build recreates the same path,
  # so strip the kernel suffix and retry until the file is executable again.
  if [[ "${argv[0]}" == *" (deleted)" && ! -x "${argv[0]}" ]]; then
    argv[0]="${argv[0]% (deleted)}"
  fi

  local attempt
  for attempt in $(seq 1 "$launch_attempts"); do
    pid=""
    if [ ! -x "${argv[0]}" ]; then
      log "captured binary is not executable yet (attempt $attempt/$launch_attempts): ${argv[0]}"
      sleep "$launch_retry_sleep"
      continue
    fi

    defer_to_existing_serve
    respawn_headroom_ok
    log "respawning captured command (attempt $attempt/$launch_attempts, log: $respawn_log): ${argv[*]}"
    setsid "${argv[@]}" </dev/null >"$respawn_log" 2>&1 &
    pid=$!
    disown "$pid" 2>/dev/null || true

    # Give exec a moment to fail (for example, a deleted binary or bad shebang).
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi

    log "captured launch exited early (attempt $attempt/$launch_attempts); tail of $respawn_log:"
    tail -n 5 "$respawn_log" 2>/dev/null || true
    if [ "$attempt" -lt "$launch_attempts" ]; then
      sleep "$launch_retry_sleep"
    fi
  done

  pid=""
  return 1
}

launch_fallback() {
  if [ ! -x "$launch_script" ]; then
    log "ERROR: no captured command and launch script not executable: $launch_script"
    exit 1
  fi
  defer_to_existing_serve
  respawn_headroom_ok
  log "no captured command; falling back to $launch_script (log: $respawn_log)"
  setsid "$launch_script" </dev/null >"$respawn_log" 2>&1 &
  pid=$!
  disown "$pid" 2>/dev/null || true
}

pid=""
if [ -s "$cmdline_file" ]; then launch_captured || pid=""; fi
[ -n "$pid" ] || launch_fallback
log "server launching (pid $pid); polling $url (timeout ${ready_timeout}s)"

waited=0
body=""
while [ "$waited" -lt "$ready_timeout" ]; do
  if ! kill -0 "$pid" 2>/dev/null; then
    log "ERROR: server process exited early; tail of $respawn_log:"
    tail -n 20 "$respawn_log" 2>/dev/null || true
    exit 1
  fi
  body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
  if [ -n "$body" ] && { [ -z "$expect_model" ] || printf '%s' "$body" | grep -q "$expect_model"; }; then
    log "server ready after ${waited}s"
    if [ "$settle" -gt 0 ]; then
      log "settling ${settle}s before returning"
      sleep "$settle"
    fi
    exit 0
  fi
  sleep 5
  waited=$((waited + 5))
done

log "ERROR: server not ready within ${ready_timeout}s; tail of $respawn_log:"
tail -n 20 "$respawn_log" 2>/dev/null || true
exit 1