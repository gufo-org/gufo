#!/usr/bin/env bash
# One guarded Flash-Next bench run. Usage: run1.sh <hip|hrx> <tag> [bench args...]
set -uo pipefail
SP=${HRX_PROBE_DIR:?set HRX_PROBE_DIR to the probe work directory}
RT=$1; TAG=$2; shift 2
M=/persist/models/store/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
if pgrep -x .gufo-wrapped >/dev/null || pgrep -x gpu_probe >/dev/null || pgrep -x llama-server >/dev/null; then
  echo "model process already running; refusing" >&2; exit 2
fi
temp() { cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1; }
# Cool down to <= 38 C, then 60 s idle.
for _ in $(seq 1 120); do t=$(temp); [ -z "$t" ] && break; [ "$t" -le 38000 ] && break; sleep 5; done
sleep 60
echo "start temp $(temp)" > "$SP/res/$TAG.log"
ENV=()
[ "$RT" = hrx ] && ENV=(LD_LIBRARY_PATH="$SP/hrx-build2/libhrx/src/binding/hip")
systemd-run --user --scope -q -p MemoryMax=118G -p TasksMax=256 -- \
  env "${ENV[@]}" "$SP/gufo-main/bin/gufo" bench -m "$M" "$@" >> "$SP/res/$TAG.log" 2>&1
rc=$?
echo "exit $rc end temp $(temp)" >> "$SP/res/$TAG.log"
exit $rc
