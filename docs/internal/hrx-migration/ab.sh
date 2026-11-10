#!/usr/bin/env bash
# One guarded Flash-Next bench run of the same Gufo binary on stock HIP or HRX.
#
#   GUFO_HRX=<.#gufo-hrx out-link> MODEL=<first GGUF shard> OUT=<dir> \
#     ab.sh <hip|hrx> <tag> [gufo bench args...]
#
# Refuses to start beside another model process, cools the GPU to 38 C or below
# plus 60 s idle, runs inside a memory-capped systemd scope, and appends any
# amdgpu fault from the kernel log. Extra environment (for example a probe
# switch) passes through to Gufo.
set -uo pipefail
RT=$1; TAG=$2; shift 2
: "${GUFO_HRX:?set GUFO_HRX to a .#gufo-hrx out-link}"
: "${MODEL:?set MODEL to the first GGUF shard}"
: "${OUT:?set OUT to a results directory}"
HRX_GUFO=$GUFO_HRX/bin/gufo
# The wrapper names the unwrapped Gufo it runs; that binary is the HIP baseline.
HIP_GUFO=$(grep -oE '/nix/store/[^ "]+-gufo-[^ "/]+/bin/gufo' "$HRX_GUFO" | head -1)
if pgrep -x .gufo-wrapped >/dev/null || pgrep -x gpu_probe >/dev/null ||
   pgrep -x llama-server >/dev/null; then
  echo "model process already running; refusing" >&2; exit 2
fi
temp() { cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1; }
for _ in $(seq 1 120); do
  t=$(temp); [ -z "$t" ] && break; [ "$t" -le 38000 ] && break; sleep 5
done
sleep 60
mkdir -p "$OUT"
echo "start temp $(temp)" > "$OUT/$TAG.log"
G=$HIP_GUFO; [ "$RT" = hrx ] && G=$HRX_GUFO
systemd-run --user --scope -q -p MemoryMax=118G -p TasksMax=256 -- \
  "$G" bench -m "$MODEL" "$@" >> "$OUT/$TAG.log" 2>&1
rc=$?
echo "exit $rc end temp $(temp)" >> "$OUT/$TAG.log"
k=$(journalctl -k --since '-15 min' --no-pager | grep -i amdgpu |
    grep -E 'fault|MES|reset|Illegal' | tail -1)
[ -n "$k" ] && echo "KLOG: $k" >> "$OUT/$TAG.log"
exit $rc
