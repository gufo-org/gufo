#!/bin/sh
# usage: fabric-clock.sh hold | release | status   (hold and release need root)
# Pins the GPU fabric clock (pp_dpm_fclk) at its top level. The NPU build of
# gufo needs it held while the GPU and the NPU work together.
set -u
for dev in /sys/class/drm/card*/device; do
  [ -e "$dev/pp_dpm_fclk" ] && break
done
[ -e "$dev/pp_dpm_fclk" ] || { echo "fabric-clock: no amdgpu pp_dpm_fclk" >&2; exit 1; }
top=$(grep -c . "$dev/pp_dpm_fclk")
top=$((top - 1))

held() {
  [ "$(cat "$dev/power_dpm_force_performance_level")" = manual ] &&
    [ "$(grep -c '\*' "$dev/pp_dpm_fclk")" = 1 ] &&
    grep '\*' "$dev/pp_dpm_fclk" | grep -q "^$top:"
}

case "${1:-status}" in
  hold)
    echo manual > "$dev/power_dpm_force_performance_level" &&
      echo "$top" > "$dev/pp_dpm_fclk" || exit 1
    i=0
    until held || [ "$i" -ge 50 ]; do sleep 0.1; i=$((i + 1)); done
    held && echo "fclk held at level $top" || { echo "fabric-clock: could not hold it" >&2; exit 1; } ;;
  release)
    echo auto > "$dev/power_dpm_force_performance_level" && echo released ;;
  status)
    if held; then echo "held"; else echo "NOT held"; exit 1; fi ;;
  *) echo "usage: $0 hold | release | status" >&2; exit 2 ;;
esac
