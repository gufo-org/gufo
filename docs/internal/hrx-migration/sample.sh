#!/usr/bin/env bash
# Sample GPU sclk, power, temp, busy, fclk, mclk and host CPU busy jiffies every 50 ms.
# Usage: sample.sh <out>
D=$(ls -d /sys/class/drm/card*/device | head -1)
H=$(ls -d "$D"/hwmon/hwmon* | head -1)
act() { awk '/\*/ {gsub(/Mhz/,"",$2); print $2}' "$1" 2>/dev/null; }
while :; do
  f=$(cat "$H/freq1_input"); p=$(cat "$H/power1_average"); t=$(cat "$H/temp1_input")
  b=$(cat "$D/gpu_busy_percent"); fc=$(act "$D/pp_dpm_fclk"); mc=$(act "$D/pp_dpm_mclk")
  read -r _ u n s i _ < /proc/stat
  echo "$(date +%s.%N) $((f/1000000)) $((p/1000000)) $((t/1000)) $b ${fc:-0} ${mc:-0} $((u+n+s)) $i"
  sleep 0.05
done > "$1"
