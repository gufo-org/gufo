#!/usr/bin/env bash
# E7 on every E2 trace, plus simulated restarts at fixed depths.
set -u
cd "$(dirname "$0")"
for model in fn q27; do
  for w in w1 w2 w3 w4; do
    "${PYTHON:-python3}" simulate_e7.py "$model" "$w" > /dev/null || echo "FAIL $model $w"
    echo "done $model $w $(date +%T)"
  done
  for r in 12 20 28; do
    "${PYTHON:-python3}" simulate_e7.py "$model" w1 --restart-at "$r" > /dev/null || echo "FAIL $model w1 r$r"
    echo "done $model w1 restart $r $(date +%T)"
  done
  "${PYTHON:-python3}" simulate_e7.py "$model" w2 --restart-at 18 > /dev/null || echo "FAIL $model w2 r18"
  "${PYTHON:-python3}" simulate_e7.py "$model" w3 --restart-at 30 > /dev/null || echo "FAIL $model w3 r30"
  echo "done $model restarts $(date +%T)"
done
echo "=== e7 done $(date +%T)"
