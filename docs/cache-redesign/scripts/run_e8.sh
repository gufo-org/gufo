#!/usr/bin/env bash
# E8 on every trace (two-session runs and concurrency-4 runs) plus restarts.
set -u
cd "$(dirname "$0")"
for model in fn q27; do
  for w in w1 w2 w3 w4; do
    "${PYTHON:-python3}" simulate_e8.py "$model" "$w" > /dev/null || echo "FAIL $model $w"
  done
  for w in w2-c4 w3-c4; do
    "${PYTHON:-python3}" simulate_e8.py "$model" "$w" --sessions 4 > /dev/null || echo "FAIL $model $w"
  done
  for r in 12 20 28; do
    "${PYTHON:-python3}" simulate_e8.py "$model" w1 --restart-at "$r" > /dev/null || echo "FAIL $model w1 r$r"
  done
  echo "done $model $(date +%T)"
done
echo "=== e8 done $(date +%T)"
