#!/usr/bin/env bash
# E8 revision 2 on every trace (two-session and concurrency-4 runs) plus
# simulated restarts during the W1 agent sessions, graceful and abrupt.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for model in fn q27; do
  for w in w1 w2 w3 w4; do
    "$PY" simulate_e8.py "$model" "$w" > /dev/null || echo "FAIL $model $w"
  done
  for w in w2-c4 w3-c4; do
    "$PY" simulate_e8.py "$model" "$w" --sessions 4 > /dev/null || echo "FAIL $model $w"
  done
  for r in 12 20 28; do
    "$PY" simulate_e8.py "$model" w1 --restart-at "$r" > /dev/null || echo "FAIL $model w1 r$r"
    "$PY" simulate_e8.py "$model" w1 --restart-at "$r" --abrupt > /dev/null || echo "FAIL $model w1 r$r abrupt"
  done
  echo "done $model $(date +%T)"
done
echo "=== e8 done $(date +%T)"
