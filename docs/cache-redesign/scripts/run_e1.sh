#!/usr/bin/env bash
# Runs every E1 configuration sequentially; skips a run if another model
# server is on the GPU, and stops if free disk space drops below 30 GB.
set -u
cd "$(dirname "$0")"
FN=${FN_MODEL:-/persist/models/qwen38-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
MTP=${FN_MTP:-/persist/models/qwen38-flash-next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
Q27=${Q27_MODEL:-/persist/models/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf}
DF=${Q27_DFLASH:-/persist/models/Qwen3.8-27B-DFlash2-Q8_0.gguf}

run() {
  local name=$1; shift
  if pgrep -f 'llama-server|bin/gufo serve' >/dev/null; then
    echo "SKIP $name: another model server is running"; return
  fi
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt 30 ]; then echo "STOP: only ${free_gb}G free"; exit 1; fi
  echo "=== $name (free ${free_gb}G) $(date +%T)"
  python3 e1_snapshot_size.py --name "$name" "$@" || echo "FAIL $name"
}

run fn-ar-260k --model "$FN" --context 260000 --depths 2048,8192,32768,65536,131072 \
  --extra "--prefill-chunk 2048"
run fn-mtp-260k --model "$FN" --context 260000 --depths 2048,32768,131072 \
  --extra "--prefill-chunk 2048 --speculative mtp --mtp-model $MTP"
run fn-ar-32k --model "$FN" --context 32768 --depths 2048,8192,16384,30000 \
  --extra "--prefill-chunk 2048"
run q27-ar-256k --model "$Q27" --context 256000 --depths 2048,8192,32768,65536
run q27-df-256k --model "$Q27" --context 256000 --depths 2048,32768 \
  --extra "--speculative dflash2 --dflash-model $DF --draft-tokens 7"
run q27-ar-32k --model "$Q27" --context 32768 --depths 2048,8192,16384,30000
echo "=== done $(date +%T)"
