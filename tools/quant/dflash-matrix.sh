#!/usr/bin/env bash
# Benchmark every DFlash-2 companion against every target quantization.
#
# The autoregressive reference depends only on the target shard, so it is
# measured once per target and shared by all companions through the runner's
# --ar-cache. Companions are rotated within a target on each repetition so
# thermal drift is spread across them instead of biasing the last arm.
set -euo pipefail

BINARY=${BINARY:-./result/bin/gufo}
SUITE=${SUITE:-benchmarks/qwen3.8-27b/speculative-corpus.json}
MAX_TOKENS=${MAX_TOKENS:-512}
DRAFT_TOKENS=${DRAFT_TOKENS:-7}
DRAFT_POLICY=${DRAFT_POLICY:-auto}
# Framing and decode length are the measurement, not a detail. A raw prompt puts
# the model outside the instruction distribution it was tuned on, and a short
# generation is dominated by the unpredictable opening tokens: on one corpus
# prompt the same target and companion read 24.66 tok/s at 53.4% acceptance raw
# at 128 tokens, 27.41 at 63.7% chat-framed, and 33.53 at 77.0% chat-framed at
# 512. Served traffic is chat-framed and answers are long, so that is what this
# sweep measures.
PROMPT_MODE=${PROMPT_MODE:-chat}
REPETITIONS=${REPETITIONS:-1}
OUT=${OUT:-/tmp/dflash-matrix}
SUITE_TAG=${SUITE_TAG:-$(basename "${SUITE%.json}")}
TIMEOUT=${TIMEOUT:-1800}

declare -A TARGETS=(
  [q8]=models/Qwen3.8-27B-UD-Q8_K_L.gguf
  [q4]=models/Qwen3.8-27B-UD-Q4_K_XL.gguf
)
declare -A DRAFTS=(
  [dq4]=models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
  [dq8]=models/Qwen3.8-27B-DFlash2-Q8_0.gguf
  [dbf16]=models/Qwen3.8-27B-DFlash2-BF16.gguf
)

TARGET_ORDER=${TARGET_ORDER:-"q8 q4"}
DRAFT_ORDER=${DRAFT_ORDER:-"dq4 dq8 dbf16"}

mkdir -p "$OUT"

for rep in $(seq 1 "$REPETITIONS"); do
  for target in $TARGET_ORDER; do
    drafts=$DRAFT_ORDER
    if [ $((rep % 2)) -eq 0 ]; then
      drafts=$(echo "$DRAFT_ORDER" | tr ' ' '\n' | tac | tr '\n' ' ')
    fi
    # A cold first touch of the target shard costs far more than the whole
    # measurement: faulting in the 28 GB Q8 shard against a full page cache
    # took 791 s against 6.6 s warm, and that latency would land entirely on
    # whichever companion happened to run first. Warm the shard before the
    # arms so every companion sees the same resident state.
    cat "${TARGETS[$target]}" > /dev/null 2>&1 || true
    for draft in $drafts; do
      cat "${DRAFTS[$draft]}" > /dev/null 2>&1 || true
      tag="${target}-${draft}-t${MAX_TOKENS}-w${DRAFT_TOKENS}-${PROMPT_MODE}-${SUITE_TAG}-r${rep}"
      report="$OUT/$tag.json"
      if [ -f "$report" ]; then
        echo "skip $tag (already measured)"
        continue
      fi
      echo "=== $tag ==="
      python3 tools/quant/speculative-corpus.py \
        --binary "$BINARY" \
        --model "${TARGETS[$target]}" \
        --draft-model "${DRAFTS[$draft]}" \
        --backend dflash2 \
        --suite "$SUITE" \
        --max-tokens "$MAX_TOKENS" \
        --draft-tokens "$DRAFT_TOKENS" \
        --draft-policy "$DRAFT_POLICY" \
        --prompt-mode "$PROMPT_MODE" \
        --timeout "$TIMEOUT" \
        --allow-mismatch \
        --ar-cache "$OUT/ar-${target}-t${MAX_TOKENS}-${PROMPT_MODE}.json" \
        --label "$tag" \
        --json "$report" \
        2>&1 | tee "$OUT/$tag.log"
    done
  done
done
