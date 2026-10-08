#!/usr/bin/env bash
# Builds tools/gemma4/llama_logits and llama_vision against the pinned
# llama.cpp reference (.#llama-cpp-reference). Output: artifacts/gemma4/
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
lib="$(nix build "$root#llama-cpp-reference" --no-link --print-out-paths)"
dev="$(nix build "$root#llama-cpp-reference^dev" --no-link --print-out-paths)"
mkdir -p "$root/artifacts/gemma4"
nix develop "$root" -c c++ -std=c++17 -O2 -o "$root/artifacts/gemma4/llama_logits" \
  "$root/tools/gemma4/llama_logits.cpp" -DGUFO_LLAMA_BACKEND_DIR="\"$lib/bin\"" -I"$dev/include" -L"$lib/lib" -L"$dev/lib" \
  -Wl,-rpath,"$lib/lib" -lllama -lggml -lggml-base
nix develop "$root" -c c++ -std=c++17 -O2 -o "$root/artifacts/gemma4/llama_vision" \
  "$root/tools/gemma4/llama_vision.cpp" -DGUFO_LLAMA_BACKEND_DIR="\"$lib/bin\"" -I"$dev/include" -L"$lib/lib" -L"$dev/lib" \
  -Wl,-rpath,"$lib/lib" -lmtmd -lllama -lggml -lggml-base
echo "$root/artifacts/gemma4/llama_logits"
echo "$root/artifacts/gemma4/llama_vision"
