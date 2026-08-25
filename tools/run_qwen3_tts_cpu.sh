#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${QWEN3_TTS_BUILD_ROOT:-}"
if [[ -z "$build_root" ]]; then
  build_root="$(mktemp -d /tmp/strix-qwen3-tts-cpu.XXXXXX)"
fi

git -C "$repo_root" add \
  CMakeLists.txt flake.nix .devops/nix/package.nix \
  src/models/qwen3_tts src/cli tests/models/qwen3_tts tests/cli tools

exec nix develop --command bash -c "
  cmake -S '$repo_root' -B '$build_root' -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF
  cmake --build '$build_root' --target \
    strix qwen3_tts_loader_test qwen3_tts_audio_api_test \
    qwen3_tts_config_test
  ctest --test-dir '$build_root' --output-on-failure -R qwen3_tts
"
