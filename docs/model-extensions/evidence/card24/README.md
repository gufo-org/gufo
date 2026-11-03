# Card 24 Evidence: Second-Provider Conformance (Text Generation via `lfm3-gufo-extension`)

## 1. Summary

**Status: complete** for the `lfm3-gufo-extension` text-generation evidence recorded here.

- **External Repository**: `git@github.com:gufo-org/lfm3-gufo-extension.git` (`lfm3-gufo-extension`, commit `6360b0fceacc7e98910c6d686c29984dc90374ea`)
- **Capability Covered**: `text_generation` (genuine external model port of Liquid LFM3 1.7B (`gguf:lfm3`, model IDs `["LFM3-1.7B", "lfm3-1.7b"]`) extracted from the `lfm3` branch and adapted to `<gufo/ext/...>` public headers).
- **Zero-Core-Diff Proof**: Building, testing, probing, benchmarking, and serving `lfm3-gufo-extension` via `-DGUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension` required **zero** changes to Gufo core, build, preset, workflow, Nix, or registry source files (`jj diff` across `src/`, `include/`, `cmake/`, `flake.nix`, `CMakeLists.txt`, `CMakePresets.json`, and `extensions.lock` is empty).
- **Scope**: All LFM3 build, numerical, CLI, HTTP serving, and HIP-free CPU checks documented below are complete and passed. Broader capability coverage and exclusion-matrix requirements are tracked in [Card 24](../../cards/24-second-provider-conformance.md); this record establishes the external `text_generation` second-provider result.

## 2. Capability-by-Provider Matrix (Current Status)

| Capability | Primary In-Tree Providers | Second External Provider | Conformance Status |
| --- | --- | --- | --- |
| `text_generation` | `qwen27-q4`, `flash-next-q4`, `deepseek4` | `lfm3` (`lfm3-gufo-extension` @ `6360b0fc`) | **PASS** (CPU/GPU build, unit/ops/session tests, CPU vs GPU logit probe, CLI `prompt`/`bench`, live `serve llm` HTTP `/v1/models` + `/v1/chat/completions` non-stream/stream/validation/2-way batching, HIP-free CPU probe & diagnostics) |
| `embeddings` | `embeddinggemma-2` | — | Pending |
| `speech_recognition` | `qwen3-asr` | — | Pending |
| `speech_synthesis` | `qwen3-tts` | — | Pending |
| `image_generation` | `qwen-image-21` | — | Pending |
| `video_generation` | `minimax-h3` | — | Pending |

## 3. Executed Commands & Retained Verdicts (`lfm3-gufo-extension`)

### 3.1 GPU Build & Extension Provenance (`gpu-test`)

```sh
nix develop -c cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension
nix develop -c cmake --build --preset gpu-test --target lfm3_config_test lfm3_ops_test lfm3_session_test lfm3_probe gufo --parallel 8
/home/mixer/gufo/build/gpu-test/gufo diagnose --section extensions --json
```

- **Result**: Exit `0` (0 warnings, 0 errors).
- **Diagnostics excerpt**:
  ```json
  {
    "extensionApiVersion": "0.1.9",
    "selectionMode": "discovery",
    "builtExtensions": [
      {"name": "lfm3", "version": "0.1.0", "source": "lfm3-gufo-extension"}
    ],
    "providers": [
      {"id": "lfm3", "extension": "lfm3", "available": true}
    ]
  }
  ```

### 3.2 Standalone Unit, HIP Kernel, Multi-Session, and CPU/GPU Logit Probe Tests

| Target / Command | Verdict | Key Output |
| --- | --- | --- |
| `build/gpu-test/extensions/build/lfm3/build/lfm3_config_test` | **PASS** | `PASS LFM3 metadata and chat framing` |
| `build/gpu-test/extensions/build/lfm3/build/lfm3_ops_test` | **PASS** | `PASS LFM3 convolution, RoPE, gated GQA and sliding ring against independent formulas` |
| `build/gpu-test/extensions/build/lfm3/build/lfm3_session_test models/lfm3/LFM3-1.7B-Q8_0.gguf` | **PASS** | `513-token batched/serial max_abs=3.19481e-05` (`PASS LFM3 513-token prefill crosses sliding window boundary`, `PASS LFM3 session state isolation and prefix restore`, `2/3/4/8-session batch/serial max_abs <= 9.17912e-06`, `PASS LFM3 multi-session batched prefill and decode`) |
| `lfm3_probe models/lfm3/LFM3-1.7B-Q8_0.gguf cpu` vs `gpu` | **PASS** | `count=128000 max_abs=3.458261e-04 mean_abs=5.191900e-05 top1_cpu=4741 top1_gpu=4741` (`4741 15.856  Paris`) |
| `gufo prompt -m models/lfm3/LFM3-1.7B-Q8_0.gguf -p "The capital of France is" -n 32` | **PASS** | Dispatched via `lfm3` `TextProvider::prompt_driver` |
| `gufo bench -m models/lfm3/LFM3-1.7B-Q8_0.gguf -p 128 -n 32 -r 1` | **PASS** | `pp128`: `4725.66 tok/s`, `tg32`: `106.153 tok/s` |

### 3.3 Live HTTP Serving Conformance (`gufo serve llm -j 2 -c 1024`)

- `GET /v1/models`:
  ```json
  {"object": "list", "data": [{"id": "LFM3-1.7B", "object": "model", "owned_by": "gufo", "context_length": 1024, "architecture": {"input_modalities": ["text"]}}]}
  ```
- `POST /v1/chat/completions` (`enable_thinking=false`):
  ```json
  {"index": 0, "message": {"role": "assistant", "content": "The capital of France is **Paris**. It is the country's largest city and a major cultural, political, and economic center. Paris is known for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."}, "finish_reason": "stop"}
  ```
- `POST /v1/chat/completions` (`enable_thinking=true`):
  ```json
  {"index": 0, "message": {"role": "assistant", "reasoning_content": "The user asks: \"What is the capital of France?\" This is a straightforward factual question. The answer: Paris.\n\nWe should respond concisely. There's no policy violation. So answer: The capital of France is Paris.\n\nThus final.", "content": "The capital of France is **Paris**."}, "finish_reason": "stop"}
  ```
- `POST /v1/chat/completions` (`stream=true`):
  - `{"chunk_count": 14, "done_seen": true, "first_role": "assistant", "last_finish": "length"}`
- Request validation:
  - Unknown model (`"unknown-model"`): HTTP `404` (`{"error":{"message":"model 'unknown-model' is not served by this process","type":"invalid_request_error","code":"model_not_found"}}`)
  - Image input on text-only `LFM3-1.7B`: HTTP `400` (`{"error":{"message":"model does not support image input","type":"invalid_request_error","code":"invalid_messages"}}`)
- Concurrent 2-session batching (`-j 2`):
  - `[[200, "length", 8], [200, "length", 8]]`

### 3.4 HIP-Free CPU Build, Probe & Diagnostics (`cpu-test`)

```sh
nix develop -c cmake --preset cpu-test -DGUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension
nix develop -c cmake --build --preset cpu-test --target gufo lfm3_config_test lfm3_probe --parallel 8
/home/mixer/gufo/build/cpu-test/extensions/build/lfm3/build/lfm3_config_test
/home/mixer/gufo/build/cpu-test/extensions/build/lfm3/build/lfm3_probe /home/mixer/gufo/models/lfm3/LFM3-1.7B-Q8_0.gguf cpu
/home/mixer/gufo/build/cpu-test/gufo diagnose --section extensions --json
```

- **Result**: Exit `0`.
- `lfm3_config_test`: `PASS LFM3 metadata and chat framing`
- `lfm3_probe ... cpu`: `4741 15.856  Paris`
- `gufo diagnose --section extensions --json`:
  - `"builtExtensions"` includes `{"name":"lfm3","version":"0.1.0","source":"lfm3-gufo-extension"}`
  - `"providers"` includes `{"id":"lfm3","extension":"lfm3","available":false,"reason":"requires hip"}`
