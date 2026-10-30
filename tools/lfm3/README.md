# Independent LFM3 teacher

Use llama.cpp revision `8642541b255bfb5dc39cd351a178be09013a55d6`.
The [model graph](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/models/lfm3.cpp)
and [official implementation](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/main)
are the independent architecture references. No llama.cpp runtime is linked
into production Gufo.

Build upstream's CPU `llama` and `llama-quantize` targets in a checkout at
`/tmp/gufo-lfm3-llama`, then inside `nix develop`:

```sh
c++ -std=c++17 -O2 -I/tmp/gufo-lfm3-llama/include \
  -I/tmp/gufo-lfm3-llama/ggml/include tools/lfm3/llama_probe.cpp \
  -L/tmp/gufo-lfm3-llama/build/bin \
  -Wl,-rpath,/tmp/gufo-lfm3-llama/build/bin -lllama -o /tmp/lfm3-llama-probe
/tmp/gufo-lfm3-llama/build/bin/llama-quantize \
  models/lfm3/LFM3-1.7B-BF16.gguf models/lfm3/LFM3-1.7B-F32.gguf F32 8
cmake --build --preset gpu-test --target lfm3_probe lfm3_session_test
python3 tests/functional/lfm3_logits.py \
  --probe build/gpu-test/tests/models/lfm3/lfm3_probe \
  --llama-probe /tmp/lfm3-llama-probe \
  --bf16 models/lfm3/LFM3-1.7B-BF16.gguf --q8 models/lfm3/LFM3-1.7B-Q8_0.gguf \
  --teacher-f32 models/lfm3/LFM3-1.7B-F32.gguf --output artifacts/lfm3/logits
build/gpu-test/tests/models/lfm3/lfm3_session_test models/lfm3/LFM3-1.7B-Q8_0.gguf
```

The BF16-to-F32 weight conversion is lossless. It removes upstream's activation
quantization from the comparison, isolating architecture and FP32 operator
semantics. Q8 quality is separately compared with BF16 on identical tokens.
Full logits and process logs remain in ignored `artifacts/lfm3/`.

HTTP checks are opt-in:

```sh
python3 tests/functional/lfm3.py --url http://127.0.0.1:18083 \
  --output artifacts/lfm3/http.json
```

Start a two-session server first, with `--max-pending-per-client 4` for the
concurrency case. The script retains every request and response, including
per-phase timings. It exercises only LFM3 and is outside hosted model CI.

The teacher explicitly uses FP32 K/V caches and disables flash attention. Earlier
retained results used the upstream default FP16 K/V cache; the newer
`logits-wired-fp32kv` run replaces those numbers for architecture parity.
`llama_probe --gpu MODEL OUTPUT TOKEN...` is an optional GPU diagnostic; the
qualification command above keeps the independent CPU teacher.

`lfm3_chat_probe MODEL REQUEST.json OUTPUT.json` is a model-private reproduction
tool, built with `cmake --build --preset gpu-test --target lfm3_chat_probe`.
Recipes contain Chat-style `messages`, optional function `tools`, a `thinking`
boolean, optional `response_schema`, and `max_tokens` (zero dumps the prompt
without model execution). `tool_choice` may be `required` or auto. It saves the
rendered prompt, exact token IDs and raw greedy generation, using LFM sampling
defaults with temperature zero. This bounded helper does not implement every
HTTP option. Use its fixed token histories with the independent logit probe
when classifying a failed HTTP answer.
