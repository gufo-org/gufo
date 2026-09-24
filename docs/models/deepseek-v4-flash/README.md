# DeepSeek V4 Flash

Text generation on gfx1151 from the Flash 0731 mixed IQ2/Q2/Q8 GGUF.
Target weights occupy **80.76 GiB**; use a 128 GiB system and leave room for
request state. AR and optional DSpark support prompt, chat and HTTP serving.
Target-model parity remains open; see the quality limits before relying on
AR/DSpark agreement as a quality claim.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download antirez/deepseek-v4-gguf \
  --revision 1cd7b564460821938add0475a60b942c409295e0 \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --local-dir models/deepseek-v4-flash
nix develop -c hf download antirez/deepseek-v4-gguf \
  --revision e7f04037032990db0346398d249baf9fb9df1ccc \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --local-dir models/deepseek-v4-flash
nix build
MODEL=/path/to/target.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --context 32768 --sessions 1
```

Add `--dspark-model /path/to/DSpark-support.gguf` to enable DSpark.
It verifies proposals against the target, commits only accepted prefixes and
keeps sampling/controller state private to each request. Speed depends on
acceptance and context; the support model is optional.

The server exposes OpenAI-compatible chat, streaming, tools, reasoning controls,
and RAM/disk continuation caches. See [server configuration](../../SERVER.md)
and [CLI options](../../CLI.md). Images are unsupported.

## Evidence

`artifacts/` retains controller calibration, sampling/qualification summaries and
three unresolved target-arithmetic audits. Historical speed matrices, duplicate
capability samples and discarded kernel experiments are available in Git history.
Maintained executable fixtures remain with their tests.
