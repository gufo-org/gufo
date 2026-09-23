# <Model name> on Gufo

<!--
Template for docs/models/<model>/BENCHMARKS.md of a GGUF language model.
Copy it, replace every <placeholder>, delete the tables the model does not
have (speculative ones for a model without a drafter, image encoder for a
text-only model), and run `model-bench.py render` to fill the marked blocks.
Keep the prose to what the tables and charts cannot show: provenance, the
timed scope, why a cell is TODO or n/a, and findings a reader would
otherwise have to derive. Never restate numbers that sit in the table above.
-->

| | |
| --- | --- |
| Host | Linux x86-64, AMD `gfx1151`, <RAM> GB unified memory |
| Gufo | `nix build` at revision `<revision>`, binary SHA-256 `<prefix>…` |
| Target | `<repo>` revision `<revision>`, **<quantization>** (<size>) |
| Speculative | <drafter file and controller, or "none"> |
| Reference | llama.cpp `llama-server` release `<bNNNNN>` (`<version string>`), ROCm gfx1151 from `flake.nix`, same GGUF |
| Method | HTTP on both servers, same prompts and timed scope, greedy, thinking off (except the thinking table), one sample per point, fresh server per table and concurrency level; `tools/bench/model-bench.py`, <date>. llama.cpp runs `--cache-ram 0` since the driver never reuses prompts. Every artifact records its server command |
| Gain | Gufo over llama.cpp, positive when Gufo is better |
| Identities | `artifacts/model-identities.json` (link after copying the template) |
| Layout | `../../../.agents/skills/benchmark-model/SKILL.md` (link after copying) |

<Model-specific caveats: chat template and default reasoning mode, image
support, which parity claims are unqualified, which correctness suite ran.>
Unmeasured cells are **TODO**; cells this host cannot produce for the
reference (a hardware limit, not a missing feature) are **n/a**, with the
reason next to the table.

## Loading

Cold-file-cache launch to `/ready`. <What each server loaded, and at which
context capacity.>

<!-- bench:loading -->
<!-- /bench -->

<Snapshot size and restore time, warm readiness for comparison.>

## Single user, autoregressive

Standard sweep: **pp2048 / tg128**, C1, greedy. Cells are tok/s from the
servers' own `timings` (`prompt_n` / `prompt_ms`, `predicted_n` /
`predicted_ms`). Depth is a cached conversation prefix: the driver sends a
user turn of about `d` tokens answered with an 8-token generated reply, then
the timed request continues that conversation with a new ~2048-token user
turn that asks for a long natural-prose answer, and 128 output tokens.
<Context capacity per server.> Artifacts:
`artifacts/single-ar-{gufo,reference}.json`.

<!-- bench:single-ar -->
<!-- /bench -->

<Only what the table does not show: a regime change such as the
prompt-cache-miss first turn, or an unexplained dip.>

## Single user, <SPEC>

Same workload with <Gufo speculative flags>. `accepted/step` is the mean
number of accepted draft tokens per verification step
(`draft_tokens_accepted / (completion_tokens − draft_tokens_accepted)`);
tokens per step is this plus one. It is proportional to the speedup and
independent of how many tokens each controller proposes, unlike an
acceptance rate. The measured turn asks for a detailed summary and a story,
so the generated text is ordinary prose.

<!-- bench:single-<spec> -->
<!-- /bench -->

Repetitive workload: same prefixes and depths, but the measured turn asks
the model to repeat the passage word for word, so the output is fully
predictable — the single-user analogue of the `repetition` corpus below.

<!-- bench:single-<spec>-repetition -->
<!-- /bench -->

Thinking workload: the model's own template decides (thinking on), and the
measured turn asks a question that requires reasoning, so the timed tokens
are chain-of-thought rather than prose. Everything else matches the tables
above.

<!-- bench:single-<spec>-thinking -->
<!-- /bench -->

<Provenance of the reference speculative build when it is not the release
pin, the cost of speculation in prefill, and where each workload changes the
picture.>

## Multiple users

Sum of individual request decode rates, averaged across measured cohorts.
Context 4096 per user, greedy, thinking
off, 128 output tokens, `cache_prompt=false`, one warm-up round, fresh
server per point (`gufo serve --sessions C`; `llama-server -np C -c 4096·C`).
Workloads come from the
`../qwen3.8-27b/artifacts/speculative-corpus.json` (link after copying): `repetition`
runs `repetition_word`; `mixed` cycles distinct requests through
<case ids>. `Exact` counts llama.cpp AR completions whose hash matches the
Gufo AR C1 reference.

<!-- bench:multi-repetition -->
<!-- /bench -->

<!-- bench:multi-mixed -->
<!-- /bench -->

<Cache-hit outcome, what `Exact` means for these completions, and why any
cell is n/a.>

## Memory

Peak device-global HIP memory in use (`hipMemGetInfo` total − free, the
counter Gufo's loader logs as `gpu_device_used_mib`), sampled every 250 ms by
the driver; idle baseline <x> GiB before either server started. C1, context
capacity <N> on both servers, both autoregressive, no draft and no projector
loaded.

<!-- bench:memory -->
<!-- /bench -->

<How each server's footprint grows with the prefix.>

## Image encoder

Prefill time (`prompt_ms`) of a chat request carrying one gradient PNG and a
one-line text turn, `max_tokens` 1, `cache_prompt=false`, one warm-up then
three timed samples per size (mean ± sd), both servers loading the same BF16
projector. The time covers the projector encode plus the prefill of the
image and text tokens; the encode alone is not separable over HTTP.

<!-- bench:image-encoder -->
<!-- /bench -->

## Reproduction

```sh
nix build
FILES="--gguf /path/to/<target>.gguf --<role> /path/to/<sidecar>.gguf"
BENCH="nix develop -c python3 tools/bench/model-bench.py --model <model>"
$BENCH tables
$BENCH $FILES run --target gufo --fresh
$BENCH $FILES run --target reference --fresh
$BENCH render
# Loading needs --drop-caches "<privileged command>".
# Split the reference sweep with --depths/--context when it cannot hold the
# deepest capacity, and --mode ar when it cannot load the sidecar.
```

<Open work: tables still TODO and why, qualification gaps, upstream pull
requests the reference columns depend on.>
