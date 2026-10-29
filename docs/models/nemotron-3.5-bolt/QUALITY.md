# Nemotron 3.5 Bolt 74B-A9B quality

There is no capability score yet. This page records numerical checks of the
float32 CPU reference (`src/models/nemotron35_bolt/reference.cpp`), the ROCm
runtime and the LiLiCorr drafter against llama.cpp on the pinned GGUFs, and
of the tokenizer and chat template against Hugging Face.

## ROCm runtime versus llama.cpp

Same prompts and token IDs as the CPU reference comparison below. "Prefill"
runs each prompt in one pass (FP16 WMMA Mamba projections, BF16 WMMA
routed experts on exact IQ2_XXS weights, MMQ elsewhere); "decode" appends
it one token at a time through the captured decode graph
(`nemotron35_bolt_runtime_probe --step`: IQ2_XXS vector kernel on BF16
activations, MMVQ on Q8_1). Both return every position's logits. Measured
2026-10-29.

| Prompt | Path | Pair | Top-1 agreement | KL median | KL p90 | KL max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Short chat, 23 tokens | prefill | Gufo ROCm vs llama.cpp Vulkan | 23/23 | 4.0e-4 | 2.4e-3 | 1.1e-2 |
| | prefill | Gufo ROCm vs llama.cpp CPU | 23/23 | 4.1e-4 | 5.5e-3 | 4.7e-2 |
| | decode | Gufo ROCm vs llama.cpp Vulkan | 22/23 | 3.8e-4 | 4.0e-3 | 5.4e-2 |
| | decode | Gufo ROCm vs llama.cpp CPU | 22/23 | 5.8e-4 | 2.8e-3 | 1.3e-2 |
| Chat with code and digits, 146 tokens | prefill | Gufo ROCm vs llama.cpp Vulkan | 144/146 | 1.0e-4 | 2.5e-3 | 5.3e-2 |
| | prefill | Gufo ROCm vs llama.cpp CPU | 146/146 | 8.6e-5 | 1.7e-3 | 3.0e-2 |
| | decode | Gufo ROCm vs llama.cpp Vulkan | 145/146 | 1.0e-4 | 2.2e-3 | 2.0e-2 |
| | decode | Gufo ROCm vs llama.cpp CPU | 145/146 | 8.0e-5 | 2.4e-3 | 3.3e-2 |

The disagreements are near ties that also split llama.cpp's own backends
(table below): position 10 of the long prompt, in the system turn, where
Vulkan puts 0.23 on its top token; position 111, a digit of fib(90), where
Vulkan puts 0.51 on one digit and CPU 0.62 on another; and the short
prompt's position 18 (p = 0.39). `nemotron35_bolt.runtime` checks that
decoding a prompt one token at a time through the captured decode graph
matches one bulk prefill (KL 7.4e-5, same top-1), that a restored snapshot
reproduces a greedy continuation bit for bit, and that invalidation restarts
the session. Greedy decoding of the short prompt answers "Paris" after a
short reasoning block.

The vendored products quantize activations to Q8_1 like llama.cpp;
`nemotron35_bolt.mmq_ops` compares them with float64 products of the
dequantized weights (relative RMS error 0.005, correlation ≥ 0.99998).
`nemotron35_bolt.gemv_ops` compares the model-private IQ2_XXS vector kernel
and the NVFP4 WMMA product with float64 products over the activations as
they round them (BF16, FP16): every output within 2e-6 of the summed
magnitudes, observed ≤ 1.1e-6, over partial tiles, 1–8 tokens, routed slots
and the fused ReLU². The routed IQ2_XXS prefill product, on exact weights and
BF16 activations, stays within 1e-7 over experts spanning several tiles,
partial tiles and per-slot inputs, and gives equal rows equal outputs in
any tile column.
`nemotron35_bolt.kernel_ops` checks the Mamba-2 and routing kernels against
direct CPU evaluations (relative error ≤ 3e-7, exact expert IDs), that the
FP16 outputs of RMSNorm and the gated norm are their F32 results rounded to
nearest even, bit for bit, and the
attention against an exact softmax (relative error 1.9e-4 chunked, 2.4e-4
split; bound 1e-3) and against an oracle with the kernel's FP16 rounding
points (7.9e-5; FP16 probabilities can round differently where the GPU's
exp2 differs in the last bit).

## LiLiCorr drafter versus llama.cpp

2026-10-29, llama.cpp `c3df4b7e2` Vulkan with an F32 drafter KV cache.
`tools/llama_parity/llama_dflash.cpp` decodes a 280-token sequence (a chat
prompt and the trunk's own 240-token greedy answer) with the trunk, injects
the extracted layer inputs into the drafter as `common/speculative.cpp`
does, and drafts one block at each of 20 anchors (every 12th position from
40). `nemotron35_bolt_drafter_probe` injects the same llama.cpp features into
Gufo's drafter and drafts at the same anchors, so the rows compare the
drafter alone.

| Check | Result |
| --- | --- |
| Drafted slot logits, 140 rows | top-1 140/140; KL median 5.8e-6, p90 1.8e-4, max 5.4e-4 |
| Slot candidate lists (top 8 by drafter logit) | 110/140 in the same order, 131/140 the same set |
| Greedy lattice walks (7 tokens per anchor) | 20/20 identical |
| Lattice scores where both lists agree (scale 8) | median difference 0.018, p90 0.053 |

The candidate differences are rank swaps and replacements at the bottom of
the top 8, where drafter logits nearly tie; every chosen path agrees.
`nemotron35_bolt.drafter_ops` checks the drafter kernels against direct
double evaluations (relative error ≤ 1e-5; RoPE ≤ 5e-4, its angle is float
as in ggml).

`nemotron35_bolt.runtime` (with the drafter) checks the verification path:
a 4-token block committed in part leaves the session bit-identical to
appending the kept 2 tokens, a discarded verification leaves no trace, a
restored snapshot drafts exactly as the original session, and greedy
draft-and-verify rounds reproduce plain greedy decoding (48 of 48 tokens on
its prompt; 240 of 240 on the Fibonacci answer above). Each verified row is
bit-identical to the row of decoding its token alone:
`nemotron35_bolt.kernel_ops` compares verified blocks of attention with
one-token calls at their positions, also across 4,096 and 8,192 keys. The
block's later tokens are already in the KV cache, and the WMMA product
rounds differently beside their values even with zero probabilities, so the
kernel zeroes them per row; and each split takes every 64th key tile, so a
tile's split does not depend on the number of keys. Before that, greedy
LiLiCorr output over HTTP left plain decoding at near ties (2 of 8 depths of
the mixed benchmark workload).

## Sampling from device heads

Random sampling (a temperature with top-k or top-p, no penalties) draws
from a head the GPU builds for each returned logits row: the largest finite
logit, the softmax sum over the row in F64, and every token within a
1/16-nat bin of the cut the filters need, at most 1,024 tokens.
`SamplerState::Sample(logits, head)` sorts and filters the head with the
same arithmetic as the full row and falls back to the row when the head
does not cover top-k or the top-p mass, or was summed at another
temperature. Only the softmax sum's addition order differs from the host's
sequential sum, which moves the top-p cut by about 1e-16 relative.

| Check | Result |
| --- | --- |
| `sampling_test`: 8 configurations × 300 rows of 4,096 tokens (sharp, moderate, flat; ties, NaN, ±inf; heads of 1, 64 and all tokens; pending draws), host heads with the exact sum | same token and RNG state for every draw; covering heads are read, other temperatures ignored |
| `nemotron35_bolt.kernel_ops`: device heads of 8 rows × 4 configurations at vocabularies 131,072 and 5,000 | maximum exact, softmax sum within 1e-12, every token above the head's floor, coverage, same token and RNG for 32 seeds |
| `nemotron35_bolt.runtime`: real rows of a verified block and a decode step | heads on 9/9 rows at T 1.0, top-p 0.95 and 6/9 at T 0.6, top-k 20 (a 20th token deeper than the 32-nat bins falls back); same token and RNG for 16 seeds |
| HTTP, a 33K-token code prompt and a 31K-token documentation prompt, seed 1, T 1.0, top-p 0.95, 1,024 tokens, with and without the drafter | byte-identical answers from device heads and from full rows on the host |

## CPU reference versus llama.cpp

2026-10-29, llama.cpp `c3df4b7e2` (build 11969), F32 KV cache in llama.cpp.
Both sides decode the same token IDs, produced by llama.cpp's tokenizer with
special tokens parsed. "Vulkan" offloads every layer; "CPU" runs llama.cpp's
CPU backend with no layer or operation offload. Rows compare next-token
distributions at every prompt position.

| Prompt | Pair | Top-1 agreement | KL median | KL p90 | KL max |
| --- | --- | ---: | ---: | ---: | ---: |
| Short chat, 23 tokens | llama.cpp Vulkan vs CPU | 23/23 | 6.7e-4 | 4.3e-3 | 4.2e-2 |
| | Gufo reference vs Vulkan | 23/23 | 2.8e-4 | 3.0e-3 | 6.1e-3 |
| | Gufo reference vs CPU | 23/23 | 4.3e-4 | 5.5e-3 | 4.1e-2 |
| Chat with code and digits, 146 tokens | llama.cpp Vulkan vs CPU | 144/146 | 1.0e-4 | 2.0e-3 | 4.1e-2 |
| | Gufo reference vs Vulkan | 145/146 | 9.8e-5 | 2.2e-3 | 5.2e-2 |
| | Gufo reference vs CPU | 145/146 | 6.8e-5 | 1.6e-3 | 1.2e-1 |

The reference agrees with each llama.cpp backend as closely as the two
backends agree with each other. The largest divergences fall on the near
ties above: at the fib(90) digit (position 111) the reference sides with
Vulkan, the ROCm prefill with CPU. llama.cpp quantizes activations for its
quantized matmuls; the reference keeps them in float32 and accumulates in
float64.

Gaps: no BF16 or Transformers comparison (the BF16 checkpoint does not fit in
memory); prompts up to 146 tokens only; vision is not checked.

## Tokenizer and chat template

Hugging Face goldens from the BF16 checkpoint's `tokenizer.json` and
`chat_template.jinja` (revision `e09a4ced`), regenerated by the scripts in
`tools/models/nemotron35_bolt/`:

| Check | Fixture | Result |
| --- | --- | --- |
| Pre-tokenizer pieces (`pixtral` split with Unicode categories) | 26 texts: scripts, combining marks, emoji, digits, code, CR/LF and trailing whitespace, added tokens | identical byte ranges |
| Token IDs, control tokens parsed and as text | same 26 texts | identical; `<think>`/`<tool_call>` stay added tokens either way |
| Token IDs from llama.cpp `c3df4b7e2` | same 26 texts | identical |
| Rendered chat bytes (Jinja2 with transformers' settings) | 9 conversations: thinking on/off, efficient mode, truncated and preserved history reasoning, tools with nested schemas, tool calls with typed arguments, consecutive tool results | identical |
| Rendered prompt token IDs | same 9 conversations | identical |

The renderer reads JSON numbers as doubles, so a schema value written as an
integral float (`5.0`) renders as `5`, where Python renders `5.0`.

## Benchmark method

All tables in [BENCHMARKS.md](BENCHMARKS.md) were measured on October 30,
2026 with `tools/bench/model-bench.py` and
[artifacts/bench.json](artifacts/bench.json): HTTP on both sides, greedy,
seed 1, thinking off, one warmed sample per point. Gufo is a clean
`cmake --preset release` build with ROCm 10.1 (the Nix build would compile
ROCm from source on this host); llama.cpp is `c3df4b7e2` on Vulkan, the
reference the port is qualified against.
[Model identities](artifacts/model-identities.json) record the commits,
binary hashes and GGUF checksums. Every step ran alone on the host (a lock
rejects runs with foreign CPU or GPU load) and started after the APU cooled
below 55 °C (GPU) and 75 °C (CPU). Under load the package power averaged
120–138 W, the GPU ran at 2.6–2.8 GHz and reached up to 98 °C.

Single-user rows continue a cached conversation prefix of 0 to 131,072
tokens with a turn of about 2,048 tokens and generate 128 tokens, at a
capacity of 133,760 on both engines. Both draft up to 7 tokens per round
with NVIDIA's LiLiCorr drafter (`--spec-type draft-dflash
--spec-draft-n-max 7` for llama.cpp). Gufo's LiLiCorr completions equal its
autoregressive completions at every depth and concurrency (completion
hashes), and its autoregressive completions equal llama.cpp's at depths 0
and 131,072. llama.cpp's `draft-dflash` completions differ from its own
autoregressive ones at every depth and at the prepared C1 control, so its
speculative cells are N/A; it measured 19–30 tok/s (mixed) and 25–38 tok/s
(repetitive) single-user, and 28–43 and 35–59 tok/s summed over 1–8 users
([unavailable.json](artifacts/unavailable.json), artifacts).

Concurrency uses the single-user d0 prompts at a capacity of 4,096 per user.
Every session is prepared before the timed cohort; Gufo prepares from a C1
prompt checkpoint (`gufo.checkpoint_preparation`), because a second cold
prefill beside an active decode runs in 512-token scheduler chunks, which
changes Nemotron's prefill arithmetic. Gufo's C2–C8 completions equal its
C1 autoregressive ones in both modes. Loading starts from model files
evicted with `POSIX_FADV_DONTNEED` and verified absent with `mincore`
(target and drafter; runtime libraries may stay warm), at C1, capacity
262,144, LiLiCorr. Memory is the peak device-global HIP allocation; the
Vulkan reference allocates outside HIP, so its cells are N/A.

## Reproduce

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target nemotron35_bolt_reference_probe
MODEL=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
# llama_logits: see tools/llama_parity/README.md
TOKENS=$(llama_logits "$MODEL" --special --text "$PROMPT")
llama_logits "$MODEL" --tokens "$TOKENS" --dump llama.bin --f32-kv   # Vulkan
llama_logits "$MODEL" --tokens "$TOKENS" --dump cpu.bin --f32-kv --cpu
build/cpu-test/tests/models/nemotron35_bolt/nemotron35_bolt_reference_probe \
  --model "$MODEL" --tokens "$TOKENS" --dump gufo.bin
python3 tools/llama_parity/compare_logits.py llama.bin gufo.bin --vocab 131072
# ROCm runtime, prefill and token-by-token decode
build/gpu-test/tests/models/nemotron35_bolt/nemotron35_bolt_runtime_probe \
  --model "$MODEL" --tokens "$TOKENS" --dump rocm.bin [--step]
```

The drafter comparison runs on the Fibonacci request (a user turn "Write a
Python function that returns the n-th Fibonacci number iteratively, with a
docstring and two doctests." after an empty system turn, then `<think>`)
followed by the trunk's 240-token greedy answer, which
`nemotron35_bolt_runtime_probe --text "$PROMPT" --generate 240` prints:

```sh
DRAFT=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
TOKENS="$(llama_logits "$MODEL" --special --text "$PROMPT"),$ANSWER_IDS"
# llama_dflash: see tools/llama_parity/README.md
llama_dflash "$MODEL" "$DRAFT" --tokens "$TOKENS" \
  --anchors 40,52,64,76,88,100,112,124,136,148,160,172,184,196,208,220,232,244,256,268 \
  --dump llama_dflash.bin --f32-kv
cmake --build --preset gpu-test --target nemotron35_bolt_drafter_probe
build/gpu-test/tests/models/nemotron35_bolt/nemotron35_bolt_drafter_probe \
  --model "$MODEL" --drafter "$DRAFT" --dump llama_dflash.bin
```

The short prompt is a single user turn, "What is the capital of France?
Answer in one word.", followed by an open assistant turn starting with
`<think>`. The long prompt has a system message, a request for a Fibonacci
function, an assistant answer with a Python block, its O(n) cost and the
value of fib(90), and a follow-up user turn "And recursively?", ending at the
next assistant header. The reference needs about 1.4 s per token on 32
threads.
