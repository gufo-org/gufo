# Qwen3.6 35B-A3B

Sparse Mixture-of-Experts text model (`qwen35moe`) on gfx1151. 256 routed
experts with 8 active per token, plus one always-on shared expert. Layers
alternate gated-delta-net (GDN) recurrence and full attention in a 3:1 ratio;
the attention layers use 16 query / 2 key-value heads at `head_dim` 256.
Production target is the **UD-Q8_K_XL** GGUF (37 GiB weights); request state
and KV/SSM caches need additional memory. **UD-Q6_K_XL** (30.5 GiB) is faster
for both prefill and decode, see [Benchmarks](BENCHMARKS.md). Both GGUFs carry
an MTP head (`blk.40.nextn.*`) that Gufo uses as a speculative draft; a
separate DFlash2 draft is published as a GGUF or can be converted (below).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix build
MODEL=path/to/Qwen3.6-35B-A3B-MTP-UD-Q8_K_XL.gguf
./result/bin/gufo serve llm --model "$MODEL" --sessions 1 --context 32768
./result/bin/gufo prompt --model "$MODEL" --prompt "Explain virtual memory." --temperature 0
```

Text-only model; there is no matching `mmproj` projector in scope. In prefill,
each routed-expert projection picks its route by format: Q8_0 / Q6_K take
Flash-Next's routed F16 WMMA GEMM, BF16 the grouped WMMA GEMM, and anything
else the per-slot GEMV. Chunks of 1024+ tokens run gate and up as one paired
launch when the down projection takes the routed GEMM too. Decode runs the
vendored llama.cpp-derived MMQ vector kernels; their device context is bound
at model load when the config reports an MoE architecture.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| MTP | `--speculative mtp --mtp-model PATH` (`gufo prompt`, `gufo bench`, `gufo serve`) | Drafts with the GGUF's own MTP head; pass the same file as the target. `--draft-tokens N` caps the draft, `--min-draft-tokens N` sets its floor. In `gufo serve`, `--mtp-prime-chunk N` sets the rows per batched prompt-prime slice (default 1024; 0 restores the per-token prime). |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` (`gufo prompt`, `gufo bench`, `gufo serve`) | Block-diffusion draft converted to GGUF (below); it taps 8 target layers. `--draft-tokens N` caps proposals, `--draft-policy fixed` fixes them. |

### Speculative decoding (MTP)

```sh
./result/bin/gufo prompt --model "$MODEL" --speculative mtp \
  --mtp-model "$MODEL" --draft-tokens 2 --min-draft-tokens 2 \
  --prompt "Explain virtual memory." --temperature 0
```

In the September 26, 2026 sweep n=2 was fastest for both quants: tg128
68.39 t/s on UD-Q6_K_XL and 65.01 t/s on UD-Q8_K_XL, versus 54.02 and 51.59
without MTP (+26%). An earlier UD-Q8_K_XL run favoured n=1, and acceptance
depends on the generated text, so re-check n on your own workload.

As in Qwen3.8 Flash-Next, each MTP draft token scores only the first 98,304
output rows plus a per-request table of 1,024 output rows (about 2–4 MB per
session, depending on the head's quant): the special tokens and the request's
own rarer tokens. A request that keeps bringing new rare tokens drafts over the
full head. Only proposals change; the target still verifies over the full
vocabulary.

### Speculative decoding (DFlash2)

A Q8_0 conversion of
[`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)
is published as
[`aminya/Qwen3.6-35B-A3B-DFlash2-GGUF`](https://huggingface.co/aminya/Qwen3.6-35B-A3B-DFlash2-GGUF)
(`Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf`). The numbers below were measured with a
local conversion of the same checkpoint, made with a llama.cpp checkout whose
converter registers `DFlash2DraftModel` (tested at `b81c99b47`); the converter
takes the tokenizer from the target checkpoint:

```sh
nix develop -c hf download incoai/Qwen3.6-35B-A3B-DFlash2 --local-dir draft
nix develop -c hf download Qwen/Qwen3.6-35B-A3B config.json tokenizer.json \
  tokenizer_config.json vocab.json merges.txt --local-dir target
python3 llama.cpp/convert_hf_to_gguf.py draft --target-model-dir target \
  --outtype q8_0 --outfile Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
./result/bin/gufo prompt --model "$MODEL" --speculative dflash2 \
  --dflash-model Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --draft-tokens 7 \
  --prompt "Explain virtual memory." --temperature 0
```

Run the converter with numpy 2. With numpy 1.26 on Python 3.14, Q8_0
conversion drops the sign of every quant in tensors with rows of 4,096 values
or more (`attn_output`, `ffn_down`, `fc`). The draft still loads, but accepts
no drafts.

On UD-Q6_K_XL, 512 greedy tokens with thinking off, a prose explanation ran at
73.5 tok/s with `--draft-tokens 3 --draft-policy fixed` and a code rewrite at
89.7 tok/s with 7, versus 52 tok/s without a draft; the best MTP setting
reached 63.6 and 68.7. The adaptive policy (66.6 / 88.8) does not find the
shorter draft on prose, so pick the length for the workload. `gufo bench`
understates DFlash2 because its synthetic prompt drafts poorly.

Greedy speculation on this model is not bit-exact against AR: MTP and DFlash2
agree with each other, but verifying a batch of tokens takes a different
MoE numeric route than single-token decode, so near-tied tokens can flip late
in a long answer. The Qwen3.8 27B keeps that contract; this model does not yet.

### Serving

```sh
./result/bin/gufo serve --sessions 4 llm --model "$MODEL" --context 262144 \
  --speculative dflash2 --dflash-model Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf \
  --draft-tokens 7 --draft-policy adaptive \
  --prefill-chunk 2048 --prefill-decode-rounds 8
```

For MTP, replace the draft arguments with
`--speculative mtp --mtp-model "$MODEL" --draft-tokens 2`.

Concurrent requests share one batched verification step. Each extra row reads
mostly different experts, so a step costs close to one decode step per
stacked row; the server therefore shortens DFlash2 drafts as more requests are
active (2 draft tokens at 2 requests, 1 at 3 or more) and a lone request keeps
the full draft. MTP takes the same batched step, with the sessions' drafts
proposed in one batch, and keeps 1 draft token at 2 requests and none at 3 or
more, where one draft token no longer pays for its row; there it decodes within
3% of the plain rate, the cost of keeping each session's draft context current.

Prefill is not batched across sessions: requests take turns, and while other
sessions decode, a prompt is prefilled in `--prefill-chunk` token slices.
`--prefill-decode-rounds N` gives the decoding sessions N rounds between two
slices, so a larger chunk does not lengthen their stalls in proportion. Each
session holds its own KV/SSM state: at context 262144, one session used
38.7 GiB of device memory in total and four 59.7 GiB. On this hardware DFlash2
decodes faster than MTP at most depths and under concurrency, while MTP
prefills short and medium prompts slightly faster; see
[Benchmarks](BENCHMARKS.md#serving).

Thinking/template controls and HTTP sampling defaults are described in
[the server guide](../../SERVER.md#reasoning-controls).

## Scope

- Quantized MoE expert GEMM (Q8_0 / Q6_K) and the fused gate/up/down expert
  path; dense attention and GDN layers are shared with the other Qwen runners.
- No vision, no audio. Speculative decoding is MTP (the GGUF's own head) or a
  converted DFlash2 draft; both run in `gufo prompt`, `gufo bench` and
  `gufo serve`.
- The prefill attention kernel is compiled only for head shapes listed in
  `detail/attention_policy.hpp`; unsupported shapes fall back to the baseline
  path (see [Experiments](EXPERIMENTS.md)).
