# Qwen3.6 35B-A3B

Sparse Mixture-of-Experts text model (`qwen35moe`) on gfx1151. 256 routed
experts with 8 active per token, plus one always-on shared expert. Layers
alternate gated-delta-net (GDN) recurrence and full attention in a 3:1 ratio;
the attention layers use 16 query / 2 key-value heads at `head_dim` 256.
Production target is the **UD-Q8_K_XL** GGUF (37 GiB weights); request state
and KV/SSM caches need additional memory. **UD-Q6_K_XL** (30.5 GiB) is faster
for both prefill and decode, see [Benchmarks](BENCHMARKS.md). Both GGUFs carry
an MTP head (`blk.40.nextn.*`) that Gufo uses as a speculative draft; a
separate DFlash2 draft can be converted from the published checkpoint (below).

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
Flash-Next's routed F16 WMMA GEMM (gate and up paired for chunks of 1024+
tokens when the down projection takes it too), BF16 the grouped WMMA GEMM, and
anything else the per-slot GEMV. Decode runs the vendored llama.cpp-derived MMQ
vector kernels; their device context is bound at model load when the config
reports an MoE architecture.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| MTP | `--speculative mtp --mtp-model PATH` (`gufo prompt`, `gufo bench`) | Drafts with the GGUF's own MTP head; pass the same file as the target. `--draft-tokens N --min-draft-tokens N` fixes the draft length. Not available through `gufo serve` yet. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` (`gufo prompt`, `gufo bench`) | Block-diffusion draft converted to GGUF (below); it taps 8 target layers. `--draft-tokens N` caps proposals, `--draft-policy fixed` fixes them. `gufo serve` is untested for this model. |

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

### Speculative decoding (DFlash2)

No DFlash2 GGUF is published for this model. Convert
[`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)
with a llama.cpp checkout whose converter registers `DFlash2DraftModel`
(tested at `b81c99b47`); the converter takes the tokenizer from the target
checkpoint:

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

Thinking/template controls and HTTP sampling defaults are described in
[the server guide](../../SERVER.md#reasoning-controls).

## Scope

- Quantized MoE expert GEMM (Q8_0 / Q6_K) and the fused gate/up/down expert
  path; dense attention and GDN layers are shared with the other Qwen runners.
- No vision, no audio. Speculative decoding is MTP or a converted DFlash2
  draft, from `gufo prompt` and `gufo bench`; the HTTP server rejects MTP for
  this architecture (it wires MTP for Flash-Next only).
- The prefill attention kernel is compiled only for head shapes listed in
  `detail/attention_policy.hpp`; unsupported shapes fall back to the baseline
  path (see [Experiments](EXPERIMENTS.md)).
