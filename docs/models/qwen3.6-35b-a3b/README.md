# Qwen3.6 35B-A3B

Sparse Mixture-of-Experts text model (`qwen35moe`) on gfx1151. 256 routed
experts with 8 active per token, plus one always-on shared expert. Layers
alternate gated-delta-net (GDN) recurrence and full attention in a 3:1 ratio;
the attention layers use 16 query / 2 key-value heads at `head_dim` 256.
Production target is the **UD-Q8_K_XL** GGUF (37 GiB weights); request state
and KV/SSM caches need additional memory. **UD-Q6_K_XL** (30.5 GiB) is faster
for both prefill and decode, see [Benchmarks](BENCHMARKS.md). Both GGUFs carry
an MTP head (`blk.40.nextn.*`) that Gufo uses as a speculative draft.

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

Thinking/template controls and HTTP sampling defaults are described in
[the server guide](../../SERVER.md#reasoning-controls).

## Scope

- Quantized MoE expert GEMM (Q8_0 / Q6_K) and the fused gate/up/down expert
  path; dense attention and GDN layers are shared with the other Qwen runners.
- No vision, no audio. Speculative decoding is MTP only (no DFlash2 draft),
  from `gufo prompt` and `gufo bench`; the HTTP server rejects MTP for this
  architecture (it wires MTP for Flash-Next only).
- The prefill attention kernel is compiled only for head shapes listed in
  `detail/attention_policy.hpp`; unsupported shapes fall back to the baseline
  path (see [Experiments](EXPERIMENTS.md)).
