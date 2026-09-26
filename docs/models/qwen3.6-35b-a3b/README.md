# Qwen3.6 35B-A3B

Sparse Mixture-of-Experts text model (`qwen35moe`) on gfx1151. 256 routed
experts with 8 active per token, plus one always-on shared expert. Layers
alternate gated-delta-net (GDN) recurrence and full attention in a 3:1 ratio;
the attention layers use 16 query / 2 key-value heads at `head_dim` 256.
Production target is the **UD-Q8_K_XL** GGUF (37 GiB weights); request state
and KV/SSM caches need additional memory. The GGUF also carries an MTP head
(`blk.64.nextn.*`) that Gufo does not load or use.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix build
MODEL=path/to/Qwen3.6-35B-A3B-MTP-UD-Q8_K_XL.gguf
./result/bin/gufo serve llm --model "$MODEL" --sessions 1 --context 32768
./result/bin/gufo prompt --model "$MODEL" --prompt "Explain virtual memory." --temperature 0
```

Text-only model; there is no matching `mmproj` projector in scope. Routed-expert
GEMMs run on the vendored llama.cpp-derived MMQ kernels; their device context is
bound at model load when the config reports an MoE architecture.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |

Thinking/template controls and HTTP sampling defaults are described in
[the server guide](../../SERVER.md#reasoning-controls).

## Scope

- Quantized MoE expert GEMM (Q8_0 / Q6_K) and the fused gate/up/down expert
  path; dense attention and GDN layers are shared with the other Qwen runners.
- No vision, no audio, no MTP/speculative decoding for this architecture yet.
- The prefill attention kernel is compiled only for head shapes listed in
  `detail/attention_policy.hpp`; unsupported shapes fall back to the baseline
  path (see [Experiments](EXPERIMENTS.md)).
