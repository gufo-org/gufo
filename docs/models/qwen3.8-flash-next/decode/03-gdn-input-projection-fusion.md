# Track 3: fuse the small and same-input projections

**Est. gain:** +2-2.5% AR decode.
**Effort:** low. **Risk:** low — same kernels, same arithmetic, fewer launches.

## Goal

Merge launches that read the same activation row and are independent of each
other, so the decode chain has fewer serialisation points.

## Current behaviour

Per linear layer, `LinearAttention`
([`executor.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/executor.cpp),
search `Dense(l.ssm_qkv`) issues:

```cpp
Q8Input xq;
Quantize(x, n_tokens, c.hidden_size, &xq, ...);   // one shared Q8_1 row
Dense(l.ssm_qkv,        xq, s_.qkv, ...);         // 2560 -> 10240
Dense(l.ssm_gate,       xq, s_.z,   ...);         // 2560 ->  6144
Dense(l.ssm_alpha_beta, x,  s_.alpha_beta, ...);  // 2560 ->    96
```

and per attention layer, `Attention` issues `attn_qkv` (one stacked tensor,
already fused) plus:

```cpp
Dense(l.indexer_k, ...);   // 2560 ->  128
Dense(l.indexer_q, ...);   // 2560 ->  512   (sparse mode only)
```

All of these read the same activation row and none depends on another. They are
separate launches only because they are separate call sites.

`ssm_alpha_beta` is a stacked device tensor built at load
(`Stack({&l.ssm_alpha, &l.ssm_beta}, true)`,
[`device_model.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/device_model.cpp),
search `ssm_alpha_beta`), so it is already one matrix with 96 rows.

The router (`ffn_gate_inp`, 2560 -> 512, F32) is the same story one stage
later; it is covered by [track 5](05-moe-router-fusion.md) and the two should
be designed together.

## Why this is worth doing: the small projections are pure latency

The measured small-projection costs are far above their bandwidth:

| Projection | Params | Bytes (F32) | Measured | Implied GB/s |
| --- | ---: | ---: | ---: | ---: |
| Router 512x2560 | 1.31 M | 5.24 MB | 17.7 µs | 296 |
| alpha/beta 96x2560 | 0.25 M | 0.98 MB | 8.1 µs | 121 |

Both measurements are from
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) "Vector loads for F16/BF16 small
projections". The router is partly L2-resident (5.24 MB fits the 32 MB MALL),
which is why it looks fast; alpha/beta at 121 GB/s is latency, not bandwidth.
An 8 µs kernel that moves 1 MB is a launch and a ramp, nothing else.

`SmallGemmKernel`
([`kernels.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp),
search `kSmallGemmRows`) uses 4 rows per block and 128 threads, so
`alpha_beta` launches 24 blocks and the router 128 blocks. On a 40-CU part
that is 0.6 and 3.2 blocks per CU.

## Change

### 3a. One launch for the small projections (the main win)

Add a grouped small-projection kernel that takes a descriptor array:

```
{ weights, type, rows, out }  x  N
```

with `grid.y` indexing the descriptor and `grid.x` covering rows, exactly the
shape `SmallGemmKernel` already uses. Per descriptor it dispatches on type
(F32 / BF16 / F16 / Q8_0 / Q6_K) and calls the existing per-row body with the
same lane mapping, product order and reduction.

Group per layer:

- linear layers: `alpha_beta` (96 rows, F32/BF16/Q8_0)
- attention layers: `indexer_k` (128), `indexer_q` (512, sparse only)
- all layers: the router (512, F32) — see [track 5](05-moe-router-fusion.md)

Saves 1-3 launches per layer. At ~8 µs each that is **~0.5-0.7 ms, 1.5-2%**.

Design note: `SmallGemmForType` already uses `grid.y` for token groups
(`groups = tokens / 8`). Do not overload that dimension; a decode pass has one
token group, so reusing `grid.y` for descriptors is safe there but the two uses
must not be mixed. Either pass a `groups == 1` invariant or add a separate
dimension via a flattened `grid.x` with a per-block descriptor index.

### 3b. One launch for `ssm_qkv` + `ssm_gate` — already done at load

**This is already implemented**, and not by a fused kernel. `DeviceModel`
stacks same-input Q8_0 projections at load:

```cpp
if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
  d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
}
```

([`device_model.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/device_model.cpp),
search `d.ssm_in = Stack`), so the decode path issues one
`Dense(l.ssm_in, x, s_.qkvz, ...)` covering all 16,384 rows, and
`qkv = s_.qkvz; z = s_.qkvz + channels;` reads both halves of its output.

`stackable` requires both tensors to be Q8_0 with the same `cols`. When it
holds, there is nothing left to fuse here. When it does not hold (mixed
types), the two projections stay separate and a fused kernel would be worth
the ~0.5-1% — but that is not the shipped artifact.

The same stacking applies to attention layers
(`d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v})`), so the attention
Q/K/V projections are already one launch too.

`alpha_beta` is stacked separately and unconditionally
(`Stack({&l.ssm_alpha, &l.ssm_beta}, true)`), so it is one launch as well.
That leaves 3a as the only remaining item in this track.

## Traps

- **Type dispatch cost.** A per-descriptor branch is fine for 96-row matrices
  and not fine for a 10,240-row one. Keep 3a and 3b separate for that reason.
- **`Quantize` caching.** `Dense(w, x, ...)` calls `Quantize` internally and
  reuses the cached Q8_1 rows via `ReadyQ8`. A fused kernel must take the
  already-quantized `xq` explicitly, or it will quantize three times.
- **Scratch aliasing.** `s_.qkv` and `s_.z` are adjacent regions of `s_.qkvz`
  (`qkv = s_.qkvz; z = s_.qkvz + channels;`). A fused kernel writing both must
  keep them disjoint; verify the strides, not just the base pointers.
- **Graph keys.** The decode graph key includes `n`, `n_logits`, `speculative`,
  `sparse` and `logits != nullptr`. A fused launch changes the node set for
  every key, so re-warm all of them; do not assume the previously captured
  graphs are still valid.

## Validation

Correctness (byte-identical):

- `projection_ops` test, including the 1/7/9/95/96-row passes at the head,
  middle and tail of a 1,619- or 1,549-row pass.
- Fixed-history logit dumps at widths 1-8 against the F32 reference fixture.
- `gdn_ops` for the `ssm_qkv`/`ssm_gate` fusion.
- Greedy output hashes and draft counts at d0/4096/32768, C1-C8; sparse mode
  on and off so `indexer_q` is covered both ways.

Timing:

- `prof.py` before/after: the `ssm` and `dense: small projections` stages
  should both shrink, and the total node count should drop by the expected
  amount.
- Unprofiled `gufo bench` at d0 and d4096.

## Decision criteria

Keep if byte-identical and the unprofiled tg at d0 improves. 3a is the safer
half; if 3b is flat, keep 3a and drop 3b rather than carrying a fused kernel
that does not pay.
