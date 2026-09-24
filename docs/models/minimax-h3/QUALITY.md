# MiniMax H3 quality

**BF16 FL2VA passes the frozen component checks; full-video perceptual quality
is not qualified.** Independent teachers follow the released
[Diffusers implementation](https://github.com/huggingface/diffusers/tree/fe15005a333d1270b490e4885a6ea13b66b1092a).
The [quality contract](../../../tests/fixtures/minimax_h3/quality-contract-v1.json)
binds weights, inputs, shapes and thresholds.

| Boundary versus independent teacher | Relative L2 | Relative maximum error |
| --- | ---: | ---: |
| Prompt layers 1 / 50, six-token fixture | 0 | 0 |
| BF16 transformer block, 528 rows | 0.004610 | 0.004831 |
| Complete forward, 256×256×22 video velocity | 0.01936 | 0.04586 |
| Complete forward, 256×256×22 audio velocity | 0.01126 | 0.02162 |
| Complete forward, 512×512×22 video velocity | 0.01327 | 0.01546 |
| Complete forward, 512×512×22 audio velocity | 0.007311 | 0.01499 |
| Selected VisualVAE frames | 8.15e-7 | 2.25e-6 |
| AudioVAE waveform | 1.09e-5 | Absolute maximum 6.03e-5 |
| AudioVAE frequency magnitudes | 3.63e-6 | 5.83e-6 |

The block limit is 0.01 for both relative errors; denoiser/VAE limits are
component-specific in the contract. All comparisons require finite outputs and
identical input tensors. Equal Gufo/PyTorch seeds do not generate equal noise.

## Full-resolution kernel qualification

At **1344×768×124**, the optimized 50-block forward preserves every baseline
video/audio velocity value: **3,580,416 video + 13,248 audio FP32 values**.
This is one forward with six conditioning rows and zero initial latents,
not a full video or an independent full-resolution teacher comparison.
Full-size attention samples have FP64-relative L2 **0.002379**.
[Evidence, September 21, 2026](artifacts/full-resolution-kernels.json).

**Remaining limits:** a separate 7,136-row real-weight block has relative
maximum error **0.010417**, above the frozen small-block 0.01 ceiling;
the optimization does not add error but this stress case is not a strict pass.
[Attention evidence](artifacts/native-attention.json).
Full-video prompt adherence, temporal consistency, perceptual quality and A/V
sync remain unmeasured. `fast`/`aggressive` are approximate presets and do not
inherit the exact route's numerical qualification.

## Reproduce

Run the smallest affected oracle; complete video generation is separate,
explicitly requested work.

```sh
nix develop -c cmake --build --preset gpu-test --target minimax_h3_dit_hip_test
nix develop -c ctest --preset gpu-full -R '^minimax_h3_dit_analytic$' \
  --no-tests=error --output-on-failure
```

Real-weight [block](../../../tests/fixtures/minimax_h3/dit-block0-oracle-v1.json),
[denoiser](../../../tests/fixtures/minimax_h3/denoiser-oracle-v1.json),
[video](../../../tests/fixtures/minimax_h3/video-vae-oracle-v1.json) and
[audio](../../../tests/fixtures/minimax_h3/audio-vae-oracle-v1.json) manifests
define the external payloads; [model tools](../../../tools/README.md)
verify their hashes before comparison. Never loosen a ceiling for an optimization.

The exact route follows Diffusers' 50-point/49-forward schedule and decoder
context. Adaptations from [h3.c `8974cc0`](https://github.com/antirez/h3.c/tree/8974cc055ea9c02fcd14cc27dfda3e1027c05153)
retain Salvatore Sanfilippo's MIT attribution in
[third-party notices](../../../THIRD_PARTY_NOTICES.md); model weights retain
their separate terms in the [model guide](README.md#model-acquisition).
