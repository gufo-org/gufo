# d1 implementation decisions

| Decision | Evidence / outcome |
| --- | --- |
| Reuse the LFM3 backbone with complete bidirectional prompts | Independent HTTP probabilities and FP64 attention checks pass; convolution and sliding attention remain causal. |
| Compile a separate bidirectional attention specialization | Keeps the causal specialization's mask and arithmetic; existing LFM3 operator checks pass. |
| Reuse the LFM3-Audio input encoder without requiring generation tensors | Official d1 mmproj has audio input and vision, but no depth decoder/codec. Audio and spoken-instruction reference checks pass. Voice creation explicitly rejects this input-only projector. |
| Add native vision convolution, axial RoPE and spatial merging | Image and mixed-media reference checks pass. No Python/reference runtime dependency. |
| Packed sets of 16 by default | Matches LiquidAI's recorded training mode; exact layout goldens and FP32 teacher logits pass, including a 17-question partial tail. |
| Sparse output head for packed/box slots | Projects only the union of vocabulary entries read by the answer slots; all selected logits agree with the independent teacher within `0.000063`. |
| Preserve the original single-slot primitive path | A compact replacement shifted legacy probabilities by up to `1.3e-7`. Retaining the full-vocabulary route restores exact `set_size=1` responses without weakening the compatibility check. |
| Calibrated single/multiple boxes | Median thousandth edges, presence tables, fixed Halton confidence and reading order match the pinned Python reference; 1/4/16 blocks and absence pass. |
| Files before images | Initial image-first ordering failed the mixed-media probability gate; reference ordering passes without changing tolerance. |

Current measurements are in [BENCHMARKS.md](BENCHMARKS.md); numerical and API
limits are in [QUALITY.md](QUALITY.md).
