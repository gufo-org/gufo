# d1 implementation decisions

| Decision | Evidence / outcome |
| --- | --- |
| Reuse the LFM3 backbone with complete bidirectional prompts | Independent HTTP probabilities and FP64 attention checks pass; convolution and sliding attention remain causal. |
| Compile a separate bidirectional attention specialization | Keeps the causal specialization's mask and arithmetic; existing LFM3 operator checks pass. |
| Reuse the LFM3-Audio input encoder without requiring generation tensors | Official d1 mmproj has audio input and vision, but no depth decoder/codec. Audio and spoken-instruction reference checks pass. Voice creation explicitly rejects this input-only projector. |
| Add native vision convolution, axial RoPE and spatial merging | Image and mixed-media reference checks pass. No Python/reference runtime dependency. |
| One question per prompt | Matches the official GGUF path. Packed set masks and typed extraction are deferred. Repeated state work is visible in per-request timings. |
| Files before images | Initial image-first ordering failed the mixed-media probability gate; reference ordering passes without changing tolerance. |

Current measurements are in [BENCHMARKS.md](BENCHMARKS.md); numerical and API
limits are in [QUALITY.md](QUALITY.md).
