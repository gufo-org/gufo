# YuE2 quality

`yue2_hip_oracle_test` compares each stage with the upstream CPU BF16 path
from fixed inputs (`src/models/yue2/reference/run_official.py` writes the fixtures; set
`YUE2_ORACLE_DIR`, `YUE2_MODEL_DIR`, `YUE2_VAE_DIR`). As of 2026-11-14 on
gfx1151, with the `examples/song.json` request (715-token prefix, 300 codec
frames):

| Stage | Result |
| --- | --- |
| Tokenizer, 8 fixtures | identical ids |
| Plan and semantic prefixes | identical |
| Teacher-forced logits, 26 rows over the full vocabulary | worst cosine 0.99997, argmax 26/26 |
| Velocity at t = 1 | cosine 0.99994, relative error 1.1% |
| Latents after 32 midpoint steps | cosine 0.99994, relative error 1.1% |
| Audio from the upstream latents | relative error 6.4e-7 |

On 2026-11-15, after the NAR GEMMs moved to hipBLASLt, every stage still
passes its gate (cosine > 0.999, audio relative error < 1e-3).

Later on 2026-11-15 the fixtures were regenerated from upstream `main`
1116bea with torch 2.14.1 on the CPU (725-token prefix: upstream planned 595
score tokens, then 300 codec frames), and the solver switched to upstream's
BF16 state:

| Stage | Result |
| --- | --- |
| Tokenizer, 8 fixtures | identical ids |
| Plan and semantic prefixes | identical |
| Teacher-forced logits, 26 rows over the full vocabulary | worst cosine 0.99968, argmax 26/26 |
| Velocity at t = 1 | cosine 0.99994, relative error 1.13% |
| Latents after 32 midpoint steps, BF16 state | cosine 0.99994, relative error 1.12% |
| The same with the previous FP32 state | cosine 0.99991, relative error 1.33% |
| Audio from the upstream latents | relative error 6.5e-7 |

On 2026-11-20 one-token decode moved to the 12-bit rows. The oracle test
prints the same digits as before on every stage, and the seeded
fixed-score song is byte-identical.

`tests/functional/model_services.py music` against `gufo serve music` passed
on 2026-11-15 in about 65 s. The 8-second song from an edited score measured
RMS 0.16 with no clipped samples, and a same-seed replay was byte-identical.
