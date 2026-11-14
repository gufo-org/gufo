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
