# Third-party material

This file comes with the code imported from `headbouyJB/inkling2-gufo` at 8bb6cd5 (MIT, see `LICENSE.inkling2`); paths below are relative to `src/models/inkling2/` as the code sits in Gufo. No third-party source file is vendored: images are decoded by Gufo's own image decoder.
Everything else in this repository is original code under the MIT license (LICENSE). Algorithms from llama.cpp are written out in our own structure, with these exceptions, which are copied or adapted:

- `kFp4` and `kIq4nl` (lookup tables in `hip/kernels.hip`): copied data from llama.cpp/ggml's `kvalues_fp4` and `kvalues_iq4nl` tables (`ggml-common.h`, e412c6f). The same e2m1 values also appear as packed byte constants in `hip/fast.hip`.
- `scale_min_k4` in `hip/kernels.hip` mirrors ggml's `get_scale_min_k4` (`ggml-quants.c`, e412c6f).
- `Covered()` in `cache_adapter.cpp`: a near-verbatim copy of the range-coverage check in Gufo's `src/models/qwen/continuation_adapter.cpp` (v0.35.0), and the identity text in `BuildIdentity()` (`runner.cpp`) follows the format of Gufo's `continuation_identity.cpp`.

The MIT licence texts of llama.cpp and of Gufo are reproduced in full below, as MIT requires for copied portions.

Other facts about the formats and algorithms were taken from, and checked against, the projects in the table. They are cited, not copied.

| Source | Revision | License | What it informed |
| --- | --- | --- | --- |
| llama.cpp (`ggml/src/ggml-common.h`, `ggml/src/ggml.c`) | e412c6f | MIT | NVFP4 block layout: 64 values per block, `block_nvfp4` is 36 bytes (4 E4M3 scales + 32 bytes of e2m1 pairs); ggml type id 40 |
| llama.cpp (`src/models/inkling2-loop.cpp`, `src/llama-arch.cpp`, `conversion/inkling2.py`) | 29b52d3 | MIT | `inkling2_loop` metadata keys, tensor names, message-query packing |
| unslothai/llama.cpp PR 389 (`inkling2` arch, loop keys) | 3b34f4f2f | MIT | root-layout (`inkling2`) metadata keys and tensor names |
| Gufo (`include/gufo/ext/*`, `src/models/qwen38_flash_next/`) | v0.35.0 (0260e11) | MIT | extension API, manifest and build conventions the layout follows |
| Gufo (`src/models/qwen/continuation_adapter.cpp`, `src/models/qwen/continuation_identity.cpp`) | v0.35.0 (0260e11) | MIT | adapted, not only informed by: `Covered()` in `cache_adapter.cpp` is a near-verbatim copy of the range-coverage check, and the identity text built in `BuildIdentity()` (`runner.cpp`) follows the format of `continuation_identity.cpp` (copyright Gufo contributors) |

Model weights (Thinking Machines Inkling 2 Mini and the GGUF conversions by
ggml-org and Unsloth) are not part of this repository and carry their own
licenses.

## llama.cpp license (reproduced verbatim from llama.cpp's LICENSE)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Gufo license (reproduced verbatim from Gufo's LICENSE)

```
MIT License

Copyright (c) 2026 gufo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## hip/kernels.hip, engine.cpp (forward pass)
- Dequantisation of NVFP4, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS blocks, the `kvalues_fp4` / `kvalues_iq4nl` tables, `ggml_ue4m3_to_fp32` and `get_scale_min_k4` are re-implemented as HIP device functions from `ggml/src/ggml-quants.c`, `ggml-common.h` and `ggml-impl.h` (llama.cpp e412c6f, MIT). The `kvalues_fp4` / `kvalues_iq4nl` tables are copied data and `scale_min_k4` mirrors `get_scale_min_k4` (see above); the rest is written out in our own structure (item-wise dequantisation into registers).
- The forward pass (block attention residual `mix`, `control`, `convolve`, `clip`, relative-bias attention with sink, SITU MoE, loop schedule and slot layout) follows `src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp` (llama.cpp e412c6f + 29b52d3, MIT) as described in `docs/internal/models/inkling-2-mini/FORWARD.md`; no source was copied.
- `tests/oracle/llama_dump.cpp` links libllama (llama.cpp e412c6f + the PR #32704 cherry-pick, MIT) as the numerical oracle.

## vision/ (vision encoder)
- Algorithm (patch layout, 2-D RoPE, position-table resize, merge permute, preprocessing and size limits) re-implemented from llama.cpp e412c6f (MIT): `tools/mtmd/models/inkling2v.cpp`, `tools/mtmd/mtmd-graph.cpp` (`build_vit`, `build_inp`, `resize_position_embeddings`, `build_rope_2d`, `build_patch_merge_permute`), `tools/mtmd/mtmd-image.cpp` (`mtmd_image_preprocessor_inkling2v`, `calc_size_preserved_ratio`), `ggml/src/ggml-cpu/ops.cpp` (bilinear `upscale_f32`). No code copied verbatim.

## Other files and data

- `text/*` (tokenizer, pre-tokenizer, chat-template renderer, JSON, tool dialect): original code. It reproduces the behaviour of the official Inkling 2 Mini `tokenizer.json` and `chat_template.jinja` (Apache 2.0, from the model repository); the tests compare against output of those files. The split regex in `pretokenizer.hpp` is the one in the official `tokenizer.json`.
- `text/unicode_tables.cpp`: generated by `text/tools/gen_unicode_tables.py` from Python's `unicodedata` (Unicode 16.0.0); Unicode data is under the Unicode License v3.
- `tests/text/goldens/*.json`: expected tokenisations and renderings produced by `tests/text/tools/gen_goldens.py` with the official tokenizer, tiktoken and jinja2. Some cases contain excerpts of the WikiText-2 test split (Wikipedia text, CC BY-SA 3.0).
- Everything outside `text/`, `vision/`: original code; Gufo's extension API headers (`include/gufo/ext/*`, MIT, copyright Gufo contributors) are included at build time from the Gufo checkout and are not copied into this repository.
- `hip/fast.hip`: original code (from the extension's kernel lab); the 128-token prefill GEMMs, fused MoE tail and decode launch changes were added in Gufo.
- `tests/`: original test programs; `tests/oracle/llama_dump.cpp` links libllama of a llama.cpp build as the oracle and contains no llama.cpp source.
- No code was taken from AGPL-licensed projects.

## Model weights

Model weights are not included in this repository and are not covered by its license. Inkling 2 Mini
(Thinking Machines) is released under Apache 2.0 according to its model card. The GGUF files (Unsloth,
ggml-org, bartowski) carry their own terms; see each model page before using or redistributing them.
