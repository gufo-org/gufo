# Third-party notices

Gufo's original source is MIT licensed; this does not relicense dependencies or
model weights. This inventory covers direct dependencies and adapted code.
Transitive system/toolchain dependencies retain their upstream licenses.
`flake.lock` pins Nixpkgs revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a`;
its package expressions record source revisions, patches and build options.
Non-Nix builds must retain the notices of the versions they actually distribute.

## Runtime and compiled code

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| ROCm / HIP | Linked runtime | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/clr](https://github.com/ROCm/clr) |
| hipBLAS | Linked | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) |
| hipBLASLt | Linked | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) |
| rocBLAS | Linked | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) |
| MIOpen | Linked; ASR convolutions | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) |
| AOTriton | Linked; H3 attention images | `MIT` | 0.11.1b; flake.lock | [ROCm/aotriton](https://github.com/ROCm/aotriton) |
| hipCUB | Headers compiled into kernels | `BSD-3-Clause` | ROCm 7.2.3; flake.lock | [ROCm/hipCUB](https://github.com/ROCm/hipCUB) |
| rocPRIM | Headers compiled into kernels | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) |
| rocWMMA | Headers compiled into kernels | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocWMMA](https://github.com/ROCm/rocWMMA) |
| Composable Kernel | Headers compiled into attention kernels | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| ICU | Linked; Unicode normalization/tokenization | `Unicode-3.0` | 78.3; flake.lock | [unicode-org/icu](https://github.com/unicode-org/icu) |
| curl / libcurl | Linked; image HTTPS and evaluation client | `curl` | 8.21.0; flake.lock | [curl/curl](https://github.com/curl/curl) |
| OpenSSL | Linked; cryptographic hashes and HTTPS dependency | `Apache-2.0` | 3.6.3; flake.lock | [openssl/openssl](https://github.com/openssl/openssl) |
| libpng | Linked; image decoding | `libpng-2.0` | 1.6.58; flake.lock | [pnggroup/libpng](https://github.com/pnggroup/libpng) |
| libjpeg-turbo | Linked; JPEG decoding | `IJG AND BSD-3-Clause AND Zlib` | 3.1.4.1; flake.lock | [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) |
| FFmpeg | Separate ffmpeg/ffprobe executables for media | `GPL-3.0-or-later` (Nix build with GPL/version3 components) | 8.1.2; flake.lock | [FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| GNU C/C++/OpenMP runtimes | System runtime libraries; no Gufo source import | `LGPL-2.1-or-later AND (GPL-3.0-or-later WITH GCC-exception-3.1)` | glibc/GCC packages in flake.lock | [GNU](https://www.gnu.org/software/) |
| llama.cpp / ggml | Adapted quantization, attention and model-private HIP kernels | `MIT` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`; attention `e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0` | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Qwen chat templates | Reference Jinja and adapted renderer behavior | `Apache-2.0` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (27B), `de4b8e4d43b917e7706784d8bb445c9af86a3540` (Flash-Next) | [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) |
| DS4 | Adapted loader, tokenizer, sessions and HIP kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| DS4 GB10/GX10 fork | Adapted paired MoE launch code, now HIP | `MIT` | `910501e` | [xangel82/DS4](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA) |
| h3.c | Adapted H3 geometry, scheduling and sampler; implementation reference | `MIT` | `8974cc055ea9c02fcd14cc27dfda3e1027c05153` | [antirez/h3.c](https://github.com/antirez/h3.c) |
| ccv TensorOps ancestry | Attribution retained through h3.c | `BSD-3-Clause` | h3.c pin above | [liuliu/ccv](https://github.com/liuliu/ccv) |

Upstream license texts are retained in `licenses/` and
[src/models/deepseek_v4_flash/LICENSE.ds4](src/models/deepseek_v4_flash/LICENSE.ds4).
CMake installs these with `LICENSE`, `NOTICE` and this inventory under
`share/licenses/gufo`. Preserve embedded copyright notices when modifying
adapted code. Model-private changes and import boundaries are recorded in the
[DS4 provenance](src/models/deepseek_v4_flash/UPSTREAM.md),
[DS4 HIP import](src/models/deepseek_v4_flash/kernels/rocm/mmq/VENDOR.md),
[Flash-Next HIP import](src/models/qwen38_flash_next/kernels/rocm/mmq/VENDOR.md)
and [H3 evaluation record](docs/models/minimax-h3/EVALUATION.md).

This software is based in part on the work of the Independent JPEG Group.

FFmpeg is invoked as a separate process, not linked into Gufo. The pinned Nix
build enables GPL/version3 components; a system FFmpeg build can have different
terms. Redistributors bundling FFmpeg or other copyleft dependencies must also
provide their applicable license texts and corresponding source/build materials,
not just this MIT license. The pinned Nixpkgs expressions and `flake.lock`
identify those sources and build recipes; a source URL alone is not a substitute
for fulfilling the applicable distribution terms.

The official 0731 DeepSeek continuations in
`tests/models/deepseek_v4_flash/fixtures/official-0731.json` come from DS4 revision
`6289c516273979173abbc062209a81dd3706b804`; the fixture retains source hashes and
its upstream MIT notice. External model files are never part of the binary package.

## Build and development only

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| LLVM/Clang | HIP build toolchain | `Apache-2.0 WITH LLVM-exception` | ROCm LLVM from flake.lock | [ROCm/llvm-project](https://github.com/ROCm/llvm-project) |
| Triton | Build-time compilation of retained H3 kernels | `MIT` | 3.7.0; flake.lock | [triton-lang/triton](https://github.com/triton-lang/triton) |
| ROCprofiler SDK / ROCTx | Development profiling and benchmark markers only | `MIT` | ROCm 7.2.3; flake.lock | [ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) |
| PyTorch ROCm | Independent evaluation; not shipped | `BSD-3-Clause` | 2.12.0; flake.lock | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| Torchvision | Evaluation; not shipped | `BSD-3-Clause` | 0.27.0; flake.lock | [pytorch/vision](https://github.com/pytorch/vision) |
| LPIPS | Evaluation; not shipped | `BSD-2-Clause` | 0.1.4; flake.lock | [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) |
| Transformers / Accelerate / Safetensors / Hugging Face Hub | Evaluation, artifact inspection and model acquisition | `Apache-2.0` | flake.lock | [Hugging Face](https://github.com/huggingface) |
| NumPy / SciPy | Evaluation arrays and signal analysis | `BSD-3-Clause` | flake.lock | [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy) |
| Requests | Evaluation HTTP client | `Apache-2.0` | flake.lock | [Requests](https://github.com/psf/requests) |
| SoX | Development audio utility | `GPL-2.0-or-later` | flake.lock | [SoX](https://sourceforge.net/projects/sox/) |
| Torchvision AlexNet weights | Evaluation data only; not shipped | `NOASSERTION` | SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` | [PyTorch model distribution](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |

Python/PyTorch and reference scripts are not installed with Gufo. Triton is
needed while building H3's native code, not during inference. Kernel/tuning
executables are built only with `GUFO_BUILD_TOOLS=ON`; ROCprofiler is supplied
by the development shell. Ordinary build tools (CMake, Ninja, pkg-config,
Python and the host compiler) retain their own licenses and are not Gufo code.

The LPIPS evaluator checks the AlexNet digest above and calibration digest
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`, records the
loaded module hash and library versions, and requires locally available weights.

## External Model Artifacts (Non-Distributed)

Model checkpoints are not part of the gufo source or binary
distribution. Operators obtain them directly from their publisher and remain
responsible for the terms governing their location and use.

### Qwen-Image-2.1

- **Artifact**: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
  revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`; separate BF16
  safetensors, tokenizer and model configuration.
- **License**: [Qwen Research License Agreement, September 20, 2026](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE).
  The granted use is non-commercial; commercial use requires a separate license.
  Gufo neither ships nor automatically downloads these weights.
- **Reference implementation**: Diffusers and Transformers, Apache-2.0.
  Native inference follows the [pinned operator contract](src/models/qwen_image_21/UPSTREAM.md);
  Python reference dependencies are development-only.
- **Publisher notice**: “Qwen is licensed under the Qwen RESEARCH LICENSE
  AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd.
  All Rights Reserved.”

### MiniMax H3 FL2VA

- **Component Name**: MiniMax H3 Base FL2VA checkpoint
- **Upstream URL**: https://huggingface.co/MiniMaxAI/MiniMax-H3
- **Pinned Revision**: `42ed227ee7df40d41602854ae760620d6eb651fe`
- **Component Used**: Locally supplied BF16 text encoder, Omni Transformer,
  VisualVAE, AudioVAE, tokenizer, scheduler configuration, and metadata under
  `FL2VA/`
- **License**: MiniMax H3 Community License Agreement dated August 2, 2026, or
  separate operator-specific authorization where required
- **License File SHA-256 at Pinned Revision**:
  `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`
- **Copyright / Notice Source**: Copyright © 2026 MiniMax. All Rights Reserved.
- **Relationship**: External, access-controlled, operator-supplied runtime
  artifact. It is never committed, packaged, mirrored, automatically
  downloaded, or redistributed by gufo.
- **Operational Boundary**: Every operator must independently obtain access
  from MiniMax, accept or obtain the terms applicable to that operator, and
  configure a local checkpoint path. Possession of the gufo source
  does not grant model rights.
- **Serving Boundary**: The engine supplies numerical execution only. Anyone
  exposing H3 through an API is responsible for the publisher's user terms,
  acceptable-use, safeguards, disclosures, reporting, attribution, and
  territorial requirements.

The project records an operator attestation that the dedicated development
machine is authorized for this work. The repository does not contain private
license correspondence or credentials and does not independently make a legal
determination about a downstream operator.

### Qwen3-VL-32B Encoder Component

- **Component Name**: Qwen3-VL-32B encoder weights used by MiniMax H3
- **Upstream URL**: https://github.com/QwenLM/Qwen3-VL
- **Pinned Revision**: Supplied as part of the pinned MiniMax H3 FL2VA package
- **Component Used**: H3 prompt encoder through layer 50
- **SPDX License Identifier**: `Apache-2.0`
- **Relationship**: External model component within the operator-supplied H3
  checkpoint; not distributed by gufo

See [the MiniMax H3 guide](docs/models/minimax-h3/README.md) for the acquisition, release, and
runtime boundary.
