# Experimental Qwen family library

Publish the shared Qwen family code as an experimental extension library. The
headers under `include/gufo/ext/families/qwen/` forward to the family libraries
and carry no promise, like `include/gufo/ext/experimental/`:
- `text/tokenizer.hpp`, `text/chat_template.hpp` (`gufo::tokenization`).
- `control_tokens/control_tokens.hpp` (`gufo::tokenization`).
- `draft_vocab/draft_vocab.hpp` (`gufo::speculative`).
- `vision/prompt.hpp`, `vision/encoder.hpp`, `vision/image_cache.hpp`,
  `vision/device_input.hpp`, `hip_common/rope.hpp`
  (`gufo::models::qwen::vision`).
- `kernels/prefill_quant_gemm.hpp`, `kernels/ops/token.hpp` (`gufo::hip`).

Their transitive closure adds these `src/families/qwen/` headers, besides the
eleven forwarded ones, to `SURFACE`: `hip_common/attention_policy.hpp`,
`hip_common/continuation_hooks.hpp`, `hip_common/execution_policy.hpp`,
`hip_common/gemm_route.hpp`, `hip_common/quant_ops.hpp`, `kernels/ops/gemm.hpp`
and `kernels/small_batch_gemm.hpp`.

No-promise paths: a `SURFACE` path reachable only through
`include/gufo/ext/experimental/` or `include/gufo/ext/families/` is listed in the
generated `include/gufo/ext/NO_PROMISE`. Its changes need no `ext-api` label or
record and do not advance the revision; a path a versioned header also reaches
keeps the rule. These paths stay in `SURFACE` and the extension build ID. This
record is needed for the `gufo::qwen_family` CMake contract below.

Public CMake contract: `gufo::qwen_family` (alias of `gufo_qwen_family`) links
the family implementation and adds no include path; link it beside
`gufo_ext_runtime`, and beside `gufo_ext_hip` for the HIP headers, which exist
only in HIP builds. Extensions outside the tree may link it; a loadable
extension module may not yet, and configuring one that does fails.

Layer rules: a header under `families/<family>/` outside `models/` never reaches
a `families/<family>/models/` directory (public or under `src/`) or
`src/models/`. A header under `families/<family>/models/<model>/` never reaches
another family's or model's `models/` directory, and reaches `src/models/` only
inside the one directory declared for it in `tools/ci/check-extension-api.py`.
Headers directly under `families/` or `families/<family>/models/` are rejected,
as are macro includes and quoted includes that resolve nowhere. Additive.
