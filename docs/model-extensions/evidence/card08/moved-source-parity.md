# Card 08 — moved-source parity (parent `build/card07-release` → candidate `build/card08-release`)

Card 07 moved sources to new paths. This card keeps every path and changes the
owning target, so the audit is the mirror image: the same source must compile
with the same effective flags and emit the same code after the move.

`tools/prof/card08_parity.py` (new) derives the moved set instead of trusting a
list: `compile_commands.json` records each object below
`CMakeFiles/<target>.dir/`, so a source whose object owner changed is a moved
source. Differences outside the tool's `ALLOWED` table fail, and an `ALLOWED`
entry that matches nothing also fails, so a justification cannot outlive the
difference it explains.

Derivation result (27 sources):

| Owner change | Sources |
| --- | --- |
| `gufo_core` → `gufo_minimax_h3` | 21 (13 C++, 8 HIP, including the `media_posix.cpp` / `media_windows.cpp` generator-expression pair) |
| `gufo_core` → `gufo_eval` | 3 (`dataset.cpp`, `extract.cpp`, `regrade.cpp`) |
| `gufo_core` → `gufo_compare` | 1 (`logit_comparator.cpp`) |
| `gufo_deepseek_v4_flash` → `gufo_deepseek_v4_flash_text` | 2 (`chat_template.cpp`, `runtime/sampling.cpp`) |

## Compile flags

```
nix develop -c python3 tools/prof/card08_parity.py --mode flags \
  --before build/card07-release/compile_commands.json \
  --after build/card08-release/compile_commands.json
```

`flags: compared=27 failures=0` — 21 identical, 6 justified, exit 0. All
justified rows and their reasons:

- The four `src/eval/*.cpp` and `src/testing/compare/logit_comparator.cpp`
  sources lost `-DGUFO_FFMPEG_EXECUTABLE="ffmpeg"` and
  `-DGUFO_FFPROBE_EXECUTABLE="ffprobe"`. Those discovery macros sat on
  `gufo_core` only because the H3 media sources were core objects; a grep over
  the tree shows `src/core/**` never reads them, and they remain private
  compile definitions of `gufo_minimax_h3`, where both media sources still see
  the identical values.
- The two DeepSeek text sources lost the model's ROCm include directories
  (`runtime` spelled from the model root, `kernels/rocm`,
  `kernels/rocm/detail`, `kernels/rocm/mmq`) and gained
  `src/models/deepseek_v4_flash/text/../runtime` (the same directory, written
  relative to the new target) plus `-Wall -Wextra -Wpedantic`. Those two files
  include only core headers, `tool_calls.hpp` (header-only constants) and
  `native_internal.h`; they compiled with the project warnings before, as
  test-executable sources. `-D_GNU_SOURCE`, `-DDS4_ROCM_BUILD=1` and the
  shipped `-O3 -ffast-math -fno-finite-math-only` sampling codegen are
  unchanged — `DS4_ROCM_BUILD` is read by no source or header in the tree.

## Host objects

```
nix develop -c python3 tools/prof/card08_parity.py --mode object \
  --before build/card07-release/compile_commands.json \
  --after build/card08-release/compile_commands.json
```

`object: compared=27 failures=0` — `symbols identical: 19`,
`symbols identical apart from the TU id: 8`,
`code identical or move artifact: 27`, exit 0. Every function's mnemonic
sequence matched; the only accepted operand differences are the per-TU
`__hip_cuid_*` identifier and `__hip_gpubin_handle_*` (both follow the object
path) plus `__FILE__`-derived local-label offsets. The first run failed symbol
comparison on the eight `.hip` host wrappers for exactly that TU-id pair, so
the tool now masks both names and still fails anything else.

## Device code (moved H3 HIP tier)

```
nix develop -c python3 tools/prof/card08_parity.py --mode device \
  --before build/card07-release/compile_commands.json \
  --after build/card08-release/compile_commands.json \
  --scratch /tmp/card08-parity
```

`device: compared=8 failures=0`, exit 0: for `attention.hip`, `audio_vae.hip`,
`denoiser.hip`, `dense.hip`, `dit.hip`, `prompt_encoder.hip`,
`sampling.hip`, `video_vae.hip`, the gfx1151 `.text`, `.rodata` and `.note`
sections are byte-identical between parent and candidate (24 of 24 sections).
`dit.hip` keeps its appended `-O2` (no flag difference reported), and the
`HIPCUB`/`ROCPRIM` system includes resolve identically now that they are
private to `gufo_minimax_h3`. Scratch was removed after the run.

## Kernel resources

```
nix develop -c python3 tools/ci/check-kernel-resources.py build/card08-release/gufo \
  --compare build/card07-release/gufo
```

`0 worse, 0 better, 2692 same, 0 one side only; scratch increased in 0`, exit 0.

Against the registered budget:

```
nix develop -c python3 tools/ci/check-kernel-resources.py build/card07-release/gufo \
  --baseline tools/ci/kernel-resources.json
nix develop -c python3 tools/ci/check-kernel-resources.py build/card08-release/gufo \
  --baseline tools/ci/kernel-resources.json
```

Both exit 1 with `2692 kernels checked, 246 use scratch, 30 over budget`, and
the sorted `FAIL` rows are identical between the two runs (`diff` empty: same
kernel names, scratch bytes, vgpr and spill). The registered
`kernel_resources_test` failure is therefore pre-existing on the parent
revision and untouched by this card, exactly as recorded for Card 07.

## Numerical checks for the moved H3 code

The H3 HIP oracle tests that would add numerical evidence
(`minimax_h3_denoiser_oracle`, `minimax_h3_dit_block0_real`,
`minimax_h3_prompt_encoder_real`, `minimax_h3_video_vae_real`,
`minimax_h3_audio_vae_real`, `minimax_h3_runtime_hip_test`,
`minimax_h3_denoiser_{smoke,reuse_smoke,production_shape_oracle}`) skip with
exit 77 for missing weights on this host, so they are
[pending](pending-commands.md), not passes. What is proven here is code
identity: identical device sections plus identical kernel scratch usage mean
the move cannot change H3 numerical behaviour.
