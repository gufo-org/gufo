# Gemma 4 31B QAT quality

**Decode matches the scalar reference; greedy MTP matches autoregressive
output; llama.cpp agrees on long context.** Unsloth QAT target (Q4_0) and the
Q4_0 QAT `gemma4-assistant` drafter;
[identities](artifacts/model-identities.json). The arbiter is Gufo's scalar
CPU implementation of the llama.cpp `gemma4` graph over the same GGUF (FP32
activations, binary16 KV, `src/models/gemma4/reference.cpp`); these are not
unquantized-model or GGUF-conversion checks. Measured September 28, 2026 with
the limits of the [standard 31B](../gemma-4-31b/QUALITY.md).

| Check | Result |
| --- | --- |
| GPU decode (FP32 activations, split-K GEMV and attention) vs scalar reference, 22-token chat prompt | Mean KL 1.7e-7 (limit 1e-5), top-1 22/22 (1.3e-7 before the own-key attention step, 2026-10-05) |
| Eight-row verification vs scalar reference | Mean KL 1.6e-7, top-1 22/22 (1.4e-7 before the own-key step) (WMMA projections since 2026-10-05; 1.2e-7 before); verification rows reproduce single-token decode with the drafter loaded bit for bit (Q4_0 WMMA projection widths 1–16 in `gemma4.projection_ops`, whole-model rows in `gemma4.target`) |
| GPU prefill (binary16 activations × Q4_0 weights, binary16 WMMA projections and attention) vs scalar reference, same prompt | Mean KL 2.2e-6 (limit 0.015), max 1.8e-5, top-1 22/22 (Q8_1 activations until 2026-10-05: 0.0011). llama.cpp b11069 prefill on the same tokens: 0.0010, max 0.0094, 22/22 |
| Bulk prefill vs exact rows, 1542-token conversation past the window and ring | Mean KL 8.5e-6 (limit 0.15), top-1 1542/1542 (Q8_1 activations: 0.0027, 1517/1542) |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported, not a gate) | Mean KL 0.00069, top-1 1353/1353 (Q8_1 activations: 0.050, 1329/1353) |
| Long context vs llama.cpp b11069, 16K templated turn (16,022-token prose prompt, 640-token greedy Gufo reply, teacher-forced) | KL(llama.cpp ‖ Gufo) 0.0028, max 0.109, top-1 626/641, true-token NLL 0.106 (llama.cpp 0.111; the reply is Gufo's own greedy output) |
| Greedy MTP vs single-token decode with the drafter loaded, three prompts, plus prompt-lookup copies | Identical token IDs (`gemma4.target`); with sibling drafts every cycle's logits equal decode's bit for bit (22 of 171 siblings accepted) |
| Greedy MTP vs an AR-only server | Not equal in general, as for the standard 31B: AR-only decode uses the split-K GEMV, whose FP32 summation order differs from the verification kernels |
| Sampled MTP | Seeded replay repeats the same tokens (`gemma4.target`) |
| Session state | Prefix extension, rewind and snapshot restore after a sliding-ring wrap continue bit for bit; snapshots sharing position blocks hold the same payload as fresh copies, also after rewritten or ring-wrapped blocks (`gemma4.target`) |
| Vision | `gemma4.vision_encoder` and `gemma4.vision_session` pass with this repository's `mmproj-BF16.gguf` |

Prefill rows remeasured October 5, 2026, after prefill moved to binary16
activations. Q4_0 weights (q − 8) d round once to binary16; the remaining
prefill error is the binary16 activation rounding. llama.cpp's integer-dot
prefill keeps the Q8_1 activation rounding Gufo used before (0.0010 vs
0.0011).

## Reproduce

```sh
export GUFO_GEMMA4_MODEL=/path/to/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
export GUFO_GEMMA4_MTP_MODEL=/path/to/mtp-gemma-4-31B-it.gguf
export GUFO_GEMMA4_MMPROJ=/path/to/mmproj-BF16.gguf
nix develop -c cmake --build --preset gpu-test --target gemma4_target_test \
  gemma4_projection_ops_test gemma4_vision_encoder_test \
  gemma4_vision_session_test
nix develop -c ctest --preset gpu-full -R '^gemma4\.' --output-on-failure
```

Long context: `gemma4_gpu_probe --model M --chat "$(cat prompt.txt)"
--generate 640 --tokens-out T.i32`, then `gemma4_gpu_probe --tokens T.i32
--logits-out G.g4lg --first <prompt tokens − 1>` and `llama_logits --model M
--tokens T.i32 --output L.g4lg --first <same>` (`tools/gemma4/build_llama_logits.sh`),
compared with `tools/gemma4/compare_logits.py L.g4lg G.g4lg`. Short prompt:
`gemma4_reference_probe --chat "Write one sentence about the sea near Genoa."
--half-kv --tokens-out T.i32 --logits-out R.g4lg` as the reference.

## Benchmark method

Gufo October 17, 2026 (revision a9ff98e, `nix build`, binary SHA-256 prefix
d92d910545862f6b) on a reduced grid (single-user depths 0, 4096, 16384 and
32768, one and two users; the other Gufo cells are TODO), llama.cpp
September 28 (not re-measured); one warmed sample per point (sampled MTP: five
seeds), greedy, thinking off, the
same driver workloads, server flags and table grid as the
[standard 31B](../gemma-4-31b/QUALITY.md#benchmark-method). MTP drafts up to
15 tokens under Gufo's calibrated policy (the default) and up to four on
llama.cpp. Loading drops the page cache
(`sync; echo 3 > /proc/sys/vm/drop_caches`) before each launch; loading and
memory follow the standard 31B's projector-free and calibration-free setup.
Every Gufo multi-user completion matches the isolated AR C1 hash (regenerated
October 17). Commands, counts and server flags are recorded per row in the
artifacts.
Against the October 7 build: prefill +1% at d0 to +11% at 16K (AR), AR decode
unchanged, MTP decode +4–16% except mixed text at 32K (+0.6%), memory
+0.06 GiB.
