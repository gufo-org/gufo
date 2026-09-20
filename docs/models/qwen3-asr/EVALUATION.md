# Qwen3-ASR evaluation

The native quality gate compares the official and native frontend, audio
encoder, text decoder, prefill logits, every generated token, and final
transcript.

| Boundary | Cosine | Relative L2 | Maximum absolute error |
|---|---:|---:|---:|
| CPU log-mel frontend | `1.0` | `9.43e-7` | `7.21e-5` |
| HIP convolutional frontend | `1.0` | `1.11e-4` | `0.015625` |
| Audio encoder layer 0 | `0.999998` | `0.00199` | `0.0625` |
| Complete audio encoder | `0.999855` | `0.01705` | `0.00830` |
| Text decoder layer 0 | `0.999998` | `0.00203` | `0.0625` |
| Prefill logits | `0.999703` | `0.02444` | `0.63281` |

Greedy generation reproduces all 49 official token IDs and the exact
transcript. The model-private argmax uses a `0.25` BF16 tie window for the
retained decode GEMV route: the official sequence has a `35.5/35.5` tie, while
the alternate accumulation order produces `35.25/35.5`; every captured
non-tied official margin is at least `1.0`.

## Reference and focused checks

Official source `7c6daf77a2421100f5fb066495372c00129d39ff`; checkpoint
`7278e1e70fe206f11671096ffdd38061171dd6e5`. Test fixtures are maintained under
`tests/models/qwen3_asr`; generate oracle tensors outside Git with
`src/models/qwen3_asr/reference/run_official.py`.

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target qwen3_asr_transcription_hip_test
nix develop -c ctest --preset gpu-full -R qwen3_asr --output-on-failure
```

Use the affected frontend, encoder or decoder check before full fixture replay.
Finite tensors, established boundary tolerances, all generated IDs and final
transcript must pass. Test request cancellation and byte/layout limits when
changing HTTP or audio handling.

The focused runtime check also runs a 60.2-second recording through a 512-token
context. The merged token IDs and text equal independent inference of the same
low-energy chunks. File-streaming text equals buffered output; concurrent
requests and cancellation/reuse retain the exact 49-token reference. This
validates splitting and state isolation, not long-form recognition accuracy.

Transport checks use OpenAI file-transcription SSE and GA Realtime manual
commits (24-kHz PCM16). No timestamp/aligner model is involved. The native model
gets a complete committed utterance; it does not use the upstream wrapper's
revisable rolling-prefix mode.

## Limits

The retained 15-second fixture is not a multilingual/long-form qualification.
Full audio attention matches the pinned upstream eager/SDPA path: the encoder
calls its layers without an attention mask. Its cumulative sequence lengths
are consumed by FlashAttention instead; parity with that windowed path is
unqualified. Do not replace full attention with the rejected 104-token window
on the strength of a speed measurement. Long-form and broader independent
capability evaluation remain **TODO**.
