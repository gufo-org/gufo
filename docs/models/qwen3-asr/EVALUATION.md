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
| Text decoder layer 0 | `0.999998` | `0.00195` | `0.0625` |
| Prefill logits | `0.999211` | `0.04138` | `0.9375` |

Greedy generation reproduces all **49/49** official token IDs and the exact
transcript using strict argmax. Only exactly equal logits choose the lower
token ID; a lower logit is never promoted through a tolerance window.
Any nonfinite logit fails generation. Independent GPU checks cover near-ties,
large negative logits, invalid values, and causal/cached attention boundaries.

The 2026-09-21 component-only weight upload retains all 49 tokens and the
long-audio, streaming, cancellation and concurrent-request checks. The loader
test verifies that each selected tensor remains in range and aligned while
audio-encoder weights are excluded from the text decoder's device copy.

The current matched oracle explicitly uses **text eager / audio SDPA**.
The old runner requested eager at the outer wrapper, but both nested attention
modules actually used SDPA. The runner now selects and records their actual
backends. Text attention retains the official eager BF16 QK, scaling and
probability casts. The table's text metrics therefore supersede a comparison
against a different backend; they are not evidence of a general accuracy gain.
Full eager and SDPA upstream execution can produce different transcripts.

## Reference and focused checks

Official source `7c6daf77a2421100f5fb066495372c00129d39ff`; checkpoint
`7278e1e70fe206f11671096ffdd38061171dd6e5`. Tests are maintained under
`src/models/qwen3_asr/tests`; generate oracle tensors outside Git with
`src/models/qwen3_asr/reference/run_official.py --text-attention eager
--audio-attention sdpa`. The runner also supports CPU reference execution.

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

Four complete-recording checks retain the previous transcripts: the 15-second
fixture and natural-EOS speech from CustomVoice, VoiceDesign and Base. The
CustomVoice connector remains `in`; the other two synthesized paragraphs have
zero word errors. This remains a small English control, not a corpus score.

The final 2026-09-20 production transport check preserves the uploaded-file
transcript through SSE and two consecutive Realtime utterances. Warm file SSE
completes the 15.05-second recording in 996 ms (RTF 0.066); the resident CLI median
is 989 ms versus 987 ms in the matched unchanged control. No meaningful speed
regression or throughput gain is established. Loading/first-request
initialization is excluded from this comparison.

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
