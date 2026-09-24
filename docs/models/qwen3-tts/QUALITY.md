# Qwen3-TTS quality

**All three 1.7B variants reproduce seeded WAVs; streamed and buffered PCM
match exactly on the retained checks.** Upstream comparisons use the original
BF16 checkpoints and pinned [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e).
Latest state/loader qualification: September 21, 2026.

| Check | Result |
| --- | --- |
| Talker prompt versus official, all variants | Cosine 1.0 |
| CustomVoice prefill / cached logits versus official | Cosine 0.999938 / 0.999975 |
| Decoder waveform versus official, fixed codes | Mean absolute error 1.33e-7; maximum 1.28e-6 (limits 1e-5 / 1e-4) |
| Base speaker embedding versus official | Cosine 0.999997 |
| VoiceDesign / Base bounded greedy codes versus official | 5/5 semantic tokens / 80/80 codec codes match |
| Base speech reconstruction versus official | Cosine 0.939502; mean absolute error 0.00827 |
| State and streaming | Seeded replay, cancellation/reuse, reference replacement and decoder history across the 300-frame reset pass |

## Intelligibility

One English paragraph, default sampling, seed 42, natural EOS; recognized by
Qwen3-ASR 1.7B. WER measures word errors, not voice similarity or prosody.

| Variant | Word-error rate | Exact full-WAV replay |
| --- | ---: | --- |
| CustomVoice | 2.70% (1/37 words) | Yes |
| VoiceDesign | 0% (0/37) | Yes |
| Base voice cloning | 0% (0/37) | Yes |

CustomVoice seeds 43/44 each score 0%; pooled across the three seeds, **0.90%
(1/111)**. Both native and official ASR hear seed 42's `and` as `in`.
This small control does not establish corpus-wide quality or improvement.
The paragraph is:

> When the first light reaches the harbor, speak gently, with quiet confidence
> and a warm, resonant tone. The boats are still resting beside the old wooden
> pier, and the town is slowly waking to another bright morning.

## Known limitations

- **Upstream codec choices differ:** CustomVoice agrees on 5/5 main codes but
  only 36/80 total codes across five greedy frames. With identical official
  inputs/history, predictor agreement is 41/45; small BF16 rounding differences
  affect close choices. Native reproducibility does not prove upstream parity.
- Gufo and PyTorch use different RNGs. Equal seeds across runtimes need not
  produce the same speech. Greedy generation can fail to reach EOS; bound tests.
- Base's aggregate speech-code agreement is 57.74%; official BF16/F32
  self-agreement is 55.14%. Waveform and embedding gates still apply.
- Multilingual, voice-similarity and long-form quality, including references
  beyond the 250-frame encoder window, remain unqualified. Incremental text
  segmentation is not guaranteed to preserve the full-text waveform.

## Reproduce

Use the [variant contracts and reference checks](../../../tests/models/qwen3_tts)
and [intelligibility tool](../../../tests/models/qwen3_tts/quality/README.md).
Base requires a reference WAV and its complete matching transcript.

```sh
nix develop -c cmake --build --preset gpu-test --target qwen3_tts_synthesis_hip_test
nix develop -c ./build/gpu-test/qwen3_tts_synthesis_hip_test "$TTS_MODEL"
nix develop -c tools/audio/run_ref_tts.sh greedy
```

Use fixed-code waveform comparisons to isolate decoder error and natural-EOS
sampled speech for intelligibility. Run affected checks; do not relax thresholds.
