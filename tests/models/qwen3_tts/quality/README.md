# Qwen3-TTS long-form quality input

The canonical exactness contract uses the short sentence in
`reference/generate_artifacts.py`. Long-form listening tests use a bounded
excerpt fetched on demand from:

`https://dgoldberg.sdsu.edu/515/harrypotter.txt`

Run from the repository root:

```sh
nix develop --command tools/audio/run_ref_tts_quality.sh
```

The fetch requests only the beginning of the source and retains at most 2,200
characters, beginning with `Mr. and Mrs. Dursley`. The excerpt, discrete codec
tokens, and waveform are written below `artifacts/qwen3_tts/quality/`, which is
gitignored. They are a listening/continuity fixture, not the exact PR contract.
The quality route keeps upstream sampling enabled; fully greedy generation is
reserved for the short bounded exactness artifact.

For sampled implementations with a different RNG stream, waveform identity is
not a meaningful quality gate. Compare intelligibility using the same native
Qwen3-ASR-1.7B runtime for the upstream and native waveforms:

```sh
nix develop --command python3 \
  tests/models/qwen3_tts/quality/compare_intelligibility.py \
  --reference-npy artifacts/qwen3_tts/rocm/waveform.npy \
  --candidate-wav /tmp/qwen3-tts-native.wav \
  --asr-model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision> \
  --gufo result/bin/gufo
```

For native optimization A/B tests, pass the retained baseline directly with
`--reference-wav` instead of `--reference-npy`.

The report checks word error rate against the requested text, transcript
agreement, duration, finite samples, peak, and RMS. It complements—not
replaces—the exact talker-logit and speech-decoder waveform gates.
