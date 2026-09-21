# Qwen3-TTS evaluation

The upstream Python implementation never serves a request. It is an offline
oracle for tensor, logit, codec, waveform, and listening comparisons, pinned at
commit `022e286b`.

Sampled codec frames are not expected to be byte-identical, because PyTorch ROCm
and the native C++ sampler use different random-number generators. Parity is
therefore judged from exact tokens, tensor and logit metrics, near-exact decoder
waveform comparison, and an end-to-end intelligibility gate.

State/restart qualification, 2026-09-21: Base voice cloning reproduces its
complete seeded WAV across fresh server processes and resident requests.
The speaker softmax matches an independent FP64 calculation at 1536 channels;
traced, untraced and reloaded encoders return identical embeddings.
Official embedding cosine remains **0.999997**. The 64-frame synthesis check
passes cancellation/reuse and exact buffered/streamed PCM. Decoder snapshots
retain only live attention and convolution history, including reference
replacement and endpoints around the 300-frame reset.

Current complete-sentence checks use the same 37-word English paragraph,
default sampling and automatic language selection. Native Qwen3-ASR-1.7B
transcribes the output; WER measures intelligibility, not voice similarity or
prosody. Base uses the complete matching transcript of its reference recording.

| Variant | Seed-42 WER | Natural EOS | Exact full-WAV replay |
| --- | ---: | ---: | --- |
| CustomVoice | 2.70% (one `and` → `in` substitution) | 235 frames / 18.80 s | yes |
| VoiceDesign | 0% | 187 frames / 14.96 s | yes |
| Base ICL | 0% | 166 frames / 13.28 s | yes |

CustomVoice seeds 43 and 44 both score 0% WER. Across these three seeds the
current result is 1/111 word errors (0.90%); the pre-correction control has
0/111. This small check does not establish a quality improvement or corpus-wide
equivalence. The speed optimizations preserve the corrected waveform exactly;
the numerical corrections themselves change sampled trajectories. The official
Qwen3-ASR wrapper also transcribes the seed-42 connector as `in`; the discrepancy
is not unique to the native recognizer.

The paragraph used for these checks is:

> When the first light reaches the harbor, speak gently, with quiet confidence
> and a warm, resonant tone. The boats are still resting beside the old wooden
> pier, and the town is slowly waking to another bright morning.

### Performance qualification

Same production build, fixed 64-frame workload (5.12 seconds of audio), one
warmup and three timed buffered requests:

| Variant | Median buffered request | First streamed PCM | Buffered/streamed PCM |
| --- | ---: | ---: | --- |
| CustomVoice | 2.018 s | 0.201 s | byte-identical |
| VoiceDesign | 2.024 s | 0.201 s | byte-identical |
| Base ICL | 2.234 s | 0.345 s | byte-identical |

The latest matched Base control is 2.693 s buffered and 0.882 s to first PCM:
the retained reference-state cache reduces warm request time by **17.0%** and
first-audio latency by **60.9%**. All three optimized waveforms match the previous
corrected build exactly. CustomVoice and VoiceDesign remain at approximately
their previous speeds; small differences do not establish a throughput gain.
Base includes reference-code talker prefill. See
[benchmark settings](BENCHMARKS.md) for scope and identities.

The Base decoder cache retains one immutable reference frontier, with exact
convolution history and attention KV. Checks cover replacement by another voice,
unrelated intervening requests, and reference endpoints before/at/after the
300-frame reset. Cached and cold suffix waveforms are byte-identical. Existing
seeded synthesis, stream cancellation and shorter-request replay checks remain
in the same model test targets. The complete 166-frame Base natural-EOS waveform
also matches the pre-cache production build exactly, buffered and streamed.

### Gate metrics

| Variant | Metric | Measured | Gate |
|---|---|---|---|
| All | Talker prompt cosine | `1.0` | `> 0.99` |
| All | BF16 norm, RoPE, codec sum and causal attention formulas | pass | exact operator outputs |
| CustomVoice | First-layer cosine / MAE | `0.999999` / `0.00011446` | `> 0.99` |
| CustomVoice | Prefill-logit cosine | `0.999938` | `> 0.95` |
| CustomVoice | Cached-logit cosine | `0.999975` | tracked |
| CustomVoice | First greedy codec frame | `16/16` | exact |
| CustomVoice | Predictor-logit cosine | `0.999715` | `> 0.95` |
| All | Decoder waveform MAE / max | `1.33e-7` / `1.28e-6` | `< 1e-5` / `< 1e-4` |
| CustomVoice | Five-frame greedy codec / main-code agreement | `36/80` / `5/5` | Main codes exact; later acoustic-code differences remain |
| CustomVoice | Fixed official inputs/history, CPU and ROCm | `41/45` predictor choices each; cosine `0.999676`–`0.999908` | Cosine `> 0.999`; track discrete differences |
| VoiceDesign | Prompt cosine, semantic greedy tokens | `1.0`, 5/5 | `> 0.9999`, exact |
| Base | Prompt cosine, bounded greedy codes | `1.0`, 80/80 | `> 0.999`, exact |
| Base | Speaker-embedding cosine | `0.999997` | `> 0.9999` |
| Base | Speech-code agreement, aggregate | `0.577351` | `> 0.55` |
| Base | Speech-code agreement, groups 0 / 1 | `0.970` / `0.950` | `> 0.95` / `> 0.90` |
| Base | Speech reconstruction cosine / MAE | `0.939502` / `0.00827` | `> 0.93` / `< 0.02` |

### Known limitations

- **Replay coverage:** current 64-frame sampled checks pass exact codec and
  waveform replay for CustomVoice, VoiceDesign and Base ICL, including
  cancellation and reuse with a shorter intervening prompt. Streaming, with
  the first audio emitted after four frames, is exact against buffered synthesis;
  the isolated decoder also matches through its 300-frame context boundary.
  The historical frame-38 divergence did not reproduce here; this is bounded
  coverage, not an arbitrary-length replay guarantee.
  The complete-sentence production checks above also repeat their full WAVs.
- **Upstream generation:** replay within Gufo does not establish identical
  full greedy codec trajectories to upstream. Even the pinned upstream CPU and
  ROCm greedy fixtures differ from the third frame. On fixed official predictor
  inputs and history, remaining choice differences have tied or 0.125-logit
  margins. A layer trace first differs at one of 2,048 projection values by
  `0.0000305176`: Gufo matches the FP64 calculation rounded to BF16 there;
  upstream differs. Small rounding differences then accumulate across layers.
  This explains an observed source of drift, not every later token difference.
- **Greedy EOS:** a diagnostic reached 3000 frames without EOS. Bound greedy
  checks; use sampled output for complete-sentence quality.
- **Base speech codes:** aggregate agreement 0.577351 is close to the official
  BF16-versus-F32 self-control (0.5514); discrete RVQ decisions amplify rounding.
  This does not justify relaxing the retained waveform/embedding gates.
- **Long reference clips:** the official 250-frame encoder window is wired,
  but the retained 202-frame clip does not exercise it. A longer oracle is TODO.
- Broader multilingual, voice-similarity and long-form corpus evaluation remains
  unqualified; the paragraph checks above are deliberately bounded.

## Reference and focused checks

Official QwenLM/Qwen3-TTS revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e` (Apache-2.0). Model fixtures and quality
scripts live in `tests/models/qwen3_tts`; tensor payloads stay in ignored
`artifacts/qwen3_tts`. Full logits are regenerated, not stored as experiment dumps.

```sh
nix develop -c cmake --build --preset gpu-test --target qwen3_tts_synthesis_hip_test
build/gpu-test/qwen3_tts_synthesis_hip_test "$TTS_MODEL"
# For Base, also pass a reference WAV and its complete transcript file.
build/gpu-test/qwen3_tts_synthesis_hip_test \
  "$TTS_BASE_MODEL" reference.wav reference.txt
nix develop -c tools/audio/run_ref_tts.sh greedy
nix develop -c tools/audio/run_ref_tts.sh
nix develop -c python3 tests/models/qwen3_tts/quality/compare_intelligibility.py \
  --reference-npy artifacts/qwen3_tts/rocm/waveform.npy \
  --candidate-wav native.wav --asr-model "$ASR_MODEL" --gufo result/bin/gufo
```

Base ICL checks require a complete transcript matching the reference recording;
a partial transcript cannot qualify synthesis quality.

Use five-frame greedy oracles for exact codes, sampled seed-42 oracles for
complete speech, and the same ASR runtime to transcribe both waveforms.
`--contract` supplies variant-specific text. VoiceDesign/Base captures include
prompt and speaker/speech-encoder boundaries. Keep pinned contract hashes,
shapes, dtype and upstream provenance with each generated reference.

The talker test also checks independent BF16 formulas, causal attention around
the 64-token dispatch boundary, and cached versus full attention. The native
fusions retain the upstream casts: codec sum before text addition, normalized
values before norm weights, RoPE products before addition, and QK/scaling/
softmax probabilities before attention output.

For a predictor comparison unaffected by earlier native token choices, generate
a short official trace and feed its hidden states and codec history back:

```sh
nix develop -c python3 tests/models/qwen3_tts/reference/probe_official.py \
  --reference-root "$TTS_REFERENCE" --dependency-root "$TTS_DEPENDENCIES" \
  --model "$TTS_MODEL" --device cuda:0 --max-new-tokens 4 \
  --out artifacts/qwen3_tts/teacher-rocm
build/gpu-test/qwen3_tts_talker_hip_prefill_test \
  "$TTS_MODEL" artifacts/qwen3_tts artifacts/qwen3_tts/teacher-rocm
```

`cuda:0` is PyTorch's ROCm device spelling. Use `--device cpu` for the CPU
control. The trace includes an FP64 projection control rounded to BF16; it
does not assume that every upstream rounding choice is more accurate.

CPU request checks cover independent talker/predictor controls, nucleus
semantics, top-k ties, stream cancellation and rejection of nonfinite waveform
samples before WAV/PCM conversion. `audio_websocket_test` covers
the actual socket protocol, authentication, fragmented Unicode, ping/close and
multiple utterances without loading models. The retained waveform tolerances
remain MAE `<1e-5`, maximum error `<1e-4`; do not relax them for streaming.

Interfaces were checked against OpenAI's audio streaming / GA Realtime schemas
and vLLM-Omni `serving_speech_stream.py` at
`23f41264456684c793283502f811aab7dcda2c88`. Sentence/clause input segmentation
is a separate latency/quality tradeoff, never a lossless waveform assertion.
