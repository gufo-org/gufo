# LFM3-Audio numerical qualification

October 29, 2026, Strix Halo `gfx1151`. Weights: LiquidAI/LFM3-Audio-2B-GGUF
`02355c1a141e464a064f2ab2c276638dbcb7f38a`.

## Independent reference

[`lfm3_audio_ref.py`](../../../tests/models/lfm3_audio/reference/lfm3_audio_ref.py)
is an FP32 PyTorch implementation written from the official Hugging Face remote
code (LiquidAI/LFM3-Audio-2B `cddb56b5`: `modeling_lfm3_audio.py`,
`modeling_lfm3.py`, `modeling_liquid_codec.py`, the feature extractor and
processor). It reads the lossless BF16 GGUF pair, shares no code with Gufo and
derives the FSQ table in closed form (checked against the GGUF). On its own it
transcribes the English fixture exactly and its greedy TTS of "The quick
brown fox jumps over the lazy dog." transcribes back exactly, which checks
the reference end to end.

`lfm3_audio_probe` compares the native stages with its fixtures (BF16
backbone and BF16 mmproj, maximum absolute error / relative to the
reference's largest magnitude):

| Stage | max abs | relative |
| --- | ---: | ---: |
| Log-mel features (FP32 DFT vs FFT; near-floor bins) | 4.9e-4 | 2.5e-4 |
| Projected audio embeddings, 125 tokens | 9.1e-7 | 8.8e-7 |
| ASR prompt final hidden state (145 tokens) | 1.5e-5 | 1.6e-7 |
| Depth decoder logits, 12 × 2017, half masked | 9.5e-6 | 6.2e-7 |
| Frame embedding (sum of 12 codebook rows) | 0 | 0 |
| Codec audio, 40 random frames (76800 samples) | 3.6e-6 | 8.1e-6 |
| Streamed codec (chunks of 1–5 frames) vs one call | 4.5e-8 | 1.0e-7 |
| TTS prompt hidden state | 2.0e-6 | 6.0e-7 |
| Streamed encoder (100 ms, 2.3 ms and 250 ms pieces) vs one call | 0 | 0 |

Greedy ASR returns the reference transcript and token IDs, also when the
turn is prefilled incrementally from streamed audio rows after an erased
earlier turn (the context is recomputed from its recorded inputs). Live ASR,
fed in 100 ms pieces, returns the reference's chunked `transcribe_live`
decode exactly, including the final block (`<|mixed_end|>` then " Extract":
the clip is cut from a longer reading). Greedy TTS with
the reference's fixed reveal order reproduces all **46/46** reference frames
code for code, including the end frame, through the device sampler.

The Q8_0 backbone with BF16 towers keeps the transcript; its ASR prompt hidden
state differs from the BF16 reference by 1.9e-3 relative and greedy TTS
diverges from the BF16 codes after a few frames (6/46 identical), as expected
for a quantized backbone. With the Q8_0 mmproj, audio embeddings differ by
5.6e-3, depth logits by 1.1e-2 and codec audio by 7.4e-2 relative; this is why
the BF16 mmproj is the recommended pair.

## End-to-end checks

The [service suites](../../../tests/functional/README.md) pass against
`serve asr` and `serve tts` (Q8_0 backbone, BF16 mmproj): exact transcript of
the English fixture over JSON/text/verbose/SSE with greedy replay, concurrent
requests, rejections, malformed audio recovery and stream cancellation; for
TTS, voice discovery, valid 24 kHz PCM16, exact same-seed replay, WAV = PCM =
SSE, concurrent texts, rejections and cancellation. Both TTS texts transcribe
with **word error rate 0.0** by the independent Qwen3-ASR 1.7B evaluator.
The [realtime suite](../../../tests/functional/README.md) passes against
`serve asr` (Q8_0 backbone, BF16 mmproj): live transcription streams 22 deltas
before the commit and returns the fixture transcript; a spoken "What is the
capital of France?" receives "The capital of France is Paris. …" in text and
speech, and the server transcribes that speech back to the same words; a typed
follow-up about Italy is answered "The capital of Italy is Rome." in text; a
spoken "Set the volume to thirty, please." calls `set_volume` with
`{"volume": 30}` and, after the function output, the reply confirms it in
speech; reasoning text replies stream their reasoning before "The capital of
France is Paris.". Streamed words need the audio after them: a synthesized
question cut at its last word transcribes live without that word, in the FP32
reference too.
Native TTS of a two-sentence text transcribes back exactly with LFM3-Audio's
own ASR, and the spoken question "What is the capital of France?" receives
the interleaved answer "The capital of France is Paris. …", whose audio
transcribes to the same text.

Digital silence transcribes as "Okay." (Gufo and the FP32 reference agree);
the ASR suite records the silence transcript for LFM3-Audio instead of
requiring it to be empty, and still requires it to be identical when it runs
beside speech.

## Reproduction

```sh
M=models/lfm3-audio R=artifacts/lfm3-audio/reference
for c in asr depth codec tts live; do
  nix develop -c python3 -I tests/models/lfm3_audio/reference/lfm3_audio_ref.py $c \
    --backbone $M/LFM3-Audio-2B-BF16.gguf --mmproj $M/mmproj-LFM3-Audio-2B-BF16.gguf \
    --tokenizer $M/tokenizer.json --wav tests/functional/fixtures/audio/clear-english-voice.wav \
    --frames 40 --out $R/$c
done
cmake --build --preset gpu-test --target lfm3_audio_probe
build/gpu-test/tests/models/lfm3_audio/lfm3_audio_probe $M/LFM3-Audio-2B-BF16.gguf \
  $M/mmproj-LFM3-Audio-2B-BF16.gguf $R tests/functional/fixtures/audio/clear-english-voice.wav
nix develop -c python3 tests/functional/model_services.py tts --binary build/release/gufo \
  --model $M/LFM3-Audio-2B-Q8_0.gguf --mmproj $M/mmproj-LFM3-Audio-2B-BF16.gguf \
  --variant custom --voice female --asr-binary build/release/gufo \
  --asr-model models/Qwen3-ASR-1.7B --output artifacts/model-services/lfm3-audio-tts
```

`tokenizer.json` comes from LiquidAI/LFM3-Audio-2B at `cddb56b5`; the `tts`
fixture uses `--frames 80`. The reference needs about 16 GB of RAM.
