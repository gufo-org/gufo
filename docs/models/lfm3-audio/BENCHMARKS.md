# LFM3-Audio benchmarks

October 29, 2026, Strix Halo `gfx1151`, release build of `3c46e082`
(GCC 16.2, ROCm 10.1 HIP Clang 24). Weights: LiquidAI/LFM3-Audio-2B-GGUF
`02355c1a`, Q8_0 backbone. One process per measurement, so first requests
include lazy kernel loading. Single stream; no other model process loaded.

## Single stream

| Workload (Q8_0 backbone) | BF16 mmproj | Q8_0 mmproj |
| --- | ---: | ---: |
| ASR, 10.0 s English clip, mean of 5 after 1 warmup | 292.3 ms (**34.2×** real time) | 280.5 ms (35.7×) |
| ASR audio encoder (mel + towers + projector) | 17.7 ms | 6.2 ms |
| TTS, 105-character sentence, seeds 1–3 | **6.44–6.48×** real time | 7.06–7.08× |
| TTS time to first audio (prompt to first 80 ms frame) | **37.5–37.9 ms** | 46.3–47.0 ms |
| TTS backbone step per frame | 8.60–8.64 ms | 8.48–8.50 ms |
| TTS depth decoder per frame (3 MaskGIT passes, sampling) | 3.51–3.54 ms | 2.54–2.55 ms |
| Voice chat, 1.9 s spoken question: first text / first audio | 77.3 / 148.2 ms | 74.2 / 143.7 ms |

Commands (`$M` backbone, `$P` mmproj):

```sh
build/release/gufo transcribe --model $M --mmproj $P --language en \
  --repeat 5 --warmup 1 --format json tests/functional/fixtures/audio/clear-english-voice.wav
build/release/gufo speak --model $M --mmproj $P --seed 1 --format json \
  --output tts.wav --text 'The quick brown fox jumps over the lazy dog. Then it runs back home through the forest, happy and tired.'
build/release/gufo speak --model $M --mmproj $P --mode chat --audio question.wav \
  --output reply.wav --format json
```

The interleaved reply starts with six text tokens, so its first audio follows
the first text block.

## Ceiling

Every speech frame needs one backbone step, then the depth decoder, strictly in
sequence (the next step consumes the frame). The Q8_0 backbone reads about
1.55 GB of weights per step; its projection kernels already stream at about
220–240 GB/s, close to the measured ~240 GB/s large-buffer ceiling, so one
step cannot drop much below 6.6 ms. A single stream is therefore bounded near
**10× real time** with the depth decoder at its own bandwidth floor
(about 1 ms of BF16 weights per frame); 50× is not reachable for one stream on
this memory system. Remaining measurable overheads are launch gaps in the
backbone step (about 1 ms) and the depth decoder's 12-row projections, which
reach about 80 GB/s (see [experiments](EXPERIMENTS.md)). Concurrent streams
sharing backbone steps are not implemented.
