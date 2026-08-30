# Qwen3-TTS 12Hz 1.7B — gufo port

Status: **IMPLEMENTED** — CustomVoice, VoiceDesign, and Base.

A model-private HIP implementation of the Qwen3-TTS 12 Hz 1.7B family for
gfx1151. `gufo serve` serves every request from this implementation; the
upstream Python package is used only offline, as the oracle the tests compare
against. Decode runs at `31.0` ms per codec frame, or `2.55x` faster than
playback, with the talker projections at 98% of the measured DRAM roofline.

- [Models](#models)
- [Performance](#performance)
- [Quality](#quality)
- [Reproducing](#reproducing)
- [Appendix](#appendix)

## Models

The loaded `config.json` selects the variant, which fixes the advertised model
ID and the conditioning the endpoint requires. All three use
`POST /v1/audio/speech`.

| Checkpoint | Model ID | Conditioning | Required fields |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | `qwen3-tts-12hz-1.7b-customvoice` | named checkpoint speaker | `voice` |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `qwen3-tts-12hz-1.7b-voice-design` | natural-language voice description | `instruct` |
| `Qwen3-TTS-12Hz-1.7B-Base` | `qwen3-tts-12hz-1.7b-base` | reference audio, cloned | `reference_audio`, plus `reference_text` for ICL |

`voice_clone_mode: "speaker_embedding_only"` makes Base skip transcript and
speech-code conditioning. The 0.6B and 25 Hz checkpoints are rejected by
configuration validation and are not part of this implementation.

### Architecture

`Qwen3TTSForConditionalGeneration`:

* `talker` — 28-layer decoder: hidden 2048, FFN 6144, 16 heads / 8 KV heads,
  head_dim 128, RMSNorm eps 1e-6 with per-head q/k RMSNorm, 3D interleaved RoPE
  `mrope_section=[24,20,20]`, theta 1e6, GQA. `codec_embedding` 3072→2048,
  `text_embedding` 151936→2048, `text_projection` MLP 2048→2048→2048 (silu),
  `codec_head` → vocab 3072.
* `code_predictor` (MTP) — 5 layers, hidden 1024, 16/8 heads, vocab 2048.
  Predicts codebook slots 1..15 after the talker head produces slot 0.
* speech tokenizer v2 (12 Hz) — RVQ decoder, transformer decoder, conv upsample.
  Runs float32, matching the official float32 reference.

Only the prompt and encoder paths are variant-specific. CustomVoice injects the
configured speaker and language codec embeddings; VoiceDesign uses the
instruction-conditioned prompt with no fixed speaker; Base runs the native
speaker encoder for every clone mode, and ICL additionally tokenizes the
reference transcript and runs the native speech encoder to prepend reference
codec frames. A bounded single-entry cache reuses those reference features across
repeated requests.

### Sampling policy

The API defaults to the checkpoint's own policy: top-k 50 at temperature 0.9 for
both the talker and its code predictor, plus repetition penalty 1.05. `greedy:
true` exists for bounded deterministic diagnostics only — see
[greedy does not terminate](#greedy-does-not-terminate).

## Performance

Measured on gfx1151 with the release build, warm, for the canonical sentence at
seed 42.

| | Native | Official ROCm | Ratio |
|---|---|---|---|
| Canonical request | `9.05`–`9.16` s for 23.12 s of audio | `98.00` s for 21.84 s | `10.9x` throughput |
| Real-time factor | `2.55x` | `0.22x` | — |
| Per codec frame | `31.0` ms | — | — |

Per-variant endpoint latency:

| Variant | Audio | Request |
|---|---|---|
| CustomVoice | 23.12 s | `9.09` s |
| VoiceDesign | 6.96 s | `2.90` s |
| Base ICL | 5.20 s | `2.69` s, or `2.45` s reusing the reference clip |

### Where the frame goes

From a kernel trace of a bounded 20-frame request:

| Component | Per frame | Status |
|---|---|---|
| Talker + predictor GEMV | `22.7` ms | 98% of the `242` GB/s DRAM roofline |
| Speech-tokenizer decode | `2.2` ms | at the rocBLAS f32 ceiling |
| Other kernels | `3.1` ms | nine launches per layer |
| Inter-kernel gaps | `3.8` ms | `2.4` us per dispatch, hardware floor |

Each frame streams about `5.2` GB of BF16 weights — `2.7` GB for one pass over
the talker, `2.5` GB because the 15 code-predictor heads are sequentially
dependent and each re-reads all five predictor layers. At `242` GB/s that traffic
alone sets a floor near `21` ms, and every GEMV bucket now sits within 15% of its
own roofline. Going meaningfully faster requires moving fewer bytes, which means
trading weight precision, so it is deliberately not attempted. The two remaining
non-precision levers are a hand-written f32 GEMM for the decoder convolutions
(see [appendix C](#rocblas-f32-ceiling-on-gfx1151)) and sparse
attention.

## Quality

The upstream Python implementation never serves a request. It is an offline
oracle for tensor, logit, codec, waveform, and listening comparisons, pinned at
commit `022e286b`.

Sampled codec frames are not expected to be byte-identical, because PyTorch ROCm
and the native C++ sampler use different random-number generators. Parity is
therefore judged from exact tokens, tensor and logit metrics, near-exact decoder
waveform comparison, and an end-to-end intelligibility gate.

The intelligibility gate transcribes the native and official waveforms with the
same native HIP Qwen3-ASR-1.7B runtime and scores both against the requested
text. The migration was rerun on CustomVoice: native output matches the
requested text exactly, while the official ROCm waveform has a `3.92%` WER.

| Variant | Native WER | Official WER | Transcript LCS |
|---|---|---|---|
| CustomVoice | `0%` | `3.92%` | `97.29%` |

VoiceDesign and Base ICL retain their tensor, code, and waveform gates below;
their long-form intelligibility fixtures have not yet been rerun with the new
ASR backend.

### Gate metrics

| Variant | Metric | Measured | Gate |
|---|---|---|---|
| All | Talker prompt cosine | `1.0` | `> 0.99` |
| All | Prefill-logit cosine | `0.999938` | `> 0.95` |
| All | Cached-logit cosine | `0.99997` | `> 0.95` |
| All | Greedy argmax boundaries | match | exact |
| All | Predictor-logit cosine | `0.999539` | `> 0.95` |
| All | Decoder waveform MAE / max | `1.33e-7` / `1.28e-6` | `< 1e-5` / `< 1e-4` |
| CustomVoice | Qwen3-ASR WER / transcript LCS | `0%` / `97.29%` | `< 30%` / `> 75%` |
| VoiceDesign | Prompt cosine, semantic greedy tokens | `1.0`, 5/5 | `> 0.9999`, exact |
| Base | Prompt cosine, bounded greedy codes | `1.0`, 80/80 | `> 0.999`, exact |
| Base | Speaker-embedding cosine | `0.999997` | `> 0.9999` |
| Base | Speech-code agreement, aggregate | `0.577351` | `> 0.55` |
| Base | Speech-code agreement, groups 0 / 1 | `0.970` / `0.950` | `> 0.95` / `> 0.90` |
| Base | Speech reconstruction cosine / MAE | `0.939502` / `0.00827` | `> 0.93` / `< 0.02` |

### Known limitations

#### Run-to-run variation

Fixed-seed identity holds for the bounded eight-frame
tests but not for a full-length request: two identical requests in the same
process agree for a long prefix and then diverge. A 60-frame request diverges at
frame 38 in every pairing, with a peak difference of 21 to 80 of 32,768 full
scale — too small for a changed first-codebook token, so what flips is a deep
residual codec group whose logits already sit near the noise floor. Root cause
not found; ruled out are sampler amplification (greedy diverges identically),
kernel races (`AMD_SERIALIZE_KERNEL=3` changes nothing), and rocBLAS split-K
atomics (the talker pins `HIPBLAS_ATOMICS_NOT_ALLOWED` and the variation
survives). The talker is reproducible for eight frames across processes and the
decoder is reproducible at 5 and 101 frames, so it is neither in isolation. This
predates the variant work and applies equally to the previous release, so A/B
tests compare per-frame cost and gate metrics, never output hashes.

#### Greedy does not terminate

The model is sampling-trained. With both samplers
disabled it did not emit EOS before a 3,000-frame diagnostic cap, so a greedy
request on a four-second sentence returns 240 seconds of looping audio where
upstream stops at 262 frames. The checked-in greedy contract is therefore bounded
to five frames, and full-sentence quality is validated from the seeded sampled
artifact instead.

#### Speech-code agreement is bfloat16-limited, not wrong

`0.577351` aggregate agreement and `0.939502` reconstruction look alarming but are intrinsic: the
official implementation compared against *itself* at bf16 versus f32 agrees on
only `0.5514` of codes. See
[appendix D](#speech-encoder-code-agreement-is-bfloat16).

#### Long reference clips

The speech encoder's transformer is windowed, not fully
causal, and now receives `encoder_config.sliding_window` (250 frames) instead of
the sequence length. The retained 8.08-second fixture is 202 encoder frames, so
the window was inactive for it and no gate metric moved; clips longer than about
10 seconds were previously attending outside the official window. Gating the fix
needs a longer-clip oracle, which is not retained yet.

## Reproducing

```sh
nix build .#checks.x86_64-linux.pr        # canonical PR gate
nix develop -c ctest --preset hardware-full -R qwen3_tts
```

The intelligibility gate uses the native Qwen3-ASR-1.7B model under the
Hugging Face cache on the development host:

```sh
nix develop -c python3 tests/models/qwen3_tts/quality/compare_intelligibility.py \
  --reference-npy artifacts/qwen3_tts/rocm/waveform.npy \
  --candidate-wav <native.wav> \
  --asr-model /var/llms/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/<revision> \
  --gufo result/bin/gufo
```

Pass `--contract` with a `{"text": ...}` JSON file to score a variant against its
own sentence rather than the CustomVoice contract.

Regenerate the oracle artifacts, which land gitignored in
`artifacts/qwen3_tts/`:

```sh
nix develop --command tools/audio/run_ref_tts.sh           # sampled quality, seed 42
nix develop --command tools/audio/run_ref_tts.sh greedy    # five-frame exactness
nix develop --command tools/audio/run_ref_tts_quality.sh   # opt-in long-form fixture
```

| Artifact | Contents | Determinism |
|---|---|---|
| `talker_codes.npy` | full-sentence codec book, sampled, seed 42 | torch seed |
| `talker_codes_greedy.npy` | bounded five-frame codec book, argmax | fully deterministic |
| `waveform*.npy` | 24 kHz float32 audio | derived |
| `internals/*.npy` | prefill embeddings, layer boundaries, logits | fully deterministic |
| `reference_contract.json` | pinned hashes, shapes, provenance | checked in |

VoiceDesign and Base oracles live under `voice_design/` and `base/`; Base also
retains the reference WAV plus speaker- and speech-encoder intermediates.
`probe_speaker_encoder.py` and `probe_speech_encoder.py` capture per-stage
tensors at a chosen dtype, suffixing anything that is not bfloat16 so a float32
reference can sit beside the retained one.

Runtime switches, all defaulting off or to the fast path:

| Variable | Effect |
|---|---|
| `GUFO_QWEN3_TTS_WEIGHT_MODE=mapped` | memory-map decoder weights instead of copying them |
| `GUFO_QWEN3_TTS_PRECOMPUTE_SNAKE=1` | precompute SnakeBeta exponents (measured slower) |

## Appendix

### A. Provenance and licensing

- Official implementation: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
  pinned at
  [`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/commit/022e286b98fbec7e1e916cb940cdf532cd9f488e)
  (Apache-2.0).
- Checkpoints: [Qwen3-TTS collection](https://huggingface.co/collections/Qwen/qwen3-tts) —
  [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
  [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign),
  [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
- Architecture reference: [Qwen3-TTS Technical Report,
  arXiv:2601.15621](https://arxiv.org/abs/2601.15621).
- Parity methodology was inspected in
  [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) at
  [`54ea3d11ba4726f15d072b23a9244dc623f7cfc7`](https://github.com/0xShug0/audio.cpp/commit/54ea3d11ba4726f15d072b23a9244dc623f7cfc7)
  (Apache-2.0). This implementation stays model-private under this directory and
  is validated independently against the official Qwen oracle.
- The upstream optional Gradio demo selects
  [Source Sans Pro](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/cli/demo.py#L272)
  from Google Fonts with Arial fallbacks. The native server does not embed,
  download, or redistribute those fonts. Source Sans is maintained by
  [Adobe Fonts](https://github.com/adobe-fonts/source-sans) under the SIL Open
  Font License 1.1.
- The long-form listening fixture is fetched in bounded form from the
  [user-provided SDSU text source](https://dgoldberg.sdsu.edu/515/harrypotter.txt)
  and is not vendored here.

### B. Optimization history

Per-frame decode went `129.7` → `32.4` → `31.01` ms. Every step is an exact
rewrite: no gate metric in this file changed at any point.

Bandwidth pass (`129.7` → `32.4` ms). hipBLAS reached only 40-60 GB/s on decode
projections, because a single token is padded into a 128x32 macro tile with a
split-K reduction.

- Model-private GEMV, one output row per workgroup, eight wavefronts each walking
  the row with an independent coalesced stream. Sustains 208-242 GB/s, fp32
  accumulation in fixed order, and is *more* accurate than the batched path it
  replaced (`1.95e-4` versus `2.35e-4` maximum relative error against fp64).
- One weight pass shared across up to four input columns, so the two-token
  predictor step costs single-token traffic.
- Device-resident coarse-grained weight pages instead of host-registered ones
  (12% faster despite the shared physical pool), shifted so the safetensors
  payload lands 16-byte aligned and the widest packed loads are legal.
- q/k per-head RMSNorm, q/k rotation and V rounding in one launch; each residual
  add folded into the following RMSNorm.
- Top-k sampling partially sorts only the requested candidates, preserving
  comparator, candidate order, distribution, and RNG sequence.
- Repetition penalty via a per-vocabulary flag, replacing a scan of generated
  codes that made every frame cost more than the one before.

Dispatch pass (`32.4` → `31.01` ms). With the GEMV at its roofline, the residual
`3.8` ms per frame of `2.4` us dispatch gaps over ~1,440 launches was the only
non-precision lever. Per-layer launches fell 14 → 9, about 1,440 → 930 per frame:

- query/key/value share one input, so they run as one grouped GEMV; gate and up
  likewise. Each workgroup still reduces one row in the same fixed order.
- the fused q/k norm and rotation kernel also writes the attention cache, and the
  attention kernel writes its BF16 context directly, removing two launches per
  layer.
- a frame's 16 codec embedding lookups plus their summing pass became one batched
  launch accumulating in the same group order.

Decoder weight loading. Copying the 651 MiB shard into aligned device pages by
default cut median five-frame decode `33.19` → `30.37` ms (`8.5%`).

### C. Rejected experiments

Kept so they are not retried. All were implemented and measured.

| Experiment | Result |
|---|---|
| HIP graph replay | `2.03` us versus `1.82` us per dispatch; no end-to-end change. The gfx1151 dispatch floor is ~`2` us either way |
| GPU-side masked top-k | greedy replaces 15 of 16 host sorts per frame and gains only `0.7%`, so the whole host sampling path is ~`0.2` ms/frame |
| Precomputed SnakeBeta exponents | `33.32` versus `32.97` ms, a `1.0%` regression |
| hipBLASLt, direct rocBLAS, alternative attention routes | neutral, slower, or failed exactness |
| Decoder GEMM restructuring | at the library ceiling, detailed below |

#### rocBLAS f32 ceiling on gfx1151

The decoder convolutions run im2col plus an f32 GEMM at a flat `2.3-2.7` TFLOPS,
18% of the `14.8` TFLOPS vector-FMA peak. Nothing about the call shape explains
it:

| Variation | Result |
|---|---|
| `rocblas_sgemm` instead of `rocblas_gemm_ex` | 1.00x, byte-identical |
| Cache-blocking the time dimension, 512-8192 columns | 0.47-1.00x, byte-identical |
| Transposed orientation, long dimension as M | 0.99-1.04x |
| hipBLASLt across 32 heuristic algorithms | 0.84-1.01x |
| Square `sgemm` at 1024 / 2048 / 4096 | `2.63` / `2.70` / `2.60` TFLOPS |

A 4096-cubed square GEMM hitting the same `2.6` TFLOPS is decisive: this is the
library's f32 ceiling for this part, not a property of the decoder's shapes.
Beating it needs a hand-written kernel — a tiled f32 GEMM, or the three-pass BF16
split that emulates f32 through the `59` TFLOPS WMMA path, for which the waveform
gate has room (`1.33e-7` against a `1e-5` limit). Both are specialist work.

### D. Investigations

#### Speaker-encoder residual aliasing (fixed)

The native speaker embedding measured `0.986726` cosine against the official
encoder — a 9.35-degree angle in 2048 dimensions, or 16% relative L2, against a
port that reaches `1e-7` elsewhere.

Per-stage capture put the break at the first SE-Res2Net block: mel `0.999999`,
initial TDNN `0.999996`, then block 1 collapsing to `0.909117`. The cause was
buffer aliasing — `RunBlock` uses the `second` scratch buffer for its Res2Net
branch outputs, and the caller passed that same buffer as the first block's
input, so the branches overwrote the leading `rows * 64` floats of the residual
the block was about to add back, corrupting roughly an eighth of the frames. Only
the first block aliased, which is why the error appeared once and then persisted
instead of compounding.

Routing the initial TDNN through a buffer the branches never touch restores
`0.999997`, with every stage at or below the official implementation's own
bfloat16 noise (block 0 at `0.00281` relative L2 against a bf16-versus-f32 gap of
`0.00281`). `RunBlock` now rejects an aliasing input, and the gate moved from
`> 0.98`, which admitted the 16% error, to `> 0.9999`.

Two things were cleared along the way: the retained oracle's bfloat16 precision
is not a factor (float32 moves the official embedding by cosine `0.999998`), and
the mel front end matches `mel_spectrogram` exactly — Slaney scale and
normalization, periodic Hann, `(n_fft - hop) / 2` reflect padding,
`sqrt(|X|^2 + 1e-9)`, `log(clamp(x, 1e-5))`, filterbank agreeing to `1.9e-9`.

#### Speech-encoder code agreement is bfloat16

`0.577351` aggregate code agreement and `0.939502` reconstruction are not a
defect, and neither is a decoder number: the reconstruction figure decodes native
and official codes through the same decoder, itself exact to `1.33e-7`, so it
measures only how far the codes differ.

Comparing the official implementation against *itself* at bf16 versus f32
reproduces the native numbers at every one of twenty stages:

| Stage | Official bf16 vs f32 | Native vs official f32 |
|---|---|---|
| layer00, first convolution | `0.00274` | `0.00274` |
| layer01, first residual unit | `0.01123` | `0.01123` |
| layer08 | `0.05241` | `0.05242` |
| layer13 | `0.06692` | `0.06696` |
| convolutional | `0.06941` | `0.06939` |
| acoustic projection | `0.04047` | `0.04049` |

Agreement to three or four significant figures means the native encoder
reproduces the official bfloat16 computation, and the `7%` relative L2 against a
float32 reference is what bfloat16 costs this network. The apparent jumps at the
ELU layers are a saturating nonlinearity shrinking the signal — layer 12 has
standard deviation `22.0`, layer 13 has `2.33` — so the same absolute difference
reads as a larger relative one.

On codes, the official bfloat16 run agrees with its own float32 run on `0.5514`,
per group `0.98, 0.93, 0.80, 0.78, 0.68, ...` down to `0.29`. Native agreement is
`0.577351`, slightly *closer* to the float32 reference. A residual vector
quantizer is this sensitive by construction: each of sixteen stages picks the
nearest of 2,048 centroids in 256 dimensions and subtracts it, so a few percent
of feature difference flips decisions near Voronoi boundaries and compounds into
the next residual. The `> 0.55` gate sits just above the official
implementation's own self-agreement for that reason.

The remaining lever is precision, not correctness: the decoder already runs
float32, while the encoder emulates the official bfloat16 activations. Dropping
that emulation would move reference-audio codes closer to the float32 ideal for a
stage that runs once per unique clip, at the cost of no longer matching the
official implementation as shipped. That is a quality-policy decision and is not
taken here.

### E. Key ids (CustomVoice)

```
codec_eos=2150 codec_bos=2149 codec_pad=2148 codec_think=2154 codec_nothink=2155
codec_think_bos=2156 codec_think_eos=2157
tts_bos=151672 tts_eos=151673 tts_pad=151671 im_start=151644 im_end=151645
spk_id: serena=3066, vivian=3065, uncle_fu=3010, ryan=3061, aiden=2861,
        ono_anna=2873, sohee=2864, eric=2875, dylan=2878
lang codec ids: english=2050 chinese=2055 ... (see config.json)
```
