# EmbeddingGemma 2 quality

**On ten probe texts the GPU embeddings match sentence-transformers float32 output at cosine 0.999985 or better, and every token ID matches Hugging Face.**  Qualified October 23, 2026, on gfx1151, with the original BF16 checkpoint at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.  The cosines below are from the matrix-core attention build, measured on October 24, 2026.

| Check | Result |
| --- | --- |
| Token IDs versus Hugging Face `tokenizers` | 95/95 fixture cases identical, covering whitespace runs, byte fallback and special tokens |
| GPU embedding versus sentence-transformers float32 | Cosine 0.999985 worst of 10 probes, 0.999998 best |
| Same probes, Hugging Face BF16 versus float32 | Cosine 0.999867 worst, which is the checkpoint's own precision noise |
| Matryoshka 128 / 256 / 512 after renormalization | Worst cosine 0.999987 against the same truncation of the reference |
| GPU versus Gufo float32 CPU reference | Cosine 0.999992 or better on four sequences |
| Packing sequences in one batch versus alone | Cosine 0.999987 or better.  The tuned GEMM can choose a different algorithm for a different batch size |
| Windowed attention beyond the 512-token radius | A 2,349-token probe matches the reference at cosine 0.999990 |

The probes are English, Spanish, and Chinese queries and passages, code, emoji, whitespace runs, literal special-token text, and one long document.  They measure agreement with the reference implementation, not retrieval quality on a benchmark, and Gufo reports no MTEB score.  Activations stay float32 between BF16 matrix products, so Gufo is closer to the float32 reference than the model's BF16 execution is.

Google published the retrieval accuracy of this checkpoint on MTEB, code, and multilingual tasks, and Gufo did not re-measure it.  Image, video and audio qualification is below.  Quantized variants are unsupported and unmeasured.  The model card states a 1,024-token sliding window.  The checkpoint's configuration and the Hugging Face code use a radius of 512 positions on each side, or 1,025 keys, and Gufo follows them.  Hugging Face's `sliding_window_bidirectional_overlay` in `transformers/masking_utils.py` keeps keys where `abs(q_idx - kv_idx) <= sliding_window`, and `create_bidirectional_sliding_window_mask` passes `config.sliding_window` (512).  The `+ 1` in `modeling_embedding_gemma2.py` applies only to the flash-attention path.

## Reproduce

The [ROCm 10.1 migration check](ROCM.md) requalifies text, image and audio
against these independent fixtures on both pinned flake toolchains, with
unchanged bounds and qualified resident HTTP timing controls.

The [reference model](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) is pinned to the qualified revision.  The probes and their reference vectors are in [`tests/fixtures/embedding_gemma2`](../../../tests/fixtures/embedding_gemma2).  The fixture stores each reference value with five significant digits, which moves a cosine by less than 1e-6.  [`tools/models/embedding_gemma2/reference_embeddings.py`](../../../tools/models/embedding_gemma2/reference_embeddings.py) regenerates them with transformers that include `embedding_gemma2` and sentence-transformers 6.1.  Tests are in [`tests/models/embedding_gemma2`](../../../tests/models/embedding_gemma2), and the model package is [`src/models/embedding_gemma2`](../../../src/models/embedding_gemma2).

## Image and mixed-input parity

Requalified October 24, 2026, on gfx1151 after rebasing onto main `1fafd5ad`,
with the same pinned BF16 checkpoint and main's hipBLASLt text GEMMs.
Seven independent probes (eight images) cover upsampling, downsampling,
unchanged size, landscape/portrait/extreme aspect ratios, image plus text,
two images in one input, and Frigate's trailing newline.

| Check | Result |
| --- | --- |
| Resized patch RGB bytes versus official processor | SHA256 identical for every image, including the entire patch order |
| GPU / HTTP embeddings versus official float32 model | Worst cosine **0.999957**, best **0.999995** with matrix-core attention, against 0.999961 and 0.999996 before it; all exceed the retained 0.9999 bound |
| Mixed text/image batches and concurrent requests | Correct order, finite unit vectors and reference agreement |
| Image Matryoshka 128 / 256 / 512, base64 | All exceed 0.9999 cosine against the same reference prefix |
| Expanded context, image count, invalid destinations and malformed images | Rejected with 400; a subsequent text request succeeds |
| Existing text golden and OpenAI SDK checks | Pass on the candidate |

These are reference-parity probes using deterministic RGB patterns, rather than
a retrieval benchmark or a full Frigate installation. Matching the official
text, image, and mixed-input outputs preserves their shared vector space.
Existing llama.cpp image indexes must be re-embedded because its preprocessing
differs. llama.cpp's non-OpenAI `/embeddings` route is served too, with the
same preprocessing; see [Embeddings](../../SERVER.md#embeddings).

The oracle is [`reference_images.py`](../../../tools/models/embedding_gemma2/reference_images.py),
using official transformers commit `50e35cc0ca18ec19b80653690de553e6decf74f5`
(`5.21.0.dev0`), Torch 2.14.1, and the checkpoint's processor configuration.
It covers the default 280-token budget and, for three shapes each, the
processor's other `max_soft_tokens` budgets (70, 140, 560 and 1120). The
280-token cases kept their patch hashes and token IDs from the earlier Torch
2.12.0 run, and their reference values moved by at most 3.4e-7.
It loads the multimodal model in float32 with unused audio disabled and
mean-pools/L2-normalizes its outputs independently of Gufo. The script records
processor patch hashes, token IDs, full reference vectors and library versions
in [`image_golden.json`](../../../tests/fixtures/embedding_gemma2/image_golden.json).
The Torchvision uint8 bicubic path uses adaptive signed fixed-point filter
weights and clamps/rounds after each resize pass; replacing it with a single
floating-point cast changes pixels.

Reproduce the oracle with that transformers checkout and compatible Python
dependencies on `PYTHONPATH`:

```sh
python3 tools/models/embedding_gemma2/reference_images.py \
  --model models/embeddinggemma-2 \
  --transformers-revision 50e35cc0ca18ec19b80653690de553e6decf74f5 \
  --output tests/fixtures/embedding_gemma2/image_golden.json
python3 tools/models/embedding_gemma2/reference_image_boundaries.py \
  --model models/embeddinggemma-2 \
  --transformers-revision 50e35cc0ca18ec19b80653690de553e6decf74f5 \
  --output tests/fixtures/embedding_gemma2/image_boundaries.json
```

The processor-only boundary oracle records exact 8,192/8,193-token interleaved
inputs for every budget, plus 32/33 square images at budget 70 (2,114/2,180
tokens). It uses the official processor without a Gufo implementation or model
forward. These counts qualify expansion and the deliberate server image cap;
they are not additional final-vector reference fixtures.

`embedding_gemma2_image_processor_test` checks every resized byte without a GPU
or weights. `embedding_gemma2_image_golden_hip_test` additionally runs the native
vision/text encoders. [`embedding_images.py`](../../../tests/functional/embedding_images.py)
checks the HTTP path. Raw request evidence is retained with the
[review qualification record](artifacts/gfx1151-pr759-review.json).

Requalified October 29, 2026, after rebasing the image-budget PR onto video
support in main `a3edb56f`, on gfx1151 with the pinned ROCm 10.1 / GCC 16.2.0
toolchain. All 19 patch/vector fixtures pass the retained 0.9999 cosine bound;
the HTTP fixture minimum is 0.9999569. Four focused native tests pass without
skips (API, image processor, image HIP golden and video processor).

The final image HTTP suite passes 89 requests: all budgets with mixed batches,
128/256/512/768 dimensions, exact usage, interleaved 8,192/8,193-token inputs,
whole-image truncation, concurrent scratch growth/reuse, 32-small-budget-image
acceptance, 33-image rejection within and across inputs, malformed inputs and
recovery. Six video smoke requests retain independent joint multimodal parity
and verify that image-budget options leave the 140-token video default intact.
Both HTTP runs, source/binary/fixture hashes, native results and every timing
arm are retained in local review evidence.
The initial mixed-media timing flags remain inconclusive; the affected-history
follow-up and unchanged-main controls are documented in
[benchmarks](BENCHMARKS.md#image-budget-rebase-controls). This is Linux
qualification; Windows runtime checks remain the CI workflow's responsibility.

## Audio and interleaved input parity

Qualified October 26, 2026, on gfx1151. The checkpoint SHA256 is
`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`,
matching its pinned Hugging Face download metadata. The independent oracle is
[`reference_audio.py`](../../../tools/models/embedding_gemma2/reference_audio.py),
using transformers `50e35cc0ca18ec19b80653690de553e6decf74f5`, Torch 2.12.0,
and the official float32 model. Its decoder explicitly selects the official
`librosa` backend, rather than relying on optional backend discovery.

| Check | Result |
| --- | --- |
| Eight WAV/FLAC/MP3/Ogg decoder probes, mono/stereo and 8/16/24/44.1/48 kHz | Exact decoded/resampled float32 waveforms |
| Fifteen frontend probes, including 1/160/161 samples, odd lengths, partial frames and longer batch padding | Maximum absolute feature error 4.77e-7; every frame mask and valid-token count exact |
| Audio encoder and projector versus independent float32 states | Pass relative L2 < 5e-4; projector maximum observed relative L2 3.75e-6 |
| Ten final-vector probes: short clips, tone, silence, synthetic music/environmental bursts, packaged English speech, audio/text, two clips and audio/image/text | Worst cosine 0.9999687, above the retained 0.999 threshold |
| Native cancellation between audio layers and valid inference afterward | Pass |
| Real HTTP audio suite | 39 requests pass, including mixed text/audio batching, ordering, concurrency, dimensions/base64, expanded context, whole-media truncation, malformed data, refused destinations, server-observed cancellation and recovery |
| Existing text/image HTTP suites and generated-audio family smoke | Pass |
| Focused native tests and Nix hosted PR contracts | Five focused tests pass without skips; PR contracts pass |

Float32 stage fixtures retain all encoder/projector values for the four isolated
audio probes; all ten probes retain full final vectors and framing token IDs.
Feature and decoder fixtures are independent of the production C++ implementation.
These establish reference agreement, rather than speech recognition, audio
classification or retrieval benchmark accuracy. The environmental and music
probes are synthetic; the speech recording is the existing reviewed functional
fixture. Video qualification is below.

The [retained evidence](artifacts/gfx1151-2026-10-26-audio.json) records revision,
binary/harness/fixture hashes, every reference cosine, memory observations,
per-request latency and matched affected text/image controls. The first timing
pass overlapped CPU contract work and remains visibly inconclusive. Subsequent
idle main/candidate controls preserve byte-identical vectors and work counts;
every request and phase is within the 5% and 10 ms margins. No overall speedup
is claimed. Native Windows compilation is checked by CI; local numerical and
HTTP evidence here is Linux gfx1151 only.


## Video reference parity

Qualified on gfx1151 with checkpoint
`914f7f89142e33e77833254d9c9b90c3cef7303b` and the official transformers
implementation at `50e35cc0ca18ec19b80653690de553e6decf74f5`.
`tools/models/embedding_gemma2/reference_video.py` decodes retained encoded clips
with PyAV and runs the official processor on RGB arrays with explicit frame
count, average FPS and duration metadata. This also covers containers such as
WebM that omit PyAV's header frame count. Reference metadata retains the PyAV,
FFmpeg library, torch and Hugging Face Hub versions and attention backends.
The teacher uses float32; no Gufo code participates in fixture generation.

Thirteen probes cover sub-second and fractional-FPS clips, portrait and landscape,
32-frame and overflow boundaries, floating linspace rounding, uniform/truncate selection, timestamped
upsampling, repeated clips, embedded soundtracks, and joint text/video/image/audio.
All selected frame indices, unpadded patch shapes and resized patch bytes match
in the CPU test. HTTP expanded-token usage agrees with reference token counts;
the worst cosine across thirteen video/mixed vectors is 0.99997998 versus the teacher, with
finite normalized outputs. Existing image controls exceed 0.9999 and text/image
GPU golden tests pass with weights loaded. This establishes reference parity on
these fixtures, not a video retrieval benchmark.

The full focused video HTTP run retains 29 requests, including Matryoshka
128/256/512, concurrent/repeated requests, malformed and oversized media,
private/local URL refusal, pixel/duration/frame/request-work limits, context
overflow with and without truncation, cancellation correlated with its server
request ID, and successful recovery. Peak logged process RSS was 628 MiB in
that run; this is host RSS, not a GPU memory measurement. Raw reports, responses,
server logs and speed observations are retained under `artifacts/issue-761/`.

Reproduce the bounded smoke check with `model_services.py embedding` (see
[functional tests](../../../tests/functional/README.md)); use
`--embedding-suite full` for all modality suites. Generate goldens in the pinned
reference environment with:

```sh
PYTHONPATH=/path/to/pinned/transformers/src python3 \
  tools/models/embedding_gemma2/reference_video.py \
  --model models/embeddinggemma-2 \
  --output tests/fixtures/embedding_gemma2/video_golden.json
```

That source needs Hugging Face Hub >=1.31.0 (qualification used 1.31.0), in
addition to the reference tools in `nix develop`. Existing encoded fixtures are
reused and their SHA256 values are recorded. `--case NAME` generates only one
probe for focused investigation; a subset is not full qualification.


The [retained qualification evidence](artifacts/gfx1151-2026-10-29-video.json)
records revisions, checkpoint/binary/harness/fixture hashes, reference cosines,
per-request latency and memory observations. The linspace rounding regression
and affected overflow controls also pass focused HTTP replay after the review
fix. The final three-request smoke check takes about four seconds including
startup on this host. The standard 11-workload speed benchmark retains 60
requests per workload. All 36 resident text/image timing pairs and nine warm-up
pairs pass the per-request 5% and 10 ms thresholds against main `6feba906`, with
identical work counts. No aggregate speedup is claimed. Initial harness failures
and missing-model skips remain explicitly unqualified in the evidence record.
Three fresh-context review rounds found one sampling issue, which was fixed;
the last two rounds reported no findings.


### CI follow-up

The initial Windows package built and passed CPU tests, but its dependency
validator misclassified the Windows CNG `ncrypt.dll` imported by FFmpeg. Adding
that inbox DLL to the allowlist fixed the native Windows package job. Codec
DLLs remain mandatory. Tests also cover explicit manifests for libraries loaded
on demand, and the extracted release ZIP is checked with an external video CPU
harness without shipping test executables or fixtures to users.

Devlab additionally measured a consistent 18–24 ms process-init regression in
embedding, ASR and TTS: eager FFmpeg linkage initialized its transitive libraries
for every service. FFmpeg now loads on the first video request through an OS
library leaf. It is absent from the executable's startup dependency graph.
Three local matched startup pairs pass the 5%/10 ms thresholds; video CPU
fixtures, all 30 video HTTP requests and the owned multimodal smoke check pass.
The [CI follow-up evidence](artifacts/gfx1151-2026-10-29-video-ci.json) retains the
original flags, matched startup observations, binary/harness identity and checks.
Two additional independent reviews covered the fix; a release validation gap was
addressed and the final review reported no findings.
