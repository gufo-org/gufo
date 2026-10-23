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

Google published the retrieval accuracy of this checkpoint on MTEB, code, and multilingual tasks, and Gufo did not re-measure it.  Video and audio embeddings are unsupported; image qualification is below.  Quantized variants are unsupported and unmeasured.  The model card states a 1,024-token sliding window.  The checkpoint's configuration and the Hugging Face code use a radius of 512 positions on each side, or 1,025 keys, and Gufo follows them.  Hugging Face's `sliding_window_bidirectional_overlay` in `transformers/masking_utils.py` keeps keys where `abs(q_idx - kv_idx) <= sliding_window`, and `create_bidirectional_sliding_window_mask` passes `config.sliding_window` (512).  The `+ 1` in `modeling_embedding_gemma2.py` applies only to the flash-attention path.

## Reproduce

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
differs. The legacy `/embeddings` route is not served.

The oracle is [`reference_images.py`](../../../tools/models/embedding_gemma2/reference_images.py),
using official transformers commit `50e35cc0ca18ec19b80653690de553e6decf74f5`
(`5.21.0.dev0`), Torch 2.12.0, and the checkpoint's processor configuration.
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
```

`embedding_gemma2_image_processor_test` checks every resized byte without a GPU
or weights. `embedding_gemma2_image_golden_hip_test` additionally runs the native
vision/text encoders. [`embedding_images.py`](../../../tests/functional/embedding_images.py)
checks the HTTP path. Raw request evidence is retained with the
[review qualification record](artifacts/gfx1151-pr759-review.json).
