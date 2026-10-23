# EmbeddingGemma 2 quality

**On ten probe texts the GPU embeddings match sentence-transformers float32 output at cosine 0.999981 or better, and every token ID matches Hugging Face.**  Qualified October 23, 2026, on gfx1151, with the original BF16 checkpoint at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.

| Check | Result |
| --- | --- |
| Token IDs versus Hugging Face `tokenizers` | 95/95 fixture cases identical, covering whitespace runs, byte fallback and special tokens |
| GPU embedding versus sentence-transformers float32 | Cosine 0.999981 worst of 10 probes, 0.999999 best |
| Same probes, Hugging Face BF16 versus float32 | Cosine 0.999867 worst, which is the checkpoint's own precision noise |
| Matryoshka 128 / 256 / 512 after renormalization | Worst cosine 0.999982 against the same truncation of the reference |
| GPU versus Gufo float32 CPU reference | Cosine 0.999992 or better on four sequences |
| Packing sequences in one batch versus alone | Cosine 0.999992 or better.  The tuned GEMM can choose a different algorithm for a different batch size |
| Windowed attention beyond the 512-token radius | A 2,349-token probe matches the reference at cosine 0.999994 |

The probes are English, Spanish, and Chinese queries and passages, code, emoji, whitespace runs, literal special-token text, and one long document.  They measure agreement with the reference implementation, not retrieval quality on a benchmark, and Gufo reports no MTEB score.  Activations stay float32 between BF16 matrix products, so Gufo is closer to the float32 reference than the model's BF16 execution is.

Google published the retrieval accuracy of this checkpoint on MTEB, code, and multilingual tasks, and Gufo did not re-measure it.  Image, video, and audio embeddings are unsupported.  Quantized variants are unsupported and unmeasured.  The model card states a 1,024-token sliding window.  The checkpoint's configuration and the Hugging Face code use a radius of 512 positions on each side, or 1,025 keys, and Gufo follows them.  Hugging Face's `sliding_window_bidirectional_overlay` in `transformers/masking_utils.py` keeps keys where `abs(q_idx - kv_idx) <= sliding_window`, and `create_bidirectional_sliding_window_mask` passes `config.sliding_window` (512).  The `+ 1` in `modeling_embedding_gemma2.py` applies only to the flash-attention path.

## Reproduce

The [reference model](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) is pinned to the qualified revision.  The probes and their reference vectors are in [`tests/fixtures/embedding_gemma2`](../../../tests/fixtures/embedding_gemma2).  The fixture stores each reference value with five significant digits, which moves a cosine by less than 1e-6.  [`tools/models/embedding_gemma2/reference_embeddings.py`](../../../tools/models/embedding_gemma2/reference_embeddings.py) regenerates them with transformers that include `embedding_gemma2` and sentence-transformers 6.1.  Tests are in [`tests/models/embedding_gemma2`](../../../tests/models/embedding_gemma2), and the model package is [`src/models/embedding_gemma2`](../../../src/models/embedding_gemma2).
