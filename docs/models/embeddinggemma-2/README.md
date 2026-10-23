# EmbeddingGemma 2

Native BF16 safetensors text embeddings on gfx1151.  Gufo runs the text backbone of [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), a 768-dimensional embedding model for 100+ languages and code, through `POST /v1/embeddings`.  The checkpoint also carries image, video, and audio encoders.  Gufo does not load them, and those inputs are unsupported.  Python runs only in offline reference checks.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

```sh
nix develop -c hf download google/embeddinggemma-2 \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --local-dir models/embeddinggemma-2
nix build
./result/bin/gufo serve embedding --model models/embeddinggemma-2
```

```sh
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["task: search result | query: What causes the northern lights?"]}'
```

The server returns L2-normalized mean-pooled vectors, so cosine similarity and dot product agree.  `input` is a string or an array of up to 2,048 strings, each at most 8,192 tokens including the `<bos>` and `<eos>` the tokenizer adds.  The server rejects token-array input because the vocabulary is the model's own.  `encoding_format` is `float` by default or `base64`.

## Dimensions

`dimensions` selects a Matryoshka size of 128, 256, 512, or 768, and defaults to 768.  Gufo keeps the first components of the full vector and renormalizes.  Quality falls slowly down to 256 and faster at 128, which Google recommends only for text.  Vectors of different sizes, or from different models, are not comparable.  Changing either requires re-embedding an index.

## Task prefixes

The model is trained with short prefixes, and Gufo does not add them.  Send the prefix as part of the input:

| Use | Query | Document |
| --- | --- | --- |
| Search | `task: search result \| query: {text}` | `title: {title or none} \| text: {text}` |
| Question answering | `task: question answering \| query: {text}` | `title: {title or none} \| text: {text}` |
| Fact checking | `task: fact checking \| query: {text}` | `title: {title or none} \| text: {text}` |
| Code retrieval | `task: code retrieval \| query: {text}` | `title: {title or filename} \| text: {code}` |
| Classification | `task: classification \| query: {text}` | n/a |
| Clustering | `task: clustering \| query: {text}` | n/a |
| Similarity | `task: sentence similarity \| query: {text}` | n/a |

Symmetric tasks use the same prefix on every input.  Omitting the prefixes still works but lowers retrieval quality.

## Requests

Loading uploads the 540 MB text backbone, unmaps the checkpoint so the weights stay only in device memory, and then runs a short warm-up.  The warm-up times each hipBLASLt algorithm that needs no scratch space for each matrix shape and row bucket the encoder uses and keeps the fastest, so the first request pays no tuning cost.  The choice can differ between loads, which changes float32 accumulation order but not the result beyond rounding.

Sequences from one request or from concurrent requests run one device batch at a time.  A batch packs up to 8,192 tokens, and a single request may span more than one batch.  Concurrent requests take turns in arrival order, one batch per turn, so a short request waits for at most one batch of a long one.  A device failure that a probe confirms answers 503 `device_lost` and shuts the server down for its supervisor to restart, as the text server does.  Attention is bidirectional, so there is no cache to reuse between requests.  Gufo checks for cancellation once the request reaches the device and between its device batches.  `--served-model-name` sets the public model ID.  Gufo supports only BF16 safetensors, not GGUF or quantized variants.

See [server limits](../../SERVER.md) and [quality gaps](QUALITY.md).
