# EmbeddingGemma 2

Native BF16 safetensors text and image embeddings on gfx1151. Gufo runs [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), a 768-dimensional embedding model for 100+ languages and code, through `POST /v1/embeddings`. Text and images share one vector space. Audio and video inputs remain unsupported. Python runs only in offline reference checks.

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

The server returns L2-normalized mean-pooled vectors, so cosine similarity and dot product agree.  `input` is a string or an array of up to 2,048 strings or typed content objects, each at most 8,192 tokens including the `<bos>` and `<eos>` the tokenizer adds.  The server rejects token-array input because the vocabulary is the model's own.  `encoding_format` is `float` by default or `base64`.

## Images

Use llama.cpp's typed input form. Each `content` object produces one embedding;
strings and content objects may be mixed in the same request. Text parts are
concatenated in order, including whitespace; the server adds no task prefix.

```json
{
  "model": "embeddinggemma-2",
  "input": [
    {"content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "\n"}
    ]},
    {"content": [{"type": "text", "text": "task: search result | query: a red car"}]}
  ],
  "encoding_format": "float"
}
```

This is the request form used by Frigate 0.19's development client. The legacy
`POST /embeddings` endpoint used by Frigate 0.18 is not served.

Images may be PNG, JPEG, or WebP, supplied as base64 data URLs or public HTTPS
URLs. Shared [image read limits](../../SERVER.md#embeddings) apply: at most 20 MiB
of encoded images across a request, 32 megapixels per decoded image, and a
15-second read budget. Embedding images additionally have a 32,768-pixel side
limit and a maximum of 32 images per request. The HTTP request-body limit also
applies to base64. Private destinations and local files are refused.

Preprocessing follows the pinned checkpoint's `Gemma4ImageProcessor`: RGB,
antialiased uint8 bicubic resize preserving aspect ratio, no crop, rescale by
1/255, and 16×16 patches. The 3×3 spatial pool yields at most 280 soft tokens
per image; the actual count depends on aspect ratio. These tokens and their
boundary markers count toward the 8,192-token input limit and reported usage.
Text parts containing literal `<|image|>` placeholders are refused when images
are present. Unsupported processor or vision configurations are refused.

Gufo's reference-matched resize differs from llama.cpp's preprocessing.
Re-embed an existing llama.cpp image index before switching servers; the
vectors are not interchangeable. [Quality](QUALITY.md) records agreement with
the official float32 model, rather than a retrieval benchmark or full Frigate
integration test.

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

The first image request uploads the vision tower and projector; subsequent requests reuse them. Audio weights are never uploaded.

Sequences from one request or from concurrent requests run one device batch at a time.  A batch packs up to 8,192 tokens, and a single request may span more than one batch.  Concurrent requests take turns in arrival order, one batch per turn, so a short request waits for at most one batch of a long one.  A device failure that a probe confirms answers 503 `device_lost` and shuts the server down for its supervisor to restart, as the text server does.  Attention is bidirectional, so there is no cache to reuse between requests.  Gufo checks for cancellation once the request reaches the device and between its device batches.  `--served-model-name` sets the public model ID.  Gufo supports only BF16 safetensors, not GGUF or quantized variants.

See [server limits](../../SERVER.md) and [quality gaps](QUALITY.md).
