# EmbeddingGemma 2

EmbeddingGemma 2 turns text, images, audio and video into 768-dimensional
vectors in one shared space, for search, clustering and classification across
100+ languages and code. Gufo serves it through `POST /v1/embeddings`.

## Files

```sh
hf download google/embeddinggemma-2 \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --local-dir models/embeddinggemma-2
```

Only the BF16 safetensors checkpoint is supported, not GGUF or quantized files.

## Run

```sh
gufo serve embedding --model models/embeddinggemma-2
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["task: search result | query: What causes the northern lights?"]}'
```

`input` is a string or an array of up to 2,048 inputs. Vectors are normalized,
so cosine similarity and dot product give the same ranking. `encoding_format`
is `float` (the default) or `base64`.

Each input can hold up to 8,192 tokens. A longer one returns 400
`context_length_exceeded`, unless the request sets `"truncate": true` to keep
its first tokens. Send text, not token IDs: Gufo refuses token arrays, because
they come from another model's tokenizer.

## Task prefixes

The model was trained with short prefixes that say what a text is for. Gufo
adds none by default, so put them in the input:

| Use | Query | Document |
| --- | --- | --- |
| Search | `task: search result \| query: {text}` | `title: {title or none} \| text: {text}` |
| Question answering | `task: question answering \| query: {text}` | `title: {title or none} \| text: {text}` |
| Fact checking | `task: fact checking \| query: {text}` | `title: {title or none} \| text: {text}` |
| Code retrieval | `task: code retrieval \| query: {text}` | `title: {title or filename} \| text: {code}` |
| Classification | `task: classification \| query: {text}` | n/a |
| Clustering | `task: clustering \| query: {text}` | n/a |
| Similarity | `task: sentence similarity \| query: {text}` | n/a |

Inputs without a prefix still work, with lower retrieval quality. If a client
can't change its input, the server can add a prefix from the checkpoint's
list: `"prompt_name": "query"` (or `"document"`, `"Retrieval-query"`, ...) per
request, or `--default-prompt-name NAME` for text inputs of every request that
names none. Changing the prefix changes every vector, so rebuild the index.

## Images, audio and video

Each input can mix text with media parts, as in llama.cpp. One `content` list
produces one vector:

```json
{"model": "embeddinggemma-2", "input": [{"content": [
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  {"type": "input_audio", "input_audio": {"data": "<base64 WAV>"}},
  {"type": "input_video", "input_video": {"data": "<base64 MP4>"}},
  {"type": "text", "text": "near the entrance"}
]}]}
```

Media comes as base64, a data URL or a public HTTPS URL; Gufo refuses local
files and private network addresses. A video's own soundtrack is ignored; add
an `input_audio` part to embed sound.

| | Images | Audio | Video |
| --- | --- | --- | --- |
| Formats | PNG, JPEG, WebP | WAV, FLAC, MP3, Ogg | MP4, MOV, WebM, Matroska |
| Per item | 32 megapixels, 32,768 px per side | 30 s, 8 channels, 8–192 kHz | 300 s, 4,096 frames, 4 megapixels per frame |
| Per request | 32 images | 32 clips | 32 clips, 64 sampled frames |

All media in a request share 20 MiB and 15 seconds of downloading, and their
tokens count toward the 8,192-token limit. Truncation never cuts a media item
in half.

An image takes up to 280 tokens by default. `"mm_processor_kwargs":
{"max_soft_tokens": N}` with N of 70, 140, 280, 560 or 1120 trades detail for
speed and memory. Videos are sampled at 1 frame per second, up to 32 frames, at
140 tokens per frame; `video_options` changes `fps`, `max_frames`,
`overflow_strategy` (`uniform` or `truncate`), `add_timestamps` and
`max_soft_tokens`.

Gufo resizes images like the original model, not like llama.cpp: re-embed a
llama.cpp image index before switching.

## Dimensions

`dimensions` picks 128, 256, 512 or 768 (the default). Quality drops slowly
down to 256 and faster at 128, which Google recommends only for text. Vectors
of different sizes, or from different models, aren't comparable.

## Clients

OpenAI embeddings clients work once they send strings and the right model ID:

- **LangChain** sends token arrays by default. Set
  `check_embedding_ctx_length=False` on `OpenAIEmbeddings`.
- **paperless-ngx** asks for `text-embedding-3-small` unless
  `PAPERLESS_AI_LLM_EMBEDDING_MODEL=embeddinggemma-2` is set, with
  `PAPERLESS_AI_LLM_EMBEDDING_BACKEND=openai-like`.
- **Frigate** 0.19 sends the typed input above; 0.18 uses llama.cpp's
  `POST /embeddings`, which Gufo also serves.

A request for another model ID returns 404 `model_not_found`;
`--served-model-name` adds names the server answers to.

## Performance

As of October 23, 2026, on Strix Halo, Gufo embeds up to **41.5k tokens/s**
over HTTP and answers a 404-token request in **13.0 ms**. Loading takes about
2 seconds. See the [full benchmarks](../internal/models/embeddinggemma-2/BENCHMARKS.md)
and the [quality report](../internal/models/embeddinggemma-2/QUALITY.md).
