# LFM3-VL 2B

Native HIP inference for LiquidAI LFM3-VL-2B on Strix Halo `gfx1151`.
Text and PNG/JPEG/WebP images produce generated text, reasoning, structured
JSON or native tool calls. Images can be interleaved with text and other
images in user messages. Use `gufo prompt`, `gufo chat`, or the Chat
Completions, Responses and Messages HTTP APIs.

## Weights

Download the official GGUF pair at
[revision b36554cf](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/tree/b36554cfd35854b06429900131730cd80f092cdb):

```sh
hf download LiquidAI/LFM3-VL-2B-GGUF \
  LFM3-VL-2B-Q8_0.gguf mmproj-LFM3-VL-2B-BF16.gguf \
  --revision b36554cfd35854b06429900131730cd80f092cdb \
  --local-dir models/lfm3-vl
```

Q8_0 is the qualified backbone format; BF16/F16 use the existing LFM3
backbone paths. Numerical evidence covers Q8_0 plus the BF16 projector.
Q4/Q5/Q6 are unsupported. Supply the matching `--mmproj` explicitly; the
loader rejects mismatched model families. Production Gufo needs neither
Python nor llama.cpp.

## Use

```sh
gufo prompt --model models/lfm3-vl/LFM3-VL-2B-Q8_0.gguf \
  --mmproj models/lfm3-vl/mmproj-LFM3-VL-2B-BF16.gguf \
  --image photo.png --think off --max-tokens 128 \
  --prompt 'Describe this picture briefly.'

gufo serve llm --model models/lfm3-vl/LFM3-VL-2B-Q8_0.gguf \
  --mmproj models/lfm3-vl/mmproj-LFM3-VL-2B-BF16.gguf \
  --context 16384 --sessions 2 --speculative off --port 8080
```

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "LFM3-VL-2B",
    "temperature": 0,
    "reasoning_effort": "none",
    "max_completion_tokens": 128,
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://placecats.com/300/200"}},
      {"type": "text", "text": "Describe this image."}
    ]}]
  }'
```

HTTP image URLs may be base64 data URLs or public HTTPS. Standard shared
image size, body and network limits apply. `image_url.detail: "low"` fits the
image within 512 pixels per side where the model's minimum area permits;
`auto`/`high` use native sizing. `--image` attaches each file before the first
user message's text. Interactive chat retains those images in its history.

Thinking is enabled by default. `--think off` or HTTP
`reasoning_effort: "none"` selects direct answers. Positive efforts enable
thinking; `reasoning_budget_tokens` can bound it. Tools and JSON use the
existing LFM3 native constraint and reasoning paths.

## Context and implementation

The native HTTP context is 131072 slots, including images and generated
text; `--context` can lower it. CLI prompt/chat reserve 8192 slots. These are
capacity limits, not full-depth quality or performance qualifications.
Images use the official smart resize, antialiased uint8 bicubic filtering,
16-pixel patches, temporal repetition, and 2×2 spatial merging. The native
image pixel budget is 65536–16777216 pixels.

The LFM3 backbone remains causal. Its vision encoder is shared with d1,
which retains its own smaller pixel budget and decision semantics. Image
embeddings replace only their own input rows; ordinary text and subsequent
batched decoding use the shared LFM3 sessions. Cache identities cover the
projector, processed patches, geometry and placement. RAM prefixes and disk
snapshots therefore distinguish different pixels even when image token IDs
are identical. Counting image tokens performs preprocessing without GPU
encoding.

Video ingestion, image content in non-user roles, speculative decoding and
CPU generation are not supported here. The checkpoint supports video
upstream, but this integration covers text and still images.
See [quality and remaining scope](QUALITY.md) and
[measurement scope](BENCHMARKS.md).
