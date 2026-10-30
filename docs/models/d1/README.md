# d1-Omni-2B decisions

[LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) answers
structured questions over text, JSON, pictures and audio without generating
text. Gufo implements the official GGUF interface's `noul` (P(yes)), `choice`
and ordinal `score` questions, including spoken instructions.

Download the backbone and projector from
[LiquidAI/d1-Omni-2B-GGUF](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF/tree/c53ae0b1f92b17065e9158d34019c9ca1ea5bcf8):

- `d1-Omni-2B-Q8_0.gguf`
- `mmproj-d1-Omni-2B-BF16.gguf`

The loader also accepts F16/BF16 weights. Numerical and performance evidence
here covers the Q8_0 backbone and BF16 projector on gfx1151. The projector is
optional for text-only requests. This is a native HIP implementation; the
production package does not link llama.cpp or require Python.

## Serve

```sh
gufo serve decision \
  --model d1-Omni-2B-Q8_0.gguf \
  --mmproj mmproj-d1-Omni-2B-BF16.gguf \
  --context 8192 --port 8080

curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "state": "I was charged twice. Please refund one charge.",
    "questions": {
      "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
      "team": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges and refunds", "shipping": "Delivery"}},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["Can wait", "Today", "Blocking the customer now"]}
    }
  }'
```

Responses contain named `answers` and `usage` with `output_tokens: 0`.
The GGUF's per-type temperatures calibrate the max-pooled vocabulary scores.
A score selects the median level; its confidence is the largest level
probability, matching the reference. `/ready` and `/v1/models` advertise the
loaded decision service. `--api-key`, connection limits and request body limits
use the shared HTTP server controls.

For pictures, add `"images": ["data:image/png;base64,..."]`. For audio, add
`"files": ["data:audio/wav;base64,..."]`. These fields also accept public HTTPS
URLs through the bounded media reader. When both are present, files precede
images, matching the reference. `"state": null` makes the media the whole state.

Alternatively, `media` lists ordered objects such as
`{"type":"image","url":"data:image/png;base64,..."}` or
`{"type":"audio","url":"data:audio/wav;base64,..."}`. Named questions may use
spoken instructions:

```json
{"type":"noul","instructions":{"type":"input_audio","input_audio":{"data":"BASE64_WAV","format":"wav"}}}
```

PNG, JPEG and WebP images use RGB normalization, antialiased bicubic resizing,
16-pixel patches and 2×2 spatial merging. Images are bounded at 1,638,400 pixels
before patchification. Audio uses mono WAV decoding, resampling to 16 kHz and
the causal LFM2 audio encoder. Each clip is limited to 300 seconds and all media
must fit the complete prompt's context.

## CLI

```sh
gufo decide --model d1-Omni-2B-Q8_0.gguf \
  --mmproj mmproj-d1-Omni-2B-BF16.gguf --request request.json

# Keep the model loaded across requests; one JSON object per input line.
gufo decide --model d1-Omni-2B-Q8_0.gguf --jsonl < requests.jsonl
```

Without `--request`, the CLI reads standard input. Answers go to standard
output; inference duration goes to standard error. Media use the same URL/data
interface as HTTP, rather than reading filesystem paths from request JSON.

## Scope

Each question runs as its own complete prompt (`set_size=1`). Full attention
is bidirectional; short convolution and sliding attention stay causal. The
state is read again for each question. Requests are serialized over one model
session. Context defaults to 8192 and is bounded at 16384 tokens; oversized
prompts are rejected instead of truncated or processed as causal chunks.

Extraction (`text`, `number`, `date`), estimates, multiple choice, boxes, video,
packed question sets and speech generation are not implemented. These typed
outputs also remain outside the upstream GGUF interface documented at the
pinned model revision. `gufo serve llm` directs decision models to
`gufo serve decision`.

See [quality](QUALITY.md), [measurements](BENCHMARKS.md) and
[implementation decisions](EXPERIMENTS.md).
