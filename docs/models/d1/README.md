# d1-Omni-2B decisions

[LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) answers
structured questions over text, JSON, pictures and audio without generating
text. Gufo implements the official GGUF interface's `noul` (P(yes)), `choice`
and ordinal `score` questions, plus `box` and `boxes` localization, including
spoken instructions. Questions are packed in sets of 16 by default.

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

## Boxes

Add exactly one picture and ask for one object or every instance:

```json
{
  "state": null,
  "images": ["data:image/png;base64,BASE64_PICTURE"],
  "questions": {
    "remote": {"type":"box", "instructions":"Where is the remote control on the left?"},
    "cats": {"type":"boxes", "instructions":"Every cat.", "max":4}
  }
}
```

A `box` answer contains `box`, `found` and `confidence`. Coordinates are
`top`, `bottom`, `left` and `right` fractions of the picture. An absent object
returns `box: null`. A `boxes` answer contains a sorted `boxes` list, `count`,
per-block `presence`, and `at_least` (true when every block read is present).
`max` reads 1–16 blocks; omitting it reads 16. Results follow left-to-right,
then top-to-bottom reading order.

Presence and confidence use LiquidAI's pinned calibration tables, embedded
because the GGUF does not contain typed-output calibration. Edges use medians
of the model's 0–999 thousandth distributions. Confidence estimates the share
of sampled rectangles with IoU at least 0.5 using the reference's fixed
512-point Halton sequence. Box questions require exactly one image; audio and
text may accompany it.

## Packed question sets

Requests pack up to 16 questions after their state in one forward pass. Override
this with `"set_size": N` (1–128); `1` retains the previous single-question
behavior. More questions create additional packs, including a partial final
pack. `usage.input_tokens` counts actual packed tokens, with the state once
per pack; no answer tokens are generated.

On full-attention layers, state tokens read the state and every question;
each question reads the state and its own question. Convolution and sliding
attention remain causal over the whole packed sequence. Consequently, changing
the questions or set size can change answers, as in LiquidAI's trained `set`
mode. Larger sets trade accuracy for throughput and must fit `--context`.

Requests remain serialized over one model session. Context defaults to 8192
and is bounded at 16384 tokens. Oversized packs return an error suggesting a
smaller set size or larger context; they are never truncated or converted to
causal chunks. Extraction (`text`, `number`, `date`), estimates, multiple choice,
video and speech generation remain unsupported. `gufo serve llm` directs
decision models to `gufo serve decision`.

See [quality](QUALITY.md), [measurements](BENCHMARKS.md) and
[implementation decisions](EXPERIMENTS.md).
