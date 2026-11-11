# MiniMax H3

MiniMax H3 generates short videos with sound from a text prompt. Gufo runs its
text-to-video-and-audio path, FL2VA, from BF16 weights and writes an MP4.

It can also start from a first frame, end on a last frame, or both. Image
references, reference videos (Ref2VA) and 2K regeneration aren't supported.

## Files

> **Warning:** You obtain the checkpoint and accept the MiniMax H3 Community
> License Agreement yourself. Gufo ships no H3 weights or tokenizer, and never
> downloads the model. See the
> [third-party notices](../../THIRD_PARTY_NOTICES.md).

| Item | Pinned identity |
| --- | --- |
| MiniMaxAI/MiniMax-H3, FL2VA | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Source manifest SHA-256 | `00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89` |
| License, August 2, 2026, SHA-256 | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |

After you have access, download the FL2VA files and verify them:

```sh
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include LICENSE README.md model_index.json "FL2VA/**" \
  --local-dir /var/llms/huggingface/MiniMax-H3
python3 tools/h3/gufo-h3-manifest.py \
  --model-root /var/llms/huggingface/MiniMax-H3 \
  --verify src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
```

The check runs from a Gufo source checkout and executes no code from the
download. Gufo also checks the files when it loads them and refuses missing,
partial or modified files.

## Presets

| Preset | Output | Speed and quality |
| --- | --- | --- |
| `exact` | 512 × 512 | The released schedule, 50 steps. |
| `exact-1344x768` | 1344 × 768 | The released schedule at full resolution. |
| `fast` | 512 × 512 | 20 steps at a smaller internal size, skipping some work; approximate. |
| `aggressive` | 512 × 512 | Like `fast`, skipping more; approximate. |
| `dev` | 256 × 256 frames | Five steps and three frames, no audio, for checking a setup. |

## Run

Check a setup with the `dev` preset, which writes three frames:

```sh
gufo video --model /var/llms/huggingface/MiniMax-H3 \
  --preset dev --seed 42 --frames-dir /tmp/h3-preview --profile \
  "A red fox walking through snow"
```

For a full video, use `--preset exact --output /tmp/h3.mp4`.

Start a video server:

```sh
gufo serve --host 127.0.0.1 --port 8080 video \
  --model /var/llms/huggingface/MiniMax-H3 \
  --root /var/llms/huggingface/gufo-h3-jobs --ttl 3600
```

`POST /v1/videos` takes JSON or multipart form data:

```json
{
  "model": "minimax-h3",
  "prompt": "A red fox walking through snow",
  "size": "512x512",
  "seconds": 5,
  "gufo": {"seed": 42}
}
```

## First and last frames

Pass `--first-frame FILE`, `--last-frame FILE` or both to `gufo video`, or
`first_frame` and `last_frame` to `POST /v1/videos` as multipart file parts or
base64 `data:` URLs. The server never fetches remote URLs. PNG, JPEG and WebP
work.

The first frame (or a lone last frame) sets the video's aspect ratio, between
1:4 and 4:1; the preset sets its size. A 1664 × 928 frame gives a 1376 × 768
video with `exact-1344x768` and 672 × 384 with `exact`. A last frame next to a
first one is cropped to fit. With `dev`, keyframes far from square are refused
because the video would be too small. `fast` and `aggressive` follow the frames less
closely than `exact`.

## Limits

- `size` is `512x512` or `1344x768`, at 24 frames per second. `seconds` is 1,
  or 5 to 14.375; Gufo rounds up to the frame counts the model supports (5
  seconds gives 124 frames).
- Finished videos are deleted after `--ttl` seconds, 3600 by default.
- The prompt can be up to 4,096 tokens.
- Over HTTP, `dev` needs `size: "256x256"` and `gufo.output_format: "ppm"`,
  and returns one frame.
- The server runs one video at a time and queues one more; further requests
  get HTTP 429.

Video jobs run in the background. The [server guide](../server.md#videos)
explains status, progress, downloads, cancellation and expiry.
`DELETE /v1/videos/{id}` cancels a job at any stage and deletes its files.
Saved job status keeps a hash of the prompt, never the prompt itself.

## Performance

As of November 2026, complete-video timings are still being measured. See the
[benchmarks and memory notes](../internal/models/minimax-h3/BENCHMARKS.md) and
the [quality report](../internal/models/minimax-h3/QUALITY.md).
