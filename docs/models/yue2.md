# YuE2 3B

YuE2 writes a song from a style description and lyrics: it plans an ABC score,
then generates 48 kHz stereo audio that follows it. You can edit the score and
generate again.

## Licence

The YuE2 code is Apache-2.0, but the weights are **CC BY-NC 4.0** with an
extra permission for individual creators: individuals may monetise the songs
they make; companies need a commercial licence from HKUST / M-A-P. Gufo doesn't
ship the weights.

## Files

YuE2 needs two directories: the model and an audio decoder. `YuE2-Vae` is the
decoder for listening; `YuE2-Vae-legacy` is the one behind the published
WildSongBench scores.

```sh
gufo model install yue2-3b          # or yue2-3b-legacy
# or by hand:
hf download m-a-p/YuE2-3B --revision 8f24312f187763b9854adeec87e437e2b03bbff1 \
  --exclude "assets/*" --exclude "*.whl" --local-dir models/YuE2-3B
hf download m-a-p/YuE2-Vae --revision 152733a19ad43aa67e367f9b5503ef8075bb5126 \
  --exclude "assets/*" --local-dir models/YuE2-Vae
```

Gufo finds `YuE2-Vae` next to the model directory, or a saved pipeline
directory (`pipeline.json` with both). Pass `--vae-model DIR` to choose
another decoder.

## Run

```sh
gufo music --model models/YuE2-3B \
  --prompt "English, warm piano pop, expressive female voice, acoustic piano, 88 BPM" \
  --lyrics-file lyrics.txt --output song.wav --score-output song.abc
```

Write lyrics with section tags such as `[Verse]` and `[Chorus]`; leave them out
for an instrumental. The model chooses the length and stops on its own, at most
after 360 s. `--seconds S` sets a shorter maximum. There are no tempo, key or
language fields: put them in the prompt or in the score.

| Option | Meaning |
| --- | --- |
| `--planning full\|melody\|off` | Plan a score with chords (default), a melody only, or no score |
| `--score PATH` | Play this ABC score instead of planning one |
| `--plan-only` | Write the score (`--score-output`) and stop |
| `--guidance G` | Classifier-free guidance; default 1 (off), 1.01 with `--planning off` |
| `--steps N` | Acoustic solver steps (default 32) |
| `--seed N` | Seed (default 831001) |

To edit a song, run with `--plan-only`, edit the ABC file, then run again with
`--score`.

## Serve

```sh
gufo serve music --model models/YuE2-3B --root music-jobs
```

Songs take minutes, so the server runs them as jobs, like videos:

1. `POST /v1/audio/music` creates a job and returns its `id` (202).
2. `GET /v1/audio/music/{id}` reports its status, phase and progress.
3. `GET /v1/audio/music/{id}/content` downloads the WAV, with range requests.
4. `GET /v1/audio/music/{id}/score` downloads the ABC score.
5. `DELETE /v1/audio/music/{id}` cancels the job and deletes its files.

```json
{
  "model": "YuE2-3B",
  "prompt": "English, warm piano pop, expressive female voice, 88 BPM",
  "lyrics": "[Verse]\n...\n[Chorus]\n...",
  "seconds": null,
  "response_format": "wav",
  "gufo": {
    "seed": 42,
    "score": {"format": "abc", "text": "X:1\n..."},
    "plan_only": false,
    "guidance_scale": null,
    "steps": null,
    "options": {"planning": "full"}
  }
}
```

Without `gufo.seed` the server uses 831001, the same default as `gufo music`.
`response_format` is `wav` (PCM16) or `wav_float`. `gufo.options` also takes
`abc_sampling` and `semantic_sampling` objects that override `temperature`,
`top_p`, `top_k`, `repetition_penalty`, `penalty_window`, `min_tokens` and
`max_tokens` of each stage. `gufo.bpm`, `gufo.key`, `gufo.time_signature` and
`gufo.language` return 400 `unsupported_parameter`: YuE2 takes them in the
prompt. The job's `gufo.phase` is `plan`, `semantic`, `acoustic` or `decode`;
`gufo.total` is 0 while the model is still choosing the length. A job whose
model hit a token budget reports `gufo.finish: "truncated"`; its audio still
plays. `GET /v1/models` lists what the model accepts under `gufo.music`.

## How it runs

The four upstream stages run on the GPU: planning and semantic tokens on the
autoregressive half of the model, a 32-step midpoint flow-matching solve on its
non-autoregressive half, and the FP32 Oobleck decoder. Each stage is checked
against the upstream CPU reference; see
[the implementation notes](../internal/models/yue2/README.md).

## Performance

As of November 15, 2026, on Strix Halo, an 80-second song from a supplied
score takes 66 s: 36 s for the semantic tokens (55 tokens/s), 27 s for the
acoustic solve and 2 s for the decoder. Planning the score first adds about
20 s for a 930-token score. See the
[benchmarks](../internal/models/yue2/BENCHMARKS.md).
