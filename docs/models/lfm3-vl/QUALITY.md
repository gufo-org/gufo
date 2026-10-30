# LFM3-VL qualification

October 30, 2026; production Linux x86-64 Strix Halo `gfx1151` build in the
pinned development shell. Q8_0 backbone plus BF16 mmproj from official GGUF
revision `b36554cfd35854b06429900131730cd80f092cdb`.

## Independent numerical reference

[The reference tool](../../../tools/lfm3_vl/reference.py) loads unmodified
LiquidAI source at revision `ba44bd51db20a2753f28c273a1571ccaa210d954`,
checks its pinned file hashes, and strictly maps dequantized GGUF tensors
into the upstream language, vision and projector classes. The teacher runs
CPU FP32 using transformers 5.21.0 and Torch 2.14.1. It does not call Gufo's
preprocessing, vision math or template implementation.

Six cases cover a single color image, a natural photograph with thinking
on, a 529-token wide image prompt, low detail, two interleaved images and a
follow-up retaining the image. All image grids and complete prompt token IDs
match. Maximum patch error is `5.92e-8`; maximum image-embedding error is
`2.64e-5`. Every case agrees on the next token. Across all 128000 logits,
maximum error is `1.03e-4`, KL below `4.04e-10` and total variation below
`1.18e-5`. Gates are patch error `1e-6`, embedding error `0.002`, matching
top-1, KL `1e-5` and total variation `0.002`.

The [model-private probe](../../../tests/models/lfm3_vl/probe.cpp) saves
patches, image embeddings, token IDs and full logits. CPU contracts retain
independent Jinja byte goldens, placement/count rejection and native versus
low-detail resizing. Shared vision operator checks retain their FP64
convolution/RoPE oracles, and d1 preprocessing retains independent torchvision
pixel goldens.

## API and regression evidence

All 19 cases (28 requests) in the focused
[HTTP suite](../../../tests/functional/lfm3_vl.py) pass. They check text-only
and image answers, exact streamed replay, token counting, image replacement,
follow-up reuse, image/text ordering, detail sizing, structured JSON, bounded
reasoning, native tools, Responses/Messages, concurrent isolation, literal
marker text, rejection/recovery, cancellation, and persistent image-prefix
restoration after restart. Changed pixels cannot restore an incompatible
image prefix, including from disk. CLI image generation also describes the
reviewed two-cat fixture correctly.

All 26 d1 packed/box/audio HTTP checks pass after sharing the vision encoder.
The parent-build single-decision control matches exactly; the earlier local
`artifacts/d1/packed/http.json` predates that final dispatch and differs by
about `1.38e-8` on the single primitive. Its image/box responses match exactly.
That older file is not used as a matched revision control.

Eleven identical text-only histories on fresh parent/candidate servers, with
an unchanged-parent control, retain identical outputs and prefill/cache
work. All 22 request/phase comparisons pass the existing 5% and 10 ms gates.
This is a text-backend regression check, not a VL speedup claim.

[Retained metrics](artifacts/qualification.json). Raw requests, responses,
SSE, process logs, tensors and the standard backbone benchmark remain in
ignored `artifacts/lfm3-vl/`. Initial CLI/harness failures are retained there:
one invocation used `--reasoning` instead of `--think`, and the cancellation
check initially looked for the audio service's `outcome` field rather than
the text server's `finish=cancelled` field.

## Limits

These checks establish bounded graph/template/API correctness. They do not
qualify OCR, spatial reasoning, multilingual accuracy, the full text API/tool
matrix, maximum native context, maximum image size or video. Existing
[LFM3 text qualification failures](../lfm3/QUALITY.md) remain visible; this
new checkpoint has not undergone that full matrix. Only user-message still
images are admitted. BF16/F16 backbones and quantized projectors have not
received this end-to-end numerical qualification.
