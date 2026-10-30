# d1-Omni numerical and API checks

October 30, 2026, Strix Halo gfx1151, GCC 16.2 / ROCm 10.1 HIP Clang 24.
Weights: `LiquidAI/d1-Omni-2B-GGUF` revision
`c53ae0b1f92b17065e9158d34019c9ca1ea5bcf8`, Q8_0 backbone and BF16 mmproj.
The original single-question reference is llama.cpp
`8642541b255bfb5dc39cd351a178be09013a55d6`, with every layer and the projector
on the GPU, flash attention off, `-b 8192 -ub 8192 -c 16384 -t 8 -np 1`.
Gufo uses the production release binary and context 8192.

## Packed sets and boxes

The independent teacher uses unmodified LiquidAI Python model, prompt, packing,
box and calibration code from revision
`431f1d667de5afcc9b0c58b2d9c170a7b38b358b`, transformers 5.19.0 and PyTorch
2.14.1 CPU FP32. It dequantizes the same GGUF weights into those model classes
and loads every parameter strictly. It also runs the independent vision
encoder/projector. No original safetensors or production Python dependency is
required. The local PyTorch HIP path crashed during setup, so it is not used
as quality evidence or a performance reference.

Eleven cases cover packed text, set sizes 1/2/16, a 17-question partial tail,
a single object, 1/4/16 box blocks, absence, mixed primitive/box questions and
JSON/Unicode state. Across 15 physical prompts, tokenizer IDs, group tags and
all answer-slot positions match the independent Python layout exactly. The
native forward's selected logits differ by at most `0.000063` against the fixed
`0.002` gate. End-to-end box coordinates and counts match exactly; the maximum
probability difference is below `0.000001`. The fixed API limits are `0.02`
for probabilities and `0.002` for box coordinates. No gate was relaxed.

`tests/models/d1/forward_probe.cpp` checks the independent layouts and logits
on real weights, including rejection of invalid groups/slots before mutation.
`tests/functional/d1_packed.py` passes 26 HTTP checks for default grouping,
partial tails, calibrated bounds, audio/spoken instructions, mixed media,
replay, concurrent isolation, bad set sizes/max values, wrong image counts and
context overflow. Independent packed numerical evidence covers text and
vision; audio placement uses these functional checks and the prior encoder
qualification. [Compact evidence](artifacts/packed-qualification.json), with
full prompts, embeddings, logits and responses retained locally under
`artifacts/d1/packed/`.

Matched `set_size=1` controls against commit `431efcc7` retain exact text,
image, audio and long-prompt responses. An initial compact single-slot path
changed probabilities by at most `1.3e-7`; it was rejected for this compatibility
route. The original full-vocabulary single-slot path is retained, while packed
and box slots project only the vocabulary entries they read. Both attempts
remain in local raw benchmark reports. Current timings are in
[BENCHMARKS.md](BENCHMARKS.md).

Reproduce the numerical teacher with [tools/d1](../../../tools/d1/README.md).
These model checks remain opt-in and outside hosted CI.

## Operator contracts

`d1.ops` compares bidirectional attention to an independent FP64 softmax and
weighted-sum formula at 1/7/15/17/33/65 queries (16 query heads, four KV heads,
128 dimensions), with both unmasked and packed state/question masks. Encoded
row-gather tests cover reordered and ragged rows. The tolerance is `2e-6`. Separate FP64 checks cover gated
spatial convolution at image edges, merge-block ordering and axial RoPE.
The original causal `lfm3.ops` suite still passes. Causal and bidirectional
attention are separate compile-time specializations.

`d1.contract` covers max-pooled token forms before temperature scaling,
ordinal median versus argmax, zero-temperature rejection, image normalization,
merge-block/channel order, a torchvision antialiased bicubic pixel golden,
and malformed RGB input. `lfm3.config` accepts the
recognized d1 decision metadata while rejecting unknown noncausal models.
Additional box contracts cover absent objects, ordered/crossed edges, Halton
confidence, calibration, monotonically decreasing presence and `at_least`.
The audio WAV/resampling host checks pass. HTTP transport tests cover
unauthorized access, malformed JSON, invalid requests, model mismatch,
readiness, discovery and decision responses without a text backend.

## Independent model checks

`tests/functional/d1.py` compares real HTTP responses for eight histories:
text with all three primitive types, image, audio, JSON/Unicode, literal control
token spellings, a 734-token prompt, mixed image/audio and audio instructions.
All discrete answers and input-token counts match. Every response generates
zero tokens. The fixed absolute probability gate is `0.02`; the largest
observed difference is `0.003213`. Quantized activation arithmetic differs
between the implementations, so this is bounded probability agreement rather
than bitwise logit equivalence or a downstream accuracy benchmark.

Concurrent text/image requests preserve the text control's exact response.
Missing/empty questions, unsupported output types, invalid scores, malformed
images and context overflow return 400. The retained compact
[qualification record](artifacts/qualification.json) contains each matched
request's timing/error, reference identity and fixture hashes. Raw responses,
server logs and process results remain in local ignored `artifacts/d1/`.

The first mixed-media check failed: English P(yes) differed by `0.0338` because
Gufo placed images before audio. The official interface reads `files` before
`images`. Correcting that ordering reduced the mixed-media maximum difference
to `0.001272`. The failed run remains retained as `artifacts/d1/functional.json`;
its passing follow-up is `functional-final.json`. No tolerance was changed.

The audio-instructions fixture uses the JFK recording as instructions; it
checks encoding, placement and reference parity, not the ability to answer a
particular spoken question. Beyond the box evidence above, no extraction qualification, large-image
quality sweep, maximum-context accuracy, F16/BF16-backbone comparison or full
LFM3-Audio speech-generation replay is claimed.

## Reproduction

Start the servers as in the usage guide and the reference configuration above,
then run:

```sh
python3 tests/functional/d1.py \
  --image cats.png --audio jfk.wav --output artifacts/d1/replay.json
```

The image is the Hugging Face documentation `coco_sample.png`; audio is
whisper.cpp's `samples/jfk.wav`. Their exact SHA-256 identities are retained in
the qualification record. The script requires both inputs and does not count
missing-model skips as passes. These model checks remain outside hosted CI;
only the small CPU contracts join the hosted PR target.
