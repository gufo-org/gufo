# Ornith 1.5 122B quality

## Logits against llama.cpp

Teacher-forced comparison over 64 texts of 256 tokens each, Gufo AR against
llama.cpp `23f7aab` on the same GGUF, measured October 21, 2026. The floor row
compares llama.cpp with itself at two batch sizes (ubatch 8 against 256), which
bounds how far two correct implementations drift on this model.

| Run | PPL | KL mean | KL p99 | KL max | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp reference | 12.571 | | | | |
| Gufo, Q4_K experts on MMQ vector kernels | 12.450 | 0.0355 | 0.4501 | 12.973 | 92.90% |
| llama.cpp batch-size floor | 12.528 | 0.0386 | 0.5308 | 11.438 | 92.73% |

Gufo's mean KL and top-1 agreement are inside the floor. The PPL ratio against
the reference has a 95% bootstrap interval of 0.9839 to 0.9961 over texts.
The KL maximum is one chat-template token, where the floor run is also above 11.

Decode quantizes activations to Q8_1 before the MMQ vector kernels, as llama.cpp
does, so replies can differ in wording from a build that keeps activations in
F32.

## Speculative decoding

With MTP at two draft tokens, replies matched MTP off on six prompts of 256
tokens at temperature 0. The per-stream scratch arenas, the 16-head shared-KV
attention and the device-memory weights each left those reply hashes
unchanged, with MTP off and at depth 2.

Sampled drafting was checked against MTP off with 300 seeded completions at
temperature 1.0 (six tokens each, one prompt). 298 of the MTP requests accepted
at least one draft and only 133 of the 300 texts matched their seed's MTP-off
text, yet the item distributions agree: chi-square 0.88 (7 df) on the first
item and 5.99 (11 df, p = 0.87) on the second, which comes from drafted
tokens.

## Functional suites

On the tested stack, the functional suites pass at context 65536 with
`--sessions 1`:

- MTP on: discovery, messages-count-tokens, conversation, tool-native-types,
  long-context.
- MTP off: discovery, conversation, tool-native-types, long-context.

`long-context` used to fail its unchanged-retry check with MTP off: a small
MoE prompt chunk on the verification forward copied its logits without
reporting them, so the full-prompt checkpoint was rejected on the retry. The
fix is in #299 (`8e509128`), and the check passes on Ornith with it.

## Kernel checks

- 16-head shared-KV decode attention matches the generic kernel bit for bit,
  for one row at four depths and for 2 to 4 verification rows at seven depths,
  with the gate on and off. `qwen_attention_long_context_ops_test` checks it
  against an FP64 oracle and checks each verification row against its own
  single-row decode.
- The MMVQ output-projection route was checked against the previous GEMV with
  the 64-text comparison above.
