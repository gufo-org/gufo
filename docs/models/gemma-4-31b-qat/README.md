# Gemma 4 31B QAT

Google's quantization-aware-trained Gemma 4 31B IT in Unsloth's GGUF:
`unsloth/gemma-4-31B-it-qat-GGUF`, **gemma-4-31B-it-qat-UD-Q4_K_XL.gguf**
(16.1 GiB). Despite the file name, every projection and the tied
embedding/LM head is Q4_0. The architecture, tokenizer and chat template
match the [standard 31B](../gemma-4-31b/README.md); the same Gemma 4 engine
serves both, so every mode, option and serving feature described there
applies here.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

Use the files from this repository together:

- **MTP drafter:** the Q4_0 QAT `gemma4-assistant` at the repository root,
  `mtp-gemma-4-31B-it.gguf` (identical to `MTP/mtp-gemma-4-31B-it-Q4_0.gguf`).
  It pairs with the QAT target; the standard 31B drafter does not.
- **Vision sidecar:** this repository's `mmproj-BF16.gguf`. Its weights differ
  from the standard 31B sidecar; it is found beside the model or passed with
  `--mmproj`.

## Load and run

```sh
nix develop -c hf download unsloth/gemma-4-31B-it-qat-GGUF \
  --revision 43cc1aeb31adf47ec06a854507ce552cd9862e6f \
  --include gemma-4-31B-it-qat-UD-Q4_K_XL.gguf mtp-gemma-4-31B-it.gguf \
  mmproj-BF16.gguf \
  --local-dir models/gemma-4-31b-qat
nix build
MODEL=models/gemma-4-31b-qat/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
MTP=models/gemma-4-31b-qat/mtp-gemma-4-31B-it.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --context 131072
```

Autoregressive decode reads Q4_0 through Gemma's split-K GEMV; verification
rows and single-token decode with the drafter loaded use the shared exact
small-batch kernels, so greedy speculative output equals single-token decoding
of the same configuration. Prefill multiplies Q4_0 codes against Q8_1
activations with WMMA. A 131072 context needs the same 8.8 GiB of KV and rings
as the standard 31B next to 16.1 GiB of weights.
