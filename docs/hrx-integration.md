# HRX integration

Target: AMD Strix Halo (`gfx1151`) with the Qwen3.8-27B strict Q8_0 model.

## Build and run

The `.#hrx` package contains both HRX execution routes:

- the normal HIP executor through HRX's `libamdhip64.so` compatibility layer;
- the direct native HRX/Loom executor selected with
  `--qwen-backend hrx-native`.

The package wrapper preloads the compatibility library for the normal route.
For `hrx-native`, it removes any preloaded `libamdhip64.so` before starting the
process so native validation cannot accidentally run through HIP interception.

```sh
git add .
nix build .#hrx

./result/bin/gufo diagnose
./result/bin/gufo bench \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --qwen-backend hrx-native \
  --n-gen 8 \
  --repetitions 3
```

The native route currently supports sequential greedy prompt and decode
benchmarks with all model layers at depth zero. It rejects unsupported GGUF
types, speculative decoding, sampling, depth tests, and batched-prefill
validation instead of silently changing their semantics. HIP remains the
default backend, and `serve` continues to use the production HIP executor.

## Native executor

`src/core/hrx/` owns the native runtime, buffers, and artifact loader.
`src/models/qwen/hrx/` implements the complete single-token Qwen3.8-27B path:

- strict metadata and tensor-type validation for the fixed 64-layer Q8_0
  contract;
- offset-preserving bindings over the mapped GGUF weights;
- native embedding, RMSNorm, and Q8_0 projections;
- full-attention Q/K/V, per-head normalization, RoPE, KV-cache update,
  causal attention, output projection, and residual;
- 48-head Gated DeltaNet convolution, parameter preparation, recurrent-state
  update, output projection, and residual;
- SwiGLU FFN and the final RMSNorm, full-vocabulary projection, and argmax;
- reset plus recurrent and KV state snapshots used by parity validation.

The required runtime set consists of 20 Loom artifacts. A separate signed-Q8
decode oracle is compiled with them. The package also carries the retained
256-thread FFN-down artifact described below; if it cannot be loaded, execution
falls back to the required 544-thread reference artifact.

`ModelExecutionReady()` is the authoritative model-level gate. Creation fails
closed and reports missing artifacts or unsupported tensors before executing a
token.

## Correctness validation

`--validate-hrx` captures an independent HIP reference, releases that executor,
then starts native HRX and compares the complete 248,320-element logit vector.
The validation covers position zero, a deterministic four-token prompt, and
the requested number of greedy decode steps.

```sh
./result/bin/gufo bench \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --qwen-backend hrx-native \
  --validate-hrx 1
```

At the current revision all five checkpoints match HIP top-1 output. The
retained FFN-down route measured cosine similarity `1.0`, worst RMSE
`2.03e-6`, and maximum absolute error `8.64e-6`.

## Retained FFN-down optimization

The reference `K=17408` Q8_0 down projection assigns all 544 quantization
blocks to one 544-thread workgroup. The retained artifact uses 256 threads that
stride over those blocks. This reduces the number of wave32s per workgroup
from 17 to 8 and gives the GPU scheduler more resident workgroups.

On the release `.#hrx` build, interleaved stage samples improved the complete
FFN portion from `168.63 ms` to `163.90 ms` on average. The repository's
three-repetition decode benchmark improved from `tg8 = 3.63 tok/s` to
`3.69 tok/s` (+1.7%). The 256-thread path is selected automatically when its
artifact is present. Use the reference fallback for an A/B run:

```sh
GUFO_HRX_Q8_K17408_WG544=1 \
  ./result/bin/gufo bench \
    --model models/Qwen3.8-27B-Q8_0.gguf \
    --qwen-backend hrx-native \
    --n-gen 8 \
    --repetitions 3
```

Set `GUFO_HRX_TRACE_STAGES=1` to print synchronized embedding, attention, SSM,
FFN, final-head, and full-token timings. Native HRX currently cannot be traced
with `rocprofv3`: the profiler aborts while intercepting executable freeze, so
the synchronized stage trace is the supported fallback.

## Gates

```sh
git add .
nix build .#hrx
nix build .#checks.x86_64-linux.pr
```

Use binaries from the release Nix package for model measurements. The focused
CMake test tree is intentionally unoptimized and is only for correctness and
debugging.
