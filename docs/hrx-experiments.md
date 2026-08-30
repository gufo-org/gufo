# HRX and XDNA2 Experiment Log

This is the durable record for native HRX and DFlash-on-XDNA2 experiments.
Each entry records the exact revision, artifact hashes, machine/driver state,
commands, raw samples, correctness evidence, and the retain/reject decision.

## Baseline

| Route | Status | Contract |
| --- | --- | --- |
| HIP DFlash | baseline | Fixed Qwen3.8-27B DFlash GGUF; target-only final-token parity. |
| Native HRX | in progress | Dedicated `.#hrx` package; fixed Qwen3.8-27B artifact ABI. |
| DFlash XDNA2 hybrid | blocked | No DFlash XCLBIN is packaged yet. |
| DFlash XDNA2 full | blocked | No DFlash XCLBIN is packaged yet. |
| DFlash XDNA2 compact | blocked | No DFlash XCLBIN is packaged yet. |

## Entry template

### `<route> — <revision>`

- Hypothesis:
- Model and artifact hashes:
- Hardware, firmware, XRT, ROCm, and HRX versions:
- Commands and raw samples:
- Correctness: CPU/XRT oracle, target-only parity, and speculative acceptance:
- Profiling: DMA, submission, completion, overlap, and shared-memory traffic:
- Decision: retained or rejected, with reason.
