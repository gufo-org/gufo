# Qwen3.8 Flash-Next on the XDNA2 NPU

The XDNA2 NPU path is optional. It is compiled in with `-DGUFO_ENABLE_NPU=ON`
(see [Build from source](../../../README.md#build-from-source)) and switched on at run time by pointing
`GUFO_NPU_DIR` at `tools/npu`. Without `GUFO_NPU_DIR` the same binary runs
GPU-only.

## Prerequisites at run time

| Requirement | Command |
| --- | --- |
| GPU fabric clock held at its top level | `sudo tools/npu/fabric-clock.sh hold` (`status`, `release`) |
| NPU power mode `balanced` or `performance` (not `turbo`) | `sudo xrt-smi configure --pmode performance` |
| NPU kept out of runtime suspend | `echo on \| sudo tee /sys/bus/pci/devices/<npu pci id>/power/control` |
| XRT libraries on the loader path | `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH` |

The fabric clock changing speed while the NPU is busy can hang the machine, so
hold it before starting. The clock and power mode reset at boot.

## Environment variables

### NPU on and off

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_DIR` | unset (NPU off) | Directory with the NPU kernel files (`tools/npu`). Setting it turns the NPU on. |

### What the NPU computes

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_SLICES` | `6` | 1024-row slices of the linear-attention `ssm_in` projection (z rows) computed on the NPU, 1 to 6. |
| `GUFO_NPU_GATE_SLICES` | `8` | 1024-row slices of the attention `attn_qkv` projection (query and gate rows, two heads per slice) on the NPU, 0 to 12. `0` turns this part off. |
| `GUFO_NPU_SHEXP` | on | `0` keeps the shared-expert gate and up projections on the GPU. |
| `GUFO_NPU_LAYERS` | all | Limit how many linear-attention layers use the NPU. |

### Reliability and diagnostics

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_TIMEOUT_MS` | `8000` | Time a single NPU run may take before it is treated as failed. |
| `GUFO_NPU_KEEPALIVE` | on | `0` stops the 1 s heartbeat that keeps the NPU awake. |
| `GUFO_NPU_WARMUP_MS` | `0` | Keep the NPU out of the first N ms of each prefill burst so the GPU's idle-to-full step happens alone. |
| `GUFO_NPU_ISOLATE` | unset | `1` gives the NPU private buffers and copies through the CPU. Slower; for debugging. |
| `GUFO_NPU_STATS` | unset | Print wall-clock accounting of the NPU path. |
| `GUFO_NPU_TRACE` | unset | Path of a synchronous event log, one line per NPU event. |

## Examples

NPU on, benchmark:

```sh
GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3
```

Same, GPU-only: leave `GUFO_NPU_DIR` unset.

Only the z rows on the NPU:

```sh
GUFO_NPU_DIR=tools/npu GUFO_NPU_GATE_SLICES=0 GUFO_NPU_SHEXP=0 build/release/gufo serve llm ...
```
