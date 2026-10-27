# Fish Audio Drama-3 benchmarks

2026-10-27, FEVM FAEX1 (AMD Ryzen AI MAX+ 395 / Radeon 8060S `gfx1151`, 128 GiB
LPDDR5X unified memory, Fedora Linux 44 `7.2.9-200.fc44.x86_64`, ROCm 10.1.0,
machine fingerprint `b8be64d5d2e77f0ca9ebbeb8987678b779b7c766ee7dbf0a978f560a061e57ee`),
native `F8_E4M3` + `BF16` + `F32` checkpoint (`fishaudio/drama-3`).

## Loading

Cold launch of all 8 safetensors shards + `codec.pth` to HTTP readiness at
context 4096: **6.54 s** (`rss_mib = 1825`, `gpu_device_used_mib = 67833`).

## Single-step operator latency

Measured on Layer 0 (`hidden_size = 2048`, `256` routed experts -> `top-8` +
`1` shared expert, `32` Gated DeltaNet value heads of dimension `128`):

| Operator | Latency per layer | Effective bandwidth |
| --- | ---: | ---: |
| Register-resident Gated DeltaNet recurrent step | 14.15 us | — |
| Native `F8_E4M3` MoE (router + gate/up + down + shared) | 132.19 – 170.04 us | ~180.0 GB/s |
| 13-layer BF16 Audio Tower step (10 codebooks) | 10.52 ms total | 200.9 GB/s |

## End-to-end synthesis

62-frame prompt prefill, 96 generated codec frames = **4.46 seconds** of
44.1 kHz mono audio (`196,608` samples), `temperature = 0.7`, `top_k = 30`,
`top_p = 0.8`, `seed = 42`.
RTF = request time / generated audio duration, lower is better.

| Phase | Time | Rate / RTF |
| --- | ---: | ---: |
| Prefill (62 frames: backbone + tower) | 411.2 ms | 150.8 frames/s |
| Autoregressive decode (96 frames: backbone + tower) | 3,011.4 ms | **31.88 frames/s** (1.48x realtime) |
| Offline Fish DAC codec decode (96 frames) | 326.6 ms | Codec RTF 0.073 |
| **Complete buffered WAV** | **3,751.9 ms** | **RTF 0.844 (1.18x realtime)** |
| **Streaming first PCM chunk (4 frames / 185.8 ms)** | **610.0 ms** | Main-thread codec wait: **23.8 ms** across 24 chunks |

Buffered WAV and 4-frame streamed PCM return bitwise identical audio samples.
