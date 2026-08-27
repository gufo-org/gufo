# HRX + DFlash2 Integration Task Log

## Status: Card 0 Complete -> Starting Card 1

## Environment & Hardware Fingerprint
- Target Platform: x86_64-linux (Linux 7.1.8)
- GPU: AMD Radeon 8060S Graphics [gfx1151, 20 CUs, 32-wave, 124 GiB VRAM pool]
- NPU: RyzenAI-npu5 [XDNA2, 32 AIE tiles, PCI 1022:17F0, firmware npu.sbin 1.1.2.64/65]
- Toolchain: ROCm 7.2.3, XRT 2.21.0 (86617617), Nix 2.x
- Ambient LD_PRELOAD: None
- Base Revision: jj parent `orxwlsxr 33f29f88 docs(hrx): align integration with gufo runtime (#200)`

## Frozen Model Hashes (SHA-256)
- `models/Qwen3.8-27B-Q8_0.gguf`: `f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8`
- `models/Qwen3.8-27B-UD-Q8_K_XL.gguf`: `2a13bba36d2efa213f9275abc430c40e4d914b145bfde976f30f1bb5b7e23ac2`
- `models/Qwen3.8-27B-DFlash2-Q8_0.gguf`: `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`

## Frozen Reference Outputs & Baselines
- Deterministic 4-token prompt top-1 sequence: `[220, 198, 157, 157]`
- Deterministic 4-token decode top-1 sequence: `[101, 102, 157, 101]`
- Parity metrics envelope: cosine_similarity = 1.00000000, max_abs_diff < 1e-5, rmse < 2.5e-6 (all pass)
- DFlash2 10-prompt suite corpus hash: `59321d75dbd1`
- DFlash2 300-token stress suite corpus hash: `baea40559c61`
- HIP Q8_0 throughput: pp128=411.44 t/s, pp512=551.04 t/s, pp2048=548.84 t/s, tg128=7.62 t/s
- HIP Q8_K_XL throughput: pp128=354.01 t/s, tg16=6.90 t/s
- HRX Native Q8_0 throughput: pp128=4.02 t/s, tg16=3.69 t/s

## Card 0 Gate: PASSED
- Clean rebuild reproduces strict-Q8_0 top-1 trajectory and HIP baselines.
