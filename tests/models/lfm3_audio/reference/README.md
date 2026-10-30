# LFM3-Audio reference

`lfm3_audio_ref.py` is an independent FP32 PyTorch implementation written from
the official Hugging Face remote code of LiquidAI/LFM3-Audio-2B at revision
`cddb56b5ec9b4703f9b3b2d0500143d1fff2fee0` (Apache 2.0). It reads the BF16
GGUF pair of LiquidAI/LFM3-Audio-2B-GGUF `02355c1a` and writes the fixtures
read by `lfm3_audio_probe`; see
[QUALITY.md](../../../../docs/models/lfm3-audio/QUALITY.md) for the commands.
Production does not use Python.
