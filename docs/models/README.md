# Models

These are the official models: the ones Gufo releases build in and the
maintainers test on AMD Strix Halo. Each page lists the files to download, how
to run the model and what speed to expect.

<!-- BEGIN GENERATED MODEL TABLE -->
| Model | Weights / supported inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF; text | [Usage and modes](deepseek-v4-flash.md) |
| EmbeddingGemma 2 | BF16 safetensors; text/image to embeddings | [Usage](embeddinggemma-2.md) |
| MiniMax H3 FL2VA | Pinned safetensors; text to audiovisual output | [Presets and usage](minimax-h3.md) |
| Qwen-Image-2.1 | BF16 safetensors; generation and image editing | [Usage](qwen-image-2.1.md) |
| Qwen3-ASR 1.7B | BF16 safetensors; audio to text | [Usage](qwen3-asr.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors; text/reference audio to speech | [Voice modes](qwen3-tts.md) |
| Qwen3.8 27B | Q4/Q8 GGUF; text/images | [Usage and modes](qwen3.8-27b.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF; text/images | [Usage and modes](qwen3.8-flash-next.md) |
<!-- END GENERATED MODEL TABLE -->

Gufo doesn't ship weights; download them from Hugging Face as each page shows.
Every speed on these pages was measured on Strix Halo with 128 GB of memory,
on the date given next to it.

Community extensions add other models and variants; see
[extensions](../extensions.md).
