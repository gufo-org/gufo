# Qwen3-ASR quality

**The 15.05-second English fixture reproduces all 49/49 official greedy token
IDs and the transcript.** Qualified September 21, 2026, with the original
1.7B BF16 checkpoint, official text eager / audio SDPA attention.

| Check | Result |
| --- | --- |
| Log-mel frontend versus official | Cosine 1.0; relative L2 9.43e-7 |
| Complete audio encoder versus official | Cosine 0.999852; relative L2 0.01723 |
| Prefill logits versus official | Cosine 0.999211; relative L2 0.04138 |
| Native convolution versus PyTorch | 32,317,440/32,317,440 BF16 values match exactly |
| Long-audio splitting | 60.2-second input in a 512-token context matches separate inference of the same chunks |
| Streaming and concurrent requests | Uploaded-file SSE matches buffered text; cancellation/reuse and concurrent requests retain the 49-token fixture |

These measure implementation agreement. The long-input fixture checks chunking
and state isolation, not recognition accuracy on a long-form corpus. Native
convolution uses independent FP64 controls; [evidence](artifacts/native-convolution.json).

**Limits:** multilingual and long-form corpus WER remain unmeasured. The official
FlashAttention windowed path is not qualified; eager/SDPA and FlashAttention
can differ. Realtime processes complete committed utterances, not a revisable
rolling transcript. Timestamps and the 0.6B model are unsupported.

## Reproduce

Official [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff)
and [checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5).
Tests and the independent runner live in
[`src/models/qwen3_asr`](../../../src/models/qwen3_asr).
Select the changed frontend, encoder, decoder or transcription test:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen3_asr_transcription_hip_test
nix develop -c ctest --preset gpu-full -R '^qwen3_asr_transcription_hip_test$' \
  --no-tests=error --output-on-failure
```

Generate official tensors with `reference/run_official.py --text-attention eager
--audio-attention sdpa`; compare finite tensors, full logits, every generated ID
and the transcript. Keep raw captures outside Git.
