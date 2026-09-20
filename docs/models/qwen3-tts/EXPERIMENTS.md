# Qwen3-TTS experiments

| Experiment | Decision / evidence |
| --- | --- |
| Model-private GEMV and shared input columns | Retained; fixed FP32 reduction, independent FP64 controls and model gates. |
| Aligned device weight copies | Retained; better resident throughput than mapped execution. |
| QKV/gate-up, norm/RoPE/cache and codec-embedding fusion | Retained; same group order and boundary checks, fewer dispatches. |
| Partial top-k and vocabulary repetition flags | Retained; preserves candidates, distribution and RNG sequence. |
| Windowed-sinc input resampling | Retained with identity/passband/stopband controls. |
| Speaker residual buffer isolation | Retained; independent embedding cosine 0.999997. |
| Mapped prompt embeddings | Rejected: no credible end-to-end gain; route removed. |
| HIP graph replay | Rejected: no end-to-end improvement at this dispatch floor. |
| GPU masked top-k | Screened: only 0.7% gain in bounded greedy request; no production replacement. |
| Precomputed SnakeBeta exponents | Rejected: about 1% slower. |
| Decoder GEMM layouts / hipBLASLt | Rejected: neutral/slower or failed exactness; F32 library ceiling remains. |
| Stateful waveform streaming | Retained: causal convolution histories and attention KV; exact buffered waveform through the 300-frame boundary, without re-decoding growing prefixes. |
| Sampling controls | Retained: temperature before nucleus filtering and all boundary ties for top-k; independent predictor controls. |

Next: broaden long-request replay coverage and qualify long reference clips
before wider optimization. See [evaluation](EVALUATION.md); no lower-precision
production change is justified by these measurements.
