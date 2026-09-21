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
| Upstream BF16 operation boundaries | Retained: correct codec/text addition, RMSNorm, RoPE and eager attention casts; independent exact operator checks and fixed-history predictor traces. |
| Library GEMM for predictor parity | Rejected: fixed-history ROCm argmax agreement was 40/45 versus 41/45 with the native GEMV; it did not resolve the remaining rounding differences. |
| Four-wave short GEMV | Rejected: exact output but no complete-request speed improvement. |
| 128-thread short attention | Retained: about 3% faster kernel at 2–32 keys, with exact output; 64 threads were slower. |
| Remove redundant KV clearing and FP32 attention scratch | Retained: visible rows are overwritten before use; exact cancellation/shorter-prompt replay, 32 MiB less scratch at context 4096. |
| First streamed audio at four codec frames | Retained: exact buffered/streamed waveforms for all three variants; subsequent chunks remain 16 frames. |
| Reusable Base waveform-reference state | Retained: exact convolution/KV snapshot, one bounded prefix per runtime; skips reference decoding on repeated voice-clone requests. |
| Lossless BF16 exponent packing | Rejected: exact audio, but only about 2% faster CustomVoice requests for 2.15 GiB additional device memory. |
| Row regrouping / LDS activation staging | Rejected: exact output but slower projection controls. |
| Non-temporal weight loads | Rejected: microbenchmark gains became about a 2% complete-request regression. |
| Generic talker hipBLASLt prefill | Rejected: failed the five-main-code oracle gate. |
| Finish normalization within one wave | Retained: fewer barriers with the same sum tree; independent operator and exact waveform replay checks. |
| Parallel mmap population before upload | Retained: cold startup includes weight reads; no additional persistent weight copy. |
| Compact reference snapshots | Retained: only live convolution/attention rows, exact replacement/cancellation and 300-frame boundary replay. |
| Speaker softmax scratch synchronization | Corrected: independent FP64 check and fresh-process voice-cloning replay; cached references previously hid nondeterminism. |

Next: broaden long-request replay coverage and qualify long reference clips
before wider optimization. See [evaluation](EVALUATION.md); no lower-precision
production change is justified by these measurements.
