#ifndef GUFO_MODELS_LFM3_AUDIO_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_AUDIO_HIP_OPS_HPP_
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/lfm3/weights.hpp"
namespace gufo::models::lfm3_audio::gpu {
using lfm3::Tensor;
enum class Activation { kNone, kSilu, kGelu };
// y[n, M] = act(x[n, K] W[M, K]^T + bias), with an F32 weight.
void Sgemm(const float* x, const float* w, const float* bias, float* y, int n,
           int M, int K, Activation act, hipStream_t stream);
// y = W x for one row of a BF16 or Q8_0 matrix whose columns are a
// multiple of 128. Returns false (launching nothing) for other shapes.
bool WideGemv(const Tensor& w, const float* x, float* y, hipStream_t stream);
// y[n, M] = act(y + bias) in place.
void BiasAct(float* y, const float* bias, int n, int M, Activation act,
             hipStream_t stream);
// Windowed STFT frames: frames[t, i] = samples[t * hop + i] * window[i].
void Frames(const float* samples, const float* window, float* frames, int n,
            int length, int hop, hipStream_t stream);
// power[t, b] = re^2 + im^2 of rows [re (bins) | im (bins)].
void Power(const float* spectrum, float* power, int n, int bins,
           hipStream_t stream);
// x = (max(log10(x), floor) + offset) / divisor, in place.
void LogMelNormalize(float* x, std::size_t count, float floor, float offset,
                     float divisor, hipStream_t stream);
// First downsampler convolution: 1 -> C channels, 3x3, stride 2, padding 1,
// then SiLU. mel is frame-major [frames][bins]; out is channel-last
// [W][H][C] (W over frames, H over bins).
void ConvFirst(const float* mel, const float* w, const float* bias, float* out,
               int bins, int frames, int channels, hipStream_t stream);
// Depthwise 3x3, stride 2, padding 1, plus bias, over channel-last planes.
void ConvDepthwise(const float* in, const float* w, const float* bias,
                   float* out, int width, int height, int channels,
                   hipStream_t stream);
// [W][H][C] -> [W][C * H] (feature c * H + h, as torch flatten(1, 2)).
void FlattenChannels(const float* in, float* out, int width, int height,
                     int channels, hipStream_t stream);
void LayerNorm(const float* x, const float* w, const float* b, float* y, int n,
               int dim, float eps, hipStream_t stream);
// Attention for head dimensions that are multiples of 32, up to 128. Queries
// of the current chunk sit at positions position..position + n - 1; earlier
// keys come from a ring of capacity rows, the chunk's own keys from k/v.
// window > 0 limits causal attention to the last window keys; causal false
// attends to every row of the chunk (no cache). gate (n x heads) is optional.
void Attention(const float* q, const float* k, const float* v,
               const float* gate, const float* cache_k, const float* cache_v,
               float* out, int n, int heads, int kv_heads, int dim,
               int position, int capacity, int window, bool causal,
               hipStream_t stream);
// out[c, :] = base[c, :] + (masked[c] ? mask : table[c * size + codes[c]]).
void DepthInputs(const float* base, const float* mask, const Tensor& table,
                 const std::int32_t* codes, const std::int32_t* masked,
                 float* out, int codebooks, int size, hipStream_t stream);
// out[r, :] = sum_c table[c * size + codes[r * codebooks + c]].
void FrameEmbeddings(const Tensor& table, const std::int32_t* codes, float* out,
                     int rows, int codebooks, int size, hipStream_t stream);
// latents[r, c * 4 + j] = fsq[codes[r * codebooks + c], j].
void FsqLatents(const float* fsq, const std::int32_t* codes, float* latents,
                int rows, int codebooks, hipStream_t stream);
// out[r * repeat + i, :] = in[r, :].
void Repeat(const float* in, float* out, int rows, int dim, int repeat,
            hipStream_t stream);
// Rows of [log magnitude | angle] (bins each) to [real | imaginary].
void Polar(float* spectrum, int n, int bins, hipStream_t stream);
// out[i, :] = rms_norm(x[c, :], norms[c, :]) for c = which[i], the head
// inputs of the codebooks scored by one depth pass.
void CodebookNorm(const float* x, const float* norms, const std::int32_t* which,
                  float* out, int count, int dim, float eps,
                  hipStream_t stream);
// logits[i, r] = table[which[i] * size + r] . x[i] for count codebooks.
void CodebookLogits(const Tensor& table, const float* x,
                    const std::int32_t* which, float* logits, int count,
                    int size, hipStream_t stream);
// Device state of speech sampling: codes and mask of the frame in progress
// and the repetition window of earlier frames.
struct FrameState {
  std::int32_t* codes;     // [codebooks]
  std::int32_t* masked;    // [codebooks]
  std::int32_t* counts;    // [codebooks][size]: codes in the window
  std::int32_t* ring;      // [window][codebooks]
  std::int32_t* position;  // [2]: ring head, filled rows
};
struct FrameSampling {
  float temperature, penalty;
  int top_k, window;
};
// codes = 0 and every codebook masked.
void StartFrame(FrameState state, int codebooks, hipStream_t stream);
// Samples the codebooks which[0..count) from their logits rows (repetition
// penalty over the window, temperature, top-k, inverse CDF of uniforms[c]),
// then reveals them.
void SampleCodes(const float* logits, const std::int32_t* which,
                 const float* uniforms, FrameState state, FrameSampling s,
                 int count, int size, hipStream_t stream);
// Ends the frame: a frame holding the end code anywhere becomes all end
// codes; others enter the repetition window. Writes the backbone input
// embedding of the frame.
void FinishFrame(const Tensor& audio_out, FrameState state, FrameSampling s,
                 float* embedding, int codebooks, int size, int end_code,
                 hipStream_t stream);
}  // namespace gufo::models::lfm3_audio::gpu
#endif
