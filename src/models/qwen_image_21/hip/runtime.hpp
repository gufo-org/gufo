#ifndef GUFO_MODELS_QWEN_IMAGE_21_HIP_RUNTIME_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_HIP_RUNTIME_HPP_

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

#include <array>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "src/models/qwen_image_21/hip/blas.hpp"
#include "src/models/qwen_image_21/model.hpp"
#include "src/models/qwen_image_21/weights.hpp"

namespace gufo::models::qwen_image_21::hip {

void Check(hipError_t status);

// Model-private operators; callers supply validated dimensions and buffers.
bool DenseShape(int channels, int inner);
// Columns must be a multiple of 16.
void PackDense16(const std::uint16_t* input, std::uint16_t* output, int rows,
                 int cols, hipStream_t stream);
bool DenseBf16(const std::uint16_t* weights, const std::uint16_t* input,
               std::uint16_t* output, int rows, int channels, int inner,
               hipStream_t stream);
// All inputs and the output use PackDense16 layout.
bool DenseSwiGlu(const std::uint16_t* gate, const std::uint16_t* up,
                 const std::uint16_t* input, std::uint16_t* output, int rows,
                 int channels, int inner, hipStream_t stream);
bool Convolve3Shape(int channels, int output_channels);
// Weights use PackDense16 layout.
bool Convolve3(const std::uint16_t* input, const std::uint16_t* weights,
               const std::uint16_t* bias, std::uint16_t* output, int height,
               int width, int channels, int output_channels, hipStream_t stream,
               int stride = 1, int padding = 1, int upscale = 1);
bool ExpandConvolution3(const std::uint16_t* input, std::uint16_t* output,
                        int height, int width, int channels, int stride,
                        int padding, int upscale, int output_width, int start,
                        int count, hipStream_t stream);
void ReorderHeads(const std::uint16_t* input, std::uint16_t* output, int rows,
                  int heads, int kv_heads, int dim, bool inverse,
                  hipStream_t stream);
/// Up to three head-major key/value buffers read as one key sequence:
/// segment i holds rows [end[i-1], end[i]) at a per-head stride of stride[i]
/// rows. Unused trailing segments repeat the previous end.
struct KeySegments {
  const std::uint16_t* keys[3]{};
  const std::uint16_t* values[3]{};
  int end[3]{};
  int stride[3]{};
};
void FusedAttention128Segmented(const std::uint16_t* queries,
                                const KeySegments& segments,
                                std::uint16_t* output, int heads, int rows,
                                int query_start, int query_count,
                                hipStream_t stream, const int* limits);
void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count = 128,
                       const int* limits = nullptr);

struct Matrix {
  std::shared_ptr<void> owner;
  std::uint16_t* data{nullptr};
  int rows{0};
  int cols{0};
  bool packed{false};
  std::size_t size() const { return static_cast<std::size_t>(rows) * cols; }
  Matrix Slice(int start, int count) const;
  Matrix Reshape(int r, int c) const;
};

enum class Norm { kLayer, kRms, kZeroRms, kVae };
enum class Activation { kSilu, kGeluTanh, kGelu, kIdentity };
void NormalizeRows(const std::uint16_t* input, std::uint16_t* output,
                   const std::uint16_t* weight, const std::uint16_t* bias,
                   int rows, int cols, Norm kind, float epsilon,
                   hipStream_t stream, bool silu = false);
void NormalizeRotary128(const std::uint16_t* input, std::uint16_t* output,
                        const std::uint16_t* weight, const float* frequencies,
                        int rows, int heads, float epsilon, hipStream_t stream);

class Runtime {
public:
  explicit Runtime(const Weights& weights);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Matrix New(int rows, int cols);
  Matrix Upload(std::span<const float> values, int rows, int cols);
  /// FP32 rotary tables; logical storage uses two BF16-sized slots per float.
  Matrix UploadFloat(std::span<const float> values, int rows, int cols);
  Matrix Weight(std::string_view name);
  Matrix Linear(const Matrix& x, const std::string& name);
  Matrix Normalize(const Matrix& x, Norm kind, const std::string& weight = {},
                   float epsilon = 1e-6F, bool silu = false);
  Matrix Activate(const Matrix& x, Activation activation);
  Matrix SwiGlu(const Matrix& gate, const Matrix& up);
  Matrix GatedLinear(const Matrix& x, const std::string& gate,
                     const std::string& up);
  Matrix Add(const Matrix& a, const Matrix& b);
  Matrix ModulationFactors(const Matrix& mod, bool gates);
  Matrix Modulate(const Matrix& x, const Matrix& mod, int part, int prefix,
                  const Matrix* residual = nullptr);
  Matrix NormalizeModulate(const Matrix& x, const Matrix& mod, int part,
                           int prefix);
  Matrix Rope(const Matrix& x, int heads, int dim, const Matrix& frequencies,
              int mode);
  Matrix NormalizeRope(const Matrix& x, const std::string& weight, int heads,
                       const Matrix& frequencies);
  Matrix Attention(const Matrix& q, const Matrix& k, const Matrix& v, int heads,
                   int kv_heads, int dim,
                   const std::vector<int>& key_limits = {});
  Matrix Concat(std::span<const Matrix> matrices);
  /// 32-head, 128-dim head-major copy of a row-major K or V matrix.
  Matrix HeadLayout(const Matrix& x);
  /// Rows [0, rows) of head-major key/value buffers (per-head stride:
  /// keys.rows).
  struct HeadKeys {
    Matrix keys, values;
    int rows{0};
  };
  /// Attention of row-major queries over the concatenated segments without
  /// copying them; limits as in Attention.
  Matrix SegmentedAttention(const Matrix& q, std::span<const HeadKeys> segments,
                            const std::vector<int>& key_limits = {});
  std::size_t CachedBytes() const;
  /// Bytes of a weight group that are not resident yet.
  std::size_t MissingBytes(std::string_view prefix) const;
  Matrix Columns(const Matrix& x, int start, int count);
  Matrix Embed(const Matrix& weights, std::span<const std::uint32_t> ids);
  Matrix ReplaceRows(const Matrix& x, const Matrix& values,
                     std::span<const int> positions, bool add);
  Matrix Conv(const Matrix& x, int height, int width, const std::string& name,
              int stride = 1, int padding = 1, int upscale = 1);
  Matrix DownShortcut(const Matrix& x, int height, int width, int out_channels,
                      int temporal, int spatial);
  Matrix UpShortcut(const Matrix& x, int height, int width, int out_channels,
                    int temporal);
  Matrix Euler(const Matrix& x, const Matrix& velocity, float dt);
  Matrix Guidance(const Matrix& positive, const Matrix& negative, float scale);
  Matrix NoiseMix(const Matrix& noise, const Matrix& image, float sigma);
  Matrix LatentScale(const Matrix& x, std::span<const float> mean,
                     std::span<const float> stddev, bool encode);
  Matrix VisionPosition(const Matrix& x, int height, int width);
  std::vector<float> Download(const Matrix& matrix);
  void Observe(const Observer& observer, std::string_view name,
               const Matrix& matrix);
  void Synchronize();
  /// Drop a resident weight component (e.g. the text encoder after the
  /// prompts are encoded); it reloads on next use. Also trims the pool.
  void Evict(std::string_view prefix);
  /// Return cached pool blocks to HIP. Freed tensors otherwise stay cached
  /// (up to 8 GB) and push the next phase into shared (host) memory.
  void Trim();
  /// Upper bound of freed blocks kept for reuse (default 8 GB).
  void SetCacheLimit(std::size_t bytes);
  /// Copy `source` into rows [row, row + source.rows) of `destination`.
  void CopyRows(const Matrix& destination, int row, const Matrix& source);
  std::size_t WeightBytes() const { return weight_bytes_; }
  hipStream_t stream() const { return stream_; }

private:
  struct Pool;
  std::shared_ptr<Pool> pool_;
  const Weights& weights_;
  hipStream_t stream_{nullptr};
  hipblasHandle_t blas_{nullptr};
  Gemm gemm_;
  std::map<std::string, Matrix, std::less<>> device_weights_;
  std::vector<std::string_view> prefetched_groups_;
  Matrix packed_source_, packed_input_;
  std::size_t weight_bytes_{0};
  // Pinned buffers for weight uploads; see Runtime::UploadMapped.
  std::array<void*, 2> staging_{};
  std::array<hipEvent_t, 2> staging_done_{};
  /// Device buffers for one weight: `m` in BF16, plus `raw` for F32 sources.
  struct StagedWeight {
    Matrix m, raw;
  };
  Matrix Raw(std::size_t bytes);
  Matrix PackedInput(const Matrix& x);
  void UploadMapped(void* device, const void* host, std::size_t bytes);
  StagedWeight Stage(const qwen_image_21::Weight& w);
  bool UploadGroup(std::string_view prefix);
  Matrix Store(std::string_view name, const qwen_image_21::Weight& w,
               StagedWeight target);
  void Matmul(const Matrix& x, const Matrix& w, Matrix& output,
              const Matrix* bias = nullptr, bool convolution = false);
};

}  // namespace gufo::models::qwen_image_21::hip
#endif
