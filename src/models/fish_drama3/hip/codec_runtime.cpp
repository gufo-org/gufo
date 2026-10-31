#include "src/models/fish_drama3/hip/codec_runtime.hpp"

#include <filesystem>
#include <unordered_map>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/fish_drama3/hip/codec_ops.hpp"
#include "src/models/fish_drama3/loader.hpp"

namespace gufo::models::fish_drama3::hip {
namespace {

constexpr std::size_t kCodecDim = 1024;
constexpr std::size_t kCodecHeads = 16;
constexpr std::size_t kCodecHeadDim = 64;
constexpr std::size_t kCodecInter = 3072;
constexpr std::size_t kCodebookDim = 8;
constexpr float kCodecNormEps = 1.0e-5F;
constexpr float kConvNextNormEps = 1.0e-6F;
constexpr float kCodecRopeTheta = 10000.0F;
constexpr std::size_t kPostModuleWindow = 128;
constexpr std::size_t kPreModuleWindow = 128;
constexpr std::size_t kEncoderTransformerWindow = 512;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void RequireRocblas(rocblas_status status, std::string_view operation) {
  if (status != rocblas_status_success) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void Reset(std::size_t count = 0) {
    if (count > 0 && count <= count_ && data_ != nullptr) {
      return;
    }
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Fish DAC codec device buffer overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Fish DAC codec");
    count_ = count;
  }

  void Upload(const Element* host_data, std::size_t count,
              hipStream_t stream = nullptr) {
    Reset(count);
    if (count > 0) {
      if (stream != nullptr) {
        RequireHip(hipMemcpyAsync(data_, host_data, count * sizeof(Element),
                                  hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync Fish DAC codec");
      } else {
        RequireHip(hipMemcpy(data_, host_data, count * sizeof(Element),
                             hipMemcpyHostToDevice),
                   "hipMemcpy Fish DAC codec");
      }
    }
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

class F32Gemm {
public:
  F32Gemm() {
    RequireRocblas(rocblas_create_handle(&handle_), "rocblas_create_handle");
    RequireRocblas(
        rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed),
        "rocblas_set_atomics_mode");
  }

  ~F32Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }

  F32Gemm(const F32Gemm&) = delete;
  F32Gemm& operator=(const F32Gemm&) = delete;

  void Run(const float* weight, const float* input, float* output,
           std::size_t rows, std::size_t output_columns, std::size_t reduction,
           hipStream_t stream) {
    constexpr std::size_t max_int =
        static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max());
    if (rows == 0 || output_columns == 0 || reduction == 0 || rows > max_int ||
        output_columns > max_int || reduction > max_int) {
      throw std::length_error("Invalid Fish DAC codec GEMM shape");
    }
    RequireRocblas(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    RequireRocblas(
        rocblas_gemm_ex(
            handle_, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<rocblas_int>(output_columns),
            static_cast<rocblas_int>(rows), static_cast<rocblas_int>(reduction),
            &alpha, weight, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(reduction), input, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(reduction), &beta, output,
            rocblas_datatype_f32_r, static_cast<rocblas_int>(output_columns),
            output, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(output_columns), rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "Fish DAC codec GEMM");
  }

private:
  rocblas_handle handle_{nullptr};
};

struct LinearWeights {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t input_columns{0};
  std::size_t output_columns{0};
};

struct ConvWeights {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t input_channels{0};
  std::size_t output_channels{0};
  std::size_t kernel{0};
  std::size_t dilation{1};
  std::size_t stride{1};
  bool depthwise{false};
  bool transpose{false};
};

struct TransformerLayerWeights {
  DeviceBuffer<float> attention_norm;
  LinearWeights q_proj;
  LinearWeights k_proj;
  LinearWeights v_proj;
  LinearWeights wo;
  DeviceBuffer<float> attention_scale;
  DeviceBuffer<float> ffn_norm;
  LinearWeights w1;  // gate
  LinearWeights w2;  // down
  LinearWeights w3;  // up
  DeviceBuffer<float> ffn_scale;
};

struct TransformerWeights {
  std::vector<TransformerLayerWeights> layers;
  DeviceBuffer<float> norm;
  std::size_t sliding_window{128};
};

struct ConvNeXtWeights {
  ConvWeights depthwise;
  DeviceBuffer<float> norm_weight;
  DeviceBuffer<float> norm_bias;
  LinearWeights pwconv1;
  LinearWeights pwconv2;
  DeviceBuffer<float> gamma;
};

struct UpsampleStageWeights {
  ConvWeights upsample;
  ConvNeXtWeights convnext;
};

struct DownsampleStageWeights {
  ConvWeights downsample;
  ConvNeXtWeights convnext;
};

struct ResidualUnitWeights {
  DeviceBuffer<float> snake1_alpha;
  ConvWeights conv1;
  DeviceBuffer<float> snake2_alpha;
  ConvWeights conv2;
};

struct DecoderBlockWeights {
  DeviceBuffer<float> snake_alpha;
  ConvWeights upsample;
  std::array<ResidualUnitWeights, 3> residuals;
};

struct EncoderBlockWeights {
  std::array<ResidualUnitWeights, 3> residuals;
  DeviceBuffer<float> snake_alpha;
  ConvWeights downsample;
  std::optional<TransformerWeights> transformer;
};

struct QuantizerUnitWeights {
  DeviceBuffer<float> in_proj_weight;       // [8, 1024]
  DeviceBuffer<float> in_proj_bias;         // [8]
  DeviceBuffer<float> out_proj_weight;      // [1024, 8]
  DeviceBuffer<float> out_proj_bias;        // [1024]
  DeviceBuffer<float> codebook;             // [codebook_size, 8]
  DeviceBuffer<float> normalized_codebook;  // [codebook_size, 8]
  std::size_t codebook_size{0};
};

struct ConvHistory {
  DeviceBuffer<float> input;
  std::size_t rows{0};
};

struct AttentionHistory {
  DeviceBuffer<float> key;
  DeviceBuffer<float> value;
  std::size_t frames{0};
};

struct StreamingState {
  std::unordered_map<const ConvWeights*, ConvHistory> convolutions;
  std::vector<AttentionHistory> post_module_attention;

  void Reset() {
    for (auto& [w, hist] : convolutions) {
      hist.rows = 0;
    }
    for (auto& attn : post_module_attention) {
      attn.frames = 0;
    }
  }
};

struct Workspace {
  std::size_t max_samples{0};
  DeviceBuffer<std::int32_t> codes;
  std::array<DeviceBuffer<float>, 6> scratch;
  DeviceBuffer<float> columns;
  DeviceBuffer<float> history_input;
  DeviceBuffer<float> history_output;

  void EnsureForSamples(std::size_t samples) {
    if (samples <= max_samples) {
      return;
    }
    // At 1/2 downsample (samples/2), channels can be 768 (decoder block 1
    // output) or at samples/1, channels = 96. Max intermediate elements is
    // max(samples * 96, (samples / 2) * 768) = samples * 384.
    const std::size_t scratch_elements =
        std::max<std::size_t>(samples * 384U, 65536U * 1024U);
    const std::size_t col_elements = scratch_elements * 7U;
    codes.Reset(4096U * 10U);
    for (auto& buf : scratch) {
      buf.Reset(scratch_elements);
    }
    columns.Reset(col_elements);
    history_input.Reset(scratch_elements);
    history_output.Reset(scratch_elements);
    max_samples = samples;
  }
};

float* FindScratch(Workspace& workspace,
                   std::initializer_list<const float*> excluded) {
  for (auto& buffer : workspace.scratch) {
    if (std::ranges::find(excluded, buffer.get()) == excluded.end()) {
      return buffer.get();
    }
  }
  throw std::runtime_error("Fish DAC codec scratch buffer exhaustion");
}

std::vector<float> NormalizeCodebookRows(const float* data, std::size_t rows,
                                         std::size_t cols) {
  std::vector<float> out(rows * cols, 0.0F);
  for (std::size_t r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const float val = data[r * cols + c];
      sum += static_cast<double>(val) * static_cast<double>(val);
    }
    const float inv =
        sum > 0.0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0F;
    for (std::size_t c = 0; c < cols; ++c) {
      out[r * cols + c] = data[r * cols + c] * inv;
    }
  }
  return out;
}

}  // namespace

struct FishDacCodecHipRuntime::Impl {
  CodecConfig config;
  hipStream_t stream{nullptr};
  F32Gemm gemm;
  Workspace workspace;
  StreamingState stream_state;
  StreamingState* active_stream{nullptr};

  // Encoder weights
  ConvWeights encoder_first;
  std::vector<EncoderBlockWeights> encoder_blocks;
  DeviceBuffer<float> encoder_final_snake;
  ConvWeights encoder_final;
  std::vector<DownsampleStageWeights> downsample_stages;
  TransformerWeights pre_module;

  // Quantizers
  QuantizerUnitWeights semantic_quantizer;
  std::vector<QuantizerUnitWeights> residual_quantizers;

  // Decoder weights
  TransformerWeights post_module;
  std::vector<UpsampleStageWeights> upsample_stages;
  ConvWeights decoder_first;
  std::vector<DecoderBlockWeights> decoder_blocks;
  DeviceBuffer<float> decoder_final_snake;
  ConvWeights decoder_final;

  Impl(const std::string& model_dir, const CodecConfig& cfg) : config(cfg) {
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags Fish DAC codec");
    auto store = LoadCodecSafetensorsOrPth(model_dir);
    if (!store) {
      throw std::runtime_error("Cannot load Fish DAC codec from " + model_dir);
    }
    LoadWeights(*store);
    workspace.EnsureForSamples(128U *
                               static_cast<std::size_t>(config.frame_length));
    {
      FishDacCodes warm_codes;
      warm_codes.codebooks = static_cast<std::int64_t>(config.total_codebooks);
      warm_codes.frames = 1;
      warm_codes.codes.assign(config.total_codebooks, 0);
      active_stream = &stream_state;
      (void)RunDecodePass(warm_codes);
      active_stream = nullptr;
      stream_state.Reset();
    }
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Fish DAC codec weights");
  }

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  const float* RequireF32(const ModelSafetensors& store,
                          const std::string& name,
                          std::initializer_list<std::uint64_t> expected_shape) {
    const TensorView& t = store.Require(name);
    if (t.dtype != DType::kF32 ||
        !std::ranges::equal(t.shape, expected_shape)) {
      throw std::runtime_error("Unexpected dtype/shape for codec tensor: " +
                               name);
    }
    return reinterpret_cast<const float*>(t.data);
  }

  DeviceBuffer<float> LoadVector(const ModelSafetensors& store,
                                 const std::string& name, std::size_t size) {
    const TensorView& t = store.Require(name);
    if (t.dtype != DType::kF32 || t.NumElements() != size) {
      throw std::runtime_error("Unexpected vector tensor shape: " + name);
    }
    DeviceBuffer<float> buf;
    buf.Upload(reinterpret_cast<const float*>(t.data), size, stream);
    return buf;
  }

  LinearWeights LoadLinear(const ModelSafetensors& store,
                           const std::string& prefix, std::size_t in_cols,
                           std::size_t out_cols, bool has_bias) {
    LinearWeights res;
    res.input_columns = in_cols;
    res.output_columns = out_cols;
    const float* w = RequireF32(store, prefix + ".weight", {out_cols, in_cols});
    res.weight.Upload(w, out_cols * in_cols, stream);
    if (has_bias) {
      const float* b = RequireF32(store, prefix + ".bias", {out_cols});
      res.bias.Upload(b, out_cols, stream);
    }
    return res;
  }

  ConvWeights LoadConv(const ModelSafetensors& store, const std::string& prefix,
                       std::size_t in_ch, std::size_t out_ch,
                       std::size_t kernel, std::size_t stride = 1,
                       std::size_t dilation = 1, bool depthwise = false) {
    ConvWeights res;
    res.input_channels = in_ch;
    res.output_channels = out_ch;
    res.kernel = kernel;
    res.stride = stride;
    res.dilation = dilation;
    res.depthwise = depthwise;
    res.transpose = false;
    const std::size_t group_in = depthwise ? 1U : in_ch;
    const float* w =
        RequireF32(store, prefix + ".weight", {out_ch, group_in, kernel});
    const float* b = RequireF32(store, prefix + ".bias", {out_ch});
    if (!depthwise && kernel > 1U && group_in > 1U) {
      std::vector<float> reordered(out_ch * kernel * group_in);
      for (std::size_t oc = 0; oc < out_ch; ++oc) {
        for (std::size_t ic = 0; ic < group_in; ++ic) {
          for (std::size_t k = 0; k < kernel; ++k) {
            reordered[(oc * kernel + k) * group_in + ic] =
                w[(oc * group_in + ic) * kernel + k];
          }
        }
      }
      res.weight.Upload(reordered.data(), reordered.size(), stream);
    } else {
      res.weight.Upload(w, out_ch * group_in * kernel, stream);
    }
    res.bias.Upload(b, out_ch, stream);
    return res;
  }

  ConvWeights LoadTransposeConv(const ModelSafetensors& store,
                                const std::string& prefix, std::size_t in_ch,
                                std::size_t out_ch, std::size_t kernel,
                                std::size_t stride) {
    ConvWeights res;
    res.input_channels = in_ch;
    res.output_channels = out_ch;
    res.kernel = kernel;
    res.stride = stride;
    res.dilation = 1;
    res.depthwise = false;
    res.transpose = true;
    const float* w =
        RequireF32(store, prefix + ".weight", {in_ch, out_ch, kernel});
    const float* b = RequireF32(store, prefix + ".bias", {out_ch});
    DeviceBuffer<float> raw_w;
    raw_w.Upload(w, in_ch * out_ch * kernel, stream);
    res.weight.Reset(in_ch * out_ch * kernel);
    LaunchCodecTransposeConvWeight(raw_w.get(), res.weight.get(), in_ch, out_ch,
                                   kernel, stream);
    RequireHip(hipStreamSynchronize(stream), "TransposeConvWeight synchronize");
    res.bias.Upload(b, out_ch, stream);
    return res;
  }

  ResidualUnitWeights LoadResidualUnit(const ModelSafetensors& store,
                                       const std::string& prefix,
                                       std::size_t channels,
                                       std::size_t dilation) {
    ResidualUnitWeights unit;
    unit.snake1_alpha = LoadVector(store, prefix + ".block.0.alpha", channels);
    unit.conv1 = LoadConv(store, prefix + ".block.1.conv", channels, channels,
                          7, 1, dilation);
    unit.snake2_alpha = LoadVector(store, prefix + ".block.2.alpha", channels);
    unit.conv2 =
        LoadConv(store, prefix + ".block.3.conv", channels, channels, 1, 1, 1);
    return unit;
  }

  ConvNeXtWeights LoadConvNeXt(const ModelSafetensors& store,
                               const std::string& prefix,
                               std::size_t channels) {
    ConvNeXtWeights cn;
    cn.depthwise = LoadConv(store, prefix + ".dwconv.conv", channels, channels,
                            7, 1, 1, true);
    cn.norm_weight = LoadVector(store, prefix + ".norm.weight", channels);
    cn.norm_bias = LoadVector(store, prefix + ".norm.bias", channels);
    cn.pwconv1 =
        LoadLinear(store, prefix + ".pwconv1", channels, channels * 4U, true);
    cn.pwconv2 =
        LoadLinear(store, prefix + ".pwconv2", channels * 4U, channels, true);
    cn.gamma = LoadVector(store, prefix + ".gamma", channels);
    return cn;
  }

  TransformerWeights LoadTransformer(const ModelSafetensors& store,
                                     const std::string& prefix,
                                     std::size_t num_layers,
                                     std::size_t sliding_window) {
    TransformerWeights tf;
    tf.sliding_window = sliding_window;
    tf.layers.reserve(num_layers);
    for (std::size_t l = 0; l < num_layers; ++l) {
      const std::string lp = prefix + ".layers." + std::to_string(l);
      TransformerLayerWeights lw;
      lw.attention_norm =
          LoadVector(store, lp + ".attention_norm.weight", kCodecDim);
      lw.q_proj = LoadLinear(store, lp + ".attention.q_proj", kCodecDim,
                             kCodecDim, false);
      lw.k_proj = LoadLinear(store, lp + ".attention.k_proj", kCodecDim,
                             kCodecDim, false);
      lw.v_proj = LoadLinear(store, lp + ".attention.v_proj", kCodecDim,
                             kCodecDim, false);
      lw.wo =
          LoadLinear(store, lp + ".attention.wo", kCodecDim, kCodecDim, false);
      lw.attention_scale =
          LoadVector(store, lp + ".attention_layer_scale.gamma", kCodecDim);
      lw.ffn_norm = LoadVector(store, lp + ".ffn_norm.weight", kCodecDim);
      lw.w1 = LoadLinear(store, lp + ".feed_forward.w1", kCodecDim, kCodecInter,
                         false);
      lw.w2 = LoadLinear(store, lp + ".feed_forward.w2", kCodecInter, kCodecDim,
                         false);
      lw.w3 = LoadLinear(store, lp + ".feed_forward.w3", kCodecDim, kCodecInter,
                         false);
      lw.ffn_scale =
          LoadVector(store, lp + ".ffn_layer_scale.gamma", kCodecDim);
      tf.layers.push_back(std::move(lw));
    }
    tf.norm = LoadVector(store, prefix + ".norm.weight", kCodecDim);
    return tf;
  }

  QuantizerUnitWeights LoadQuantizerUnit(const ModelSafetensors& store,
                                         const std::string& prefix,
                                         std::size_t cb_size) {
    QuantizerUnitWeights q;
    q.codebook_size = cb_size;
    const float* in_w = RequireF32(store, prefix + ".in_proj.weight",
                                   {kCodebookDim, kCodecDim, 1});
    const float* in_b =
        RequireF32(store, prefix + ".in_proj.bias", {kCodebookDim});
    const float* out_w = RequireF32(store, prefix + ".out_proj.weight",
                                    {kCodecDim, kCodebookDim, 1});
    const float* out_b =
        RequireF32(store, prefix + ".out_proj.bias", {kCodecDim});
    const float* cb_w =
        RequireF32(store, prefix + ".codebook.weight", {cb_size, kCodebookDim});

    q.in_proj_weight.Upload(in_w, kCodebookDim * kCodecDim, stream);
    q.in_proj_bias.Upload(in_b, kCodebookDim, stream);
    q.out_proj_weight.Upload(out_w, kCodecDim * kCodebookDim, stream);
    q.out_proj_bias.Upload(out_b, kCodecDim, stream);
    q.codebook.Upload(cb_w, cb_size * kCodebookDim, stream);
    const auto norm_cb = NormalizeCodebookRows(cb_w, cb_size, kCodebookDim);
    q.normalized_codebook.Upload(norm_cb.data(), norm_cb.size(), stream);
    return q;
  }

  void LoadWeights(const ModelSafetensors& store) {
    // 1. Encoder
    encoder_first = LoadConv(store, "encoder.block.0.conv", 1, 64, 7, 1, 1);
    const std::array<std::size_t, 4> enc_strides{2, 4, 8, 8};
    std::size_t enc_ch = 64;
    for (std::size_t b = 0; b < 4; ++b) {
      const std::string bp =
          "encoder.block." + std::to_string(b + 1) + ".block";
      EncoderBlockWeights block;
      block.residuals[0] = LoadResidualUnit(store, bp + ".0", enc_ch, 1);
      block.residuals[1] = LoadResidualUnit(store, bp + ".1", enc_ch, 3);
      block.residuals[2] = LoadResidualUnit(store, bp + ".2", enc_ch, 9);
      block.snake_alpha = LoadVector(store, bp + ".3.alpha", enc_ch);
      block.downsample = LoadConv(store, bp + ".4.conv", enc_ch, enc_ch * 2U,
                                  2U * enc_strides[b], enc_strides[b], 1);
      enc_ch *= 2U;
      if (b == 3) {
        block.transformer =
            LoadTransformer(store, bp + ".5", 4, kEncoderTransformerWindow);
      }
      encoder_blocks.push_back(std::move(block));
    }
    encoder_final_snake = LoadVector(store, "encoder.block.5.alpha", kCodecDim);
    encoder_final =
        LoadConv(store, "encoder.block.6.conv", kCodecDim, kCodecDim, 3, 1, 1);

    // 2. Quantizer downsample & pre_module
    for (std::size_t i = 0; i < 2; ++i) {
      const std::string dp = "quantizer.downsample." + std::to_string(i);
      DownsampleStageWeights ds;
      ds.downsample =
          LoadConv(store, dp + ".0.conv", kCodecDim, kCodecDim, 2, 2, 1);
      ds.convnext = LoadConvNeXt(store, dp + ".1", kCodecDim);
      downsample_stages.push_back(std::move(ds));
    }
    pre_module =
        LoadTransformer(store, "quantizer.pre_module", 8, kPreModuleWindow);

    // 3. Quantizer codebooks
    semantic_quantizer =
        LoadQuantizerUnit(store, "quantizer.semantic_quantizer.quantizers.0",
                          config.semantic_codebook_size);
    residual_quantizers.reserve(config.quantizer_codebooks);
    for (std::size_t i = 0; i < config.quantizer_codebooks; ++i) {
      residual_quantizers.push_back(LoadQuantizerUnit(
          store, "quantizer.quantizer.quantizers." + std::to_string(i),
          config.residual_codebook_size));
    }

    // 4. Quantizer post_module & upsample
    post_module =
        LoadTransformer(store, "quantizer.post_module", 8, kPostModuleWindow);
    for (std::size_t i = 0; i < 2; ++i) {
      const std::string up = "quantizer.upsample." + std::to_string(i);
      UpsampleStageWeights us;
      us.upsample =
          LoadTransposeConv(store, up + ".0.conv", kCodecDim, kCodecDim, 2, 2);
      us.convnext = LoadConvNeXt(store, up + ".1", kCodecDim);
      upsample_stages.push_back(std::move(us));
    }

    // 5. Decoder
    decoder_first =
        LoadConv(store, "decoder.model.0.conv", kCodecDim, 1536, 7, 1, 1);
    const std::array<std::size_t, 4> dec_strides{8, 8, 4, 2};
    std::size_t dec_ch = 1536;
    for (std::size_t b = 0; b < 4; ++b) {
      const std::string bp =
          "decoder.model." + std::to_string(b + 1) + ".block";
      DecoderBlockWeights block;
      block.snake_alpha = LoadVector(store, bp + ".0.alpha", dec_ch);
      block.upsample =
          LoadTransposeConv(store, bp + ".1.conv", dec_ch, dec_ch / 2U,
                            2U * dec_strides[b], dec_strides[b]);
      dec_ch /= 2U;
      block.residuals[0] = LoadResidualUnit(store, bp + ".2", dec_ch, 1);
      block.residuals[1] = LoadResidualUnit(store, bp + ".3", dec_ch, 3);
      block.residuals[2] = LoadResidualUnit(store, bp + ".4", dec_ch, 9);
      decoder_blocks.push_back(std::move(block));
    }
    decoder_final_snake = LoadVector(store, "decoder.model.5.alpha", 96);
    decoder_final = LoadConv(store, "decoder.model.6.conv", 96, 1, 7, 1, 1);
  }

  void RunLinear(const LinearWeights& weights, const float* input,
                 float* output, std::size_t rows) {
    gemm.Run(weights.weight.get(), input, output, rows, weights.output_columns,
             weights.input_columns, stream);
    if (weights.bias.size() > 0) {
      LaunchCodecAddBias(output, weights.bias.get(), rows,
                         weights.output_columns, stream);
    }
  }

  void RunConvRaw(const ConvWeights& weights, const float* input, float* output,
                  std::size_t input_length, std::size_t output_length) {
    if (weights.depthwise) {
      LaunchCodecCausalDepthwiseConv1d(
          input, weights.weight.get(), weights.bias.get(), output, input_length,
          weights.input_channels, weights.kernel, weights.dilation, stream);
      return;
    }
    if (weights.transpose) {
      const std::size_t expanded_columns =
          weights.output_channels * weights.kernel;
      gemm.Run(weights.weight.get(), input, workspace.columns.get(),
               input_length, expanded_columns, weights.input_channels, stream);
      LaunchCodecAssembleCausalConvTranspose1d(
          workspace.columns.get(), weights.bias.get(), output, input_length,
          output_length, weights.output_channels, weights.kernel,
          weights.stride, stream);
      return;
    }
    if (weights.kernel == 1U && weights.stride == 1U) {
      gemm.Run(weights.weight.get(), input, output, output_length,
               weights.output_channels, weights.input_channels, stream);
      LaunchCodecAddBias(output, weights.bias.get(), output_length,
                         weights.output_channels, stream);
      return;
    }
    const std::size_t reduction = weights.input_channels * weights.kernel;
    LaunchCodecCausalConv1dIm2Col(input, workspace.columns.get(), input_length,
                                  weights.input_channels, output_length,
                                  weights.kernel, weights.stride,
                                  weights.dilation, stream);
    gemm.Run(weights.weight.get(), workspace.columns.get(), output,
             output_length, weights.output_channels, reduction, stream);
    LaunchCodecAddBias(output, weights.bias.get(), output_length,
                       weights.output_channels, stream);
  }

  void RunConv(const ConvWeights& weights, const float* input, float* output,
               std::size_t input_length, std::size_t output_length) {
    const std::size_t keep =
        weights.transpose
            ? (weights.kernel - 1U) / weights.stride
            : (weights.kernel - 1U) * weights.dilation + 1U - weights.stride;
    if (active_stream == nullptr || keep == 0) {
      RunConvRaw(weights, input, output, input_length, output_length);
      return;
    }
    auto& history = active_stream->convolutions[&weights];
    if (history.input.size() < keep * weights.input_channels) {
      history.input.Reset(keep * weights.input_channels);
    }
    // Fish DAC streaming zero-pads the initial history of `keep` frames on the
    // first chunk (see CodecStreamingState::prepare in
    // fish_dac_codec_runtime.cpp).
    const std::size_t previous = keep;
    if (history.rows == 0) {
      RequireHip(
          hipMemsetAsync(history.input.get(), 0,
                         keep * weights.input_channels * sizeof(float), stream),
          "zero Fish DAC conv history");
      history.rows = keep;
    }
    const std::size_t rows = previous + input_length;
    const std::size_t stride = weights.transpose ? weights.stride : 1U;
    const std::size_t input_elements = rows * weights.input_channels;
    const std::size_t output_elements = rows * stride * weights.output_channels;
    if (workspace.history_input.size() < input_elements) {
      workspace.history_input.Reset(input_elements);
    }
    if (workspace.history_output.size() < output_elements) {
      workspace.history_output.Reset(output_elements);
    }
    float* joined = workspace.history_input.get();
    RequireHip(hipMemcpyAsync(joined, history.input.get(),
                              previous * weights.input_channels * sizeof(float),
                              hipMemcpyDeviceToDevice, stream),
               "copy Fish DAC conv history");
    RequireHip(
        hipMemcpyAsync(joined + previous * weights.input_channels, input,
                       input_length * weights.input_channels * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "append Fish DAC conv input");
    RequireHip(hipMemcpyAsync(history.input.get(),
                              joined + input_length * weights.input_channels,
                              keep * weights.input_channels * sizeof(float),
                              hipMemcpyDeviceToDevice, stream),
               "retain Fish DAC conv history");
    RunConvRaw(weights, joined, workspace.history_output.get(), rows,
               rows * stride);
    RequireHip(
        hipMemcpyAsync(output,
                       workspace.history_output.get() +
                           previous * stride * weights.output_channels,
                       output_length * weights.output_channels * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "copy Fish DAC incremental conv output");
  }

  float* RunConvNeXt(const ConvNeXtWeights& cn, float* input,
                     std::size_t length) {
    float* depthwise = FindScratch(workspace, {input});
    RunConv(cn.depthwise, input, depthwise, length, length);
    float* normed = FindScratch(workspace, {input, depthwise});
    LaunchCodecLayerNorm(depthwise, cn.norm_weight.get(), cn.norm_bias.get(),
                         normed, length, kCodecDim, kConvNextNormEps, stream);
    float* wide = FindScratch(workspace, {input, depthwise, normed});
    RunLinear(cn.pwconv1, normed, wide, length);
    LaunchCodecGelu(wide, length * kCodecDim * 4U, stream);
    RunLinear(cn.pwconv2, wide, depthwise, length);
    LaunchCodecAddScaledChannels(input, depthwise, cn.gamma.get(), length,
                                 kCodecDim, stream);
    return input;
  }

  float* RunResidualUnit(const ResidualUnitWeights& unit, float* input,
                         std::size_t length, std::size_t channels) {
    float* branch = FindScratch(workspace, {input});
    LaunchCodecSnake1d(input, unit.snake1_alpha.get(), branch, length, channels,
                       stream);
    float* work = FindScratch(workspace, {input, branch});
    RunConv(unit.conv1, branch, work, length, length);
    LaunchCodecSnake1d(work, unit.snake2_alpha.get(), branch, length, channels,
                       stream);
    RunConv(unit.conv2, branch, work, length, length);
    LaunchCodecAdd(input, work, input, length * channels, stream);
    return input;
  }

  float* RunTransformer(const TransformerWeights& tf, float* hidden,
                        std::size_t frames, bool use_stream_cache) {
    if (use_stream_cache && active_stream != nullptr) {
      active_stream->post_module_attention.resize(tf.layers.size());
    }
    for (std::size_t l = 0; l < tf.layers.size(); ++l) {
      const auto& layer = tf.layers[l];
      float* normed = FindScratch(workspace, {hidden});
      LaunchCodecRmsNorm(hidden, layer.attention_norm.get(), normed, frames,
                         kCodecDim, kCodecNormEps, stream);
      float* q = FindScratch(workspace, {hidden, normed});
      float* k = FindScratch(workspace, {hidden, normed, q});
      float* v = FindScratch(workspace, {hidden, normed, q, k});
      RunLinear(layer.q_proj, normed, q, frames);
      RunLinear(layer.k_proj, normed, k, frames);
      RunLinear(layer.v_proj, normed, v, frames);

      std::size_t start_pos = 0;
      const float* k_ctx = k;
      const float* v_ctx = v;
      std::size_t query_offset = 0;
      std::size_t key_frames = frames;

      if (use_stream_cache && active_stream != nullptr) {
        auto& cache = active_stream->post_module_attention[l];
        start_pos = cache.frames;
        LaunchCodecInterleavedRope(q, k, frames, kCodecHeads, kCodecHeadDim,
                                   kCodecRopeTheta, stream, start_pos);
        const std::size_t keep_prev =
            std::min<std::size_t>(cache.frames, tf.sliding_window - 1U);
        key_frames = keep_prev + frames;
        query_offset = keep_prev;
        const std::size_t needed_cap =
            std::max<std::size_t>(key_frames * kCodecDim, 256U * kCodecDim);
        if (cache.key.size() < needed_cap) {
          DeviceBuffer<float> new_k(needed_cap);
          DeviceBuffer<float> new_v(needed_cap);
          if (keep_prev > 0) {
            const std::size_t src_off = (cache.frames - keep_prev) * kCodecDim;
            RequireHip(hipMemcpyAsync(new_k.get(), cache.key.get() + src_off,
                                      keep_prev * kCodecDim * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream),
                       "grow codec KV cache K");
            RequireHip(hipMemcpyAsync(new_v.get(), cache.value.get() + src_off,
                                      keep_prev * kCodecDim * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream),
                       "grow codec KV cache V");
          }
          cache.key = std::move(new_k);
          cache.value = std::move(new_v);
        } else if (keep_prev > 0 && cache.frames > keep_prev) {
          const std::size_t src_off = (cache.frames - keep_prev) * kCodecDim;
          // Slide window left via scratch buffer
          float* tmp = FindScratch(workspace, {hidden, normed, q, k, v});
          RequireHip(hipMemcpyAsync(tmp, cache.key.get() + src_off,
                                    keep_prev * kCodecDim * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream),
                     "slide codec K tmp");
          RequireHip(hipMemcpyAsync(cache.key.get(), tmp,
                                    keep_prev * kCodecDim * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream),
                     "slide codec K");
          RequireHip(hipMemcpyAsync(tmp, cache.value.get() + src_off,
                                    keep_prev * kCodecDim * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream),
                     "slide codec V tmp");
          RequireHip(hipMemcpyAsync(cache.value.get(), tmp,
                                    keep_prev * kCodecDim * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream),
                     "slide codec V");
        }
        RequireHip(hipMemcpyAsync(cache.key.get() + keep_prev * kCodecDim, k,
                                  frames * kCodecDim * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream),
                   "append codec K");
        RequireHip(hipMemcpyAsync(cache.value.get() + keep_prev * kCodecDim, v,
                                  frames * kCodecDim * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream),
                   "append codec V");
        cache.frames += frames;
        k_ctx = cache.key.get();
        v_ctx = cache.value.get();
      } else {
        LaunchCodecInterleavedRope(q, k, frames, kCodecHeads, kCodecHeadDim,
                                   kCodecRopeTheta, stream, 0);
      }

      float* attn_out = FindScratch(workspace, {hidden, normed, q, k, v});
      LaunchCodecSlidingCausalAttention(
          q, k_ctx, v_ctx, attn_out, frames, kCodecHeads, kCodecHeadDim,
          tf.sliding_window, stream, query_offset, key_frames);
      RunLinear(layer.wo, attn_out, normed, frames);
      LaunchCodecAddScaledChannels(hidden, normed, layer.attention_scale.get(),
                                   frames, kCodecDim, stream);

      LaunchCodecRmsNorm(hidden, layer.ffn_norm.get(), normed, frames,
                         kCodecDim, kCodecNormEps, stream);
      float* gate = q;
      float* up = k;
      float* act = v;
      RunLinear(layer.w1, normed, gate, frames);
      RunLinear(layer.w3, normed, up, frames);
      LaunchCodecSwiGlu(gate, up, act, frames * kCodecInter, stream);
      RunLinear(layer.w2, act, normed, frames);
      LaunchCodecAddScaledChannels(hidden, normed, layer.ffn_scale.get(),
                                   frames, kCodecDim, stream);
    }
    float* out = FindScratch(workspace, {hidden});
    LaunchCodecRmsNorm(hidden, tf.norm.get(), out, frames, kCodecDim,
                       kCodecNormEps, stream);
    return out;
  }

  std::vector<float> RunDecodePass(const FishDacCodes& codes) {
    const std::size_t frames = static_cast<std::size_t>(codes.frames);
    const std::size_t total_samples = frames * config.frame_length;
    workspace.EnsureForSamples((frames + 16U) *
                               static_cast<std::size_t>(config.frame_length));
    workspace.codes.Upload(codes.codes.data(), codes.codes.size(), stream);

    float* z_q = workspace.scratch[0].get();
    LaunchCodecDequantizeCodebook(workspace.codes.get(),
                                  semantic_quantizer.codebook_size,
                                  semantic_quantizer.codebook.get(),
                                  semantic_quantizer.out_proj_weight.get(),
                                  semantic_quantizer.out_proj_bias.get(), z_q,
                                  frames, kCodecDim, true, stream);
    for (std::size_t i = 0; i < residual_quantizers.size(); ++i) {
      const auto& rq = residual_quantizers[i];
      LaunchCodecDequantizeCodebook(
          workspace.codes.get() + (i + 1U) * frames, rq.codebook_size,
          rq.codebook.get(), rq.out_proj_weight.get(), rq.out_proj_bias.get(),
          z_q, frames, kCodecDim, false, stream);
    }

    float* hidden =
        RunTransformer(post_module, z_q, frames, active_stream != nullptr);
    std::size_t length = frames;
    for (const auto& stage : upsample_stages) {
      const std::size_t next_len = length * stage.upsample.stride;
      float* upsampled = FindScratch(workspace, {hidden});
      RunConv(stage.upsample, hidden, upsampled, length, next_len);
      hidden = RunConvNeXt(stage.convnext, upsampled, next_len);
      length = next_len;
    }

    float* dec_in = FindScratch(workspace, {hidden});
    RunConv(decoder_first, hidden, dec_in, length, length);
    hidden = dec_in;
    std::size_t channels = 1536;
    for (const auto& block : decoder_blocks) {
      float* act = FindScratch(workspace, {hidden});
      LaunchCodecSnake1d(hidden, block.snake_alpha.get(), act, length, channels,
                         stream);
      const std::size_t next_len = length * block.upsample.stride;
      float* upsampled = FindScratch(workspace, {hidden, act});
      RunConv(block.upsample, act, upsampled, length, next_len);
      channels /= 2U;
      hidden = upsampled;
      length = next_len;
      for (const auto& res : block.residuals) {
        hidden = RunResidualUnit(res, hidden, length, channels);
      }
    }

    float* final_act = FindScratch(workspace, {hidden});
    LaunchCodecSnake1d(hidden, decoder_final_snake.get(), final_act, length,
                       96U, stream);
    float* waveform = FindScratch(workspace, {hidden, final_act});
    RunConv(decoder_final, final_act, waveform, length, length);
    LaunchCodecTanh(waveform, length, stream);

    std::vector<float> out_samples(total_samples);
    RequireHip(hipMemcpyAsync(out_samples.data(), waveform,
                              total_samples * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "copy Fish DAC decoded waveform");
    RequireHip(hipStreamSynchronize(stream), "synchronize Fish DAC decode");
    return out_samples;
  }

  FishDacCodes RunEncodePass(std::span<const float> mono_samples) {
    const std::size_t frame_len = config.frame_length;
    const std::size_t frames =
        (mono_samples.size() + frame_len - 1U) / frame_len;
    const std::size_t padded_samples = frames * frame_len;
    workspace.EnsureForSamples(padded_samples);

    std::vector<float> padded(padded_samples, 0.0F);
    std::copy(mono_samples.begin(), mono_samples.end(), padded.begin());

    float* hidden = workspace.scratch[0].get();
    RequireHip(
        hipMemcpyAsync(hidden, padded.data(), padded_samples * sizeof(float),
                       hipMemcpyHostToDevice, stream),
        "upload Fish DAC encoder input");

    float* next = FindScratch(workspace, {hidden});
    RunConvRaw(encoder_first, hidden, next, padded_samples, padded_samples);
    hidden = next;
    std::size_t length = padded_samples;
    std::size_t channels = 64;

    for (const auto& block : encoder_blocks) {
      for (const auto& res : block.residuals) {
        hidden = RunResidualUnit(res, hidden, length, channels);
      }
      float* act = FindScratch(workspace, {hidden});
      LaunchCodecSnake1d(hidden, block.snake_alpha.get(), act, length, channels,
                         stream);
      const std::size_t next_len = length / block.downsample.stride;
      float* down = FindScratch(workspace, {hidden, act});
      RunConvRaw(block.downsample, act, down, length, next_len);
      hidden = down;
      length = next_len;
      channels *= 2U;
      if (block.transformer.has_value()) {
        hidden = RunTransformer(*block.transformer, hidden, length, false);
      }
    }

    float* final_act = FindScratch(workspace, {hidden});
    LaunchCodecSnake1d(hidden, encoder_final_snake.get(), final_act, length,
                       kCodecDim, stream);
    float* enc_out = FindScratch(workspace, {hidden, final_act});
    RunConvRaw(encoder_final, final_act, enc_out, length, length);
    hidden = enc_out;

    for (const auto& stage : downsample_stages) {
      const std::size_t next_len = length / stage.downsample.stride;
      float* down = FindScratch(workspace, {hidden});
      RunConvRaw(stage.downsample, hidden, down, length, next_len);
      hidden = RunConvNeXt(stage.convnext, down, next_len);
      length = next_len;
    }

    hidden = RunTransformer(pre_module, hidden, length, false);
    if (length != frames) {
      throw std::runtime_error("Fish DAC encoder output frame count mismatch");
    }

    const std::size_t total_cb = config.total_codebooks;
    workspace.codes.Reset(total_cb * frames);
    LaunchCodecQuantizeCodebookStep(
        hidden, semantic_quantizer.in_proj_weight.get(),
        semantic_quantizer.in_proj_bias.get(),
        semantic_quantizer.codebook.get(),
        semantic_quantizer.normalized_codebook.get(),
        semantic_quantizer.out_proj_weight.get(),
        semantic_quantizer.out_proj_bias.get(), workspace.codes.get(), frames,
        kCodecDim, semantic_quantizer.codebook_size, stream);

    for (std::size_t i = 0; i < residual_quantizers.size(); ++i) {
      const auto& rq = residual_quantizers[i];
      LaunchCodecQuantizeCodebookStep(
          hidden, rq.in_proj_weight.get(), rq.in_proj_bias.get(),
          rq.codebook.get(), rq.normalized_codebook.get(),
          rq.out_proj_weight.get(), rq.out_proj_bias.get(),
          workspace.codes.get() + (i + 1U) * frames, frames, kCodecDim,
          rq.codebook_size, stream);
    }

    FishDacCodes result;
    result.codebooks = static_cast<std::int64_t>(total_cb);
    result.frames = static_cast<std::int64_t>(frames);
    result.codes.resize(total_cb * frames);
    RequireHip(hipMemcpyAsync(result.codes.data(), workspace.codes.get(),
                              result.codes.size() * sizeof(std::int32_t),
                              hipMemcpyDeviceToHost, stream),
               "download Fish DAC encoded codes");
    RequireHip(hipStreamSynchronize(stream), "synchronize Fish DAC encode");
    return result;
  }
};

FishDacCodecHipRuntime::FishDacCodecHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

FishDacCodecHipRuntime::~FishDacCodecHipRuntime() = default;

std::unique_ptr<FishDacCodecHipRuntime> FishDacCodecHipRuntime::Create(
    const std::string& model_dir, const CodecConfig& config,
    std::string* error) {
  try {
    return std::unique_ptr<FishDacCodecHipRuntime>(
        new FishDacCodecHipRuntime(std::make_unique<Impl>(model_dir, config)));
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return nullptr;
  }
}

bool FishDacCodecHipRuntime::Encode(std::span<const float> mono_samples,
                                    FishDacCodes* output, std::string* error) {
  if (output == nullptr || mono_samples.empty()) {
    SetError(error, "Fish DAC Encode requires non-empty mono samples");
    return false;
  }
  try {
    impl_->active_stream = nullptr;
    *output = impl_->RunEncodePass(mono_samples);
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

bool FishDacCodecHipRuntime::Decode(const FishDacCodes& codes,
                                    std::vector<float>* output_samples,
                                    std::string* error) {
  if (output_samples == nullptr || codes.frames <= 0 ||
      codes.codebooks !=
          static_cast<std::int64_t>(impl_->config.total_codebooks) ||
      codes.codes.size() !=
          static_cast<std::size_t>(codes.codebooks * codes.frames)) {
    SetError(error, "Invalid FishDacCodes shape for Decode");
    return false;
  }
  try {
    impl_->active_stream = nullptr;
    *output_samples = impl_->RunDecodePass(codes);
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

void FishDacCodecHipRuntime::ResetStream() {
  impl_->stream_state.Reset();
}

bool FishDacCodecHipRuntime::DecodeStreamChunk(
    const FishDacCodes& chunk_codes, std::vector<float>* output_samples,
    std::string* error) {
  if (output_samples == nullptr || chunk_codes.frames <= 0 ||
      chunk_codes.codebooks !=
          static_cast<std::int64_t>(impl_->config.total_codebooks) ||
      chunk_codes.codes.size() !=
          static_cast<std::size_t>(chunk_codes.codebooks *
                                   chunk_codes.frames)) {
    SetError(error, "Invalid FishDacCodes shape for DecodeStreamChunk");
    return false;
  }
  try {
    impl_->active_stream = &impl_->stream_state;
    struct Guard {
      Impl* impl;
      ~Guard() { impl->active_stream = nullptr; }
    } guard{impl_.get()};
    *output_samples = impl_->RunDecodePass(chunk_codes);
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

std::uint32_t FishDacCodecHipRuntime::sample_rate() const noexcept {
  return impl_->config.sample_rate;
}

std::uint32_t FishDacCodecHipRuntime::frame_length() const noexcept {
  return impl_->config.frame_length;
}

}  // namespace gufo::models::fish_drama3::hip

#else

namespace gufo::models::fish_drama3::hip {

struct FishDacCodecHipRuntime::Impl {};

FishDacCodecHipRuntime::FishDacCodecHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
FishDacCodecHipRuntime::~FishDacCodecHipRuntime() = default;

std::unique_ptr<FishDacCodecHipRuntime> FishDacCodecHipRuntime::Create(
    const std::string&, const CodecConfig&, std::string* error) {
  if (error != nullptr) {
    *error = "Fish DAC codec runtime requires a HIP-enabled build";
  }
  return nullptr;
}

bool FishDacCodecHipRuntime::Encode(std::span<const float>, FishDacCodes*,
                                    std::string* error) {
  if (error != nullptr) {
    *error = "Fish DAC codec runtime requires a HIP-enabled build";
  }
  return false;
}

bool FishDacCodecHipRuntime::Decode(const FishDacCodes&, std::vector<float>*,
                                    std::string* error) {
  if (error != nullptr) {
    *error = "Fish DAC codec runtime requires a HIP-enabled build";
  }
  return false;
}

void FishDacCodecHipRuntime::ResetStream() {}

bool FishDacCodecHipRuntime::DecodeStreamChunk(const FishDacCodes&,
                                               std::vector<float>*,
                                               std::string* error) {
  if (error != nullptr) {
    *error = "Fish DAC codec runtime requires a HIP-enabled build";
  }
  return false;
}

std::uint32_t FishDacCodecHipRuntime::sample_rate() const noexcept {
  return 44100;
}
std::uint32_t FishDacCodecHipRuntime::frame_length() const noexcept {
  return 2048;
}

}  // namespace gufo::models::fish_drama3::hip

#endif
