#include "src/models/fish_drama3/hip/tower_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/fish_drama3/hip/tower_ops.hpp"

namespace gufo::models::fish_drama3::hip {
namespace {

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

float HostBf16ToFloat(std::uint16_t bits) {
  const std::uint32_t u32 = static_cast<std::uint32_t>(bits) << 16U;
  float val = 0.0F;
  std::memcpy(&val, &u32, sizeof(float));
  return val;
}

std::uint16_t HostFloatToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(float));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16U);
}

bool IsSemanticFrame(const Drama3Frame& frame, const TowerConfig& cfg) {
  return frame[0] >= cfg.semantic_begin &&
         frame[0] <
             cfg.semantic_begin + static_cast<std::int32_t>(cfg.codebook_size);
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
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Drama 3 tower device buffer overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Drama 3 tower");
    count_ = count;
  }

  void Upload(const Element* host_data, std::size_t count,
              hipStream_t stream = nullptr) {
    Reset(count);
    if (count > 0) {
      if (stream != nullptr) {
        RequireHip(hipMemcpyAsync(data_, host_data, count * sizeof(Element),
                                  hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync Drama 3 tower");
      } else {
        RequireHip(hipMemcpy(data_, host_data, count * sizeof(Element),
                             hipMemcpyHostToDevice),
                   "hipMemcpy Drama 3 tower");
      }
    }
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

struct LayerWeights {
  DeviceBuffer<std::uint16_t> embedding_bf16;  // [codebook_size, hidden_size]
  DeviceBuffer<float> enorm;                   // [hidden_size] (1 + w)
  DeviceBuffer<float> hnorm;                   // [hidden_size] (1 + w)
  DeviceBuffer<std::uint16_t> eh_bf16;         // [hidden_size, 2 * hidden_size]
  DeviceBuffer<float> input_norm;              // [hidden_size] (1 + w)
  DeviceBuffer<std::uint16_t>
      qkv_bf16;                         // [2*q_width + 2*kv_width, hidden_size]
  DeviceBuffer<float> q_norm;           // [head_dim] (1 + w)
  DeviceBuffer<float> k_norm;           // [head_dim] (1 + w)
  DeviceBuffer<std::uint16_t> wo_bf16;  // [hidden_size, q_width]
  DeviceBuffer<float> post_norm;        // [hidden_size] (1 + w)
  DeviceBuffer<std::uint16_t> w1_bf16;  // [intermediate_size, hidden_size]
  DeviceBuffer<std::uint16_t> w3_bf16;  // [intermediate_size, hidden_size]
  DeviceBuffer<std::uint16_t> w2_bf16;  // [hidden_size, intermediate_size]
  DeviceBuffer<float> final_norm;       // [hidden_size] (1 + w)
  DeviceBuffer<std::uint16_t>
      head_bf16;  // [codebook_size, hidden_size] (residual layers only)

  DeviceBuffer<float> key_cache;    // [capacity, kv_heads * head_dim]
  DeviceBuffer<float> value_cache;  // [capacity, kv_heads * head_dim]
};

struct Workspace {
  std::size_t max_steps{0};
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> next_hidden;
  DeviceBuffer<float> extra;
  DeviceBuffer<float> label_mask;
  DeviceBuffer<std::int32_t> codes;
  DeviceBuffer<std::int32_t> sampled_books;
  DeviceBuffer<float> concat_eh;
  DeviceBuffer<float> mixed;
  DeviceBuffer<float> normed;
  DeviceBuffer<float> qkv;
  DeviceBuffer<float> q;
  DeviceBuffer<float> attended;
  DeviceBuffer<float> ffn_gate;
  DeviceBuffer<float> ffn_up;
  DeviceBuffer<float> ffn_act;
  DeviceBuffer<float> logits;
  DeviceBuffer<std::uint16_t> bf16_in;

  void Ensure(std::size_t steps, const TowerConfig& cfg) {
    if (steps <= max_steps) {
      return;
    }
    const std::size_t dim = cfg.hidden_size;
    const std::size_t q_width = cfg.heads * cfg.head_dim;
    const std::size_t kv_width = cfg.kv_heads * cfg.head_dim;
    const std::size_t qkv_width = 2U * q_width + 2U * kv_width;
    const std::size_t inter = cfg.intermediate_size;

    hidden.Reset(steps * dim);
    next_hidden.Reset(steps * dim);
    extra.Reset(steps * dim);
    label_mask.Reset(steps);
    codes.Reset(steps);
    sampled_books.Reset(cfg.residual_layers);
    concat_eh.Reset(steps * 2U * dim);
    mixed.Reset(steps * dim);
    normed.Reset(steps * dim);
    qkv.Reset(steps * qkv_width);
    q.Reset(steps * q_width);
    attended.Reset(steps * q_width);
    ffn_gate.Reset(steps * inter);
    ffn_up.Reset(steps * inter);
    ffn_act.Reset(steps * inter);
    logits.Reset(steps * cfg.codebook_size);
    bf16_in.Reset(steps * std::max({2U * dim, q_width, inter}));
    max_steps = steps;
  }
};

}  // namespace

struct TowerHipRuntime::Impl {
  TowerConfig cfg;
  std::size_t capacity{4096};
  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};
  std::vector<float>
      code_embeddings_f32;  // [10 * codebook_size, hidden_size] on host
  std::vector<LayerWeights> layers;
  Workspace ws;

  Impl(const ModelSafetensors& store, const TowerConfig& config,
       std::size_t max_cache_steps)
      : cfg(config), capacity(max_cache_steps) {
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags Drama 3 tower");
    RequireRocblas(rocblas_create_handle(&blas),
                   "rocblas_create_handle Drama 3 tower");
    RequireRocblas(rocblas_set_atomics_mode(blas, rocblas_atomics_not_allowed),
                   "rocblas_set_atomics_mode Drama 3 tower");
    RequireRocblas(rocblas_set_stream(blas, stream),
                   "rocblas_set_stream Drama 3 tower");
    LoadWeights(store);
    ws.Ensure(64, cfg);
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Drama 3 tower init");
  }

  ~Impl() {
    if (blas != nullptr) {
      (void)rocblas_destroy_handle(blas);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  std::vector<float> ReadTensorAsF32(
      const ModelSafetensors& store, const std::string& name,
      std::initializer_list<std::uint64_t> expected_shape) {
    const TensorView& t = store.Require(name);
    if (!std::ranges::equal(t.shape, expected_shape)) {
      throw std::runtime_error("Unexpected shape for Drama 3 tower tensor: " +
                               name);
    }
    const std::size_t n = t.NumElements();
    std::vector<float> out(n);
    if (t.dtype == DType::kF32) {
      std::memcpy(out.data(), t.data, n * sizeof(float));
    } else if (t.dtype == DType::kBF16) {
      const auto* src = reinterpret_cast<const std::uint16_t*>(t.data);
      for (std::size_t i = 0; i < n; ++i) {
        out[i] = HostBf16ToFloat(src[i]);
      }
    } else {
      throw std::runtime_error("Unsupported dtype for Drama 3 tower tensor: " +
                               name);
    }
    return out;
  }

  DeviceBuffer<float> LoadGemmaNorm(const ModelSafetensors& store,
                                    const std::string& name,
                                    std::uint64_t length) {
    auto vals = ReadTensorAsF32(store, name, {length});
    for (float& v : vals) {
      v += 1.0F;
    }
    DeviceBuffer<float> buf;
    buf.Upload(vals.data(), vals.size(), stream);
    return buf;
  }

  DeviceBuffer<std::uint16_t> LoadBf16Matrix(const ModelSafetensors& store,
                                             const std::string& name,
                                             std::uint64_t rows,
                                             std::uint64_t cols) {
    const TensorView& t = store.Require(name);
    const std::array<std::uint64_t, 2> expected{rows, cols};
    if (!std::ranges::equal(t.shape, expected)) {
      throw std::runtime_error("Unexpected shape for Drama 3 tower matrix: " +
                               name);
    }
    const std::size_t n = t.NumElements();
    DeviceBuffer<std::uint16_t> buf;
    if (t.dtype == DType::kBF16) {
      buf.Upload(reinterpret_cast<const std::uint16_t*>(t.data), n, stream);
    } else if (t.dtype == DType::kF32) {
      const auto* src = reinterpret_cast<const float*>(t.data);
      std::vector<std::uint16_t> converted(n);
      for (std::size_t i = 0; i < n; ++i) {
        converted[i] = HostFloatToBf16(src[i]);
      }
      buf.Upload(converted.data(), n, stream);
    } else {
      throw std::runtime_error("Unsupported dtype for Drama 3 tower matrix: " +
                               name);
    }
    return buf;
  }

  void LoadWeights(const ModelSafetensors& store) {
    code_embeddings_f32 =
        ReadTensorAsF32(store, "audio_decoder.codebook_embeddings.weight",
                        {10ULL * cfg.codebook_size, cfg.hidden_size});

    const std::size_t total_layers = cfg.TotalLayers();
    const std::uint64_t q_width =
        static_cast<std::uint64_t>(cfg.heads) * cfg.head_dim;
    const std::uint64_t kv_width =
        static_cast<std::uint64_t>(cfg.kv_heads) * cfg.head_dim;

    layers.reserve(total_layers);
    for (std::size_t idx = 0; idx < total_layers; ++idx) {
      const bool shared = idx < cfg.shared_layers;
      const std::size_t local_idx = shared ? idx : (idx - cfg.shared_layers);
      const std::string prefix =
          std::string(shared ? "audio_decoder_mtp_shared.layers."
                             : "audio_decoder_mtp.layers.") +
          std::to_string(local_idx);
      LayerWeights w;
      w.embedding_bf16 =
          LoadBf16Matrix(store,
                         prefix + (shared ? ".semantic_embedding.weight"
                                          : ".codebook_embedding.weight"),
                         cfg.codebook_size, cfg.hidden_size);
      w.enorm = LoadGemmaNorm(store, prefix + ".enorm.weight", cfg.hidden_size);
      w.hnorm = LoadGemmaNorm(store, prefix + ".hnorm.weight", cfg.hidden_size);
      w.eh_bf16 = LoadBf16Matrix(store, prefix + ".eh_proj.weight",
                                 cfg.hidden_size, 2ULL * cfg.hidden_size);
      w.final_norm = LoadGemmaNorm(store, prefix + ".final_layernorm.weight",
                                   cfg.hidden_size);

      const std::string tf = prefix + ".transformer_layer";
      w.input_norm =
          LoadGemmaNorm(store, tf + ".attention_norm.weight", cfg.hidden_size);
      w.post_norm =
          LoadGemmaNorm(store, tf + ".ffn_norm.weight", cfg.hidden_size);
      w.qkv_bf16 =
          LoadBf16Matrix(store, tf + ".attention.wqkv.weight",
                         2ULL * q_width + 2ULL * kv_width, cfg.hidden_size);
      w.q_norm =
          LoadGemmaNorm(store, tf + ".attention.q_norm.weight", cfg.head_dim);
      w.k_norm =
          LoadGemmaNorm(store, tf + ".attention.k_norm.weight", cfg.head_dim);
      w.wo_bf16 = LoadBf16Matrix(store, tf + ".attention.wo.weight",
                                 cfg.hidden_size, q_width);
      w.w1_bf16 = LoadBf16Matrix(store, tf + ".feed_forward.w1.weight",
                                 cfg.intermediate_size, cfg.hidden_size);
      w.w3_bf16 = LoadBf16Matrix(store, tf + ".feed_forward.w3.weight",
                                 cfg.intermediate_size, cfg.hidden_size);
      w.w2_bf16 = LoadBf16Matrix(store, tf + ".feed_forward.w2.weight",
                                 cfg.hidden_size, cfg.intermediate_size);
      if (!shared) {
        w.head_bf16 = LoadBf16Matrix(
            store,
            "audio_decoder_mtp.heads." + std::to_string(local_idx) + ".weight",
            cfg.codebook_size, cfg.hidden_size);
      }
      w.key_cache.Reset(capacity * kv_width);
      w.value_cache.Reset(capacity * kv_width);
      layers.push_back(std::move(w));
    }
  }

  void AddCodeEmbeddings(float* out, const Drama3Frame& frame,
                         std::size_t begin_book, std::size_t end_book) const {
    for (std::size_t book = begin_book; book < end_book; ++book) {
      const std::int32_t code = frame[book + 1U];
      if (code < 0 || static_cast<std::uint32_t>(code) >= cfg.codebook_size) {
        throw std::runtime_error("Drama 3 frame code out of range");
      }
      const float* emb =
          code_embeddings_f32.data() +
          (book * cfg.codebook_size + static_cast<std::size_t>(code)) *
              cfg.hidden_size;
      for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
        out[c] += emb[c];
      }
    }
  }

  void RunLinearBf16(const std::uint16_t* weight_bf16, const float* input_f32,
                     float* output_f32, std::size_t batch, std::size_t rows,
                     std::size_t cols) {
    if (batch <= 4U) {
      LaunchTowerBf16Gemv(weight_bf16, input_f32, output_f32, batch, rows, cols,
                          stream);
      return;
    }
    LaunchTowerFloatToBf16(input_f32, ws.bf16_in.get(), batch * cols, stream);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    RequireRocblas(
        rocblas_gemm_ex(
            blas, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
            static_cast<rocblas_int>(cols), &alpha, weight_bf16,
            rocblas_datatype_bf16_r, static_cast<rocblas_int>(cols),
            ws.bf16_in.get(), rocblas_datatype_bf16_r,
            static_cast<rocblas_int>(cols), &beta, output_f32,
            rocblas_datatype_f32_r, static_cast<rocblas_int>(rows), output_f32,
            rocblas_datatype_f32_r, static_cast<rocblas_int>(rows),
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
        "rocblas_gemm_ex Drama 3 tower");
  }

  void RunLayerDevice(std::size_t layer_idx, std::size_t steps,
                      std::size_t start_pos, const float* hidden_in,
                      const std::int32_t* codes_in, const float* mask_in,
                      const float* extra_in, float* hidden_out) {
    const auto& w = layers[layer_idx];
    const std::size_t dim = cfg.hidden_size;
    const std::size_t q_width = cfg.heads * cfg.head_dim;
    const std::size_t kv_width = cfg.kv_heads * cfg.head_dim;
    const std::size_t qkv_stride = 2U * q_width + 2U * kv_width;
    const std::size_t inter = cfg.intermediate_size;

    // 1. Embed + enorm/hnorm + concat
    LaunchTowerEmbedAndConcatNorm(w.embedding_bf16.get(), codes_in, mask_in,
                                  extra_in, hidden_in, w.enorm.get(),
                                  w.hnorm.get(), ws.concat_eh.get(), steps, dim,
                                  cfg.norm_eps, stream);

    // 2. eh_proj -> mixed -> input_norm -> normed
    RunLinearBf16(w.eh_bf16.get(), ws.concat_eh.get(), ws.mixed.get(), steps,
                  dim, 2U * dim);
    LaunchTowerRmsNorm(ws.mixed.get(), w.input_norm.get(), ws.normed.get(),
                       steps, dim, cfg.norm_eps, stream);

    // 3. wqkv -> Split + QK norm + Interleaved RoPE + KV cache write
    RunLinearBf16(w.qkv_bf16.get(), ws.normed.get(), ws.qkv.get(), steps,
                  qkv_stride, dim);
    LaunchTowerSplitQkvNormRopeCache(
        ws.qkv.get(), w.q_norm.get(), w.k_norm.get(), ws.q.get(),
        w.key_cache.get(), w.value_cache.get(), steps, start_pos, cfg.heads,
        cfg.kv_heads, cfg.head_dim, cfg.RotaryDim(), cfg.rope_theta,
        cfg.norm_eps, stream);

    // 4. Causal GQA Attention + output gate
    LaunchTowerGqaAttentionAndGate(
        ws.q.get(), w.key_cache.get(), w.value_cache.get(), ws.qkv.get(),
        ws.attended.get(), steps, start_pos, cfg.heads, cfg.kv_heads,
        cfg.head_dim, qkv_stride, stream);

    // 5. wo + residual add into mixed + post_norm -> normed
    RunLinearBf16(w.wo_bf16.get(), ws.attended.get(), ws.normed.get(), steps,
                  dim, q_width);
    LaunchTowerResidualAddRmsNorm(ws.mixed.get(), ws.normed.get(),
                                  w.post_norm.get(), ws.normed.get(), steps,
                                  dim, cfg.norm_eps, stream);

    // 6. FFN SwiGLU + w2 + residual add + final_norm -> hidden_out
    if (steps == 1U) {
      LaunchTowerBf16SwiGluGemv(w.w1_bf16.get(), w.w3_bf16.get(),
                                ws.normed.get(), ws.ffn_act.get(), inter, dim,
                                stream);
    } else {
      RunLinearBf16(w.w1_bf16.get(), ws.normed.get(), ws.ffn_gate.get(), steps,
                    inter, dim);
      RunLinearBf16(w.w3_bf16.get(), ws.normed.get(), ws.ffn_up.get(), steps,
                    inter, dim);
      LaunchTowerSwiGlu(ws.ffn_gate.get(), ws.ffn_up.get(), ws.ffn_act.get(),
                        steps * inter, stream);
    }
    RunLinearBf16(w.w2_bf16.get(), ws.ffn_act.get(), ws.normed.get(), steps,
                  dim, inter);
    LaunchTowerResidualAddRmsNorm(ws.mixed.get(), ws.normed.get(),
                                  w.final_norm.get(), hidden_out, steps, dim,
                                  cfg.norm_eps, stream);
  }
};

TowerHipRuntime::TowerHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TowerHipRuntime::~TowerHipRuntime() = default;

std::unique_ptr<TowerHipRuntime> TowerHipRuntime::Create(
    const ModelSafetensors& store, const TowerConfig& config,
    std::size_t max_cache_steps, std::string* error) {
  try {
    return std::unique_ptr<TowerHipRuntime>(new TowerHipRuntime(
        std::make_unique<Impl>(store, config, max_cache_steps)));
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return nullptr;
  }
}

void TowerHipRuntime::Reset() {
  const std::size_t kv_bytes = impl_->capacity * impl_->cfg.kv_heads *
                               impl_->cfg.head_dim * sizeof(float);
  for (auto& layer : impl_->layers) {
    (void)hipMemsetAsync(layer.key_cache.get(), 0, kv_bytes, impl_->stream);
    (void)hipMemsetAsync(layer.value_cache.get(), 0, kv_bytes, impl_->stream);
  }
}

void TowerHipRuntime::MixBackboneEmbeddings(
    std::vector<float>* embeddings, std::span<const Drama3Frame> frames) const {
  const auto& cfg = impl_->cfg;
  if (embeddings == nullptr ||
      embeddings->size() != frames.size() * cfg.hidden_size) {
    throw std::runtime_error("Drama 3 frame embedding size mismatch");
  }
  const float scale =
      1.0F / std::sqrt(1.0F + static_cast<float>(cfg.backbone_codebooks));
  for (std::size_t row = 0; row < frames.size(); ++row) {
    if (!IsSemanticFrame(frames[row], cfg)) {
      continue;
    }
    float* out = embeddings->data() + row * cfg.hidden_size;
    impl_->AddCodeEmbeddings(out, frames[row], 1U, 1U + cfg.backbone_codebooks);
    for (std::size_t c = 0; c < cfg.hidden_size; ++c) {
      out[c] *= scale;
    }
  }
}

std::vector<float> TowerHipRuntime::SharedInput(
    std::span<const Drama3Frame> frames) const {
  const auto& cfg = impl_->cfg;
  std::vector<float> result(frames.size() * cfg.hidden_size, 0.0F);
  for (std::size_t row = 0; row < frames.size(); ++row) {
    if (IsSemanticFrame(frames[row], cfg)) {
      impl_->AddCodeEmbeddings(result.data() + row * cfg.hidden_size,
                               frames[row], 1U + cfg.backbone_codebooks, 10U);
    }
  }
  return result;
}

bool TowerHipRuntime::Prefill(std::span<const float> backbone_hidden,
                              std::span<const Drama3Frame> frames,
                              std::string* error) {
  const auto& cfg = impl_->cfg;
  if (frames.empty() ||
      backbone_hidden.size() != frames.size() * cfg.hidden_size) {
    SetError(error, "Drama 3 tower prompt hidden size mismatch");
    return false;
  }
  const std::size_t steps = frames.size() - 1U;
  if (steps == 0) {
    return true;
  }
  if (steps > impl_->capacity) {
    SetError(error, "Drama 3 tower prefill exceeds cache capacity");
    return false;
  }

  try {
    impl_->ws.Ensure(steps, cfg);
    const auto extra_host = SharedInput(frames.first(steps));
    std::vector<float> masks_host(steps);
    for (std::size_t r = 0; r < steps; ++r) {
      masks_host[r] = IsSemanticFrame(frames[r + 1U], cfg) ? 1.0F : 0.0F;
    }

    RequireHip(hipMemcpyAsync(impl_->ws.hidden.get(), backbone_hidden.data(),
                              steps * cfg.hidden_size * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "upload tower prefill hidden");
    RequireHip(hipMemcpyAsync(impl_->ws.extra.get(), extra_host.data(),
                              extra_host.size() * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "upload tower prefill extra");
    RequireHip(hipMemcpyAsync(impl_->ws.label_mask.get(), masks_host.data(),
                              steps * sizeof(float), hipMemcpyHostToDevice,
                              impl_->stream),
               "upload tower prefill masks");

    std::vector<std::int32_t> codes_host(steps);
    float* cur_h = impl_->ws.hidden.get();
    float* next_h = impl_->ws.next_hidden.get();

    for (std::size_t layer = 0; layer < impl_->layers.size(); ++layer) {
      const std::size_t book =
          layer < cfg.shared_layers ? 0U : (layer - cfg.shared_layers);
      for (std::size_t r = 0; r < steps; ++r) {
        codes_host[r] = frames[r + 1U][1U + book];
      }
      RequireHip(hipMemcpyAsync(impl_->ws.codes.get(), codes_host.data(),
                                steps * sizeof(std::int32_t),
                                hipMemcpyHostToDevice, impl_->stream),
                 "upload tower prefill book codes");
      impl_->RunLayerDevice(layer, steps, 0, cur_h, impl_->ws.codes.get(),
                            impl_->ws.label_mask.get(),
                            layer == 0 ? impl_->ws.extra.get() : nullptr,
                            next_h);
      std::swap(cur_h, next_h);
    }
    RequireHip(hipStreamSynchronize(impl_->stream),
               "synchronize Drama 3 tower prefill");
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

bool TowerHipRuntime::RunFrame(std::span<const float> backbone_hidden,
                               const Drama3Frame& current_frame,
                               std::int32_t semantic_code, std::size_t position,
                               float temperature, int top_k, float top_p,
                               const std::array<float, 9>& uniforms,
                               Drama3Frame* out_frame, std::string* error) {
  const auto& cfg = impl_->cfg;
  if (out_frame == nullptr || backbone_hidden.size() != cfg.hidden_size ||
      position >= impl_->capacity || semantic_code < 0 ||
      static_cast<std::uint32_t>(semantic_code) >= cfg.codebook_size) {
    SetError(error, "Invalid Drama 3 tower RunFrame arguments");
    return false;
  }

  try {
    impl_->ws.Ensure(1, cfg);
    const std::array<Drama3Frame, 1> cur_span{current_frame};
    const auto extra_host = SharedInput(cur_span);
    const float one_mask = 1.0F;

    RequireHip(hipMemcpyAsync(impl_->ws.hidden.get(), backbone_hidden.data(),
                              cfg.hidden_size * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "upload tower frame hidden");
    RequireHip(hipMemcpyAsync(impl_->ws.extra.get(), extra_host.data(),
                              cfg.hidden_size * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "upload tower frame extra");
    RequireHip(
        hipMemcpyAsync(impl_->ws.label_mask.get(), &one_mask, sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream),
        "upload tower frame mask");
    RequireHip(hipMemcpyAsync(impl_->ws.codes.get(), &semantic_code,
                              sizeof(std::int32_t), hipMemcpyHostToDevice,
                              impl_->stream),
               "upload tower frame semantic code");

    float* cur_h = impl_->ws.hidden.get();
    float* next_h = impl_->ws.next_hidden.get();

    for (std::size_t layer = 0; layer < impl_->layers.size(); ++layer) {
      impl_->RunLayerDevice(layer, 1U, position, cur_h, impl_->ws.codes.get(),
                            impl_->ws.label_mask.get(),
                            layer == 0 ? impl_->ws.extra.get() : nullptr,
                            next_h);
      if (layer >= cfg.shared_layers) {
        const std::size_t book = layer - cfg.shared_layers;
        impl_->RunLinearBf16(impl_->layers[layer].head_bf16.get(), next_h,
                             impl_->ws.logits.get(), 1U, cfg.codebook_size,
                             cfg.hidden_size);
        LaunchTowerSampleTopKTopP(
            impl_->ws.logits.get(), cfg.codebook_size, temperature, top_k,
            top_p, uniforms[book], impl_->ws.codes.get(),
            impl_->ws.sampled_books.get(), book, impl_->stream);
      }
      std::swap(cur_h, next_h);
    }

    std::array<std::int32_t, 9> books_host{};
    RequireHip(hipMemcpyAsync(books_host.data(), impl_->ws.sampled_books.get(),
                              books_host.size() * sizeof(std::int32_t),
                              hipMemcpyDeviceToHost, impl_->stream),
               "download tower sampled books");
    RequireHip(hipStreamSynchronize(impl_->stream),
               "synchronize Drama 3 tower frame");

    Drama3Frame res{};
    res[0] = cfg.semantic_begin + semantic_code;
    res[1] = semantic_code;
    for (std::size_t b = 0; b < 9U; ++b) {
      res[b + 2U] = books_host[b];
    }
    *out_frame = res;
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

const TowerConfig& TowerHipRuntime::config() const noexcept {
  return impl_->cfg;
}

}  // namespace gufo::models::fish_drama3::hip

#else

namespace gufo::models::fish_drama3::hip {

struct TowerHipRuntime::Impl {
  TowerConfig cfg;
};

TowerHipRuntime::TowerHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TowerHipRuntime::~TowerHipRuntime() = default;

std::unique_ptr<TowerHipRuntime> TowerHipRuntime::Create(
    const ModelSafetensors&, const TowerConfig&, std::size_t,
    std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 tower runtime requires a HIP-enabled build";
  }
  return nullptr;
}

void TowerHipRuntime::Reset() {}
void TowerHipRuntime::MixBackboneEmbeddings(
    std::vector<float>*, std::span<const Drama3Frame>) const {}
std::vector<float> TowerHipRuntime::SharedInput(
    std::span<const Drama3Frame>) const {
  return {};
}
bool TowerHipRuntime::Prefill(std::span<const float>,
                              std::span<const Drama3Frame>,
                              std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 tower runtime requires a HIP-enabled build";
  }
  return false;
}
bool TowerHipRuntime::RunFrame(std::span<const float>, const Drama3Frame&,
                               std::int32_t, std::size_t, float, int, float,
                               const std::array<float, 9>&, Drama3Frame*,
                               std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 tower runtime requires a HIP-enabled build";
  }
  return false;
}
const TowerConfig& TowerHipRuntime::config() const noexcept {
  return impl_->cfg;
}

}  // namespace gufo::models::fish_drama3::hip

#endif
