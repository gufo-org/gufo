#include "src/models/fish_drama3/hip/backbone_runtime.hpp"

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

#include "src/models/fish_drama3/hip/backbone_ops.hpp"

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
      throw std::overflow_error("Drama 3 backbone device buffer overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Drama 3 backbone");
    count_ = count;
  }

  void Upload(const Element* host_data, std::size_t count,
              hipStream_t stream = nullptr) {
    Reset(count);
    if (count > 0) {
      if (stream != nullptr) {
        RequireHip(hipMemcpyAsync(data_, host_data, count * sizeof(Element),
                                  hipMemcpyHostToDevice, stream),
                   "hipMemcpyAsync Drama 3 backbone");
      } else {
        RequireHip(hipMemcpy(data_, host_data, count * sizeof(Element),
                             hipMemcpyHostToDevice),
                   "hipMemcpy Drama 3 backbone");
      }
    }
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

struct FullAttentionWeights {
  DeviceBuffer<std::uint16_t> q_proj_bf16;  // [8192, 2048]
  DeviceBuffer<std::uint16_t> k_proj_bf16;  // [512, 2048]
  DeviceBuffer<std::uint16_t> v_proj_bf16;  // [512, 2048]
  DeviceBuffer<std::uint16_t> o_proj_bf16;  // [2048, 4096]
  DeviceBuffer<float> q_norm;               // [256] (1 + w)
  DeviceBuffer<float> k_norm;               // [256] (1 + w)

  DeviceBuffer<float> key_cache;    // [capacity, 512]
  DeviceBuffer<float> value_cache;  // [capacity, 512]
  DeviceBuffer<float> prefix_key_cache;
  DeviceBuffer<float> prefix_value_cache;
};

struct LinearAttentionWeights {
  DeviceBuffer<std::uint16_t> in_proj_qkv_bf16;  // [8192, 2048]
  DeviceBuffer<std::uint16_t> in_proj_z_bf16;    // [4096, 2048]
  DeviceBuffer<std::uint16_t> in_proj_b_bf16;    // [32, 2048]
  DeviceBuffer<std::uint16_t> in_proj_a_bf16;    // [32, 2048]
  DeviceBuffer<float> conv_weight;               // [8192, 4]
  DeviceBuffer<float> neg_exp_a;                 // [32] (-exp(A_log))
  DeviceBuffer<float> dt_bias;                   // [32]
  DeviceBuffer<float> norm_weight;               // [128] (raw RMSNorm weight)
  DeviceBuffer<std::uint16_t> out_proj_bf16;     // [2048, 4096]

  DeviceBuffer<float> conv_state;  // [3, 8192]
  DeviceBuffer<float> ssm_state;   // [32, 128, 128]
  DeviceBuffer<float> prefix_conv_state;
  DeviceBuffer<float> prefix_ssm_state;
};

struct LayerWeights {
  bool is_full_attention{false};
  DeviceBuffer<float> input_norm;  // [2048] (1 + w)
  DeviceBuffer<float> post_norm;   // [2048] (1 + w)

  FullAttentionWeights full_attn;
  LinearAttentionWeights linear_attn;

  // MoE Router + Shared Expert (BF16)
  DeviceBuffer<std::uint16_t> router_weight_bf16;       // [num_experts, 2048]
  DeviceBuffer<std::uint16_t> shared_gate_weight_bf16;  // [1, 2048]
  DeviceBuffer<std::uint16_t> shared_gate_proj_bf16;    // [512, 2048]
  DeviceBuffer<std::uint16_t> shared_up_proj_bf16;      // [512, 2048]
  DeviceBuffer<std::uint16_t> shared_down_proj_bf16;    // [2048, 512]

  // Routed Experts (Native F8_E4M3 + 128x128 block scale_inv F32)
  std::uint32_t num_experts{256};
  DeviceBuffer<std::uint8_t> experts_gate_fp8;  // [num_experts, 512, 2048]
  DeviceBuffer<float> experts_gate_scale;       // [num_experts, 4, 16]
  DeviceBuffer<std::uint8_t> experts_up_fp8;    // [num_experts, 512, 2048]
  DeviceBuffer<float> experts_up_scale;         // [num_experts, 4, 16]
  DeviceBuffer<std::uint8_t> experts_down_fp8;  // [num_experts, 2048, 512]
  DeviceBuffer<float> experts_down_scale;       // [num_experts, 16, 4]
};

struct Workspace {
  std::size_t max_steps{0};
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> normed;
  DeviceBuffer<float> qkv;
  DeviceBuffer<float> qkv_conv;
  DeviceBuffer<float> z_or_k;
  DeviceBuffer<float> v_buf;
  DeviceBuffer<float> b_buf;
  DeviceBuffer<float> a_buf;
  DeviceBuffer<float> attn_out;
  DeviceBuffer<float> proj_out;
  DeviceBuffer<std::int32_t> top_indices;
  DeviceBuffer<float> top_weights;
  DeviceBuffer<float> shared_gate_scale;
  DeviceBuffer<float> expert_inter;
  DeviceBuffer<float> logits;
  DeviceBuffer<std::uint16_t> bf16_in;

  void Ensure(std::size_t steps, const BackboneConfig& cfg,
              std::size_t lm_head_rows) {
    if (steps <= max_steps) {
      return;
    }
    const std::size_t dim = cfg.hidden_size;
    const std::size_t q_full_width =
        cfg.num_attention_heads * cfg.head_dim * 2U;  // 8192
    const std::size_t kv_width = cfg.num_key_value_heads * cfg.head_dim;  // 512
    const std::size_t lin_q_dim =
        cfg.linear_num_key_heads * cfg.linear_key_head_dim;  // 2048
    const std::size_t lin_v_dim =
        cfg.linear_num_value_heads * cfg.linear_value_head_dim;    // 4096
    const std::size_t lin_qkv_width = 2U * lin_q_dim + lin_v_dim;  // 8192

    hidden.Reset(steps * dim);
    normed.Reset(steps * dim);
    qkv.Reset(steps * std::max(q_full_width, lin_qkv_width));
    qkv_conv.Reset(steps * lin_qkv_width);
    z_or_k.Reset(steps * std::max(lin_v_dim, kv_width));
    v_buf.Reset(steps * kv_width);
    b_buf.Reset(steps * cfg.linear_num_value_heads);
    a_buf.Reset(steps * cfg.linear_num_value_heads);
    attn_out.Reset(steps *
                   std::max<std::size_t>(cfg.num_attention_heads * cfg.head_dim,
                                         lin_v_dim));
    proj_out.Reset(steps * dim);
    top_indices.Reset(steps * cfg.num_experts_per_tok);
    top_weights.Reset(steps * cfg.num_experts_per_tok);
    shared_gate_scale.Reset(steps);
    expert_inter.Reset(steps * (cfg.num_experts_per_tok + 1U) *
                       cfg.intermediate_size);
    logits.Reset(lm_head_rows);
    bf16_in.Reset(steps * std::max(dim, lin_v_dim));
    max_steps = steps;
  }
};

}  // namespace

struct BackboneHipRuntime::Impl {
  ModelConfig model_cfg;
  BackboneConfig cfg;
  std::size_t capacity{4096};
  std::size_t cur_steps{0};
  std::size_t prefix_steps{0};
  std::size_t compact_lm_head_rows{0};

  hipStream_t stream{nullptr};
  rocblas_handle blas{nullptr};

  std::vector<std::uint16_t>
      embed_tokens_bf16;  // [vocab_size, hidden_size] on host
  std::vector<LayerWeights> layers;
  DeviceBuffer<float> final_norm;                    // [hidden_size] (1 + w)
  DeviceBuffer<std::uint16_t> compact_lm_head_bf16;  // [4097, hidden_size]
  Workspace ws;

  Impl(const ModelSafetensors& store, const ModelConfig& mcfg,
       std::size_t max_cache_steps)
      : model_cfg(mcfg), cfg(mcfg.backbone), capacity(max_cache_steps) {
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags Drama 3 backbone");
    RequireRocblas(rocblas_create_handle(&blas),
                   "rocblas_create_handle Drama 3 backbone");
    RequireRocblas(rocblas_set_atomics_mode(blas, rocblas_atomics_not_allowed),
                   "rocblas_set_atomics_mode Drama 3 backbone");
    RequireRocblas(rocblas_set_stream(blas, stream),
                   "rocblas_set_stream Drama 3 backbone");
    LoadWeights(store);
    ws.Ensure(64, cfg, compact_lm_head_rows);
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Drama 3 backbone init");
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
      throw std::runtime_error(
          "Unexpected shape for Drama 3 backbone tensor: " + name);
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
      throw std::runtime_error(
          "Unsupported dtype for Drama 3 backbone tensor: " + name);
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
      throw std::runtime_error(
          "Unexpected shape for Drama 3 backbone matrix: " + name);
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
      throw std::runtime_error(
          "Unsupported dtype for Drama 3 backbone matrix: " + name);
    }
    return buf;
  }

  void LoadExpertBankFp8(const ModelSafetensors& store,
                         const std::string& layer_prefix,
                         const std::string& proj_name,
                         std::uint32_t num_experts, std::uint64_t rows,
                         std::uint64_t cols,
                         DeviceBuffer<std::uint8_t>* out_fp8,
                         DeviceBuffer<float>* out_scale) {
    const std::uint64_t rb = rows / 128ULL;
    const std::uint64_t cb = cols / 128ULL;
    const std::size_t mat_elems = static_cast<std::size_t>(rows * cols);
    const std::size_t scale_elems = static_cast<std::size_t>(rb * cb);

    out_fp8->Reset(static_cast<std::size_t>(num_experts) * mat_elems);
    out_scale->Reset(static_cast<std::size_t>(num_experts) * scale_elems);

    const std::array<std::uint64_t, 2> mat_shape{rows, cols};
    const std::array<std::uint64_t, 2> scale_shape{rb, cb};

    for (std::uint32_t e = 0; e < num_experts; ++e) {
      const std::string base =
          layer_prefix + ".mlp.experts." + std::to_string(e) + "." + proj_name;
      const TensorView& w_t = store.Require(base + ".weight");
      const TensorView& s_t = store.Require(base + ".weight_scale_inv");
      if (w_t.dtype != DType::kF8_E4M3 ||
          !std::ranges::equal(w_t.shape, mat_shape)) {
        throw std::runtime_error("Expected F8_E4M3 expert weight: " + base +
                                 ".weight");
      }
      if (s_t.dtype != DType::kF32 ||
          !std::ranges::equal(s_t.shape, scale_shape)) {
        throw std::runtime_error("Expected F32 expert weight_scale_inv: " +
                                 base + ".weight_scale_inv");
      }
      RequireHip(hipMemcpyAsync(out_fp8->get() + e * mat_elems, w_t.data,
                                mat_elems * sizeof(std::uint8_t),
                                hipMemcpyHostToDevice, stream),
                 "upload FP8 expert weight");
      RequireHip(hipMemcpyAsync(out_scale->get() + e * scale_elems,
                                reinterpret_cast<const float*>(s_t.data),
                                scale_elems * sizeof(float),
                                hipMemcpyHostToDevice, stream),
                 "upload FP8 expert scale_inv");
    }
  }

  void LoadWeights(const ModelSafetensors& store) {
    // 1. Token embeddings (keep BF16 on host for instant lookup)
    {
      const TensorView& emb_t =
          store.Require("model.language_model.embed_tokens.weight");
      const std::array<std::uint64_t, 2> expected{cfg.vocab_size,
                                                  cfg.hidden_size};
      if (emb_t.dtype != DType::kBF16 ||
          !std::ranges::equal(emb_t.shape, expected)) {
        throw std::runtime_error(
            "Unexpected shape/dtype for embed_tokens.weight");
      }
      embed_tokens_bf16.resize(emb_t.NumElements());
      std::memcpy(embed_tokens_bf16.data(), emb_t.data,
                  emb_t.NumElements() * sizeof(std::uint16_t));
    }

    // 2. Compact LM Head: [1 + codebook_size, hidden_size] = [4097, 2048]
    compact_lm_head_rows = 1U + model_cfg.tower.codebook_size;
    {
      const std::string head_name =
          cfg.tie_word_embeddings ? "model.language_model.embed_tokens.weight"
                                  : "lm_head.weight";
      const TensorView& head_t = store.Require(head_name);
      if (head_t.dtype != DType::kBF16 || head_t.shape.size() != 2 ||
          head_t.shape[1] != cfg.hidden_size) {
        throw std::runtime_error("Unexpected shape/dtype for " + head_name);
      }
      const auto* src_bf16 =
          reinterpret_cast<const std::uint16_t*>(head_t.data);
      std::vector<std::uint16_t> compact_host(compact_lm_head_rows *
                                              cfg.hidden_size);
      if (head_t.shape[0] == compact_lm_head_rows) {
        std::memcpy(compact_host.data(), src_bf16,
                    compact_host.size() * sizeof(std::uint16_t));
      } else {
        // Row 0: eos_token_id
        std::memcpy(
            compact_host.data(),
            src_bf16 + static_cast<std::size_t>(model_cfg.eos_token_id) *
                           cfg.hidden_size,
            cfg.hidden_size * sizeof(std::uint16_t));
        // Rows 1..4096: semantic_start_token_id .. semantic_end_token_id
        std::memcpy(compact_host.data() + cfg.hidden_size,
                    src_bf16 + static_cast<std::size_t>(
                                   model_cfg.semantic_start_token_id) *
                                   cfg.hidden_size,
                    model_cfg.tower.codebook_size * cfg.hidden_size *
                        sizeof(std::uint16_t));
      }
      compact_lm_head_bf16.Upload(compact_host.data(), compact_host.size(),
                                  stream);
    }

    // 3. Layers 0..39
    const std::uint64_t dim = cfg.hidden_size;
    const std::uint64_t q_width =
        static_cast<std::uint64_t>(cfg.num_attention_heads) * cfg.head_dim;
    const std::uint64_t kv_width =
        static_cast<std::uint64_t>(cfg.num_key_value_heads) * cfg.head_dim;
    const std::uint64_t lin_q_dim =
        static_cast<std::uint64_t>(cfg.linear_num_key_heads) *
        cfg.linear_key_head_dim;
    const std::uint64_t lin_v_dim =
        static_cast<std::uint64_t>(cfg.linear_num_value_heads) *
        cfg.linear_value_head_dim;
    const std::uint64_t lin_qkv_dim = 2ULL * lin_q_dim + lin_v_dim;

    layers.resize(cfg.num_hidden_layers);
    for (std::uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
      const std::string lp = "model.language_model.layers." + std::to_string(l);
      auto& lw = layers[l];
      lw.is_full_attention = cfg.IsFullAttentionLayer(l);
      lw.input_norm = LoadGemmaNorm(store, lp + ".input_layernorm.weight", dim);
      lw.post_norm =
          LoadGemmaNorm(store, lp + ".post_attention_layernorm.weight", dim);

      if (lw.is_full_attention) {
        const std::string ap = lp + ".self_attn";
        lw.full_attn.q_proj_bf16 =
            LoadBf16Matrix(store, ap + ".q_proj.weight", 2ULL * q_width, dim);
        lw.full_attn.k_proj_bf16 =
            LoadBf16Matrix(store, ap + ".k_proj.weight", kv_width, dim);
        lw.full_attn.v_proj_bf16 =
            LoadBf16Matrix(store, ap + ".v_proj.weight", kv_width, dim);
        lw.full_attn.o_proj_bf16 =
            LoadBf16Matrix(store, ap + ".o_proj.weight", dim, q_width);
        lw.full_attn.q_norm =
            LoadGemmaNorm(store, ap + ".q_norm.weight", cfg.head_dim);
        lw.full_attn.k_norm =
            LoadGemmaNorm(store, ap + ".k_norm.weight", cfg.head_dim);
        lw.full_attn.key_cache.Reset(capacity * kv_width);
        lw.full_attn.value_cache.Reset(capacity * kv_width);
      } else {
        const std::string ap = lp + ".linear_attn";
        lw.linear_attn.in_proj_qkv_bf16 =
            LoadBf16Matrix(store, ap + ".in_proj_qkv.weight", lin_qkv_dim, dim);
        lw.linear_attn.in_proj_z_bf16 =
            LoadBf16Matrix(store, ap + ".in_proj_z.weight", lin_v_dim, dim);
        lw.linear_attn.in_proj_b_bf16 = LoadBf16Matrix(
            store, ap + ".in_proj_b.weight", cfg.linear_num_value_heads, dim);
        lw.linear_attn.in_proj_a_bf16 = LoadBf16Matrix(
            store, ap + ".in_proj_a.weight", cfg.linear_num_value_heads, dim);
        lw.linear_attn.out_proj_bf16 =
            LoadBf16Matrix(store, ap + ".out_proj.weight", dim, lin_v_dim);

        const auto conv_w =
            ReadTensorAsF32(store, ap + ".conv1d.weight",
                            {lin_qkv_dim, 1ULL, cfg.linear_conv_kernel_dim});
        lw.linear_attn.conv_weight.Upload(conv_w.data(), conv_w.size(), stream);

        auto a_log =
            ReadTensorAsF32(store, ap + ".A_log", {cfg.linear_num_value_heads});
        for (float& v : a_log) {
          v = -std::exp(v);
        }
        lw.linear_attn.neg_exp_a.Upload(a_log.data(), a_log.size(), stream);

        const auto dt_b = ReadTensorAsF32(store, ap + ".dt_bias",
                                          {cfg.linear_num_value_heads});
        lw.linear_attn.dt_bias.Upload(dt_b.data(), dt_b.size(), stream);

        const auto norm_w = ReadTensorAsF32(store, ap + ".norm.weight",
                                            {cfg.linear_value_head_dim});
        lw.linear_attn.norm_weight.Upload(norm_w.data(), norm_w.size(), stream);

        lw.linear_attn.conv_state.Reset((cfg.linear_conv_kernel_dim - 1U) *
                                        lin_qkv_dim);
        lw.linear_attn.ssm_state.Reset(cfg.linear_num_value_heads *
                                       cfg.linear_key_head_dim *
                                       cfg.linear_value_head_dim);
      }

      // MoE
      lw.num_experts = cfg.LayerExperts(l);
      lw.router_weight_bf16 =
          LoadBf16Matrix(store, lp + ".mlp.gate.weight", lw.num_experts, dim);
      lw.shared_gate_weight_bf16 = LoadBf16Matrix(
          store, lp + ".mlp.shared_expert_gate.weight", 1ULL, dim);
      lw.shared_gate_proj_bf16 =
          LoadBf16Matrix(store, lp + ".mlp.shared_expert.gate_proj.weight",
                         cfg.shared_expert_intermediate_size, dim);
      lw.shared_up_proj_bf16 =
          LoadBf16Matrix(store, lp + ".mlp.shared_expert.up_proj.weight",
                         cfg.shared_expert_intermediate_size, dim);
      lw.shared_down_proj_bf16 =
          LoadBf16Matrix(store, lp + ".mlp.shared_expert.down_proj.weight", dim,
                         cfg.shared_expert_intermediate_size);

      LoadExpertBankFp8(store, lp, "gate_proj", lw.num_experts,
                        cfg.intermediate_size, dim, &lw.experts_gate_fp8,
                        &lw.experts_gate_scale);
      LoadExpertBankFp8(store, lp, "up_proj", lw.num_experts,
                        cfg.intermediate_size, dim, &lw.experts_up_fp8,
                        &lw.experts_up_scale);
      LoadExpertBankFp8(store, lp, "down_proj", lw.num_experts, dim,
                        cfg.intermediate_size, &lw.experts_down_fp8,
                        &lw.experts_down_scale);
    }

    final_norm = LoadGemmaNorm(store, "model.language_model.norm.weight", dim);
  }

  void RunLinearBf16(const std::uint16_t* weight_bf16, const float* input_f32,
                     float* output_f32, std::size_t batch, std::size_t rows,
                     std::size_t cols) {
    if (batch <= 4U) {
      LaunchBackboneBf16Gemv(weight_bf16, input_f32, output_f32, batch, rows,
                             cols, stream);
      return;
    }
    LaunchBackboneFloatToBf16(input_f32, ws.bf16_in.get(), batch * cols,
                              stream);
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
        "rocblas_gemm_ex Drama 3 backbone");
  }
};

BackboneHipRuntime::BackboneHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

BackboneHipRuntime::~BackboneHipRuntime() = default;

std::unique_ptr<BackboneHipRuntime> BackboneHipRuntime::Create(
    const ModelSafetensors& store, const ModelConfig& model_config,
    std::size_t max_cache_steps, std::string* error) {
  try {
    auto rt = std::unique_ptr<BackboneHipRuntime>(new BackboneHipRuntime(
        std::make_unique<Impl>(store, model_config, max_cache_steps)));
    rt->Reset();
    return rt;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return nullptr;
  }
}

void BackboneHipRuntime::Reset() {
  impl_->cur_steps = 0;
  for (auto& lw : impl_->layers) {
    if (lw.is_full_attention) {
      (void)hipMemsetAsync(lw.full_attn.key_cache.get(), 0,
                           lw.full_attn.key_cache.size() * sizeof(float),
                           impl_->stream);
      (void)hipMemsetAsync(lw.full_attn.value_cache.get(), 0,
                           lw.full_attn.value_cache.size() * sizeof(float),
                           impl_->stream);
    } else {
      (void)hipMemsetAsync(lw.linear_attn.conv_state.get(), 0,
                           lw.linear_attn.conv_state.size() * sizeof(float),
                           impl_->stream);
      (void)hipMemsetAsync(lw.linear_attn.ssm_state.get(), 0,
                           lw.linear_attn.ssm_state.size() * sizeof(float),
                           impl_->stream);
    }
  }
}

bool BackboneHipRuntime::SavePrefixState(std::string* error) {
  if (impl_->cur_steps == 0 || impl_->cur_steps > impl_->capacity) {
    SetError(error, "Invalid Drama 3 backbone prefix steps to save");
    return false;
  }
  try {
    impl_->prefix_steps = impl_->cur_steps;
    const std::size_t kv_width =
        impl_->cfg.num_key_value_heads * impl_->cfg.head_dim;
    const std::size_t kv_elems = impl_->prefix_steps * kv_width;
    for (auto& lw : impl_->layers) {
      if (lw.is_full_attention) {
        if (lw.full_attn.prefix_key_cache.size() < kv_elems) {
          lw.full_attn.prefix_key_cache.Reset(kv_elems);
          lw.full_attn.prefix_value_cache.Reset(kv_elems);
        }
        RequireHip(hipMemcpyAsync(lw.full_attn.prefix_key_cache.get(),
                                  lw.full_attn.key_cache.get(),
                                  kv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "save backbone prefix key_cache");
        RequireHip(hipMemcpyAsync(lw.full_attn.prefix_value_cache.get(),
                                  lw.full_attn.value_cache.get(),
                                  kv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "save backbone prefix value_cache");
      } else {
        const std::size_t conv_elems = lw.linear_attn.conv_state.size();
        const std::size_t ssm_elems = lw.linear_attn.ssm_state.size();
        if (lw.linear_attn.prefix_conv_state.size() != conv_elems) {
          lw.linear_attn.prefix_conv_state.Reset(conv_elems);
        }
        if (lw.linear_attn.prefix_ssm_state.size() != ssm_elems) {
          lw.linear_attn.prefix_ssm_state.Reset(ssm_elems);
        }
        RequireHip(hipMemcpyAsync(lw.linear_attn.prefix_conv_state.get(),
                                  lw.linear_attn.conv_state.get(),
                                  conv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "save backbone prefix conv_state");
        RequireHip(hipMemcpyAsync(lw.linear_attn.prefix_ssm_state.get(),
                                  lw.linear_attn.ssm_state.get(),
                                  ssm_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "save backbone prefix ssm_state");
      }
    }
    RequireHip(hipStreamSynchronize(impl_->stream),
               "synchronize save backbone prefix state");
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

bool BackboneHipRuntime::RestorePrefixState(std::string* error) {
  if (impl_->prefix_steps == 0 || impl_->prefix_steps > impl_->capacity) {
    SetError(error, "No saved Drama 3 backbone prefix state to restore");
    return false;
  }
  try {
    impl_->cur_steps = impl_->prefix_steps;
    const std::size_t kv_width =
        impl_->cfg.num_key_value_heads * impl_->cfg.head_dim;
    const std::size_t kv_elems = impl_->prefix_steps * kv_width;
    for (auto& lw : impl_->layers) {
      if (lw.is_full_attention) {
        RequireHip(hipMemcpyAsync(lw.full_attn.key_cache.get(),
                                  lw.full_attn.prefix_key_cache.get(),
                                  kv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "restore backbone prefix key_cache");
        RequireHip(hipMemcpyAsync(lw.full_attn.value_cache.get(),
                                  lw.full_attn.prefix_value_cache.get(),
                                  kv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "restore backbone prefix value_cache");
      } else {
        const std::size_t conv_elems = lw.linear_attn.conv_state.size();
        const std::size_t ssm_elems = lw.linear_attn.ssm_state.size();
        RequireHip(hipMemcpyAsync(lw.linear_attn.conv_state.get(),
                                  lw.linear_attn.prefix_conv_state.get(),
                                  conv_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "restore backbone prefix conv_state");
        RequireHip(hipMemcpyAsync(lw.linear_attn.ssm_state.get(),
                                  lw.linear_attn.prefix_ssm_state.get(),
                                  ssm_elems * sizeof(float),
                                  hipMemcpyDeviceToDevice, impl_->stream),
                   "restore backbone prefix ssm_state");
      }
    }
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

std::size_t BackboneHipRuntime::saved_prefix_steps() const noexcept {
  return impl_->prefix_steps;
}

std::vector<float> BackboneHipRuntime::TokenEmbedding(
    std::span<const std::int32_t> token_ids) const {
  const std::size_t dim = impl_->cfg.hidden_size;
  std::vector<float> out(token_ids.size() * dim);
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    const std::int32_t id = token_ids[i];
    if (id < 0 || static_cast<std::uint32_t>(id) >= impl_->cfg.vocab_size) {
      throw std::runtime_error("Drama 3 token ID out of range");
    }
    const std::uint16_t* row_bf16 =
        impl_->embed_tokens_bf16.data() + static_cast<std::size_t>(id) * dim;
    float* row_f32 = out.data() + i * dim;
    for (std::size_t c = 0; c < dim; ++c) {
      row_f32[c] = HostBf16ToFloat(row_bf16[c]);
    }
  }
  return out;
}

bool BackboneHipRuntime::ForwardEmbeddings(
    std::span<const float> embeddings, std::size_t steps,
    std::vector<float>* out_hidden, std::string* error,
    std::vector<float>* out_last_logits) {
  const auto& cfg = impl_->cfg;
  const std::size_t dim = cfg.hidden_size;
  if (out_hidden == nullptr || steps == 0 || embeddings.size() != steps * dim ||
      impl_->cur_steps + steps > impl_->capacity) {
    SetError(error, "Invalid Drama 3 Backbone ForwardEmbeddings arguments");
    return false;
  }

  try {
    impl_->ws.Ensure(steps, cfg, impl_->compact_lm_head_rows);
    RequireHip(hipMemcpyAsync(impl_->ws.hidden.get(), embeddings.data(),
                              steps * dim * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "upload backbone embeddings");

    const std::size_t start_pos = impl_->cur_steps;
    const std::size_t q_width = cfg.num_attention_heads * cfg.head_dim;
    const std::size_t kv_width = cfg.num_key_value_heads * cfg.head_dim;
    const std::size_t lin_q_dim =
        cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const std::size_t lin_v_dim =
        cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const std::size_t lin_qkv_dim = 2U * lin_q_dim + lin_v_dim;

    for (std::size_t l = 0; l < impl_->layers.size(); ++l) {
      auto& lw = impl_->layers[l];
      LaunchBackboneRmsNorm(impl_->ws.hidden.get(), lw.input_norm.get(),
                            impl_->ws.normed.get(), steps, dim,
                            cfg.rms_norm_eps, impl_->stream);

      if (lw.is_full_attention) {
        auto& fa = lw.full_attn;
        if (steps == 1U) {
          LaunchBackboneFullQkvProjFusedGemv(
              fa.q_proj_bf16.get(), fa.k_proj_bf16.get(), fa.v_proj_bf16.get(),
              impl_->ws.normed.get(), impl_->ws.qkv.get(),
              impl_->ws.z_or_k.get(), impl_->ws.v_buf.get(), 2U * q_width,
              kv_width, dim, impl_->stream);
        } else {
          impl_->RunLinearBf16(fa.q_proj_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.qkv.get(), steps, 2U * q_width, dim);
          impl_->RunLinearBf16(fa.k_proj_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.z_or_k.get(), steps, kv_width, dim);
          impl_->RunLinearBf16(fa.v_proj_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.v_buf.get(), steps, kv_width, dim);
        }

        LaunchBackboneFullAttnPrep(
            impl_->ws.qkv.get(), impl_->ws.z_or_k.get(), impl_->ws.v_buf.get(),
            fa.q_norm.get(), fa.k_norm.get(), impl_->ws.qkv_conv.get(),
            fa.key_cache.get(), fa.value_cache.get(), steps, start_pos,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            cfg.RotaryDim(), cfg.rope_theta, cfg.rms_norm_eps, impl_->stream);

        LaunchBackboneFullAttnGqaAndGate(
            impl_->ws.qkv_conv.get(), fa.key_cache.get(), fa.value_cache.get(),
            impl_->ws.qkv.get(), impl_->ws.attn_out.get(), steps, start_pos,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            impl_->stream);

        impl_->RunLinearBf16(fa.o_proj_bf16.get(), impl_->ws.attn_out.get(),
                             impl_->ws.proj_out.get(), steps, dim, q_width);
      } else {
        auto& la = lw.linear_attn;
        if (steps == 1U) {
          LaunchBackboneLinearProjFusedGemv(
              la.in_proj_qkv_bf16.get(), la.in_proj_z_bf16.get(),
              la.in_proj_b_bf16.get(), la.in_proj_a_bf16.get(),
              impl_->ws.normed.get(), impl_->ws.qkv.get(),
              impl_->ws.z_or_k.get(), impl_->ws.b_buf.get(),
              impl_->ws.a_buf.get(), lin_qkv_dim, lin_v_dim,
              cfg.linear_num_value_heads, dim, impl_->stream);
        } else {
          impl_->RunLinearBf16(la.in_proj_qkv_bf16.get(),
                               impl_->ws.normed.get(), impl_->ws.qkv.get(),
                               steps, lin_qkv_dim, dim);
          impl_->RunLinearBf16(la.in_proj_z_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.z_or_k.get(), steps, lin_v_dim, dim);
          impl_->RunLinearBf16(la.in_proj_b_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.b_buf.get(), steps,
                               cfg.linear_num_value_heads, dim);
          impl_->RunLinearBf16(la.in_proj_a_bf16.get(), impl_->ws.normed.get(),
                               impl_->ws.a_buf.get(), steps,
                               cfg.linear_num_value_heads, dim);
        }

        LaunchBackboneLinearConv1dSilu(
            impl_->ws.qkv.get(), la.conv_weight.get(), la.conv_state.get(),
            impl_->ws.qkv_conv.get(), steps, lin_qkv_dim, impl_->stream);

        LaunchBackboneGatedDeltaNetRecurrent(
            impl_->ws.qkv_conv.get(), impl_->ws.z_or_k.get(),
            impl_->ws.b_buf.get(), impl_->ws.a_buf.get(), la.dt_bias.get(),
            la.neg_exp_a.get(), la.norm_weight.get(), la.ssm_state.get(),
            impl_->ws.attn_out.get(), steps, cfg.linear_num_key_heads,
            cfg.linear_num_value_heads, cfg.linear_value_head_dim,
            cfg.rms_norm_eps, impl_->stream);

        impl_->RunLinearBf16(la.out_proj_bf16.get(), impl_->ws.attn_out.get(),
                             impl_->ws.proj_out.get(), steps, dim, lin_v_dim);
      }

      // Fused Residual Add + Post-Attention RMSNorm
      LaunchBackboneResidualAddRmsNorm(
          impl_->ws.hidden.get(), impl_->ws.proj_out.get(), lw.post_norm.get(),
          impl_->ws.normed.get(), steps, dim, cfg.rms_norm_eps, impl_->stream);

      // MoE Router + FP8 Routed Experts + BF16 Shared Expert
      LaunchBackboneMoERouter(
          lw.router_weight_bf16.get(), lw.shared_gate_weight_bf16.get(),
          impl_->ws.normed.get(), impl_->ws.top_indices.get(),
          impl_->ws.top_weights.get(), impl_->ws.shared_gate_scale.get(), steps,
          lw.num_experts, cfg.num_experts_per_tok, dim, impl_->stream);

      LaunchBackboneMoEGateUpSwiGlu(
          lw.experts_gate_fp8.get(), lw.experts_gate_scale.get(),
          lw.experts_up_fp8.get(), lw.experts_up_scale.get(),
          lw.shared_gate_proj_bf16.get(), lw.shared_up_proj_bf16.get(),
          impl_->ws.normed.get(), impl_->ws.top_indices.get(),
          impl_->ws.expert_inter.get(), steps, cfg.num_experts_per_tok,
          cfg.intermediate_size, dim, impl_->stream);

      LaunchBackboneMoEDownCombine(
          lw.experts_down_fp8.get(), lw.experts_down_scale.get(),
          lw.shared_down_proj_bf16.get(), impl_->ws.expert_inter.get(),
          impl_->ws.top_indices.get(), impl_->ws.top_weights.get(),
          impl_->ws.shared_gate_scale.get(), impl_->ws.hidden.get(), steps,
          cfg.num_experts_per_tok, cfg.intermediate_size, dim, impl_->stream);
    }

    // Final RMSNorm
    LaunchBackboneRmsNorm(impl_->ws.hidden.get(), impl_->final_norm.get(),
                          impl_->ws.normed.get(), steps, dim, cfg.rms_norm_eps,
                          impl_->stream);

    if (out_last_logits != nullptr) {
      const float* last_normed = impl_->ws.normed.get() + (steps - 1U) * dim;
      LaunchBackboneBf16Gemv(impl_->compact_lm_head_bf16.get(), last_normed,
                             impl_->ws.logits.get(), 1U,
                             impl_->compact_lm_head_rows, dim, impl_->stream);
      out_last_logits->resize(impl_->compact_lm_head_rows);
      RequireHip(hipMemcpyAsync(out_last_logits->data(), impl_->ws.logits.get(),
                                impl_->compact_lm_head_rows * sizeof(float),
                                hipMemcpyDeviceToHost, impl_->stream),
                 "download fused lm_head logits");
    }

    out_hidden->resize(steps * dim);
    RequireHip(hipMemcpyAsync(out_hidden->data(), impl_->ws.normed.get(),
                              steps * dim * sizeof(float),
                              hipMemcpyDeviceToHost, impl_->stream),
               "download backbone final hidden");
    RequireHip(hipStreamSynchronize(impl_->stream),
               "synchronize Drama 3 backbone forward");
    impl_->cur_steps += steps;
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

bool BackboneHipRuntime::ComputeLmHeadLogits(std::span<const float> last_hidden,
                                             std::vector<float>* out_logits,
                                             std::string* error) {
  const std::size_t dim = impl_->cfg.hidden_size;
  if (out_logits == nullptr || last_hidden.size() != dim) {
    SetError(error, "Invalid Drama 3 ComputeLmHeadLogits arguments");
    return false;
  }
  try {
    RequireHip(hipMemcpyAsync(impl_->ws.normed.get(), last_hidden.data(),
                              dim * sizeof(float), hipMemcpyHostToDevice,
                              impl_->stream),
               "upload lm_head input");
    LaunchBackboneBf16Gemv(impl_->compact_lm_head_bf16.get(),
                           impl_->ws.normed.get(), impl_->ws.logits.get(), 1U,
                           impl_->compact_lm_head_rows, dim, impl_->stream);
    out_logits->resize(impl_->compact_lm_head_rows);
    RequireHip(hipMemcpyAsync(out_logits->data(), impl_->ws.logits.get(),
                              impl_->compact_lm_head_rows * sizeof(float),
                              hipMemcpyDeviceToHost, impl_->stream),
               "download lm_head logits");
    RequireHip(hipStreamSynchronize(impl_->stream),
               "synchronize Drama 3 lm_head");
    return true;
  } catch (const std::exception& ex) {
    SetError(error, ex.what());
    return false;
  }
}

std::size_t BackboneHipRuntime::valid_steps() const noexcept {
  return impl_->cur_steps;
}

const BackboneConfig& BackboneHipRuntime::config() const noexcept {
  return impl_->cfg;
}

}  // namespace gufo::models::fish_drama3::hip

#else

namespace gufo::models::fish_drama3::hip {

struct BackboneHipRuntime::Impl {
  BackboneConfig cfg;
};

BackboneHipRuntime::BackboneHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
BackboneHipRuntime::~BackboneHipRuntime() = default;

std::unique_ptr<BackboneHipRuntime> BackboneHipRuntime::Create(
    const ModelSafetensors&, const ModelConfig&, std::size_t,
    std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 backbone runtime requires a HIP-enabled build";
  }
  return nullptr;
}

void BackboneHipRuntime::Reset() {}
bool BackboneHipRuntime::SavePrefixState(std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 backbone runtime requires a HIP-enabled build";
  }
  return false;
}
bool BackboneHipRuntime::RestorePrefixState(std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 backbone runtime requires a HIP-enabled build";
  }
  return false;
}
std::size_t BackboneHipRuntime::saved_prefix_steps() const noexcept {
  return 0;
}
std::vector<float> BackboneHipRuntime::TokenEmbedding(
    std::span<const std::int32_t>) const {
  return {};
}
bool BackboneHipRuntime::ForwardEmbeddings(std::span<const float>, std::size_t,
                                           std::vector<float>*,
                                           std::string* error,
                                           std::vector<float>*) {
  if (error != nullptr) {
    *error = "Drama 3 backbone runtime requires a HIP-enabled build";
  }
  return false;
}
bool BackboneHipRuntime::ComputeLmHeadLogits(std::span<const float>,
                                             std::vector<float>*,
                                             std::string* error) {
  if (error != nullptr) {
    *error = "Drama 3 backbone runtime requires a HIP-enabled build";
  }
  return false;
}
std::size_t BackboneHipRuntime::valid_steps() const noexcept {
  return 0;
}
const BackboneConfig& BackboneHipRuntime::config() const noexcept {
  return impl_->cfg;
}

}  // namespace gufo::models::fish_drama3::hip

#endif
