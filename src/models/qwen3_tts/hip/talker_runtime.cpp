#include "src/models/qwen3_tts/hip/talker_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen3_tts/hip/talker_ops.hpp"
#include "src/models/qwen3_tts/loader.hpp"

namespace gufo::models::qwen3_tts::hip {
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

void RequireHipblas(hipblasStatus_t status, std::string_view operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(status));
  }
}

float Bfloat16ToFloat(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { Reset(count); }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      RequireHip(hipFree(data_), "hipFree");
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Qwen3-TTS HIP buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc");
    count_ = count;
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

class DeviceRegion {
public:
  DeviceRegion() = default;
  ~DeviceRegion() {
    if (owns_device_ && allocation_ != nullptr) {
      (void)hipFree(allocation_);
    }
    if (registered_ && host_ != nullptr) {
      (void)hipHostUnregister(const_cast<std::byte*>(host_));
    }
  }

  DeviceRegion(const DeviceRegion&) = delete;
  DeviceRegion& operator=(const DeviceRegion&) = delete;

  bool Initialize(MappedRegion region, std::string* error) {
    host_ = region.data;
    size_ = region.size;
    payload_offset_ = region.payload_offset;
    core::PrefaultMappedRange(host_, size_);
    // gfx1151 shares one physical memory pool with the host, but
    // device-resident weights still use coarse-grained pages the GPU caches and
    // streams far faster than host-registered pages, so decode prefers the
    // copy.
    if (TryCopy() || TryMap())
      return true;
    SetError(error, "cannot make Qwen3-TTS safetensors GPU-visible");
    return false;
  }

  [[nodiscard]] const void* Resolve(const std::byte* pointer) const {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base = reinterpret_cast<std::uintptr_t>(host_);
    if (address < base || address - base >= size_) {
      return nullptr;
    }
    return static_cast<const std::byte*>(device_) + (address - base);
  }

private:
  bool TryMap() {
    const hipError_t registered =
        hipHostRegister(const_cast<std::byte*>(host_), size_,
                        hipHostRegisterMapped | hipHostRegisterReadOnly);
    if (registered != hipSuccess) {
      return false;
    }
    registered_ = true;
    if (hipHostGetDevicePointer(&device_, const_cast<std::byte*>(host_), 0) ==
        hipSuccess) {
      return true;
    }
    (void)hipHostUnregister(const_cast<std::byte*>(host_));
    registered_ = false;
    device_ = nullptr;
    return false;
  }

  bool TryCopy() {
    // Safetensors headers are an arbitrary length, so the payload rarely starts
    // on a 16-byte boundary. Padding the copy by that much lets every tensor
    // land where the widest packed loads are legal.
    constexpr std::size_t kLoadAlignment = 16;
    const std::size_t shift =
        (kLoadAlignment - (payload_offset_ % kLoadAlignment)) % kLoadAlignment;
    void* allocation = nullptr;
    if (hipMalloc(&allocation, size_ + shift) != hipSuccess) {
      return false;
    }
    allocation_ = allocation;
    owns_device_ = true;
    device_ = static_cast<std::byte*>(allocation) + shift;
    if (hipMemcpy(device_, host_, size_, hipMemcpyHostToDevice) == hipSuccess) {
      return true;
    }
    (void)hipFree(allocation_);
    owns_device_ = false;
    allocation_ = nullptr;
    device_ = nullptr;
    return false;
  }

  const std::byte* host_{nullptr};
  void* device_{nullptr};
  void* allocation_{nullptr};
  std::size_t size_{0};
  std::size_t payload_offset_{0};
  bool owns_device_{false};
  bool registered_{false};
};

const Tensor& RequireTensor(const TensorStore& store, std::string_view name,
                            std::span<const std::uint64_t> shape) {
  const Tensor* tensor = store.Find(name);
  if (tensor == nullptr) {
    throw std::runtime_error("missing Qwen3-TTS tensor: " + std::string(name));
  }
  if (tensor->dtype != DType::kBF16 ||
      !std::ranges::equal(tensor->shape, shape)) {
    throw std::runtime_error("unexpected Qwen3-TTS tensor layout: " +
                             std::string(name));
  }
  return *tensor;
}

std::vector<float> DecodeBfloat16(const Tensor& tensor) {
  const auto elements = tensor.NumElements();
  if (!elements.has_value()) {
    throw std::runtime_error("invalid Qwen3-TTS BF16 tensor");
  }
  std::vector<float> result(*elements);
  const auto* source = reinterpret_cast<const std::uint16_t*>(tensor.data);
  std::transform(source, source + *elements, result.begin(), Bfloat16ToFloat);
  return result;
}

struct DeviceNorm {
  DeviceBuffer<float> values;
};

DeviceNorm UploadNorm(const TensorStore& store, std::string_view name,
                      std::size_t size) {
  const std::array<std::uint64_t, 1> shape{static_cast<std::uint64_t>(size)};
  const Tensor& tensor = RequireTensor(store, name, shape);
  const std::vector<float> host = DecodeBfloat16(tensor);
  DeviceNorm result{DeviceBuffer<float>(host.size())};
  RequireHip(hipMemcpy(result.values.get(), host.data(),
                       host.size() * sizeof(float), hipMemcpyHostToDevice),
             "hipMemcpy norm");
  return result;
}

struct LayerWeights {
  const void* q{nullptr};
  const void* k{nullptr};
  const void* v{nullptr};
  const void* o{nullptr};
  const void* gate{nullptr};
  const void* up{nullptr};
  const void* down{nullptr};
  DeviceNorm input_norm;
  DeviceNorm q_norm;
  DeviceNorm k_norm;
  DeviceNorm post_norm;
};

struct PredictorLayerWeights {
  const void* q{nullptr};
  const void* k{nullptr};
  const void* v{nullptr};
  const void* o{nullptr};
  const void* gate{nullptr};
  const void* up{nullptr};
  const void* down{nullptr};
  DeviceNorm input_norm;
  DeviceNorm q_norm;
  DeviceNorm k_norm;
  DeviceNorm post_norm;
};

std::string LayerName(std::size_t layer, std::string_view suffix) {
  return "talker.model.layers." + std::to_string(layer) + "." +
         std::string(suffix);
}

std::string PredictorLayerName(std::size_t layer, std::string_view suffix) {
  return "talker.code_predictor.model.layers." + std::to_string(layer) + "." +
         std::string(suffix);
}

const void* ResolveMatrix(const TensorStore& store, const DeviceRegion& region,
                          std::string_view name, std::uint64_t rows,
                          std::uint64_t columns) {
  const std::array<std::uint64_t, 2> shape{rows, columns};
  const Tensor& tensor = RequireTensor(store, name, shape);
  const void* result = region.Resolve(tensor.data);
  if (result == nullptr) {
    throw std::runtime_error(
        "Qwen3-TTS tensor is outside the mapped weights: " + std::string(name));
  }
  return result;
}

const void* ResolveVector(const TensorStore& store, const DeviceRegion& region,
                          std::string_view name, std::uint64_t elements) {
  const std::array<std::uint64_t, 1> shape{elements};
  const Tensor& tensor = RequireTensor(store, name, shape);
  const void* result = region.Resolve(tensor.data);
  if (result == nullptr) {
    throw std::runtime_error(
        "Qwen3-TTS tensor is outside the mapped weights: " + std::string(name));
  }
  return result;
}

std::string AsciiLower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

std::uint32_t SelectMainCode(
    std::span<const float> logits,
    std::span<const std::uint8_t> generated_first_codes,
    std::size_t generation_step, std::uint32_t eos_token,
    const SamplingOptions& sampling, std::mt19937* random,
    std::vector<TokenScore>* scratch) {
  std::vector<TokenScore>& candidates = *scratch;
  candidates.clear();
  candidates.reserve(2049);
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    if ((token >= 2048 && token != eos_token) ||
        (generation_step < 2 && token == eos_token)) {
      continue;
    }
    float score = logits[token];
    // A flag per vocabulary entry keeps the penalty test constant time;
    // scanning the generated codes made every frame cost more than the one
    // before it.
    if (token < generated_first_codes.size() &&
        generated_first_codes[token] != 0) {
      score = score < 0.0F ? score * sampling.repetition_penalty
                           : score / sampling.repetition_penalty;
    }
    candidates.push_back({token, score});
  }
  return SampleCodec(candidates, sampling.top_k, sampling.top_p,
                     sampling.temperature, random);
}

}  // namespace

struct TalkerHipRuntime::Impl {
  Impl(LoadResult loaded, std::size_t token_capacity)
      : model(std::move(loaded)),
        maximum_tokens(token_capacity),
        hidden(token_capacity * model.config.talker.hidden_size),
        normalized(token_capacity * model.config.talker.hidden_size),
        q(token_capacity * model.config.talker.hidden_size),
        k(token_capacity * model.config.talker.num_key_value_heads *
          model.config.talker.head_dim),
        v(token_capacity * model.config.talker.num_key_value_heads *
          model.config.talker.head_dim),
        attention_output(token_capacity * model.config.talker.hidden_size),
        gate(token_capacity * model.config.talker.intermediate_size),
        up(token_capacity * model.config.talker.intermediate_size),
        feed_forward_output(token_capacity * model.config.talker.hidden_size),
        bfloat16_scratch(token_capacity *
                         std::max(model.config.talker.hidden_size,
                                  model.config.talker.intermediate_size)),
        key_cache(
            static_cast<std::size_t>(model.config.talker.num_hidden_layers) *
            model.config.talker.num_key_value_heads * token_capacity *
            model.config.talker.head_dim),
        value_cache(
            static_cast<std::size_t>(model.config.talker.num_hidden_layers) *
            model.config.talker.num_key_value_heads * token_capacity *
            model.config.talker.head_dim),
        predictor_key_cache(static_cast<std::size_t>(
                                model.config.code_predictor.num_hidden_layers) *
                            model.config.code_predictor.num_key_value_heads *
                            model.config.talker.num_code_groups *
                            model.config.code_predictor.head_dim),
        predictor_value_cache(
            static_cast<std::size_t>(
                model.config.code_predictor.num_hidden_layers) *
            model.config.code_predictor.num_key_value_heads *
            model.config.talker.num_code_groups *
            model.config.code_predictor.head_dim),
        logits(model.config.talker.vocab_size),
        prompt_ids(token_capacity),
        reference_code_ids(token_capacity *
                           model.config.talker.num_code_groups),
        codec_embedding_tables(model.config.talker.num_code_groups),
        predictor_token(1) {
    const auto talker_regions = model.RegionsFor("talker.");
    if (talker_regions.size() != 1 ||
        !main_weights.Initialize(talker_regions.front(), nullptr)) {
      throw std::runtime_error(
          "cannot initialize Qwen3-TTS GPU weight mapping");
    }
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags");
    RequireHipblas(hipblasCreate(&blas), "hipblasCreate");
    RequireHipblas(hipblasSetStream(blas, stream), "hipblasSetStream");
    // Prefill is the one projection that still runs through the batched GEMM,
    // and its split-K reduction is only reproducible without atomics. Top-k
    // sampling sorts near-equal candidates, so a last-bit prefill difference
    // permutes the distribution and changes the drawn codec token.
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode");
    LoadWeights();
  }

  ~Impl() {
    if (blas != nullptr) {
      (void)hipblasDestroy(blas);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  void LoadWeights() {
    const TensorStore& store = *model.store;
    const auto& config = model.config.talker;
    layers.reserve(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      LayerWeights weights;
      weights.q = ResolveMatrix(
          store, main_weights, LayerName(layer, "self_attn.q_proj.weight"),
          config.num_attention_heads * config.head_dim, config.hidden_size);
      weights.k = ResolveMatrix(
          store, main_weights, LayerName(layer, "self_attn.k_proj.weight"),
          config.num_key_value_heads * config.head_dim, config.hidden_size);
      weights.v = ResolveMatrix(
          store, main_weights, LayerName(layer, "self_attn.v_proj.weight"),
          config.num_key_value_heads * config.head_dim, config.hidden_size);
      weights.o = ResolveMatrix(
          store, main_weights, LayerName(layer, "self_attn.o_proj.weight"),
          config.hidden_size, config.num_attention_heads * config.head_dim);
      weights.gate = ResolveMatrix(
          store, main_weights, LayerName(layer, "mlp.gate_proj.weight"),
          config.intermediate_size, config.hidden_size);
      weights.up = ResolveMatrix(store, main_weights,
                                 LayerName(layer, "mlp.up_proj.weight"),
                                 config.intermediate_size, config.hidden_size);
      weights.down = ResolveMatrix(
          store, main_weights, LayerName(layer, "mlp.down_proj.weight"),
          config.hidden_size, config.intermediate_size);
      weights.input_norm =
          UploadNorm(store, LayerName(layer, "input_layernorm.weight"),
                     config.hidden_size);
      weights.q_norm = UploadNorm(
          store, LayerName(layer, "self_attn.q_norm.weight"), config.head_dim);
      weights.k_norm = UploadNorm(
          store, LayerName(layer, "self_attn.k_norm.weight"), config.head_dim);
      weights.post_norm =
          UploadNorm(store, LayerName(layer, "post_attention_layernorm.weight"),
                     config.hidden_size);
      layers.push_back(std::move(weights));
    }
    final_norm =
        UploadNorm(store, "talker.model.norm.weight", config.hidden_size);
    codec_head = ResolveMatrix(store, main_weights, "talker.codec_head.weight",
                               config.vocab_size, config.hidden_size);
    codec_embedding = ResolveMatrix(store, main_weights,
                                    "talker.model.codec_embedding.weight",
                                    config.vocab_size, config.hidden_size);
    text_embedding =
        ResolveMatrix(store, main_weights, "talker.model.text_embedding.weight",
                      config.text_vocab_size, config.text_hidden_size);
    text_projection_fc1 = ResolveMatrix(
        store, main_weights, "talker.text_projection.linear_fc1.weight",
        config.text_hidden_size, config.text_hidden_size);
    text_projection_fc1_bias = ResolveVector(
        store, main_weights, "talker.text_projection.linear_fc1.bias",
        config.text_hidden_size);
    text_projection_fc2 = ResolveMatrix(
        store, main_weights, "talker.text_projection.linear_fc2.weight",
        config.hidden_size, config.text_hidden_size);
    text_projection_fc2_bias = ResolveVector(
        store, main_weights, "talker.text_projection.linear_fc2.bias",
        config.hidden_size);

    const auto& predictor = model.config.code_predictor;
    predictor_layers.reserve(predictor.num_hidden_layers);
    for (std::size_t layer = 0; layer < predictor.num_hidden_layers; ++layer) {
      PredictorLayerWeights weights;
      weights.q =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "self_attn.q_proj.weight"),
                        predictor.num_attention_heads * predictor.head_dim,
                        predictor.hidden_size);
      weights.k =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "self_attn.k_proj.weight"),
                        predictor.num_key_value_heads * predictor.head_dim,
                        predictor.hidden_size);
      weights.v =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "self_attn.v_proj.weight"),
                        predictor.num_key_value_heads * predictor.head_dim,
                        predictor.hidden_size);
      weights.o =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "self_attn.o_proj.weight"),
                        predictor.hidden_size,
                        predictor.num_attention_heads * predictor.head_dim);
      weights.gate =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "mlp.gate_proj.weight"),
                        predictor.intermediate_size, predictor.hidden_size);
      weights.up = ResolveMatrix(
          store, main_weights, PredictorLayerName(layer, "mlp.up_proj.weight"),
          predictor.intermediate_size, predictor.hidden_size);
      weights.down =
          ResolveMatrix(store, main_weights,
                        PredictorLayerName(layer, "mlp.down_proj.weight"),
                        predictor.hidden_size, predictor.intermediate_size);
      weights.input_norm =
          UploadNorm(store, PredictorLayerName(layer, "input_layernorm.weight"),
                     predictor.hidden_size);
      weights.q_norm = UploadNorm(
          store, PredictorLayerName(layer, "self_attn.q_norm.weight"),
          predictor.head_dim);
      weights.k_norm = UploadNorm(
          store, PredictorLayerName(layer, "self_attn.k_norm.weight"),
          predictor.head_dim);
      weights.post_norm = UploadNorm(
          store, PredictorLayerName(layer, "post_attention_layernorm.weight"),
          predictor.hidden_size);
      predictor_layers.push_back(std::move(weights));
    }
    predictor_norm =
        UploadNorm(store, "talker.code_predictor.model.norm.weight",
                   predictor.hidden_size);
    predictor_projection =
        ResolveMatrix(store, main_weights,
                      "talker.code_predictor.small_to_mtp_projection.weight",
                      predictor.hidden_size, config.hidden_size);
    predictor_projection_bias =
        ResolveVector(store, main_weights,
                      "talker.code_predictor.small_to_mtp_projection.bias",
                      predictor.hidden_size);
    predictor_embeddings.reserve(talker_code_groups());
    predictor_heads.reserve(talker_code_groups());
    for (std::size_t group = 0; group + 1 < model.config.talker.num_code_groups;
         ++group) {
      predictor_embeddings.push_back(
          ResolveMatrix(store, main_weights,
                        "talker.code_predictor.model.codec_embedding." +
                            std::to_string(group) + ".weight",
                        predictor.vocab_size, config.hidden_size));
      predictor_heads.push_back(ResolveMatrix(
          store, main_weights,
          "talker.code_predictor.lm_head." + std::to_string(group) + ".weight",
          predictor.vocab_size, predictor.hidden_size));
    }
    std::vector<const void*> embedding_tables;
    embedding_tables.reserve(model.config.talker.num_code_groups);
    embedding_tables.push_back(codec_embedding);
    embedding_tables.insert(embedding_tables.end(),
                            predictor_embeddings.begin(),
                            predictor_embeddings.end());
    RequireHip(hipMemcpy(codec_embedding_tables.get(), embedding_tables.data(),
                         embedding_tables.size() * sizeof(const void*),
                         hipMemcpyHostToDevice),
               "hipMemcpy codec embedding table pointers");
  }

  [[nodiscard]] std::size_t talker_code_groups() const {
    return model.config.talker.num_code_groups - 1;
  }

  void Gemm(const void* weights, const void* inputs_bfloat16, float* output,
            std::size_t batch, std::size_t rows, std::size_t columns) {
    // Decode drives one or two tokens at a time, where the batched GEMM pads
    // the few columns into a tile and reaches only a fraction of DRAM
    // bandwidth. The model-private GEMV streams each weight row instead.
    if (LaunchBfloat16Gemv(weights, inputs_bfloat16, output, batch, rows,
                           columns, stream)) {
      return;
    }
    gufo::hip::LaunchHipblasGEMMBF16(blas, weights, inputs_bfloat16, output,
                                     batch, rows, columns, stream);
  }

  /// Runs projections that share one input. The grouped GEMV produces the same
  /// values as the per-projection calls, so this only trades dispatches; when
  /// it declines the shape, each group runs on its own.
  void GemmGrouped(std::span<const Bfloat16GemvGroup> groups,
                   const void* inputs_bfloat16, std::size_t batch,
                   std::size_t columns) {
    if (LaunchBfloat16GemvGrouped(groups.data(), groups.size(), inputs_bfloat16,
                                  batch, columns, stream)) {
      return;
    }
    for (const Bfloat16GemvGroup& group : groups) {
      Gemm(group.weights_bfloat16, inputs_bfloat16, group.output, batch,
           group.rows, columns);
    }
  }

  /// Normalizes and rotates q/k and rounds v in one launch where the head fits
  /// a workgroup, otherwise falls back to the separate kernels.
  /// Returns true when the fused kernel also filled the attention cache, which
  /// lets the caller skip the standalone cache-write launch.
  [[nodiscard]] bool NormalizeRotateQkv(
      float* q_values, float* k_values, float* v_values, const float* q_weight,
      const float* k_weight, std::size_t batch_size, std::uint32_t q_heads,
      std::uint32_t kv_heads, std::uint32_t head_dim,
      std::uint32_t start_position, float rope_theta, float epsilon,
      float* key_cache_values, float* value_cache_values,
      std::size_t layer_index, std::size_t max_context) {
    if (LaunchBfloat16QkNormRoPE(
            q_values, k_values, v_values, q_weight, k_weight, batch_size,
            q_heads, kv_heads, head_dim, start_position, rope_theta, epsilon,
            key_cache_values, value_cache_values,
            static_cast<std::uint32_t>(layer_index),
            static_cast<std::uint32_t>(max_context), stream)) {
      return key_cache_values != nullptr;
    }
    LaunchBfloat16PerHeadRMSNorm(q_values, q_weight, batch_size, q_heads,
                                 head_dim, epsilon, stream);
    LaunchBfloat16PerHeadRMSNorm(k_values, k_weight, batch_size, kv_heads,
                                 head_dim, epsilon, stream);
    LaunchBfloat16RoPEAndRoundV(q_values, k_values, v_values, batch_size,
                                q_heads, kv_heads, head_dim, start_position,
                                rope_theta, stream);
    return false;
  }

  void RmsNormToBfloat16(float* input, const float* weight,
                         std::size_t batch_size, std::size_t dimension,
                         float epsilon) {
    LaunchBfloat16ResidualAddRMSNorm(input, nullptr, weight,
                                     bfloat16_scratch.get(), batch_size,
                                     dimension, epsilon, stream);
  }

  /// Folds a residual add into the RMSNorm that always follows it. The float
  /// normalization output is unused inside the layers, so only the BF16 GEMM
  /// input is produced.
  void ResidualAddNormToBfloat16(const float* update, const float* weight,
                                 std::size_t batch_size, std::size_t dimension,
                                 float epsilon) {
    LaunchBfloat16ResidualAddRMSNorm(hidden.get(), update, weight,
                                     bfloat16_scratch.get(), batch_size,
                                     dimension, epsilon, stream);
  }

  std::vector<float> ProjectText(std::span<const std::uint32_t> token_ids) {
    const auto& config = model.config.talker;
    if (token_ids.empty()) {
      return {};
    }
    if (token_ids.size() > maximum_tokens) {
      throw std::length_error("Qwen3-TTS text projection exceeds capacity");
    }
    for (const std::uint32_t token : token_ids) {
      if (token >= config.text_vocab_size) {
        throw std::out_of_range("Qwen3-TTS text token is out of range");
      }
    }
    const std::size_t elements = token_ids.size() * config.text_hidden_size;
    RequireHip(hipMemcpyAsync(prompt_ids.get(), token_ids.data(),
                              token_ids.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync text token ids");
    gufo::hip::LaunchBatchedEmbeddingLookup(
        text_embedding, core::GgmlType::kBF16, prompt_ids.get(), hidden.get(),
        token_ids.size(), config.text_hidden_size, stream);
    gufo::hip::LaunchFloatToBfloat16(hidden.get(), bfloat16_scratch.get(),
                                     elements, stream);
    Gemm(text_projection_fc1, bfloat16_scratch.get(), normalized.get(),
         token_ids.size(), config.text_hidden_size, config.text_hidden_size);
    LaunchBfloat16BiasSilu(normalized.get(), text_projection_fc1_bias,
                           hidden.get(), bfloat16_scratch.get(),
                           token_ids.size(), config.text_hidden_size, stream);
    Gemm(text_projection_fc2, bfloat16_scratch.get(), normalized.get(),
         token_ids.size(), config.hidden_size, config.text_hidden_size);
    LaunchBfloat16Bias(normalized.get(), text_projection_fc2_bias, hidden.get(),
                       token_ids.size(), config.hidden_size, stream);
    std::vector<float> result(token_ids.size() * config.hidden_size);
    RequireHip(hipMemcpyAsync(result.data(), hidden.get(),
                              result.size() * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync projected text");
    RequireHip(hipStreamSynchronize(stream), "hipStreamSynchronize text");
    RequireHip(hipGetLastError(), "Qwen3-TTS text projection");
    return result;
  }

  std::vector<float> LookupCodec(std::span<const std::uint32_t> token_ids) {
    const auto& config = model.config.talker;
    if (token_ids.empty()) {
      return {};
    }
    if (token_ids.size() > maximum_tokens) {
      throw std::length_error("Qwen3-TTS codec lookup exceeds capacity");
    }
    for (const std::uint32_t token : token_ids) {
      if (token >= config.vocab_size) {
        throw std::out_of_range("Qwen3-TTS codec token is out of range");
      }
    }
    RequireHip(hipMemcpyAsync(prompt_ids.get(), token_ids.data(),
                              token_ids.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync codec token ids");
    gufo::hip::LaunchBatchedEmbeddingLookup(
        codec_embedding, core::GgmlType::kBF16, prompt_ids.get(), hidden.get(),
        token_ids.size(), config.hidden_size, stream);
    std::vector<float> result(token_ids.size() * config.hidden_size);
    RequireHip(hipMemcpyAsync(result.data(), hidden.get(),
                              result.size() * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync codec embeddings");
    RequireHip(hipStreamSynchronize(stream), "hipStreamSynchronize codec");
    RequireHip(hipGetLastError(), "Qwen3-TTS codec lookup");
    return result;
  }

  void ProjectPredictorInput(const float* input, std::size_t rows) {
    const auto& talker = model.config.talker;
    const auto& predictor = model.config.code_predictor;
    if (rows == 0 || rows > talker.num_code_groups) {
      throw std::length_error("Qwen3-TTS predictor input shape is invalid");
    }
    gufo::hip::LaunchFloatToBfloat16(input, bfloat16_scratch.get(),
                                     rows * talker.hidden_size, stream);
    Gemm(predictor_projection, bfloat16_scratch.get(), hidden.get(), rows,
         predictor.hidden_size, talker.hidden_size);
    LaunchBfloat16Bias(hidden.get(), predictor_projection_bias, hidden.get(),
                       rows, predictor.hidden_size, stream);
  }

  std::uint32_t RunPredictor(std::size_t tokens, std::uint32_t start_position,
                             std::size_t head_index,
                             std::vector<float>* trace_logits) {
    const auto& config = model.config.code_predictor;
    const std::size_t context = model.config.talker.num_code_groups;
    if (tokens == 0 || start_position + tokens > context ||
        head_index >= predictor_heads.size()) {
      throw std::length_error("Qwen3-TTS predictor step is out of range");
    }
    const std::size_t ffn_elements = tokens * config.intermediate_size;

    const auto launch_predictor = [&] {
      RmsNormToBfloat16(hidden.get(),
                        predictor_layers.front().input_norm.values.get(),
                        tokens, config.hidden_size, config.rms_norm_eps);
      for (std::size_t layer = 0; layer < predictor_layers.size(); ++layer) {
        const PredictorLayerWeights& weights = predictor_layers[layer];
        const std::array<Bfloat16GemvGroup, 3> qkv{{
            {weights.q, q.get(), config.num_attention_heads * config.head_dim},
            {weights.k, k.get(), config.num_key_value_heads * config.head_dim},
            {weights.v, v.get(), config.num_key_value_heads * config.head_dim},
        }};
        GemmGrouped(qkv, bfloat16_scratch.get(), tokens, config.hidden_size);
        const bool cached = NormalizeRotateQkv(
            q.get(), k.get(), v.get(), weights.q_norm.values.get(),
            weights.k_norm.values.get(), tokens, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, start_position,
            config.rope_theta, config.rms_norm_eps, predictor_key_cache.get(),
            predictor_value_cache.get(), layer, context);
        LaunchBfloat16Attention(
            q.get(), k.get(), v.get(), predictor_key_cache.get(),
            predictor_value_cache.get(), bfloat16_scratch.get(), layer,
            start_position, tokens, context, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, stream, cached);
        Gemm(weights.o, bfloat16_scratch.get(), attention_output.get(), tokens,
             config.hidden_size, config.num_attention_heads * config.head_dim);
        ResidualAddNormToBfloat16(attention_output.get(),
                                  weights.post_norm.values.get(), tokens,
                                  config.hidden_size, config.rms_norm_eps);
        const std::array<Bfloat16GemvGroup, 2> gate_up{{
            {weights.gate, gate.get(), config.intermediate_size},
            {weights.up, up.get(), config.intermediate_size},
        }};
        GemmGrouped(gate_up, bfloat16_scratch.get(), tokens,
                    config.hidden_size);
        LaunchBfloat16SwiGlu(gate.get(), up.get(), nullptr,
                             bfloat16_scratch.get(), ffn_elements, stream);
        Gemm(weights.down, bfloat16_scratch.get(), feed_forward_output.get(),
             tokens, config.hidden_size, config.intermediate_size);
        // The next layer opens with an RMSNorm, so the trailing residual folds
        // into it; the last layer folds into the predictor's final norm.
        ResidualAddNormToBfloat16(
            feed_forward_output.get(),
            layer + 1 < predictor_layers.size()
                ? predictor_layers[layer + 1].input_norm.values.get()
                : predictor_norm.values.get(),
            tokens, config.hidden_size, config.rms_norm_eps);
      }

      const auto* last_hidden_bfloat16 =
          bfloat16_scratch.get() + ((tokens - 1) * config.hidden_size);
      Gemm(predictor_heads[head_index], last_hidden_bfloat16, logits.get(), 1,
           config.vocab_size, config.hidden_size);
      LaunchRoundBfloat16InPlace(logits.get(), config.vocab_size, stream);
    };

    launch_predictor();
    std::vector<float>* host_logits = trace_logits;
    if (sampling_rng != nullptr && host_logits == nullptr) {
      host_logits = &predictor_logit_scratch;
    }
    if (host_logits != nullptr) {
      host_logits->resize(config.vocab_size);
      RequireHip(hipMemcpyAsync(host_logits->data(), logits.get(),
                                config.vocab_size * sizeof(float),
                                hipMemcpyDeviceToHost, stream),
                 "hipMemcpyAsync predictor logits");
    }
    if (sampling_rng != nullptr) {
      RequireHip(hipStreamSynchronize(stream),
                 "hipStreamSynchronize predictor sampling");
      RequireHip(hipGetLastError(), "Qwen3-TTS code predictor sampling");
      candidate_scratch.clear();
      candidate_scratch.reserve(host_logits->size());
      for (std::uint32_t token = 0; token < host_logits->size(); ++token) {
        candidate_scratch.push_back({token, (*host_logits)[token]});
      }
      return SampleCodec(candidate_scratch, predictor_top_k, predictor_top_p,
                         predictor_temperature, sampling_rng);
    }
    gufo::hip::LaunchGPUArgmax(logits.get(), predictor_token.get(),
                               config.vocab_size, stream);
    std::uint32_t token = 0;
    RequireHip(hipMemcpyAsync(&token, predictor_token.get(), sizeof(token),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync predictor token");
    RequireHip(hipStreamSynchronize(stream), "hipStreamSynchronize predictor");
    RequireHip(hipGetLastError(), "Qwen3-TTS code predictor");
    return gufo::hip::CheckedSampleToken(token, config.vocab_size);
  }

  void RunTalkerToken(TalkerPrefillOutput* output) {
    const auto& config = model.config.talker;
    if (talker_cache_tokens == 0 || talker_cache_tokens >= maximum_tokens) {
      throw std::length_error("Qwen3-TTS talker cache position is invalid");
    }
    constexpr std::size_t tokens = 1;
    const std::size_t ffn_elements = config.intermediate_size;

    RmsNormToBfloat16(hidden.get(), layers.front().input_norm.values.get(),
                      tokens, config.hidden_size, config.rms_norm_eps);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
      const LayerWeights& weights = layers[layer];
      const std::array<Bfloat16GemvGroup, 3> qkv{{
          {weights.q, q.get(), config.num_attention_heads * config.head_dim},
          {weights.k, k.get(), config.num_key_value_heads * config.head_dim},
          {weights.v, v.get(), config.num_key_value_heads * config.head_dim},
      }};
      GemmGrouped(qkv, bfloat16_scratch.get(), tokens, config.hidden_size);
      const bool cached = NormalizeRotateQkv(
          q.get(), k.get(), v.get(), weights.q_norm.values.get(),
          weights.k_norm.values.get(), tokens, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim,
          static_cast<std::uint32_t>(talker_cache_tokens), config.rope_theta,
          config.rms_norm_eps, key_cache.get(), value_cache.get(), layer,
          maximum_tokens);
      LaunchBfloat16Attention(
          q.get(), k.get(), v.get(), key_cache.get(), value_cache.get(),
          bfloat16_scratch.get(), layer, talker_cache_tokens, tokens,
          maximum_tokens, config.num_attention_heads,
          config.num_key_value_heads, config.head_dim, stream, cached);
      Gemm(weights.o, bfloat16_scratch.get(), attention_output.get(), tokens,
           config.hidden_size, config.num_attention_heads * config.head_dim);
      ResidualAddNormToBfloat16(attention_output.get(),
                                weights.post_norm.values.get(), tokens,
                                config.hidden_size, config.rms_norm_eps);
      const std::array<Bfloat16GemvGroup, 2> gate_up{{
          {weights.gate, gate.get(), config.intermediate_size},
          {weights.up, up.get(), config.intermediate_size},
      }};
      GemmGrouped(gate_up, bfloat16_scratch.get(), tokens, config.hidden_size);
      LaunchBfloat16SwiGlu(gate.get(), up.get(), nullptr,
                           bfloat16_scratch.get(), ffn_elements, stream);
      Gemm(weights.down, bfloat16_scratch.get(), feed_forward_output.get(),
           tokens, config.hidden_size, config.intermediate_size);
      // The next layer opens with an RMSNorm, so the trailing residual folds
      // into it; the last layer folds into the talker's final norm.
      ResidualAddNormToBfloat16(
          feed_forward_output.get(),
          layer + 1 < layers.size() ? layers[layer + 1].input_norm.values.get()
                                    : final_norm.values.get(),
          tokens, config.hidden_size, config.rms_norm_eps);
    }

    Gemm(codec_head, bfloat16_scratch.get(), logits.get(), 1, config.vocab_size,
         config.hidden_size);
    LaunchRoundBfloat16InPlace(logits.get(), config.vocab_size, stream);
    LaunchBfloat16ToFloat(bfloat16_scratch.get(), attention_output.get(),
                          config.hidden_size, stream);

    output->last_hidden.resize(config.hidden_size);
    output->logits.resize(config.vocab_size);
    RequireHip(
        hipMemcpyAsync(output->last_hidden.data(), attention_output.get(),
                       config.hidden_size * sizeof(float),
                       hipMemcpyDeviceToHost, stream),
        "hipMemcpyAsync talker hidden");
    RequireHip(hipMemcpyAsync(output->logits.data(), logits.get(),
                              config.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync talker logits");
    RequireHip(hipStreamSynchronize(stream), "hipStreamSynchronize talker");
    RequireHip(hipGetLastError(), "Qwen3-TTS cached talker step");
    ++talker_cache_tokens;
  }

  void PrepareCodeFrameEmbedding(std::span<const std::uint32_t> codes,
                                 std::span<const float> text_embedding) {
    const auto& talker = model.config.talker;
    if (codes.size() != talker.num_code_groups) {
      throw std::length_error("Qwen3-TTS codec frame group count is invalid");
    }
    RequireHip(hipMemcpyAsync(feed_forward_output.get(), text_embedding.data(),
                              talker.hidden_size * sizeof(float),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync trailing text embedding");
    // One dispatch replaces a lookup per code group plus the summing pass. The
    // accumulation walks the groups in the same order, so the rounded row is
    // unchanged.
    RequireHip(hipMemcpyAsync(reference_code_ids.get(), codes.data(),
                              codes.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync codec frame codes");
    LaunchBatchedCodecEmbeddingSum(
        codec_embedding_tables.get(), reference_code_ids.get(),
        feed_forward_output.get(), hidden.get(), 1, talker.num_code_groups,
        talker.hidden_size, stream);
  }

  std::vector<float> BuildReferenceCodecEmbeddings(
      std::span<const std::uint32_t> codes, std::size_t frames) {
    const auto& talker = model.config.talker;
    if (frames == 0 || frames > maximum_tokens ||
        codes.size() != frames * talker.num_code_groups) {
      throw std::length_error(
          "Qwen3-TTS reference codec embedding shape is invalid");
    }
    RequireHip(hipMemcpyAsync(reference_code_ids.get(), codes.data(),
                              codes.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync reference codec codes");
    LaunchBatchedCodecEmbeddingSum(codec_embedding_tables.get(),
                                   reference_code_ids.get(), nullptr,
                                   hidden.get(), frames, talker.num_code_groups,
                                   talker.hidden_size, stream);
    std::vector<float> result(frames * talker.hidden_size);
    RequireHip(hipMemcpyAsync(result.data(), hidden.get(),
                              result.size() * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync reference codec embeddings");
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize reference codec embeddings");
    RequireHip(hipGetLastError(), "Qwen3-TTS reference codec embeddings");
    return result;
  }

  LoadResult model;
  std::size_t maximum_tokens;
  std::size_t talker_cache_tokens{0};
  DeviceRegion main_weights;
  std::vector<LayerWeights> layers;
  std::vector<PredictorLayerWeights> predictor_layers;
  DeviceNorm final_norm;
  DeviceNorm predictor_norm;
  const void* codec_head{nullptr};
  const void* codec_embedding{nullptr};
  const void* text_embedding{nullptr};
  const void* text_projection_fc1{nullptr};
  const void* text_projection_fc1_bias{nullptr};
  const void* text_projection_fc2{nullptr};
  const void* text_projection_fc2_bias{nullptr};
  const void* predictor_projection{nullptr};
  const void* predictor_projection_bias{nullptr};
  std::vector<const void*> predictor_embeddings;
  std::vector<const void*> predictor_heads;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> normalized;
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> attention_output;
  DeviceBuffer<float> gate;
  DeviceBuffer<float> up;
  DeviceBuffer<float> feed_forward_output;
  DeviceBuffer<hip_bfloat16> bfloat16_scratch;
  DeviceBuffer<float> key_cache;
  DeviceBuffer<float> value_cache;
  DeviceBuffer<float> predictor_key_cache;
  DeviceBuffer<float> predictor_value_cache;
  DeviceBuffer<float> logits;
  DeviceBuffer<std::uint32_t> prompt_ids;
  DeviceBuffer<std::uint32_t> reference_code_ids;
  DeviceBuffer<const void*> codec_embedding_tables;
  DeviceBuffer<std::uint32_t> predictor_token;
  std::mt19937* sampling_rng{nullptr};
  std::size_t predictor_top_k{50};
  float predictor_temperature{0.9F};
  float predictor_top_p{1.0F};
  // Reused across the sixteen sampling steps of every frame so that decode does
  // not allocate per step.
  std::vector<float> predictor_logit_scratch;
  std::vector<TokenScore> candidate_scratch;
  std::vector<std::uint8_t> generated_first_code_flags;
};

TalkerHipRuntime::TalkerHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TalkerHipRuntime::~TalkerHipRuntime() = default;

std::unique_ptr<TalkerHipRuntime> TalkerHipRuntime::Create(
    const std::string& model_root, std::size_t maximum_tokens,
    std::string* error) {
  if (maximum_tokens == 0) {
    SetError(error, "Qwen3-TTS maximum token count must be positive");
    return nullptr;
  }
  try {
    LoadResult model = LoadModelDirectory(model_root);
    if (!model.ok) {
      SetError(error, std::move(model.error));
      return nullptr;
    }
    auto impl = std::make_unique<Impl>(std::move(model), maximum_tokens);
    return std::unique_ptr<TalkerHipRuntime>(
        new TalkerHipRuntime(std::move(impl)));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool TalkerHipRuntime::Prefill(std::span<const float> input_embeddings,
                               std::size_t tokens, TalkerPrefillOutput* output,
                               std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prefill output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config.talker;
  if (tokens == 0 || tokens > impl_->maximum_tokens ||
      input_embeddings.size() != tokens * config.hidden_size) {
    SetError(error, "Qwen3-TTS prefill embedding shape is invalid");
    return false;
  }

  try {
    impl_->talker_cache_tokens = 0;
    const std::size_t hidden_elements = tokens * config.hidden_size;
    const std::size_t ffn_elements = tokens * config.intermediate_size;
    RequireHip(hipMemcpyAsync(impl_->hidden.get(), input_embeddings.data(),
                              hidden_elements * sizeof(float),
                              hipMemcpyHostToDevice, impl_->stream),
               "hipMemcpyAsync prefill embeddings");
    // Each layer overwrites the entire visible prefix before causal attention.
    // Unused cache capacity is never read, including after a cancelled request.

    impl_->RmsNormToBfloat16(impl_->hidden.get(),
                             impl_->layers.front().input_norm.values.get(),
                             tokens, config.hidden_size, config.rms_norm_eps);
    for (std::size_t layer = 0; layer < impl_->layers.size(); ++layer) {
      const LayerWeights& weights = impl_->layers[layer];
      const std::array<Bfloat16GemvGroup, 3> qkv{{
          {weights.q, impl_->q.get(),
           config.num_attention_heads * config.head_dim},
          {weights.k, impl_->k.get(),
           config.num_key_value_heads * config.head_dim},
          {weights.v, impl_->v.get(),
           config.num_key_value_heads * config.head_dim},
      }};
      impl_->GemmGrouped(qkv, impl_->bfloat16_scratch.get(), tokens,
                         config.hidden_size);
      const bool cached = impl_->NormalizeRotateQkv(
          impl_->q.get(), impl_->k.get(), impl_->v.get(),
          weights.q_norm.values.get(), weights.k_norm.values.get(), tokens,
          config.num_attention_heads, config.num_key_value_heads,
          config.head_dim, 0, config.rope_theta, config.rms_norm_eps,
          impl_->key_cache.get(), impl_->value_cache.get(), layer,
          impl_->maximum_tokens);
      LaunchBfloat16Attention(impl_->q.get(), impl_->k.get(), impl_->v.get(),
                              impl_->key_cache.get(), impl_->value_cache.get(),
                              impl_->bfloat16_scratch.get(), layer, 0, tokens,
                              impl_->maximum_tokens, config.num_attention_heads,
                              config.num_key_value_heads, config.head_dim,
                              impl_->stream, cached);
      impl_->Gemm(weights.o, impl_->bfloat16_scratch.get(),
                  impl_->attention_output.get(), tokens, config.hidden_size,
                  config.num_attention_heads * config.head_dim);
      impl_->ResidualAddNormToBfloat16(impl_->attention_output.get(),
                                       weights.post_norm.values.get(), tokens,
                                       config.hidden_size, config.rms_norm_eps);
      const std::array<Bfloat16GemvGroup, 2> gate_up{{
          {weights.gate, impl_->gate.get(), config.intermediate_size},
          {weights.up, impl_->up.get(), config.intermediate_size},
      }};
      impl_->GemmGrouped(gate_up, impl_->bfloat16_scratch.get(), tokens,
                         config.hidden_size);
      LaunchBfloat16SwiGlu(impl_->gate.get(), impl_->up.get(), nullptr,
                           impl_->bfloat16_scratch.get(), ffn_elements,
                           impl_->stream);
      impl_->Gemm(weights.down, impl_->bfloat16_scratch.get(),
                  impl_->feed_forward_output.get(), tokens, config.hidden_size,
                  config.intermediate_size);
      // The next layer opens with an RMSNorm, so the trailing residual folds
      // into it; the last layer folds into the talker's final norm.
      impl_->ResidualAddNormToBfloat16(
          impl_->feed_forward_output.get(),
          layer + 1 < impl_->layers.size()
              ? impl_->layers[layer + 1].input_norm.values.get()
              : impl_->final_norm.values.get(),
          tokens, config.hidden_size, config.rms_norm_eps);

      if (layer == 0) {
        output->layer0_output.resize(hidden_elements);
        RequireHip(
            hipMemcpyAsync(output->layer0_output.data(), impl_->hidden.get(),
                           hidden_elements * sizeof(float),
                           hipMemcpyDeviceToHost, impl_->stream),
            "hipMemcpyAsync layer0 output");
      }
    }

    const auto* last_hidden_bfloat16 =
        impl_->bfloat16_scratch.get() + ((tokens - 1) * config.hidden_size);
    impl_->Gemm(impl_->codec_head, last_hidden_bfloat16, impl_->logits.get(), 1,
                config.vocab_size, config.hidden_size);
    LaunchRoundBfloat16InPlace(impl_->logits.get(), config.vocab_size,
                               impl_->stream);

    output->last_hidden.resize(config.hidden_size);
    output->logits.resize(config.vocab_size);
    LaunchBfloat16ToFloat(last_hidden_bfloat16, impl_->attention_output.get(),
                          config.hidden_size, impl_->stream);
    RequireHip(hipMemcpyAsync(output->last_hidden.data(),
                              impl_->attention_output.get(),
                              config.hidden_size * sizeof(float),
                              hipMemcpyDeviceToHost, impl_->stream),
               "hipMemcpyAsync final hidden");
    RequireHip(hipMemcpyAsync(output->logits.data(), impl_->logits.get(),
                              config.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, impl_->stream),
               "hipMemcpyAsync logits");
    RequireHip(hipStreamSynchronize(impl_->stream), "hipStreamSynchronize");
    RequireHip(hipGetLastError(), "Qwen3-TTS HIP prefill");
    impl_->talker_cache_tokens = tokens;
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool TalkerHipRuntime::BuildCustomVoicePrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
    std::string_view language, CustomVoicePromptOutput* output,
    std::string* error) {
  return BuildConditionedPrompt(input_ids, instruction_ids, speaker, language,
                                true, output, error);
}

bool TalkerHipRuntime::BuildVoiceDesignPrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view language,
    TalkerPromptOutput* output, std::string* error) {
  return BuildConditionedPrompt(input_ids, instruction_ids, {}, language, false,
                                output, error);
}

bool TalkerHipRuntime::BuildConditionedPrompt(
    std::span<const std::uint32_t> input_ids,
    std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
    std::string_view language, bool include_speaker, TalkerPromptOutput* output,
    std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config;
  const auto& talker = config.talker;
  if (input_ids.size() < 8) {
    SetError(error, "Qwen3-TTS assistant prompt is too short");
    return false;
  }
  try {
    std::string normalized_language = AsciiLower(language);
    if (normalized_language.empty()) {
      normalized_language = "auto";
    }

    std::optional<std::uint32_t> speaker_token;
    if (include_speaker) {
      const std::string normalized_speaker = AsciiLower(speaker);
      const auto speaker_id = talker.spk_id.find(normalized_speaker);
      if (speaker_id == talker.spk_id.end()) {
        throw std::invalid_argument("unsupported Qwen3-TTS speaker: " +
                                    std::string(speaker));
      }
      speaker_token = speaker_id->second;
      if (normalized_language == "chinese" || normalized_language == "auto") {
        const auto dialect = talker.spk_dialect.find(normalized_speaker);
        if (dialect != talker.spk_dialect.end() &&
            dialect->second.has_value()) {
          normalized_language = *dialect->second;
        }
      }
    }

    std::vector<std::uint32_t> codec_ids;
    if (normalized_language == "auto") {
      codec_ids = {talker.codec_nothink_id, talker.codec_think_bos_id,
                   talker.codec_think_eos_id};
    } else {
      const auto language_id =
          talker.codec_language_id.find(normalized_language);
      if (language_id == talker.codec_language_id.end()) {
        throw std::invalid_argument("unsupported Qwen3-TTS language: " +
                                    std::string(language));
      }
      codec_ids = {talker.codec_think_id, talker.codec_think_bos_id,
                   language_id->second, talker.codec_think_eos_id};
    }
    if (speaker_token.has_value()) {
      codec_ids.push_back(*speaker_token);
    }
    codec_ids.push_back(talker.codec_pad_id);
    codec_ids.push_back(talker.codec_bos_id);

    const std::array<std::uint32_t, 3> special_ids{config.tts_bos_token_id,
                                                   config.tts_eos_token_id,
                                                   config.tts_pad_token_id};
    const std::vector<float> special = impl_->ProjectText(special_ids);
    const std::size_t hidden = talker.hidden_size;
    const std::span<const float> tts_bos(special.data(), hidden);
    const std::span<const float> tts_eos(special.data() + hidden, hidden);
    const std::span<const float> tts_pad(special.data() + (2 * hidden), hidden);

    std::vector<float> instruction;
    if (!instruction_ids.empty()) {
      instruction = impl_->ProjectText(instruction_ids);
    }
    const std::vector<float> role = impl_->ProjectText(input_ids.first(3));
    const std::vector<float> codec = impl_->LookupCodec(codec_ids);
    const auto text_ids = input_ids.subspan(3, input_ids.size() - 8);
    const std::vector<float> text = impl_->ProjectText(text_ids);
    const std::array<std::uint32_t, 1> codec_pad{talker.codec_pad_id};
    const std::vector<float> codec_pad_embedding =
        impl_->LookupCodec(codec_pad);
    const NonIclPromptInput prompt_input{
        .hidden_size = hidden,
        .instruction = instruction,
        .role = role,
        .codec = codec,
        .text = text,
        .tts_bos = tts_bos,
        .tts_eos = tts_eos,
        .tts_pad = tts_pad,
        .codec_pad = codec_pad_embedding,
    };
    if (!BuildNonIclPrompt(prompt_input, output, error)) {
      return false;
    }
    if (output->tokens == 0 || output->tokens > impl_->maximum_tokens) {
      throw std::length_error("Qwen3-TTS prompt exceeds runtime capacity");
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

bool TalkerHipRuntime::BuildVoiceClonePrompt(const VoiceClonePromptInput& input,
                                             TalkerPromptOutput* output,
                                             std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS prompt output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config;
  const auto& talker = config.talker;
  if (input.input_ids.size() < 8) {
    SetError(error, "Qwen3-TTS assistant prompt is too short");
    return false;
  }
  if (input.speaker_embedding.size() != talker.hidden_size) {
    SetError(error, "Qwen3-TTS speaker embedding shape is invalid");
    return false;
  }
  if (input.icl_mode &&
      (input.reference_ids.size() < 5 || input.reference_frames == 0 ||
       input.reference_codes.size() !=
           input.reference_frames * talker.num_code_groups)) {
    SetError(error, "Qwen3-TTS reference prompt is invalid");
    return false;
  }
  try {
    std::string normalized_language = AsciiLower(input.language);
    if (normalized_language.empty()) {
      normalized_language = "auto";
    }
    std::vector<std::uint32_t> codec_ids;
    if (normalized_language == "auto") {
      codec_ids = {talker.codec_nothink_id, talker.codec_think_bos_id,
                   talker.codec_think_eos_id};
    } else {
      const auto language_id =
          talker.codec_language_id.find(normalized_language);
      if (language_id == talker.codec_language_id.end()) {
        throw std::invalid_argument("unsupported Qwen3-TTS language: " +
                                    std::string(input.language));
      }
      codec_ids = {talker.codec_think_id, talker.codec_think_bos_id,
                   language_id->second, talker.codec_think_eos_id};
    }

    const std::array<std::uint32_t, 3> special_ids{config.tts_bos_token_id,
                                                   config.tts_eos_token_id,
                                                   config.tts_pad_token_id};
    const std::vector<float> special = impl_->ProjectText(special_ids);
    const std::size_t hidden = talker.hidden_size;
    const std::span<const float> tts_bos(special.data(), hidden);
    const std::span<const float> tts_eos(special.data() + hidden, hidden);
    const std::span<const float> tts_pad(special.data() + (2 * hidden), hidden);
    const std::vector<float> role =
        impl_->ProjectText(input.input_ids.first(3));

    std::vector<float> codec = impl_->LookupCodec(codec_ids);
    codec.insert(codec.end(), input.speaker_embedding.begin(),
                 input.speaker_embedding.end());
    const std::array<std::uint32_t, 2> codec_suffix{talker.codec_pad_id,
                                                    talker.codec_bos_id};
    const std::vector<float> suffix = impl_->LookupCodec(codec_suffix);
    codec.insert(codec.end(), suffix.begin(), suffix.end());

    const auto target_ids =
        input.input_ids.subspan(3, input.input_ids.size() - 8);
    if (!input.icl_mode) {
      const std::vector<float> text = impl_->ProjectText(target_ids);
      const std::array<std::uint32_t, 1> codec_pad{talker.codec_pad_id};
      const std::vector<float> codec_pad_embedding =
          impl_->LookupCodec(codec_pad);
      const NonIclPromptInput prompt_input{
          .hidden_size = hidden,
          .role = role,
          .codec = codec,
          .text = text,
          .tts_bos = tts_bos,
          .tts_eos = tts_eos,
          .tts_pad = tts_pad,
          .codec_pad = codec_pad_embedding,
      };
      if (!BuildNonIclPrompt(prompt_input, output, error)) {
        return false;
      }
    } else {
      const auto reference_text_ids =
          input.reference_ids.subspan(3, input.reference_ids.size() - 5);
      std::vector<std::uint32_t> combined_ids(reference_text_ids.begin(),
                                              reference_text_ids.end());
      combined_ids.insert(combined_ids.end(), target_ids.begin(),
                          target_ids.end());
      const std::vector<float> combined_text = impl_->ProjectText(combined_ids);

      const std::array<std::uint32_t, 1> reference_bos{talker.codec_bos_id};
      std::vector<float> reference_codec = impl_->LookupCodec(reference_bos);
      for (std::size_t frame_index = 0; frame_index < input.reference_frames;
           ++frame_index) {
        const auto codes = input.reference_codes.subspan(
            frame_index * talker.num_code_groups, talker.num_code_groups);
        for (std::size_t group = 0; group < codes.size(); ++group) {
          const std::uint32_t vocabulary =
              group == 0 ? talker.vocab_size : config.code_predictor.vocab_size;
          if (codes[group] >= vocabulary) {
            throw std::out_of_range(
                "Qwen3-TTS reference codec token is out of range");
          }
        }
      }
      const std::vector<float> frame_embeddings =
          impl_->BuildReferenceCodecEmbeddings(input.reference_codes,
                                               input.reference_frames);
      reference_codec.insert(reference_codec.end(), frame_embeddings.begin(),
                             frame_embeddings.end());
      const IclPromptInput prompt_input{
          .hidden_size = hidden,
          .role = role,
          .codec = codec,
          .combined_text = combined_text,
          .reference_codec = reference_codec,
          .tts_bos = tts_bos,
          .tts_eos = tts_eos,
          .tts_pad = tts_pad,
      };
      if (!BuildIclPrompt(prompt_input, output, error)) {
        return false;
      }
    }
    if (output->tokens == 0 || output->tokens > impl_->maximum_tokens) {
      throw std::length_error("Qwen3-TTS prompt exceeds runtime capacity");
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

bool TalkerHipRuntime::PredictCodeFrame(std::span<const float> talker_hidden,
                                        std::uint32_t first_code,
                                        std::vector<std::uint32_t>* codes,
                                        std::string* error) {
  if (codes == nullptr) {
    SetError(error, "Qwen3-TTS codec output must not be null");
    return false;
  }
  CodePredictorOutput output;
  if (!PredictFrame(talker_hidden, first_code, &output, false, error)) {
    codes->clear();
    return false;
  }
  *codes = std::move(output.codes);
  return true;
}

bool TalkerHipRuntime::PredictCodeFrameTrace(
    std::span<const float> talker_hidden, std::uint32_t first_code,
    CodePredictorOutput* output, std::string* error,
    std::span<const std::uint32_t> reference_codes) {
  return PredictFrame(talker_hidden, first_code, output, true, error,
                      reference_codes);
}

bool TalkerHipRuntime::PredictFrame(
    std::span<const float> talker_hidden, std::uint32_t first_code,
    CodePredictorOutput* output, bool collect_logits, std::string* error,
    std::span<const std::uint32_t> reference_codes) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS predictor output must not be null");
    return false;
  }
  *output = {};
  const auto& talker = impl_->model.config.talker;
  const auto& predictor = impl_->model.config.code_predictor;
  if (talker_hidden.size() != talker.hidden_size) {
    SetError(error, "Qwen3-TTS talker hidden shape is invalid");
    return false;
  }
  if (first_code >= talker.vocab_size) {
    SetError(error, "Qwen3-TTS first codec token is out of range");
    return false;
  }
  if (!reference_codes.empty() &&
      (reference_codes.size() != talker.num_code_groups ||
       reference_codes.front() != first_code ||
       std::any_of(reference_codes.begin() + 1, reference_codes.end(),
                   [&](std::uint32_t token) {
                     return token >= predictor.vocab_size;
                   }))) {
    SetError(error, "Qwen3-TTS predictor reference history is invalid");
    return false;
  }
  try {
    // Every frame starts at position zero. Its two prefix rows and each later
    // codec row overwrite K/V before attention can see them.
    RequireHip(
        hipMemcpyAsync(impl_->attention_output.get(), talker_hidden.data(),
                       talker.hidden_size * sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream),
        "hipMemcpyAsync predictor talker hidden");
    gufo::hip::LaunchEmbeddingLookup(
        impl_->codec_embedding, core::GgmlType::kBF16, first_code,
        impl_->attention_output.get() + talker.hidden_size, talker.hidden_size,
        impl_->stream);
    impl_->ProjectPredictorInput(impl_->attention_output.get(), 2);
    if (collect_logits) {
      output->projected_input.resize(2 * predictor.hidden_size);
      RequireHip(
          hipMemcpyAsync(output->projected_input.data(), impl_->hidden.get(),
                         output->projected_input.size() * sizeof(float),
                         hipMemcpyDeviceToHost, impl_->stream),
          "hipMemcpyAsync predictor projected input");
      RequireHip(hipStreamSynchronize(impl_->stream),
                 "hipStreamSynchronize predictor input trace");
    }

    output->codes.reserve(talker.num_code_groups);
    output->codes.push_back(first_code);
    std::vector<float> head_logits;
    // Generation only needs the codes, so the per-head logit rows are copied
    // out for the trace entry point alone.
    std::vector<float>* trace = collect_logits ? &head_logits : nullptr;
    if (collect_logits) {
      output->logits.reserve((talker.num_code_groups - 1) *
                             predictor.vocab_size);
    }
    std::uint32_t code = impl_->RunPredictor(2, 0, 0, trace);
    if (collect_logits) {
      output->logits.insert(output->logits.end(), head_logits.begin(),
                            head_logits.end());
    }
    output->codes.push_back(code);
    for (std::size_t head = 1; head < impl_->predictor_heads.size(); ++head) {
      gufo::hip::LaunchEmbeddingLookup(
          impl_->predictor_embeddings[head - 1], core::GgmlType::kBF16,
          reference_codes.empty() ? code : reference_codes[head],
          impl_->attention_output.get(), talker.hidden_size, impl_->stream);
      impl_->ProjectPredictorInput(impl_->attention_output.get(), 1);
      code = impl_->RunPredictor(1, static_cast<std::uint32_t>(head + 1), head,
                                 trace);
      if (collect_logits) {
        output->logits.insert(output->logits.end(), head_logits.begin(),
                              head_logits.end());
      }
      output->codes.push_back(code);
    }
    if (output->codes.size() != talker.num_code_groups ||
        (collect_logits &&
         output->logits.size() !=
             (talker.num_code_groups - 1) * predictor.vocab_size)) {
      throw std::runtime_error(
          "Qwen3-TTS code predictor output shape mismatch");
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

bool TalkerHipRuntime::DecodeCodeFrame(std::span<const std::uint32_t> codes,
                                       std::span<const float> text_embedding,
                                       TalkerPrefillOutput* output,
                                       std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS talker output must not be null");
    return false;
  }
  *output = {};
  const auto& talker = impl_->model.config.talker;
  if (codes.size() != talker.num_code_groups ||
      text_embedding.size() != talker.hidden_size) {
    SetError(error, "Qwen3-TTS codec frame shape is invalid");
    return false;
  }
  try {
    for (std::size_t group = 0; group < codes.size(); ++group) {
      const std::uint32_t code = codes[group];
      const std::uint32_t vocabulary =
          group == 0 ? talker.vocab_size
                     : impl_->model.config.code_predictor.vocab_size;
      if (code >= vocabulary) {
        throw std::out_of_range("Qwen3-TTS codec token is out of range");
      }
    }
    impl_->PrepareCodeFrameEmbedding(codes, text_embedding);
    impl_->RunTalkerToken(output);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

bool TalkerHipRuntime::BuildCodeFrameEmbedding(
    std::span<const std::uint32_t> codes, std::span<const float> text_embedding,
    std::vector<float>* embedding, std::string* error) {
  if (embedding == nullptr) {
    SetError(error, "Qwen3-TTS frame embedding output must not be null");
    return false;
  }
  embedding->clear();
  const auto& talker = impl_->model.config.talker;
  if (codes.size() != talker.num_code_groups ||
      text_embedding.size() != talker.hidden_size) {
    SetError(error, "Qwen3-TTS codec frame shape is invalid");
    return false;
  }
  try {
    for (std::size_t group = 0; group < codes.size(); ++group) {
      const std::uint32_t vocabulary =
          group == 0 ? talker.vocab_size
                     : impl_->model.config.code_predictor.vocab_size;
      if (codes[group] >= vocabulary) {
        throw std::out_of_range("Qwen3-TTS codec token is out of range");
      }
    }
    impl_->PrepareCodeFrameEmbedding(codes, text_embedding);
    embedding->resize(talker.hidden_size);
    RequireHip(hipMemcpyAsync(embedding->data(), impl_->hidden.get(),
                              talker.hidden_size * sizeof(float),
                              hipMemcpyDeviceToHost, impl_->stream),
               "hipMemcpyAsync frame embedding");
    RequireHip(hipStreamSynchronize(impl_->stream),
               "hipStreamSynchronize frame embedding");
    RequireHip(hipGetLastError(), "Qwen3-TTS frame embedding");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    embedding->clear();
    return false;
  }
}

bool TalkerHipRuntime::GenerateGreedy(const CustomVoicePromptOutput& prompt,
                                      std::size_t maximum_new_tokens,
                                      TalkerGenerationOutput* output,
                                      std::string* error) {
  SamplingOptions sampling;
  sampling.sample = false;
  sampling.predictor_sample = false;
  return Generate(prompt, maximum_new_tokens, sampling, output, error);
}

bool TalkerHipRuntime::Generate(const CustomVoicePromptOutput& prompt,
                                std::size_t maximum_new_tokens,
                                const SamplingOptions& sampling,
                                TalkerGenerationOutput* output,
                                std::string* error) {
  return Generate(prompt, maximum_new_tokens, sampling, {}, output, error);
}

bool TalkerHipRuntime::Generate(
    const CustomVoicePromptOutput& prompt, std::size_t maximum_new_tokens,
    const SamplingOptions& sampling, const CancellationCheck& is_cancelled,
    TalkerGenerationOutput* output, std::string* error,
    const std::function<bool(const TalkerGenerationOutput&)>& on_codes) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS generation output must not be null");
    return false;
  }
  *output = {};
  const auto& config = impl_->model.config.talker;
  if (prompt.tokens == 0 ||
      prompt.embeddings.size() != prompt.tokens * config.hidden_size ||
      prompt.tts_pad.size() != config.hidden_size ||
      prompt.trailing_text.empty() ||
      prompt.trailing_text.size() % config.hidden_size != 0 ||
      maximum_new_tokens == 0) {
    SetError(error, "Qwen3-TTS generation prompt is invalid");
    return false;
  }
  if (!ValidSamplingOptions(sampling)) {
    SetError(error, "Qwen3-TTS sampling options are invalid");
    return false;
  }
  std::mt19937 random(sampling.seed);
  impl_->sampling_rng = sampling.predictor_sample ? &random : nullptr;
  impl_->predictor_top_k = sampling.predictor_top_k;
  impl_->predictor_temperature = sampling.predictor_temperature;
  impl_->predictor_top_p = sampling.predictor_top_p;
  struct PredictorSamplingReset {
    Impl* impl;
    ~PredictorSamplingReset() { impl->sampling_rng = nullptr; }
  };
  const PredictorSamplingReset reset{impl_.get()};
  try {
    const auto cancelled = [&] {
      if (!is_cancelled || !is_cancelled()) {
        return false;
      }
      SetError(error, "Qwen3-TTS native generation cancelled");
      *output = {};
      return true;
    };
    if (cancelled()) {
      return false;
    }
    TalkerPrefillOutput current;
    if (!Prefill(prompt.embeddings, prompt.tokens, &current, error)) {
      return false;
    }
    output->code_groups = config.num_code_groups;
    output->codes.reserve(maximum_new_tokens * config.num_code_groups);
    std::vector<std::uint8_t>& generated_first_codes =
        impl_->generated_first_code_flags;
    generated_first_codes.assign(config.vocab_size, 0);
    const std::size_t trailing_rows =
        prompt.trailing_text.size() / config.hidden_size;
    for (std::size_t step = 0; step < maximum_new_tokens; ++step) {
      if (cancelled()) {
        return false;
      }
      const std::uint32_t first_code = SelectMainCode(
          current.logits, generated_first_codes, step,
          config.codec_eos_token_id, sampling,
          sampling.sample ? &random : nullptr, &impl_->candidate_scratch);
      if (first_code == config.codec_eos_token_id) {
        break;
      }
      generated_first_codes[first_code] = 1;
      std::vector<std::uint32_t> frame;
      if (!PredictCodeFrame(current.last_hidden, first_code, &frame, error)) {
        return false;
      }
      output->codes.insert(output->codes.end(), frame.begin(), frame.end());
      ++output->frames;
      if (on_codes && !on_codes(*output)) {
        SetError(error, "Qwen3-TTS audio stream cancelled or failed");
        *output = {};
        return false;
      }
      if (step + 1 == maximum_new_tokens) {
        break;
      }
      const std::span<const float> text_embedding =
          step < trailing_rows
              ? std::span<const float>(
                    prompt.trailing_text.data() + (step * config.hidden_size),
                    config.hidden_size)
              : std::span<const float>(prompt.tts_pad);
      TalkerPrefillOutput next;
      if (!DecodeCodeFrame(frame, text_embedding, &next, error)) {
        return false;
      }
      current = std::move(next);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::hip

#else

namespace gufo::models::qwen3_tts::hip {

struct TalkerHipRuntime::Impl {};

TalkerHipRuntime::TalkerHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TalkerHipRuntime::~TalkerHipRuntime() = default;

std::unique_ptr<TalkerHipRuntime> TalkerHipRuntime::Create(const std::string&,
                                                           std::size_t,
                                                           std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return nullptr;
}

bool TalkerHipRuntime::Prefill(std::span<const float>, std::size_t,
                               TalkerPrefillOutput*, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::BuildCustomVoicePrompt(std::span<const std::uint32_t>,
                                              std::span<const std::uint32_t>,
                                              std::string_view,
                                              std::string_view,
                                              CustomVoicePromptOutput*,
                                              std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::BuildVoiceDesignPrompt(std::span<const std::uint32_t>,
                                              std::span<const std::uint32_t>,
                                              std::string_view,
                                              TalkerPromptOutput*,
                                              std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::BuildVoiceClonePrompt(const VoiceClonePromptInput&,
                                             TalkerPromptOutput*,
                                             std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::PredictCodeFrame(std::span<const float>, std::uint32_t,
                                        std::vector<std::uint32_t>*,
                                        std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::PredictCodeFrameTrace(std::span<const float>,
                                             std::uint32_t,
                                             CodePredictorOutput*,
                                             std::string* error,
                                             std::span<const std::uint32_t>) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::DecodeCodeFrame(std::span<const std::uint32_t>,
                                       std::span<const float>,
                                       TalkerPrefillOutput*,
                                       std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::BuildCodeFrameEmbedding(std::span<const std::uint32_t>,
                                               std::span<const float>,
                                               std::vector<float>*,
                                               std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::GenerateGreedy(const CustomVoicePromptOutput&,
                                      std::size_t, TalkerGenerationOutput*,
                                      std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::Generate(const CustomVoicePromptOutput&, std::size_t,
                                const SamplingOptions&, TalkerGenerationOutput*,
                                std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

bool TalkerHipRuntime::Generate(
    const CustomVoicePromptOutput&, std::size_t, const SamplingOptions&,
    const CancellationCheck&, TalkerGenerationOutput*, std::string* error,
    const std::function<bool(const TalkerGenerationOutput&)>&) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

}  // namespace gufo::models::qwen3_tts::hip

#endif  // defined(ENGINE_ENABLE_HIP)
