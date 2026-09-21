#include "src/models/qwen3_asr/hip/text_decoder_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/models/qwen3_asr/hip/blas.hpp"
#include "src/models/qwen3_asr/hip/text_ops.hpp"
#include "src/models/qwen3_asr/loader.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {

constexpr std::uint32_t kAudioPadToken = 151676;
constexpr std::uint32_t kEndOfTextToken = 151643;
constexpr std::uint32_t kImEndToken = 151645;

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
        count_(std::exchange(other.count_, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0U);
    }
    return *this;
  }

  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      RequireHip(hipFree(data_), "hipFree Qwen3-ASR text buffer");
      data_ = nullptr;
      count_ = 0U;
    }
    if (count == 0U) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Qwen3-ASR text buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Qwen3-ASR text buffer");
    count_ = count;
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0U};
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

  [[nodiscard]] bool Initialize(MappedRegion region, hipStream_t stream) {
    host_ = region.data;
    size_ = region.size;
    payload_offset_ = region.payload_offset;
    core::PrefaultMappedRange(host_, size_);
    return TryCopy(stream) || TryMap();
  }

  [[nodiscard]] const void* Resolve(const std::byte* pointer) const noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base = reinterpret_cast<std::uintptr_t>(host_);
    if (address < base || address - base >= size_) {
      return nullptr;
    }
    return static_cast<const std::byte*>(device_) + (address - base);
  }

private:
  [[nodiscard]] bool TryMap() {
    if (hipHostRegister(const_cast<std::byte*>(host_), size_,
                        hipHostRegisterMapped | hipHostRegisterReadOnly) !=
        hipSuccess) {
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

  [[nodiscard]] bool TryCopy(hipStream_t stream) {
    constexpr std::size_t kLoadAlignment = 16U;
    const std::size_t shift =
        (kLoadAlignment - (payload_offset_ % kLoadAlignment)) % kLoadAlignment;
    if (hipMalloc(&allocation_, size_ + shift) != hipSuccess) {
      allocation_ = nullptr;
      return false;
    }
    owns_device_ = true;
    device_ = static_cast<std::byte*>(allocation_) + shift;
    if (hipMemcpyAsync(device_, host_, size_, hipMemcpyHostToDevice, stream) ==
        hipSuccess) {
      return true;
    }
    (void)hipFree(allocation_);
    allocation_ = nullptr;
    device_ = nullptr;
    owns_device_ = false;
    return false;
  }

  const std::byte* host_{nullptr};
  void* device_{nullptr};
  void* allocation_{nullptr};
  std::size_t size_{0U};
  std::size_t payload_offset_{0U};
  bool owns_device_{false};
  bool registered_{false};
};

const Tensor& RequireTensor(const TensorStore& store, std::string_view name,
                            std::span<const std::uint64_t> shape) {
  const Tensor* tensor = store.Find(name);
  if (tensor == nullptr || tensor->dtype != DType::kBF16 ||
      !std::ranges::equal(tensor->shape, shape)) {
    throw std::runtime_error("unexpected Qwen3-ASR text tensor: " +
                             std::string(name));
  }
  return *tensor;
}

const void* ResolveTensor(
    const TensorStore& store,
    const std::vector<std::unique_ptr<DeviceRegion>>& regions,
    std::string_view name, std::span<const std::uint64_t> shape) {
  const Tensor& tensor = RequireTensor(store, name, shape);
  for (const auto& region : regions) {
    if (const void* result = region->Resolve(tensor.data); result != nullptr) {
      return result;
    }
  }
  throw std::runtime_error("Qwen3-ASR tensor is outside GPU weight regions: " +
                           std::string(name));
}

struct DeviceNorm {
  DeviceBuffer<float> values;
};

DeviceNorm UploadNorm(const TensorStore& store,
                      const std::vector<std::unique_ptr<DeviceRegion>>& regions,
                      std::string_view name, std::size_t size,
                      hipStream_t stream) {
  const std::array<std::uint64_t, 1> shape{static_cast<std::uint64_t>(size)};
  const void* source = ResolveTensor(store, regions, name, shape);
  DeviceNorm result{DeviceBuffer<float>(size)};
  LaunchTextBfloat16ToFloat(source, result.values.get(), size, stream);
  return result;
}

std::string LayerName(std::size_t layer, std::string_view suffix) {
  return "thinker.model.layers." + std::to_string(layer) + "." +
         std::string(suffix);
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

bool IsStopToken(std::uint32_t token) {
  return token == kEndOfTextToken || token == kImEndToken;
}

}  // namespace

struct TextDecoderHipRuntime::Impl {
  Impl(LoadResult loaded, std::size_t token_capacity)
      : model(std::move(loaded)),
        maximum_tokens(token_capacity),
        hidden(token_capacity * model.config.text.hidden_size),
        q(token_capacity * model.config.text.num_attention_heads *
          model.config.text.head_dim),
        k(token_capacity * model.config.text.num_key_value_heads *
          model.config.text.head_dim),
        v(token_capacity * model.config.text.num_key_value_heads *
          model.config.text.head_dim),
        attention(token_capacity * model.config.text.hidden_size),
        projection(token_capacity * model.config.text.hidden_size),
        gate(token_capacity * model.config.text.intermediate_size),
        up(token_capacity * model.config.text.intermediate_size),
        feed_forward(token_capacity * model.config.text.hidden_size),
        bfloat16_scratch(token_capacity *
                         std::max(model.config.text.hidden_size,
                                  model.config.text.intermediate_size)),
        gemv_input(std::max(model.config.text.hidden_size,
                            model.config.text.intermediate_size)),
        key_cache(
            static_cast<std::size_t>(model.config.text.num_hidden_layers) *
            model.config.text.num_key_value_heads * token_capacity *
            model.config.text.head_dim),
        value_cache(
            static_cast<std::size_t>(model.config.text.num_hidden_layers) *
            model.config.text.num_key_value_heads * token_capacity *
            model.config.text.head_dim),
        logits(model.config.text.vocab_size),
        argmax_scratch(TextArgmaxScratchElements()),
        token_ids(token_capacity),
        selected_token(1U) {
    if (maximum_tokens == 0U) {
      throw std::invalid_argument(
          "Qwen3-ASR maximum text token count must be positive");
    }
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreate Qwen3-ASR text");
    RequireHipblas(hipblasCreate(&blas), "hipblasCreate Qwen3-ASR text");
    RequireHipblas(hipblasSetStream(blas, stream),
                   "hipblasSetStream Qwen3-ASR text");
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode Qwen3-ASR text");
    prefill_lt = std::make_unique<GemmLt>();
    weight_regions.reserve(model.mapped_regions.size());
    for (const MappedRegion region :
         model.RegionsFor({"thinker.model.", "thinker.lm_head."})) {
      auto device_region = std::make_unique<DeviceRegion>();
      if (!device_region->Initialize(region, stream)) {
        throw std::runtime_error(
            "cannot make Qwen3-ASR safetensors GPU-visible");
      }
      weight_regions.push_back(std::move(device_region));
    }
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
    const TextConfig& config = model.config.text;
    layers.reserve(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      LayerWeights weights;
      weights.q = ResolveTensor(
          store, weight_regions, LayerName(layer, "self_attn.q_proj.weight"),
          std::array<std::uint64_t, 2>{
              static_cast<std::uint64_t>(config.num_attention_heads) *
                  config.head_dim,
              config.hidden_size});
      weights.k = ResolveTensor(
          store, weight_regions, LayerName(layer, "self_attn.k_proj.weight"),
          std::array<std::uint64_t, 2>{
              static_cast<std::uint64_t>(config.num_key_value_heads) *
                  config.head_dim,
              config.hidden_size});
      weights.v = ResolveTensor(
          store, weight_regions, LayerName(layer, "self_attn.v_proj.weight"),
          std::array<std::uint64_t, 2>{
              static_cast<std::uint64_t>(config.num_key_value_heads) *
                  config.head_dim,
              config.hidden_size});
      weights.o = ResolveTensor(
          store, weight_regions, LayerName(layer, "self_attn.o_proj.weight"),
          std::array<std::uint64_t, 2>{
              config.hidden_size,
              static_cast<std::uint64_t>(config.num_attention_heads) *
                  config.head_dim});
      weights.gate = ResolveTensor(
          store, weight_regions, LayerName(layer, "mlp.gate_proj.weight"),
          std::array<std::uint64_t, 2>{config.intermediate_size,
                                       config.hidden_size});
      weights.up = ResolveTensor(
          store, weight_regions, LayerName(layer, "mlp.up_proj.weight"),
          std::array<std::uint64_t, 2>{config.intermediate_size,
                                       config.hidden_size});
      weights.down = ResolveTensor(
          store, weight_regions, LayerName(layer, "mlp.down_proj.weight"),
          std::array<std::uint64_t, 2>{config.hidden_size,
                                       config.intermediate_size});
      weights.input_norm = UploadNorm(
          store, weight_regions, LayerName(layer, "input_layernorm.weight"),
          config.hidden_size, stream);
      weights.q_norm = UploadNorm(store, weight_regions,
                                  LayerName(layer, "self_attn.q_norm.weight"),
                                  config.head_dim, stream);
      weights.k_norm = UploadNorm(store, weight_regions,
                                  LayerName(layer, "self_attn.k_norm.weight"),
                                  config.head_dim, stream);
      weights.post_norm =
          UploadNorm(store, weight_regions,
                     LayerName(layer, "post_attention_layernorm.weight"),
                     config.hidden_size, stream);
      layers.push_back(std::move(weights));
    }
    final_norm = UploadNorm(store, weight_regions, "thinker.model.norm.weight",
                            config.hidden_size, stream);
    embedding = ResolveTensor(
        store, weight_regions, "thinker.model.embed_tokens.weight",
        std::array<std::uint64_t, 2>{config.vocab_size, config.hidden_size});
    lm_head = ResolveTensor(
        store, weight_regions, "thinker.lm_head.weight",
        std::array<std::uint64_t, 2>{config.vocab_size, config.hidden_size});
  }

  struct GemmProjection {
    const void* weights{nullptr};
    float* output{nullptr};
    std::size_t rows{0U};
  };

  void Gemm(const void* weights, const void* inputs_bfloat16, float* output,
            std::size_t batch, std::size_t rows, std::size_t columns) {
    const GemmProjection projection{
        .weights = weights, .output = output, .rows = rows};
    GemmFused(&projection, 1U, inputs_bfloat16, batch, columns);
  }

  /// Runs projections that share one activation vector. Batch-one decode takes
  /// them as a single fused dispatch over their concatenated rows; prefill
  /// still issues one hipBLAS GEMM per tensor because it is matrix-throughput
  /// bound rather than occupancy bound.
  void GemmFused(const GemmProjection* projections, std::uint32_t count,
                 const void* inputs_bfloat16, std::size_t batch,
                 std::size_t columns) {
    if (batch == 1U && count <= kTextDecodeGemvMaxTensors) {
      std::array<TextDecodeGemvTensor, kTextDecodeGemvMaxTensors> tensors{};
      bool representable = true;
      for (std::uint32_t index = 0U; index < count; ++index) {
        if (projections[index].rows >
            std::numeric_limits<std::uint32_t>::max()) {
          representable = false;
          break;
        }
        tensors[index] = {
            .weights_bfloat16 = projections[index].weights,
            .output = projections[index].output,
            .rows = static_cast<std::uint32_t>(projections[index].rows)};
      }
      if (representable &&
          LaunchTextDecodeGemv(tensors.data(), count, inputs_bfloat16, columns,
                               stream)) {
        return;
      }
    }
    for (std::uint32_t index = 0U; index < count; ++index) {
      if (batch == 1U) {
        // Retained fallback for a geometry the fused kernel rejects.
        LaunchTextBfloat16ToFloat(inputs_bfloat16, gemv_input.get(), columns,
                                  stream);
        LaunchTextRowGemv(projections[index].weights, gemv_input.get(),
                          projections[index].output, projections[index].rows,
                          columns, stream);
        continue;
      }
      if (prefill_lt != nullptr &&
          prefill_lt->Run(projections[index].weights, inputs_bfloat16,
                          projections[index].output, batch,
                          projections[index].rows, columns, stream)) {
        continue;
      }
      LaunchGemmBf16(blas, projections[index].weights, inputs_bfloat16,
                     projections[index].output, batch, projections[index].rows,
                     columns, stream);
    }
  }

  void PreparePrompt(std::span<const std::uint32_t> prompt_ids,
                     const float* audio_embeddings,
                     std::size_t audio_embedding_elements,
                     std::size_t audio_tokens, hipMemcpyKind copy_kind) {
    const TextConfig& config = model.config.text;
    if (prompt_ids.empty() || prompt_ids.size() > maximum_tokens ||
        audio_embeddings == nullptr || audio_tokens == 0U ||
        audio_embedding_elements != audio_tokens * config.hidden_size) {
      throw std::invalid_argument("Qwen3-ASR text prompt shape is invalid");
    }
    std::size_t audio_offset = prompt_ids.size();
    std::size_t found_audio_tokens = 0U;
    for (std::size_t index = 0; index < prompt_ids.size(); ++index) {
      if (prompt_ids[index] >= config.vocab_size) {
        throw std::out_of_range("Qwen3-ASR prompt token is out of range");
      }
      if (prompt_ids[index] == kAudioPadToken) {
        if (found_audio_tokens == 0U) {
          audio_offset = index;
        } else if (index != audio_offset + found_audio_tokens) {
          throw std::invalid_argument(
              "Qwen3-ASR audio prompt tokens must be contiguous");
        }
        ++found_audio_tokens;
      }
    }
    if (found_audio_tokens != audio_tokens) {
      throw std::invalid_argument(
          "Qwen3-ASR prompt/audio embedding count differs");
    }

    RequireHip(hipMemcpyAsync(token_ids.get(), prompt_ids.data(),
                              prompt_ids.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-ASR prompt ids");
    LaunchTextEmbeddingLookup(embedding, token_ids.get(), hidden.get(),
                              prompt_ids.size(), config.hidden_size, stream);
    float* audio_destination =
        hidden.get() + (audio_offset * config.hidden_size);
    RequireHip(hipMemcpyAsync(audio_destination, audio_embeddings,
                              audio_embedding_elements * sizeof(float),
                              copy_kind, stream),
               "hipMemcpyAsync Qwen3-ASR audio embeddings");
    LaunchTextRoundBfloat16(audio_destination, audio_embedding_elements,
                            stream);
  }

  void PrepareToken(std::uint32_t token) {
    const TextConfig& config = model.config.text;
    if (token >= config.vocab_size) {
      throw std::out_of_range("Qwen3-ASR generated token is out of range");
    }
    RequireHip(hipMemcpyAsync(token_ids.get(), &token, sizeof(token),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-ASR generated token");
    LaunchTextEmbeddingLookup(embedding, token_ids.get(), hidden.get(), 1U,
                              config.hidden_size, stream);
  }

  void RunTokens(std::size_t tokens, std::uint32_t start_position,
                 std::vector<float>* layer0_trace) {
    const TextConfig& config = model.config.text;
    if (tokens == 0U ||
        static_cast<std::size_t>(start_position) + tokens > maximum_tokens) {
      throw std::length_error("Qwen3-ASR text step exceeds KV capacity");
    }
    const std::size_t hidden_elements = tokens * config.hidden_size;
    const std::size_t ffn_elements = tokens * config.intermediate_size;
    LaunchTextRMSNorm(hidden.get(), layers.front().input_norm.values.get(),
                      bfloat16_scratch.get(), tokens, config.hidden_size,
                      config.rms_norm_eps, stream);

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
      if (cancellation && cancellation()) {
        throw std::runtime_error("Qwen3-ASR generation cancelled");
      }
      const LayerWeights& weights = layers[layer];
      // q/k/v read the same normalized activations, so decode issues them as
      // one dispatch over 4096 concatenated rows instead of three grids of
      // 2048/1024/1024 that each leave the machine under half occupied.
      const std::array<GemmProjection, 3> qkv{
          GemmProjection{.weights = weights.q,
                         .output = q.get(),
                         .rows = config.num_attention_heads * config.head_dim},
          GemmProjection{.weights = weights.k,
                         .output = k.get(),
                         .rows = config.num_key_value_heads * config.head_dim},
          GemmProjection{.weights = weights.v,
                         .output = v.get(),
                         .rows = config.num_key_value_heads * config.head_dim}};
      GemmFused(qkv.data(), 3U, bfloat16_scratch.get(), tokens,
                config.hidden_size);
      if (!LaunchTextQkNormRoPE(
              q.get(), k.get(), v.get(), weights.q_norm.values.get(),
              weights.k_norm.values.get(), tokens, config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, start_position,
              config.rope_theta, config.rms_norm_eps, key_cache.get(),
              value_cache.get(), static_cast<std::uint32_t>(layer),
              static_cast<std::uint32_t>(maximum_tokens), stream)) {
        throw std::runtime_error(
            "Qwen3-ASR text q/k normalization shape is unsupported");
      }
      if (tokens != 1U ||
          !LaunchTextDecodeAttention(
              q.get(), key_cache.get(), value_cache.get(), attention.get(),
              bfloat16_scratch.get(), static_cast<std::uint32_t>(layer),
              start_position, static_cast<std::uint32_t>(maximum_tokens),
              config.num_attention_heads, config.num_key_value_heads,
              config.head_dim, stream)) {
        LaunchTextBatchedAttention(
            q.get(), key_cache.get(), value_cache.get(), attention.get(),
            bfloat16_scratch.get(), static_cast<std::uint32_t>(layer),
            start_position, tokens, static_cast<std::uint32_t>(maximum_tokens),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, stream);
      }
      Gemm(weights.o, bfloat16_scratch.get(), projection.get(), tokens,
           config.hidden_size, config.num_attention_heads * config.head_dim);
      LaunchTextResidualAddRMSNorm(
          hidden.get(), projection.get(), weights.post_norm.values.get(),
          bfloat16_scratch.get(), tokens, config.hidden_size,
          config.rms_norm_eps, stream);
      // gate and up likewise share their input.
      const std::array<GemmProjection, 2> gate_up{
          GemmProjection{.weights = weights.gate,
                         .output = gate.get(),
                         .rows = config.intermediate_size},
          GemmProjection{.weights = weights.up,
                         .output = up.get(),
                         .rows = config.intermediate_size}};
      GemmFused(gate_up.data(), 2U, bfloat16_scratch.get(), tokens,
                config.hidden_size);
      LaunchTextSwiGLU(gate.get(), up.get(), bfloat16_scratch.get(),
                       ffn_elements, stream);
      Gemm(weights.down, bfloat16_scratch.get(), feed_forward.get(), tokens,
           config.hidden_size, config.intermediate_size);
      LaunchTextResidualAddRMSNorm(
          hidden.get(), feed_forward.get(),
          layer + 1U < layers.size()
              ? layers[layer + 1U].input_norm.values.get()
              : final_norm.values.get(),
          bfloat16_scratch.get(), tokens, config.hidden_size,
          config.rms_norm_eps, stream);

      if (layer == 0U && layer0_trace != nullptr) {
        layer0_trace->resize(hidden_elements);
        RequireHip(hipMemcpyAsync(layer0_trace->data(), hidden.get(),
                                  hidden_elements * sizeof(float),
                                  hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync Qwen3-ASR text layer 0");
      }
    }

    const auto* last_hidden =
        bfloat16_scratch.get() + ((tokens - 1U) * config.hidden_size);
    Gemm(lm_head, last_hidden, logits.get(), 1U, config.vocab_size,
         config.hidden_size);
    LaunchTextRoundBfloat16(logits.get(), config.vocab_size, stream);
  }

  void Prefill(std::span<const std::uint32_t> prompt_ids,
               std::span<const float> audio_embeddings,
               std::size_t audio_tokens, TextDecoderTrace* trace) {
    PreparePrompt(prompt_ids, audio_embeddings.data(), audio_embeddings.size(),
                  audio_tokens, hipMemcpyHostToDevice);
    RunTokens(prompt_ids.size(), 0U,
              trace != nullptr ? &trace->layer0 : nullptr);
    cache_tokens = prompt_ids.size();
    if (trace != nullptr) {
      trace->logits.resize(model.config.text.vocab_size);
      RequireHip(hipMemcpyAsync(trace->logits.data(), logits.get(),
                                trace->logits.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream),
                 "hipMemcpyAsync Qwen3-ASR prefill logits");
      RequireHip(hipStreamSynchronize(stream),
                 "hipStreamSynchronize Qwen3-ASR prefill");
      RequireHip(hipGetLastError(), "Qwen3-ASR text prefill");
    }
  }

  void PrefillDevice(std::span<const std::uint32_t> prompt_ids,
                     const float* audio_embeddings, std::size_t audio_tokens) {
    PreparePrompt(prompt_ids, audio_embeddings,
                  audio_tokens * model.config.text.hidden_size, audio_tokens,
                  hipMemcpyDeviceToDevice);
    RunTokens(prompt_ids.size(), 0U, nullptr);
    cache_tokens = prompt_ids.size();
  }

  std::uint32_t SelectToken() {
    LaunchTextArgmax(logits.get(), selected_token.get(),
                     model.config.text.vocab_size, argmax_scratch.get(),
                     stream);
    std::uint32_t result = 0U;
    RequireHip(hipMemcpyAsync(&result, selected_token.get(), sizeof(result),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync Qwen3-ASR selected token");
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Qwen3-ASR selected token");
    RequireHip(hipGetLastError(), "Qwen3-ASR text generation");
    return result;
  }

  void Decode(std::uint32_t input_token) {
    if (cache_tokens >= maximum_tokens) {
      throw std::length_error("Qwen3-ASR generation exceeds KV capacity");
    }
    PrepareToken(input_token);
    RunTokens(1U, static_cast<std::uint32_t>(cache_tokens), nullptr);
    ++cache_tokens;
  }

  LoadResult model;
  std::size_t maximum_tokens;
  CancellationCheck cancellation;
  std::size_t cache_tokens{0U};
  std::vector<std::unique_ptr<DeviceRegion>> weight_regions;
  std::vector<LayerWeights> layers;
  DeviceNorm final_norm;
  const void* embedding{nullptr};
  const void* lm_head{nullptr};
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<GemmLt> prefill_lt;
  DeviceBuffer<float> hidden;
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> attention;
  DeviceBuffer<float> projection;
  DeviceBuffer<float> gate;
  DeviceBuffer<float> up;
  DeviceBuffer<float> feed_forward;
  DeviceBuffer<hip_bfloat16> bfloat16_scratch;
  DeviceBuffer<float> gemv_input;
  DeviceBuffer<float> key_cache;
  DeviceBuffer<float> value_cache;
  DeviceBuffer<float> logits;
  DeviceBuffer<float> argmax_scratch;
  DeviceBuffer<std::uint32_t> token_ids;
  DeviceBuffer<std::uint32_t> selected_token;
};

TextDecoderHipRuntime::TextDecoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TextDecoderHipRuntime::~TextDecoderHipRuntime() = default;

std::unique_ptr<TextDecoderHipRuntime> TextDecoderHipRuntime::Create(
    const std::string& model_root, std::size_t maximum_tokens,
    std::string* error) {
  try {
    LoadResult loaded = LoadModelDirectory(model_root);
    if (!loaded.ok) {
      SetError(error, std::move(loaded.error));
      return nullptr;
    }
    return std::unique_ptr<TextDecoderHipRuntime>(new TextDecoderHipRuntime(
        std::make_unique<Impl>(std::move(loaded), maximum_tokens)));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool TextDecoderHipRuntime::Prefill(std::span<const std::uint32_t> prompt_ids,
                                    std::span<const float> audio_embeddings,
                                    std::size_t audio_tokens,
                                    TextDecoderTrace* output,
                                    std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR text trace output must not be null");
    return false;
  }
  *output = {};
  try {
    impl_->Prefill(prompt_ids, audio_embeddings, audio_tokens, output);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool TextDecoderHipRuntime::Generate(std::span<const std::uint32_t> prompt_ids,
                                     std::span<const float> audio_embeddings,
                                     std::size_t audio_tokens,
                                     std::size_t maximum_new_tokens,
                                     std::vector<std::uint32_t>* generated_ids,
                                     std::string* error) {
  if (generated_ids == nullptr) {
    SetError(error, "Qwen3-ASR generated-token output must not be null");
    return false;
  }
  generated_ids->clear();
  if (maximum_new_tokens == 0U) {
    return true;
  }
  try {
    if (prompt_ids.size() + maximum_new_tokens > impl_->maximum_tokens) {
      throw std::length_error(
          "Qwen3-ASR requested generation exceeds KV capacity");
    }
    impl_->Prefill(prompt_ids, audio_embeddings, audio_tokens, nullptr);
    std::uint32_t token = impl_->SelectToken();
    if (token >= impl_->model.config.text.vocab_size) {
      throw std::runtime_error(
          "Qwen3-ASR decoder produced non-finite prefill logits");
    }
    generated_ids->push_back(token);
    while (!IsStopToken(token) && generated_ids->size() < maximum_new_tokens) {
      impl_->Decode(token);
      token = impl_->SelectToken();
      if (token >= impl_->model.config.text.vocab_size) {
        throw std::runtime_error(
            "Qwen3-ASR decoder produced non-finite logits at generation "
            "step " +
            std::to_string(generated_ids->size()));
      }
      generated_ids->push_back(token);
    }
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    generated_ids->clear();
    SetError(error, exception.what());
    return false;
  }
}

bool TextDecoderHipRuntime::GenerateDevice(
    std::span<const std::uint32_t> prompt_ids,
    const float* audio_embeddings_device, std::size_t audio_tokens,
    std::size_t maximum_new_tokens, std::vector<std::uint32_t>* generated_ids,
    std::string* error,
    const std::function<bool(std::span<const std::uint32_t>)>& on_tokens,
    const CancellationCheck& is_cancelled) {
  if (generated_ids == nullptr) {
    SetError(error, "Qwen3-ASR generated-token output must not be null");
    return false;
  }
  generated_ids->clear();
  if (maximum_new_tokens == 0U) {
    return true;
  }
  impl_->cancellation = is_cancelled;
  struct ClearCancellation {
    Impl* impl;
    ~ClearCancellation() { impl->cancellation = {}; }
  } clear{impl_.get()};
  try {
    if (prompt_ids.size() > impl_->maximum_tokens ||
        maximum_new_tokens > impl_->maximum_tokens - prompt_ids.size()) {
      throw std::length_error(
          "Qwen3-ASR requested generation exceeds KV capacity");
    }
    impl_->PrefillDevice(prompt_ids, audio_embeddings_device, audio_tokens);
    std::uint32_t token = impl_->SelectToken();
    if (token >= impl_->model.config.text.vocab_size) {
      throw std::runtime_error(
          "Qwen3-ASR decoder produced non-finite prefill logits");
    }
    generated_ids->push_back(token);
    if (on_tokens && !on_tokens(*generated_ids)) {
      throw std::runtime_error("Qwen3-ASR generation cancelled");
    }
    while (!IsStopToken(token) && generated_ids->size() < maximum_new_tokens) {
      impl_->Decode(token);
      token = impl_->SelectToken();
      if (token >= impl_->model.config.text.vocab_size) {
        throw std::runtime_error(
            "Qwen3-ASR decoder produced non-finite logits at generation "
            "step " +
            std::to_string(generated_ids->size()));
      }
      generated_ids->push_back(token);
      if (on_tokens && !on_tokens(*generated_ids)) {
        throw std::runtime_error("Qwen3-ASR generation cancelled");
      }
    }
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    generated_ids->clear();
    SetError(error, exception.what());
    return false;
  }
}

}  // namespace gufo::models::qwen3_asr::hip

#else

namespace gufo::models::qwen3_asr::hip {

TextDecoderHipRuntime::~TextDecoderHipRuntime() = default;

std::unique_ptr<TextDecoderHipRuntime> TextDecoderHipRuntime::Create(
    const std::string&, std::size_t, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR text decoder requires ENGINE_ENABLE_HIP";
  }
  return nullptr;
}

bool TextDecoderHipRuntime::Prefill(std::span<const std::uint32_t>,
                                    std::span<const float>, std::size_t,
                                    TextDecoderTrace*, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR text decoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

bool TextDecoderHipRuntime::Generate(std::span<const std::uint32_t>,
                                     std::span<const float>, std::size_t,
                                     std::size_t, std::vector<std::uint32_t>*,
                                     std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR text decoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

bool TextDecoderHipRuntime::GenerateDevice(
    std::span<const std::uint32_t>, const float*, std::size_t, std::size_t,
    std::vector<std::uint32_t>*, std::string* error,
    const std::function<bool(std::span<const std::uint32_t>)>&,
    const CancellationCheck&) {
  if (error != nullptr) {
    *error = "Qwen3-ASR text decoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
