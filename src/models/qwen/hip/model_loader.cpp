#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

#include "src/models/qwen/hip/detail/weight_regions.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace strix::hip {
namespace detail {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_memory && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    }
    if (region.host_registered && region.host_data != nullptr) {
      (void)hipHostUnregister(const_cast<void*>(region.host_data));
    }
    region.device_data = nullptr;
    region.owns_device_memory = false;
    region.host_registered = false;
  }
}

}  // namespace detail

using detail::ReleaseWeightRegions;

namespace {

enum class WeightMappingMode {
  kAuto,
  kMapped,
  kCopy,
};

[[nodiscard]] WeightMappingMode GetWeightMappingMode() noexcept {
  const char* value = std::getenv("STRIX_GPU_WEIGHT_MODE");
  if (value == nullptr) {
    return WeightMappingMode::kAuto;
  }
  const std::string_view mode{value};
  if (mode == "mapped") {
    return WeightMappingMode::kMapped;
  }
  if (mode == "copy") {
    return WeightMappingMode::kCopy;
  }
  return WeightMappingMode::kAuto;
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source,
    QwenGpuWeightRegion& destination) noexcept {
  const auto register_error =
      hipHostRegister(const_cast<void*>(source.data), source.size,
                      hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess) {
    return register_error;
  }

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, const_cast<void*>(source.data), 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(const_cast<void*>(source.data));
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .size = source.size,
                 .owns_device_memory = false,
                 .host_registered = true};
  return hipSuccess;
}

[[nodiscard]] hipError_t CopyRegionToDevice(
    const core::GgufMappedRegion& source,
    QwenGpuWeightRegion& destination) noexcept {
  void* device_data = nullptr;
  const auto malloc_error = hipMalloc(&device_data, source.size);
  if (malloc_error != hipSuccess) {
    return malloc_error;
  }
  const auto copy_error =
      hipMemcpy(device_data, source.data, source.size, hipMemcpyHostToDevice);
  if (copy_error != hipSuccess) {
    (void)hipFree(device_data);
    return copy_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .size = source.size,
                 .owns_device_memory = true,
                 .host_registered = false};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  hipDeviceProp_t properties{};
  const bool integrated =
      hipGetDeviceProperties(&properties, 0) == hipSuccess &&
      properties.integrated != 0;
  const auto mode = GetWeightMappingMode();
  const bool prefer_mapped = mode == WeightMappingMode::kMapped ||
                             (mode == WeightMappingMode::kAuto && integrated);

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t first_error = hipSuccess;
    hipError_t second_error = hipSuccess;

    if (prefer_mapped) {
      first_error = MapRegisteredRegion(source, destination);
      if (first_error != hipSuccess && mode == WeightMappingMode::kAuto) {
        second_error = CopyRegionToDevice(source, destination);
      }
    } else {
      first_error = CopyRegionToDevice(source, destination);
      if (first_error != hipSuccess && mode == WeightMappingMode::kAuto) {
        second_error = MapRegisteredRegion(source, destination);
      }
    }

    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        const auto final_error =
            second_error != hipSuccess ? second_error : first_error;
        *error_msg =
            "Failed to make GGUF shard " + std::to_string(i) +
            " GPU-visible in " +
            (prefer_mapped ? std::string("mapped") : std::string("copy")) +
            " mode: " + hipGetErrorString(final_error);
      }
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)) {}

QwenGpuModel::~QwenGpuModel() {
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer || tokenizer->GetVocabSize() <= 256) {
    auto bin_tok = tokenization::QwenTokenizer::CreateFromBinaryFile(
        "models/qwen_vocab.bin");
    if (bin_tok) {
      tokenizer = std::move(bin_tok);
    } else if (!tokenizer) {
      return nullptr;
    }
  }

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = remap(weights_opt->token_embd) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  const auto pre_dequantize = [&](models::QwenTensorRef& tensor) {
    if (tensor.empty()) {
      return true;
    }
    if (tensor.type == core::GgmlType::kQ6_K ||
        tensor.type == core::GgmlType::kQ5_K ||
        tensor.type == core::GgmlType::kQ8_K) {
      void* d_bf16 = nullptr;
      const std::size_t n_bytes = tensor.num_elements * sizeof(hip_bfloat16);
      if (hipMalloc(&d_bf16, n_bytes) != hipSuccess) {
        return false;
      }
      LaunchDequantizeToBf16(tensor.type, tensor.data,
                             static_cast<hip_bfloat16*>(d_bf16),
                             tensor.num_elements, nullptr);
      weight_regions.push_back({
          .host_data = nullptr,
          .device_data = d_bf16,
          .size = n_bytes,
          .owns_device_memory = true,
          .host_registered = false,
      });
      tensor.data = d_bf16;
      tensor.type = core::GgmlType::kBF16;
      tensor.available_bytes = n_bytes;
    }
    return true;
  };

  bool dequant_ok = pre_dequantize(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    dequant_ok =
        dequant_ok && pre_dequantize(layer.attn_q) &&
        pre_dequantize(layer.attn_k) && pre_dequantize(layer.attn_v) &&
        pre_dequantize(layer.attn_output) && pre_dequantize(layer.attn_qkv) &&
        pre_dequantize(layer.attn_gate) && pre_dequantize(layer.ssm_out) &&
        pre_dequantize(layer.ssm_alpha) && pre_dequantize(layer.ssm_beta) &&
        pre_dequantize(layer.ffn_gate) && pre_dequantize(layer.ffn_up) &&
        pre_dequantize(layer.ffn_down);
  }
  if (!dequant_ok) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg =
          "Failed to pre-dequantize non-Q8_0 weight tensors to GPU BF16";
    }
    return nullptr;
  }
  (void)hipDeviceSynchronize();

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions));
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
