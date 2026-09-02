#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/detail/weight_regions.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {
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

  weight_regions.resize(source_regions.size());
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    const auto map_error = MapRegisteredRegion(source, destination);

    if (destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible through mapped registration: " +
                     hipGetErrorString(map_error);
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

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
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

  // opt-q4kxl: whether this shard runs its K-quants natively.
  //
  // Q6_K, Q5_K and Q8_K used to be dequantized to BF16 at load, because they
  // had no in-kernel decoder. They have had one since `opt-q4kxl`, and keeping
  // them packed is now better on all three axes:
  //
  //  - **Exactness.** The dequantized copies land on the BF16 projection route,
  //    whose batched spelling (`LaunchExactBf16GEMMFp32SmallBatch`, used by the
  //    speculative verifier) does not reproduce the dense BF16 GEMV that
  //    single-token decode uses. A drafted token is only acceptable if the
  //    verifier reproduces decode exactly, so on any shard that hit this path
  //    speculation silently stopped being greedy-faithful: UD-Q8_K_L was 0/10
  //    exact against UD-Q4_K_XL's 10/10, diverging first at the earliest layer
  //    whose `ffn_down` was converted. The packed routes are bit-exact against
  //    the decode GEMV and covered by `qwen_q4kxl_quant_ops_test` and
  //    `qwen_quant_gemv_ops_test`.
  //  - **Speed.** Q6_K is 0.82 bytes per element against BF16's two, and decode
  //    is bandwidth bound. Interleaved `tg128` on UD-Q8_K_L, six pairs:
  //    dequantized 5.65/4.44/5.89/6.29/4.33/6.28 (median 5.77), packed
  //    5.39/5.50/6.83/6.94/6.99/6.53 (median 6.68), **+15.8%**, five pairs of
  //    six.
  //  - **Footprint.** The conversion allocated a device-resident BF16 copy of
  //    every converted projection, roughly 3 GB on this shard.
  //
  // UD-Q4_K_XL never took this path -- it carries Q4_K/IQ4_XS/IQ3_S, which the
  // old heuristic already recognized as native-only -- which is exactly why it
  // was bit-exact while the Q8 shard was not.
  // GUFO_QUANT_NATIVE_KQUANT=0 restores the dequantizing behaviour.
  bool native_kquant = true;
  if (const char* forced = std::getenv("GUFO_QUANT_NATIVE_KQUANT");
      forced != nullptr) {
    const std::string_view setting{forced};
    native_kquant = setting != "0" && setting != "false" && setting != "off";
  }

  const auto pre_dequantize = [&](models::QwenTensorRef& tensor) {
    if (tensor.empty() || native_kquant) {
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

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
