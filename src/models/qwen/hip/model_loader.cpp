#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <atomic>
#include <cstring>
#include <gufo/ext/experimental/heap_placement.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/platform/pages.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.owns_device_allocation && region.device_data != nullptr) {
      (void)hipFree(region.device_data);
    } else if (region.host_copy != nullptr) {
      (void)hipHostUnregister(region.host_copy);
      platform::ReleaseCommittedPages(region.host_copy, region.size);
    }
    region = {};
  }
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    bool& source_was_read) {
  source_was_read = false;
  void* host_copy =
      platform::CloneMappedWeightsToAnonymousPages(source.data, source.size);
  if (host_copy == nullptr)
    return hipErrorOutOfMemory;
  source_was_read = true;
  const auto unmap = [&](void* pointer) {
    platform::ReleaseCommittedPages(pointer, source.size);
  };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  const auto register_error = hipHostRegister(
      host_copy, source.size, hipHostRegisterMapped | hipHostRegisterReadOnly);
  if (register_error != hipSuccess)
    return register_error;

  void* device_data = nullptr;
  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

/// Device-resident placement: one hipMalloc copy in (on this APU target)
/// carve-out memory, filled from the cached mapping or by FillDeviceRegions.
[[nodiscard]] hipError_t AllocateDeviceRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination) {
  void* device_data = nullptr;
  const auto status = hipMalloc(&device_data, source.size);
  if (status != hipSuccess)
    return status;
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size,
                 .owns_device_allocation = true};
  return hipSuccess;
}

/// Streams one region from the read-only mapping into device memory. Where
/// the platform can drop the mapping's pages as it goes, eight workers each
/// copy 64 MiB chunks through their own pinned staging buffer, so a shard in
/// the page cache uploads at memory speed. A hipMemcpy straight from the
/// pageable mapping pins it page by page and runs at a small fraction of
/// that. Elsewhere one staging buffer streams the prefaulted mapping; that is
/// the fallback for sources WeightUpload cannot read, such as in-memory images
/// without a file.
[[nodiscard]] hipError_t CopyMappedRegion(const core::GgufMappedRegion& source,
                                          void* device_data) {
  const auto free_staging = [](void* pointer) { (void)hipHostFree(pointer); };
  using Staging = std::unique_ptr<void, decltype(free_staging)>;
  if (platform::CanVisitMappedChunks()) {
    constexpr std::size_t kChunkBytes = 64ULL << 20;
    constexpr std::size_t kWorkers = 8;
    std::vector<Staging> staging;
    staging.reserve(kWorkers);
    for (std::size_t i = 0; i < kWorkers; ++i)
      staging.emplace_back(nullptr, free_staging);
    std::atomic<hipError_t> failure{hipSuccess};
    platform::VisitMappedChunks(
        source.data, source.size, kChunkBytes, kWorkers,
        [&](std::size_t worker, std::size_t offset, const void* chunk,
            std::size_t size) {
          auto& buffer = staging[worker];
          if (!buffer) {
            void* pointer = nullptr;
            if (const auto status = hipHostMalloc(&pointer, kChunkBytes);
                status != hipSuccess) {
              failure.store(status, std::memory_order_relaxed);
              return false;
            }
            buffer.reset(pointer);
          }
          std::memcpy(buffer.get(), chunk, size);
          const auto status =
              hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                        buffer.get(), size, hipMemcpyHostToDevice);
          if (status != hipSuccess) {
            failure.store(status, std::memory_order_relaxed);
            return false;
          }
          return true;
        });
    return failure.load();
  }

  platform::PrefaultMappedRange(source.data, source.size);

  constexpr std::size_t kChunkBytes = 16ULL << 20;
  void* staging = nullptr;
  hipError_t status = hipHostMalloc(&staging, kChunkBytes);
  if (status != hipSuccess)
    return status;
  Staging owned_staging(staging, free_staging);

  const auto chunks =
      source.size / kChunkBytes + (source.size % kChunkBytes != 0);
  for (std::size_t index = 0; index < chunks; ++index) {
    const auto offset = index * kChunkBytes;
    const auto bytes = std::min(kChunkBytes, source.size - offset);
    std::memcpy(staging, static_cast<const std::uint8_t*>(source.data) + offset,
                bytes);
    status = hipMemcpy(static_cast<std::uint8_t*>(device_data) + offset,
                       staging, bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess)
      return status;
  }
  return hipSuccess;
}

/// Fills device-resident regions from their shard files without touching the
/// mappings. Copying through a mapping faults the whole shard into host memory
/// first; on Windows those pages also stay in the process working set, so a
/// 16 GiB target needs as much free RAM beside the GPU carve-out. WeightUpload
/// keeps host residency at its staging buffers.
[[nodiscard]] bool FillDeviceRegions(
    std::span<const core::GgufMappedRegion> sources,
    std::span<QwenGpuWeightRegion> destinations,
    std::span<const std::size_t> indices, std::string* error_msg) {
  std::string upload_error;
  if (auto uploader = WeightUpload::Create(sources, &upload_error)) {
    for (const auto index : indices) {
      if (!uploader->Copy(static_cast<std::uint32_t>(index), 0,
                          sources[index].size, destinations[index].device_data,
                          error_msg))
        break;
    }
    // Finish waits for every queued copy, also after a failed Copy, so the
    // destinations stay valid until the uploader is gone.
    return uploader->Finish(error_msg);
  }
  for (const auto index : indices) {
    const auto status =
        CopyMappedRegion(sources[index], destinations[index].device_data);
    if (status != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to upload GGUF shard " + std::to_string(index) +
                     ": " + hipGetErrorString(status);
      }
      return false;
    }
  }
  return true;
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
  std::vector<std::size_t> device_regions;
  for (std::size_t i = 0; i < source_regions.size(); ++i) {
    const auto& source = source_regions[i];
    auto& destination = weight_regions[i];
    hipError_t map_error;
    try {
      // Device memory is not on the kernel's page LRU, so compaction never
      // migrates it. A registered anonymous copy is: every migration makes
      // the driver evict and restore the GPU mapping, and kernels reading it
      // stall for seconds.
      // Read through the page cache where the platform allows it, so a
      // warm reload does not wait on the disk; otherwise (Windows) the
      // bounded file pipeline fills the region without touching the mapping.
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess &&
          HeapFitsAboveReserve(free_bytes, source.size) &&
          AllocateDeviceRegion(source, destination) == hipSuccess) {
        if (platform::CanVisitMappedChunks())
          map_error = CopyMappedRegion(source, destination.device_data);
        else {
          map_error = hipSuccess;
          device_regions.push_back(i);
        }
      } else {
        (void)hipGetLastError();
        destination = {};
        bool source_was_read = false;
        map_error = MapRegisteredRegion(source, destination, source_was_read);
        if (map_error != hipSuccess) {
          // The device-resident upload preserves single-copy residency
          // wherever registration is refused.
          destination = {};
          map_error = AllocateDeviceRegion(source, destination);
          if (map_error == hipSuccess) {
            if (source_was_read) {
              // A successful clone has already read the mapping. Reuse those
              // cached bytes if registration fails instead of rereading the
              // shard with direct I/O. When cloning is unavailable (Windows),
              // keep the bounded file pipeline without touching the mapping.
              map_error = CopyMappedRegion(source, destination.device_data);
            } else {
              device_regions.push_back(i);
            }
          }
        }
      }
    } catch (const std::exception& e) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg)
        *error_msg = e.what();
      return false;
    }
    if (map_error != hipSuccess || destination.device_data == nullptr) {
      ReleaseWeightRegions(weight_regions);
      if (error_msg != nullptr) {
        *error_msg = "Failed to make GGUF shard " + std::to_string(i) +
                     " GPU-visible: " + hipGetErrorString(map_error);
      }
      return false;
    }
  }
  if (!device_regions.empty() &&
      !FillDeviceRegions(source_regions, weight_regions, device_regions,
                         error_msg)) {
    ReleaseWeightRegions(weight_regions);
    return false;
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
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
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
  if (!tokenizer)
    return nullptr;

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

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  return std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision));
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
