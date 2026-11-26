#include "src/models/minimax_h3/runtime.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <gufo/ext/safetensors.hpp>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <utility>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace gufo::minimax_h3 {

namespace {

constexpr std::string_view kManifestSchema = "gufo.minimax-h3-source.v1";
constexpr std::string_view kModelKind = "minimax-h3-fl2va-bf16";
constexpr std::string_view kRevision =
    "42ed227ee7df40d41602854ae760620d6eb651fe";
constexpr std::string_view kSourceManifestSha256 =
    "00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89";
constexpr std::size_t kMaximumJsonBytes = 64U << 20U;
constexpr std::size_t kDeviceAlignment = 256;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

const json::Value& RequireField(const json::Value& object,
                                std::string_view key) {
  const json::Value* value = object.Find(key);
  if (value == nullptr) {
    throw json::Error("missing required field " + std::string(key));
  }
  return *value;
}

std::filesystem::path SafeRelativePath(std::string_view value) {
  const std::filesystem::path path(value);
  if (value.empty() || path.is_absolute() ||
      path.lexically_normal().generic_string() != value) {
    throw json::Error("unsafe relative path " + std::string(value));
  }
  for (const auto& part : path) {
    if (part == ".." || part == ".") {
      throw json::Error("unsafe relative path " + std::string(value));
    }
  }
  return path;
}

DType ParseDType(std::string_view value) {
  if (value == "BF16") {
    return DType::kBFloat16;
  }
  if (value == "F32") {
    return DType::kFloat32;
  }
  if (value == "I32") {
    return DType::kInt32;
  }
  throw json::Error("unsupported H3 tensor dtype " + std::string(value));
}

Phase TensorPhase(std::string_view component, std::string_view name) {
  if (component == "text_encoder") {
    return Phase::kPromptEncoder;
  }
  if (component == "video_vae") {
    return Phase::kVisualVae;
  }
  if (component == "audio_vae") {
    return Phase::kAudioVae;
  }
  if (component != "transformer") {
    throw json::Error("unsupported H3 component " + std::string(component));
  }
  if (name.find("adaln_proj") != std::string_view::npos ||
      name.starts_with("time_embedder.")) {
    return Phase::kAdaLnPrecompute;
  }
  return Phase::kDitCore;
}

bool ReadExact(platform::PlatformFile descriptor, void* output,
               std::size_t bytes, std::uint64_t offset) {
  auto* destination = static_cast<std::byte*>(output);
  std::size_t completed = 0;
  while (completed < bytes) {
    const std::size_t result =
        platform::ReadFileAt(descriptor, destination + completed,
                             bytes - completed, offset + completed);
    if (result == 0) {
      return false;
    }
    completed += result;
  }
  return true;
}

struct HeaderTensor {
  DType dtype{DType::kBFloat16};
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct ParsedHeader {
  std::uint64_t payload_offset{0};
  std::uint64_t file_bytes{0};
  std::map<std::string, HeaderTensor, std::less<>> tensors;
};

ParsedHeader ParseSafetensorsHeader(const std::filesystem::path& path,
                                    InspectionTelemetry* telemetry) {
  std::string open_error;
  const platform::PlatformFile descriptor =
      platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(descriptor)) {
    throw json::Error(open_error.empty() ? "cannot open " + path.string()
                                         : open_error);
  }
  // Closes the file on every way out, exceptions included.
  struct Closer {
    platform::PlatformFile file;
    ~Closer() { platform::CloseFile(file); }
  } const closer{descriptor};
  const auto file_size = platform::FileSizeBytes(descriptor);
  if (!file_size.has_value()) {
    throw json::Error("cannot stat " + path.string());
  }
  // A file shorter than the length gives a shorter prefix, which the reader
  // rejects.
  std::array<std::byte, safetensors::kLengthBytes> encoded_length{};
  const auto prefix =
      std::span(encoded_length)
          .first(static_cast<std::size_t>(
              std::min<std::uint64_t>(*file_size, encoded_length.size())));
  if (!ReadExact(descriptor, prefix.data(), prefix.size(), 0)) {
    throw json::Error("cannot read safetensors length " + path.string());
  }
  std::uint64_t header_bytes = 0;
  try {
    header_bytes = safetensors::HeaderLength(prefix, *file_size);
  } catch (const safetensors::Error& error) {
    throw json::Error(path.string() + ": " + error.what());
  }
  std::string header(static_cast<std::size_t>(header_bytes), '\0');
  if (!ReadExact(descriptor, header.data(), header.size(), 8)) {
    throw json::Error("cannot read safetensors header " + path.string());
  }
  telemetry->safetensors_header_bytes_read += 8U + header_bytes;

  safetensors::Header checked;
  try {
    checked =
        safetensors::ParseHeader(std::as_bytes(std::span(header)), *file_size);
  } catch (const safetensors::Error& error) {
    throw json::Error(path.string() + ": " + error.what());
  }
  ParsedHeader parsed;
  parsed.payload_offset = checked.payload_offset;
  parsed.file_bytes = *file_size;
  for (auto& entry : checked.tensors) {
    const DType dtype = ParseDType(safetensors::Name(entry.dtype));
    if (entry.begin == entry.end) {
      throw json::Error(path.string() + ": " + entry.name +
                        ": tensor has no elements");
    }
    // The reader rejects duplicate names, so every emplace inserts.
    parsed.tensors.emplace(
        std::move(entry.name),
        HeaderTensor{dtype, std::move(entry.shape), entry.begin, entry.end});
  }
  return parsed;
}

std::map<std::string, std::string, std::less<>> ParseWeightMap(
    const std::filesystem::path& path, InspectionTelemetry* telemetry) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw json::Error("cannot stat index " + path.string());
  }
  telemetry->json_bytes_read += size;
  const json::Value root = json::ParseFile(path, kMaximumJsonBytes);
  const auto& weight_map = RequireField(root, "weight_map").AsObject();
  std::map<std::string, std::string, std::less<>> result;
  for (const auto& [name, shard_value] : weight_map) {
    const std::filesystem::path shard =
        SafeRelativePath(shard_value.AsString());
    const auto [unused, inserted] =
        result.emplace(name, shard.generic_string());
    (void)unused;
    if (!inserted) {
      throw json::Error("duplicate tensor in index " + name);
    }
  }
  return result;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("allocation size overflow");
  }
  return (value + alignment - 1) & ~(alignment - 1);
}

bool Cancelled(const CancellationToken* cancellation, std::string* error) {
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    SetError(error, "MiniMax H3 phase load cancelled");
    return true;
  }
  return false;
}

bool InjectFailure(FailureInjector* failures, std::string_view operation,
                   std::string* error) {
  if (failures != nullptr && failures->ShouldFail(operation)) {
    SetError(error,
             "injected MiniMax H3 load failure at " + std::string(operation));
    return true;
  }
  return false;
}

struct FileInterval {
  std::filesystem::path shard;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

std::vector<FileInterval> BuildIntervals(
    std::span<const TensorDescriptor* const> tensors) {
  const std::uint64_t page_size =
      static_cast<std::uint64_t>(platform::SystemPageSize());
  std::vector<FileInterval> intervals;
  intervals.reserve(tensors.size());
  for (const TensorDescriptor* tensor : tensors) {
    const std::uint64_t begin = (tensor->file_offset / page_size) * page_size;
    if (tensor->file_offset >
        std::numeric_limits<std::uint64_t>::max() - tensor->payload_bytes) {
      throw std::overflow_error("tensor file interval overflow");
    }
    const std::uint64_t unaligned_end =
        tensor->file_offset + tensor->payload_bytes;
    intervals.push_back({tensor->shard, begin, unaligned_end});
  }
  std::sort(intervals.begin(), intervals.end(),
            [](const FileInterval& left, const FileInterval& right) {
              return std::tie(left.shard, left.begin, left.end) <
                     std::tie(right.shard, right.begin, right.end);
            });
  std::vector<FileInterval> merged;
  for (const auto& interval : intervals) {
    if (!merged.empty() && merged.back().shard == interval.shard &&
        interval.begin <= merged.back().end) {
      merged.back().end = std::max(merged.back().end, interval.end);
    } else {
      merged.push_back(interval);
    }
  }
  return merged;
}

void UpdatePeak(MemoryTelemetry* telemetry) {
  telemetry->current_live_bytes =
      telemetry->registered_host_bytes + telemetry->device_weight_bytes +
      telemetry->persistent_bytes + telemetry->scratch_bytes;
  telemetry->peak_live_bytes =
      std::max(telemetry->peak_live_bytes, telemetry->current_live_bytes);
}

}  // namespace

std::string_view ToString(DType dtype) noexcept {
  switch (dtype) {
    case DType::kBFloat16:
      return "BF16";
    case DType::kFloat32:
      return "F32";
    case DType::kInt32:
      return "I32";
  }
  return "unknown";
}

std::size_t DTypeSize(DType dtype) noexcept {
  switch (dtype) {
    case DType::kBFloat16:
      return 2;
    case DType::kFloat32:
    case DType::kInt32:
      return 4;
  }
  return 0;
}

std::string_view ToString(Phase phase) noexcept {
  switch (phase) {
    case Phase::kPromptEncoder:
      return "prompt-encoder";
    case Phase::kAdaLnPrecompute:
      return "adaln-precompute";
    case Phase::kDitCore:
      return "dit-core";
    case Phase::kVisualVae:
      return "visual-vae";
    case Phase::kAudioVae:
      return "audio-vae";
  }
  return "unknown";
}

std::string_view ToString(ResidencyMode mode) noexcept {
  switch (mode) {
    case ResidencyMode::kDeviceCopy:
      return "device-copy";
    case ResidencyMode::kMappedReadOnly:
      return "mapped-read-only";
  }
  return "unknown";
}

std::optional<ModelInventory> ModelInventory::Inspect(
    const std::filesystem::path& model_root,
    const std::filesystem::path& source_manifest, std::string* error) {
  return Inspect(model_root, source_manifest, InspectionOptions{}, error);
}

std::optional<ModelInventory> ModelInventory::Inspect(
    const std::filesystem::path& model_root,
    const std::filesystem::path& source_manifest,
    const InspectionOptions& options, std::string* error) {
  try {
    ModelInventory inventory;
    inventory.model_root_ = std::filesystem::weakly_canonical(model_root);
    if (!std::filesystem::is_directory(inventory.model_root_)) {
      throw json::Error("model root is not a directory");
    }
    std::error_code size_error;
    const std::uintmax_t manifest_size =
        std::filesystem::file_size(source_manifest, size_error);
    if (size_error) {
      throw json::Error("cannot stat source manifest");
    }
    inventory.telemetry_.json_bytes_read += manifest_size;
    if (options.require_pinned_manifest &&
        Sha256File(source_manifest) != kSourceManifestSha256) {
      throw json::Error(
          "source manifest SHA-256 does not match the compiled contract");
    }
    const json::Value manifest =
        json::ParseFile(source_manifest, kMaximumJsonBytes);
    if (RequireField(manifest, "schema").AsString() != kManifestSchema ||
        RequireField(manifest, "model_kind").AsString() != kModelKind ||
        RequireField(manifest, "revision").AsString() != kRevision) {
      throw json::Error("source manifest identity does not match MiniMax H3");
    }

    std::map<std::filesystem::path, std::uint64_t> expected_file_sizes;
    for (const auto& file : RequireField(manifest, "files").AsArray()) {
      const auto relative =
          SafeRelativePath(RequireField(file, "path").AsString());
      const std::uint64_t size = RequireField(file, "size").AsUint64();
      if (!expected_file_sizes.emplace(relative, size).second) {
        throw json::Error("duplicate file in source manifest");
      }
    }

    struct ExpectedTensor {
      TensorDescriptor descriptor;
      std::uint64_t relative_begin{0};
      std::uint64_t relative_end{0};
    };
    std::vector<ExpectedTensor> expected_tensors;
    for (const auto& tensor : RequireField(manifest, "tensors").AsArray()) {
      TensorDescriptor descriptor;
      descriptor.component = RequireField(tensor, "component").AsString();
      descriptor.name = RequireField(tensor, "name").AsString();
      descriptor.dtype = ParseDType(RequireField(tensor, "dtype").AsString());
      for (const auto& dimension : RequireField(tensor, "shape").AsArray()) {
        descriptor.shape.push_back(dimension.AsUint64());
      }
      descriptor.shard =
          SafeRelativePath(RequireField(tensor, "shard").AsString());
      descriptor.payload_bytes = RequireField(tensor, "byte_count").AsUint64();
      descriptor.phase = TensorPhase(descriptor.component, descriptor.name);
      const auto& offsets = RequireField(tensor, "data_offsets").AsArray();
      if (offsets.size() != 2) {
        throw json::Error("source manifest tensor has invalid offsets");
      }
      const std::uint64_t begin = offsets[0].AsUint64();
      const std::uint64_t end = offsets[1].AsUint64();
      if (begin >= end || end - begin != descriptor.payload_bytes) {
        throw json::Error("source manifest tensor byte count differs");
      }
      expected_tensors.push_back({std::move(descriptor), begin, end});
    }
    if (expected_tensors.empty()) {
      throw json::Error("source manifest contains no tensors");
    }

    std::map<std::string, std::string, std::less<>> indexed;
    for (const auto& [component, relative_index] :
         std::array<std::pair<std::string_view, std::string_view>, 2>{
             std::pair{"text_encoder",
                       "FL2VA/text_encoder/model.safetensors.index.json"},
             std::pair{"transformer",
                       "FL2VA/transformer/model.safetensors.index.json"}}) {
      const auto index_path = inventory.model_root_ / relative_index;
      auto weight_map = ParseWeightMap(index_path, &inventory.telemetry_);
      for (auto& [name, shard] : weight_map) {
        const std::string key = std::string(component) + ":" + name;
        const std::filesystem::path full_relative =
            std::filesystem::path(relative_index).parent_path() / shard;
        if (!indexed.emplace(key, full_relative.generic_string()).second) {
          throw json::Error("duplicate indexed tensor " + key);
        }
      }
    }

    std::map<std::filesystem::path, std::vector<ExpectedTensor*>>
        expected_by_shard;
    for (auto& expected : expected_tensors) {
      expected_by_shard[expected.descriptor.shard].push_back(&expected);
      if (expected.descriptor.component == "text_encoder" ||
          expected.descriptor.component == "transformer") {
        const std::string key =
            expected.descriptor.component + ":" + expected.descriptor.name;
        const auto iterator = indexed.find(key);
        if (iterator == indexed.end() ||
            iterator->second != expected.descriptor.shard.generic_string()) {
          throw json::Error("safetensors index differs for " + key);
        }
        indexed.erase(iterator);
      }
    }
    if (!indexed.empty()) {
      throw json::Error("safetensors index contains an unknown tensor");
    }

    inventory.tensors_.reserve(expected_tensors.size());
    inventory.shards_.reserve(expected_by_shard.size());
    for (auto& [relative_shard, expected] : expected_by_shard) {
      const auto size_iterator = expected_file_sizes.find(relative_shard);
      if (size_iterator == expected_file_sizes.end()) {
        throw json::Error("source manifest lacks shard file inventory");
      }
      const ParsedHeader actual = ParseSafetensorsHeader(
          inventory.model_root_ / relative_shard, &inventory.telemetry_);
      if (actual.file_bytes != size_iterator->second ||
          actual.tensors.size() != expected.size()) {
        throw json::Error("safetensors shard inventory differs: " +
                          relative_shard.string());
      }
      for (ExpectedTensor* tensor : expected) {
        const auto actual_iterator =
            actual.tensors.find(tensor->descriptor.name);
        if (actual_iterator == actual.tensors.end()) {
          throw json::Error("safetensors shard lacks tensor " +
                            tensor->descriptor.name);
        }
        const HeaderTensor& header = actual_iterator->second;
        if (header.dtype != tensor->descriptor.dtype ||
            header.shape != tensor->descriptor.shape ||
            header.begin != tensor->relative_begin ||
            header.end != tensor->relative_end) {
          throw json::Error("safetensors tensor metadata differs: " +
                            tensor->descriptor.name);
        }
        tensor->descriptor.file_offset =
            actual.payload_offset + tensor->relative_begin;
      }
      inventory.shards_.push_back({relative_shard, actual.file_bytes,
                                   actual.payload_offset, expected.size()});
    }

    std::sort(
        expected_tensors.begin(), expected_tensors.end(),
        [](const ExpectedTensor& left, const ExpectedTensor& right) {
          return std::tie(left.descriptor.component, left.descriptor.name) <
                 std::tie(right.descriptor.component, right.descriptor.name);
        });
    for (auto& expected : expected_tensors) {
      const std::size_t index = inventory.tensors_.size();
      if (!inventory.tensor_index_.emplace(expected.descriptor.name, index)
               .second) {
        throw json::Error("duplicate tensor name across H3 components");
      }
      inventory.tensors_.push_back(std::move(expected.descriptor));
    }
    return inventory;
  } catch (const std::exception& exception) {
    SetError(error,
             std::string("MiniMax H3 inspection failed: ") + exception.what());
    return std::nullopt;
  }
}

const TensorDescriptor* ModelInventory::FindTensor(
    std::string_view name) const noexcept {
  const auto iterator = tensor_index_.find(std::string(name));
  if (iterator == tensor_index_.end()) {
    return nullptr;
  }
  return &tensors_[iterator->second];
}

std::vector<const TensorDescriptor*> ModelInventory::TensorsForPhase(
    Phase phase) const {
  std::vector<const TensorDescriptor*> result;
  for (const auto& tensor : tensors_) {
    if (tensor.phase == phase) {
      result.push_back(&tensor);
    }
  }
  return result;
}

bool FailureInjector::ShouldFail(std::string_view operation) {
  last_operation_ = operation;
  const bool result = fail_after_.has_value() && operations_ == *fail_after_;
  ++operations_;
  return result;
}

#if !defined(ENGINE_ENABLE_HIP)
std::unique_ptr<ResidencyBackend> CreateHipResidencyBackend(
    std::string* error) {
  SetError(error, "MiniMax H3 HIP residency requires ENGINE_ENABLE_HIP");
  return nullptr;
}
#endif

PhaseSession::~PhaseSession() {
  Release();
}

PhaseSession::PhaseSession(PhaseSession&& other) noexcept
    : backend_(std::exchange(other.backend_, nullptr)),
      phase_(other.phase_),
      mode_(other.mode_),
      stream_(std::exchange(other.stream_, nullptr)),
      device_weights_(std::exchange(other.device_weights_, nullptr)),
      persistent_(std::exchange(other.persistent_, nullptr)),
      scratch_(std::exchange(other.scratch_, nullptr)),
      mappings_(std::move(other.mappings_)),
      tensors_(std::move(other.tensors_)),
      telemetry_(other.telemetry_),
      timed_execution_(other.timed_execution_),
      phase_lease_(std::exchange(other.phase_lease_, false)) {}

PhaseSession& PhaseSession::operator=(PhaseSession&& other) noexcept {
  if (this != &other) {
    Release();
    backend_ = std::exchange(other.backend_, nullptr);
    phase_ = other.phase_;
    mode_ = other.mode_;
    stream_ = std::exchange(other.stream_, nullptr);
    device_weights_ = std::exchange(other.device_weights_, nullptr);
    persistent_ = std::exchange(other.persistent_, nullptr);
    scratch_ = std::exchange(other.scratch_, nullptr);
    mappings_ = std::move(other.mappings_);
    tensors_ = std::move(other.tensors_);
    telemetry_ = other.telemetry_;
    timed_execution_ = other.timed_execution_;
    phase_lease_ = std::exchange(other.phase_lease_, false);
  }
  return *this;
}

void PhaseSession::Release() noexcept {
  if (backend_ == nullptr) {
    return;
  }
  if (scratch_ != nullptr) {
    backend_->Free(scratch_);
    scratch_ = nullptr;
  }
  if (persistent_ != nullptr) {
    backend_->Free(persistent_);
    persistent_ = nullptr;
  }
  if (device_weights_ != nullptr) {
    backend_->Free(device_weights_);
    device_weights_ = nullptr;
  }
  for (auto iterator = mappings_.rbegin(); iterator != mappings_.rend();
       ++iterator) {
    if (iterator->registered) {
      backend_->Unregister(iterator->host);
    }
    iterator->mapping_file.Close();
    iterator->host = nullptr;
  }
  mappings_.clear();
  if (stream_ != nullptr) {
    backend_->DestroyStream(stream_);
    stream_ = nullptr;
  }
  tensors_.clear();
  telemetry_.current_live_bytes = 0;
  if (phase_lease_) {
    backend_->ReleasePhase();
    phase_lease_ = false;
  }
  backend_ = nullptr;
}

std::optional<PhaseSession> PhaseSession::Load(
    const ModelInventory& inventory, Phase phase, const LoadOptions& options,
    ResidencyBackend& backend, const CancellationToken* cancellation,
    FailureInjector* failures, std::string* error) {
  PhaseSession session(backend);
  session.phase_ = phase;
  session.mode_ = options.mode;
  try {
    std::string backend_error;
    if (InjectFailure(failures, "target-validation", error) ||
        !backend.IsSupportedTarget(&backend_error)) {
      if (error != nullptr && error->empty()) {
        SetError(error, backend_error);
      }
      return std::nullopt;
    }
    if (!backend.TryAcquirePhase()) {
      SetError(error,
               "MiniMax H3 backend already has an active residency phase");
      return std::nullopt;
    }
    session.phase_lease_ = true;
    if (Cancelled(cancellation, error)) {
      return std::nullopt;
    }
    if (InjectFailure(failures, "stream-create", error) ||
        !backend.CreateStream(&session.stream_, &backend_error)) {
      if (error != nullptr && error->empty()) {
        SetError(error, backend_error);
      }
      return std::nullopt;
    }

    const std::vector<const TensorDescriptor*> phase_tensors =
        inventory.TensorsForPhase(phase);
    if (phase_tensors.empty()) {
      SetError(error, "MiniMax H3 phase has no tensors: " +
                          std::string(ToString(phase)));
      return std::nullopt;
    }
    const auto intervals = BuildIntervals(phase_tensors);
    session.mappings_.reserve(intervals.size());
    for (const auto& interval : intervals) {
      if (Cancelled(cancellation, error) ||
          InjectFailure(failures, "file-map", error)) {
        return std::nullopt;
      }
      const auto path = inventory.model_root() / interval.shard;
      const std::uint64_t length64 = interval.end - interval.begin;
      if (length64 > std::numeric_limits<std::size_t>::max()) {
        SetError(error, "MiniMax H3 mapping is too large");
        return std::nullopt;
      }
      const std::size_t length = static_cast<std::size_t>(length64);
      Mapping mapping;
      mapping.shard = interval.shard;
      std::string map_error;
      if (!mapping.mapping_file.OpenRange(path, interval.begin, length,
                                          platform::MapHint::kSequential,
                                          &map_error)) {
        SetError(error, "cannot map " + path.string() + ": " + map_error);
        return std::nullopt;
      }
      mapping.host = mapping.mapping_file.data;
      mapping.bytes = length;
      mapping.file_offset = interval.begin;
      session.mappings_.push_back(std::move(mapping));
      session.telemetry_.file_backed_bytes += length;
      if (options.prefault) {
        if (InjectFailure(failures, "prefault", error)) {
          return std::nullopt;
        }
        const std::size_t page_size = platform::SystemPageSize();
        volatile std::uint8_t accumulator = 0;
        const auto* bytes =
            static_cast<const std::uint8_t*>(session.mappings_.back().host);
        for (std::size_t offset = 0; offset < length; offset += page_size) {
          accumulator = static_cast<std::uint8_t>(accumulator ^ bytes[offset]);
        }
        if (length != 0) {
          accumulator =
              static_cast<std::uint8_t>(accumulator ^ bytes[length - 1]);
        }
        (void)accumulator;
        session.telemetry_.prefaulted_bytes += length;
      }
    }

    const auto find_mapping =
        [&](const TensorDescriptor& tensor) -> const Mapping* {
      for (auto& mapping : session.mappings_) {
        if (mapping.shard == tensor.shard &&
            tensor.file_offset >= mapping.file_offset &&
            tensor.file_offset + tensor.payload_bytes <=
                mapping.file_offset + mapping.bytes) {
          return &mapping;
        }
      }
      return nullptr;
    };

    if (options.mode == ResidencyMode::kMappedReadOnly) {
      for (auto& mapping : session.mappings_) {
        if (Cancelled(cancellation, error) ||
            InjectFailure(failures, "host-register", error)) {
          return std::nullopt;
        }
        if (!backend.RegisterReadOnlyMapped(mapping.host, mapping.bytes,
                                            &mapping.device_alias,
                                            &backend_error)) {
          SetError(error, backend_error);
          return std::nullopt;
        }
        mapping.registered = true;
        session.telemetry_.registered_host_bytes += mapping.bytes;
        UpdatePeak(&session.telemetry_);
      }
      for (const TensorDescriptor* tensor : phase_tensors) {
        const Mapping* mapping = find_mapping(*tensor);
        if (mapping == nullptr) {
          SetError(error, "MiniMax H3 tensor mapping was not found");
          return std::nullopt;
        }
        const std::size_t delta = static_cast<std::size_t>(
            tensor->file_offset - mapping->file_offset);
        session.tensors_.push_back(
            {tensor,
             static_cast<const std::byte*>(mapping->device_alias) + delta,
             static_cast<std::size_t>(tensor->payload_bytes)});
      }
    } else {
      std::size_t total_bytes = 0;
      std::vector<std::size_t> offsets;
      offsets.reserve(phase_tensors.size());
      for (const TensorDescriptor* tensor : phase_tensors) {
        total_bytes = AlignUp(total_bytes, kDeviceAlignment);
        offsets.push_back(total_bytes);
        if (tensor->payload_bytes >
            std::numeric_limits<std::size_t>::max() - total_bytes) {
          SetError(error, "MiniMax H3 device weight allocation overflows");
          return std::nullopt;
        }
        total_bytes += static_cast<std::size_t>(tensor->payload_bytes);
      }
      if (InjectFailure(failures, "weight-allocate", error) ||
          !backend.Allocate(total_bytes, &session.device_weights_,
                            &backend_error)) {
        if (error != nullptr && error->empty()) {
          SetError(error, backend_error);
        }
        return std::nullopt;
      }
      session.telemetry_.device_weight_bytes = total_bytes;
      UpdatePeak(&session.telemetry_);
      for (std::size_t index = 0; index < phase_tensors.size(); ++index) {
        if (Cancelled(cancellation, error) ||
            InjectFailure(failures, "weight-copy", error)) {
          return std::nullopt;
        }
        const TensorDescriptor& tensor = *phase_tensors[index];
        const Mapping* mapping = find_mapping(tensor);
        if (mapping == nullptr) {
          SetError(error, "MiniMax H3 tensor mapping was not found");
          return std::nullopt;
        }
        const std::size_t source_delta =
            static_cast<std::size_t>(tensor.file_offset - mapping->file_offset);
        void* destination =
            static_cast<std::byte*>(session.device_weights_) + offsets[index];
        if (!backend.CopyToDevice(
                destination,
                static_cast<const std::byte*>(mapping->host) + source_delta,
                static_cast<std::size_t>(tensor.payload_bytes), session.stream_,
                &backend_error)) {
          SetError(error, backend_error);
          return std::nullopt;
        }
        session.tensors_.push_back(
            {&tensor, destination,
             static_cast<std::size_t>(tensor.payload_bytes)});
      }
      if (InjectFailure(failures, "weight-synchronize", error) ||
          !backend.Synchronize(session.stream_, &backend_error)) {
        if (error != nullptr && error->empty()) {
          SetError(error, backend_error);
        }
        return std::nullopt;
      }
      for (auto& mapping : session.mappings_) {
        mapping.mapping_file.Close();
        mapping.host = nullptr;
      }
      session.mappings_.clear();
    }

    if (options.persistent_bytes != 0) {
      if (InjectFailure(failures, "persistent-allocate", error) ||
          !backend.Allocate(options.persistent_bytes, &session.persistent_,
                            &backend_error)) {
        if (error != nullptr && error->empty()) {
          SetError(error, backend_error);
        }
        return std::nullopt;
      }
      session.telemetry_.persistent_bytes = options.persistent_bytes;
      UpdatePeak(&session.telemetry_);
    }
    if (options.scratch_bytes != 0) {
      if (InjectFailure(failures, "scratch-allocate", error) ||
          !backend.Allocate(options.scratch_bytes, &session.scratch_,
                            &backend_error)) {
        if (error != nullptr && error->empty()) {
          SetError(error, backend_error);
        }
        return std::nullopt;
      }
      session.telemetry_.scratch_bytes = options.scratch_bytes;
      UpdatePeak(&session.telemetry_);
    }
    if (Cancelled(cancellation, error)) {
      return std::nullopt;
    }
    return session;
  } catch (const std::exception& exception) {
    SetError(error,
             std::string("MiniMax H3 phase load failed: ") + exception.what());
    return std::nullopt;
  }
}

const ResidentTensor* PhaseSession::FindTensor(
    std::string_view name) const noexcept {
  for (const auto& tensor : tensors_) {
    if (tensor.descriptor != nullptr && tensor.descriptor->name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

}  // namespace gufo::minimax_h3
