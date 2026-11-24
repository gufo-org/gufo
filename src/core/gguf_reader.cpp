#include "src/core/gguf_reader.hpp"

#include <gufo/gguf.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/rust/callback.hpp"

namespace gufo::core {

namespace {

// Host memory the descriptors of one file may take.
constexpr std::size_t kDescriptorBudget = std::size_t{128} * 1024 * 1024;

std::string UnsizedError(std::string_view name) {
  return "Invalid GGUF tensor shape or unsupported storage type: " +
         std::string(name);
}

std::string_view View(GufoGgufSlice slice) {
  return {reinterpret_cast<const char*>(slice.data), slice.len};
}

// The items of the array a block belongs to; a block at index 0 starts it
// and reserves all its items.
template<class T>
std::vector<T>& ArrayItems(GgufMetadataValue& array,
                           const GufoGgufItems& block) {
  if (block.first != 0)
    return std::get<std::vector<T>>(array.value);
  auto& items = array.value.emplace<std::vector<T>>();
  items.reserve(block.len);
  return items;
}

template<class T>
void AppendItems(GgufMetadataValue& array, const GufoGgufItems& block,
                 const T* items) {
  auto& all = ArrayItems<T>(array, block);
  // The items of an empty block point nowhere.
  if (block.count == 0)
    return;
  all.insert(all.end(), items, items + block.count);
}

// Adds a block of items to the array the next metadata entry holds.
void Append(GgufMetadataValue& array, const GufoGgufItems& block) {
  switch (block.kind) {
    case GUFO_GGUF_STRINGS: {
      auto& items = ArrayItems<std::string_view>(array, block);
      for (std::size_t i = 0; i < block.count; ++i)
        items.push_back(View(block.strings[i]));
      return;
    }
    case GUFO_GGUF_UNSIGNEDS:
      AppendItems(array, block, block.unsigneds);
      return;
    case GUFO_GGUF_SIGNEDS:
      AppendItems(array, block, block.signeds);
      return;
    case GUFO_GGUF_FLOATS:
      AppendItems(array, block, block.floats);
      return;
    case GUFO_GGUF_UNSIGNED:
    case GUFO_GGUF_SIGNED:
    case GUFO_GGUF_FLOAT:
    case GUFO_GGUF_BOOL:
    case GUFO_GGUF_STRING:
      break;
  }
  throw std::logic_error("Rust GGUF header parser sent a block of no array");
}

// The value of a metadata entry; an array takes the items `array` collected.
GgufMetadataValue MetadataValue(const GufoGgufMetadata& entry,
                                GgufMetadataValue& array) {
  GgufMetadataValue value;
  switch (entry.kind) {
    case GUFO_GGUF_UNSIGNED:
      value.value.emplace<std::uint64_t>(entry.unsigned_value);
      break;
    case GUFO_GGUF_SIGNED:
      value.value.emplace<std::int64_t>(entry.signed_value);
      break;
    case GUFO_GGUF_FLOAT:
      value.value.emplace<double>(entry.float_value);
      break;
    case GUFO_GGUF_BOOL:
      value.value.emplace<bool>(entry.unsigned_value != 0);
      break;
    case GUFO_GGUF_STRING:
      value.value.emplace<std::string_view>(View(entry.string));
      break;
    case GUFO_GGUF_STRINGS:
    case GUFO_GGUF_UNSIGNEDS:
    case GUFO_GGUF_SIGNEDS:
    case GUFO_GGUF_FLOATS:
      value.value = std::move(array.value);
      break;
  }
  value.type = static_cast<GgufValueType>(entry.value_type);
  return value;
}

}  // namespace

GgufReader::~GgufReader() = default;

GgufReader::GgufReader(GgufReader&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      source_path_(std::move(other.source_path_)),
      mapping_(std::move(other.mapping_)),
      version_(other.version_),
      alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)),
      tensors_(std::move(other.tensors_)),
      tensor_index_(std::move(other.tensor_index_)),
      mapped_regions_(std::move(other.mapped_regions_)),
      shards_(std::move(other.shards_)) {
  other.data_ = nullptr;
  other.size_ = 0;
}

GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
  if (this != &other) {
    mapping_.Close();
    shards_.clear();
    mapped_regions_.clear();

    data_ = other.data_;
    size_ = other.size_;
    source_path_ = std::move(other.source_path_);
    mapping_ = std::move(other.mapping_);
    version_ = other.version_;
    alignment_ = other.alignment_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
    mapped_regions_ = std::move(other.mapped_regions_);
    shards_ = std::move(other.shards_);

    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

std::unique_ptr<GgufReader> GgufReader::OpenFile(
    const std::filesystem::path& path, std::string* error_msg,
    GgufOpenOptions options) {
  auto first = OpenSingleFile(path, error_msg, options);
  if (!first) {
    return nullptr;
  }

  // `split.count` is optional; when present it is the number of shards. Some
  // writers emit split.count = 0 for a single-file model, which llama.cpp also
  // reads as "not split", so treat 0/1 as one file instead of rejecting it.
  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  if (first->FindMetadata("split.count") &&
      (!first->GetMetadataUint32("split.count") || split_count > 65535)) {
    if (error_msg)
      *error_msg = "Invalid GGUF split count";
    return nullptr;
  }
  if (split_count <= 1) {
    return first;
  }
  return OpenSplitFileSet(path, std::move(first), error_msg, options);
}

std::unique_ptr<GgufReader> GgufReader::OpenSingleFile(
    const std::filesystem::path& path, std::string* error_msg,
    GgufOpenOptions options) {
  std::string open_error;
  auto mapping = platform::ReadOnlyMappedFile::Open(path, &open_error);
  if (!mapping.valid()) {
    if (error_msg != nullptr) {
      *error_msg =
          "Failed to open or map file: " + path.string() + ": " + open_error;
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->data_ = mapping.data();
  reader->size_ = mapping.size();
  reader->source_path_ = path;
  reader->mapping_ = std::move(mapping);

  try {
    if (!reader->ParseHeaders(error_msg, options)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back({reader->data_, reader->size_,
                                     reader->mapping_.PlatformHandle(),
                                     reader->source_path_});
  return reader;
}

std::unique_ptr<GgufReader> GgufReader::OpenSplitFileSet(
    const std::filesystem::path& path, std::unique_ptr<GgufReader> first,
    std::string* error_msg, GgufOpenOptions options) {
  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  const auto split_no = first->GetMetadataUint32("split.no");
  const auto total_tensor_count =
      first->GetMetadataUint64("split.tensors.count");
  if (!split_no.has_value() || *split_no >= split_count ||
      !total_tensor_count.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid split GGUF metadata in " + path.string();
    }
    return nullptr;
  }

  const std::string filename = path.filename().string();
  const std::regex split_pattern{R"(^(.*)-([0-9]+)-of-([0-9]+)\.gguf$)"};
  std::smatch match;
  if (!std::regex_match(filename, match, split_pattern)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Split GGUF filename does not match "
          "<name>-00001-of-00002.gguf: " +
          filename;
    }
    return nullptr;
  }

  const std::string prefix = match[1].str();
  const std::size_t index_width = match[2].str().size();
  const std::size_t count_width = match[3].str().size();
  const auto build_path = [&](std::uint32_t index) {
    std::ostringstream name;
    name << prefix << '-' << std::setfill('0')
         << std::setw(static_cast<int>(index_width)) << (index + 1) << "-of-"
         << std::setw(static_cast<int>(count_width)) << split_count << ".gguf";
    return path.parent_path() / name.str();
  };

  std::vector<std::unique_ptr<GgufReader>> shards(split_count);
  shards[*split_no] = std::move(first);
  for (std::uint32_t i = 0; i < split_count; ++i) {
    if (shards[i]) {
      continue;
    }
    const auto shard_path = build_path(i);
    shards[i] = OpenSingleFile(shard_path, error_msg, options);
    if (!shards[i]) {
      if (error_msg != nullptr && error_msg->empty()) {
        *error_msg = "Failed to open split GGUF shard: " + shard_path.string();
      }
      return nullptr;
    }
  }

  const auto architecture =
      shards[0]->GetMetadataString("general.architecture");
  for (std::uint32_t i = 0; i < split_count; ++i) {
    const auto shard_no = shards[i]->GetMetadataUint32("split.no");
    const auto shard_count = shards[i]->GetMetadataUint32("split.count");
    const auto shard_architecture =
        shards[i]->GetMetadataString("general.architecture");
    if (!shard_no.has_value() || *shard_no != i || !shard_count.has_value() ||
        *shard_count != split_count ||
        shards[i]->GetVersion() != shards[0]->GetVersion() ||
        (shard_architecture.has_value() &&
         shard_architecture != architecture)) {
      if (error_msg != nullptr) {
        *error_msg =
            "Inconsistent split GGUF shard metadata: " + build_path(i).string();
      }
      return nullptr;
    }
  }

  auto combined = std::unique_ptr<GgufReader>(new GgufReader());
  combined->version_ = shards[0]->version_;
  combined->alignment_ = shards[0]->alignment_;
  combined->metadata_ = shards[0]->metadata_;
  std::size_t actual_count = 0;
  for (const auto& shard : shards) {
    if (shard->tensors_.size() >
            std::numeric_limits<std::size_t>::max() - actual_count ||
        shard->size_ >
            std::numeric_limits<std::size_t>::max() - combined->size_)
      return nullptr;
    actual_count += shard->tensors_.size();
    combined->size_ += shard->size_;
  }
  if (actual_count != *total_tensor_count) {
    if (error_msg)
      *error_msg = "Split GGUF tensor count mismatch";
    return nullptr;
  }
  combined->tensors_.reserve(actual_count);
  combined->tensor_index_.reserve(
      static_cast<std::size_t>(*total_tensor_count));
  combined->mapped_regions_.reserve(split_count);

  for (const auto& shard : shards) {
    combined->mapped_regions_.push_back({shard->data_, shard->size_,
                                         shard->mapping_.PlatformHandle(),
                                         shard->source_path_});
    for (const auto& tensor : shard->tensors_) {
      if (combined->tensor_index_.contains(tensor.name)) {
        if (error_msg != nullptr) {
          *error_msg = "Duplicate tensor across split GGUF shards: " +
                       std::string(tensor.name);
        }
        return nullptr;
      }
      combined->tensor_index_[tensor.name] = combined->tensors_.size();
      combined->tensors_.push_back(tensor);
    }
  }

  if (combined->tensors_.size() != *total_tensor_count) {
    if (error_msg != nullptr) {
      *error_msg = "Split GGUF tensor count mismatch: expected " +
                   std::to_string(*total_tensor_count) + ", found " +
                   std::to_string(combined->tensors_.size());
    }
    return nullptr;
  }

  combined->shards_ = std::move(shards);
  return combined;
}

std::unique_ptr<GgufReader> GgufReader::OpenMemory(const void* data,
                                                   std::size_t size,
                                                   std::string* error_msg,
                                                   GgufOpenOptions options) {
  if (data == nullptr || size < 24) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid memory buffer for GGUF";
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->data_ = static_cast<const std::uint8_t*>(data);
  reader->size_ = size;

  try {
    if (!reader->ParseHeaders(error_msg, options)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back(
      {reader->data_, reader->size_, platform::kInvalidFile, {}});
  return reader;
}

bool GgufReader::ParseHeaders(std::string* error_msg, GgufOpenOptions options) {
  struct Context {
    GgufReader* reader;
    std::string* error_msg;
    // Reserved once the first tensor arrives, when all passed their checks.
    std::size_t tensor_count;
    // The array the blocks so far hold.
    GgufMetadataValue array;
    ffi::CallbackFailure failure;
  } context{this, error_msg, 0, {}, {}};
  // Bound host-side descriptor allocations independently of mapped weight
  // size, charging what this reader's descriptors take.
  const GufoGgufLimits limits{
      .descriptor_budget = kDescriptorBudget,
      .metadata_entry_bytes = sizeof(GgufMetadataValue) + 64,
      .tensor_entry_bytes = sizeof(GgufTensorInfo) + 128,
      .string_item_bytes = sizeof(std::string_view),
      .allow_unsized_tensors = options.allow_unsized_tensors ? 1 : 0};
  // No exception may cross the Rust frames; it is rethrown below.
  const auto tensor_bytes = [](void*, std::uint32_t type,
                               std::uint64_t elements,
                               std::uint64_t first_dimension) noexcept {
    const auto ggml_type = static_cast<GgmlType>(type);
    const auto block = quant::QuantizedBlockElements(ggml_type);
    return block != 0 && first_dimension % block != 0
               ? std::size_t{0}
               : quant::EncodedSizeBytes(ggml_type, elements);
  };
  const auto counts = [](void* opaque, std::size_t metadata_count,
                         std::size_t tensor_count) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      context.reader->metadata_.reserve(metadata_count);
      context.tensor_count = tensor_count;
      return 0;
    });
  };
  const auto items = [](void* opaque, const GufoGgufItems* block) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      Append(context.array, *block);
      return 0;
    });
  };
  const auto metadata = [](void* opaque,
                           const GufoGgufMetadata* entry) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      context.reader->metadata_.emplace(View(entry->key),
                                        MetadataValue(*entry, context.array));
      return 0;
    });
  };
  const auto tensor = [](void* opaque, const GufoGgufTensor* parsed) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      auto& reader = *context.reader;
      if (reader.tensors_.empty()) {
        reader.tensors_.reserve(context.tensor_count);
        reader.tensor_index_.reserve(context.tensor_count);
      }
      reader.tensor_index_.emplace(View(parsed->name), reader.tensors_.size());
      reader.tensors_.push_back(
          {.name = View(parsed->name),
           .dimensions = {parsed->dimensions,
                          parsed->dimensions + parsed->dimension_count},
           .type = static_cast<GgmlType>(parsed->ggml_type),
           .offset = parsed->offset,
           .data = reader.data_ + parsed->start,
           .size_bytes = parsed->size,
           .sized = parsed->sized != 0});
      return 0;
    });
  };
  // Without error_msg, the parser makes no message.
  GufoGgufBytesFn error = nullptr;
  if (error_msg != nullptr) {
    error = [](void* opaque, const std::uint8_t* data,
               std::size_t len) noexcept {
      auto& context = *static_cast<Context*>(opaque);
      context.failure.Guard([&] {
        context.error_msg->assign(reinterpret_cast<const char*>(data), len);
      });
    };
  }
  GufoGgufHeader header{};
  const GufoGgufParse parsed =
      gufo_gguf_parse_header(data_, size_, &limits, tensor_bytes, counts, items,
                             metadata, tensor, error, &context, &header);
  context.failure.Rethrow();
  switch (parsed) {
    case GUFO_GGUF_COMPLETE:
      version_ = header.version;
      alignment_ = header.alignment;
      return true;
    case GUFO_GGUF_REJECTED:
      return false;
    case GUFO_GGUF_NO_MEMORY:
      throw std::bad_alloc();
    case GUFO_GGUF_STOPPED:
    case GUFO_GGUF_INTERNAL:
      break;
  }
  throw std::runtime_error("Rust GGUF header parser failed internally");
}

std::string GgufReader::UnsizedTensorError() const {
  const auto* tensor = FirstUnsizedTensor();
  return tensor != nullptr ? UnsizedError(tensor->name) : std::string{};
}

const GgufMetadataValue* GgufReader::FindMetadata(
    std::string_view key) const noexcept {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::optional<std::string_view> GgufReader::GetMetadataString(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::string_view>(meta->value)) {
    return std::get<std::string_view>(meta->value);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GgufReader::GetMetadataUint32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  } else if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufReader::GetMetadataUint64(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint64_t>(*value)) {
      return static_cast<std::uint64_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<float> GgufReader::GetMetadataFloat32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<double>(meta->value)) {
    return static_cast<float>(std::get<double>(meta->value));
  }
  return std::nullopt;
}

std::optional<bool> GgufReader::GetMetadataBool(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<bool>(meta->value)) {
    return std::get<bool>(meta->value);
  }
  return std::nullopt;
}

std::vector<std::string_view> GgufReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::vector<std::string_view>>(meta->value)) {
    return std::get<std::vector<std::string_view>>(meta->value);
  }
  return {};
}

const GgufTensorInfo* GgufReader::FindTensor(
    std::string_view name) const noexcept {
  auto it = tensor_index_.find(name);
  if (it != tensor_index_.end()) {
    return &tensors_[it->second];
  }
  return nullptr;
}

bool GgufReader::HasTensor(std::string_view name) const noexcept {
  return tensor_index_.contains(name);
}

std::string GgufReader::GetQuantizationLabel() const {
  // `general.file_type` is deliberately not consulted. The Unsloth UD-*_K_XL
  // artifacts this repo targets carry a stale one -- UD-Q8_K_XL records ftype
  // 15 (Q4_K_M), which is what llama-bench prints for it -- so the recorded
  // value is not evidence about the weights. The tensor table always is.
  //
  // Weighted by element count, not by tensor count: an artifact carries
  // hundreds of tiny F32 norms that would otherwise outvote every projection.
  // Element count also avoids needing each format's byte geometry here, which
  // lives in quant::QuantizedRowBytes and would invert this file's dependency
  // direction.
  std::unordered_map<std::uint16_t, std::uint64_t> elements_by_type;
  std::uint64_t total_elements = 0;
  for (const auto& tensor : tensors_) {
    const std::uint64_t count = tensor.ElementCount();
    elements_by_type[static_cast<std::uint16_t>(tensor.type)] += count;
    total_elements += count;
  }
  if (elements_by_type.empty() || total_elements == 0) {
    return "unknown";
  }
  const auto dominant = std::ranges::max_element(
      elements_by_type, {}, [](const auto& entry) { return entry.second; });
  std::string label(ToString(static_cast<GgmlType>(dominant->first)));
  // The UD-*_K_XL targets spread seven formats across the projections, so
  // naming only the largest would read as a claim about the whole file. Four
  // fifths is the line between "this artifact is that format" and "this is a
  // mix".
  if (dominant->second * 5 < total_elements * 4) {
    label += " mixed";
  }
  return label;
}

bool GgufReader::HasMtpTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("mtp.") ||
           t.name.find(".mtp.") != std::string_view::npos ||
           t.name.find(".nextn.") != std::string_view::npos;
  });
}

bool GgufReader::HasVisionTensors() const noexcept {
  return std::ranges::any_of(tensors_, [](const auto& t) {
    return t.name.starts_with("model.visual") ||
           t.name.starts_with("visual.") || t.name.starts_with("v.");
  });
}

}  // namespace gufo::core
