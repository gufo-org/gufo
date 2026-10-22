#include "src/models/embedding_gemma2/loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {

class MappedFile {
public:
  MappedFile() = default;
  ~MappedFile() { Reset(); }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] bool Open(const std::filesystem::path& path) {
    struct stat status{};
    if (stat(path.c_str(), &status) != 0 || status.st_size <= 0) {
      return false;
    }
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      return false;
    }
    void* mapping = mmap(nullptr, static_cast<std::size_t>(status.st_size),
                         PROT_READ, MAP_PRIVATE, descriptor, 0);
    close(descriptor);
    if (mapping == MAP_FAILED) {
      return false;
    }
    data_ = mapping;
    size_ = static_cast<std::size_t>(status.st_size);
    return true;
  }

  [[nodiscard]] const std::byte* data() const noexcept {
    return static_cast<const std::byte*>(data_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)munmap(data_, size_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  void* data_{nullptr};
  std::size_t size_{0};
};

DType ParseDType(std::string_view name) noexcept {
  if (name == "F32") {
    return DType::kF32;
  }
  if (name == "BF16") {
    return DType::kBF16;
  }
  if (name == "F16") {
    return DType::kF16;
  }
  return DType::kUnknown;
}

std::uint64_t ReadLittleEndian64(const std::byte* bytes) noexcept {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |=
        static_cast<std::uint64_t>(std::to_integer<unsigned char>(bytes[index]))
        << (index * 8U);
  }
  return value;
}

/// Indexes the text-backbone tensors of one safetensors file. Tensors that
/// belong to the other towers are skipped without validation.
bool LoadSafetensors(const std::filesystem::path& path, TensorStore* store,
                     std::vector<std::shared_ptr<void>>* mappings,
                     std::string* error) {
  auto mapping = std::make_shared<MappedFile>();
  if (!mapping->Open(path) || mapping->size() < 8U) {
    *error = "cannot map " + path.string();
    return false;
  }
  const std::uint64_t header_size = ReadLittleEndian64(mapping->data());
  if (header_size == 0 || header_size > mapping->size() - 8U) {
    *error = "invalid safetensors header in " + path.string();
    return false;
  }
  json::Value header;
  try {
    header = json::parse(
        std::string_view(reinterpret_cast<const char*>(mapping->data() + 8),
                         static_cast<std::size_t>(header_size)));
  } catch (const std::exception&) {
    *error = "unparsable safetensors header in " + path.string();
    return false;
  }
  if (!header.is_object()) {
    *error = "invalid safetensors header in " + path.string();
    return false;
  }

  const std::uint64_t payload_offset = 8U + header_size;
  for (const auto& [name, specification] : header.members()) {
    if (name == "__metadata__" || !name.starts_with(kTextTensorPrefix)) {
      continue;
    }
    if (!specification.is_object()) {
      *error = "invalid tensor entry " + name;
      return false;
    }
    Tensor tensor;
    tensor.name = name;
    tensor.dtype = ParseDType(specification.member_str("dtype"));
    if (const json::Value* shape = specification.find("shape");
        shape != nullptr && shape->is_array()) {
      for (const auto& dimension : shape->items()) {
        tensor.shape.push_back(dimension.as_size());
      }
    }
    const json::Value* offsets = specification.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2) {
      *error = "missing data_offsets for " + name;
      return false;
    }
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    const auto elements = tensor.NumElements();
    const std::size_t element_size = DTypeSize(tensor.dtype);
    if (element_size == 0) {
      *error = "unsupported dtype for " + name;
      return false;
    }
    if (!elements.has_value() || begin >= end ||
        end > mapping->size() - payload_offset ||
        *elements > std::numeric_limits<std::uint64_t>::max() / element_size ||
        end - begin != static_cast<std::uint64_t>(*elements) * element_size) {
      *error = "inconsistent tensor extent for " + name;
      return false;
    }
    tensor.data = mapping->data() + payload_offset + begin;
    tensor.byte_count = end - begin;
    if (!store->Add(std::move(tensor))) {
      *error = "duplicate tensor " + name;
      return false;
    }
  }
  mappings->push_back(std::move(mapping));
  return true;
}

void AppendLayerTensors(const TextConfig& config, std::size_t index,
                        std::vector<ExpectedTensor>* tensors) {
  const LayerGeometry& layer = config.layers[index];
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t per_layer = config.hidden_size_per_layer_input;
  const std::uint64_t query = config.num_attention_heads * layer.head_dim;
  const std::uint64_t key = layer.num_key_value_heads * layer.head_dim;
  const std::string prefix =
      std::string(kTextTensorPrefix) + "layers." + std::to_string(index) + ".";
  const auto add = [&](const char* suffix, std::vector<std::uint64_t> shape) {
    tensors->push_back({prefix + suffix, std::move(shape)});
  };
  add("input_layernorm.weight", {hidden});
  add("post_attention_layernorm.weight", {hidden});
  add("pre_feedforward_layernorm.weight", {hidden});
  add("post_feedforward_layernorm.weight", {hidden});
  add("layer_scalar", {1});
  add("self_attn.q_proj.weight", {query, hidden});
  add("self_attn.k_proj.weight", {key, hidden});
  add("self_attn.v_proj.weight", {key, hidden});
  add("self_attn.o_proj.weight", {hidden, query});
  add("self_attn.q_norm.weight", {layer.head_dim});
  add("self_attn.k_norm.weight", {layer.head_dim});
  add("mlp.gate_proj.weight", {config.intermediate_size, hidden});
  add("mlp.up_proj.weight", {config.intermediate_size, hidden});
  add("mlp.down_proj.weight", {hidden, config.intermediate_size});
  add("ple_block.per_layer_input_gate.weight", {per_layer, hidden});
  add("ple_block.per_layer_projection.weight", {hidden, per_layer});
  add("ple_block.post_per_layer_input_norm.weight", {hidden});
}

}  // namespace

std::vector<ExpectedTensor> ExpectedTextTensors(const TextConfig& config) {
  std::vector<ExpectedTensor> tensors;
  const std::string prefix = kTextTensorPrefix;
  tensors.push_back({prefix + "embed_tokens.weight",
                     {config.vocab_size, config.hidden_size}});
  tensors.push_back({prefix + "norm.weight", {config.hidden_size}});
  tensors.push_back({prefix + "embedding_projection.weight",
                     {config.embedding_dim, config.hidden_size}});
  tensors.push_back({prefix + "ple.per_layer_model_projection.weight",
                     {static_cast<std::uint64_t>(config.num_hidden_layers) *
                          config.hidden_size_per_layer_input,
                      config.hidden_size}});
  tensors.push_back({prefix + "ple.per_layer_projection_norm.weight",
                     {config.hidden_size_per_layer_input}});
  for (std::size_t index = 0; index < config.layers.size(); ++index) {
    AppendLayerTensors(config, index, &tensors);
  }
  return tensors;
}

LoadResult LoadModelDirectory(const std::string& model_dir) {
  LoadResult result;
  result.store = std::make_unique<TensorStore>();
  const auto config = LoadModelConfigFromPath(model_dir);
  if (!config.has_value()) {
    result.error =
        "embedding_gemma2: cannot parse " + model_dir + "/config.json";
    return result;
  }
  result.config = *config;
  if (!IsSupportedModelConfig(result.config)) {
    result.error =
        "embedding_gemma2: expected the google/embeddinggemma-2 BF16 "
        "configuration";
    return result;
  }

  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "model.safetensors";
  std::string error;
  if (!LoadSafetensors(path, result.store.get(), &result.mappings, &error)) {
    result.error = "embedding_gemma2: " + error;
    return result;
  }
  for (const ExpectedTensor& expected :
       ExpectedTextTensors(result.config.text)) {
    const Tensor* tensor = result.store->Find(expected.name);
    if (tensor == nullptr) {
      result.error = "embedding_gemma2: missing tensor " + expected.name;
      return result;
    }
    if (tensor->dtype != DType::kBF16 || tensor->shape != expected.shape) {
      result.error =
          "embedding_gemma2: unexpected dtype or shape for " + expected.name;
      return result;
    }
  }
  result.ok = true;
  return result;
}

bool LooksLikeEmbeddingGemma2(const std::string& model_dir) {
  const auto config = LoadModelConfigFromPath(model_dir);
  return config.has_value() && IsSupportedModelConfig(*config);
}

}  // namespace gufo::models::embedding_gemma2
