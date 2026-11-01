#include "src/models/embedding_gemma2/loader.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace gufo::models::embedding_gemma2 {
namespace {

using MappedFile = platform::RawMappedFile;

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

/// Indexes the requested backbones. Unused towers are skipped.
bool LoadSafetensors(const std::filesystem::path& path, TensorStore* store,
                     std::vector<std::shared_ptr<void>>* mappings,
                     std::string* error, bool include_vision,
                     bool include_audio) {
  // RawMappedFile has no destructor; the owner closes the mapping.
  std::shared_ptr<MappedFile> mapping(new MappedFile(), [](MappedFile* m) {
    m->Close();
    delete m;
  });
  if (!mapping->Open(path) || mapping->size < 8U) {
    *error = "cannot map " + path.string();
    return false;
  }
  const std::uint64_t header_size =
      ReadLittleEndian64(static_cast<const std::byte*>(mapping->data));
  if (header_size == 0 || header_size > mapping->size - 8U) {
    *error = "invalid safetensors header in " + path.string();
    return false;
  }
  json::Value header;
  try {
    header = json::parse(
        std::string_view(reinterpret_cast<const char*>(
                             static_cast<const std::byte*>(mapping->data) + 8),
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
    if (name == "__metadata__" ||
        !(name.starts_with(kTextTensorPrefix) ||
          (include_vision && (name.starts_with("vision_tower.") ||
                              name.starts_with("embed_vision."))) ||
          (include_audio && (name.starts_with("audio_tower.") ||
                             name.starts_with("embed_audio."))))) {
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
        end > mapping->size - payload_offset ||
        *elements > std::numeric_limits<std::uint64_t>::max() / element_size ||
        end - begin != static_cast<std::uint64_t>(*elements) * element_size) {
      *error = "inconsistent tensor extent for " + name;
      return false;
    }
    tensor.data =
        static_cast<const std::byte*>(mapping->data) + payload_offset + begin;
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

std::vector<ExpectedTensor> ExpectedVisionTensors() {
  std::vector<ExpectedTensor> tensors = {
      {"vision_tower.patch_embedder.input_proj.weight", {768, 768}},
      {"vision_tower.patch_embedder.position_embedding_table", {2, 10240, 768}},
      {"embed_vision.embedding_projection.weight", {512, 768}},
  };
  for (unsigned i = 0; i < 16; ++i) {
    const std::string p =
        "vision_tower.encoder.layers." + std::to_string(i) + ".";
    for (const auto* name :
         {"input_layernorm.weight", "post_attention_layernorm.weight",
          "pre_feedforward_layernorm.weight",
          "post_feedforward_layernorm.weight"}) {
      tensors.push_back({p + name, {768}});
    }
    for (const auto* name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
      tensors.push_back(
          {p + "self_attn." + name + ".linear.weight", {768, 768}});
    }
    for (const auto* name : {"q_norm.weight", "k_norm.weight"}) {
      tensors.push_back({p + "self_attn." + name, {64}});
    }
    for (const auto* name : {"gate_proj", "up_proj"}) {
      tensors.push_back({p + "mlp." + name + ".linear.weight", {3072, 768}});
    }
    tensors.push_back({p + "mlp.down_proj.linear.weight", {768, 3072}});
  }
  return tensors;
}

std::vector<ExpectedTensor> ExpectedAudioTensors() {
  std::vector<ExpectedTensor> tensors = {
      {"audio_tower.subsample_conv_projection.layer0.conv.weight",
       {128, 1, 3, 3}},
      {"audio_tower.subsample_conv_projection.layer0.norm.weight", {128}},
      {"audio_tower.subsample_conv_projection.layer1.conv.weight",
       {32, 128, 3, 3}},
      {"audio_tower.subsample_conv_projection.layer1.norm.weight", {32}},
      {"audio_tower.subsample_conv_projection.input_proj_linear.weight",
       {1024, 1024}},
      {"audio_tower.output_proj.weight", {1536, 1024}},
      {"audio_tower.output_proj.bias", {1536}},
      {"embed_audio.embedding_projection.weight", {512, 1536}},
  };
  for (unsigned i = 0; i < 12; ++i) {
    const std::string p = "audio_tower.layers." + std::to_string(i) + ".";
    const auto linear = [&](const std::string& name, std::uint64_t out,
                            std::uint64_t in) {
      tensors.push_back({p + name + ".linear.weight", {out, in}});
      for (const auto* bound :
           {"input_min", "input_max", "output_min", "output_max"})
        tensors.push_back({p + name + "." + bound, {}});
    };
    for (const auto* ff : {"feed_forward1.", "feed_forward2."}) {
      linear(std::string(ff) + "ffw_layer_1", 4096, 1024);
      linear(std::string(ff) + "ffw_layer_2", 1024, 4096);
      for (const auto* norm :
           {"pre_layer_norm.weight", "post_layer_norm.weight"})
        tensors.push_back({p + ff + norm, {1024}});
    }
    for (const auto* proj : {"q_proj", "k_proj", "v_proj", "post"})
      linear(std::string("self_attn.") + proj, 1024, 1024);
    tensors.push_back({p + "self_attn.relative_k_proj.weight", {1024, 1024}});
    tensors.push_back({p + "self_attn.per_dim_scale", {128}});
    linear("lconv1d.linear_start", 2048, 1024);
    linear("lconv1d.linear_end", 1024, 1024);
    tensors.push_back({p + "lconv1d.depthwise_conv1d.weight", {1024, 1, 5}});
    for (const auto* norm :
         {"lconv1d.pre_layer_norm.weight", "lconv1d.conv_norm.weight",
          "norm_pre_attn.weight", "norm_post_attn.weight", "norm_out.weight"})
      tensors.push_back({p + norm, {1024}});
  }
  return tensors;
}

LoadResult LoadModelDirectory(const std::string& model_dir, bool include_vision,
                              bool include_audio) {
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
  if (!LoadSafetensors(path, result.store.get(), &result.mappings, &error,
                       include_vision, include_audio)) {
    result.error = "embedding_gemma2: " + error;
    return result;
  }
  auto expected_tensors = ExpectedTextTensors(result.config.text);
  if (include_vision) {
    if (!result.config.supported_vision) {
      result.error = "embedding_gemma2: unsupported vision configuration";
      return result;
    }
    auto vision = ExpectedVisionTensors();
    expected_tensors.insert(expected_tensors.end(), vision.begin(),
                            vision.end());
  }
  if (include_audio) {
    if (!result.config.supported_audio) {
      result.error = "embedding_gemma2: unsupported audio configuration";
      return result;
    }
    auto audio = ExpectedAudioTensors();
    expected_tensors.insert(expected_tensors.end(), audio.begin(), audio.end());
  }
  for (const ExpectedTensor& expected : expected_tensors) {
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
