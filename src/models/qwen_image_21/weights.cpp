#include "src/models/qwen_image_21/weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

#include "src/core/mapped_prefetch.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

std::size_t Integer(const json::Value& value) {
  const double number = value.as_double(-1);
  if (!value.is_number() || !std::isfinite(number) || number < 0 ||
      number > (1ULL << 40) ||
      number != static_cast<double>(static_cast<std::uint64_t>(number)))
    throw std::runtime_error("invalid Qwen-Image integer metadata");
  return static_cast<std::size_t>(number);
}

void Boolean(const json::Value& object, std::string_view key, bool expected) {
  const auto* value = object.find(std::string(key));
  if (!value || !value->is_bool() || value->as_bool() != expected)
    throw std::runtime_error("unsupported Qwen-Image configuration: " +
                             std::string(key));
}

void Equal(const json::Value& object, std::string_view key, double expected) {
  const auto* value = object.find(std::string(key));
  if (!value || !value->is_number() || value->as_double() != expected)
    throw std::runtime_error("unsupported Qwen-Image configuration: " +
                             std::string(key));
}

void ArrayEqual(const json::Value& object, std::string_view key,
                std::initializer_list<int> expected) {
  const auto* value = object.find(std::string(key));
  if (!value || !value->is_array() || value->size() != expected.size())
    throw std::runtime_error("invalid Qwen-Image array: " + std::string(key));
  std::size_t index = 0;
  for (int item : expected) {
    if (Integer(value->items()[index++]) != static_cast<std::size_t>(item))
      throw std::runtime_error("unsupported Qwen-Image array: " +
                               std::string(key));
  }
}

}  // namespace

json::Value ReadJson(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const auto size = in.tellg();
  if (!in || size <= 0 || size > (32 << 20))
    throw std::runtime_error("cannot read Qwen-Image metadata: " +
                             path.string());
  std::string data(static_cast<std::size_t>(size), '\0');
  in.seekg(0);
  if (!in.read(data.data(), size))
    throw std::runtime_error("truncated Qwen-Image metadata: " + path.string());
  return json::parse(data);
}

Weights::Weights(const std::filesystem::path& root) {
  const auto index = ReadJson(root / "model_index.json");
  if (index.member_str("_class_name") != "QwenImage21Pipeline")
    throw std::runtime_error("expected a Qwen-Image-2.1 Diffusers directory");
  const auto dit = ReadJson(root / "transformer/config.json");
  for (auto [key, value] : {std::pair{"num_layers", 32},
                            {"num_attention_heads", 32},
                            {"attention_head_dim", 128},
                            {"context_in_dim", 4096},
                            {"in_channels", 64},
                            {"out_channels", 64},
                            {"patch_size", 1},
                            {"mlp_ratio", 3}})
    Equal(dit, key, value);
  Equal(dit, "eps", 1e-6);
  ArrayEqual(dit, "axes_dims_rope", {16, 56, 56});
  if (!dit.find("causal_condition") ||
      !dit.find("causal_condition")->is_bool() ||
      !dit.find("causal_condition")->as_bool())
    throw std::runtime_error("Qwen-Image requires causal conditioning");
  const auto text = ReadJson(root / "text_encoder/config.json");
  if (text.member_str("model_type") != "qwen3_vl")
    throw std::runtime_error("expected Qwen3-VL image text encoder");
  const auto* tc = text.find("text_config");
  const auto* vc = text.find("vision_config");
  if (!tc || !vc)
    throw std::runtime_error("missing Qwen3-VL configuration");
  for (auto [key, value] : {std::pair{"hidden_size", 4096},
                            {"intermediate_size", 12288},
                            {"num_hidden_layers", 36},
                            {"num_attention_heads", 32},
                            {"num_key_value_heads", 8},
                            {"head_dim", 128},
                            {"vocab_size", 151936}})
    Equal(*tc, key, value);
  Equal(*tc, "rms_norm_eps", 1e-6);
  Equal(*tc, "rope_theta", 5000000);
  const auto* rope = tc->find("rope_scaling");
  if (!rope || rope->member_str("rope_type") != "default" ||
      !rope->find("mrope_interleaved") ||
      !rope->find("mrope_interleaved")->as_bool())
    throw std::runtime_error("unsupported Qwen3-VL rotary configuration");
  ArrayEqual(*rope, "mrope_section", {24, 20, 20});
  for (auto [key, value] : {std::pair{"hidden_size", 1152},
                            {"intermediate_size", 4304},
                            {"depth", 27},
                            {"num_heads", 16},
                            {"patch_size", 16},
                            {"spatial_merge_size", 2},
                            {"temporal_patch_size", 2},
                            {"num_position_embeddings", 2304},
                            {"out_hidden_size", 4096}})
    Equal(*vc, key, value);
  ArrayEqual(*vc, "deepstack_visual_indexes", {8, 16, 24});
  if (tc->member_str("hidden_act") != "silu" ||
      vc->member_str("hidden_act") != "gelu_pytorch_tanh")
    throw std::runtime_error("unsupported Qwen3-VL activation");
  Boolean(*tc, "attention_bias", false);
  vae_config_ = ReadJson(root / "vae/config.json");
  for (auto [key, value] : {std::pair{"z_dim", 64},
                            {"base_dim", 96},
                            {"decoder_base_dim", 144},
                            {"in_channels", 4},
                            {"out_channels", 4},
                            {"num_res_blocks", 2},
                            {"scale_factor_spatial", 16}})
    Equal(vae_config_, key, value);
  ArrayEqual(vae_config_, "dim_mult", {1, 2, 4, 8, 8});
  Boolean(vae_config_, "is_residual", true);
  const auto* temporal = vae_config_.find("temperal_downsample");
  const auto* attention = vae_config_.find("attn_scales");
  if (!temporal || !temporal->is_array() || temporal->size() != 4 ||
      !attention || !attention->is_array() || attention->size())
    throw std::runtime_error("unsupported VAE block layout");
  for (int i = 0; i < 4; ++i)
    if (!temporal->items()[i].is_bool() ||
        temporal->items()[i].as_bool() != (i != 0))
      throw std::runtime_error("unsupported VAE temporal layout");
  for (const char* key : {"latents_mean", "latents_std"}) {
    const auto* values = vae_config_.find(key);
    if (!values || !values->is_array() || values->size() != 64)
      throw std::runtime_error("invalid VAE latent normalization");
    for (const auto& value : values->items())
      if (!value.is_number() || !std::isfinite(value.as_double()) ||
          (std::string_view(key) == "latents_std" && value.as_double() <= 0))
        throw std::runtime_error("invalid VAE latent normalization");
  }
  const auto scheduler = ReadJson(root / "scheduler/scheduler_config.json");
  for (auto [key, value] : {std::pair{"base_image_seq_len", 256.0},
                            {"max_image_seq_len", 8192.0},
                            {"base_shift", 0.5},
                            {"max_shift", 0.9},
                            {"shift_terminal", 0.02},
                            {"num_train_timesteps", 1000.0}})
    Equal(scheduler, key, value);
  if (scheduler.member_str("time_shift_type") != "exponential" ||
      !scheduler.find("use_dynamic_shifting") ||
      !scheduler.find("use_dynamic_shifting")->as_bool())
    throw std::runtime_error("unsupported Qwen-Image flow schedule");
  for (const char* key :
       {"invert_sigmas", "stochastic_sampling", "use_beta_sigmas",
        "use_exponential_sigmas", "use_karras_sigmas"})
    Boolean(scheduler, key, false);
  const auto processor = ReadJson(root / "processor/preprocessor_config.json");
  for (auto [key, value] : {std::pair{"patch_size", 16},
                            {"temporal_patch_size", 2},
                            {"merge_size", 2}})
    Equal(processor, key, value);
  Equal(processor, "rescale_factor", 1.0 / 255.0);
  for (const char* key :
       {"do_convert_rgb", "do_normalize", "do_rescale", "do_resize"})
    Boolean(processor, key, true);
  for (const char* key : {"image_mean", "image_std"}) {
    const auto* values = processor.find(key);
    if (!values || !values->is_array() || values->size() != 3)
      throw std::runtime_error("invalid vision normalization");
    for (const auto& value : values->items())
      if (!value.is_number() || value.as_double() != 0.5)
        throw std::runtime_error("unsupported vision normalization");
  }
  for (const auto& [component, filename] :
       {std::pair{"text_encoder", "model.safetensors.index.json"},
        {"transformer", "diffusion_pytorch_model.safetensors.index.json"}}) {
    const auto manifest = ReadJson(root / component / filename);
    const auto* weight_map = manifest.find("weight_map");
    if (!weight_map || !weight_map->is_object())
      throw std::runtime_error("missing safetensors weight map");
    std::set<std::string> shards;
    for (const auto& [name, file] : weight_map->members()) {
      (void)name;
      if (!file.is_string() ||
          std::filesystem::path(file.str()).filename().string() != file.str())
        throw std::runtime_error("invalid safetensors shard path");
      shards.insert(file.str());
    }
    for (const auto& shard : shards)
      Load(root / component / shard, std::string(component) + ".");
    for (const auto& [name, file] : weight_map->members()) {
      (void)file;
      if (!Contains(std::string(component) + "." + name))
        throw std::runtime_error("missing safetensors tensor " + name);
    }
  }
  Load(root / "vae/diffusion_pytorch_model.safetensors", "vae.");
}

void Weights::Load(const std::filesystem::path& path,
                   const std::string& prefix) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  struct stat st{};
  if (fd < 0)
    throw std::runtime_error("cannot open " + path.string());
  if (fstat(fd, &st) != 0 || st.st_size < 8) {
    close(fd);
    throw std::runtime_error("invalid safetensors file " + path.string());
  }
  const auto size = static_cast<std::size_t>(st.st_size);
  void* data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (data == MAP_FAILED)
    throw std::runtime_error("cannot map " + path.string());
  auto owner =
      std::shared_ptr<void>(data, [size](void* p) { munmap(p, size); });
  std::uint64_t header_size;
  std::memcpy(&header_size, data, 8);
  if (!header_size || header_size > (32U << 20U) || header_size > size - 8)
    throw std::runtime_error("invalid safetensors header");
  const auto header = json::parse(
      std::string_view(static_cast<const char*>(data) + 8, header_size));
  if (!header.is_object())
    throw std::runtime_error("invalid safetensors header");
  const auto payload_size = size - 8 - header_size;
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (const auto& [name, entry] : header.members()) {
    if (name == "__metadata__")
      continue;
    const auto dtype = entry.member_str("dtype");
    if (dtype != "BF16" && dtype != "F32")
      throw std::runtime_error("unsupported Qwen-Image weight dtype " + dtype);
    const auto* shape = entry.find("shape");
    const auto* offsets = entry.find("data_offsets");
    if (!shape || !shape->is_array() || shape->size() > 5 || !offsets ||
        !offsets->is_array() || offsets->size() != 2)
      throw std::runtime_error("invalid tensor layout " + name);
    Weight weight;
    weight.bf16 = dtype == "BF16";
    std::size_t bytes = weight.bf16 ? 2 : 4;
    for (const auto& dim : shape->items()) {
      const auto value = Integer(dim);
      if (!value || value > std::numeric_limits<int>::max() ||
          bytes > payload_size / value)
        throw std::runtime_error("invalid tensor shape " + name);
      weight.shape.push_back(static_cast<int>(value));
      bytes *= value;
    }
    const auto begin = Integer(offsets->items()[0]);
    const auto end = Integer(offsets->items()[1]);
    if (begin > end || end > payload_size || end - begin != bytes)
      throw std::runtime_error("invalid tensor extent " + name);
    weight.data = static_cast<const char*>(data) + 8 + header_size + begin;
    weight.bytes = bytes;
    ranges.emplace_back(begin, end);
    if (!weights_.emplace(prefix + name, std::move(weight)).second)
      throw std::runtime_error("duplicate tensor " + name);
  }
  std::sort(ranges.begin(), ranges.end());
  std::size_t cursor = 0;
  for (auto [begin, end] : ranges) {
    if (begin != cursor)
      throw std::runtime_error("noncontiguous safetensors payload");
    cursor = end;
  }
  if (cursor != payload_size)
    throw std::runtime_error("unexpected safetensors data");
  mappings_.push_back(std::move(owner));
  regions_.emplace_back(static_cast<const std::byte*>(data), size);
}

void Weights::Prefetch(std::string_view prefix) const {
  for (const auto region : regions_) {
    const auto base = reinterpret_cast<std::uintptr_t>(region.data());
    std::size_t begin = region.size(), end = 0;
    for (auto it = weights_.lower_bound(prefix);
         it != weights_.end() && it->first.starts_with(prefix); ++it) {
      const auto address = reinterpret_cast<std::uintptr_t>(it->second.data);
      if (address < base || address - base >= region.size())
        continue;
      const auto offset = address - base;
      begin = std::min(begin, offset);
      end = std::max(end, offset + it->second.bytes);
    }
    if (end > begin)
      core::PrefaultMappedRange(region.data() + begin, end - begin);
  }
}

const Weight& Weights::Get(std::string_view name) const {
  const auto it = weights_.find(name);
  if (it == weights_.end())
    throw std::runtime_error("missing Qwen-Image tensor " + std::string(name));
  return it->second;
}

bool Weights::Contains(std::string_view name) const {
  return weights_.contains(name);
}

}  // namespace gufo::models::qwen_image_21
