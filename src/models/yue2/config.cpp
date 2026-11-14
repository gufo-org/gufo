#include "src/models/yue2/config.hpp"

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <gufo/ext/json.hpp>

namespace gufo::models::yue2 {
namespace {

std::optional<json::Value> ReadJson(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return std::nullopt;
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    json::Value value = json::parse(text);
    if (!value.is_object())
      return std::nullopt;
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

bool IsFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

bool IsModelDirectory(const std::filesystem::path& root) {
  const auto config = ReadJson(root / "config.json");
  const auto generation = ReadJson(root / "yue2_generation_config.json");
  return config && config->member_str("model_type") == "yue2" && generation &&
         generation->member_str("version") == kProtocolVersion &&
         IsFile(root / "qwen.tiktoken") && IsFile(root / "model.safetensors");
}

bool IsVaeDirectory(const std::filesystem::path& root) {
  const auto config = ReadJson(root / "config.json");
  return config && config->member_str("model_type") == "yue2_vae" &&
         IsFile(root / "model.safetensors");
}

// A saved pipeline: `pipeline.json` names the model and VAE subdirectories.
std::optional<ModelPaths> PipelinePaths(const std::filesystem::path& root) {
  const auto pipeline = ReadJson(root / "pipeline.json");
  if (!pipeline)
    return std::nullopt;
  const std::filesystem::path model = pipeline->member_str("model");
  const std::filesystem::path vae = pipeline->member_str("vae");
  // Only plain child names: the file must not point outside the root.
  if (model.empty() || vae.empty() || model.has_parent_path() ||
      vae.has_parent_path() || model == ".." || vae == "..")
    return std::nullopt;
  return ModelPaths{root / model, root / vae};
}

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

bool ReadSampling(const json::Value* value, Sampling* sampling,
                  std::string* error) {
  if (value == nullptr)
    return true;
  if (!value->is_object())
    return Fail(error, "generation sampling must be an object");
  sampling->temperature = static_cast<float>(
      value->member_double("temperature", sampling->temperature));
  sampling->top_p =
      static_cast<float>(value->member_double("top_p", sampling->top_p));
  sampling->top_k =
      static_cast<std::uint32_t>(value->member_size("top_k", sampling->top_k));
  sampling->repetition_penalty = static_cast<float>(
      value->member_double("repetition_penalty", sampling->repetition_penalty));
  sampling->penalty_window = static_cast<std::uint32_t>(
      value->member_size("penalty_window", sampling->penalty_window));
  sampling->min_tokens = static_cast<std::uint32_t>(
      value->member_size("min_tokens", sampling->min_tokens));
  sampling->max_tokens = static_cast<std::uint32_t>(
      value->member_size("max_tokens", sampling->max_tokens));
  return ValidateSampling(*sampling, error);
}

}  // namespace

bool LooksLikeYue2(const std::filesystem::path& model_root) {
  if (IsModelDirectory(model_root))
    return true;
  const auto pipeline = PipelinePaths(model_root);
  return pipeline && IsModelDirectory(pipeline->model);
}

std::optional<std::filesystem::path> FindModelDirectory(
    const std::filesystem::path& root) {
  if (IsModelDirectory(root))
    return root;
  if (auto pipeline = PipelinePaths(root);
      pipeline && IsModelDirectory(pipeline->model))
    return pipeline->model;
  return std::nullopt;
}

std::optional<ModelPaths> ResolveModelPaths(
    const std::filesystem::path& root, const std::filesystem::path& vae_root,
    std::string* error) {
  ModelPaths paths;
  if (IsModelDirectory(root)) {
    paths.model = root;
    paths.vae = root.parent_path() / "YuE2-Vae";
  } else if (auto pipeline = PipelinePaths(root);
             pipeline && IsModelDirectory(pipeline->model)) {
    paths = *pipeline;
  } else {
    Fail(error, "not a YuE2 model directory: " + root.string());
    return std::nullopt;
  }
  if (!vae_root.empty())
    paths.vae = vae_root;
  if (!IsVaeDirectory(paths.vae)) {
    Fail(error, "YuE2 decoder not found at " + paths.vae.string() +
                    "; pass --vae-model DIR");
    return std::nullopt;
  }
  return paths;
}

bool LoadBackboneConfig(const std::filesystem::path& model_root,
                        BackboneConfig* config, GenerationDefaults* defaults,
                        std::string* error) {
  const auto value = ReadJson(model_root / "config.json");
  if (!value || value->member_str("model_type") != "yue2")
    return Fail(error, "invalid YuE2 config.json");
  const BackboneConfig released;
  const auto size = [&](const char* key) { return value->member_size(key, 0); };
  if (size("hidden_size") != released.hidden ||
      size("num_hidden_layers") != released.layers ||
      size("num_attention_heads") != released.heads ||
      size("num_key_value_heads") != released.kv_heads ||
      size("head_dim") != released.head_dim ||
      size("intermediate_size") != released.intermediate ||
      size("vocab_size") != released.vocabulary ||
      size("max_position_embeddings") != released.context ||
      size("latent_dim") != released.latent_dim ||
      size("max_latent_frames") != released.max_latent_frames ||
      value->member_str("latent_type", "vae") != "vae")
    return Fail(error, "unsupported YuE2 backbone shape in config.json");
  *config = released;
  config->rms_epsilon =
      static_cast<float>(value->member_double("rms_norm_eps", 1e-6));
  config->rope_theta =
      static_cast<float>(value->member_double("rope_theta", 1000000.0));
  config->timestep_shift =
      static_cast<float>(value->member_double("timestep_shift", 1.0));
  if (!std::isfinite(config->timestep_shift) || config->timestep_shift <= 0.0F)
    return Fail(error, "invalid YuE2 timestep_shift");

  const auto generation = ReadJson(model_root / "yue2_generation_config.json");
  if (!generation || generation->member_str("version") != kProtocolVersion)
    return Fail(error, "yue2_generation_config.json must be version " +
                           std::string(kProtocolVersion));
  if (generation->member_str("ode_method", "midpoint") != "midpoint" ||
      generation->member_size("context", kContext) != kContext)
    return Fail(error, "YuE2 generation requires midpoint and context 24576");
  *defaults = GenerationDefaults{};
  defaults->ode_steps =
      static_cast<int>(generation->member_size("ode_steps", 32));
  if (defaults->ode_steps < 1 || defaults->ode_steps > 1000)
    return Fail(error, "invalid YuE2 ode_steps");
  return ReadSampling(generation->find("abc"), &defaults->abc, error) &&
         ReadSampling(generation->find("semantic"), &defaults->semantic, error);
}

bool LoadVaeConfig(const std::filesystem::path& vae_root, VaeConfig* config,
                   std::string* error) {
  const auto value = ReadJson(vae_root / "config.json");
  if (!value || value->member_str("model_type") != "yue2_vae")
    return Fail(error, "invalid YuE2 VAE config.json");
  VaeConfig released;
  const json::Value* decoder = value->find("decoder_config");
  if (decoder == nullptr || !decoder->is_object())
    return Fail(error, "YuE2 VAE config has no decoder_config");
  const auto list = [](const json::Value* array) {
    std::vector<int> result;
    if (array != nullptr && array->is_array()) {
      for (const auto& item : array->items())
        result.push_back(static_cast<int>(item.as_size(0)));
    }
    return result;
  };
  const std::vector<int> multipliers = list(decoder->find("c_mults"));
  const std::vector<int> strides = list(decoder->find("strides"));
  const json::Value* snake = decoder->find("use_snake");
  const json::Value* tanh = decoder->find("final_tanh");
  if (multipliers != std::vector<int>{1, 2, 4, 8, 16, 32} ||
      strides != std::vector<int>{2, 2, 4, 4, 5, 6} ||
      decoder->member_size("channels") != 64 ||
      decoder->member_size("latent_dim") != 64 ||
      decoder->member_size("out_channels") != 2 || snake == nullptr ||
      !snake->is_bool() || !snake->as_bool() ||
      decoder->member_str("snake_type", "vanilla") != "vanilla" ||
      tanh == nullptr || !tanh->is_bool() || tanh->as_bool() ||
      value->member_size("sample_rate") != 48000 ||
      value->member_size("downsampling_ratio") != 1920)
    return Fail(error, "unsupported YuE2 VAE decoder layout");
  for (const char* key :
       {"use_filter", "antialias_activation", "use_nearest_upsample"}) {
    if (const json::Value* flag = decoder->find(key);
        flag != nullptr && flag->is_bool() && flag->as_bool())
      return Fail(error, std::string("unsupported YuE2 VAE option ") + key);
  }
  *config = released;
  config->release_variant = value->member_str("release_variant", "standard");
  config->core_frames =
      static_cast<int>(value->member_size("decode_core_frames", 1024));
  config->halo_frames =
      static_cast<int>(value->member_size("decode_halo_frames", 16));
  if (config->core_frames < 1 || config->core_frames > 8192 ||
      config->halo_frames < 16)
    return Fail(error, "invalid YuE2 VAE tiling");
  return true;
}

}  // namespace gufo::models::yue2
