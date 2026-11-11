// Qwen-Image-2.1 transformer modules for the shared adapter parser
// (src/core/lora/adapter.cpp). A fused `img_mlp.gate_up` patches every row
// of gate_layer, then of proj, with a shared down matrix (ComfyUI order).
#include "src/models/qwen_image_21/lora.hpp"

#include <string>

namespace gufo::models::qwen_image_21 {
namespace {

constexpr std::string_view kPrefixes[] = {
    "diffusion_model.transformer.", "base_model.model.transformer.",
    "base_model.model.", "diffusion_model.", "transformer."};

constexpr std::string_view kBlockModules[] = {
    "attn.to_q",          "attn.to_k",    "attn.to_v",
    "attn.to_out.0",      "attn.norm_q",  "attn.norm_k",
    "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"};

constexpr std::string_view kGlobalModules[] = {
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "txt_in.text_norm",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
    "proj_out"};

// Block index of "<prefix><digits>[._]...", or -1.
long BlockIndex(std::string_view module, std::string_view prefix) {
  if (!module.starts_with(prefix))
    return -1;
  module.remove_prefix(prefix.size());
  long index = 0;
  std::size_t digits = 0;
  while (digits < module.size() && digits < 9 && module[digits] >= '0' &&
         module[digits] <= '9')
    index = index * 10 + (module[digits++] - '0');
  return digits ? index : -1;
}

lora::Schema Schema(const LoraAdapter::BaseShape& base) {
  lora::Schema schema;
  schema.base = base;
  schema.prefixes.assign(std::begin(kPrefixes), std::end(kPrefixes));
  const auto whole = [](const std::string& module) {
    return std::vector<lora::Slice>{{"transformer." + module + ".weight"}};
  };
  int blocks = 0;
  while (!base("transformer.transformer_blocks." + std::to_string(blocks) +
               ".attn.to_q.weight")
              .empty())
    ++blocks;
  for (int b = 0; b < blocks; ++b) {
    const auto block = "transformer_blocks." + std::to_string(b) + ".";
    for (auto module : kBlockModules)
      schema.modules.emplace(block + std::string(module),
                             whole(block + std::string(module)));
    schema.modules.emplace(
        block + "img_mlp.gate_up",
        std::vector<lora::Slice>{
            {"transformer." + block + "img_mlp.gate_layer.weight"},
            {"transformer." + block + "img_mlp.proj.weight"}});
  }
  for (auto module : kGlobalModules)
    schema.modules.emplace(std::string(module), whole(std::string(module)));
  schema.unknown = [blocks](std::string_view module) -> std::string {
    auto index = BlockIndex(module, "lora_unet_transformer_blocks_");
    if (index < 0)
      index = BlockIndex(module, "transformer_blocks.");
    if (index >= blocks)
      return "block " + std::to_string(index) +
             " does not exist: Qwen-Image 1.x/Edit adapters (60 blocks) "
             "do not fit Qwen-Image-2.1's " +
             std::to_string(blocks) + " blocks";
    if (module.starts_with("lora_unet_"))
      return {};
    return "not a Qwen-Image-2.1 transformer module";
  };
  return schema;
}

}  // namespace

LoraAdapter::LoraAdapter(const std::filesystem::path& path,
                         const BaseShape& base)
    : lora::Adapter(path, Schema(base)) {}

}  // namespace gufo::models::qwen_image_21
