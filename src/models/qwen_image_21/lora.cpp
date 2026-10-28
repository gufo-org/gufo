// Qwen-Image-2.1 transformer adapters, following ComfyUI's conventions
// (comfy/lora.py, comfy/weight_adapter/{lora,base}.py at 54a4e56a):
//   pair:  W += strength * scale * up @ down, scale = alpha / rank, where
//          alpha is the module's `.alpha`, else diffusers' metadata
//          `lora_alpha` (`use_rslora`: alpha / sqrt(rank)), else rank;
//   DoRA:  W_c = W + scale * up @ down; W_c *= m / (||W row|| + FLT_EPSILON)
//          with the row norm of W before the patch; W += strength*(W_c - W);
//   diff:  W += strength * diff.
// A fused `img_mlp.gate_up` patches rows [0, H) of gate_layer and [H, 2H) of
// proj with a shared down matrix.
#include "src/models/qwen_image_21/lora.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

enum Role { kDown, kUp, kAlpha, kMagnitude, kDiff, kRoles };

struct Suffix {
  std::string_view text;
  Role role;
  std::string_view layout;
};

// First match wins.
constexpr Suffix kSuffixes[] = {
    {".lora_A.default.weight", kDown, "peft-default"},
    {".lora_B.default.weight", kUp, "peft-default"},
    {".lora_A.weight", kDown, "peft"},
    {".lora_B.weight", kUp, "peft"},
    {".lora_down.weight", kDown, "down-up"},
    {".lora_up.weight", kUp, "down-up"},
    {".lora.down.weight", kDown, "diffusers-legacy"},
    {".lora.up.weight", kUp, "diffusers-legacy"},
    {".lora_A", kDown, "peft-bare"},
    {".lora_B", kUp, "peft-bare"},
    {".lora_down", kDown, "down-up-bare"},
    {".lora_up", kUp, "down-up-bare"},
    {".alpha", kAlpha, ""},
    {".dora_scale", kMagnitude, ""},
    {".diff", kDiff, ""}};

constexpr std::string_view kLayouts[] = {
    "peft",    "peft-default", "peft-bare",       "kohya",
    "down-up", "down-up-bare", "diffusers-legacy"};

constexpr std::string_view kPrefixes[] = {
    "diffusion_model.transformer.", "base_model.model.transformer.",
    "base_model.model.", "diffusion_model.", "transformer."};

constexpr std::string_view kBlockModules[] = {
    "attn.to_q",    "attn.to_k",   "attn.to_v",       "attn.to_out.0",
    "attn.norm_q",  "attn.norm_k", "img_mlp.gate_up", "img_mlp.gate_layer",
    "img_mlp.proj", "img_mlp.out"};

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

std::optional<std::size_t> Integer(const json::Value& value) {
  const double number = value.as_double(-1);
  if (!value.is_number() || !std::isfinite(number) || number < 0 ||
      number > (1ULL << 40) ||
      number != static_cast<double>(static_cast<std::uint64_t>(number)))
    return std::nullopt;
  return static_cast<std::size_t>(number);
}

[[noreturn]] void Fail(const std::filesystem::path& path, std::string_view key,
                       std::string_view reason) {
  throw std::runtime_error(
      path.string() + ": " +
      (key.empty() ? std::string() : std::string(key) + ": ") +
      std::string(reason));
}

std::size_t Elements(const LoraTensor& tensor) {
  std::size_t count = 1;
  for (int dim : tensor.shape)
    count *= static_cast<std::size_t>(dim);
  return count;
}

std::string Shape(const std::vector<int>& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    text += (i ? "," : "") + std::to_string(shape[i]);
  return text + "]";
}

std::string Underscored(std::string text) {
  std::replace(text.begin(), text.end(), '.', '_');
  return text;
}

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

struct Module {
  std::array<std::string, kRoles> keys;
  std::array<LoraTensor, kRoles> tensors;
};

}  // namespace

std::vector<float> LoraFloats(const LoraTensor& tensor, std::size_t first,
                              std::size_t count) {
  if (first > Elements(tensor) || count > Elements(tensor) - first)
    throw std::out_of_range("adapter tensor range out of bounds");
  std::vector<float> values(count);
  const auto* bytes = static_cast<const char*>(tensor.data);
  if (tensor.dtype == LoraDtype::kF32) {
    std::memcpy(values.data(), bytes + first * 4, count * 4);
    return values;
  }
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t value;
    std::memcpy(&value, bytes + (first + i) * 2, 2);
    values[i] = tensor.dtype == LoraDtype::kBf16
                    ? std::bit_cast<float>(std::uint32_t(value) << 16)
                    : gufo::quant::Fp16ToFloat(value);
  }
  return values;
}

LoraAdapter::LoraAdapter(const std::filesystem::path& path,
                         const BaseShape& base) {
  auto mapped = std::make_unique<platform::RawMappedFile>();
  std::string open_error;
  if (!mapped->Open(path, platform::MapHint::kNone, &open_error))
    Fail(path, "",
         open_error.empty() ? "invalid safetensors file"
                            : "cannot open: " + open_error);
  const auto size = mapped->size;
  void* data = mapped->data;
  // Owns the mapping from here, so every later failure unmaps it.
  mapping_ = std::shared_ptr<void>(data, [file = mapped.release()](void*) {
    file->Close();
    delete file;
  });
  if (size < 8)
    Fail(path, "", "invalid safetensors file");
  std::uint64_t header_size;
  std::memcpy(&header_size, data, 8);
  if (!header_size || header_size > (32U << 20U) || header_size > size - 8)
    Fail(path, "", "invalid safetensors header");
  json::Value header;
  try {
    header = json::parse(
        std::string_view(static_cast<const char*>(data) + 8, header_size));
  } catch (const std::exception&) {
    Fail(path, "", "invalid safetensors header");
  }
  if (!header.is_object())
    Fail(path, "", "invalid safetensors header");
  const auto payload_size = size - 8 - header_size;
  const char* payload = static_cast<const char*>(data) + 8 + header_size;

  int blocks = 0;
  while (!base("transformer.transformer_blocks." + std::to_string(blocks) +
               ".attn.to_q.weight")
              .empty())
    ++blocks;
  std::set<std::string, std::less<>> canonical;
  std::map<std::string, std::string, std::less<>> kohya;
  for (int b = 0; b < blocks; ++b)
    for (auto module : kBlockModules)
      canonical.insert("transformer_blocks." + std::to_string(b) + "." +
                       std::string(module));
  for (auto module : kGlobalModules)
    canonical.emplace(module);
  for (const auto& module : canonical)
    kohya.emplace("lora_unet_" + Underscored(module), module);
  const auto block_error = [&](std::string_view key, long index) {
    Fail(path, key,
         "block " + std::to_string(index) +
             " does not exist: Qwen-Image 1.x/Edit adapters (60 blocks) "
             "do not fit Qwen-Image-2.1's " +
             std::to_string(blocks) + " blocks");
  };

  std::map<std::string, Module, std::less<>> modules;
  std::set<std::string_view> layouts;
  std::set<std::string, std::less<>> keys;
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  const json::Value* metadata = nullptr;
  for (const auto& [key, entry] : header.members()) {
    if (key == "__metadata__") {
      metadata = &entry;
      continue;
    }
    if (!keys.insert(key).second)
      Fail(path, key, "duplicate tensor");
    for (auto text : {"lora_te", "text_encoder", "te1_", "te2_"})
      if (key.find(text) != std::string::npos)
        Fail(path, key, "text-encoder adapters are unsupported");
    for (auto text : {"hada_", "lokr_", "oft_", ".lora_mid", "boft"})
      if (key.find(text) != std::string::npos)
        Fail(path, key, "LyCORIS LoHa/LoKr/OFT/LoCon adapters are unsupported");
    if (key.find(".diff_b") != std::string::npos)
      Fail(path, key, "bias deltas are unsupported");
    if (key.ends_with(".bias"))
      Fail(path, key,
           "unsupported bias tensor (bias delta, full replacement or unknown "
           "layout)");
    for (auto text : {".set_weight", ".w_norm", ".b_norm", ".reshape_weight"})
      if (key.find(text) != std::string::npos)
        Fail(path, key, "unsupported adapter tensor");

    // Tensor extent, as in Weights::Load; keep both in step. Not shared so
    // the base loader stays untouched: this one also accepts F16, reports
    // the offending key and leaves the payload check to after the loop.
    const auto dtype = entry.member_str("dtype");
    LoraTensor tensor;
    std::size_t bytes = 2;
    if (dtype == "BF16") {
      tensor.dtype = LoraDtype::kBf16;
    } else if (dtype == "F16") {
      tensor.dtype = LoraDtype::kF16;
    } else if (dtype == "F32") {
      tensor.dtype = LoraDtype::kF32;
      bytes = 4;
    } else {
      Fail(path, key, "unsupported adapter dtype " + dtype);
    }
    const auto* shape = entry.find("shape");
    const auto* offsets = entry.find("data_offsets");
    if (!shape || !shape->is_array() || shape->size() > 5 || !offsets ||
        !offsets->is_array() || offsets->size() != 2)
      Fail(path, key, "invalid tensor layout");
    for (const auto& dim : shape->items()) {
      const auto value = Integer(dim).value_or(0);
      if (!value || value > std::numeric_limits<int>::max() ||
          bytes > payload_size / value)
        Fail(path, key, "invalid tensor shape");
      tensor.shape.push_back(static_cast<int>(value));
      bytes *= value;
    }
    const auto begin = Integer(offsets->items()[0]);
    const auto end = Integer(offsets->items()[1]);
    if (!begin || !end || *begin > *end || *end > payload_size ||
        *end - *begin != bytes)
      Fail(path, key, "invalid tensor extent");
    tensor.data = payload + *begin;
    ranges.emplace_back(*begin, *end);

    const Suffix* suffix = nullptr;
    for (const auto& candidate : kSuffixes)
      if (key.ends_with(candidate.text)) {
        suffix = &candidate;
        break;
      }
    if (!suffix)
      Fail(path, key,
           "unsupported adapter tensor (full replacement weights or unknown "
           "layout)");
    std::string_view module(key);
    module.remove_suffix(suffix->text.size());
    for (auto prefix : kPrefixes)
      if (module.starts_with(prefix)) {
        module.remove_prefix(prefix.size());
        break;
      }
    std::string_view layout = suffix->layout;
    if (module.starts_with("lora_unet_")) {
      const auto it = kohya.find(module);
      if (it == kohya.end()) {
        const auto index = BlockIndex(module, "lora_unet_transformer_blocks_");
        if (index >= blocks)
          block_error(key, index);
        Fail(path, key, "unknown kohya module");
      }
      module = it->second;
      if (!layout.empty())
        layout = "kohya";
    } else if (!canonical.contains(module)) {
      const auto index = BlockIndex(module, "transformer_blocks.");
      if (index >= blocks)
        block_error(key, index);
      Fail(path, key, "not a Qwen-Image-2.1 transformer module");
    }
    if (!layout.empty())
      layouts.insert(layout);
    auto& slot = modules[std::string(module)];
    if (!slot.keys[suffix->role].empty())
      Fail(path, key,
           "duplicate adapter tensor (also " + slot.keys[suffix->role] + ")");
    slot.keys[suffix->role] = key;
    slot.tensors[suffix->role] = std::move(tensor);
  }
  std::sort(ranges.begin(), ranges.end());
  std::size_t cursor = 0;
  for (auto [begin, end] : ranges) {
    if (begin != cursor)
      Fail(path, "", "noncontiguous safetensors payload");
    cursor = end;
  }
  if (cursor != payload_size)
    Fail(path, "", "unexpected safetensors data");

  // diffusers/PEFT `lora_adapter_metadata`; other metadata alphas are
  // ignored like ComfyUI and diffusers do.
  std::optional<double> metadata_alpha;
  bool rslora = false;
  if (metadata && metadata->is_object())
    if (const auto* text = metadata->find("lora_adapter_metadata")) {
      json::Value config;
      try {
        config = json::parse(text->str());
      } catch (const std::exception&) {
        Fail(path, "__metadata__", "invalid lora_adapter_metadata");
      }
      const auto find = [&](const std::string& name) {
        const auto* value = config.find("transformer." + name);
        return value ? value : config.find(name);
      };
      for (const char* name : {"alpha_pattern", "rank_pattern"})
        if (const auto* value = find(name);
            value && !value->is_null() &&
            !(value->is_object() && value->empty()))
          Fail(path, "__metadata__", std::string(name) + " is unsupported");
      if (const auto* value = find("use_dora"); value && value->as_bool())
        Fail(path, "__metadata__", "use_dora metadata is unsupported");
      if (const auto* value = find("lora_alpha"); value && !value->is_null()) {
        if (!value->is_number() || !std::isfinite(value->as_double()))
          Fail(path, "__metadata__", "invalid lora_alpha");
        metadata_alpha = value->as_double();
      }
      if (const auto* value = find("use_rslora"); value && value->as_bool()) {
        if (!metadata_alpha)
          Fail(path, "__metadata__", "use_rslora without lora_alpha");
        rslora = true;
      }
    }

  bool alpha = false, used_metadata = false, diff = false, dora = false,
       gate_up = false;
  std::set<std::string, std::less<>> targets;
  for (const auto& [module, slot] : modules) {
    const auto& t = slot.tensors;
    const auto has = [&](Role role) { return t[role].data != nullptr; };
    const auto& first_key = *std::find_if(slot.keys.begin(), slot.keys.end(),
                                          [](auto& k) { return !k.empty(); });
    for (int role = 0; role < kRoles; ++role) {
      if (!has(Role(role)))
        continue;
      const std::size_t count = Elements(t[role]);
      for (std::size_t offset = 0; offset < count; offset += 1 << 20)
        for (float value :
             LoraFloats(t[role], offset,
                        std::min<std::size_t>(1 << 20, count - offset)))
          if (!std::isfinite(value))
            Fail(path, slot.keys[role], "non-finite value");
    }
    if (has(kDown) != has(kUp))
      Fail(path, first_key,
           has(kDown) ? "lora_down without lora_up"
                      : "lora_up without lora_down");
    const bool pair = has(kDown);
    if (pair && has(kDiff))
      Fail(path, slot.keys[kDiff], "both a low-rank pair and a diff");
    if (!pair && !has(kDiff))
      Fail(path, first_key, "alpha/dora_scale without a pair");
    if (!pair && has(kMagnitude))
      Fail(path, slot.keys[kMagnitude], "alpha/dora_scale without a pair");
    if (!pair && has(kAlpha))
      Fail(path, slot.keys[kAlpha], "alpha/dora_scale without a pair");

    // Targets and their common base shape.
    std::vector<std::string> names;
    const bool fused = module.ends_with("img_mlp.gate_up");
    const std::string stem = module.substr(0, module.size() - 7);
    if (fused) {
      names = {"transformer." + stem + "gate_layer.weight",
               "transformer." + stem + "proj.weight"};
      gate_up = true;
    } else {
      names = {"transformer." + module + ".weight"};
    }
    std::vector<int> shape;
    for (const auto& name : names) {
      const auto found = base(name);
      if (found.empty())
        Fail(path, first_key, name + " is missing from the base model");
      if (!shape.empty() && found != shape)
        Fail(path, first_key, "gate_layer and proj shapes differ");
      shape = found;
      if (!targets.insert(name).second)
        Fail(path, first_key, name + " is patched by several modules");
    }
    if (shape.size() > 2)
      Fail(path, first_key, "unsupported base tensor " + Shape(shape));
    const int rows = shape[0];
    const int cols = shape.size() == 2 ? shape[1] : 1;
    const int all_rows = rows * static_cast<int>(names.size());

    LoraPatch patch;
    patch.rows = rows;
    patch.cols = cols;
    if (pair) {
      const auto& down = t[kDown].shape;
      const auto& up = t[kUp].shape;
      if (shape.size() != 2)
        Fail(path, first_key, "low-rank pair on a 1-D tensor");
      if (down.size() != 2)
        Fail(path, slot.keys[kDown],
             "lora_down must be 2-D, got " + Shape(down));
      if (up.size() != 2)
        Fail(path, slot.keys[kUp], "lora_up must be 2-D, got " + Shape(up));
      if (up[1] != down[0])
        Fail(path, slot.keys[kUp],
             "rank mismatch: lora_up " + Shape(up) + ", lora_down " +
                 Shape(down));
      if (down[1] != cols || up[0] != all_rows)
        Fail(path, slot.keys[kUp],
             "shape " + Shape(up) + " x " + Shape(down) +
                 " does not match base " + Shape(shape) +
                 (fused ? " (fused gate_up needs 2x rows)" : ""));
      patch.rank = down[0];
      patch.down = t[kDown];
      patch.up = t[kUp];
      if (has(kAlpha)) {
        if (Elements(t[kAlpha]) != 1)
          Fail(path, slot.keys[kAlpha], "alpha must have one element");
        patch.scale = LoraFloats(t[kAlpha], 0, 1)[0] / patch.rank;
        alpha = true;
      } else if (metadata_alpha) {
        patch.scale = static_cast<float>(
            *metadata_alpha /
            (rslora ? std::sqrt(double(patch.rank)) : patch.rank));
        used_metadata = true;
      }
      if (has(kMagnitude)) {
        const auto& m = t[kMagnitude].shape;
        if (m != std::vector{all_rows, 1} && m != std::vector{all_rows})
          Fail(path, slot.keys[kMagnitude],
               m == std::vector{1, cols} || m == std::vector{cols}
                   ? "input-axis dora_scale is unsupported"
                   : "dora_scale shape " + Shape(m) + " does not match " +
                         Shape(shape));
        patch.magnitude = t[kMagnitude];
        dora = true;
      }
    } else {
      auto expected = shape;
      expected[0] = all_rows;
      if (t[kDiff].shape != expected)
        Fail(path, slot.keys[kDiff],
             "diff shape " + Shape(t[kDiff].shape) + " does not match base " +
                 Shape(expected));
      patch.diff = t[kDiff];
      diff = true;
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      patch.target = names[i];
      patch.first_row = static_cast<int>(i) * rows;
      patches_.push_back(patch);
    }
  }
  std::sort(patches_.begin(), patches_.end(),
            [](const auto& a, const auto& b) { return a.target < b.target; });

  for (auto layout : kLayouts)
    if (layouts.contains(layout))
      format_ += (format_.empty() ? "" : "+") + std::string(layout);
  for (auto [used, name] : {std::pair{alpha, "alpha"},
                            {used_metadata, "metadata-alpha"},
                            {diff, "diff"},
                            {dora, "dora"},
                            {gate_up, "gate_up"}})
    if (used)
      format_ += (format_.empty() ? "" : "+") + std::string(name);
  if (patches_.empty())
    Fail(path, "", "no adapter tensors");
}

int LoraAdapter::min_rank() const {
  int rank = 0;
  for (const auto& patch : patches_)
    if (patch.rank && (!rank || patch.rank < rank))
      rank = patch.rank;
  return rank;
}

int LoraAdapter::max_rank() const {
  int rank = 0;
  for (const auto& patch : patches_)
    rank = std::max(rank, patch.rank);
  return rank;
}

}  // namespace gufo::models::qwen_image_21
