#include "src/models/qwen_image_21/lora.hpp"

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace gufo::models::qwen_image_21;
namespace fs = std::filesystem;

struct Tensor {
  std::string name;
  std::vector<int> shape;
  std::string dtype{"BF16"};
  float fill{0.25F};  // element i holds fill * ((i % 7) - 3)
};

float Value(const Tensor& tensor, std::size_t i) {
  return tensor.fill * static_cast<float>(static_cast<int>(i % 7) - 3);
}

// IEEE binary16 by definition (portable stand-in for _Float16).
float FromHalf(std::uint16_t bits) {
  const int exponent = (bits >> 10) & 0x1f, mantissa = bits & 0x3ff;
  const double sign = bits & 0x8000 ? -1 : 1;
  if (exponent == 0x1f)
    return mantissa ? std::nanf("") : static_cast<float>(sign * INFINITY);
  return static_cast<float>(
      sign * (exponent ? std::ldexp(1024 + mantissa, exponent - 25)
                       : std::ldexp(mantissa, -24)));
}

// Exact for the fixture values (normal halves), infinity on overflow.
std::uint16_t ToHalf(float value) {
  const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  if (value == 0)
    return sign;
  if (std::fabs(value) >= 65520)
    return sign | 0x7c00;
  int exponent;
  const double fraction = std::frexp(std::fabs(value), &exponent);
  const double mantissa = (fraction * 2 - 1) * 1024;
  assert(exponent >= -13 && mantissa == std::floor(mantissa));
  return sign | static_cast<std::uint16_t>((exponent + 14) << 10) |
         static_cast<std::uint16_t>(mantissa);
}

fs::path Directory() {
  static const fs::path dir = [] {
    auto path = fs::temp_directory_path() /
                ("gufo_lora_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(path);
    return path;
  }();
  return dir;
}

// Tiny safetensors writer; `trim` drops payload bytes, `header_size`
// overrides the 8-byte length prefix.
fs::path Write(const std::string& file, const std::vector<Tensor>& tensors,
               const std::string& metadata = "", std::size_t trim = 0,
               std::uint64_t header_size = 0) {
  std::string header = "{", payload;
  if (!metadata.empty())
    header += "\"__metadata__\":" + metadata;
  for (const auto& tensor : tensors) {
    std::size_t count = 1;
    for (int dim : tensor.shape)
      count *= static_cast<std::size_t>(dim);
    const auto begin = payload.size();
    for (std::size_t i = 0; i < count; ++i) {
      const float value = Value(tensor, i);
      if (tensor.dtype == "F32") {
        payload.append(reinterpret_cast<const char*>(&value), 4);
      } else if (tensor.dtype == "F64") {
        const double wide = value;
        payload.append(reinterpret_cast<const char*>(&wide), 8);
      } else if (tensor.dtype == "F16") {
        const auto half = ToHalf(value);
        payload.append(reinterpret_cast<const char*>(&half), 2);
      } else {
        const auto bf16 = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(value) >> 16);
        payload.append(reinterpret_cast<const char*>(&bf16), 2);
      }
    }
    if (header.size() > 1)
      header += ",";
    header += "\"" + tensor.name + "\":{\"dtype\":\"" + tensor.dtype +
              "\",\"shape\":[";
    for (std::size_t i = 0; i < tensor.shape.size(); ++i)
      header += (i ? "," : "") + std::to_string(tensor.shape[i]);
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," +
              std::to_string(payload.size()) + "]}";
  }
  header += "}";
  const auto path = Directory() / file;
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t size = header_size ? header_size : header.size();
  out.write(reinterpret_cast<const char*>(&size), 8);
  out << header << payload.substr(0, payload.size() - trim);
  return path;
}

std::vector<int> Base(std::string_view name) {
  static const std::map<std::string, std::vector<int>, std::less<>> shapes =
      [] {
        std::map<std::string, std::vector<int>, std::less<>> map;
        for (int b = 0; b < 2; ++b) {
          const auto p = "transformer.transformer_blocks." + std::to_string(b);
          for (const char* m : {"to_q", "to_k", "to_v", "to_out.0"})
            map[p + ".attn." + m + ".weight"] = {8, 8};
          map[p + ".attn.norm_q.weight"] = {4};
          map[p + ".attn.norm_k.weight"] = {4};
          map[p + ".img_mlp.gate_layer.weight"] = {12, 8};
          map[p + ".img_mlp.proj.weight"] = {12, 8};
          map[p + ".img_mlp.out.weight"] = {8, 12};
        }
        map["transformer.img_in.weight"] = {8, 4};
        map["transformer.txt_in.in_layer.weight"] = {8, 8};
        map["transformer.txt_in.out_layer.weight"] = {8, 8};
        map["transformer.txt_in.text_norm.weight"] = {8};
        map["transformer.time_text_embed.timestep_embedder.linear_1.weight"] = {
            8, 6};
        map["transformer.time_text_embed.timestep_embedder.linear_2.weight"] = {
            8, 8};
        map["transformer.modulation.1.weight"] = {32, 8};
        map["transformer.norm_out.linear.weight"] = {8, 8};
        map["transformer.proj_out.weight"] = {4, 8};
        return map;
      }();
  const auto it = shapes.find(name);
  return it == shapes.end() ? std::vector<int>{} : it->second;
}

// Low-rank pair of `module` [out, in] spelled with `down`/`up` suffixes.
std::vector<Tensor> Pair(const std::string& module, const std::string& down,
                         const std::string& up, int rank, int out, int in,
                         const std::string& dtype = "BF16") {
  return {{module + down, {rank, in}, dtype},
          {module + up, {out, rank}, dtype}};
}

std::vector<Tensor> Join(std::initializer_list<std::vector<Tensor>> parts) {
  std::vector<Tensor> all;
  for (const auto& part : parts)
    all.insert(all.end(), part.begin(), part.end());
  return all;
}

const LoraPatch& Find(const LoraAdapter& adapter, std::string_view target) {
  for (const auto& patch : adapter.patches())
    if (patch.target == target)
      return patch;
  std::cerr << "missing patch " << target << "\n";
  std::abort();
}

void Rejects(const fs::path& path, const std::string& expected) {
  try {
    LoraAdapter adapter(path, Base);
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()).find(expected) != std::string::npos)
      return;
    std::cerr << "unexpected error: " << error.what() << "\n";
    std::abort();
  }
  std::cerr << "accepted " << path << ", expected " << expected << "\n";
  std::abort();
}

constexpr const char* kQ = "transformer_blocks.0.attn.to_q";
constexpr const char* kQTarget =
    "transformer.transformer_blocks.0.attn.to_q.weight";

}  // namespace

int main() {
  // F16 conversion of every bit pattern.
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    const auto half = static_cast<std::uint16_t>(bits);
    const LoraTensor tensor{&half, LoraDtype::kF16, {1}};
    const float value = LoraFloats(tensor, 0, 1)[0];
    const float expected = FromHalf(half);
    assert(std::isnan(expected) ? std::isnan(value)
                                : std::bit_cast<std::uint32_t>(value) ==
                                      std::bit_cast<std::uint32_t>(expected));
  }

  // PEFT with every prefix.
  for (std::string prefix :
       {"diffusion_model.", "transformer.", "base_model.model.",
        "diffusion_model.transformer.", ""}) {
    const LoraAdapter adapter(
        Write("peft.safetensors",
              Pair(prefix + kQ, ".lora_A.weight", ".lora_B.weight", 2, 8, 8)),
        Base);
    assert(adapter.format() == "peft");
    assert(adapter.patches().size() == 1);
    const auto& patch = adapter.patches()[0];
    assert(patch.target == kQTarget);
    assert(patch.rows == 8 && patch.cols == 8 && patch.rank == 2);
    assert(patch.first_row == 0 && patch.scale == 1);
    assert(patch.down.shape == (std::vector{2, 8}));
    assert(patch.up.shape == (std::vector{8, 2}));
    assert(!patch.diff.data && !patch.magnitude.data);
    const auto values = LoraFloats(patch.down, 0, 16);
    for (std::size_t i = 0; i < values.size(); ++i)
      assert(values[i] == 0.25F * (static_cast<int>(i % 7) - 3));
    assert(adapter.min_rank() == 2 && adapter.max_rank() == 2);
  }

  // Other layouts, one per file.
  for (auto [down, up, format] :
       {std::tuple{".lora_A.default.weight", ".lora_B.default.weight",
                   "peft-default"},
        {".lora_A", ".lora_B", "peft-bare"},
        {".lora_down", ".lora_up", "down-up-bare"},
        {".lora.down.weight", ".lora.up.weight", "diffusers-legacy"},
        {".lora_down.weight", ".lora_up.weight", "down-up"}}) {
    const LoraAdapter adapter(
        Write("layout.safetensors", Pair(kQ, down, up, 3, 8, 8, "F32")), Base);
    assert(adapter.format() == format);
    assert(adapter.patches().size() == 1);
    assert(adapter.patches()[0].rank == 3);
    assert(adapter.patches()[0].down.dtype == LoraDtype::kF32);
  }

  // Dotted lora_down/up + alpha (BF16 and F32 scalars).
  {
    auto tensors =
        Join({Pair(std::string("diffusion_model.") + kQ, ".lora_down.weight",
                   ".lora_up.weight", 4, 8, 8),
              Pair("diffusion_model.transformer_blocks.1.attn.to_k",
                   ".lora_down.weight", ".lora_up.weight", 2, 8, 8)});
    tensors.push_back(
        {std::string("diffusion_model.") + kQ + ".alpha", {}, "BF16", -2.0F});
    tensors.push_back({"diffusion_model.transformer_blocks.1.attn.to_k.alpha",
                       {1},
                       "F32",
                       -1.0F});
    const LoraAdapter adapter(Write("alpha.safetensors", tensors), Base);
    assert(adapter.format() == "down-up+alpha");
    assert(Find(adapter, kQTarget).scale == 6.0F / 4);
    assert(Find(adapter, "transformer.transformer_blocks.1.attn.to_k.weight")
               .scale == 3.0F / 2);
    assert(adapter.min_rank() == 2 && adapter.max_rank() == 4);
  }

  // kohya, with a fused gate_up and a global module.
  {
    auto tensors =
        Join({Pair("lora_unet_transformer_blocks_0_attn_to_out_0",
                   ".lora_down.weight", ".lora_up.weight", 2, 8, 8),
              Pair("lora_unet_transformer_blocks_1_img_mlp_gate_up",
                   ".lora_down.weight", ".lora_up.weight", 2, 24, 8),
              Pair("lora_unet_time_text_embed_timestep_embedder_linear_1",
                   ".lora_down.weight", ".lora_up.weight", 2, 8, 6)});
    tensors.push_back({"lora_unet_transformer_blocks_1_img_mlp_gate_up.alpha",
                       {},
                       "F32",
                       -1.0F});
    const LoraAdapter adapter(Write("kohya.safetensors", tensors), Base);
    assert(adapter.format() == "kohya+alpha+gate_up");
    assert(adapter.patches().size() == 4);
    const auto& out =
        Find(adapter, "transformer.transformer_blocks.0.attn.to_out.0.weight");
    assert(out.rank == 2 && out.scale == 1);
    const auto& gate = Find(
        adapter, "transformer.transformer_blocks.1.img_mlp.gate_layer.weight");
    const auto& proj =
        Find(adapter, "transformer.transformer_blocks.1.img_mlp.proj.weight");
    assert(gate.rows == 12 && gate.cols == 8 && gate.first_row == 0);
    assert(proj.rows == 12 && proj.cols == 8 && proj.first_row == 12);
    assert(gate.scale == 1.5F && proj.scale == 1.5F);
    assert(gate.up.data == proj.up.data &&
           gate.up.shape == (std::vector{24, 2}));
    const auto& time =
        Find(adapter,
             "transformer.time_text_embed.timestep_embedder.linear_1.weight");
    assert(time.rows == 8 && time.cols == 6);
  }

  // Diffs (2-D, 1-D) mixed with pairs of different ranks; F16 tensors.
  {
    auto tensors =
        Join({Pair("diffusion_model.modulation.1", ".lora_down.weight",
                   ".lora_up.weight", 3, 32, 8, "F16"),
              Pair("diffusion_model.norm_out.linear", ".lora_down.weight",
                   ".lora_up.weight", 1, 8, 8, "F16")});
    tensors.push_back({"diffusion_model.modulation.1.alpha", {}, "F16", -1.0F});
    tensors.push_back({"diffusion_model.img_in.diff", {8, 4}, "F16"});
    tensors.push_back(
        {"diffusion_model.transformer_blocks.0.attn.norm_q.diff", {4}, "F16"});
    tensors.push_back({"diffusion_model.txt_in.text_norm.diff", {8}, "BF16"});
    const LoraAdapter adapter(Write("diff.safetensors", tensors), Base);
    assert(adapter.format() == "down-up+alpha+diff");
    assert(adapter.patches().size() == 5);
    assert(Find(adapter, "transformer.modulation.1.weight").scale == 1);
    assert(Find(adapter, "transformer.modulation.1.weight").rows == 32);
    const auto& img = Find(adapter, "transformer.img_in.weight");
    assert(img.rank == 0 && img.diff.data && !img.down.data);
    assert(img.rows == 8 && img.cols == 4 && img.scale == 1);
    const auto& norm =
        Find(adapter, "transformer.transformer_blocks.0.attn.norm_q.weight");
    assert(norm.rows == 4 && norm.cols == 1 && norm.diff.shape.size() == 1);
    const auto values = LoraFloats(norm.diff, 1, 3);
    assert(values == (std::vector{-0.5F, -0.25F, 0.0F}));
    assert(adapter.min_rank() == 1 && adapter.max_rank() == 3);
  }

  // DoRA [out,1], including gate_up.
  {
    auto tensors = Join({Pair(kQ, ".lora_A.weight", ".lora_B.weight", 2, 8, 8),
                         Pair("transformer_blocks.0.img_mlp.gate_up",
                              ".lora_A.weight", ".lora_B.weight", 2, 24, 8)});
    tensors.push_back({std::string(kQ) + ".dora_scale", {8, 1}});
    tensors.push_back(
        {"transformer_blocks.0.img_mlp.gate_up.dora_scale", {24}});
    const LoraAdapter adapter(Write("dora.safetensors", tensors), Base);
    assert(adapter.format() == "peft+dora+gate_up");
    assert(Find(adapter, kQTarget).magnitude.shape == (std::vector{8, 1}));
    const auto& proj =
        Find(adapter, "transformer.transformer_blocks.0.img_mlp.proj.weight");
    assert(proj.magnitude.data && proj.first_row == 12);
  }

  // diffusers metadata alpha, and rslora.
  {
    const auto tensors = Pair(kQ, ".lora_A.weight", ".lora_B.weight", 4, 8, 8);
    const LoraAdapter adapter(
        Write("meta.safetensors", tensors,
              R"({"lora_adapter_metadata":"{\"transformer.lora_alpha\": 8, )"
              R"(\"transformer.alpha_pattern\": {}, \"transformer.r\": 4}",)"
              R"("ss_network_alpha":"1"})"),
        Base);
    assert(adapter.format() == "peft+metadata-alpha");
    assert(adapter.patches()[0].scale == 2);
    const LoraAdapter rslora(
        Write("rslora.safetensors", tensors,
              R"({"lora_adapter_metadata":"{\"lora_alpha\": 8, )"
              R"(\"use_rslora\": true}"})"),
        Base);
    assert(rslora.patches()[0].scale == 4);
    // A module's .alpha wins over metadata.
    auto with_alpha = tensors;
    with_alpha.push_back({std::string(kQ) + ".alpha", {}, "F32", -1.0F});
    const LoraAdapter both(
        Write("both.safetensors", with_alpha,
              R"({"lora_adapter_metadata":"{\"transformer.lora_alpha\": 8}"})"),
        Base);
    assert(both.format() == "peft+alpha");
    assert(both.patches()[0].scale == 0.75F);
  }

  // A file shorter than the header length prefix maps but is rejected.
  {
    const auto path = Directory() / "short.safetensors";
    std::ofstream(path, std::ios::binary) << "abcd";
    Rejects(path, "invalid safetensors file");
  }

  // Rejections name the offending key.
  const auto q = Pair(kQ, ".lora_A.weight", ".lora_B.weight", 2, 8, 8);
  Rejects(Write("r.safetensors", Join({q, {{"lora_unet_x.lokr_w1", {2, 2}}}})),
          "lora_unet_x.lokr_w1: LyCORIS");
  Rejects(Write("r.safetensors",
                Pair("lora_te1_text_model_encoder_layers_0_mlp_fc1",
                     ".lora_down.weight", ".lora_up.weight", 2, 8, 8)),
          "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight: "
          "text-encoder");
  Rejects(
      Write("r.safetensors", Pair("transformer_blocks.2.attn.to_q",
                                  ".lora_A.weight", ".lora_B.weight", 2, 8, 8)),
      "transformer_blocks.2.attn.to_q.lora_A.weight: block 2 does not "
      "exist");
  Rejects(Write("r.safetensors",
                Pair("lora_unet_transformer_blocks_59_attn_to_q",
                     ".lora_down.weight", ".lora_up.weight", 2, 8, 8)),
          "block 59 does not exist");
  Rejects(
      Write("r.safetensors", Pair("transformer_blocks.0.img_mod.1",
                                  ".lora_A.weight", ".lora_B.weight", 2, 8, 8)),
      "img_mod.1.lora_A.weight: not a Qwen-Image-2.1 transformer module");
  Rejects(Write("r.safetensors",
                Pair("diffusion_model.double_blocks.0.img_attn.qkv",
                     ".lora_A.weight", ".lora_B.weight", 2, 8, 8)),
          "double_blocks.0.img_attn.qkv.lora_A.weight: not a Qwen-Image-2.1");
  Rejects(Write("r.safetensors",
                Join({q, {{"transformer_blocks.0.attn.norm_q.weight", {4}}}})),
          "transformer_blocks.0.attn.norm_q.weight: unsupported adapter "
          "tensor (full replacement");
  Rejects(
      Write("r.safetensors", {{std::string(kQ) + ".lora_A.weight", {2, 8, 1}},
                              {std::string(kQ) + ".lora_B.weight", {8, 2}}}),
      "to_q.lora_A.weight: lora_down must be 2-D");
  Rejects(Write("r.safetensors",
                Join({q, {{"transformer_blocks.0.attn.to_k.diff_b", {8}}}})),
          "to_k.diff_b: bias deltas");
  Rejects(Write("r.safetensors",
                Join({q, {{"transformer.crepa_projector.0.bias", {8}}}})),
          "crepa_projector.0.bias: unsupported bias tensor");
  Rejects(Write("r.safetensors",
                Pair(kQ, ".lora_A.weight", ".lora_B.weight", 2, 8, 6)),
          "to_q.lora_B.weight: shape [8,2] x [2,6] does not match base");
  Rejects(
      Write("r.safetensors", {{std::string(kQ) + ".lora_A.weight", {2, 8}},
                              {std::string(kQ) + ".lora_B.weight", {8, 3}}}),
      "to_q.lora_B.weight: rank mismatch");
  Rejects(
      Write("r.safetensors", {{std::string(kQ) + ".lora_A.weight", {2, 8}}}),
      "to_q.lora_A.weight: lora_down without lora_up");
  Rejects(Write("r.safetensors",
                Join({q, {{"transformer_blocks.0.attn.to_k.alpha", {}}}})),
          "to_k.alpha: alpha/dora_scale without a pair");
  Rejects(Write("r.safetensors",
                Join({q, {{std::string(kQ) + ".dora_scale", {1, 8}}}})),
          "to_q.dora_scale: input-axis dora_scale is unsupported");
  Rejects(Write("r.safetensors", Join({q,
                                       {{"transformer_blocks.0.attn.to_k.diff",
                                         {8, 8},
                                         "F32",
                                         std::nanf("")}}})),
          "to_k.diff: non-finite value");
  Rejects(Write("r.safetensors",
                Join({q, {{std::string(kQ) + ".alpha", {}, "F16", 1e30F}}})),
          "to_q.alpha: non-finite value");
  Rejects(Write("r.safetensors",
                {{std::string(kQ) + ".lora_A.weight", {2, 8}, "F64"},
                 {std::string(kQ) + ".lora_B.weight", {8, 2}}}),
          "to_q.lora_A.weight: unsupported adapter dtype F64");
  Rejects(Write("r.safetensors", q, "", 2), "invalid tensor extent");
  Rejects(Write("r.safetensors", q, "", 0, 1ULL << 40),
          "invalid safetensors header");
  Rejects(Write("r.safetensors",
                Pair("transformer_blocks.0.img_mlp.gate_up", ".lora_A.weight",
                     ".lora_B.weight", 2, 36, 8)),
          "gate_up.lora_B.weight: shape [36,2] x [2,8] does not match base "
          "[12,8] (fused gate_up needs 2x rows)");
  Rejects(
      Write("r.safetensors", Join({q, Pair(std::string("transformer.") + kQ,
                                           ".lora_A", ".lora_B", 2, 8, 8)})),
      "transformer.transformer_blocks.0.attn.to_q.lora_A: duplicate "
      "adapter tensor");
  Rejects(Write("r.safetensors",
                Join({Pair("transformer_blocks.0.img_mlp.gate_up",
                           ".lora_A.weight", ".lora_B.weight", 2, 24, 8),
                      Pair("transformer_blocks.0.img_mlp.proj",
                           ".lora_A.weight", ".lora_B.weight", 2, 12, 8)})),
          "img_mlp.proj.weight is patched by several modules");
  Rejects(Write("r.safetensors", q,
                R"({"lora_adapter_metadata":"{\"transformer.alpha_pattern\":)"
                R"( {\"to_q\": 4}}"})"),
          "__metadata__: alpha_pattern is unsupported");
  Rejects(
      Write("r.safetensors", Join({q, {{std::string(kQ) + ".diff", {8, 8}}}})),
      "to_q.diff: both a low-rank pair and a diff");
  Rejects(
      Write("r.safetensors", Pair("transformer_blocks.0.attn.norm_q",
                                  ".lora_A.weight", ".lora_B.weight", 2, 4, 1)),
      "norm_q.lora_A.weight: low-rank pair on a 1-D tensor");

  fs::remove_all(Directory());
  std::cout << "PASS qwen_image_21_lora_test\n";
  return 0;
}
