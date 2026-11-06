#include "src/models/qwen_image_21/weights.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "tests/support/temporary_directory.hpp"

#ifdef GUFO_TEST_WEIGHT_UPLOAD
#include "src/models/qwen_image_21/hip/runtime.hpp"
#endif

namespace {
using gufo::models::qwen_image_21::Weights;

void Write(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out << text;
  assert(out.good());
}

struct Fixture {
  TemporaryDirectory directory{"gufo-image-weights"};
  const std::filesystem::path root = directory.path();

  Fixture() {
    Write(root / "model_index.json",
          R"({"_class_name":"QwenImage21Pipeline"})");
    Write(root / "transformer/config.json", R"({
      "num_layers":32,"num_attention_heads":32,"attention_head_dim":128,
      "context_in_dim":4096,"in_channels":64,"out_channels":64,
      "patch_size":1,"mlp_ratio":3,"eps":0.000001,
      "axes_dims_rope":[16,56,56],"causal_condition":true})");
    Write(root / "text_encoder/config.json", R"({"model_type":"qwen3_vl",
      "text_config":{"hidden_size":4096,"intermediate_size":12288,
        "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,
        "head_dim":128,"vocab_size":151936,"rms_norm_eps":0.000001,
        "rope_theta":5000000,"hidden_act":"silu","attention_bias":false,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,
          "mrope_section":[24,20,20]}},
      "vision_config":{"hidden_size":1152,"intermediate_size":4304,"depth":27,
        "num_heads":16,"patch_size":16,"spatial_merge_size":2,
        "temporal_patch_size":2,"num_position_embeddings":2304,
        "out_hidden_size":4096,"deepstack_visual_indexes":[8,16,24],
        "hidden_act":"gelu_pytorch_tanh"}})");
    std::string latent_mean = "[0", latent_std = "[1";
    for (int i = 1; i < 64; ++i) {
      latent_mean += ",0";
      latent_std += ",1";
    }
    Write(root / "vae/config.json", R"({
      "z_dim":64,"base_dim":96,"decoder_base_dim":144,"in_channels":4,
      "out_channels":4,"num_res_blocks":2,"scale_factor_spatial":16,
      "dim_mult":[1,2,4,8,8],"is_residual":true,
      "temperal_downsample":[false,true,true,true],"attn_scales":[],
      "latents_mean":)" + latent_mean + "],\"latents_std\":" +
                                        latent_std + "]}");
    Write(root / "scheduler/scheduler_config.json", R"({
      "_class_name":"FlowMatchEulerDiscreteScheduler",
      "base_image_seq_len":256,"max_image_seq_len":8192,"base_shift":0.5,
      "max_shift":0.9,"shift_terminal":0.02,"num_train_timesteps":1000,
      "time_shift_type":"exponential","use_dynamic_shifting":true,
      "invert_sigmas":false,"stochastic_sampling":false,"use_beta_sigmas":false,
      "use_exponential_sigmas":false,"use_karras_sigmas":false})");
    Write(root / "processor/preprocessor_config.json", R"({
      "patch_size":16,"temporal_patch_size":2,"merge_size":2,
      "rescale_factor":0.00392156862745098,"do_convert_rgb":true,
      "do_normalize":true,"do_rescale":true,"do_resize":true,
      "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]})");
    Write(root / "text_encoder/model.safetensors.index.json", R"({"weight_map":{
      "model.language_model.a":"a.safetensors",
      "model.language_model.b":"b.safetensors"}})");
    Tensor(root / "text_encoder/a.safetensors", "model.language_model.a", true);
    Tensor(root / "text_encoder/b.safetensors", "model.language_model.b",
           false);
    Write(root / "transformer/diffusion_pytorch_model.safetensors.index.json",
          R"({"weight_map":{"a.weight":"a.safetensors",
             "b.weight":"b.safetensors","c.weight":"c.safetensors"}})");
    for (const char* name : {"a", "b", "c"})
      Tensor(root / "transformer" / (std::string(name) + ".safetensors"),
             std::string(name) + ".weight", false);
    Tensor(root / "vae/diffusion_pytorch_model.safetensors", "quant_conv.a",
           true);
  }

  void Tensor(const std::filesystem::path& path, const std::string& name,
              bool bf16, bool large = false, int scale = 1) {
    const std::array<float, 4> values{1.0F * scale, -2.5F * scale,
                                      0.3333F * scale, 13.0625F * scale};
    const std::size_t count = large ? 4096 * 4096 : values.size();
    const auto bytes = count * (bf16 ? 2 : 4);
    std::string header =
        "{\"" + name + "\":{\"dtype\":\"" + (bf16 ? "BF16" : "F32") +
        "\",\"shape\":" + (large ? "[4096,4096]" : "[4]") +
        ",\"data_offsets\":[0," + std::to_string(bytes) + "]}}";
    // Deliberately unaligned payload; the last tensor ends at EOF.
    while ((8 + header.size()) % 4096 != 37)
      header += ' ';
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t size = header.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out << header;
    std::vector<std::uint16_t> block(bf16 ? 16384 : 32768);
    if (bf16) {
      for (std::size_t i = 0; i < block.size(); ++i) {
        const float value = values[i % 4] * (large ? 1 << (i / 4096) : 1);
        auto bits = std::bit_cast<std::uint32_t>(value);
        block[i] = (bits + 32767 + ((bits >> 16) & 1)) >> 16;
      }
    } else {
      for (std::size_t i = 0; i < block.size() / 2; ++i) {
        const float value = values[i % 4] * (large ? 1 << (i / 4096) : 1);
        std::memcpy(block.data() + i * 2, &value, sizeof(value));
      }
    }
    for (std::size_t done = 0; done < bytes;) {
      const auto chunk = std::min(bytes - done, block.size() * 2);
      out.write(reinterpret_cast<const char*>(block.data()), chunk);
      done += chunk;
    }
    assert(out.good());
  }
};

void FilesAndGroups(const Fixture& fixture) {
  const Weights weights(fixture.root);
  assert(weights.Files().size() == 6);
  assert(weights.Group("absent.").empty());
  const auto group = weights.Group("text_encoder.model.language_model.");
  assert(group.size() == 2);
  assert(group[0].first == "text_encoder.model.language_model.a");
  assert(group[1].first == "text_encoder.model.language_model.b");
  assert(group[0].second->bf16 && !group[1].second->bf16);
  assert(group[0].second->file != group[1].second->file);
  for (const auto& [name, weight] : weights.Group("")) {
    assert(weight == &weights.Get(name));
    const auto& file = weights.Files()[weight->file];
    assert(weight->offset % 4096 == 37);
    assert(weight->data ==
           static_cast<const char*>(file.data) + weight->offset);
    // Retained handles must survive replacement of the source pathname.
    const auto old = file.source_path.string() + ".old";
    std::filesystem::rename(file.source_path, old);
    Write(file.source_path, "replacement must not supply tensor bytes");
    std::vector<char> bytes(weight->bytes);
    assert(gufo::platform::ReadFileAt(file.file_descriptor, bytes.data(),
                                      bytes.size(),
                                      weight->offset) == bytes.size());
    assert(std::memcmp(bytes.data(), weight->data, bytes.size()) == 0);
    std::filesystem::remove(file.source_path);
    std::filesystem::rename(old, file.source_path);
  }
}

#ifdef GUFO_TEST_WEIGHT_UPLOAD
void Upload(Fixture& fixture) {
  namespace gpu = gufo::models::qwen_image_21::hip;
  // Each F32 tensor needs 96 MiB before conversion. Three cross the batch
  // budget and exercise draining, conversion, packing, and pool reuse.
  for (const char* name : {"a", "b", "c"})
    fixture.Tensor(
        fixture.root / "transformer" / (std::string(name) + ".safetensors"),
        std::string(name) + ".weight", false, true, 1 << (name[0] - 'a'));
  const Weights weights(fixture.root);
  for (const auto& file : weights.Files()) {
    std::filesystem::rename(file.source_path,
                            file.source_path.string() + ".old");
    Write(file.source_path, "replacement must not supply GPU bytes");
  }
  gpu::Runtime runtime(weights);
  const auto a = runtime.Weight("text_encoder.model.language_model.a");
  const auto b = runtime.Weight("text_encoder.model.language_model.b");
  const std::vector<float> expected{1.0F, -2.5F, 0.333984375F, 13.0625F};
  assert(runtime.Download(a) == expected);
  assert(runtime.Download(b) == expected);
  assert(runtime.WeightBytes() == 16);
  // This VAE tensor is outside the eagerly streamed prefixes and exercises
  // the mapped per-tensor path through the same conversion/storage code.
  assert(runtime.Download(runtime.Weight("vae.quant_conv.a")) == expected);
  for (const char* name : {"a", "b", "c"}) {
    const auto key = "transformer." + std::string(name) + ".weight";
    const auto matrix = runtime.Weight(key);
    assert(runtime.WeightBytes() == 24 + 3 * 4096ULL * 4096 * 2);
    assert(matrix.packed && matrix.rows == 4096 && matrix.cols == 4096);
    assert(runtime.Weight(key).data == matrix.data);
    // Independent [K/16][row][16] layout and BF16 goldens.
    for (int row : {0, 19, 4095})
      for (int col : {0, 1, 2, 3, 16, 2047, 4095}) {
        std::uint16_t bits = 0;
        const auto offset =
            std::size_t(col / 16) * 4096 * 16 + row * 16 + col % 16;
        gpu::Check(hipMemcpy(&bits, matrix.data + offset, sizeof(bits),
                             hipMemcpyDeviceToHost));
        const float actual = std::bit_cast<float>(std::uint32_t(bits) << 16);
        assert(actual ==
               expected[col % 4] * (1 << (name[0] - 'a')) * (1 << (row % 4)));
      }
  }
}
#endif
}  // namespace

int main() {
  Fixture fixture;
  FilesAndGroups(fixture);
#ifdef GUFO_TEST_WEIGHT_UPLOAD
  int devices = 0;
  gufo::models::qwen_image_21::hip::Check(hipGetDeviceCount(&devices));
  if (!devices)
    return 77;
  Upload(fixture);
#endif
  std::cout << "Qwen-Image synthetic weight file contracts passed\n";
}
