#include "src/models/embedding_gemma2/config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace embedding_gemma2 = gufo::models::embedding_gemma2;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_config_test: " << message << '\n';
    std::exit(1);
  }
}

// config.json of google/embeddinggemma-2 revision 914f7f89 without the vision
// and audio towers, for the text-only configuration contract.
constexpr const char* kConfig = R"json(
{
 "architectures": [
  "EmbeddingGemma2Model"
 ],
 "audio_token_id": 258881,
 "boa_token_id": 256000,
 "boi_token_id": 255999,
 "dtype": "bfloat16",
 "eoa_token_index": 258883,
 "eoi_token_id": 258882,
 "image_token_id": 258880,
 "initializer_range": 0.02,
 "model_type": "embedding_gemma2",
 "text_config": {
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "dtype": "bfloat16",
  "embedding_dim": 768,
  "eos_token_id": 1,
  "head_dim": 256,
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 512,
  "hidden_size_per_layer_input": 512,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "layer_types": [
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "sliding_attention",
   "full_attention"
  ],
  "max_position_embeddings": 262144,
  "model_type": "embedding_gemma2_text",
  "num_attention_heads": 4,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "pad_token_id": 0,
  "per_layer_config": {
   "05": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "11": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "17": {
    "head_dim": 512,
    "num_key_value_heads": 1
   },
   "23": {
    "head_dim": 512,
    "num_key_value_heads": 1
   }
  },
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
   "full_attention": {
    "rope_theta": 1000000.0,
    "rope_type": "default"
   },
   "sliding_attention": {
    "rope_theta": 10000.0,
    "rope_type": "default"
   }
  },
  "sliding_window": 512,
  "vocab_size": 262144
 },
 "transformers_version": "5.18.0.dev0",
 "video_token_id": 258884,
 "vision_soft_tokens_per_image": 280
}
)json";

std::string Replace(std::string text, std::string_view from,
                    std::string_view to) {
  const auto at = text.find(from);
  Check(at != std::string::npos, "test edit must match the fixture");
  text.replace(at, from.size(), to);
  return text;
}

}  // namespace

int main() {
  const auto config = embedding_gemma2::ParseModelConfig(kConfig);
  Check(config.has_value(), "the released configuration must parse");
  Check(embedding_gemma2::IsSupportedModelConfig(*config),
        "the released configuration must be supported");

  const std::string vision_config =
      Replace(kConfig, "\"vision_soft_tokens_per_image\": 280",
              R"("vision_soft_tokens_per_image": 280,
         "vision_config": {
           "model_type": "gemma4_vision", "dtype": "bfloat16",
           "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 768,
           "intermediate_size": 3072, "num_hidden_layers": 16,
           "num_attention_heads": 12, "num_key_value_heads": 12,
           "head_dim": 64, "patch_size": 16, "pooling_kernel_size": 3,
           "position_embedding_size": 10240, "standardize": false,
           "use_clipped_linears": false, "attention_bias": false,
           "rms_norm_eps": 1e-6,
           "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0}
         })");
  const auto vision = embedding_gemma2::ParseModelConfig(vision_config);
  Check(vision && vision->supported_vision,
        "released vision geometry is supported");
  for (const auto& changed :
       {Replace(vision_config, "\"head_dim\": 64", "\"head_dim\": 128"),
        Replace(vision_config, "\"pooling_kernel_size\": 3",
                "\"pooling_kernel_size\": 2"),
        Replace(vision_config, "\"standardize\": false",
                "\"standardize\": true"),
        Replace(vision_config, "\"image_token_id\": 258880",
                "\"image_token_id\": 42")}) {
    const auto unsupported = embedding_gemma2::ParseModelConfig(changed);
    Check(unsupported && !unsupported->supported_vision &&
              embedding_gemma2::IsSupportedModelConfig(*unsupported),
          "unsupported vision refuses images while retaining text support");
  }

  const embedding_gemma2::TextConfig& text = config->text;
  Check(text.layers.size() == 24, "24 layers");
  Check(text.sliding_window == 512, "sliding window radius is 512");
  Check(text.embedding_dim == 768, "embedding dimension is 768");
  for (std::size_t index = 0; index < text.layers.size(); ++index) {
    const bool full = index % 6 == 5;
    const auto& layer = text.layers[index];
    Check((layer.type == embedding_gemma2::LayerType::kFullAttention) == full,
          "every sixth layer is a full-attention layer");
    Check(layer.head_dim == (full ? 512U : 256U),
          "full-attention layers use a 512-wide head");
    Check(layer.num_key_value_heads == (full ? 1U : 2U),
          "full-attention layers use one key/value head");
    Check(layer.rope_theta == (full ? 1000000.0F : 10000.0F),
          "rope base depends on the layer type");
  }

  Check(!embedding_gemma2::ParseModelConfig("{}").has_value(),
        "a config without a text backbone is rejected");
  const auto wrong_width = embedding_gemma2::ParseModelConfig(
      Replace(kConfig, "\"hidden_size\": 512", "\"hidden_size\": 768"));
  Check(wrong_width.has_value() &&
            !embedding_gemma2::IsSupportedModelConfig(*wrong_width),
        "a different width is unsupported");
  const auto short_layers = embedding_gemma2::ParseModelConfig(Replace(
      kConfig, "\"num_hidden_layers\": 24", "\"num_hidden_layers\": 23"));
  Check(!short_layers.has_value(), "layer_types must match num_hidden_layers");

  std::cout << "PASS embedding_gemma2_config_test\n";
  return 0;
}
