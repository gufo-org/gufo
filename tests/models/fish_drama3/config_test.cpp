#include "src/models/fish_drama3/config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/fish_drama3/prompt.hpp"

namespace {

namespace drama3 = gufo::models::fish_drama3;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL fish_drama3_config_test: " << message << "\n";
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main() {
  const auto cfg = drama3::ParseModelConfig(R"({
    "model_type": "fish_omni",
    "speech_template_version": "reference-bracket-v1",
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "audio_pad_token_id": 248076,
    "semantic_start_token_id": 248099,
    "semantic_end_token_id": 252194,
    "text_config": {
      "model_type": "fish_omni_text",
      "vocab_size": 252288,
      "hidden_size": 2048,
      "num_hidden_layers": 40,
      "num_attention_heads": 16,
      "num_key_value_heads": 2,
      "head_dim": 256,
      "rms_norm_eps": 1e-6,
      "layernorm_zero_centered_gamma": true,
      "linear_num_key_heads": 16,
      "linear_num_value_heads": 32,
      "linear_key_head_dim": 128,
      "linear_value_head_dim": 128,
      "linear_conv_kernel_dim": 4,
      "moe_intermediate_size": 512,
      "shared_expert_intermediate_size": 512,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "rope_parameters": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0
      }
    },
    "audio_decoder_config": {
      "model_type": "fish_qwen3_audio_decoder",
      "decoder_type": "codebook_mtp_shared",
      "vocab_size": 4096,
      "dim": 2048,
      "shared_enrichment_layers": 4,
      "n_head": 16,
      "n_local_heads": 2,
      "head_dim": 256,
      "intermediate_size": 6144,
      "norm_eps": 1e-6,
      "rope_base": 10000000.0,
      "rotary_percent": 0.25,
      "rotary_interleaved": true,
      "layernorm_zero_centered_gamma": true,
      "attention_output_gate": true,
      "num_codebooks": 10
    }
  })");

  Check(cfg.has_value(), "valid Fish Drama-3 configuration parses");
  Check(drama3::IsSupportedModelConfig(*cfg),
        "valid Fish Drama-3 configuration is supported");
  Check(cfg->backbone.num_hidden_layers == 40 &&
            cfg->backbone.num_experts == 256 &&
            cfg->backbone.num_experts_per_tok == 8,
        "MoE backbone dimensions are preserved");
  Check(cfg->tower.shared_layers == 4 && cfg->tower.residual_layers == 9 &&
            cfg->tower.codebook_size == 4096,
        "audio tower dimensions are preserved");
  Check(cfg->codec.sample_rate == 44100 && cfg->codec.frame_length == 2048,
        "44.1 kHz Fish DAC codec defaults are set");
  Check(!drama3::ParseModelConfig("{").has_value(),
        "malformed JSON fails closed");

  const auto wrong_type = drama3::ParseModelConfig(R"({
    "model_type": "qwen3_tts"
  })");
  Check(!wrong_type.has_value() || !drama3::IsSupportedModelConfig(*wrong_type),
        "non-Drama3 model_type is rejected");

  drama3::ValidateDrama3Text("Hello world.", "test");

  drama3::FishDacCodes codes;
  codes.codebooks = 10;
  codes.frames = 2;
  codes.codes.assign(20, 0);
  drama3::ValidateDrama3Codes(codes, cfg->codec.semantic_codebook_size);

  bool threw_on_bad_code = false;
  codes.codes[0] = 5000;
  try {
    drama3::ValidateDrama3Codes(codes, cfg->codec.semantic_codebook_size);
  } catch (const std::exception&) {
    threw_on_bad_code = true;
  }
  Check(threw_on_bad_code, "out-of-range semantic codebook entry is rejected");

  std::cout << "PASS fish_drama3_config_test\n";
  return 0;
}
