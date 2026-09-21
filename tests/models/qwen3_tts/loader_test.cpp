#include "src/models/qwen3_tts/loader.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/models/qwen3_tts/config.hpp"

namespace {

namespace qwen3_tts = gufo::models::qwen3_tts;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_tts_loader_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Optional argv: model dir. When absent, skips if default location missing.
  const std::string model_dir =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";

  if (!std::filesystem::exists(model_dir + "/config.json")) {
    std::cerr << "SKIP qwen3_tts_loader_test: model not present at "
              << model_dir << '\n';
    return 77;
  }

  auto config = qwen3_tts::LoadModelConfigFromPath(model_dir);
  Check(config.has_value(), "config parse");
  Check(qwen3_tts::IsSupportedModelConfig(*config), "supported 1.7B variant");
  Check(config->talker.hidden_size == 2048, "talker hidden 2048");
  Check(config->talker.num_hidden_layers == 28, "talker layers 28");
  Check(config->talker.num_attention_heads == 16, "talker heads 16");
  Check(config->talker.num_key_value_heads == 8, "talker kv heads 8");
  Check(config->talker.intermediate_size == 6144, "talker ffn 6144");
  Check(config->talker.codec_eos_token_id == 2150, "codec eos 2150");
  Check(config->code_predictor.num_hidden_layers == 5, "cp layers 5");
  Check(config->code_predictor.hidden_size == 1024, "cp hidden 1024");
  Check(config->speech_tokenizer.output_sample_rate == 24000,
        "speech tokenizer sample rate 24000");
  Check(config->speech_tokenizer.num_quantizers == 16,
        "speech tokenizer quantizers 16");
  Check(config->speech_tokenizer.latent_dim == 1024,
        "speech decoder latent 1024");
  Check(config->speech_tokenizer.codebook_dim == 512,
        "speech decoder codebook dim 512");
  Check(config->speech_tokenizer.decoder_dim == 1536,
        "speech decoder channel width 1536");
  Check(config->speech_tokenizer.hidden_size == 512 &&
            config->speech_tokenizer.intermediate_size == 1024,
        "speech decoder transformer dimensions");
  Check(config->speech_tokenizer.num_hidden_layers == 8 &&
            config->speech_tokenizer.num_attention_heads == 16 &&
            config->speech_tokenizer.head_dim == 64,
        "speech decoder transformer topology");
  Check(config->speech_tokenizer.upsample_rates ==
            std::vector<std::uint32_t>({8, 5, 4, 3}),
        "speech decoder waveform upsample rates");
  Check(config->speech_tokenizer.upsampling_ratios ==
            std::vector<std::uint32_t>({2, 2}),
        "speech decoder latent upsample ratios");

  if (config->variant == qwen3_tts::ModelVariant::kCustomVoice) {
    const auto spk = config->talker.spk_id.find("vivian");
    Check(spk != config->talker.spk_id.end() && spk->second == 3065,
          "vivian spk id 3065");
    const auto dialect = config->talker.spk_dialect.find("eric");
    Check(dialect != config->talker.spk_dialect.end() &&
              dialect->second == "sichuan_dialect",
          "Eric dialect is preserved");
  }

  auto loaded = qwen3_tts::LoadModelDirectory(model_dir);
  Check(loaded.ok, "load model dir: " + loaded.error);
  Check(loaded.store != nullptr, "store present");
  Check(loaded.store->size() > 850, "talker and speech tokenizer tensors");

  const auto* text_emb =
      loaded.store->Find("talker.model.text_embedding.weight");
  Check(text_emb != nullptr, "text_embedding present");
  Check(text_emb->shape.size() == 2 && text_emb->shape[0] == 151936,
        "text_embedding rows 151936");
  Check(text_emb->data != nullptr, "text_embedding payload mapped");
  const auto* waveform_weight =
      loaded.store->Find("decoder.decoder.6.conv.weight");
  Check(waveform_weight != nullptr && waveform_weight->shape.size() == 3,
        "speech decoder output convolution present");

  const auto talker_regions = loaded.RegionsFor("talker.");
  const auto decoder_regions = loaded.RegionsFor("decoder.");
  const auto contains = [](const auto& regions, const auto* tensor) {
    if (!tensor)
      return false;
    const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
    for (const auto& region : regions) {
      const auto base = reinterpret_cast<std::uintptr_t>(region.data);
      if (address >= base && address - base < region.size &&
          tensor->byte_count <= region.size - (address - base))
        return true;
    }
    return false;
  };
  Check(talker_regions.size() == 1 && decoder_regions.size() == 1 &&
            contains(talker_regions, text_emb) &&
            contains(decoder_regions, waveform_weight),
        "selected component weights remain available");
  const auto* encoder_weight =
      loaded.store->Find("encoder.encoder.layers.0.conv.weight");
  Check(encoder_weight && !contains(decoder_regions, encoder_weight) &&
            !contains(talker_regions, waveform_weight),
        "synthesis uploads must exclude unused component weights");
  const auto* speaker_weight =
      loaded.store->Find("speaker_encoder.blocks.0.conv.weight");
  Check(!speaker_weight || !contains(talker_regions, speaker_weight),
        "talker upload must exclude the independently loaded speaker encoder");
  std::size_t selected_bytes = 0, all_bytes = 0;
  for (const auto& region : loaded.mapped_regions)
    all_bytes += region.size;
  for (const auto& regions : {talker_regions, decoder_regions})
    for (const auto& region : regions)
      selected_bytes += region.size;
  Check(selected_bytes < all_bytes, "unused weight bytes were not excluded");
  std::cout << "PASS qwen3_tts_loader_test (tensors=" << loaded.store->size()
            << ") avoided_upload_bytes=" << all_bytes - selected_bytes << '\n';
  return 0;
}
