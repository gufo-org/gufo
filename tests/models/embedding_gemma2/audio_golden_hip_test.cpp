#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/json.hpp"
#include "src/models/embedding_gemma2/audio_decoder.hpp"
#include "src/models/embedding_gemma2/audio_processor.hpp"
#include "src/models/embedding_gemma2/hip/audio_runtime.hpp"
#include "src/models/embedding_gemma2/hip/encoder_runtime.hpp"
#include "src/models/embedding_gemma2/image_processor.hpp"

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::vector<float> Waveform(std::size_t count, const std::string& kind) {
  if (kind == "speech") {
    std::ifstream input(GUFO_EMBEDDING_GEMMA2_SPEECH_WAV, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                    {});
    Check(!bytes.empty(), "packaged speech fixture must exist");
    return model::DecodeAudio(bytes);
  }
  std::vector<float> samples(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    if (kind == "triangle") {
      const int phase = static_cast<int>(i % 80);
      samples[i] = static_cast<float>(std::min(phase, 80 - phase) - 20) / 64;
    } else if (kind == "music") {
      const int period = (i / 2000) % 2 == 0 ? 64 : 80;
      const int phase = static_cast<int>(i % period);
      samples[i] =
          static_cast<float>(std::min(phase, period - phase) - period / 4) / 64;
    } else if (kind == "environment") {
      samples[i] =
          i % 4000 < 800
              ? static_cast<float>(
                    static_cast<int>((i * 73 + i * i * 19) % 1024) - 512) /
                    1024
              : 0;
    } else if (kind == "noise") {
      samples[i] = static_cast<float>(
                       static_cast<int>((i * 73 + i * i * 19) % 1024) - 512) /
                   1024;
    }
  }
  return samples;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const char* configured = std::getenv("EMBEDDING_GEMMA2_MODEL_ROOT");
    const std::string path = argc > 1     ? argv[1]
                             : configured ? configured
                                          : "models/embeddinggemma-2";
    if (!std::filesystem::exists(std::filesystem::path(path) /
                                 "model.safetensors"))
      return 77;
    std::ifstream input(GUFO_EMBEDDING_GEMMA2_AUDIO_EMBEDDING_GOLDEN);
    Check(input.good(), "official audio embedding fixtures must exist");
    std::ostringstream contents;
    contents << input.rdbuf();
    const auto fixture = gufo::json::parse(contents.str());
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    {
      std::size_t free_before = 0, free_loaded = 0, total = 0;
      HIP_CHECK(hipMemGetInfo(&free_before, &total));
      model::hip::AudioRuntime audio(path, stream);
      HIP_CHECK(hipMemGetInfo(&free_loaded, &total));
      std::cout << "audio device bytes after load=" << free_before - free_loaded
                << std::endl;
      for (const auto& probe : fixture.find("cases")->items()) {
        if (!probe.find("audio_features"))
          continue;
        std::size_t reference_offset = 0;
        for (const auto& clip : probe.find("clips")->items()) {
          auto features = model::ProcessAudio(
              Waveform(clip.member_size("samples"), clip.member_str("kind")));
          const float* device = audio.Run(features);
          std::vector<float> result(features.soft_tokens() * 512);
          HIP_CHECK(hipMemcpyAsync(result.data(), device,
                                   result.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipStreamSynchronize(stream));
          const auto& expected = probe.find("audio_features")->items();
          Check(reference_offset + features.soft_tokens() <= expected.size(),
                "audio valid-token count");
          double squared_error = 0, squared_ref = 0, worst = 0;
          for (std::size_t t = 0; t < features.soft_tokens(); ++t) {
            const auto& row = expected[reference_offset + t].items();
            for (std::size_t c = 0; c < row.size(); ++c) {
              const double reference = row[c].as_double();
              const double error = result[t * 512 + c] - reference;
              Check(std::isfinite(result[t * 512 + c]),
                    "finite audio features");
              squared_error += error * error;
              squared_ref += reference * reference;
              worst = std::max(worst, std::abs(error));
            }
          }
          const double relative_error = std::sqrt(squared_error / squared_ref);
          std::cout << probe.member_str("name")
                    << " relative L2=" << relative_error << " max=" << worst
                    << std::endl;
          Check(relative_error < 5e-4 && worst < 0.01,
                "official audio projector features differ");
          std::vector<float> encoded(features.soft_tokens() * 1536);
          HIP_CHECK(hipMemcpyAsync(encoded.data(), audio.encoded_states(),
                                   encoded.size() * sizeof(float),
                                   hipMemcpyDeviceToHost, stream));
          HIP_CHECK(hipStreamSynchronize(stream));
          const auto& encoder_expected = probe.find("encoder_states")->items();
          squared_error = 0;
          squared_ref = 0;
          for (std::size_t t = 0; t < features.soft_tokens(); ++t) {
            for (std::size_t c = 0; c < 1536; ++c) {
              const double ref =
                  encoder_expected[reference_offset + t].items()[c].as_double();
              const double delta = encoded[t * 1536 + c] - ref;
              squared_error += delta * delta;
              squared_ref += ref * ref;
            }
          }
          Check(std::sqrt(squared_error / squared_ref) < 5e-4,
                "official audio encoder states differ");
          reference_offset += features.soft_tokens();
        }
        Check(reference_offset == probe.find("audio_features")->size(),
              "every reference audio token checked");
      }
      int polls = 0;
      bool cancelled = false;
      const auto features = model::ProcessAudio(Waveform(16000, "triangle"));
      try {
        (void)audio.Run(features, [&] { return ++polls == 2; });
      } catch (const std::runtime_error&) {
        cancelled = true;
      }
      HIP_CHECK(hipStreamSynchronize(stream));
      Check(cancelled && polls == 2,
            "audio encoder observes cancellation between layers");
      (void)audio.Run(features);
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    std::string error;
    auto encoder = model::hip::EncoderRuntime::Create(path, &error);
    Check(encoder != nullptr, error);
    for (const auto& probe : fixture.find("cases")->items()) {
      model::EncoderSequence sequence;
      for (const auto& id : probe.find("token_ids")->items())
        sequence.tokens.push_back(static_cast<std::uint32_t>(id.as_size()));
      std::size_t cursor = 0;
      for (const auto& clip : probe.find("clips")->items()) {
        auto features = model::ProcessAudio(
            Waveform(clip.member_size("samples"), clip.member_str("kind")));
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != model::kAudioTokenId)
          ++cursor;
        const auto count = features.soft_tokens();
        sequence.audio.push_back({cursor, std::move(features)});
        cursor += count;
      }
      cursor = 0;
      for (const auto& picture : probe.find("images")->items()) {
        gufo::core::Image image;
        image.width = static_cast<std::uint32_t>(picture.member_size("width"));
        image.height =
            static_cast<std::uint32_t>(picture.member_size("height"));
        const auto seed = picture.member_size("seed");
        for (std::size_t y = 0; y < image.height; ++y)
          for (std::size_t x = 0; x < image.width; ++x)
            for (std::size_t c = 0; c < 3; ++c)
              image.pixels.push_back(static_cast<std::uint8_t>(
                  (x * 17 + y * 31 + c * 67 + (x * y + seed * 13) % 251) %
                  256));
        auto patches = model::ProcessImage(image);
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != model::kImageTokenId)
          ++cursor;
        const auto count = patches.soft_tokens();
        sequence.images.push_back({cursor, std::move(patches)});
        cursor += count;
      }
      std::vector<std::vector<float>> vectors;
      Check(encoder->EmbedMultimodal({sequence}, &vectors, &error), error);
      const auto& expected = probe.find("embedding")->items();
      double dot = 0, norm = 0, ref_norm = 0;
      for (std::size_t c = 0; c < expected.size(); ++c) {
        const double reference = expected[c].as_double();
        dot += vectors[0][c] * reference;
        norm += vectors[0][c] * vectors[0][c];
        ref_norm += reference * reference;
      }
      const double cosine = dot / std::sqrt(norm * ref_norm);
      std::cout << probe.member_str("name") << " embedding cosine=" << cosine
                << std::endl;
      Check(cosine > 0.999, "official audio final vector differs");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
