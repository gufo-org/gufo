#ifndef GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
#define GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/sampling.hpp"

namespace gufo::models::qwen3_tts {

struct SynthesisRequest {
  std::string text;
  std::string speaker{"vivian"};
  std::string language{"english"};
  std::string instruct;
  AudioBuffer reference_audio;
  std::string reference_text;
  bool speaker_embedding_only{false};
  std::size_t max_new_tokens{3000};
  SamplingOptions sampling;
  // Called with newly generated mono audio at 24 kHz. Returning false cancels
  // generation. Streaming results retain codes but do not buffer the waveform.
  std::function<bool(std::span<const float>)> on_audio{};
};

struct SynthesisResult {
  std::uint32_t sample_rate{0};
  std::uint32_t code_groups{0};
  std::vector<float> samples;
  // Frame-major [frames, code_groups].
  std::vector<std::int32_t> codes;
  std::size_t sample_count{0};
};

using CancellationCheck = std::function<bool()>;

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
