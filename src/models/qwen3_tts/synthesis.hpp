#ifndef GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
#define GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_

#include "src/ext/tts.hpp"
#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/sampling.hpp"

namespace gufo::models::qwen3_tts {

using SynthesisRequest = ext::SynthesisRequest;
using SynthesisResult = ext::SynthesisResult;
using CancellationCheck = ext::CancellationCheck;

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_SYNTHESIS_HPP_
