#ifndef GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_
#define GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_

#include "src/ext/asr.hpp"

namespace gufo::models::qwen3_asr {

using TranscriptionRequest = ext::TranscriptionRequest;
using TranscriptionTimings = ext::TranscriptionTimings;
using TranscriptionResult = ext::TranscriptionResult;
using CancellationCheck = ext::CancellationCheck;

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_TRANSCRIPTION_HPP_
