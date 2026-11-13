// The Base voice-cloning encoders on the NPU against the GPU ones on the same
// recording: the speaker embedding's cosine and the share of equal speech
// codes. Skips (77) without the model, the GPU or the NPU.
// Usage: qwen3_tts_npu_encoders_test <reference wav> [Base model dir]; the
// model defaults to $QWEN3_TTS_BASE_MODEL.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/hip/speaker_encoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/speech_encoder_runtime.hpp"
#include "src/models/qwen3_tts/npu/speaker_encoder_npu.hpp"
#include "src/models/qwen3_tts/npu/speech_encoder_npu.hpp"

namespace tts = gufo::models::qwen3_tts;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: qwen3_tts_npu_encoders_test <wav> [model]\n");
    return 2;
  }
  const char* env = std::getenv("QWEN3_TTS_BASE_MODEL");
  const std::string model = argc > 2 ? argv[2] : (env ? env : "");
  if (model.empty()) {
    std::printf("SKIP: set QWEN3_TTS_BASE_MODEL\n");
    return 77;
  }
  std::ifstream in(argv[1], std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  tts::AudioBuffer audio;
  std::string error;
  if (raw.empty() ||
      !tts::DecodeWav(
          std::span<const std::byte>(
              reinterpret_cast<const std::byte*>(raw.data()), raw.size()),
          &audio, &error)) {
    std::printf("SKIP: no reference recording (%s)\n", error.c_str());
    return 77;
  }
  auto gpu_speaker = tts::hip::SpeakerEncoderHipRuntime::Create(model, &error);
  auto npu_speaker = tts::npu::SpeakerEncoderNpuRuntime::Create(model, &error);
  if (!gpu_speaker || !npu_speaker) {
    std::printf("SKIP: %s\n", error.c_str());
    return 77;
  }
  tts::hip::SpeakerEncoderOutput a, b;
  if (!gpu_speaker->Encode(audio, &a, &error) ||
      !npu_speaker->Encode(audio, &b, &error) ||
      a.embedding.size() != b.embedding.size()) {
    std::printf("FAIL: speaker encoding: %s\n", error.c_str());
    return 1;
  }
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.embedding.size(); ++i) {
    dot += double(a.embedding[i]) * b.embedding[i];
    na += double(a.embedding[i]) * a.embedding[i];
    nb += double(b.embedding[i]) * b.embedding[i];
  }
  const double cosine = dot / std::sqrt(na * nb);
  auto gpu_speech = tts::hip::SpeechEncoderHipRuntime::Create(model, &error);
  auto npu_speech = tts::npu::SpeechEncoderNpuRuntime::Create(model, &error);
  tts::hip::SpeechEncoderOutput ca, cb;
  if (!gpu_speech || !npu_speech || !gpu_speech->Encode(audio, &ca, &error) ||
      !npu_speech->Encode(audio, &cb, &error) ||
      ca.codes.size() != cb.codes.size()) {
    std::printf("FAIL: speech encoding: %s\n", error.c_str());
    return 1;
  }
  std::size_t same = 0, same_first = 0;
  for (std::size_t i = 0; i < ca.codes.size(); ++i) {
    same += ca.codes[i] == cb.codes[i];
    if (i % ca.code_groups == 0)
      same_first += ca.codes[i] == cb.codes[i];
  }
  const double share = double(same) / double(ca.codes.size());
  const double share_first = double(same_first) / double(ca.frames);
  std::printf(
      "speaker embedding cosine %.6f; speech codes equal %.1f%% (semantic "
      "%.1f%%) over %zu frames\n",
      cosine, 100 * share, 100 * share_first, ca.frames);
  const bool ok = cosine > 0.999 && share_first > 0.9;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
