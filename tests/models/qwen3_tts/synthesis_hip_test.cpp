#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include "src/models/qwen3_tts/hip/synthesis_runtime.hpp"

int main(int argc, char** argv) {
  namespace tts = gufo::models::qwen3_tts;
  const std::string root =
      argc > 1 ? argv[1]
               : "/home/fbozzo/projects/Qwen3-TTS-12Hz-1.7B-CustomVoice";
  if (!std::filesystem::exists(root + "/model.safetensors"))
    return 77;
  const auto config = tts::LoadModelConfigFromPath(root);
  assert(config);
  std::string error;
  auto runtime = tts::hip::SynthesisHipRuntime::Create(root, 1024,
                                                       config->variant, &error);
  if (!runtime) {
    std::cerr << error << '\n';
    return 1;
  }
  tts::SynthesisRequest request{
      .text =
          "When the first light reaches the harbor, speak gently, with quiet "
          "confidence "
          "and a warm, resonant tone. The boats are still resting beside the "
          "old wooden "
          "pier, and the town is slowly waking to another bright morning. We "
          "will walk "
          "along the shore together and listen to the waves.",
      .language = "english",
      .instruct = "A warm adult voice with clear diction and calm confidence.",
      .max_new_tokens = 64};
  if (config->variant == tts::ModelVariant::kBase) {
    if (argc < 3)
      return 77;
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    assert(!bytes.empty());
    assert(tts::DecodeWav(std::as_bytes(std::span(bytes)),
                          &request.reference_audio, &error));
    // Exercise full ICL: reference codec frames must be consumed, not emitted.
    request.reference_text =
        "Uh huh. Oh yeah, yeah. He wasn't even that big when I started "
        "listening to him.";
  }
  request.sampling.top_k = 20;
  request.sampling.top_p = .85F;
  request.sampling.temperature = .7F;
  request.sampling.predictor_top_k = 30;
  request.sampling.predictor_top_p = .9F;
  request.sampling.predictor_temperature = .8F;
  tts::SynthesisResult buffered, replay, streamed;
  const auto start = std::chrono::steady_clock::now();
  assert(runtime->Generate(request, {}, &buffered, &error));
  const auto buffered_end = std::chrono::steady_clock::now();
  assert(buffered.codes.size() / 16 > 38);
  assert(runtime->Generate(request, {}, &replay, &error));
  assert(buffered.codes == replay.codes && buffered.samples == replay.samples);
  std::vector<float> chunks;
  std::size_t callbacks = 0;
  double first_audio_ms = 0;
  const auto streaming_start = std::chrono::steady_clock::now();
  request.on_audio = [&](std::span<const float> audio) {
    if (callbacks++ == 0)
      first_audio_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - streaming_start)
                           .count();
    chunks.insert(chunks.end(), audio.begin(), audio.end());
    return true;
  };
  assert(runtime->Generate(request, {}, &streamed, &error));
  assert(callbacks > 1 && streamed.samples.empty());
  assert(streamed.codes == buffered.codes &&
         chunks.size() == buffered.samples.size());
  double total_error = 0, maximum_error = 0;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const double difference =
        std::abs(static_cast<double>(chunks[i]) - buffered.samples[i]);
    total_error += difference;
    maximum_error = std::max(maximum_error, difference);
  }
  assert(total_error / chunks.size() < 1e-5 && maximum_error < 1e-4);
  assert(streamed.sample_count == chunks.size());
  request.on_audio = [](std::span<const float>) { return false; };
  assert(!runtime->Generate(request, {}, &streamed, &error));
  assert(streamed.codes.empty() && streamed.samples.empty());
  request.on_audio = {};
  assert(runtime->Generate(request, {}, &replay, &error));
  assert(replay.codes == buffered.codes && replay.samples == buffered.samples);
  std::cout
      << "PASS synthesis replay/stream/cancel frames="
      << buffered.codes.size() / 16 << " buffered_ms="
      << std::chrono::duration<double, std::milli>(buffered_end - start).count()
      << " first_audio_ms=" << first_audio_ms
      << " stream_waveform_mae=" << total_error / chunks.size()
      << " max_error=" << maximum_error << '\n';
}
