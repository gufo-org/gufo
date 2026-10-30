// Aggregate throughput of batched LFM3-Audio speech synthesis:
//   lfm3_audio_batch_bench BACKBONE.gguf MMPROJ.gguf [STREAMS...]
// Each stream speaks the same sentence with its own seed; one warmup batch
// runs first. Reports audio seconds per wall second for every batch size.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/lfm3_audio/generation.hpp"
namespace audio = gufo::models::lfm3_audio;
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: lfm3_audio_batch_bench BACKBONE MMPROJ [STREAMS...]\n";
    return 2;
  }
  std::vector<std::size_t> sizes;
  for (int i = 3; i < argc; ++i)
    sizes.push_back(std::stoul(argv[i]));
  if (sizes.empty())
    sizes = {1, 2, 4, 8};
  std::string error;
  auto model = audio::Model::Load(argv[1], argv[2], &error);
  std::size_t largest = 1;
  for (auto n : sizes)
    largest = std::max(largest, n);
  auto batch = model ? audio::SpeechBatch::Create(model, 1024, largest, &error)
                     : nullptr;
  if (!batch) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  std::string system;
  audio::TtsSystemPrompt("en", "female", &system);
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].parts.push_back(
      {false,
       "The quick brown fox jumps over the lazy dog. Then it runs back home "
       "through the forest, happy and tired.",
       {}});
  const auto prompt = audio::BuildPrompt(system, messages, "<|audio_start|>");
  auto run = [&](std::size_t streams, bool report) {
    std::vector<audio::GenerateResult> results(streams);
    std::size_t done = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < streams; ++i) {
      audio::GenerateOptions options;
      options.max_new_tokens = 600;
      options.seed = 100 + i;
      if (!batch->Add(
              prompt, options, {},
              [&, i](bool ok, audio::GenerateResult result,
                     std::string failure) {
                if (!ok)
                  error = failure;
                results[i] = std::move(result);
                ++done;
              },
              &error))
        return false;
    }
    while (done < streams)
      if (!batch->Step(&error))
        return false;
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    if (!error.empty())
      return false;
    double seconds = 0, depth = 0, backbone = 0;
    std::size_t frames = 0;
    for (const auto& r : results) {
      seconds += r.samples.size() / 24000.0;
      frames = std::max(frames, r.frames);
      depth = std::max(depth, r.timings.depth_ms);
      backbone = std::max(backbone, r.timings.backbone_ms);
    }
    if (report)
      std::printf(
          "streams %zu: %.2f s audio in %.0f ms, %.1fx real time aggregate "
          "(%.1fx per stream); per step depth %.2f backbone %.2f ms\n",
          streams, seconds, ms, seconds * 1000 / ms,
          seconds * 1000 / ms / double(streams), depth / double(frames),
          backbone / double(frames));
    return true;
  };
  if (!run(largest, false)) {
    std::cerr << "error: " << error << "\n";
    return 1;
  }
  for (auto n : sizes)
    if (!run(n, true)) {
      std::cerr << "error: " << error << "\n";
      return 1;
    }
  return 0;
}
