// Prefills one 4096-token prompt of the Qwen3.6-35B-A3B target in chunks of
// several sizes. The time difference between chunk sizes is the fixed cost
// per chunk that stacking several sessions' prompts into one chunk would
// save. Skips with code 77 when GUFO_QWEN35BA3B_MODEL is unset.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

int main() {
  const char* model_path = std::getenv("GUFO_QWEN35BA3B_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "GUFO_QWEN35BA3B_MODEL not set; skipping\n";
    return 77;
  }
  try {
    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(model_path, &error);
    if (reader == nullptr)
      throw std::runtime_error(error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (model == nullptr)
      throw std::runtime_error(error);
    auto executor = gufo::hip::QwenGpuExecutor::Create(model, &error, 8192);
    if (executor == nullptr)
      throw std::runtime_error(error);

    // Natural-looking text so expert routing resembles a real prompt.
    std::string text;
    while (text.size() < 40000)
      text +=
          "The scheduler admits each request, prefills its prompt in chunks, "
          "then decodes one token at a time while the cache keeps every "
          "attention key and value. Memory bandwidth bounds decoding. ";
    auto tokens = executor->GetTokenizer().Encode(text);
    constexpr std::size_t kPrompt = 4096;
    if (tokens.size() < kPrompt)
      throw std::runtime_error("fixture text too short");
    tokens.resize(kPrompt);

    std::cout << "chunk chunks total_ms tok_per_s\n";
    for (const std::size_t chunk : {256U, 512U, 1024U, 2048U, 4096U}) {
      if (chunk > executor->GetMaxPromptBatch())
        continue;
      std::vector<double> samples;
      for (int iteration = 0; iteration < 4; ++iteration) {
        executor->Reset();
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t offset = 0; offset < kPrompt; offset += chunk) {
          (void)executor->ForwardPromptBatch(
              std::span(tokens).subspan(offset, chunk),
              static_cast<std::uint32_t>(offset), offset + chunk == kPrompt);
        }
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start;
        if (iteration > 0)
          samples.push_back(elapsed.count());
      }
      std::sort(samples.begin(), samples.end());
      const double median = samples[samples.size() / 2];
      std::cout << chunk << ' ' << kPrompt / chunk << ' ' << std::fixed
                << std::setprecision(1) << median << ' '
                << 1000.0 * static_cast<double>(kPrompt) / median << '\n';
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "prefill chunk bench failed: " << exception.what() << '\n';
    return 1;
  }
}
