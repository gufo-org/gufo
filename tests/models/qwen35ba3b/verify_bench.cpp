// Times one speculative verification step of the Qwen3.6-35B-A3B target for
// several (sessions x rows) shapes, the widths the server batches under
// concurrent DFlash2 decoding. Reports the median step time and its cost per
// row. Skips with code 77 when GUFO_QWEN35BA3B_MODEL is unset.
#include <algorithm>
#include <array>
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

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Distinct natural prompts so each session routes differently.
constexpr std::array<const char*, 4> kTexts{
    "Virtual memory lets an operating system give each process its own "
    "address space. Pages map virtual addresses to physical memory, and a page "
    "fault occurs when the requested page is not resident. The kernel then "
    "reads the page from disk, updates the page table and resumes the process. "
    "Translation lookaside buffers cache recent translations so that most "
    "memory accesses avoid walking the page table at all.",
    "def lru_cache(maxsize):\n    cache = {}\n    order = []\n    def "
    "decorator(fn):\n        def wrapper(*args):\n            if args in "
    "cache:\n                order.remove(args)\n                "
    "order.append(args)\n                return cache[args]\n            "
    "value = fn(*args)\n            cache[args] = value\n            "
    "order.append(args)\n            if len(order) > maxsize:\n              "
    "  del cache[order.pop(0)]\n            return value\n        return "
    "wrapper\n    return decorator\n",
    "The lighthouse keeper climbed the spiral stairs every evening at dusk. "
    "The salt wind rattled the windows while he trimmed the wick and polished "
    "the great lens. Far out at sea a fishing boat turned toward the beam, and "
    "for a moment he imagined the crew watching the light sweep across the "
    "dark water, grateful for the steady rhythm of his work.",
    "| Country | Capital | Continent |\n|---|---|---|\n| France | Paris | "
    "Europe |\n| Japan | Tokyo | Asia |\n| Brazil | Brasilia | South America "
    "|\n| Kenya | Nairobi | Africa |\n| Canada | Ottawa | North America |\n| "
    "Australia | Canberra | Oceania |\n| Egypt | Cairo | Africa |\n| India | "
    "New Delhi | Asia |\n",
};

struct Session {
  std::unique_ptr<Executor> executor;
  std::vector<Token> tokens;
  std::uint32_t position{0};
  std::unique_ptr<gufo::hip::QwenGpuSnapshot> snapshot;
};

double MedianMs(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

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
    Expect(reader != nullptr, error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);

    constexpr std::size_t kMaxRows = 8;
    std::vector<Session> sessions(kTexts.size());
    for (std::size_t s = 0; s < sessions.size(); ++s) {
      auto& session = sessions[s];
      session.executor = Executor::Create(model, &error, 4096);
      Expect(session.executor != nullptr, error);
      session.tokens = session.executor->GetTokenizer().Encode(kTexts[s]);
      Expect(session.tokens.size() > kMaxRows + 16, "fixture text too short");
      session.position =
          static_cast<std::uint32_t>(session.tokens.size() - kMaxRows);
      session.executor->Reset();
      (void)session.executor->ForwardPromptBatch(
          std::span(session.tokens).first(session.position));
      session.snapshot = session.executor->SaveSnapshot(session.position);
    }

    constexpr int kWarmup = 3;
    constexpr int kIterations = 9;
    std::cout << "sessions rows total_rows median_ms ms_per_row\n";
    for (const std::size_t session_count : {1U, 2U, 4U}) {
      for (const std::size_t rows : {1U, 2U, 4U, 8U}) {
        std::vector<double> samples;
        for (int iteration = 0; iteration < kWarmup + kIterations;
             ++iteration) {
          for (std::size_t s = 0; s < session_count; ++s)
            sessions[s].executor->RestoreSnapshot(*sessions[s].snapshot);
          const auto start = std::chrono::steady_clock::now();
          if (session_count == 1) {
            (void)sessions[0].executor->ForwardVerificationChunk(
                std::span(sessions[0].tokens)
                    .subspan(sessions[0].position, rows),
                sessions[0].position, true);
          } else {
            std::vector<gufo::hip::QwenGpuVerificationItem> items;
            for (std::size_t s = 0; s < session_count; ++s) {
              items.push_back({sessions[s].executor.get(),
                               std::span(sessions[s].tokens)
                                   .subspan(sessions[s].position, rows),
                               sessions[s].position, true});
            }
            (void)Executor::ForwardVerificationBatch(items);
          }
          const std::chrono::duration<double, std::milli> elapsed =
              std::chrono::steady_clock::now() - start;
          if (iteration >= kWarmup)
            samples.push_back(elapsed.count());
        }
        const double median = MedianMs(samples);
        const std::size_t total = session_count * rows;
        std::cout << session_count << ' ' << rows << ' ' << total << ' '
                  << std::fixed << std::setprecision(2) << median << ' '
                  << median / static_cast<double>(total) << '\n';
      }
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "verify bench failed: " << exception.what() << '\n';
    return 1;
  }
}
