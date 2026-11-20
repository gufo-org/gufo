// End-to-end check of RAM-cache relief under host memory pressure (#932) on
// the real Flash-Next runner, without starving the host: the pool's host
// sample is replaced to report a shortfall, while the memory released, the
// time it takes and the outputs afterwards are measured for real.
//
//   qwen38_flash_next_host_pressure_probe MODEL.gguf [PROMPTS] [TOKENS]
//
// Fills the RAM cache with PROMPTS distinct prompts of about TOKENS tokens,
// reports MemAvailable and GTT, relieves a simulated 6 GiB shortfall at idle,
// replays every prompt (greedy outputs must match), then lets the cache
// regrow and replays again.
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/ext/host_budget.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace server = gufo::server;
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::uint64_t MemAvailableMiB() {
  std::ifstream file("/proc/meminfo");
  std::string key;
  std::uint64_t kib = 0;
  while (file >> key >> kib) {
    if (key == "MemAvailable:")
      return kib / 1024;
    file.ignore(64, '\n');
  }
  return 0;
}
std::uint64_t GttMiB() {
  std::ifstream file("/sys/class/drm/card1/device/mem_info_gtt_used");
  std::uint64_t bytes = 0;
  file >> bytes;
  return bytes / kMiB;
}
struct Outcome {
  std::vector<server::TextRunnerToken> tokens;
  std::size_t cached{};
  double ms{};
};
Outcome Run(server::TextRunnerPool& pool,
            const std::vector<server::TextRunnerToken>& prompt) {
  const auto start = Clock::now();
  auto request = pool.Acquire(prompt, gufo::sampling::SamplingConfig{});
  Outcome outcome{.cached = request.cached_prompt_tokens()};
  while (!request.prefill_complete())
    (void)request.Prefill(prompt.size());
  for (int i = 0; i < 8; ++i) {
    outcome.tokens.push_back(request.SelectNext().token);
    request.Advance();
  }
  (void)request.Commit();
  outcome.ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  return outcome;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc >= 2, "usage: MODEL.gguf [PROMPTS] [TOKENS]");
    const std::size_t prompts = argc > 2 ? std::stoul(argv[2]) : 6;
    const std::size_t length = argc > 3 ? std::stoul(argv[3]) : 16000;
    std::string error;
    auto model = qfn::Model::Load(
        argv[1], {.max_context = 65536, .decode_concurrency = 2}, &error);
    Require(bool(model), error);
    server::TextRunnerPool pool(
        qfn::CreateQwenFlashNextTextRunner(model, 65536), 2);
    std::atomic<std::uint64_t> shortfall{0};
    pool.SetHostSampler([&] {
      auto sample = gufo::ext::SampleHostMemory();
      // The reserve becomes CmaFree, `missing` above what is available.
      if (const auto missing = shortfall.load())
        sample.cma_free = sample.available + missing;
      return sample;
    });
    const auto pattern = model->Tokenize(
        "Virtual memory maps pages to physical storage on demand. A train "
        "travels sixty kilometers per hour through the quiet valley. ");
    std::vector<std::vector<server::TextRunnerToken>> inputs;
    for (std::size_t p = 0; p < prompts; ++p) {
      std::vector<server::TextRunnerToken> prompt;
      prompt.push_back(static_cast<server::TextRunnerToken>(1000 + p));
      for (std::size_t i = 0; prompt.size() < length; ++i)
        prompt.push_back(pattern[(i + 7 * p) % pattern.size()]);
      inputs.push_back(std::move(prompt));
    }
    std::vector<Outcome> first;
    for (const auto& prompt : inputs)
      first.push_back(Run(pool, prompt));
    // Idle without pressure: measures the tick's cost.
    auto start = Clock::now();
    for (int i = 0; i < 100; ++i)
      pool.Idle();
    std::cout << "idle_tick_us,"
              << std::chrono::duration<double, std::micro>(Clock::now() - start)
                         .count() /
                     100
              << '\n';
    const auto avail_before = MemAvailableMiB(), gtt_before = GttMiB();
    shortfall = std::uint64_t{6} << 30;
    start = Clock::now();
    pool.Idle();
    const double relief_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto avail_after = MemAvailableMiB(), gtt_after = GttMiB();
    std::cout << "relief_ms," << relief_ms << ",mem_available_mib,"
              << avail_before << "->" << avail_after << ",gtt_mib,"
              << gtt_before << "->" << gtt_after << '\n';
    // Pressure persists with the cache already given back: cheap again.
    start = Clock::now();
    pool.Idle();
    std::cout << "second_tick_ms,"
              << std::chrono::duration<double, std::milli>(Clock::now() - start)
                     .count()
              << '\n';
    shortfall = 0;
    for (int round = 0; round < 2; ++round) {
      for (std::size_t p = 0; p < inputs.size(); ++p) {
        const auto again = Run(pool, inputs[p]);
        std::cout << (round ? "regrown" : "relieved") << ",prompt," << p
                  << ",cached," << first[p].cached << "->" << again.cached
                  << ",ms," << first[p].ms << "->" << again.ms << '\n';
        Require(again.tokens == first[p].tokens,
                "outputs changed after relief, prompt " + std::to_string(p));
      }
      std::cout << "gtt_mib_after_round," << GttMiB() << ",mem_available_mib,"
                << MemAvailableMiB() << '\n';
    }
    std::cout << "host_pressure_probe: ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "host_pressure_probe: " << error.what() << '\n';
    return 1;
  }
}
