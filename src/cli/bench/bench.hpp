#ifndef GUFO_CLI_BENCH_HPP_
#define GUFO_CLI_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::cli {

struct BenchOptions {
  std::string model_path{"models/Qwen3.5-4B-BF16.gguf"};
  std::vector<std::size_t> n_prompts{64, 128, 512};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  int n_gpu_layers{99};
  std::string speculative_backend{""};
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  float draft_p_min{0.0F};
  bool verbose{false};
};

void PrintBenchHelp(std::string_view program_name);
std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg = nullptr);
int RunBench(std::span<const char* const> args);

}  // namespace gufo::cli

#endif  // GUFO_CLI_BENCH_HPP_
