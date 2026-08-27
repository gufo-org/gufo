#include "src/cli/bench/bench.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

#if defined(ENGINE_ENABLE_HRX)
#include "src/models/qwen/hrx/qwen_hrx_executor.hpp"
#endif

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/heterogeneous/npu_drafter.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/speculative/draft_heads.hpp"
#include "src/core/speculative/prompt_lookup_backend.hpp"
#include "src/core/speculative/self_speculative.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#endif

namespace gufo::cli {

void PrintBenchHelp(std::string_view program_name) {
  BenchOptions opt;
  gufo::cli::ArgParser parser(
      std::string(program_name) + " bench",
      "Benchmark prompt processing (pp) and token generation (tg) throughput.");

  parser.AddOption(
      "-m", "--model", "PATH",
      "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)", "Model",
      &opt.model_path);
  parser.AddOption("-ngl", "--n-gpu-layers", "N",
                   "Number of layers offloaded to GPU (default: 99)", "Model",
                   &opt.n_gpu_layers);
  parser.AddOption("", "--qwen-backend", "MODE",
                   "Qwen backend: hip or hrx-native (default: hip)", "Model",
                   &opt.qwen_backend);

  parser.AddOption("-p", "--n-prompt", "n,n,...",
                   "Prompt token lengths to benchmark (default: 64,128,512)",
                   "Workload", &opt.model_path);
  parser.AddOption(
      "-n", "--n-gen", "n,n,...",
      "Number of text generation tokens to benchmark (default: 128)",
      "Workload", &opt.model_path);
  parser.AddOption("-d", "--n-depth", "n,n,...",
                   "Context depths prepared before timed region (default: 0)",
                   "Workload", &opt.model_path);
  parser.AddOption(
      "-r", "--repetitions", "N",
      "Repetitions per test point for variance reduction (default: 1)",
      "Workload", &opt.repetitions);

  parser.AddOption("", "--validate-prefill", "N",
                   "Compare batched prefill logits against sequential "
                   "reference (TODO: deepseek)",
                   "Validation", &opt.model_path);
  parser.AddOption("", "--validate-hrx", "N",
                   "Compare native HRX logits and greedy decode against HIP",
                   "Validation", &opt.model_path);

  parser.AddOption("", "--speculative", "MODE",
                   "Draft backend: dflash, dflash2, mtp, mtp-npu, npu, pld, "
                   "self, or off",
                   "Speculative", &opt.speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddOption(
      "", "--draft-tokens", "N",
      "Maximum speculative draft tokens per verification step (default: 7)",
      "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--draft-policy", "MODE",
                   "Draft sizing: fixed, rolling, or accepted-ema (default: "
                   "rolling)",
                   "Speculative", &opt.draft_policy);
  parser.AddOption("", "--min-draft-tokens", "N",
                   "Adaptive draft floor (default: 1)", "Speculative",
                   &opt.min_draft_tokens);

  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);

  parser.PrintHelp();
}

namespace {

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << load_seconds << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

std::vector<std::size_t> ParseCommaSeparatedSizes(std::string_view str,
                                                  bool allow_zero = false) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start < str.size()) {
    auto end = str.find(',', start);
    if (end == std::string_view::npos) {
      end = str.size();
    }
    const auto part = str.substr(start, end - start);
    if (!part.empty()) {
      std::size_t val = 0;
      const auto [ptr, ec] =
          std::from_chars(part.data(), part.data() + part.size(), val);
      if (ec == std::errc{} && (val > 0 || allow_zero)) {
        result.push_back(val);
      }
    }
    start = end + 1;
  }
  return result;
}

#if defined(ENGINE_ENABLE_HIP)
struct BenchStats {
  double mean{0.0};
  double stddev{0.0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {.mean = 0.0, .stddev = 0.0};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::vector<tokenization::TokenId> MakeBenchmarkTokens(
    std::size_t count, std::size_t start_pos = 0) {
  std::vector<tokenization::TokenId> tokens(count);
  for (std::size_t i = 0; i < count; ++i) {
    tokens[i] =
        static_cast<tokenization::TokenId>(((start_pos + i) % 1000) + 100);
  }
  return tokens;
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

bool ValidatePrefill(hip::QwenGpuExecutor& executor,
                     std::size_t prompt_length) {
  if (prompt_length == 0 || prompt_length > executor.GetMaxPromptBatch()) {
    std::cerr << "Error: --validate-prefill must be between 1 and "
              << executor.GetMaxPromptBatch() << " tokens.\n";
    return false;
  }

  const auto prompt_tokens = MakeBenchmarkTokens(prompt_length);

  executor.Reset();
  tokenization::TokenId sequential_token = 0;
  for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
    sequential_token =
        executor.ForwardToken(prompt_tokens[i], static_cast<std::uint32_t>(i),
                              i + 1 == prompt_tokens.size());
  }
  const auto sequential_logits_view = executor.CopyLastLogits();
  const std::vector<float> sequential_logits(sequential_logits_view.begin(),
                                             sequential_logits_view.end());

  executor.Reset();
  const auto batched_token = executor.ForwardPromptBatch(prompt_tokens);
  const auto batched_logits = executor.CopyLastLogits();
  const auto comparison =
      testing::CompareLogits(sequential_logits, batched_logits);

  std::cout << std::fixed << std::setprecision(8)
            << "[Prefill Validation] tokens=" << prompt_length
            << " sequential_top1=" << sequential_token
            << " batched_top1=" << batched_token
            << " top1_match=" << (comparison.top1_match ? "yes" : "no")
            << " finite=" << (comparison.finite ? "yes" : "no") << '\n'
            << "  max_abs_diff=" << comparison.max_abs_diff
            << " mean_abs_diff=" << comparison.mean_abs_diff
            << " rmse=" << comparison.root_mean_square_error
            << " cosine_similarity=" << comparison.cosine_similarity << '\n';

  return comparison.finite && comparison.top1_match &&
         sequential_token == batched_token;
}

#if defined(ENGINE_ENABLE_HRX)
constexpr float kHrxParityMaxRmse = 1.0e-4F;
constexpr float kHrxParityMinCosine = 0.999999F;
constexpr std::size_t kHrxValidationPromptLength = 4;

struct HrxParityStep {
  tokenization::TokenId input{0};
  tokenization::TokenId next_token{0};
  std::vector<float> logits;
};

std::vector<HrxParityStep> CaptureHrxReference(hip::QwenGpuExecutor& reference,
                                               std::size_t generation_length) {
  const auto prompt = MakeBenchmarkTokens(kHrxValidationPromptLength);
  std::vector<HrxParityStep> fixture;
  fixture.reserve(prompt.size() + generation_length);
  reference.Reset();

  const auto capture = [&reference, &fixture](tokenization::TokenId input,
                                              std::uint32_t position) {
    const auto next_token = reference.ForwardToken(input, position, true);
    const auto logits_view = reference.CopyLastLogits();
    fixture.push_back(
        {.input = input,
         .next_token = next_token,
         .logits = std::vector<float>(logits_view.begin(), logits_view.end())});
    return next_token;
  };

  tokenization::TokenId next_token = 0;
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    next_token = capture(prompt[index], static_cast<std::uint32_t>(index));
  }
  for (std::size_t step = 0; step < generation_length; ++step) {
    const auto position = static_cast<std::uint32_t>(prompt.size() + step);
    next_token = capture(next_token, position);
  }
  return fixture;
}

bool ValidateHrxStep(const HrxParityStep& reference,
                     hrx::QwenHrxExecutor& candidate, std::uint32_t position,
                     std::string_view phase, std::size_t step,
                     std::string* error_msg) {
  const auto candidate_token =
      candidate.ForwardToken(reference.input, position, true, error_msg);

  if (!candidate_token.has_value()) {
    std::cerr << "Error: native HRX " << phase << " step " << step
              << " failed: " << *error_msg << '\n';
    return false;
  }
  const auto candidate_logits = candidate.CopyLastLogits(error_msg);
  if (candidate_logits.empty()) {
    std::cerr << "Error: native HRX " << phase << " step " << step
              << " logit readback failed: " << *error_msg << '\n';
    return false;
  }

  const auto comparison =
      testing::CompareLogits(reference.logits, candidate_logits);
  const bool token_match = reference.next_token == *candidate_token;
  const bool within_envelope =
      comparison.root_mean_square_error <= kHrxParityMaxRmse &&
      comparison.cosine_similarity >= kHrxParityMinCosine;
  std::cout << std::fixed << std::setprecision(8)
            << "[HRX Validation] phase=" << phase << " step=" << step
            << " position=" << position
            << " reference_top1=" << reference.next_token
            << " candidate_top1=" << *candidate_token
            << " top1_match=" << (comparison.top1_match ? "yes" : "no")
            << " finite=" << (comparison.finite ? "yes" : "no") << '\n'
            << "  max_abs_diff=" << comparison.max_abs_diff
            << " mean_abs_diff=" << comparison.mean_abs_diff
            << " rmse=" << comparison.root_mean_square_error
            << " cosine_similarity=" << comparison.cosine_similarity
            << " envelope=" << (within_envelope ? "pass" : "fail") << '\n';

  if (!comparison.finite || !comparison.top1_match || !token_match ||
      !within_envelope) {
    std::cerr << "Error: native HRX diverged from HIP during " << phase
              << " step " << step << ".\n";
    return false;
  }
  return true;
}

bool ValidateHrxParity(std::span<const HrxParityStep> reference,
                       hrx::QwenHrxExecutor& candidate,
                       std::string* error_msg) {
  if (!candidate.Reset(error_msg)) {
    std::cerr << "Error resetting native HRX executor: " << *error_msg << '\n';
    return false;
  }

  for (std::size_t position = 0; position < reference.size(); ++position) {
    const bool prompt = position < kHrxValidationPromptLength;
    const std::size_t step =
        prompt ? position : position - kHrxValidationPromptLength;
    if (!ValidateHrxStep(reference[position], candidate,
                         static_cast<std::uint32_t>(position),
                         prompt ? "prompt" : "decode", step, error_msg)) {
      return false;
    }
  }

  std::cout << "[HRX Validation] PASS: position-zero, "
            << kHrxValidationPromptLength << "-token prompt, and "
            << reference.size() - kHrxValidationPromptLength
            << " greedy decode steps match HIP.\n";
  return true;
}
#endif

bool IsDeepSeekV4Flash(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "deepseek4";
}

std::vector<int> MakeDeepSeekBenchmarkTokens(
    const models::deepseek_v4_flash::Model& model, std::size_t count) {
  const auto pattern = model.Tokenize(
      "The quick brown fox jumps over the lazy dog. "
      "Strix Halo executes this deterministic benchmark sequence. ");
  if (pattern.empty()) {
    throw std::runtime_error("DeepSeek benchmark token pattern is empty");
  }
  std::vector<int> tokens(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = pattern[index % pattern.size()];
  }
  return tokens;
}

int RunDeepSeekBenchmark(
    const BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for DeepSeek V4 Flash\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_test_tokens =
      std::max(max_or_zero(options.n_prompts), max_or_zero(options.n_gens));
  const std::size_t required_context =
      std::max<std::size_t>(4096, max_depth + max_test_tokens + 1);
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: requested DeepSeek benchmark context is too large\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  if (options.validate_prefill_tokens != 0) {
    std::cerr << "Error: DeepSeek prefill oracle validation is not available "
                 "through --validate-prefill\n";
    return 1;
  }
  if (!options.speculative_backend.empty()) {
    std::cerr << "Error: DeepSeek speculative benchmark modes are not yet "
                 "implemented\n";
    return 1;
  }

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      options.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .prefill_chunk = 2048,
          .power_percent = 100,
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto tokens = MakeDeepSeekBenchmarkTokens(*model, required_context);
  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();

  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";

  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  std::unique_ptr<models::deepseek_v4_flash::Session> prepared_session;
  std::size_t prepared_depth = 0;
  for (const std::size_t depth : options.n_depths) {
    std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
    if (depth > 0) {
      if (prepared_session == nullptr || depth < prepared_depth) {
        prepared_session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        prepared_depth = 0;
      }
      if (prepared_session == nullptr ||
          !prepared_session->Sync(std::span(tokens.data(), depth), &error)) {
        std::cerr << "Error preparing DeepSeek depth " << depth << ": " << error
                  << '\n';
        return 1;
      }
      prepared_depth = depth;
      snapshot = prepared_session->SaveSnapshot(&error);
      if (snapshot == nullptr) {
        std::cerr << "Error snapshotting DeepSeek depth " << depth << ": "
                  << error << '\n';
        return 1;
      }
      if (options.verbose) {
        std::cerr << "Prepared DeepSeek depth " << depth
                  << " snapshot_bytes=" << snapshot->SizeBytes() << '\n';
      }
    }

    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: DeepSeek prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      runs.reserve(options.repetitions);
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
        models::deepseek_v4_flash::Session* session = prepared_session.get();
        if (depth == 0) {
          local_session = model->CreateSession(
              static_cast<std::uint32_t>(required_context), &error);
          session = local_session.get();
        } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
          std::cerr << "Error restoring DeepSeek depth: " << error << '\n';
          return 1;
        }
        if (session == nullptr) {
          std::cerr << "Error creating DeepSeek session: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens.data(), depth + prompt_length),
                           &error)) {
          std::cerr << "Error running DeepSeek prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(prompt_length) / seconds);
        if (options.verbose) {
          std::cerr << "DeepSeek pp depth=" << depth
                    << " payload_bytes=" << session->PayloadBytes() << '\n';
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: DeepSeek generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      runs.reserve(options.repetitions);
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        std::unique_ptr<models::deepseek_v4_flash::Session> local_session;
        models::deepseek_v4_flash::Session* session = prepared_session.get();
        if (depth == 0) {
          local_session = model->CreateSession(
              static_cast<std::uint32_t>(required_context), &error);
          session = local_session.get();
          if (session != nullptr &&
              !session->Sync(std::span(tokens.data(), prefix_length), &error)) {
            session = nullptr;
          }
        } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
          session = nullptr;
        }
        if (session == nullptr) {
          std::cerr << "Error preparing DeepSeek generation: " << error << '\n';
          return 1;
        }

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < generation_length; ++step) {
          const int token = session->SelectNextExcluding(model->EosToken());
          if (token < 0 || !session->Evaluate(token, &error)) {
            std::cerr << "Error running DeepSeek decode: " << error << '\n';
            return 1;
          }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          std::cerr << "DeepSeek tg depth=" << depth
                    << " payload_bytes=" << session->PayloadBytes() << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }

    if (depth > 0 && !prepared_session->RestoreSnapshot(*snapshot, &error)) {
      std::cerr << "Error restoring final DeepSeek depth: " << error << '\n';
      return 1;
    }
  }

  std::cout << '\n';
  return 0;
}
#endif

}  // namespace

std::optional<BenchOptions> ParseBenchOptions(std::span<const char* const> args,
                                              std::string* error_msg) {
  BenchOptions opt;
  bool explicit_p = false;
  bool explicit_n = false;

  gufo::cli::ArgParser parser(
      "gufo bench",
      "Benchmark prompt processing (pp) and token generation (tg) "
      "throughput.");
  parser.AddOption(
      "-m", "--model", "PATH",
      "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)", "Model",
      &opt.model_path);
  parser.AddOption("-ngl", "--n-gpu-layers", "N",
                   "Number of layers offloaded to GPU (default: 99)", "Model",
                   &opt.n_gpu_layers);
  parser.AddCustomOption(
      "", "--qwen-backend", "MODE",
      "Qwen backend: hip or hrx-native (default: hip)", "Model",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        if (value != "hip" && value != "hrx-native") {
          if (error != nullptr) {
            *error = "Invalid Qwen backend: " + std::string(value);
          }
          return false;
        }
        opt.qwen_backend = value;
        return true;
      });

  parser.AddCustomOption(
      "-p", "--n-prompt", "n,n,...",
      "Prompt token lengths to benchmark (default: 64,128,512)", "Workload",
      [&opt, &explicit_p](std::string_view, std::string_view val,
                          std::string*) -> bool {
        opt.n_prompts = ParseCommaSeparatedSizes(val);
        explicit_p = true;
        return true;
      });

  parser.AddCustomOption(
      "-n", "--n-gen", "n,n,...",
      "Number of text generation tokens to benchmark (default: 128)",
      "Workload",
      [&opt, &explicit_n](std::string_view, std::string_view val,
                          std::string*) -> bool {
        opt.n_gens = ParseCommaSeparatedSizes(val);
        explicit_n = true;
        return true;
      });

  parser.AddCustomOption(
      "-d", "--n-depth", "n,n,...",
      "Context depths prepared before timed region (default: 0)", "Workload",
      [&opt](std::string_view flag_name, std::string_view val,
             std::string* err) -> bool {
        opt.n_depths = ParseCommaSeparatedSizes(val, true);
        if (opt.n_depths.empty()) {
          if (err != nullptr) {
            *err = "Invalid argument for " + std::string(flag_name);
          }
          return false;
        }
        return true;
      });

  parser.AddOption(
      "-r", "--repetitions", "N",
      "Repetitions per test point for variance reduction (default: 1)",
      "Workload", &opt.repetitions);

  parser.AddCustomOption(
      "", "--validate-prefill", "N",
      "Compare batched prefill logits against sequential reference (TODO: "
      "deepseek)",
      "Validation",
      [&opt](std::string_view, std::string_view val, std::string* err) -> bool {
        std::size_t num = 0;
        const auto [ptr, ec] =
            std::from_chars(val.data(), val.data() + val.size(), num);
        if (ec != std::errc{} || ptr != val.data() + val.size() || num == 0) {
          if (err != nullptr) {
            *err = "Invalid argument for --validate-prefill";
          }
          return false;
        }
        opt.validate_prefill_tokens = num;
        return true;
      });
  parser.AddCustomOption(
      "", "--validate-hrx", "N",
      "Compare native HRX logits and greedy decode against HIP", "Validation",
      [&opt](std::string_view, std::string_view val, std::string* err) -> bool {
        std::size_t num = 0;
        const auto [ptr, ec] =
            std::from_chars(val.data(), val.data() + val.size(), num);
        if (ec != std::errc{} || ptr != val.data() + val.size() || num == 0 ||
            num > std::numeric_limits<std::uint32_t>::max() - 4) {
          if (err != nullptr) {
            *err = "Invalid argument for --validate-hrx";
          }
          return false;
        }
        opt.validate_hrx_tokens = num;
        return true;
      });

  const auto parse_speculative_backend =
      [&opt](std::string_view, std::string_view value, std::string*) -> bool {
    if (value == "none" || value == "off" || value == "false" ||
        value == "disabled") {
      opt.speculative_backend.clear();
    } else {
      opt.speculative_backend = value;
    }
    return true;
  };
  parser.AddCustomOption(
      "", "--speculative", "MODE",
      "Draft backend: dflash, dflash2, mtp, mtp-npu, npu, pld, self, or off",
      "Speculative", parse_speculative_backend);
  parser.AddCustomOption("", "--speculative-decoding", "MODE",
                         "Alias for --speculative", "Speculative",
                         parse_speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddCustomOption(
      "", "--draft-tokens", "N",
      "Maximum speculative draft tokens per verification step (default: 7)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid argument for --draft-tokens";
          }
          return false;
        }
        opt.draft_tokens = count;
        return true;
      });
  parser.AddCustomOption(
      "", "--draft-policy", "MODE",
      "Draft sizing: fixed, rolling, or accepted-ema (default: rolling)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        if (value != "fixed" && value != "rolling" && value != "accepted-ema") {
          if (error != nullptr) {
            *error = "Invalid draft policy: " + std::string(value);
          }
          return false;
        }
        opt.draft_policy = value;
        return true;
      });
  parser.AddCustomOption(
      "", "--min-draft-tokens", "N", "Adaptive draft floor (default: 1)",
      "Speculative",
      [&opt](std::string_view, std::string_view value,
             std::string* error) -> bool {
        std::uint32_t count = 0;
        const auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), count);
        if (ec != std::errc{} || ptr != value.data() + value.size() ||
            count == 0) {
          if (error != nullptr) {
            *error = "Invalid argument for --min-draft-tokens";
          }
          return false;
        }
        opt.min_draft_tokens = count;
        return true;
      });
  parser.AddFlag("-v", "--verbose",
                 "Print detailed timing, latency breakdown, and tok/s metrics",
                 "General", &opt.verbose);

  parser.SetPositionalHandler(
      [&opt](std::string_view arg, std::string*) -> bool {
        opt.model_path = std::string(arg);
        return true;
      });

  if (!parser.Parse(args, error_msg)) {
    return std::nullopt;
  }
  if (parser.IsHelpRequested()) {
    return std::nullopt;
  }

  if (explicit_p && !explicit_n) {
    opt.n_gens.clear();
  } else if (!explicit_p && explicit_n) {
    opt.n_prompts.clear();
  }

  if (opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }
  if (opt.validate_hrx_tokens > 0 && opt.qwen_backend != "hrx-native") {
    if (error_msg != nullptr) {
      *error_msg = "--validate-hrx requires --qwen-backend hrx-native";
    }
    return std::nullopt;
  }

  return opt;
}

int RunBench(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseBenchOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintBenchHelp("gufo");
      return 2;
    }
    PrintBenchHelp("gufo");
    return 0;
  }

  const auto& opt = *opt_res;

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = gufo::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const gufo::core::GgufReader> reader(
      std::move(reader_owner));

  if (opt.qwen_backend == "hrx-native") {
    if (!opt.speculative_backend.empty() || opt.validate_prefill_tokens != 0 ||
        opt.n_depths != std::vector<std::size_t>{0} || opt.n_gpu_layers != 99 ||
        opt.repetitions == 0) {
      std::cerr << "Error: the bare-minimum hrx-native backend supports only "
                   "depth 0, all layers, positive repetitions, greedy "
                   "sequential execution, and no prefill validation or "
                   "speculative decoding\n";
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
#if defined(ENGINE_ENABLE_HRX)
    const auto max_or_zero = [](const std::vector<std::size_t>& values) {
      return values.empty() ? std::size_t{0}
                            : *std::max_element(values.begin(), values.end());
    };
    constexpr std::size_t kGenerationPrimeTokens = 16;
    constexpr std::size_t kValidationPromptTokens = 4;
    const std::size_t max_prompt = max_or_zero(opt.n_prompts);
    const std::size_t max_generation = max_or_zero(opt.n_gens);
    if (max_generation >
        std::numeric_limits<std::uint32_t>::max() - kGenerationPrimeTokens) {
      std::cerr << "Error: requested native HRX context is too large.\n";
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    const std::size_t required_context =
        std::max({max_prompt, kGenerationPrimeTokens + max_generation,
                  kValidationPromptTokens + opt.validate_hrx_tokens});
    if (required_context == 0 ||
        required_context > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Error: native HRX benchmark requires a non-zero context.\n";
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }

#if defined(ENGINE_ENABLE_HIP)
    std::vector<HrxParityStep> hrx_reference;
    if (opt.validate_hrx_tokens > 0) {
      std::cout << "[HRX Validation] Capturing HIP reference...\n"
                << std::flush;
      auto reference_executor = hip::QwenGpuExecutor::CreateFromGguf(
          reader, &err, static_cast<std::uint32_t>(required_context));
      if (reference_executor == nullptr) {
        std::cerr << "Error creating HIP parity executor: " << err << '\n';
        PrintModelLoadTime(model_load_start, false);
        return 1;
      }
      hrx_reference =
          CaptureHrxReference(*reference_executor, opt.validate_hrx_tokens);
      reference_executor.reset();
      std::cout << "[HRX Validation] HIP reference released; starting native "
                   "HRX...\n"
                << std::flush;
    }
#endif

    auto native_executor = hrx::QwenHrxExecutor::CreateFromGguf(
        reader, static_cast<std::uint32_t>(required_context),
        GUFO_HRX_KERNEL_DIR, &err);
    if (native_executor == nullptr) {
      std::cerr << "Error creating native HRX Qwen executor: " << err << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    if (!native_executor->ModelExecutionReady()) {
      std::cerr << "Error: native HRX Q8_0 execution is not complete. Missing "
                   "capabilities:\n";
      for (const auto& capability :
           native_executor->MissingModelCapabilities()) {
        std::cerr << "  - " << capability << '\n';
      }
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    PrintModelLoadTime(model_load_start);
    std::cout << "backend=HRX-native model="
              << native_executor->GetConfig().model_name << '\n';

    if (opt.validate_hrx_tokens > 0) {
#if defined(ENGINE_ENABLE_HIP)
      return ValidateHrxParity(hrx_reference, *native_executor, &err) ? 0 : 1;
#else
      std::cerr << "Error: --validate-hrx requires ENGINE_ENABLE_HIP\n";
      return 1;
#endif
    }

    const auto print_native_result = [](std::string_view name,
                                        const std::vector<double>& runs) {
      const auto stats = ComputeStats(runs);
      std::cout << name << ": " << std::fixed << std::setprecision(2)
                << stats.mean << " +/- " << stats.stddev << " t/s\n";
    };
    for (const std::size_t prompt_length : opt.n_prompts) {
      if (prompt_length == 0) {
        std::cerr << "Error: native HRX prompt length must be non-zero.\n";
        return 1;
      }
      const auto tokens = MakeBenchmarkTokens(prompt_length);
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < opt.repetitions;
           ++repetition) {
        if (!native_executor->Reset(&err)) {
          std::cerr << "Error resetting native HRX executor: " << err << '\n';
          return 1;
        }
        const auto begin = std::chrono::high_resolution_clock::now();
        const auto next =
            native_executor->ForwardPromptBatch(tokens, 0, true, &err);
        const auto end = std::chrono::high_resolution_clock::now();
        if (!next.has_value()) {
          std::cerr << "Error in native HRX prompt execution: " << err << '\n';
          return 1;
        }
        const double seconds =
            std::chrono::duration<double>(end - begin).count();
        if (seconds > 0.0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_native_result(MakeTestName("pp", prompt_length, 0), runs);
    }

    for (const std::size_t generation_length : opt.n_gens) {
      if (generation_length == 0) {
        std::cerr << "Error: native HRX generation length must be non-zero.\n";
        return 1;
      }
      const auto prime = MakeBenchmarkTokens(kGenerationPrimeTokens);
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < opt.repetitions;
           ++repetition) {
        if (!native_executor->Reset(&err)) {
          std::cerr << "Error resetting native HRX executor: " << err << '\n';
          return 1;
        }
        auto current =
            native_executor->ForwardPromptBatch(prime, 0, true, &err);
        if (!current.has_value()) {
          std::cerr << "Error priming native HRX generation: " << err << '\n';
          return 1;
        }
        std::uint32_t position = kGenerationPrimeTokens;
        const auto begin = std::chrono::high_resolution_clock::now();
        for (std::size_t step = 0; step < generation_length; ++step) {
          current =
              native_executor->ForwardToken(*current, position, true, &err);
          if (!current.has_value()) {
            std::cerr << "Error in native HRX generation: " << err << '\n';
            return 1;
          }
          ++position;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double seconds =
            std::chrono::duration<double>(end - begin).count();
        if (seconds > 0.0) {
          runs.push_back(static_cast<double>(generation_length) / seconds);
        }
      }
      print_native_result(MakeTestName("tg", generation_length, 0), runs);
    }
    std::cout << '\n';
    return 0;
#else
    std::cerr << "Error: --qwen-backend hrx-native requires "
                 "ENGINE_ENABLE_HRX\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
#endif
  }

#if defined(ENGINE_ENABLE_HIP)
  if (IsDeepSeekV4Flash(*reader)) {
    return RunDeepSeekBenchmark(opt, reader, model_load_start);
  }

  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: No HIP GPU devices available for benchmarking.\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const double total_vram_mib =
      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);

  std::cout << "ggml_cuda_init: found " << device_count
            << " ROCm devices (Total VRAM: "
            << static_cast<std::size_t>(total_vram_mib) << " MiB):\n"
            << "  Device 0: " << prop.name << ", " << prop.gcnArchName
            << ", Wave Size: " << prop.warpSize
            << ", VRAM: " << static_cast<std::size_t>(total_vram_mib)
            << " MiB\n";

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(opt.n_depths);
  const std::size_t max_test_tokens =
      std::max({max_or_zero(opt.n_prompts), max_or_zero(opt.n_gens),
                opt.validate_prefill_tokens});
  if (max_depth > std::numeric_limits<std::uint32_t>::max() - max_test_tokens) {
    std::cerr << "Error: requested benchmark context is too large.\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::size_t required_context =
      std::max<std::size_t>(4096, max_depth + max_test_tokens);

  auto gpu_exec = hip::QwenGpuExecutor::CreateFromGguf(
      reader, &err, static_cast<std::uint32_t>(required_context));
  if (!gpu_exec) {
    std::cerr << "Error creating Qwen GPU executor: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto& config = gpu_exec->GetConfig();
  std::shared_ptr<const hip::QwenMtpGpuModel> mtp_gpu_model;
  if (opt.speculative_backend == "mtp" ||
      opt.speculative_backend == "mtp-npu") {
    std::string mtp_path = opt.mtp_model_path;
    if (mtp_path.empty()) {
      if (const char* environment = std::getenv("GUFO_MTP_MODEL");
          environment != nullptr) {
        mtp_path = environment;
      }
    }
    auto mtp_reader_owner = core::GgufReader::OpenFile(mtp_path, &err);
    if (mtp_reader_owner == nullptr) {
      std::cerr << "Error loading MTP GGUF model: " << err << '\n';
      return 1;
    }
    std::shared_ptr<const core::GgufReader> mtp_reader(
        std::move(mtp_reader_owner));
    mtp_gpu_model = hip::QwenMtpGpuModel::Create(
        std::move(mtp_reader), gpu_exec->GetSharedModel(), &err);
    if (mtp_gpu_model == nullptr) {
      std::cerr << "Error creating GPU MTP model: " << err << '\n';
      return 1;
    }
    if (opt.verbose) {
      std::cerr << "Packed GPU MTP view: "
                << mtp_gpu_model->GetPackedWeightBytes() << " bytes in "
                << mtp_gpu_model->GetPackTimeSeconds() << " s\n";
    }
  }
  const std::string model_name = config.model_name + " BF16";
  const double model_size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  std::uint64_t parameter_count = 0;
  for (const auto& tensor : reader->GetTensors()) {
    parameter_count += tensor.ElementCount();
  }
  const double model_params_b =
      static_cast<double>(parameter_count) / 1'000'000'000.0;

  if (opt.validate_prefill_tokens > 0 &&
      !ValidatePrefill(*gpu_exec, opt.validate_prefill_tokens)) {
    return 1;
  }

  std::cout << "| " << std::left << std::setw(30) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::right << std::setw(10) << "params"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(3) << "ngl"
            << " | " << std::right << std::setw(15) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n";
  std::cout << "| " << std::string(30, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(10, '-')
            << " | " << std::string(3, '-') << " | " << std::string(15, '-')
            << " | " << std::string(21, '-') << " |\n";

  // Warmup run
  {
    const auto warmup_tokens = MakeBenchmarkTokens(32);
    gpu_exec->Reset();
    (void)gpu_exec->ForwardPromptBatch(warmup_tokens);
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::ostringstream ss_size, ss_params;
  ss_size << std::fixed << std::setprecision(2) << model_size_gib << " GiB";
  ss_params << std::fixed << std::setprecision(2) << model_params_b << " B";

  const std::string_view backend_name = opt.speculative_backend == "mtp-npu"
                                            ? "HIP+XDNA2"
#if defined(ENGINE_ENABLE_HRX)
                                            : "HRX (HIP)";
#else
                                            : "ROCm (HIP)";
#endif
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream ss_ts;
    ss_ts << std::fixed << std::setprecision(2) << stats.mean << " ± "
          << stats.stddev;

    std::cout << "| " << std::left << std::setw(30) << model_name << " | "
              << std::right << std::setw(10) << ss_size.str() << " | "
              << std::right << std::setw(10) << ss_params.str() << " | "
              << std::left << std::setw(10) << backend_name << " | "
              << std::right << std::setw(3) << opt.n_gpu_layers << " | "
              << std::right << std::setw(15) << test_name << " | " << std::right
              << std::setw(21) << ss_ts.str() << " |\n"
              << std::flush;
  };

  std::size_t prepared_depth = 0;
  bool has_prepared_depth = false;
  tokenization::TokenId prepared_next_token = 0;
  bool has_prepared_next_token = false;
  for (const std::size_t depth : opt.n_depths) {
    if (depth > 0) {
      if (opt.verbose) {
        std::cerr << "Preparing context depth " << depth << "...\n";
      }
      std::size_t preparation_start = 0;
      double restore_ms = 0.0;
      if (has_prepared_depth && depth >= prepared_depth) {
        const auto restore_start = std::chrono::steady_clock::now();
        gpu_exec->RestoreState();
        const auto restore_end = std::chrono::steady_clock::now();
        restore_ms = std::chrono::duration<double, std::milli>(restore_end -
                                                               restore_start)
                         .count();
        preparation_start = prepared_depth;
      } else {
        gpu_exec->Reset();
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto depth_tokens =
          MakeBenchmarkTokens(depth - preparation_start, preparation_start);
      const auto preparation_begin = std::chrono::steady_clock::now();
      const bool compute_next_token = !opt.n_gens.empty();
      if (!depth_tokens.empty()) {
        const auto next_token = gpu_exec->ForwardPromptBatch(
            depth_tokens, static_cast<std::uint32_t>(preparation_start),
            compute_next_token);
        if (compute_next_token) {
          prepared_next_token = next_token;
          has_prepared_next_token = true;
        }
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto preparation_end = std::chrono::steady_clock::now();
      const auto cache_begin = preparation_end;
      gpu_exec->SaveState(static_cast<std::uint32_t>(depth));
      const auto cache_end = std::chrono::steady_clock::now();
      if (opt.verbose) {
        const double preparation_seconds =
            std::chrono::duration<double>(preparation_end - preparation_begin)
                .count();
        const double cache_ms =
            std::chrono::duration<double, std::milli>(cache_end - cache_begin)
                .count();
        std::cerr << "Prepared " << (depth - preparation_start)
                  << " new tokens in " << std::fixed << std::setprecision(3)
                  << preparation_seconds << " s; cached depth " << depth
                  << " in " << std::setprecision(2) << cache_ms << " ms";
        if (preparation_start > 0) {
          std::cerr << " after restoring depth " << preparation_start << " in "
                    << restore_ms << " ms";
        }
        std::cerr << "\n";
      }
      prepared_depth = depth;
      has_prepared_depth = true;
    }

    const auto restore_depth = [&] {
      if (depth == 0) {
        gpu_exec->Reset();
      } else {
        gpu_exec->RestoreState();
      }
      HIP_CHECK(hipDeviceSynchronize());
    };

    for (const std::size_t p_len : opt.n_prompts) {
      const auto prompt_tokens = MakeBenchmarkTokens(p_len, depth);

      restore_depth();
      (void)gpu_exec->ForwardPromptBatch(prompt_tokens,
                                         static_cast<std::uint32_t>(depth));
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<double> runs;
      runs.reserve(opt.repetitions);
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        const auto t0 = std::chrono::high_resolution_clock::now();
        (void)gpu_exec->ForwardPromptBatch(prompt_tokens,
                                           static_cast<std::uint32_t>(depth));
        HIP_CHECK(hipDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (elapsed_sec > 0.0) {
          runs.push_back(static_cast<double>(p_len) / elapsed_sec);
        }
      }

      print_result(MakeTestName("pp", p_len, depth), ComputeStats(runs));
    }

    for (const std::size_t g_len : opt.n_gens) {
      restore_depth();
      (void)gpu_exec->ForwardToken(
          static_cast<tokenization::TokenId>((depth % 1000) + 100),
          static_cast<std::uint32_t>(depth));
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<double> runs;
      runs.reserve(opt.repetitions);
      std::vector<double> acceptance_runs;
      acceptance_runs.reserve(opt.repetitions);
      std::vector<double> hybrid_gpu_to_host_runs;
      std::vector<double> hybrid_activation_pack_runs;
      std::vector<double> hybrid_command_runs;
      std::vector<double> hybrid_end_to_end_runs;
      std::vector<double> hybrid_host_to_gpu_runs;
      heterogeneous::NpuDrafterMetrics npu_summary_metrics{};
      bool has_npu_metrics = false;
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        std::unique_ptr<speculative::IDraftBackend> draft_backend;
        hip::QwenMtpGpuDraftBackend* mtp_backend = nullptr;
        heterogeneous::NpuDraftBackend* npu_draft_backend = nullptr;
        if (opt.speculative_backend == "dflash" ||
            opt.speculative_backend == "dflash2" ||
            opt.speculative_backend == "dflash-2") {
          std::string dflash_path = opt.dflash_model_path;
          if (dflash_path.empty()) {
            if (const char* env = std::getenv("GUFO_DFLASH_MODEL");
                env != nullptr) {
              dflash_path = env;
            }
          }
          if (dflash_path.empty()) {
            dflash_path = opt.model_path;
          }
          hip::QwenDFlashGpuDraftConfig cfg{
              .max_context = static_cast<std::uint32_t>(required_context),
              .max_draft_tokens = opt.draft_tokens,
          };
          draft_backend = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
              dflash_path, gpu_exec->GetSharedModel(), cfg, &err);
          if (draft_backend == nullptr) {
            throw std::runtime_error("DFlash initialization failed: " + err);
          }
        } else if (opt.speculative_backend == "mtp" ||
                   opt.speculative_backend == "mtp-npu") {
          hip::QwenMtpGpuDraftConfig cfg{
              .max_context = static_cast<std::uint32_t>(required_context),
              .max_draft_tokens = opt.draft_tokens,
              .execution_mode =
                  opt.speculative_backend == "mtp-npu"
                      ? hip::QwenMtpExecutionMode::kHybridNpuEhProj
                      : hip::QwenMtpExecutionMode::kGpu,
          };
          draft_backend =
              hip::QwenMtpGpuDraftBackend::Create(mtp_gpu_model, cfg, &err);
          if (draft_backend == nullptr) {
            throw std::runtime_error("MTP initialization failed: " + err);
          }
          mtp_backend =
              static_cast<hip::QwenMtpGpuDraftBackend*>(draft_backend.get());
        } else if (opt.speculative_backend == "self") {
          speculative::SelfSpeculativeConfig cfg;
          cfg.total_layers = config.num_layers;
          cfg.exit_layer = std::max<std::uint32_t>(4U, config.num_layers / 4);
          cfg.draft_step_count = opt.draft_tokens;
          draft_backend =
              std::make_unique<speculative::SelfSpeculativeBackend>(cfg);
        } else if (opt.speculative_backend == "npu" ||
                   opt.speculative_backend == "dflash-npu") {
          heterogeneous::NpuDrafterConfig cfg;
          cfg.mode = (opt.speculative_backend == "dflash-npu" ||
                      !opt.dflash_model_path.empty())
                         ? heterogeneous::NpuDraftMode::kDFlash2
                         : heterogeneous::NpuDraftMode::kMTP;
          if (!opt.dflash_model_path.empty()) {
            cfg.dflash_model_path = opt.dflash_model_path;
          }
          if (!opt.mtp_model_path.empty()) {
            cfg.mtp_model_path = opt.mtp_model_path;
          }
          cfg.max_draft_tokens = opt.draft_tokens;
          cfg.vocab_size = config.vocab_size;
          draft_backend = std::make_unique<heterogeneous::NpuDraftBackend>(cfg);
          npu_draft_backend =
              static_cast<heterogeneous::NpuDraftBackend*>(draft_backend.get());
        } else if (opt.speculative_backend == "pld" ||
                   opt.speculative_backend == "lookup") {
          speculative::PromptLookupConfig cfg;
          cfg.max_draft_tokens = opt.draft_tokens;
          draft_backend =
              std::make_unique<speculative::PromptLookupDraftBackend>(cfg);
        }

        std::unique_ptr<speculative::SpeculativeVerifier> spec_verifier;
        if (draft_backend) {
          speculative::SpeculativeOptions s_opts;
          s_opts.max_draft_tokens = opt.draft_tokens;
          s_opts.min_draft_tokens = opt.min_draft_tokens;
          s_opts.initial_draft_tokens = opt.draft_tokens;
          if (opt.draft_policy == "fixed") {
            s_opts.enable_adaptive_draft_length = false;
          } else if (opt.draft_policy == "accepted-ema") {
            s_opts.adaptive_draft_policy =
                speculative::AdaptiveDraftPolicy::kAcceptedTokenEma;
          }
          if (opt.speculative_backend == "dflash" ||
              opt.speculative_backend == "dflash2" ||
              opt.speculative_backend == "dflash-2" ||
              opt.speculative_backend == "dflash-npu" ||
              opt.speculative_backend == "npu") {
            s_opts.use_batched_verification = true;
            s_opts.use_batched_lm_head = true;
            s_opts.target_bf16_from_layer = 48;
          } else if (opt.speculative_backend == "mtp" ||
                     opt.speculative_backend == "mtp-npu") {
            s_opts.use_batched_verification = true;
            s_opts.use_batched_lm_head = true;
            s_opts.target_bf16_from_layer = 0;
            s_opts.target_fp32_from_layer = 63;
          }
          spec_verifier = std::make_unique<speculative::SpeculativeVerifier>(
              *gpu_exec, std::move(draft_backend), s_opts);
        }

        std::vector<tokenization::TokenId> speculative_sequence;
        tokenization::TokenId speculative_current_token = 0;
        std::uint32_t speculative_current_pos = 0;
        if (spec_verifier) {
          speculative_sequence =
              MakeBenchmarkTokens(depth > 0 ? depth : std::size_t{16});
          speculative_current_token =
              spec_verifier->Prime(speculative_sequence);
          speculative_current_pos =
              static_cast<std::uint32_t>(speculative_sequence.size());
          speculative_sequence.push_back(speculative_current_token);
        }
        tokenization::TokenId greedy_current_token = 0;
        std::uint32_t greedy_current_pos = 0;
        if (!spec_verifier) {
          if (depth > 0) {
            if (!has_prepared_next_token) {
              throw std::logic_error(
                  "prepared generation depth has no target token");
            }
            greedy_current_token = prepared_next_token;
            greedy_current_pos = static_cast<std::uint32_t>(depth);
          } else {
            const auto prompt = MakeBenchmarkTokens(16);
            greedy_current_token = gpu_exec->ForwardPromptBatch(prompt);
            greedy_current_pos = static_cast<std::uint32_t>(prompt.size());
          }
        }
        HIP_CHECK(hipDeviceSynchronize());

        const auto t0 = std::chrono::high_resolution_clock::now();
        if (spec_verifier) {
          std::size_t emitted = 0;
          while (emitted < g_len) {
            const auto step_res = spec_verifier->VerifyStep(
                speculative_sequence, speculative_current_pos,
                speculative_current_token, 999999);
            for (const auto t : step_res.emitted_tokens) {
              speculative_sequence.push_back(t);
              ++speculative_current_pos;
              ++emitted;
              if (emitted >= g_len) {
                break;
              }
            }
            speculative_current_token = step_res.next_token;
          }
          acceptance_runs.push_back(spec_verifier->GetStats().AcceptanceRate());
          if (mtp_backend != nullptr && opt.speculative_backend == "mtp-npu") {
            const auto& metrics = mtp_backend->GetHybridMetrics();
            if (metrics.projection_count > 0) {
              const double count =
                  static_cast<double>(metrics.projection_count);
              hybrid_gpu_to_host_runs.push_back(metrics.gpu_to_host_us / count);
              hybrid_activation_pack_runs.push_back(metrics.activation_pack_us /
                                                    count);
              hybrid_command_runs.push_back(metrics.npu_command_us / count);
              hybrid_end_to_end_runs.push_back(metrics.npu_end_to_end_us /
                                               count);
              hybrid_host_to_gpu_runs.push_back(metrics.host_to_gpu_us / count);
            }
          }
          if (npu_draft_backend != nullptr) {
            npu_summary_metrics = npu_draft_backend->GetMetrics();
            has_npu_metrics = true;
          }
        } else {
          for (std::size_t step = 0; step < g_len; ++step) {
            greedy_current_token = gpu_exec->ForwardToken(greedy_current_token,
                                                          greedy_current_pos);
            ++greedy_current_pos;
          }
        }
        HIP_CHECK(hipDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double elapsed_sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (elapsed_sec > 0.0) {
          runs.push_back(static_cast<double>(g_len) / elapsed_sec);
        }
      }

      std::string test_name = MakeTestName("tg", g_len, depth);
      if (!opt.speculative_backend.empty()) {
        test_name += "-" + opt.speculative_backend;
      }
      print_result(test_name, ComputeStats(runs));
      if (opt.verbose && !acceptance_runs.empty()) {
        const auto acceptance = ComputeStats(acceptance_runs);
        std::cerr << test_name << " acceptance=" << std::fixed
                  << std::setprecision(3) << acceptance.mean << " +/- "
                  << acceptance.stddev << '\n';
      }
      if (opt.verbose && !hybrid_command_runs.empty()) {
        const auto gpu_to_host = ComputeStats(hybrid_gpu_to_host_runs);
        const auto activation_pack = ComputeStats(hybrid_activation_pack_runs);
        const auto command = ComputeStats(hybrid_command_runs);
        const auto npu_end_to_end = ComputeStats(hybrid_end_to_end_runs);
        const auto host_to_gpu = ComputeStats(hybrid_host_to_gpu_runs);
        std::cerr << test_name << " hybrid_us_per_projection: gpu_to_host="
                  << gpu_to_host.mean
                  << " activation_pack=" << activation_pack.mean
                  << " npu_command=" << command.mean
                  << " npu_end_to_end=" << npu_end_to_end.mean
                  << " host_to_gpu=" << host_to_gpu.mean << '\n';
      }
      if (opt.verbose && has_npu_metrics) {
        std::cerr << test_name << " [NPU Profiler Telemetry]: invocations="
                  << npu_summary_metrics.proposal_invocations
                  << " npu_submissions=" << npu_summary_metrics.npu_submissions
                  << " dma_bytes=" << npu_summary_metrics.dma_bytes_transferred
                  << " dma_time_us=" << npu_summary_metrics.total_dma_time_us
                  << " npu_time_us=" << npu_summary_metrics.total_npu_time_us
                  << '\n';
      }
    }
  }

  std::cout << "\n";
  return 0;
#else
  std::cerr << "Error: Gufo GPU benchmark requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(model_load_start, false);
  return 1;
#endif
}

}  // namespace gufo::cli
