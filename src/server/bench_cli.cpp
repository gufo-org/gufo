#include "src/server/bench_cli.hpp"

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

#include "src/core/gguf_reader.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/heterogeneous/npu_drafter.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_executor.hpp"
#include "src/models/qwen/hip/qwen_mtp_gpu.hpp"
#include "src/core/speculative/draft_heads.hpp"
#include "src/core/speculative/prompt_lookup_backend.hpp"
#include "src/core/speculative/self_speculative.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#endif

namespace strix::server {
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

void PrintBenchHelp(std::string_view program_name) {
  std::cout << "Usage: " << program_name << " bench [options]\n\n"
            << "Benchmark prompt processing (pp) and token generation (tg) "
               "throughput.\n\n"
            << "Options:\n"
            << "  -h, --help                  Print help\n"
            << "  -m, --model <PATH>          Path to GGUF model file "
               "(default: models/Qwen3.5-4B-BF16.gguf)\n"
            << "  -p, --n-prompt <n,n,...>    Prompt token lengths to "
               "benchmark (default: 64,128,512)\n"
            << "  -n, --n-gen <n,n,...>       Number of text generation tokens "
               "(default: 128)\n"
            << "  -d, --n-depth <n,n,...>     Context depths prepared outside "
               "the timed region (default: 0)\n"
            << "  -r, --repetitions <N>       Number of repetitions per test "
               "(default: 1)\n"
            << "  --validate-prefill <N>      Compare batched logits against "
               "sequential prefill\n"
            << "  --speculative <MODE>        Draft backend: mtp, mtp-npu, "
               "npu, pld, or self\n"
            << "  --mtp-model <PATH>          Quantized Qwen MTP GGUF\n"
            << "  --draft-tokens <N>          Maximum speculative block "
               "length\n"
            << "  -ngl, --n-gpu-layers <N>    Number of layers offloaded to "
               "GPU (default: 99)\n"
            << "  -v, --verbose               Verbose progress output\n";
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
  bool skip_next = false;

  for (std::size_t i = 0; i < args.size(); ++i) {
    if (skip_next) {
      skip_next = false;
      continue;
    }

    const std::string_view arg = args[i];
    if (arg == "-h" || arg == "--help") {
      return std::nullopt;
    }

    if (arg == "-m" || arg == "--model") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.model_path = args[i + 1];
      skip_next = true;
      continue;
    }

    if (arg == "-p" || arg == "--n-prompt") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.n_prompts = ParseCommaSeparatedSizes(args[i + 1]);
      explicit_p = true;
      skip_next = true;
      continue;
    }

    if (arg == "-n" || arg == "--n-gen") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.n_gens = ParseCommaSeparatedSizes(args[i + 1]);
      explicit_n = true;
      skip_next = true;
      continue;
    }

    if (arg == "-d" || arg == "--n-depth") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      opt.n_depths = ParseCommaSeparatedSizes(args[i + 1], true);
      if (opt.n_depths.empty()) {
        if (error_msg != nullptr) {
          *error_msg = "Invalid argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      skip_next = true;
      continue;
    }

    if (arg == "-r" || arg == "--repetitions") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      std::size_t r = 0;
      const std::string_view val = args[i + 1];
      std::from_chars(val.data(), val.data() + val.size(), r);
      if (r > 0) {
        opt.repetitions = r;
      }
      skip_next = true;
      continue;
    }

    if (arg == "--validate-prefill") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for --validate-prefill";
        }
        return std::nullopt;
      }
      const std::string_view val = args[i + 1];
      const auto [ptr, ec] = std::from_chars(
          val.data(), val.data() + val.size(), opt.validate_prefill_tokens);
      if (ec != std::errc{} || ptr != val.data() + val.size() ||
          opt.validate_prefill_tokens == 0) {
        if (error_msg != nullptr) {
          *error_msg = "Invalid argument for --validate-prefill";
        }
        return std::nullopt;
      }
      skip_next = true;
      continue;
    }

    if (arg == "-ngl" || arg == "--n-gpu-layers") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for " + std::string(arg);
        }
        return std::nullopt;
      }
      int ngl = 0;
      const std::string_view val = args[i + 1];
      std::from_chars(val.data(), val.data() + val.size(), ngl);
      opt.n_gpu_layers = ngl;
      skip_next = true;
      continue;
    }

    if (arg == "--speculative") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for --speculative";
        }
        return std::nullopt;
      }
      opt.speculative_backend = std::string(args[i + 1]);
      skip_next = true;
      continue;
    }

    if (arg == "--mtp-model") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for --mtp-model";
        }
        return std::nullopt;
      }
      opt.mtp_model_path = std::string(args[i + 1]);
      skip_next = true;
      continue;
    }

    if (arg == "--draft-tokens") {
      if (i + 1 >= args.size()) {
        if (error_msg != nullptr) {
          *error_msg = "Missing argument for --draft-tokens";
        }
        return std::nullopt;
      }
      std::uint32_t k = 0;
      const std::string_view val = args[i + 1];
      std::from_chars(val.data(), val.data() + val.size(), k);
      if (k > 0) {
        opt.draft_tokens = k;
      }
      skip_next = true;
      continue;
    }

    if (arg == "-v" || arg == "--verbose") {
      opt.verbose = true;
      continue;
    }

    if (!arg.empty() && arg[0] != '-') {
      opt.model_path = std::string(arg);
      continue;
    }
  }

  if (explicit_p && !explicit_n) {
    opt.n_gens.clear();
  } else if (!explicit_p && explicit_n) {
    opt.n_prompts.clear();
  }

  return opt;
}

int RunBench(std::span<const char* const> args) {
  std::string parse_err;
  const auto opt_res = ParseBenchOptions(args, &parse_err);
  if (!opt_res.has_value()) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintBenchHelp("strix-server");
      return 2;
    }
    PrintBenchHelp("strix-server");
    return 0;
  }

  const auto& opt = *opt_res;

  const auto model_load_start = std::chrono::steady_clock::now();
  std::string err;
  auto reader_owner = strix::core::GgufReader::OpenFile(opt.model_path, &err);
  if (!reader_owner) {
    std::cerr << "Error loading GGUF model '" << opt.model_path << "': " << err
              << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const std::shared_ptr<const strix::core::GgufReader> reader(
      std::move(reader_owner));

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
      if (const char* environment = std::getenv("STRIX_MTP_MODEL");
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

  const std::string_view backend_name =
      opt.speculative_backend == "mtp-npu" ? "HIP+XDNA2" : "ROCm (HIP)";
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
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        std::unique_ptr<speculative::IDraftBackend> draft_backend;
        hip::QwenMtpGpuDraftBackend* mtp_backend = nullptr;
        if (opt.speculative_backend == "mtp" ||
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
        } else if (opt.speculative_backend == "npu") {
          heterogeneous::NpuDrafterConfig cfg;
          cfg.max_draft_tokens = opt.draft_tokens;
          cfg.vocab_size = config.vocab_size;
          draft_backend = std::make_unique<heterogeneous::NpuDraftBackend>(cfg);
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
          s_opts.initial_draft_tokens = opt.draft_tokens;
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
    }
  }

  std::cout << "\n";
  return 0;
#else
  std::cerr << "Error: Strix GPU benchmark requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(model_load_start, false);
  return 1;
#endif
}

}  // namespace strix::server
