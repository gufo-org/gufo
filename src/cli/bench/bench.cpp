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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/testing/compare/logit_comparator.hpp"

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
  parser.AddOption("-c", "--concurrency", "n,n,...",
                   "DS4 simultaneous requests, 1..8 (default: 1); pp is "
                   "aggregate, tg per user",
                   "Workload", &opt.model_path);
  parser.AddOption(
      "-r", "--repetitions", "N",
      "Repetitions per test point for variance reduction (default: 1)",
      "Workload", &opt.repetitions);

  parser.AddOption("", "--validate-prefill", "N",
                   "Compare batched prefill logits against sequential "
                   "reference",
                   "Validation", &opt.model_path);

  parser.AddOption(
      "", "--speculative", "MODE",
      "Draft backend: dflash, dflash2, mtp, mtp-npu, dspark, npu, pld, "
      "self, or off",
      "Speculative", &opt.speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--dspark-model", "PATH",
                   "DeepSeek V4 Flash DSpark support GGUF", "Speculative",
                   &opt.dspark_model_path);
  parser.AddOption("", "--mtp-model", "PATH",
                   "Path to quantized Qwen MTP draft head GGUF file",
                   "Speculative", &opt.mtp_model_path);
  parser.AddOption(
      "", "--draft-tokens", "N",
      "Maximum speculative draft tokens per verification step (default: 7)",
      "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--spec-draft-n-max", "N",
                   "llama.cpp-compatible alias for --draft-tokens",
                   "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--draft-vocab", "N",
                   "Qwen3.8-Flash-Next: score MTP drafts over the first N "
                   "token ids only (default: 0 = full vocabulary)",
                   "Speculative", &opt.draft_vocab);

  parser.AddOption("", "--min-draft-tokens", "N",
                   "Adaptive draft floor (default: 1)", "Speculative",
                   &opt.min_draft_tokens);
  parser.AddOption("", "--spec-draft-n-min", "N",
                   "llama.cpp-compatible alias for --min-draft-tokens",
                   "Speculative", &opt.min_draft_tokens);
  parser.AddOption("", "--temperature", "T",
                   "DeepSeek generation temperature; 0 is greedy (default: 0)",
                   "Workload", &opt.temperature);
  parser.AddOption("", "--seed", "N",
                   "DeepSeek sampling seed for --temperature (default: 0)",
                   "Workload", &opt.seed);

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

std::optional<std::vector<std::size_t>> ParseCommaSeparatedSizes(
    std::string_view text, bool keep_zero = false) {
  if (text.empty() || text.back() == ',')
    return std::nullopt;
  std::vector<std::size_t> result;
  for (std::size_t start = 0; start < text.size();) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto part = text.substr(start, end - start);
    std::size_t value = 0;
    const auto [ptr, ec] =
        std::from_chars(part.data(), part.data() + part.size(), value);
    if (ec != std::errc{} || ptr != part.data() + part.size())
      return std::nullopt;
    if (value != 0 || keep_zero)
      result.push_back(value);
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
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  constexpr std::size_t context_limit = std::numeric_limits<int>::max() - 1;
  if (max_depth > context_limit || max_prompt > context_limit - max_depth ||
      max_generation > context_limit - std::max<std::size_t>(max_depth, 16) ||
      options.validate_prefill_tokens > context_limit) {
    std::cerr << "Error: requested DeepSeek benchmark context is too large\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool dspark = options.speculative_backend == "dspark";
  if (!options.speculative_backend.empty() && !dspark) {
    std::cerr << "Error: DeepSeek supports only --speculative dspark or off\n";
    return 1;
  }
  if (dspark && options.dspark_model_path.empty()) {
    std::cerr << "Error: --speculative dspark requires --dspark-model\n";
    return 1;
  }
  if (dspark && options.min_draft_tokens != 1) {
    std::cerr << "Error: DSpark uses model-owned adaptive drafting; custom "
                 "draft floors are unsupported\n";
    return 1;
  }

  // Sampled generation draws with the shared server sampler on both paths so
  // the AR and DSpark runs of one seed produce comparable token hashes.
  const bool sampled = options.temperature > 0.0F;
  const sampling::SamplingConfig sampling_config{
      .temperature = options.temperature,
      .seed = options.seed,
  };
  const auto sample_next = [&](models::deepseek_v4_flash::Session& session,
                               sampling::SamplerState& sampler,
                               std::string* error_msg) -> int {
    int token = session.TakePendingDsparkToken();
    if (token < 0) {
      const auto logits = session.CopyLogits(error_msg);
      if (logits.empty())
        return -1;
      token = static_cast<int>(sampler.Sample(logits));
    }
    sampler.Accept(static_cast<sampling::TokenId>(token));
    return token;
  };

  std::string error;
  auto model = models::deepseek_v4_flash::Model::Load(
      options.model_path,
      models::deepseek_v4_flash::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .dspark_model_path = dspark ? options.dspark_model_path : "",
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating DeepSeek V4 Flash model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto tokens = MakeDeepSeekBenchmarkTokens(*model, required_context);
  if (options.validate_prefill_tokens != 0) {
    auto sequential = model->CreateSession(required_context, &error);
    auto batched = model->CreateSession(required_context, &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "DeepSeek prefill validation failed: " << error << '\n';
      return 1;
    }
    for (const int token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "DeepSeek scalar reference failed: " << error << '\n';
        return 1;
      }
    }
    const auto reference = sequential->CopyLogits(&error);
    const auto candidate = batched->CopyLogits(&error);
    if (reference.empty() || reference.size() != candidate.size()) {
      std::cerr << "DeepSeek validation logit readback failed: " << error
                << '\n';
      return 1;
    }
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
    }
  }
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

  for (const std::size_t concurrency : options.concurrency) {
    if (concurrency > 1) {
      using models::deepseek_v4_flash::Session;
      for (const std::size_t depth : options.n_depths) {
        auto base = model->CreateSession(
            static_cast<uint32_t>(required_context), &error);
        const std::size_t prefix = depth > 0 ? depth : 16;
        if (!base || !base->Sync(std::span(tokens).first(prefix), &error)) {
          std::cerr << "Error preparing concurrent depth: " << error << '\n';
          return 1;
        }
        auto snapshot = base->SaveSnapshot(&error);
        if (!snapshot) {
          std::cerr << "Error preparing concurrent snapshot: " << error << '\n';
          return 1;
        }
        base.reset();
        for (const bool generate : {false, true}) {
          for (const std::size_t count :
               generate ? options.n_gens : options.n_prompts) {
            std::vector<double> runs;
            for (std::size_t repeat = 0; repeat < options.repetitions;
                 ++repeat) {
              std::vector<std::unique_ptr<Session>> sessions;
              std::vector<std::vector<int>> generated(concurrency);
              std::vector<sampling::SamplerState> samplers(
                  concurrency, sampling::SamplerState(sampling_config));
              for (std::size_t i = 0; i < concurrency; ++i) {
                auto session = model->CreateSession(
                    static_cast<uint32_t>(required_context), &error);
                if (!session ||
                    ((generate || depth > 0) &&
                     !session->RestoreSnapshot(*snapshot, &error))) {
                  std::cerr << "Error restoring concurrent depth: " << error
                            << '\n';
                  return 1;
                }
                sessions.push_back(std::move(session));
                generated[i].reserve(count);
              }
              const auto start = std::chrono::steady_clock::now();
              if (!generate) {
                for (auto& session : sessions) {
                  if (!session->Sync(std::span(tokens).first(depth + count),
                                     &error)) {
                    std::cerr << "Error in concurrent prefill: " << error
                              << '\n';
                    return 1;
                  }
                }
              } else if (dspark) {
                std::vector<std::vector<int>> emitted(concurrency);
                while (true) {
                  std::vector<models::deepseek_v4_flash::SessionDsparkBatchItem>
                      items;
                  std::vector<std::size_t> active;
                  std::vector<std::unique_ptr<
                      models::deepseek_v4_flash::DsparkSamplerBridge>>
                      bridges;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    if (generated[i].size() == count)
                      continue;
                    active.push_back(i);
                    const ds4_dspark_sampler* hook = nullptr;
                    if (sampled) {
                      bridges.push_back(
                          std::make_unique<
                              models::deepseek_v4_flash::DsparkSamplerBridge>(
                              samplers[i]));
                      hook = bridges.back()->hook();
                    }
                    items.push_back({.session = sessions[i].get(),
                                     .max_tokens = count - generated[i].size(),
                                     .max_draft_tokens = options.draft_tokens,
                                     .emitted = &emitted[i],
                                     .sampler = hook});
                  }
                  if (items.empty())
                    break;
                  if (!model->DsparkStepBatch(items, &error)) {
                    std::cerr << "Error in concurrent DSpark: " << error
                              << '\n';
                    return 1;
                  }
                  for (std::size_t slot = 0; slot < active.size(); ++slot) {
                    const auto i = active[slot];
                    if (emitted[i].empty() ||
                        emitted[i].size() > count - generated[i].size()) {
                      std::cerr << "Error: DSpark returned an invalid "
                                   "benchmark token count\n";
                      return 1;
                    }
                    if (sampled) {
                      samplers[i].SetRngState(bridges[slot]->rng_state());
                      for (const int token : emitted[i]) {
                        samplers[i].Accept(
                            static_cast<sampling::TokenId>(token));
                      }
                    }
                    generated[i].insert(generated[i].end(), emitted[i].begin(),
                                        emitted[i].end());
                  }
                }
              } else {
                for (std::size_t step = 0; step < count; ++step) {
                  std::vector<models::deepseek_v4_flash::SessionBatchItem>
                      items;
                  for (std::size_t i = 0; i < concurrency; ++i) {
                    const int token =
                        sampled ? sample_next(*sessions[i], samplers[i], &error)
                                : sessions[i]->SelectNext(0.0F, nullptr);
                    if (token < 0) {
                      std::cerr << "Error in concurrent sampling: " << error
                                << '\n';
                      return 1;
                    }
                    generated[i].push_back(token);
                    items.push_back(
                        {.session = sessions[i].get(), .token = token});
                  }
                  if (!model->EvaluateBatch(items, &error)) {
                    std::cerr << "Error in concurrent target decode: " << error
                              << '\n';
                    return 1;
                  }
                }
              }
              const double seconds =
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
              runs.push_back(
                  static_cast<double>(count) *
                  (generate ? 1.0 : static_cast<double>(concurrency)) /
                  seconds);
              if (options.verbose && generate) {
                for (std::size_t i = 0; i < concurrency; ++i) {
                  const auto stats = sessions[i]->DsparkStatistics();
                  const auto* bytes =
                      reinterpret_cast<const uint8_t*>(generated[i].data());
                  std::cerr << "DeepSeek tg C=" << concurrency
                            << " depth=" << depth << " request=" << i
                            << " accepted=" << stats.support_accepted
                            << " drafted=" << stats.support_drafted
                            << " steps=" << stats.steps
                            << " skipped=" << stats.skipped << " output_sha256="
                            << crypto::Sha256Hex(std::span(
                                   bytes, generated[i].size() * sizeof(int)))
                            << '\n';
                }
              }
            }
            print_result(MakeTestName(generate ? "tg" : "pp", count, depth) +
                             " C" + std::to_string(concurrency),
                         ComputeStats(runs));
          }
        }
      }
      continue;
    }
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
          std::cerr << "Error preparing DeepSeek depth " << depth << ": "
                    << error << '\n';
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
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr && depth > 0 &&
                !session->Sync(std::span(tokens).first(depth), &error)) {
              session = nullptr;
            }
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
          if (snapshot == nullptr) {
            local_session = model->CreateSession(
                static_cast<std::uint32_t>(required_context), &error);
            session = local_session.get();
            if (session != nullptr &&
                !session->Sync(std::span(tokens.data(), prefix_length),
                               &error)) {
              session = nullptr;
            }
          } else if (!prepared_session->RestoreSnapshot(*snapshot, &error)) {
            session = nullptr;
          }
          if (session == nullptr) {
            std::cerr << "Error preparing DeepSeek generation: " << error
                      << '\n';
            return 1;
          }

          std::vector<int> generated;
          generated.reserve(generation_length);
          sampling::SamplerState sampler(sampling_config);
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t step = 0; step < generation_length;) {
            if (dspark) {
              std::vector<int> emitted;
              std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>
                  bridge;
              if (sampled)
                bridge.emplace(sampler);
              if (!session->DsparkStep(generation_length - step,
                                       options.draft_tokens, &emitted, &error,
                                       bridge ? bridge->hook() : nullptr) ||
                  emitted.empty()) {
                std::cerr << "Error running DSpark decode: " << error << '\n';
                return 1;
              }
              if (bridge) {
                sampler.SetRngState(bridge->rng_state());
                for (const int token : emitted) {
                  sampler.Accept(static_cast<sampling::TokenId>(token));
                }
              }
              generated.insert(generated.end(), emitted.begin(), emitted.end());
              step += emitted.size();
            } else {
              const int token = sampled ? sample_next(*session, sampler, &error)
                                        : session->SelectNext(0.0F, nullptr);
              if (token < 0 || !session->Evaluate(token, &error)) {
                std::cerr << "Error running DeepSeek decode: " << error << '\n';
                return 1;
              }
              generated.push_back(token);
              ++step;
            }
          }
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
          runs.push_back(static_cast<double>(generation_length) / seconds);
          if (options.verbose) {
            const auto stats = session->DsparkStatistics();
            const auto* bytes =
                reinterpret_cast<const uint8_t*>(generated.data());
            std::cerr << "DeepSeek tg C=1 depth=" << depth
                      << " request=0 accepted=" << stats.support_accepted
                      << " drafted=" << stats.support_drafted
                      << " steps=" << stats.steps
                      << " skipped=" << stats.skipped << " output_sha256="
                      << crypto::Sha256Hex(
                             std::span(bytes, generated.size() * sizeof(int)))
                      << '\n';
          }
        }
        print_result(MakeTestName("tg", generation_length, depth),
                     ComputeStats(runs));
      }

      if (snapshot && !prepared_session->RestoreSnapshot(*snapshot, &error)) {
        std::cerr << "Error restoring final DeepSeek depth: " << error << '\n';
        return 1;
      }
    }
  }
  std::cout << '\n';
  return 0;
}

bool IsQwen38FlashNext(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "qwen4exp";
}

int RunQwen38FlashNextBenchmark(
    const BenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  namespace qfn = models::qwen38_flash_next;
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Qwen3.8-Flash-Next\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Qwen3.8-Flash-Next supports only --speculative mtp "
                 "or off\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\n";
    return 1;
  }
  if (options.temperature > 0.0F) {
    std::cerr << "Error: Qwen3.8-Flash-Next benchmark decodes greedily\n";
    return 1;
  }

  std::string error;
  auto model = qfn::Model::Load(
      options.model_path,
      qfn::ModelOptions{
          .max_context = static_cast<std::uint32_t>(required_context),
          .mtp_model_path = mtp ? options.mtp_model_path : "",
          .max_batch = static_cast<std::uint32_t>(
              std::max<std::size_t>(1, options.batch_size)),
          .max_draft_tokens = std::max<std::uint32_t>(1, options.draft_tokens),
          .draft_vocab = options.draft_vocab,
      },
      &error);
  if (model == nullptr) {
    std::cerr << "Error creating Qwen3.8-Flash-Next model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  // A natural-language pattern keeps the router and the n-gram hashes on
  // realistic paths.
  std::vector<std::int32_t> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = pattern[i % pattern.size()];
    }
  }

  if (options.validate_prefill_tokens != 0) {
    // Batched prefill against one-token-at-a-time evaluation of the same
    // prefix; both run on the GPU, so this checks the batched kernels.
    auto sequential = model->CreateSession(
        static_cast<std::uint32_t>(required_context), &error);
    auto batched = model->CreateSession(
        static_cast<std::uint32_t>(required_context), &error);
    const auto prefix =
        std::span(tokens).first(options.validate_prefill_tokens);
    if (!sequential || !batched || !sequential->Sync(prefix.first(1), &error) ||
        !batched->Sync(prefix, &error)) {
      std::cerr << "Qwen3.8-Flash-Next prefill validation failed: " << error
                << '\n';
      return 1;
    }
    for (const std::int32_t token : prefix.subspan(1)) {
      if (!sequential->Evaluate(token, &error)) {
        std::cerr << "Qwen3.8-Flash-Next sequential reference failed: " << error
                  << '\n';
        return 1;
      }
    }
    const auto reference = sequential->Logits();
    const auto candidate = batched->Logits();
    const auto comparison = testing::CompareLogits(reference, candidate);
    const std::size_t rank =
        1 + std::count_if(candidate.begin(), candidate.end(), [&](float value) {
          return value > candidate[comparison.reference_argmax];
        });
    std::cout << "[Prefill Validation] tokens=" << prefix.size()
              << " finite=" << comparison.finite
              << " scalar_winner_rank=" << rank
              << " rmse=" << comparison.root_mean_square_error
              << " cosine=" << comparison.cosine_similarity
              << " max_error=" << comparison.max_abs_diff << '\n';
    if (!comparison.finite || rank > 3 ||
        comparison.root_mean_square_error > 1.12F ||
        comparison.cosine_similarity < 0.979F ||
        comparison.max_abs_diff > 5.0F) {
      return 1;
    }
  }

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

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      // One untimed pass first, as llama-bench does: GEMM plan tuning and
      // arena growth happen once per shape.
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            (depth > 0 &&
             !session->Sync(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            !session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        std::vector<std::int32_t> generated;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < generation_length;) {
          if (mtp) {
            std::vector<std::int32_t> emitted;
            if (!session->SpeculativeStep(generation_length - step, &emitted,
                                          &error) ||
                emitted.empty()) {
              std::cerr << "Error running MTP decode: " << error << '\n';
              return 1;
            }
            generated.insert(generated.end(), emitted.begin(), emitted.end());
            step += emitted.size();
          } else {
            const std::int32_t token = session->SelectNext(0.0F, nullptr);
            if (!session->Evaluate(token, &error)) {
              std::cerr << "Error running decode: " << error << '\n';
              return 1;
            }
            generated.push_back(token);
            ++step;
          }
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          const auto stats = session->Statistics();
          std::cerr << "Qwen3.8-Flash-Next tg depth=" << depth
                    << " cycles=" << stats.cycles
                    << " drafted=" << stats.drafted
                    << " accepted=" << stats.accepted << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
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
      "-p", "--n-prompt", "n,n,...",
      "Prompt token lengths to benchmark (default: 64,128,512)", "Workload",
      [&opt, &explicit_p](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_prompts = std::move(*sizes);
        explicit_p = true;
        return true;
      });

  parser.AddCustomOption(
      "-n", "--n-gen", "n,n,...",
      "Number of text generation tokens to benchmark (default: 128)",
      "Workload",
      [&opt, &explicit_n](std::string_view flag, std::string_view val,
                          std::string* error) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val);
        if (!sizes) {
          if (error)
            *error = "Invalid token counts for " + std::string(flag);
          return false;
        }
        opt.n_gens = std::move(*sizes);
        explicit_n = true;
        return true;
      });

  parser.AddCustomOption(
      "-d", "--n-depth", "n,n,...",
      "Context depths prepared before timed region (default: 0)", "Workload",
      [&opt](std::string_view flag_name, std::string_view val,
             std::string* err) -> bool {
        auto sizes = ParseCommaSeparatedSizes(val, true);
        if (!sizes) {
          if (err != nullptr) {
            *err = "Invalid argument for " + std::string(flag_name);
          }
          return false;
        }
        opt.n_depths = std::move(*sizes);
        return true;
      });

  parser.AddCustomOption(
      "-c", "--concurrency", "n,n,...",
      "DS4 simultaneous requests, 1..8 (default: 1); pp is aggregate, tg per "
      "user",
      "Workload",
      [&opt](std::string_view, std::string_view value, std::string* error) {
        auto sizes = ParseCommaSeparatedSizes(value, true);
        if (!sizes || sizes->empty() ||
            std::any_of(sizes->begin(), sizes->end(),
                        [](auto size) { return size < 1 || size > 8; })) {
          if (error)
            *error =
                "--concurrency requires DS4 request counts between 1 and 8";
          return false;
        }
        opt.concurrency = std::move(*sizes);
        return true;
      });

  parser.AddOption("-b", "--batch-size", "N",
                   "Qwen3.8-Flash-Next: prefill chunk size (default: 512)",
                   "Workload", &opt.batch_size);
  parser.AddOption(
      "-r", "--repetitions", "N",
      "Repetitions per test point for variance reduction (default: 1)",
      "Workload", &opt.repetitions);

  parser.AddCustomOption(
      "", "--validate-prefill", "N",
      "Compare batched prefill logits against sequential reference",
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

  bool speculative_explicit = false;
  const auto parse_speculative_backend =
      [&opt, &speculative_explicit](std::string_view, std::string_view value,
                                    std::string*) -> bool {
    speculative_explicit = true;
    if (value == "none" || value == "off" || value == "false" ||
        value == "disabled") {
      opt.speculative_backend.clear();
    } else {
      opt.speculative_backend = value;
    }
    return true;
  };
  parser.AddCustomOption("", "--speculative", "MODE",
                         "Draft backend: dflash, dflash2, mtp, mtp-npu, "
                         "dspark, npu, pld, self, or off",
                         "Speculative", parse_speculative_backend);
  parser.AddCustomOption("", "--speculative-decoding", "MODE",
                         "Alias for --speculative", "Speculative",
                         parse_speculative_backend);
  parser.AddOption("", "--dflash-model", "PATH",
                   "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                   "Speculative", &opt.dflash_model_path);
  parser.AddOption("", "--dspark-model", "PATH",
                   "DeepSeek V4 Flash DSpark support GGUF", "Speculative",
                   &opt.dspark_model_path);
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
  parser.AddOption("", "--spec-draft-n-max", "N",
                   "llama.cpp-compatible alias for --draft-tokens",
                   "Speculative", &opt.draft_tokens);
  parser.AddOption("", "--spec-draft-n-min", "N",
                   "llama.cpp-compatible alias for --min-draft-tokens",
                   "Speculative", &opt.min_draft_tokens);
  parser.AddOption("", "--draft-vocab", "N",
                   "Qwen3.8-Flash-Next: score MTP drafts over the first N "
                   "token ids only (default: 0 = full vocabulary)",
                   "Speculative", &opt.draft_vocab);
  parser.AddOption("", "--temperature", "T",
                   "DeepSeek generation temperature; 0 is greedy (default: 0)",
                   "Workload", &opt.temperature);
  parser.AddOption("", "--seed", "N",
                   "DeepSeek sampling seed for --temperature (default: 0)",
                   "Workload", &opt.seed);
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

  if (opt.repetitions == 0 || (opt.n_prompts.empty() && opt.n_gens.empty() &&
                               opt.validate_prefill_tokens == 0)) {
    if (error_msg)
      *error_msg =
          "Benchmark needs positive repetitions and at least one workload";
    return std::nullopt;
  }

  if (!speculative_explicit && !opt.dspark_model_path.empty()) {
    opt.speculative_backend = "dspark";
  }

  if (opt.draft_tokens == 0 || opt.min_draft_tokens == 0 ||
      opt.min_draft_tokens > opt.draft_tokens) {
    if (error_msg != nullptr) {
      *error_msg = "min-draft-tokens cannot exceed draft-tokens";
    }
    return std::nullopt;
  }
  if (!std::isfinite(opt.temperature) || opt.temperature < 0.0F) {
    if (error_msg != nullptr) {
      *error_msg = "temperature must be non-negative";
    }
    return std::nullopt;
  }
  if (opt.min_draft_tokens != 1 && (opt.speculative_backend == "dflash" ||
                                    opt.speculative_backend == "dflash2" ||
                                    opt.speculative_backend == "dflash-2")) {
    if (error_msg != nullptr)
      *error_msg = "DFlash2 uses fixed blocks; --min-draft-tokens must be 1";
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

#if defined(ENGINE_ENABLE_HIP)
  if (IsDeepSeekV4Flash(*reader)) {
    return RunDeepSeekBenchmark(opt, reader, model_load_start);
  }
  if (IsQwen38FlashNext(*reader)) {
    return RunQwen38FlashNextBenchmark(opt, reader, model_load_start);
  }

  if (opt.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: concurrent model benchmarks currently support DS4 "
                 "only; use the serving benchmark for Qwen\n";
    return 1;
  }
  if (opt.temperature > 0.0F) {
    std::cerr << "Error: sampled model benchmarks currently support DS4 only; "
                 "use the serving benchmark for Qwen\n";
    return 1;
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
      std::cerr << "MTP requires --mtp-model\n";
      return 1;
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
  const std::string model_name =
      config.model_name + " " + reader->GetQuantizationLabel();
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
        if (opt.speculative_backend == "dflash" ||
            opt.speculative_backend == "dflash2" ||
            opt.speculative_backend == "dflash-2") {
          std::string dflash_path = opt.dflash_model_path;
          if (dflash_path.empty()) {
            std::cerr << "DFlash2 requires --dflash-model\n";
            return 1;
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
          s_opts.min_draft_tokens = opt.min_draft_tokens;
          s_opts.initial_draft_tokens = opt.draft_tokens;
          const bool block_diffusion_draft =
              opt.speculative_backend == "dflash" ||
              opt.speculative_backend == "dflash2" ||
              opt.speculative_backend == "dflash-2";
          s_opts.enable_adaptive_draft_length = !block_diffusion_draft;
          if (opt.speculative_backend == "dflash" ||
              opt.speculative_backend == "dflash2" ||
              opt.speculative_backend == "dflash-2") {
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
                speculative_current_token, 999999,
                static_cast<std::uint32_t>(g_len - emitted));
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
  std::cerr << "Error: Gufo GPU benchmark requires ENGINE_ENABLE_HIP=ON\n";
  PrintModelLoadTime(model_load_start, false);
  return 1;
#endif
}

}  // namespace gufo::cli
