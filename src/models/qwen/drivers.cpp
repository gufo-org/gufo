#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "gufo/ext/text.hpp"
#include "src/families/qwen/config/qwen_config.hpp"
#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/mtp.hpp"
#include "src/models/qwen/hip/qwen_speculative_verifier.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace gufo::models::qwen {
namespace {

constexpr std::size_t kDefaultContext = 4096;

#define HIP_CHECK(expr)                                               \
  do {                                                                \
    hipError_t err_ = (expr);                                         \
    if (err_ != hipSuccess) {                                         \
      std::cerr << "HIP error: " << hipGetErrorString(err_) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;          \
      return 1;                                                       \
    }                                                                 \
  } while (0)

struct BenchStats {
  double mean = 0.0;
  double stddev = 0.0;
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty())
    return {};
  if (values.size() == 1)
    return {.mean = values.front(), .stddev = 0.0};
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  double sq_sum = 0.0;
  for (const double val : values) {
    sq_sum += (val - mean) * (val - mean);
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

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  if (start == std::chrono::steady_clock::time_point{})
    return;
  const auto end = std::chrono::steady_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cerr << "Model loaded in " << ms << " ms ("
            << (success ? "success" : "failed") << ").\n";
}

void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

static std::optional<gufo::ReasoningEffort> ParseReasoningEffort(
    std::string_view value) {
  if (value == "minimal")
    return gufo::ReasoningEffort::kMinimal;
  if (value == "low")
    return gufo::ReasoningEffort::kLow;
  if (value == "medium")
    return gufo::ReasoningEffort::kMedium;
  if (value == "high")
    return gufo::ReasoningEffort::kHigh;
  if (value == "xhigh")
    return gufo::ReasoningEffort::kXHigh;
  if (value == "max")
    return gufo::ReasoningEffort::kMax;
  return std::nullopt;
}

static gufo::ReasoningOptions PromptReasoningOptions(
    const ext::TextPromptOptions& options) {
  gufo::ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

void AttachImages(const ext::TextPromptOptions& opt,
                  tokenization::ChatMessage& user_msg) {
  for (const auto& path : opt.image_paths) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) {
      throw std::runtime_error("Failed to read image: " + path);
    }
    const auto size = is.tellg();
    is.seekg(0, std::ios::beg);
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(size);
    if (!is.read(reinterpret_cast<char*>(bytes->data()), size)) {
      throw std::runtime_error("Failed to read image: " + path);
    }
    user_msg.images.push_back(
        {0, std::move(bytes), tokenization::ChatMessage::ImageDetail::kAuto});
  }
}

std::shared_ptr<models::qwen::vision::Encoder> LoadQwenVision(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader) {
  if (opt.image_paths.empty() && opt.vision_model_path.empty())
    return {};
  if (reader.GetMetadataUint64("qwen35.embedding_length") != 5120) {
    throw std::invalid_argument(
        "image input supports Qwen3.8-27B and Flash-Next");
  }
  auto encoder = models::qwen::vision::Encoder::Open(
      opt.model_path, opt.vision_model_path, 5120);
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  return encoder;
}

std::shared_ptr<const models::qwen::vision::Prompt> PrepareVision(
    const ext::TextPromptOptions& opt,
    const tokenization::QwenTokenizer& tokenizer,
    std::span<const tokenization::ChatMessage> messages,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder) {
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  const auto reasoning = PromptReasoningOptions(opt);
  return std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, messages, {},
          tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
          encoder->identity(), kDefaultContext));
}

std::unique_ptr<speculative::SpeculativeVerifier> CreateQwenVerifier(
    const ext::TextPromptOptions& opt, hip::QwenGpuExecutor& executor) {
  std::string err;
  std::unique_ptr<speculative::IDraftBackend> draft_backend;
  if (opt.speculative_backend == "dflash2") {
    std::string dflash_path = opt.dflash_model_path;
    if (dflash_path.empty()) {
      throw std::invalid_argument("DFlash2 requires --dflash-model");
    }
    hip::QwenDFlashGpuDraftConfig cfg{
        .max_context = executor.GetMaxContext(),
        .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
        .policy = speculative::ParseDFlashDraftPolicy(opt.draft_policy),
    };
    draft_backend = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
        dflash_path, executor.GetSharedModel(), cfg, &err);
    if (draft_backend == nullptr) {
      throw std::runtime_error("Failed to initialize DFlash backend: " + err);
    }
  } else if (opt.speculative_backend == "mtp") {
    std::string mtp_path = opt.mtp_model_path;
    if (mtp_path.empty()) {
      throw std::invalid_argument("MTP requires --mtp-model");
    }
    hip::QwenMtpGpuDraftConfig cfg{
        .max_context = executor.GetMaxContext(),
        .max_draft_tokens = static_cast<std::uint32_t>(opt.draft_tokens),
        .vision_input = &executor.VisionInput(),
    };
    draft_backend = hip::QwenMtpGpuDraftBackend::CreateFromGguf(
        mtp_path, executor.GetSharedModel(), cfg, &err);
    if (draft_backend == nullptr) {
      throw std::runtime_error("Failed to initialize MTP backend: " + err);
    }
  } else {
    throw std::invalid_argument("Unknown speculative backend: " +
                                opt.speculative_backend);
  }

  speculative::SpeculativeOptions s_opts;
  s_opts.max_draft_tokens = opt.draft_tokens;
  s_opts.min_draft_tokens = opt.min_draft_tokens;
  s_opts.initial_draft_tokens = opt.draft_tokens;
  const bool block_diffusion_draft = opt.speculative_backend == "dflash2";
  s_opts.enable_adaptive_draft_length = !block_diffusion_draft;
  s_opts.use_batched_verification = true;
  return hip::CreateQwenGpuSpeculativeVerifier(
      executor, std::move(draft_backend), s_opts);
}

void GenerateQwenGpuResponse(
    const ext::TextPromptOptions& opt, hip::QwenGpuExecutor& executor,
    speculative::SpeculativeVerifier* verifier,
    std::span<const tokenization::TokenId> prompt_tokens,
    std::string* reply = nullptr) {
  std::size_t generated_count = 0;
  std::vector<tokenization::TokenId> generated_ids;
  if (opt.verbose)
    generated_ids.reserve(opt.max_tokens);
  const auto on_token = [&](tokenization::TokenId token,
                            std::string_view piece) {
    std::cout << piece << std::flush;
    if (reply != nullptr)
      reply->append(piece);
    ++generated_count;
    if (opt.verbose)
      generated_ids.push_back(token);
    return true;
  };
  const auto start = std::chrono::steady_clock::now();
  if (verifier != nullptr) {
    speculative::GenerationOptions generation;
    generation.max_new_tokens = opt.max_tokens;
    generation.sampling = opt.sampling;
    (void)verifier->Generate(prompt_tokens, generation, on_token);
  } else {
    models::GenerationOptions generation;
    generation.max_new_tokens = opt.max_tokens;
    generation.sampling = opt.sampling;
    (void)executor.Generate(prompt_tokens, generation, on_token);
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cout << '\n';
  if (opt.verbose) {
    if (verifier != nullptr) {
      const auto& stats = verifier->GetStats();
      std::cerr << "[Speculative]: acceptance=" << stats.AcceptanceRate()
                << " drafted=" << stats.total_draft_tokens
                << " accepted=" << stats.total_accepted_tokens
                << " verification_steps=" << stats.total_verification_steps
                << '\n';
    }
    const double rate = seconds > 0 ? generated_count / seconds : 0;
    std::cout << "Generated " << generated_count << " tokens on GPU in "
              << seconds << "s (" << rate << " tok/s)\n";
    PrintTokenTrace(generated_ids);
  }
}

int RunQwenBenchmarkImpl(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  std::string err;
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

  std::cout << "HIP: found " << device_count << " ROCm devices (Total VRAM: "
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
  if (opt.speculative_backend == "mtp") {
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
  // A DFlash2 pp row includes the same feature capture and draft injection as
  // prompt/serving. Keep one backend alive across warmup, depth restores and
  // TG.
  std::unique_ptr<speculative::SpeculativeVerifier> dflash_verifier;
  if (opt.speculative_backend == "dflash2") {
    hip::QwenDFlashGpuDraftConfig draft_config{
        .max_context = static_cast<std::uint32_t>(required_context),
        .max_draft_tokens = opt.draft_tokens,
        .policy = speculative::ParseDFlashDraftPolicy(opt.draft_policy),
    };
    auto draft = hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
        opt.dflash_model_path, gpu_exec->GetSharedModel(), draft_config, &err);
    if (!draft) {
      throw std::runtime_error("DFlash2 initialization failed: " + err);
    }
    speculative::SpeculativeOptions options;
    options.max_draft_tokens = opt.draft_tokens;
    options.min_draft_tokens = opt.min_draft_tokens;
    options.initial_draft_tokens = opt.draft_tokens;
    options.enable_adaptive_draft_length = false;
    options.use_batched_verification = true;
    dflash_verifier = hip::CreateQwenGpuSpeculativeVerifier(
        *gpu_exec, std::move(draft), options);
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
            << " | " << std::right << std::setw(15) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n";
  std::cout << "| " << std::string(30, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(10, '-')
            << " | " << std::string(15, '-') << " | " << std::string(21, '-')
            << " |\n";

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

  const std::string_view backend_name = "ROCm (HIP)";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream ss_ts;
    ss_ts << std::fixed << std::setprecision(2) << stats.mean << " ± "
          << stats.stddev;

    std::cout << "| " << std::left << std::setw(30) << model_name << " | "
              << std::right << std::setw(10) << ss_size.str() << " | "
              << std::right << std::setw(10) << ss_params.str() << " | "
              << std::left << std::setw(10) << backend_name << " | "
              << std::right << std::setw(15) << test_name << " | " << std::right
              << std::setw(21) << ss_ts.str() << " |\n"
              << std::flush;
  };

  // Verification owns the executor's SaveState rollback slot. A benchmark
  // prefix must have an independent snapshot across speculative TG repeats.
  std::unique_ptr<hip::QwenGpuSnapshot> prepared_target;
  std::unique_ptr<speculative::SpeculativeVerifierSnapshot> prepared_draft;
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
        gpu_exec->RestoreSnapshot(*prepared_target);
        if (dflash_verifier)
          dflash_verifier->RestoreSnapshot(*prepared_draft);
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
        const auto next_token =
            dflash_verifier
                ? (preparation_start == 0
                       ? dflash_verifier->Prime(depth_tokens,
                                                compute_next_token)
                       : dflash_verifier->ExtendPrompt(
                             depth_tokens,
                             static_cast<std::uint32_t>(preparation_start),
                             compute_next_token))
                : gpu_exec->ForwardPromptBatch(
                      depth_tokens,
                      static_cast<std::uint32_t>(preparation_start),
                      compute_next_token);
        if (compute_next_token) {
          prepared_next_token = next_token;
          has_prepared_next_token = true;
        }
      }
      HIP_CHECK(hipDeviceSynchronize());
      const auto preparation_end = std::chrono::steady_clock::now();
      const auto cache_begin = preparation_end;
      prepared_target =
          gpu_exec->SaveSnapshot(static_cast<std::uint32_t>(depth));
      if (dflash_verifier)
        prepared_draft = dflash_verifier->Snapshot();
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
        gpu_exec->RestoreSnapshot(*prepared_target);
        if (dflash_verifier)
          dflash_verifier->RestoreSnapshot(*prepared_draft);
      }
      HIP_CHECK(hipDeviceSynchronize());
    };

    for (const std::size_t p_len : opt.n_prompts) {
      const auto prompt_tokens = MakeBenchmarkTokens(p_len, depth);

      const auto prefill = [&] {
        if (dflash_verifier) {
          if (depth == 0) {
            (void)dflash_verifier->Prime(prompt_tokens);
          } else {
            (void)dflash_verifier->ExtendPrompt(
                prompt_tokens, static_cast<std::uint32_t>(depth));
          }
        } else {
          (void)gpu_exec->ForwardPromptBatch(prompt_tokens,
                                             static_cast<std::uint32_t>(depth));
        }
        HIP_CHECK(hipDeviceSynchronize());
      };
      restore_depth();
      prefill();

      std::vector<double> runs;
      runs.reserve(opt.repetitions);
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        const auto t0 = std::chrono::high_resolution_clock::now();
        prefill();
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
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        restore_depth();

        std::unique_ptr<speculative::IDraftBackend> draft_backend;
        if (opt.speculative_backend == "mtp") {
          hip::QwenMtpGpuDraftConfig cfg{
              .max_context = static_cast<std::uint32_t>(required_context),
              .max_draft_tokens = opt.draft_tokens,
          };
          draft_backend =
              hip::QwenMtpGpuDraftBackend::Create(mtp_gpu_model, cfg, &err);
          if (draft_backend == nullptr) {
            throw std::runtime_error("MTP initialization failed: " + err);
          }
        }

        std::unique_ptr<speculative::SpeculativeVerifier> owned_spec_verifier;
        auto* spec_verifier = dflash_verifier.get();
        if (draft_backend) {
          speculative::SpeculativeOptions s_opts;
          s_opts.max_draft_tokens = opt.draft_tokens;
          s_opts.min_draft_tokens = opt.min_draft_tokens;
          s_opts.initial_draft_tokens = opt.draft_tokens;
          s_opts.use_batched_verification = true;
          owned_spec_verifier = hip::CreateQwenGpuSpeculativeVerifier(
              *gpu_exec, std::move(draft_backend), s_opts);
          spec_verifier = owned_spec_verifier.get();
        }

        std::vector<tokenization::TokenId> speculative_sequence;
        tokenization::TokenId speculative_current_token = 0;
        std::uint32_t speculative_current_pos = 0;
        if (spec_verifier) {
          speculative_sequence =
              MakeBenchmarkTokens(depth > 0 ? depth : std::size_t{16});
          speculative_current_token =
              dflash_verifier && depth > 0
                  ? prepared_next_token
                  : spec_verifier->Prime(speculative_sequence);
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

        std::vector<tokenization::TokenId> generated;
        if (opt.verbose)
          generated.reserve(g_len);
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
              if (opt.verbose)
                generated.push_back(t);
              ++speculative_current_pos;
              ++emitted;
              if (emitted >= g_len) {
                break;
              }
            }
            speculative_current_token = step_res.next_token;
          }
          acceptance_runs.push_back(spec_verifier->GetStats().AcceptanceRate());
        } else {
          for (std::size_t step = 0; step < g_len; ++step) {
            greedy_current_token = gpu_exec->ForwardToken(greedy_current_token,
                                                          greedy_current_pos);
            ++greedy_current_pos;
            if (opt.verbose)
              generated.push_back(greedy_current_token);
          }
        }
        HIP_CHECK(hipDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (opt.verbose) {
          const auto* bytes =
              reinterpret_cast<const std::uint8_t*>(generated.data());
          std::cerr << "[QwenBenchTrace]: depth=" << depth
                    << " count=" << generated.size() << " sha256="
                    << crypto::Sha256Hex(std::span(
                           bytes,
                           generated.size() * sizeof(tokenization::TokenId)))
                    << '\n';
        }

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
    }
  }

  std::cout << "\n";
  return 0;
}

int RunQwenPromptImpl(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader_ref,
                      std::chrono::steady_clock::time_point model_load_start) {
  const std::shared_ptr<const core::GgufReader> reader =
      std::shared_ptr<const core::GgufReader>(
          std::shared_ptr<const core::GgufReader>{}, &reader_ref);
  std::string err;
  std::string rendered_prompt = opt.prompt_text;
  std::vector<tokenization::ContentSpan> content_spans;
  std::vector<tokenization::ChatMessage> messages;
  if (opt.use_chat_template) {
    if (!opt.system_prompt.empty()) {
      messages.push_back(
          {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
    }
    messages.push_back(
        {tokenization::ChatRole::kUser, opt.prompt_text, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    const auto rendered = tokenization::QwenChatTemplate::Render(
        messages, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (rendered.has_value()) {
      rendered_prompt = *rendered;
    } else {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }
  }

  const auto encode_prompt = [&](const tokenization::QwenTokenizer& tokenizer) {
    return tokenization::QwenChatTemplate::EncodeRendered(
        tokenizer, rendered_prompt, 0, rendered_prompt.size(), content_spans,
        {});
  };

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  if (!opt.force_cpu && hipGetDeviceCount(&dev_count) == hipSuccess &&
      dev_count > 0) {
    auto gpu_exec = gufo::hip::QwenGpuExecutor::CreateFromGguf(reader, &err);
    if (gpu_exec) {
      PrintModelLoadTime(model_load_start);
      auto prompt_tokens = encode_prompt(gpu_exec->GetTokenizer());
      try {
        auto encoder = LoadQwenVision(opt, *reader);
        if (!opt.image_paths.empty()) {
          AttachImages(opt, messages.back());
          auto vision =
              PrepareVision(opt, gpu_exec->GetTokenizer(), messages, encoder);
          prompt_tokens = vision->tokens;
          gpu_exec->ConfigureVision(std::move(vision), std::move(encoder));
        }
      } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
      }

      if (opt.verbose) {
        const auto& config = gpu_exec->GetConfig();
        std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n"
                  << "Model: " << config.architecture << " ("
                  << config.num_layers
                  << " layers, hidden=" << config.hidden_size
                  << ", heads=" << config.num_attention_heads << ")\n"
                  << "Prompt tokens (" << prompt_tokens.size() << "): ";
        for (const auto t : prompt_tokens) {
          std::cout << "[" << t << ": '"
                    << gpu_exec->GetTokenizer().DecodeToken(t) << "'] ";
        }
        std::cout << "\nMax tokens: " << opt.max_tokens << "\n"
                  << "--- Generation Output ---\n";
      }

      std::unique_ptr<speculative::SpeculativeVerifier> verifier;
      try {
        if (!opt.speculative_backend.empty()) {
          verifier = CreateQwenVerifier(opt, *gpu_exec);
        }
        GenerateQwenGpuResponse(opt, *gpu_exec, verifier.get(), prompt_tokens);
      } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
      }
      return 0;
    }
    std::cerr << "Error creating Qwen GPU executor: " << err << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
#endif

  if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
    std::cerr << "Image input requires ROCm\n";
    return 1;
  }
  if (!opt.speculative_backend.empty()) {
    std::cerr << "Qwen speculative decoding requires the ROCm backend\n";
    return 1;
  }
  auto generator = models::QwenGenerator::CreateFromGguf(*reader, &err);
  if (!generator) {
    std::cerr << "Error creating Qwen generator: " << err << "\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  const auto prompt_tokens = encode_prompt(generator->GetTokenizer());

  if (opt.verbose) {
    const auto& config = generator->GetConfig();
    std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n"
              << "Model: " << config.architecture << " (" << config.num_layers
              << " layers, hidden=" << config.hidden_size
              << ", heads=" << config.num_attention_heads << ")\n"
              << "Prompt tokens: " << prompt_tokens.size() << "\n"
              << "Max tokens: " << opt.max_tokens << "\n"
              << "Temperature: " << opt.sampling.temperature << "\n"
              << "--- Generation Output ---\n";
  }

  models::GenerationOptions gen_opts;
  gen_opts.max_new_tokens = opt.max_tokens;
  gen_opts.sampling = opt.sampling;

  const auto start_time = std::chrono::steady_clock::now();
  std::size_t generated_count = 0;

  const auto tokens = generator->Generate(
      prompt_tokens, gen_opts,
      [&](tokenization::TokenId, std::string_view piece) -> bool {
        std::cout << piece << std::flush;
        ++generated_count;
        return true;
      });

  std::cout << "\n";

  if (opt.verbose && generated_count > 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    const double sec = static_cast<double>(elapsed.count()) / 1000.0;
    const double tok_per_sec =
        (sec > 0.0) ? (static_cast<double>(generated_count) / sec) : 0.0;
    std::cout << "Generated " << generated_count << " tokens on CPU in " << sec
              << "s (" << tok_per_sec << " tok/s)\n";
  }

  return 0;
}

int RunQwenChatImpl(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start) {
  const tokenization::QwenTokenizer* tokenizer = nullptr;
  std::string architecture;
  std::unique_ptr<models::QwenGenerator> generator;
#if defined(ENGINE_ENABLE_HIP)
  std::unique_ptr<hip::QwenGpuExecutor> gpu_executor;
  std::shared_ptr<models::qwen::vision::Encoder> vision_encoder;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier;
  int device_count = 0;
  if (!opt.force_cpu && hipGetDeviceCount(&device_count) == hipSuccess &&
      device_count > 0) {
    std::string err;
    auto reader_ptr = std::shared_ptr<const core::GgufReader>(
        std::shared_ptr<const core::GgufReader>{}, &reader);
    gpu_executor = hip::QwenGpuExecutor::CreateFromGguf(reader_ptr, &err);
    if (!gpu_executor) {
      std::cerr << "Error creating Qwen GPU executor: " << err << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    try {
      vision_encoder = LoadQwenVision(opt, reader);
      if (!opt.speculative_backend.empty()) {
        verifier = CreateQwenVerifier(opt, *gpu_executor);
      }
    } catch (const std::exception& exception) {
      std::cerr << exception.what() << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    tokenizer = &gpu_executor->GetTokenizer();
    architecture = gpu_executor->GetConfig().architecture;
    if (opt.verbose)
      std::cout << "[Engine]: AMD Strix Halo gfx1151 GPU Executor\n";
  }
#endif
  if (tokenizer == nullptr) {
    if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
      std::cerr << "Image input requires ROCm\n";
      return 1;
    }
    if (!opt.speculative_backend.empty()) {
      std::cerr << "Qwen speculative decoding requires the ROCm backend\n";
      return 1;
    }
    std::string err;
    generator = models::QwenGenerator::CreateFromGguf(reader, &err);
    if (!generator) {
      std::cerr << "Error creating Qwen generator: " << err << '\n';
      PrintModelLoadTime(model_load_start, false);
      return 1;
    }
    tokenizer = &generator->GetTokenizer();
    architecture = generator->GetConfig().architecture;
    if (opt.verbose)
      std::cout << "[Engine]: CPU (OpenMP Multi-Threaded)\n";
  }
  PrintModelLoadTime(model_load_start);

  std::cout << "=== Gufo Interactive Chat (" << architecture << ") ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back(
        {tokenization::ChatRole::kSystem, opt.system_prompt, "", ""});
  }

  bool first_turn = true;
  std::string user_input;
  while (true) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, user_input)) {
      break;
    }
    if (user_input == "exit" || user_input == "quit") {
      break;
    }
    if (user_input.empty()) {
      continue;
    }

    history.push_back({tokenization::ChatRole::kUser, user_input, "", ""});
    const auto reasoning = PromptReasoningOptions(opt);
    std::vector<tokenization::ContentSpan> content_spans;
    const auto rendered_prompt = tokenization::QwenChatTemplate::Render(
        history, {},
        tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id),
        nullptr, nullptr, nullptr, &content_spans);
    if (!rendered_prompt.has_value()) {
      std::cerr << "Error formatting chat template.\n";
      return 1;
    }

    auto prompt_tokens = tokenization::QwenChatTemplate::EncodeRendered(
        *tokenizer, *rendered_prompt, 0, rendered_prompt->size(), content_spans,
        {});

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;

    try {
#if defined(ENGINE_ENABLE_HIP)
      if (first_turn)
        AttachImages(opt, history.back());
      first_turn = false;
      if (!opt.image_paths.empty()) {
        auto vision = PrepareVision(opt, *tokenizer, history, vision_encoder);
        prompt_tokens = vision->tokens;
        if (gpu_executor)
          gpu_executor->ConfigureVision(vision, vision_encoder);
      }
      if (gpu_executor != nullptr) {
        GenerateQwenGpuResponse(opt, *gpu_executor, verifier.get(),
                                prompt_tokens, &assistant_reply);
      } else
#endif
      {
        models::GenerationOptions generation;
        generation.max_new_tokens = opt.max_tokens;
        generation.sampling = opt.sampling;
        (void)generator->Generate(
            prompt_tokens, generation,
            [&](tokenization::TokenId, std::string_view piece) {
              std::cout << piece << std::flush;
              assistant_reply += piece;
              return true;
            });
        std::cout << '\n';
      }
    } catch (const std::exception& exception) {
      std::cerr << "Generation failed: " << exception.what() << '\n';
      return 1;
    }

    std::cout << '\n';
    tokenization::ChatMessage reply{tokenization::ChatRole::kAssistant, ""};
    if (tokenization::ResolveQwenChatOptions(reasoning, opt.add_vision_id)
            .enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = assistant_reply.find(end);
      reply.thought = assistant_reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        reply.content = assistant_reply.substr(boundary + end.size());
      }
    } else {
      reply.content = std::move(assistant_reply);
    }
    history.push_back(std::move(reply));
  }

  return 0;
}

}  // namespace

int RunQwenBenchmark(const ext::TextBenchOptions& opt,
                     const std::shared_ptr<const core::GgufReader>& reader,
                     std::chrono::steady_clock::time_point model_load_start) {
  return RunQwenBenchmarkImpl(opt, reader, model_load_start);
}

int RunQwenPrompt(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start) {
  return RunQwenPromptImpl(opt, reader, model_load_start);
}

int RunQwenChat(const ext::TextPromptOptions& opt,
                const core::GgufReader& reader,
                std::chrono::steady_clock::time_point model_load_start) {
  return RunQwenChatImpl(opt, reader, model_load_start);
}
}  // namespace gufo::models::qwen
