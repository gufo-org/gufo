#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

using TokenId = gufo::tokenization::TokenId;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<TokenId> GenerateDirect(gufo::hip::QwenGpuExecutor& executor,
                                    std::span<const TokenId> prompt_tokens,
                                    std::size_t max_tokens) {
  gufo::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  options.sampling.temperature = 0.0F;
  return executor.Generate(prompt_tokens, options);
}

void ExpectStableGpuMemory(std::size_t before, std::size_t after) {
  constexpr std::size_t tolerance = 16ULL * 1024ULL * 1024ULL;
  Expect(after + tolerance >= before,
         "request cleanup leaked more than 16 MiB of GPU memory");
}

void CheckSamplingStrategies(
    gufo::server::InferenceBackend& backend,
    gufo::server::InferenceBackend* speculative_backend = nullptr) {
  constexpr std::uint32_t token_count = 8;
  struct Reference {
    std::string prompt;
    gufo::sampling::SamplingConfig config;
    gufo::server::InferenceBackend::Result ar;
    gufo::server::InferenceBackend::Result speculative;
  };
  std::vector<Reference> references;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    const auto prompt = "Sampling " + std::string(test.name) +
                        ": Continue red, blue, blue, red,";
    const auto ar = backend.complete(prompt, token_count, test.config);
    const auto ar_replay = backend.complete(prompt, token_count, test.config);
    Expect(ar.completion_tokens == token_count && ar.draft_tokens == 0 &&
               ar_replay.cache_hit && ar.tokens == ar_replay.tokens,
           "AR strategy must reproduce all cold/cached token IDs");
    Reference reference{
        .prompt = prompt, .config = test.config, .ar = ar, .speculative = {}};
    if (speculative_backend != nullptr) {
      const auto spec =
          speculative_backend->complete(prompt, token_count, test.config);
      const auto replay =
          speculative_backend->complete(prompt, token_count, test.config);
      Expect(spec.completion_tokens == token_count && spec.draft_tokens > 0 &&
                 !spec.cache_hit && replay.cache_hit &&
                 spec.tokens == replay.tokens,
             "DFlash2 strategy must draft and reproduce cold/cached token IDs");
      Expect(ar.tokens.front() == spec.tokens.front(),
             "AR/DFlash2 first-token sampling differs");
      if (!test.config.uses_random_sampling())
        Expect(ar.tokens == spec.tokens,
               "deterministic DFlash2 strategy differs from AR");
      reference.speculative = spec;
    }
    references.push_back(std::move(reference));
    std::cout << "Sampling replay passed: " << test.name
              << (speculative_backend ? " (AR + DFlash2)" : " (AR)") << '\n';
  }
  const auto check_concurrent = [&](gufo::server::InferenceBackend& current,
                                    bool speculative, std::size_t width,
                                    std::size_t offset) {
    std::vector<std::future<gufo::server::InferenceBackend::Result>> pending;
    pending.reserve(width);
    for (std::size_t row = 0; row < width; ++row) {
      const auto* reference = &references[(offset + row) % references.size()];
      pending.push_back(std::async(std::launch::async, [&current, reference] {
        return current.complete(reference->prompt, token_count,
                                reference->config);
      }));
    }
    std::size_t physical_width = 1;
    for (std::size_t row = 0; row < width; ++row) {
      const auto result = pending[row].get();
      const auto& reference = references[(offset + row) % references.size()];
      const auto& expected = speculative ? reference.speculative : reference.ar;
      Expect(result.tokens == expected.tokens &&
                 result.draft_tokens == expected.draft_tokens &&
                 result.draft_accepted_tokens == expected.draft_accepted_tokens,
             "concurrent sampling changed isolated tokens or draft acceptance");
      physical_width =
          std::max(physical_width, result.physical_execution_width);
    }
    Expect(physical_width <= width,
           "concurrent sampling reported an invalid physical width");
    std::cout << "Sampling concurrency=" << width
              << " physical_width=" << physical_width
              << (speculative ? " DFlash2" : " AR") << " exact=1\n";
    return physical_width;
  };
  // Reuse the isolated oracles above. The final C8 group wraps around so all
  // 23 strategies participate, including greedy penalties and combined floors.
  std::size_t offset = 0;
  bool shared_ar = false;
  bool shared_speculative = false;
  for (const std::size_t width : {2U, 4U, 6U, 8U, 8U}) {
    shared_ar |= check_concurrent(backend, false, width, offset) > 1;
    if (speculative_backend != nullptr)
      shared_speculative |=
          check_concurrent(*speculative_backend, true, width, offset) > 1;
    offset += width;
  }
  Expect(shared_ar && (speculative_backend == nullptr || shared_speculative),
         "concurrent sampling did not exercise shared target execution");
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::array<char, 96> pattern{};
    const std::string path =
        (std::filesystem::temp_directory_path() / "gufo-qwen-disk-cache-XXXXXX")
            .string();
    Expect(path.size() + 1 <= pattern.size(),
           "temporary Qwen cache path fits fixed buffer");
    std::copy(path.begin(), path.end(), pattern.begin());
    const char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create Qwen cache directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

void RunPromptReuseSmoke(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
    gufo::server::InferenceBackend& backend) {
  const std::vector<gufo::tokenization::ChatMessage> messages = {
      {gufo::tokenization::ChatRole::kSystem, "Answer with one short sentence.",
       "", ""},
      {gufo::tokenization::ChatRole::kUser, "Name one primary color.", "", ""},
  };
  const auto rendered_chat =
      gufo::tokenization::QwenChatTemplate::Render(messages);
  Expect(rendered_chat.has_value() && !rendered_chat->empty(),
         "prefix-reuse chat prompt rendering");
  const auto prompt = model->GetTokenizer().Encode(*rendered_chat);
  const auto root = backend.chat(messages, 1, {});
  Expect(
      !root.tokens.empty() && !root.cache_hit && root.cache_snapshot_bytes > 0,
      "cold prefix-reuse chat must publish one prompt snapshot");

  const auto repeated = backend.chat(messages, 1, {});
  Expect(repeated.cache_hit && repeated.cached_prompt_tokens == prompt.size() &&
             repeated.prefill_tokens == 0,
         "repeated chat must restore the complete prompt boundary");
  Expect(repeated.tokens == root.tokens,
         "repeated chat differs from the cold request");
  Expect(repeated.cache_snapshot_bytes == 0,
         "exact prompt reuse must avoid another full snapshot copy");

  auto continued_messages = messages;
  continued_messages.emplace_back(gufo::tokenization::ChatRole::kAssistant,
                                  "Blue.");
  continued_messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                  "Name a different primary color.");
  const auto rendered_continuation =
      gufo::tokenization::QwenChatTemplate::Render(continued_messages);
  Expect(rendered_continuation.has_value(),
         "prefix-reuse continuation prompt rendering");
  const auto continuation_prompt =
      model->GetTokenizer().Encode(*rendered_continuation);
  const auto continued = backend.chat(continued_messages, 1, {});
  Expect(continued.cache_hit &&
             continued.cached_prompt_tokens == prompt.size() &&
             continued.prefill_tokens ==
                 continuation_prompt.size() - prompt.size(),
         "continuation must prefill only the suffix after the cached prompt");
  Expect(!continued.tokens.empty(),
         "cached continuation must generate at least one token");
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model_path =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
    if (model_path == nullptr) {
      std::cout << "SKIP: pass a Qwen GGUF path for the gfx1151 server test; "
                   "append --full for the exhaustive suite\n";
      return 77;
    }

    const char* draft_model_path = std::getenv("GUFO_QWEN27B_DFLASH_MODEL");
    bool run_full_suite = draft_model_path != nullptr;
    bool sampling_only = false;
    bool cache_only = false;
    auto sampling_policy = gufo::speculative::DFlashDraftPolicy::kAdaptive;
    for (int index = 2; index < argc; ++index) {
      const std::string_view argument = argv[index];
      if (argument == "--full") {
        run_full_suite = true;
      } else if (argument == "--sampling-only") {
        run_full_suite = sampling_only = true;
      } else if (argument == "--cache-only") {
        run_full_suite = cache_only = true;
      } else if (argument == "--fixed") {
        sampling_policy = gufo::speculative::DFlashDraftPolicy::kFixed;
      } else if (argument.starts_with("--")) {
        throw std::invalid_argument("unknown Qwen GPU test option");
      } else if (draft_model_path == nullptr) {
        draft_model_path = argv[index];
        run_full_suite = true;
      } else {
        throw std::invalid_argument("multiple DFlash model paths provided");
      }
    }

    std::string error;
    auto reader_owner = gufo::core::GgufReader::OpenFile(model_path, &error);
    Expect(reader_owner != nullptr, error);
    const std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(reader_owner));
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    Expect(model->GetWeightRegionCount() == reader->GetMappedRegions().size(),
           "every mapped GGUF shard must have one shared GPU region");

    const std::uint32_t context = run_full_suite ? 256U : 64U;
    const std::size_t state_count = run_full_suite ? 8U : 1U;
    gufo::server::InferenceBackend backend;
    Expect(!backend.load(
               model, &error, context, state_count, {}, {},
               {.backend = gufo::server::TextSpeculativeBackend::kMtp}) &&
               error.find("Flash-Next") != std::string::npos,
           "Qwen27B must reject MTP instead of silently running AR");
    Expect(
        !backend.load(model, &error, context, state_count, {}, {},
                      {.backend = gufo::server::TextSpeculativeBackend::kDFlash,
                       .min_draft_tokens = 2}) &&
            error.find("min-draft-tokens") != std::string::npos,
        "DFlash must reject an unused adaptive draft floor");
    Expect(backend.load(model, &error, context, state_count, {},
                        {.max_pending_requests_per_client = 8}),
           error);
    Expect(backend.model_id() == model->GetConfig().model_name,
           "HTTP model identifier");

    if (sampling_only) {
      if (draft_model_path == nullptr) {
        CheckSamplingStrategies(backend);
      } else {
        gufo::server::InferenceBackend speculative_backend;
        Expect(speculative_backend.load(
                   model, &error, context, state_count,
                   {.decode_active_tokens = 8},
                   {.max_pending_requests_per_client = 8},
                   {.backend = gufo::server::TextSpeculativeBackend::kDFlash,
                    .draft_model_path = draft_model_path,
                    .max_draft_tokens = 7,
                    .min_draft_tokens = 1,
                    .dflash_policy = sampling_policy}),
               error);
        CheckSamplingStrategies(backend, &speculative_backend);
      }
      std::cout << "All Qwen sampling strategy replays passed.\n";
      return 0;
    }

    if (!run_full_suite) {
      RunPromptReuseSmoke(model, backend);
      std::cout << "Qwen continuation prefix-reuse smoke test passed.\n";
      return 0;
    }

    auto direct = gufo::hip::QwenGpuExecutor::Create(model, &error, context);
    Expect(direct != nullptr, error);

    const std::string raw_prompt = "The capital of France is";
    const auto raw_prompt_tokens = model->GetTokenizer().Encode(raw_prompt);

    {
      auto snapshot_source =
          gufo::hip::QwenGpuExecutor::Create(model, &error, context);
      Expect(snapshot_source != nullptr, error);
      const auto frontier =
          snapshot_source->ForwardPromptBatch(raw_prompt_tokens);
      auto snapshot = snapshot_source->SaveSnapshot(
          static_cast<std::uint32_t>(raw_prompt_tokens.size()));
      Expect(snapshot != nullptr && snapshot->PayloadBytes() > 0,
             "Qwen snapshot must own an accounted payload");
      const auto uninterrupted = snapshot_source->ForwardToken(
          frontier, static_cast<std::uint32_t>(raw_prompt_tokens.size()));

      for (int fork_index = 0; fork_index < 2; ++fork_index) {
        auto fork = gufo::hip::QwenGpuExecutor::Create(model, &error, context);
        Expect(fork != nullptr, error);
        fork->RestoreSnapshot(*snapshot);
        const auto forked = fork->ForwardToken(
            frontier, static_cast<std::uint32_t>(raw_prompt_tokens.size()));
        Expect(forked == uninterrupted,
               "Qwen snapshot fork differs from uninterrupted execution");
      }
    }

    const auto direct_raw = GenerateDirect(*direct, raw_prompt_tokens, 2);
    const auto http_raw = backend.complete(raw_prompt, 2, 0.0F);
    Expect(http_raw.tokens == direct_raw,
           "raw HTTP and direct executor tokens differ");
    Expect(http_raw.text == model->GetTokenizer().Decode(direct_raw),
           "raw HTTP text must decode the exact generated tokens");
    Expect(http_raw.ttft_ms > 0.0, "raw HTTP TTFT must be reported");

    if (draft_model_path != nullptr) {
      gufo::server::InferenceBackend speculative_backend;
      Expect(
          speculative_backend.load(
              model, &error, context, state_count, {.decode_active_tokens = 8},
              {.max_pending_requests_per_client = 8},
              gufo::server::TextSpeculativeConfig{
                  .backend = gufo::server::TextSpeculativeBackend::kDFlash,
                  .draft_model_path = draft_model_path,
                  .max_draft_tokens = 7,
                  .min_draft_tokens = 1,
                  .dflash_policy =
                      gufo::speculative::DFlashDraftPolicy::kAdaptive,
              }),
          error);
      const auto direct_spec = GenerateDirect(*direct, raw_prompt_tokens, 8);
      const auto http_spec = speculative_backend.complete(raw_prompt, 8, 0.0F);
      Expect(http_spec.tokens == direct_spec,
             "DFlash HTTP and direct greedy tokens differ");
      Expect(http_spec.draft_tokens > 0,
             "DFlash HTTP request did not draft any tokens");
      Expect(http_spec.draft_accepted_tokens <= http_spec.draft_tokens,
             "DFlash HTTP acceptance metrics are invalid");
      Expect(!http_spec.cache_hit,
             "first DFlash HTTP request must be a cache miss");

      const gufo::sampling::SamplingConfig sampling{
          .temperature = 0.8F,
          .top_k = 40,
          .top_p = 0.9F,
          .min_p = 0.01F,
          .seed = 73,
          .repeat_penalty = 1.05F,
      };
      const auto sampled_spec = speculative_backend.complete(
          "Choose an unusual English noun:", 8, sampling);
      Expect(sampled_spec.completion_tokens > 0,
             "sampled DFlash-enabled request produced no tokens");
      Expect(sampled_spec.draft_tokens > 0,
             "nonzero-temperature DFlash request bypassed drafting");
      Expect(sampled_spec.draft_accepted_tokens <= sampled_spec.draft_tokens,
             "sampled DFlash acceptance metrics are invalid");
      const auto sampled_replay = speculative_backend.complete(
          "Choose an unusual English noun:", 8, sampling);
      Expect(sampled_replay.cache_hit &&
                 sampled_replay.tokens == sampled_spec.tokens,
             "seeded DFlash cached replay differs from the cold request");

      if (!cache_only)
        CheckSamplingStrategies(backend, &speculative_backend);

      // Before any proposal draws, AR and DFlash must use exactly the same
      // sampler, including when a saved host frontier replaces device logits.
      const std::string first_token_prompt =
          "Cobalt ribbons decorate the stage. Continue the sentence:";
      for (const bool filtered : {false, true}) {
        for (const auto seed : {1, 73, 808}) {
          auto first_token_sampling = sampling;
          first_token_sampling.seed = seed;
          if (!filtered) {
            first_token_sampling.top_k = 0;
            first_token_sampling.top_p = 1.0F;
            first_token_sampling.min_p = 0.0F;
            first_token_sampling.repeat_penalty = 1.0F;
          }
          const auto ar_first =
              backend.complete(first_token_prompt, 1, first_token_sampling);
          const auto spec_first = speculative_backend.complete(
              first_token_prompt, 1, first_token_sampling);
          Expect(ar_first.tokens == spec_first.tokens &&
                     spec_first.completion_tokens == 1 &&
                     spec_first.draft_tokens == 0,
                 "AR and speculative cold/cached first-token sampling differ");
        }
      }

      const gufo::sampling::SamplingConfig greedy_penalties{
          .temperature = 0.0F,
          .seed = 73,
          .repeat_penalty = 1.1F,
          .repeat_last_n = 8,
          .frequency_penalty = 0.15F,
          .presence_penalty = 0.1F,
      };
      const auto penalized_ar =
          backend.complete(raw_prompt, 16, greedy_penalties);
      const auto penalized_spec =
          speculative_backend.complete(raw_prompt, 16, greedy_penalties);
      Expect(penalized_spec.tokens == penalized_ar.tokens,
             "greedy penalized DFlash tokens differ from AR");
      Expect(penalized_spec.draft_tokens > 0,
             "greedy penalties bypassed DFlash drafting");
      const auto penalized_replay =
          speculative_backend.complete(raw_prompt, 16, greedy_penalties);
      Expect(penalized_replay.cache_hit &&
                 penalized_replay.tokens == penalized_spec.tokens,
             "greedy penalized DFlash cached replay changed tokens");

      const std::vector<gufo::tokenization::ChatMessage> spec_messages = {
          {gufo::tokenization::ChatRole::kSystem,
           "Answer with one short sentence.", "", ""},
          {gufo::tokenization::ChatRole::kUser, "Name one primary color.", "",
           ""},
      };
      // This fixture must actually reach EOS, rather than truncate the
      // now-default thinking trace at its 32-token budget.
      gufo::server::ChatRequest spec_request(spec_messages);
      spec_request.reasoning.enabled = false;
      const auto template_options =
          gufo::tokenization::ResolveQwenChatOptions(spec_request.reasoning);
      const auto first_spec_chat =
          speculative_backend.chat(spec_request, 32, {});
      const auto stop_prompt = gufo::tokenization::QwenChatTemplate::Render(
          spec_messages, template_options);
      Expect(stop_prompt.has_value(), "DFlash stop prompt rendering");
      const auto first_ar_tokens = GenerateDirect(
          *direct, model->GetTokenizer().Encode(*stop_prompt), 32);
      Expect(first_spec_chat.finish_reason ==
                     gufo::server::TextGenerationBackend::FinishReason::kStop &&
                 first_spec_chat.tokens == first_ar_tokens,
             "DFlash must reach EOS with the same published tokens as AR");
      Expect(!first_spec_chat.cache_hit,
             "first DFlash chat request must be a cache miss");
      Expect(first_spec_chat.cache_snapshot_bytes > 0,
             "DFlash root must retain target and draft snapshots");

      auto continued_spec_messages = spec_messages;
      continued_spec_messages.emplace_back(
          gufo::tokenization::ChatRole::kAssistant, first_spec_chat.text);
      continued_spec_messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                           "Name a different primary color.");
      auto forked_spec_messages = spec_messages;
      forked_spec_messages.emplace_back(
          gufo::tokenization::ChatRole::kAssistant, first_spec_chat.text);
      forked_spec_messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                        "Name one warm primary color.");
      const auto rendered_spec_continuation =
          gufo::tokenization::QwenChatTemplate::Render(continued_spec_messages,
                                                       template_options);
      Expect(rendered_spec_continuation.has_value(),
             "DFlash continuation prompt rendering");
      const auto rendered_spec_fork =
          gufo::tokenization::QwenChatTemplate::Render(forked_spec_messages,
                                                       template_options);
      Expect(rendered_spec_fork.has_value(), "DFlash fork prompt rendering");
      const auto direct_spec_continuation = GenerateDirect(
          *direct, model->GetTokenizer().Encode(*rendered_spec_continuation),
          2);
      const auto direct_spec_fork = GenerateDirect(
          *direct, model->GetTokenizer().Encode(*rendered_spec_fork), 2);

      gufo::server::ChatRequest continued_spec_request(continued_spec_messages);
      continued_spec_request.reasoning = spec_request.reasoning;
      continued_spec_request.client_id = "dflash-snapshot-branch-a";
      gufo::server::ChatRequest forked_spec_request(forked_spec_messages);
      forked_spec_request.reasoning = spec_request.reasoning;
      forked_spec_request.client_id = "dflash-snapshot-branch-b";
      auto pending_spec_continuation =
          speculative_backend.start_chat(continued_spec_request, 2, 0.0F);
      auto pending_spec_fork =
          speculative_backend.start_chat(forked_spec_request, 2, 0.0F);
      Expect(
          pending_spec_continuation != nullptr && pending_spec_fork != nullptr,
          "concurrent DFlash snapshot branches are admitted");
      const auto cached_spec_continuation = pending_spec_continuation->Wait();
      const auto cached_spec_fork = pending_spec_fork->Wait();
      for (const auto* result :
           {&cached_spec_continuation, &cached_spec_fork}) {
        std::cout << "DFlash branch: cached=" << result->cached_prompt_tokens
                  << " restore_bytes=" << result->cache_restore_bytes
                  << " snapshot_bytes=" << result->cache_snapshot_bytes << '\n';
        Expect(result->cache_hit,
               "DFlash branch must restore the retained root");
        Expect(result->cached_prompt_tokens > 0 &&
                   result->cached_prompt_tokens < result->prompt_tokens,
               "DFlash branch reports its reused prefix and cold suffix");
        Expect(
            result->cache_snapshot_bytes > 0 &&
                (result->cache_restore_bytes > 0 ||
                 result->cached_prompt_tokens >= first_spec_chat.prompt_tokens),
            "DFlash branch retains a snapshot or the completed live frontier");
      }
      Expect(cached_spec_continuation.cache_restore_bytes > 0 ||
                 cached_spec_fork.cache_restore_bytes > 0,
             "one divergent DFlash branch must restore its immutable snapshot");
      Expect(cached_spec_continuation.tokens == direct_spec_continuation,
             "cached DFlash continuation differs from cold target execution");
      Expect(cached_spec_fork.tokens == direct_spec_fork,
             "forked DFlash continuation differs from cold target execution");
    }

    // Cache-prefix fixtures disable thinking: reinserting an unfinished
    // reasoning trace as ordinary assistant content changes the template
    // prefix rather than extending the saved prompt.
    const gufo::tokenization::ChatTemplateOptions cache_template{
        .enable_thinking = false};
    const auto cache_request = [](const auto& messages) {
      gufo::server::ChatRequest request(messages);
      request.reasoning.enabled = false;
      return request;
    };
    const std::vector<gufo::tokenization::ChatMessage> messages = {
        {gufo::tokenization::ChatRole::kSystem,
         "Answer with one short sentence.", "", ""},
        {gufo::tokenization::ChatRole::kUser, "Name one primary color.", "",
         ""},
    };
    const auto rendered_chat =
        gufo::tokenization::QwenChatTemplate::Render(messages, cache_template);
    Expect(rendered_chat.has_value() && !rendered_chat->empty(),
           "CLI chat prompt rendering");
    const auto chat_prompt = model->GetTokenizer().Encode(*rendered_chat);
    const auto direct_chat = GenerateDirect(*direct, chat_prompt, 2);
    const auto http_chat = backend.chat(cache_request(messages), 2, {});
    Expect(http_chat.tokens == direct_chat,
           "chat HTTP and direct executor tokens differ");
    Expect(!http_chat.cache_hit, "first chat request must be a cache miss");
    const auto repeated_chat = backend.chat(cache_request(messages), 2, {});
    Expect(repeated_chat.cache_hit &&
               repeated_chat.cached_prompt_tokens == chat_prompt.size() &&
               repeated_chat.prefill_tokens == 0,
           "repeated Qwen chat restores the complete prompt boundary");
    Expect(repeated_chat.tokens == direct_chat,
           "repeated Qwen chat differs from cold target execution");
    Expect(repeated_chat.cache_snapshot_bytes == 0,
           "exact Qwen reuse avoids another full snapshot copy");

    auto continued_messages = messages;
    continued_messages.emplace_back(gufo::tokenization::ChatRole::kAssistant,
                                    http_chat.text);
    continued_messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                    "Name a different primary color.");
    auto forked_messages = messages;
    forked_messages.emplace_back(gufo::tokenization::ChatRole::kAssistant,
                                 http_chat.text);
    forked_messages.emplace_back(gufo::tokenization::ChatRole::kUser,
                                 "Name one warm primary color.");
    const auto rendered_continuation =
        gufo::tokenization::QwenChatTemplate::Render(continued_messages,
                                                     cache_template);
    const auto rendered_fork = gufo::tokenization::QwenChatTemplate::Render(
        forked_messages, cache_template);
    Expect(rendered_continuation.has_value(),
           "continued chat prompt rendering");
    Expect(rendered_fork.has_value(), "forked chat prompt rendering");
    const auto continuation_prompt =
        model->GetTokenizer().Encode(*rendered_continuation);
    const auto fork_prompt = model->GetTokenizer().Encode(*rendered_fork);

    {
      const auto snapshot_root = GenerateDirect(*direct, chat_prompt, 2);
      Expect(snapshot_root == direct_chat,
             "Qwen direct snapshot root is not deterministic");
      const auto root_tokens = chat_prompt.size() + snapshot_root.size();
      auto snapshot =
          direct->SaveSnapshot(static_cast<std::uint32_t>(root_tokens));
      Expect(
          snapshot->PayloadBytes() ==
                  direct->SnapshotPayloadBytes(root_tokens) &&
              snapshot->PayloadBytes() <
                  direct->GetMemoryUsage().request_state_bytes &&
              snapshot->CompactPayloadBytes() == snapshot->PayloadBytes() + 80,
          "Qwen snapshot retains only live KV and recurrent layers");
      gufo::models::GenerationOptions options;
      options.max_new_tokens = 2;
      options.sampling.temperature = 0.0F;
      // Direct prefix restoration takes exact executed IDs. Server transcript
      // rendering and prefix matching are exercised separately below.
      auto snapshot_continuation = chat_prompt;
      snapshot_continuation.insert(snapshot_continuation.end(),
                                   snapshot_root.begin(), snapshot_root.end());
      const auto suffix =
          model->GetTokenizer().Encode(" Name a different primary color.");
      snapshot_continuation.insert(snapshot_continuation.end(), suffix.begin(),
                                   suffix.end());
      const auto live_continuation = direct->GenerateFromPrefix(
          snapshot_continuation, root_tokens, options);
      const auto cold_continuation =
          GenerateDirect(*direct, snapshot_continuation, 2);
      Expect(live_continuation == cold_continuation,
             "Qwen live prefix plus divergent suffix differs from cold "
             "prefill");

      auto restored =
          gufo::hip::QwenGpuExecutor::Create(model, &error, context);
      Expect(restored != nullptr, error);
      // Restore over an unrelated, longer history. Rows after the snapshot
      // frontier may remain stale, but no attention path may read them.
      (void)GenerateDirect(*restored, fork_prompt, 4);
      restored->RestoreSnapshot(*snapshot);
      const auto restored_continuation = restored->GenerateFromPrefix(
          snapshot_continuation, root_tokens, options);
      const auto restored_cold_continuation =
          GenerateDirect(*direct, snapshot_continuation, 2);
      Expect(restored_continuation == restored_cold_continuation,
             "Qwen snapshot plus divergent suffix differs from cold prefill");
    }

    const auto direct_continuation =
        GenerateDirect(*direct, continuation_prompt, 2);
    const auto direct_fork = GenerateDirect(*direct, fork_prompt, 2);

    {
      TemporaryDirectory cache_directory;
      const gufo::server::TextDiskCacheConfig disk_cache{
          .directory = cache_directory.path(),
          .capacity_bytes = 1024ULL * 1024ULL * 1024ULL,
          .staging_capacity_bytes = 512ULL * 1024ULL * 1024ULL,
          .model_artifact_fingerprint = std::string(64, 'a'),
          .draft_model_artifact_fingerprint = {},
      };
      {
        gufo::server::InferenceBackend writer;
        Expect(writer.load(model, &error, context, 1, {}, {}, {}, disk_cache),
               error);
        const auto persistent_root =
            writer.chat(cache_request(messages), 2, {});
        Expect(persistent_root.tokens == direct_chat,
               "Qwen disk writer differs from cold target execution");
        Expect(persistent_root.text == http_chat.text,
               "Qwen disk writer produced a different reusable root");
        Expect(persistent_root.cache_disk_queued_bytes > 0,
               "Qwen disk writer did not publish a compact snapshot");
      }

      gufo::server::InferenceBackend restarted;
      Expect(restarted.load(model, &error, context, 1, {}, {}, {}, disk_cache),
             error);
      const auto restored_continuation =
          restarted.chat(cache_request(continued_messages), 2, {});
      Expect(restored_continuation.cache_hit &&
                 restored_continuation.cache_disk_hit,
             "fresh Qwen backend did not restore its disk prefix");
      Expect(restored_continuation.cached_prompt_tokens == chat_prompt.size() &&
                 restored_continuation.cached_prompt_tokens <
                     restored_continuation.prompt_tokens,
             "Qwen disk restore did not report its exact reusable prefix");
      Expect(restored_continuation.cache_restore_bytes > 0,
             "Qwen disk restore did not report restored bytes");
      Expect(restored_continuation.tokens == direct_continuation,
             "Qwen disk continuation differs from cold full prefill");

      auto incompatible_cache = disk_cache;
      incompatible_cache.model_artifact_fingerprint = std::string(64, 'b');
      gufo::server::InferenceBackend incompatible;
      Expect(incompatible.load(model, &error, context, 1, {}, {}, {},
                               incompatible_cache),
             error);
      const auto incompatible_result =
          incompatible.chat(cache_request(continued_messages), 2, {});
      Expect(
          !incompatible_result.cache_hit && !incompatible_result.cache_disk_hit,
          "changed Qwen artifact fingerprint must be a cold miss");
      Expect(incompatible_result.tokens == direct_continuation,
             "Qwen compatibility miss changed cold execution");

      if (draft_model_path != nullptr) {
        const gufo::server::TextSpeculativeConfig dflash_config{
            .backend = gufo::server::TextSpeculativeBackend::kDFlash,
            .draft_model_path = draft_model_path,
            .max_draft_tokens = 7,
            .min_draft_tokens = 1,
        };
        auto dflash_disk_cache = disk_cache;
        dflash_disk_cache.draft_model_artifact_fingerprint =
            std::string(64, 'c');
        const auto direct_dflash_continuation =
            GenerateDirect(*direct, continuation_prompt, 8);

        gufo::server::InferenceBackend warm_dflash;
        Expect(
            warm_dflash.load(model, &error, context, 1, {}, {}, dflash_config),
            error);
        const auto warm_dflash_root =
            warm_dflash.chat(cache_request(messages), 2, {});
        const auto warm_dflash_continuation =
            warm_dflash.chat(cache_request(continued_messages), 8, {});
        Expect(warm_dflash_continuation.cache_hit &&
                   !warm_dflash_continuation.cache_disk_hit,
               "warm DFlash reference restores its in-memory prefix");
        Expect(warm_dflash_continuation.tokens == direct_dflash_continuation,
               "warm DFlash continuation differs from cold target execution");
        Expect(warm_dflash_continuation.draft_tokens > 0,
               "warm DFlash continuation did not exercise drafting");

        {
          gufo::server::InferenceBackend dflash_writer;
          Expect(dflash_writer.load(model, &error, context, 1, {}, {},
                                    dflash_config, dflash_disk_cache),
                 error);
          const auto persistent_dflash_root =
              dflash_writer.chat(cache_request(messages), 2, {});
          Expect(persistent_dflash_root.tokens == warm_dflash_root.tokens,
                 "DFlash disk writer root differs from warm execution");
          Expect(persistent_dflash_root.cache_disk_queued_bytes > 0,
                 "DFlash disk writer did not publish a compact snapshot");
        }

        gufo::server::InferenceBackend restarted_dflash;
        Expect(restarted_dflash.load(model, &error, context, 1, {}, {},
                                     dflash_config, dflash_disk_cache),
               error);
        const auto restored_dflash_continuation =
            restarted_dflash.chat(cache_request(continued_messages), 8, {});
        Expect(restored_dflash_continuation.cache_hit &&
                   restored_dflash_continuation.cache_disk_hit,
               "fresh DFlash backend did not restore its disk prefix");
        Expect(restored_dflash_continuation.cached_prompt_tokens ==
                       chat_prompt.size() &&
                   warm_dflash_continuation.cached_prompt_tokens >=
                       restored_dflash_continuation.cached_prompt_tokens &&
                   restored_dflash_continuation.cached_prompt_tokens <
                       restored_dflash_continuation.prompt_tokens,
               "DFlash disk restore did not report its exact prefix");
        Expect(restored_dflash_continuation.tokens ==
                   warm_dflash_continuation.tokens,
               "DFlash disk restore changed continuation tokens");
        Expect(restored_dflash_continuation.draft_tokens ==
                       warm_dflash_continuation.draft_tokens &&
                   restored_dflash_continuation.draft_accepted_tokens ==
                       warm_dflash_continuation.draft_accepted_tokens,
               "DFlash disk restore changed the speculative trajectory");

        auto incompatible_dflash_cache = dflash_disk_cache;
        incompatible_dflash_cache.draft_model_artifact_fingerprint =
            std::string(64, 'd');
        gufo::server::InferenceBackend incompatible_dflash;
        Expect(
            incompatible_dflash.load(model, &error, context, 1, {}, {},
                                     dflash_config, incompatible_dflash_cache),
            error);
        const auto incompatible_dflash_result =
            incompatible_dflash.chat(cache_request(continued_messages), 8, {});
        Expect(!incompatible_dflash_result.cache_hit &&
                   !incompatible_dflash_result.cache_disk_hit,
               "changed DFlash artifact fingerprint must be a cold miss");
        Expect(incompatible_dflash_result.tokens == direct_dflash_continuation,
               "DFlash compatibility miss changed cold execution");
      }
    }

    // Prime exactly one live frontier. Earlier repeated requests can leave
    // two equivalent live sessions, legitimately avoiding both restore copies.
    Expect(backend.load(model, &error, context, 2), error);
    Expect(backend.chat(cache_request(messages), 2, {}).tokens == direct_chat,
           "fork fixture must start at the qualified root");
    auto continuation_request = cache_request(continued_messages);
    continuation_request.client_id = "qwen-snapshot-branch-a";
    auto fork_request = cache_request(forked_messages);
    fork_request.client_id = "qwen-snapshot-branch-b";
    auto pending_continuation =
        backend.start_chat(continuation_request, 2, 0.0F);
    auto pending_fork = backend.start_chat(fork_request, 2, 0.0F);
    Expect(pending_continuation != nullptr && pending_fork != nullptr,
           "concurrent Qwen snapshot branches are admitted");
    const auto http_continuation = pending_continuation->Wait();
    const auto http_fork = pending_fork->Wait();

    const auto root_tokens = chat_prompt.size();
    for (const auto* result : {&http_continuation, &http_fork}) {
      Expect(result->cache_hit,
             "concurrent Qwen branch must restore the retained root");
      Expect(result->cached_prompt_tokens >= root_tokens,
             "Qwen branches retain at least the shared prompt frontier");
      Expect(result->cached_prompt_tokens < result->prompt_tokens,
             "Qwen branches prefill only their suffix");
      Expect(result->cache_snapshot_bytes > 0,
             "Qwen branches account their new snapshot bytes");
      Expect((result->cache_restore_bytes == 0 ||
              result->cache_restore_ms > 0.0) &&
                 result->cache_snapshot_ms > 0.0,
             "Qwen branches account performed snapshot copies");
      Expect(result->cache_shared_bytes == 0,
             "full-copy Qwen snapshots do not claim shared bytes");
    }
    Expect(http_continuation.cache_restore_bytes > 0 ||
               http_fork.cache_restore_bytes > 0,
           "one Qwen branch must restore its immutable prompt snapshot");
    Expect(http_continuation.tokens == direct_continuation,
           "cached Qwen continuation differs from cold full prefill");
    Expect(http_fork.tokens == direct_fork,
           "forked Qwen continuation differs from cold full prefill");

    // Both prompts must generate enough tokens to exercise an actual decode
    // batch. The old "complete this phrase" fixtures ended after one token.
    // Small chunks also cover bounded prefill before the concurrent decode.
    Expect(backend.load(model, &error, context, 2, {.decode_active_tokens = 8}),
           error);
    gufo::server::ChatRequest concurrent_a({
        {gufo::tokenization::ChatRole::kUser,
         "The quick brown fox jumps over the lazy dog. Explain why this "
         "sentence is commonly used.",
         "", ""},
    });
    concurrent_a.client_id = "batch-a";
    gufo::server::ChatRequest concurrent_b({
        {gufo::tokenization::ChatRole::kUser,
         "Write a concise C++20 implementation of a fixed-capacity ring buffer "
         "with push, pop, front, and size. Explain the invariants.",
         "", ""},
    });
    concurrent_b.client_id = "batch-b";
    const auto rendered_a =
        gufo::tokenization::QwenChatTemplate::Render(concurrent_a.messages);
    const auto rendered_b =
        gufo::tokenization::QwenChatTemplate::Render(concurrent_b.messages);
    Expect(rendered_a.has_value() && rendered_b.has_value(),
           "concurrent chat prompt rendering");
    const auto direct_a =
        GenerateDirect(*direct, model->GetTokenizer().Encode(*rendered_a), 8);
    const auto direct_b =
        GenerateDirect(*direct, model->GetTokenizer().Encode(*rendered_b), 8);
    Expect(direct_a.size() == 8 && direct_b.size() == 8,
           "concurrency fixtures must retain eight generated tokens");

    auto pending_a = backend.start_chat(concurrent_a, 8, 0.0F);
    auto pending_b = backend.start_chat(concurrent_b, 8, 0.0F);
    const auto concurrent_result_a = pending_a->Wait();
    const auto concurrent_result_b = pending_b->Wait();
    Expect(concurrent_result_a.tokens == direct_a,
           "concurrent Qwen request A differs from isolated execution");
    Expect(concurrent_result_b.tokens == direct_b,
           "concurrent Qwen request B differs from isolated execution");
    // The first cold prefill uses its optimal geometry. Work overlapping a
    // decoder or its pending snapshot yields; it can grow after that peer ends.
    const auto snapshot_bounded = [](const auto& result) {
      return result.prefill_chunks > 1 && result.max_prefill_chunk_tokens <= 8;
    };
    Expect(concurrent_result_a.active_decode_prefill_chunks +
                       concurrent_result_b.active_decode_prefill_chunks >
                   0 ||
               snapshot_bounded(concurrent_result_a) ||
               snapshot_bounded(concurrent_result_b),
           "concurrent Qwen prefill must yield to decode or snapshot work");
    std::cout << "concurrency tokens=" << concurrent_result_a.tokens.size()
              << ',' << concurrent_result_b.tokens.size()
              << " widths=" << concurrent_result_a.physical_execution_width
              << ',' << concurrent_result_b.physical_execution_width
              << " plans=" << concurrent_result_a.execution_plan << ','
              << concurrent_result_b.execution_plan
              << " queue_ms=" << concurrent_result_a.queue_ms << ','
              << concurrent_result_b.queue_ms << '\n';
    Expect(concurrent_result_a.physical_execution_width == 2 &&
               concurrent_result_b.physical_execution_width == 2,
           "concurrent Qwen requests did not execute through W=2");
    Expect(concurrent_result_a.execution_plan == "batched-w2" &&
               concurrent_result_b.execution_plan == "batched-w2",
           "concurrent Qwen requests did not report the W=2 plan");

    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    HIP_CHECK(hipMemGetInfo(&free_before, &total_memory));

    std::size_t cancel_checks = 0;
    const auto cancelled =
        backend.complete(raw_prompt, 4, 0.0F, [&cancel_checks] {
          ++cancel_checks;
          return cancel_checks > 2;
        });
    Expect(cancelled.cancelled, "request cancellation must be reported");
    Expect(cancelled.completion_tokens <= 1,
           "cancelled request emitted more than one token");

    bool callback_threw = false;
    std::size_t error_checks = 0;
    try {
      (void)backend.complete(raw_prompt, 4, 0.0F, [&error_checks] {
        ++error_checks;
        if (error_checks > 2) {
          throw std::runtime_error("injected request failure");
        }
        return false;
      });
    } catch (const std::runtime_error& exception) {
      callback_threw =
          std::string_view(exception.what()) == "injected request failure";
    }
    Expect(callback_threw, "injected request error must propagate");

    const auto recovered = backend.complete(raw_prompt, 2, 0.0F);
    Expect(recovered.tokens == direct_raw,
           "session was not reusable after cancellation and error");
    // A failed mutable frontier cannot be reused, but its immutable prompt
    // snapshot remains valid. Exact recovered tokens above qualify either path.

    HIP_CHECK(hipDeviceSynchronize());
    std::size_t free_after = 0;
    HIP_CHECK(hipMemGetInfo(&free_after, &total_memory));
    ExpectStableGpuMemory(free_before, free_after);

    std::cout << "HTTP HIP inference parity and cleanup tests passed.\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
