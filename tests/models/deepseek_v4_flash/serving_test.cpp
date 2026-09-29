#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

using Model = gufo::models::deepseek_v4_flash::Model;
using ModelOptions = gufo::models::deepseek_v4_flash::ModelOptions;
using gufo::models::deepseek_v4_flash::ResolveDeepSeekChatOptions;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<gufo::tokenization::TokenId> GenerateDirect(
    const std::shared_ptr<Model>& model, std::span<const int> prompt,
    std::size_t max_tokens, const gufo::sampling::SamplingConfig& sampling = {},
    bool dspark = false) {
  std::string error;
  auto session =
      model->CreateSession(dspark ? gufo::core::SessionMode::kSpeculative
                                  : gufo::core::SessionMode::kAutoregressive,
                           dspark ? 4096 : 512, &error);
  Expect(session != nullptr, error);
  Expect(session->Sync(prompt, &error), error);
  const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                     prompt.end());
  gufo::sampling::SamplerState sampler(sampling, history);

  std::vector<gufo::tokenization::TokenId> result;
  result.reserve(max_tokens);
  if (dspark) {
    while (result.size() < max_tokens) {
      gufo::models::deepseek_v4_flash::DsparkSamplerBridge bridge(sampler);
      std::vector<int> emitted;
      Expect(session->DsparkStep(max_tokens - result.size(), 7, &emitted,
                                 &error, bridge.hook()),
             error);
      Expect(!emitted.empty(), "direct DSpark progress");
      sampler.SetRngState(bridge.rng_state());
      for (const int token : emitted) {
        if (model->IsStopToken(token))
          return result;
        sampler.Accept(token);
        result.push_back(token);
      }
    }
    return result;
  }
  for (std::size_t index = 0; index < max_tokens; ++index) {
    int token = -1;
    if (sampling.can_use_unmodified_argmax()) {
      token = session->SelectNext(0.0F, nullptr);
    } else {
      const auto logits = session->CopyLogits(&error);
      Expect(!logits.empty(), error);
      token = static_cast<int>(sampler.Sample(logits));
    }
    Expect(token >= 0, "direct token selection");
    if (model->IsStopToken(token)) {
      break;
    }
    sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
    result.push_back(static_cast<gufo::tokenization::TokenId>(token));
    if (index + 1 < max_tokens) {
      Expect(session->Evaluate(token, &error), error);
    }
  }
  return result;
}

void CheckSampledBatchReplay(const std::shared_ptr<Model>& model,
                             std::span<const int> prompt) {
  using namespace gufo::models::deepseek_v4_flash;
  // A fixed dispatch schedule separates sampler reproducibility from HTTP
  // arrival order. Different seeds/filters exercise independent request state.
  std::array<std::vector<int>, 2> reference;
  std::array<std::uint64_t, 2> reference_drafted{}, reference_accepted{};
  std::size_t probabilistic_draws = 0, verifications = 0;
  struct Probe {
    DsparkSamplerBridge* bridge;
    std::size_t *draws, *verified;
  };
  for (int run = 0; run < 2; ++run) {
    std::string error;
    std::array<std::unique_ptr<Session>, 2> sessions;
    std::array<gufo::sampling::SamplerState, 2> samplers{
        gufo::sampling::SamplerState(
            {.temperature = 1.F, .top_p = 0.95F, .seed = 7}),
        gufo::sampling::SamplerState({.temperature = 0.8F,
                                      .top_k = 40,
                                      .top_p = 0.95F,
                                      .seed = 11,
                                      .repeat_penalty = 1.1F})};
    const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                       prompt.end());
    std::array<std::vector<int>, 2> output;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      sessions[i] = model->CreateSession(
          model->HasDspark() ? gufo::core::SessionMode::kSpeculative
                             : gufo::core::SessionMode::kAutoregressive,
          4096, &error);
      Expect(sessions[i] && sessions[i]->Sync(prompt, &error), error);
      samplers[i].ResetHistory(history);
    }
    for (int cycle = 0; cycle < 16; ++cycle) {
      std::array<std::optional<DsparkSamplerBridge>, 2> bridges;
      std::array<Probe, 2> probes;
      std::array<ds4_dspark_sampler, 2> hooks;
      std::array<std::vector<int>, 2> emitted;
      std::array<SessionDsparkBatchItem, 2> items;
      for (std::size_t i = 0; i < sessions.size(); ++i) {
        bridges[i].emplace(samplers[i]);
        probes[i] = {&*bridges[i], &probabilistic_draws, &verifications};
        hooks[i] = {
            .ctx = &probes[i],
            .sample =
                [](void* ctx, const float* logits, uint32_t vocab) {
                  const auto* hook = static_cast<Probe*>(ctx)->bridge->hook();
                  return hook->sample(hook->ctx, logits, vocab);
                },
            .accept =
                [](void* ctx, int token) {
                  const auto* hook = static_cast<Probe*>(ctx)->bridge->hook();
                  hook->accept(hook->ctx, token);
                },
            .propose =
                [](void* ctx, uint32_t row,
                   const ds4_dspark_candidates* candidates) {
                  auto& probe = *static_cast<Probe*>(ctx);
                  const auto* hook = probe.bridge->hook();
                  const auto rng = probe.bridge->rng_state();
                  const int token = hook->propose(hook->ctx, row, candidates);
                  *probe.draws += rng != probe.bridge->rng_state();
                  return token;
                },
            .verify =
                [](void* ctx, uint32_t row, const float* logits, uint32_t vocab,
                   int token) {
                  auto& probe = *static_cast<Probe*>(ctx);
                  ++*probe.verified;
                  const auto* hook = probe.bridge->hook();
                  return hook->verify(hook->ctx, row, logits, vocab, token);
                }};
        items[i] = {.session = sessions[i].get(),
                    .max_tokens = 4,
                    .max_draft_tokens = 3,
                    .emitted = &emitted[i],
                    .sampler = &hooks[i]};
      }
      Expect(model->DsparkStepBatch(items, &error), error);
      for (std::size_t i = 0; i < sessions.size(); ++i) {
        Expect(!emitted[i].empty(), "sampled batch makes progress");
        samplers[i].SetRngState(bridges[i]->rng_state());
        for (const int token : emitted[i])
          samplers[i].Accept(token);
        output[i].insert(output[i].end(), emitted[i].begin(), emitted[i].end());
      }
    }
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      const auto stats = sessions[i]->DsparkStatistics();
      if (run == 0) {
        reference[i] = output[i];
        reference_drafted[i] = stats.support_drafted;
        reference_accepted[i] = stats.support_accepted;
      } else {
        Expect(output[i] == reference[i] &&
                   stats.support_drafted == reference_drafted[i] &&
                   stats.support_accepted == reference_accepted[i],
               "sampled batch replays outputs and decisions independently");
      }
    }
  }
  Expect(reference_drafted[0] + reference_drafted[1] > 0,
         "filtered sampled batch exercises DSpark");
  Expect(probabilistic_draws > 0 && verifications > 0,
         "concurrent runtime invokes stochastic proposal and verification");
}

void CheckDsparkEosBoundary(const std::shared_ptr<Model>& model) {
  using namespace gufo::models::deepseek_v4_flash;
  const auto prompt = model->Tokenize("Continue the pattern: red, blue, red,");
  struct Draw {
    int token;
    int calls = 0;
  };
  std::array<Draw, 3> draws{
      {{model->EosToken()}, {prompt.back()}, {prompt.back()}}};
  std::array<ds4_dspark_sampler, 3> hooks{};
  std::array<std::unique_ptr<Session>, 3> sessions;
  std::array<std::vector<int>, 3> emitted;
  std::array<SessionDsparkBatchItem, 3> items{};
  std::string error;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    sessions[i] = model->CreateSession(
        model->HasDspark() ? gufo::core::SessionMode::kSpeculative
                           : gufo::core::SessionMode::kAutoregressive,
        4096, &error);
    Expect(sessions[i] && sessions[i]->Sync(prompt, &error), error);
    hooks[i] = {.ctx = &draws[i],
                .sample = [](void* ctx, const float*, uint32_t) {
                  auto& draw = *static_cast<Draw*>(ctx);
                  ++draw.calls;
                  return draw.token;
                }};
    items[i] = {.session = sessions[i].get(),
                .max_tokens = 1,
                .max_draft_tokens = 1,
                .emitted = &emitted[i],
                .sampler = &hooks[i],
                .stop_at_eos = true};
  }
  const auto eos_logits = sessions[0]->CopyLogits(&error);
  Expect(!eos_logits.empty(), error);
  // Exercise no survivors, one survivor and a compacted GPU batch.
  for (std::size_t width = 1; width <= items.size(); ++width) {
    std::array<std::size_t, 3> positions{};
    for (std::size_t i = 0; i < width; ++i) {
      positions[i] = sessions[i]->Position();
      draws[i].calls = 0;
    }
    Expect(model->DsparkStepBatch(std::span(items).first(width), &error),
           error);
    for (std::size_t i = 0; i < width; ++i) {
      Expect(
          emitted[i] == std::vector<int>{draws[i].token} && draws[i].calls == 1,
          "EOS compaction emits each request's draw without resampling");
      Expect(sessions[i]->Position() == positions[i] + (i != 0),
             "only non-EOS anchors enter the request checkpoint");
    }
    Expect(sessions[0]->CopyLogits(&error) == eos_logits,
           "EOS keeps the preceding frontier logits");
  }
  std::array<std::size_t, 3> positions{};
  for (std::size_t i = 0; i < items.size(); ++i) {
    positions[i] = sessions[i]->Position();
    draws[i] = {model->EosToken()};
  }
  Expect(model->DsparkStepBatch(items, &error), error);
  for (std::size_t i = 0; i < items.size(); ++i) {
    Expect(sessions[i]->Position() == positions[i] && draws[i].calls == 1 &&
               emitted[i] == std::vector<int>{model->EosToken()},
           "an all-EOS cohort completes without committing tokens");
  }
  Expect(sessions[0]->DsparkStep(1, 1, &emitted[0], &error, &hooks[0]), error);
  Expect(sessions[0]->Position() == prompt.size() + 1 &&
             emitted[0] == std::vector<int>{model->EosToken()},
         "fixed-length decoding still consumes EOS as an ordinary token");
}

void CheckExecutionModes(const std::shared_ptr<Model>& model) {
  using gufo::core::SessionMode;
  std::string error;
  auto ar = model->CreateSession(SessionMode::kAutoregressive, 512, &error);
  auto spec = model->CreateSession(SessionMode::kSpeculative, 512, &error);
  Expect(ar && spec, error);
  Expect(!ar->DsparkEnabled() && spec->DsparkEnabled(),
         "session mode reports model capability instead of enabled execution");
  const std::string text = "The capital of France is";
  const auto prompt = model->Tokenize(text);
  Expect(ar->Sync(prompt, &error) && spec->Sync(prompt, &error), error);
  Expect(ar->CopyLogits(&error) == spec->CopyLogits(&error),
         "DSpark sidecar changes AR logits");
  Expect(ar->DsparkStatistics().context_tokens == 0 &&
             spec->DsparkStatistics().context_tokens == prompt.size(),
         "AR session maintains DSpark state");
  auto saved_ar = ar->SaveSnapshot(&error);
  auto saved_spec = spec->SaveSnapshot(&error);
  Expect(
      saved_ar && saved_spec && saved_ar->SizeBytes() < saved_spec->SizeBytes(),
      "AR snapshot includes predictor state");
  gufo::server::InferenceBackend backend;
  Expect(backend.load(model, &error, 512, 1), error);
  const auto actual = backend.complete(text, 8, 0.0F);
  Expect(actual.draft_tokens == 0 &&
             actual.tokens == GenerateDirect(model, prompt, 8),
         "disabled backend executed DSpark with a resident sidecar");
  Expect(!ar->RestoreSnapshot(*saved_spec, &error) &&
             !spec->RestoreSnapshot(*saved_ar, &error),
         "DeepSeek snapshots crossed execution modes");
  std::cout << "DSpark execution modes and snapshot isolation passed\n";
}

void CheckDsparkServing(const char* model_path, const char* support_path) {
  using namespace gufo::server;
  std::string error;
  auto model = Model::Load(
      model_path,
      ModelOptions{.max_context = 262144, .dspark_model_path = support_path},
      &error);
  Expect(model != nullptr, error);
  CheckExecutionModes(model);
  CheckDsparkEosBoundary(model);
  const TextSpeculativeConfig speculative{
      .backend = TextSpeculativeBackend::kDSpark,
      .draft_model_path = support_path};
  const ChatRequest prompt({{gufo::tokenization::ChatRole::kUser,
                             "Continue the pattern with twenty terms: red, "
                             "blue, blue, red, blue, blue,",
                             "", ""}});
  {
    InferenceBackend backend;
    Expect(backend.load(model, &error, 262144, 1, {}, {}, speculative), error);
    const auto cold = backend.chat(prompt, 32, 0.0F);
    const auto warm = backend.chat(prompt, 32, 0.0F);
    Expect(cold.draft_tokens > 0 && warm.cache_hit && warm.prefill_tokens == 0,
           "DSpark reuses the complete prompt and drafts immediately");
    Expect(cold.tokens == warm.tokens &&
               cold.draft_tokens == warm.draft_tokens &&
               cold.draft_accepted_tokens == warm.draft_accepted_tokens,
           "DSpark prefix reuse resets policy and reproduces output and draft "
           "decisions");
  }

  // C1 retains the point-mass route and its stronger AR seed-trace contract.
  {
    const std::vector<gufo::models::deepseek_v4_flash::ChatMessage>
        direct_messages = {
            {.role = "user",
             .content = "Continue the pattern with twenty terms: red, blue, "
                        "blue, red, blue, blue,"},
        };
    const auto direct_prompt =
        model->EncodeChat(direct_messages, ResolveDeepSeekChatOptions({}));
    InferenceBackend backend;
    Expect(backend.load(model, &error, 4096, 1, {}, {}, speculative), error);
    const auto greedy = backend.chat(prompt, 32, 0.0F);
    std::size_t sampled_drafts = 0;
    for (const auto& sampling : {
             gufo::sampling::SamplingConfig{.temperature = 0.6F, .seed = 7},
             gufo::sampling::SamplingConfig{.temperature = 1.0F, .seed = 7},
             gufo::sampling::SamplingConfig{.temperature = 0.6F,
                                            .top_k = 40,
                                            .top_p = 0.95F,
                                            .seed = 11,
                                            .repeat_penalty = 1.2F,
                                            .frequency_penalty = 0.3F},
         }) {
      const auto direct =
          GenerateDirect(model, direct_prompt, 32, sampling, true);
      const auto autoregressive =
          GenerateDirect(model, direct_prompt, 32, sampling);
      const auto result = backend.chat(prompt, 32, sampling);
      const auto repeat = backend.chat(prompt, 32, sampling);
      sampled_drafts += result.draft_tokens;
      Expect(result.draft_accepted_tokens <= result.draft_tokens,
             "sampled DSpark accounting includes legitimate AR fallback");
      Expect(result.tokens == direct && direct == autoregressive &&
                 repeat.tokens == result.tokens,
             "C1 DSpark, AR, frontend and cached seeded replay agree");
    }
    Expect(sampled_drafts > 0,
           "sampled C1 serving exercises DSpark verification");
    const auto greedy_again = backend.chat(prompt, 32, 0.0F);
    Expect(
        greedy_again.tokens == greedy.tokens &&
            greedy_again.draft_tokens == greedy.draft_tokens &&
            greedy_again.draft_accepted_tokens == greedy.draft_accepted_tokens,
        "greedy DSpark output is unchanged after sampled cycles");
    CheckSampledBatchReplay(model, direct_prompt);
  }

  // Greedy and seeded sampled requests share one DSpark cohort.
  const gufo::sampling::SamplingConfig sampled{.temperature = 0.8F,
                                               .top_k = 20,
                                               .top_p = 0.9F,
                                               .seed = 1234,
                                               .repeat_penalty = 1.1F};
  for (const std::size_t concurrency : {2U, 4U}) {
    InferenceBackend backend;
    Expect(backend.load(model, &error, 4096, concurrency, {}, {}, speculative),
           error);
    std::vector<std::shared_ptr<TextGenerationBackend::GenerationRequest>>
        pending;
    for (std::size_t i = 0; i < concurrency; ++i) {
      auto request = prompt;
      request.client_id = "mixed-" + std::to_string(i);
      pending.push_back(backend.start_chat(
          request, 32,
          i % 2 == 0 ? gufo::sampling::SamplingConfig{.temperature = 0.0F}
                     : sampled,
          {}, true));
      Expect(pending.back() != nullptr, "mixed DSpark request admission");
    }
    for (std::size_t i = 0; i < pending.size(); ++i) {
      const auto result = pending[i]->Wait();
      Expect(!result.tokens.empty() && result.tokens.size() <= 32,
             "mixed sampling request completes within its budget");
      if (i % 2 == 0)
        Expect(result.draft_tokens > 0, "greedy requests retain DSpark");
      Expect(result.physical_execution_width == concurrency &&
                 result.execution_plan ==
                     "batched-w" + std::to_string(concurrency),
             "mixed sampling reports one DSpark cohort");
    }
  }
  {
    InferenceBackend backend;
    Expect(backend.load(model, &error, 4096, 2, {.decode_active_tokens = 32},
                        {}, speculative),
           error);
    auto active = backend.start_chat(prompt, 64, 0.0F, {}, true);
    std::shared_ptr<TextGenerationBackend::GenerationRequest> late;
    auto long_request = prompt;
    long_request.client_id = "late-long-prompt";
    for (int i = 0; i < 256; ++i)
      long_request.messages.front().content += " additional context";
    const auto first = active->Wait([&](std::string_view) {
      if (!late)
        late = backend.start_chat(long_request, 8, 0.0F, {}, true);
      return true;
    });
    Expect(late != nullptr && !first.tokens.empty(),
           "late prompt arrives during decoding");
    const auto second = late->Wait();
    Expect(second.active_decode_prefill_chunks > 0 &&
               second.max_consecutive_active_prefill_chunks == 1,
           "real DSpark serving yields to decoding between long-prompt chunks");
  }

  {
    InferenceBackend backend;
    Expect(backend.load(model, &error, 64, 2, {}, {}, speculative), error);
    auto greedy = backend.start_chat(prompt, 128, 0.0F, {}, true);
    auto sampling = backend.start_chat(prompt, 128, sampled, {}, true);
    for (const auto& result : {greedy->Wait(), sampling->Wait()}) {
      Expect(result.prompt_tokens + result.completion_tokens == 64,
             "greedy and sampled serving stop exactly at context capacity");
    }
  }

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("ds4-support-cache-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{directory};
  TextDiskCacheConfig disk{
      .directory = directory,
      .model_artifact_fingerprint = std::string(64, 'a'),
      .draft_model_artifact_fingerprint = std::string(64, 'b')};
  std::vector<gufo::tokenization::TokenId> expected;
  for (int run = 0; run < 3; ++run) {
    if (run == 2)
      disk.draft_model_artifact_fingerprint = std::string(64, 'c');
    InferenceBackend backend;
    Expect(backend.load(model, &error, 4096, 1, {}, {}, speculative, disk),
           error);
    const auto result = backend.chat(prompt, 8, sampled);
    Expect(result.cache_disk_hit == (run == 1),
           "DSpark disk reuse requires the matching support artifact identity");
    if (run == 0)
      expected = result.tokens;
    else
      Expect(result.tokens == expected,
             "DSpark disk restore retains output quality");
  }
  std::cout << "DSpark serving: prefix/disk reuse, mixed sampling, and late "
               "prefill passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const char* model_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
      std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_MODEL is not set\n";
      return 77;
    }
    if (argc > 1) {
      Expect(argc == 2 && std::string_view(argv[1]) == "--dspark-eos",
             "usage: ds4_serving_test [--dspark-eos]");
      const char* support = std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
      if (support == nullptr || *support == '\0') {
        std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL is not set\n";
        return 77;
      }
      std::string error;
      auto model = Model::Load(
          model_path,
          ModelOptions{.max_context = 4096, .dspark_model_path = support},
          &error);
      Expect(model != nullptr, error);
      CheckExecutionModes(model);
      CheckDsparkEosBoundary(model);
      std::cout << "DSpark EOS checkpoint checks passed\n";
      return 0;
    }

    {
      std::string error;
      auto model = Model::Load(model_path,
                               ModelOptions{
                                   .max_context = 512,
                               },
                               &error);
      Expect(model != nullptr, error);

      gufo::server::InferenceBackend backend;
      Expect(backend.load(model, &error, 512, 2), error);
      Expect(backend.model_id() == model->ModelName(), "HTTP model identifier");

      const std::string raw_prompt = "The capital of France is";
      const auto direct_raw =
          GenerateDirect(model, model->Tokenize(raw_prompt), 2);
      const auto http_raw = backend.complete(raw_prompt, 2, 0.0F);
      Expect(http_raw.tokens == direct_raw, "raw direct/HTTP token parity");
      Expect(http_raw.ttft_ms > 0.0, "raw TTFT");

      const gufo::sampling::SamplingConfig sampled{
          .temperature = 0.8F,
          .top_k = 20,
          .top_p = 0.9F,
          .seed = 1234,
          .repeat_penalty = 1.1F,
      };
      const auto direct_sampled =
          GenerateDirect(model, model->Tokenize(raw_prompt), 8, sampled);
      const auto sampled_first = backend.complete(raw_prompt, 8, sampled);
      const auto sampled_repeat = backend.complete(raw_prompt, 8, sampled);
      Expect(sampled_first.tokens == direct_sampled &&
                 sampled_repeat.tokens == direct_sampled,
             "seeded sampling and penalties match direct execution and repeat");

      const std::vector<gufo::tokenization::ChatMessage> messages = {
          {gufo::tokenization::ChatRole::kSystem,
           "Answer with one short sentence.", "", ""},
          {gufo::tokenization::ChatRole::kUser, "Name one primary color.", "",
           ""},
      };
      const std::vector<gufo::models::deepseek_v4_flash::ChatMessage>
          direct_messages = {
              {.role = "system", .content = "Answer with one short sentence."},
              {.role = "user", .content = "Name one primary color."},
          };
      const auto direct_chat = GenerateDirect(
          model,
          model->EncodeChat(direct_messages, ResolveDeepSeekChatOptions({})),
          2);
      const auto http_chat = backend.chat(messages, 2, {});
      Expect(http_chat.tokens == direct_chat, "chat direct/HTTP token parity");
      Expect(!http_chat.cache_hit, "first chat request is a cache miss");
      const auto repeated_chat = backend.chat(messages, 2, {});
      Expect(repeated_chat.cache_hit &&
                 repeated_chat.cached_prompt_tokens ==
                     repeated_chat.prompt_tokens &&
                 repeated_chat.prefill_tokens == 0,
             "repeated DeepSeek chat restores the complete prompt boundary");
      Expect(repeated_chat.tokens == direct_chat,
             "repeated DeepSeek chat differs from cold target execution");
      Expect(repeated_chat.cache_snapshot_bytes == 0,
             "exact DeepSeek reuse avoids another full snapshot copy");

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
      auto continued_direct_messages = direct_messages;
      continued_direct_messages.push_back(
          {.role = "assistant", .content = http_chat.text});
      continued_direct_messages.push_back(
          {.role = "user", .content = "Name a different primary color."});
      auto forked_direct_messages = direct_messages;
      forked_direct_messages.push_back(
          {.role = "assistant", .content = http_chat.text});
      forked_direct_messages.push_back(
          {.role = "user", .content = "Name one warm primary color."});
      const auto direct_continuation =
          GenerateDirect(model,
                         model->EncodeChat(continued_direct_messages,
                                           ResolveDeepSeekChatOptions({})),
                         2);
      const auto direct_fork =
          GenerateDirect(model,
                         model->EncodeChat(forked_direct_messages,
                                           ResolveDeepSeekChatOptions({})),
                         2);

      gufo::server::ChatRequest continuation_request(continued_messages);
      continuation_request.client_id = "deepseek-snapshot-branch-a";
      gufo::server::ChatRequest fork_request(forked_messages);
      fork_request.client_id = "deepseek-snapshot-branch-b";
      auto pending_continuation =
          backend.start_chat(continuation_request, 2, 0.0F, {}, true);
      auto pending_fork = backend.start_chat(fork_request, 2, 0.0F, {}, true);
      Expect(pending_continuation != nullptr && pending_fork != nullptr,
             "concurrent DeepSeek snapshot branches are admitted");
      const auto http_continuation = pending_continuation->Wait();
      const auto http_fork = pending_fork->Wait();

      Expect(http_continuation.cache_hit && http_fork.cache_hit,
             "concurrent DeepSeek branches restore the retained root");
      Expect(http_continuation.cached_prompt_tokens > 0 &&
                 http_fork.cached_prompt_tokens > 0,
             "DeepSeek branches reuse their longest available prefix");
      Expect(http_continuation.cached_prompt_tokens <
                     http_continuation.prompt_tokens &&
                 http_fork.cached_prompt_tokens < http_fork.prompt_tokens,
             "DeepSeek branches prefill only their suffixes");
      for (const auto* result : {&http_continuation, &http_fork}) {
        Expect(
            result->cache_snapshot_bytes > 0 && result->cache_snapshot_ms > 0.0,
            "DeepSeek branches retain their new prompt snapshot");
        Expect((result->cache_restore_bytes == 0) ==
                   (result->cache_restore_ms == 0.0),
               "live reuse reports no restore copy; snapshot reuse reports it");
        Expect(result->cache_shared_bytes == 0,
               "full-copy DeepSeek snapshots do not claim shared bytes");
      }
      Expect(http_continuation.tokens == direct_continuation,
             "cached DeepSeek continuation differs from cold full prefill");
      Expect(http_fork.tokens == direct_fork,
             "forked DeepSeek continuation differs from cold full prefill");

      gufo::server::ChatRequest concurrent_a({
          {gufo::tokenization::ChatRole::kUser,
           "Continue this sequence with four short items: one, two, three,", "",
           ""},
      });
      concurrent_a.client_id = "deepseek-a";
      gufo::server::ChatRequest concurrent_b({
          {gufo::tokenization::ChatRole::kUser,
           "Continue this sequence with four short items: red, green, blue,",
           "", ""},
      });
      concurrent_b.client_id = "deepseek-b";
      const std::vector<gufo::models::deepseek_v4_flash::ChatMessage>
          direct_a_messages = {
              {.role = "user",
               .content =
                   "Continue this sequence with four short items: one, two, "
                   "three,"},
          };
      const std::vector<gufo::models::deepseek_v4_flash::ChatMessage>
          direct_b_messages = {
              {.role = "user",
               .content =
                   "Continue this sequence with four short items: red, green, "
                   "blue,"},
          };
      const auto direct_a = GenerateDirect(
          model,
          model->EncodeChat(direct_a_messages, ResolveDeepSeekChatOptions({})),
          4);
      const auto direct_b = GenerateDirect(
          model,
          model->EncodeChat(direct_b_messages, ResolveDeepSeekChatOptions({})),
          4);

      auto pending_a = backend.start_chat(concurrent_a, 4, 0.0F, {}, true);
      auto pending_b = backend.start_chat(concurrent_b, 4, 0.0F, {}, true);
      Expect(pending_a != nullptr && pending_b != nullptr,
             "concurrent DeepSeek requests are admitted");

      std::mutex event_mutex;
      std::vector<char> events;
      gufo::server::InferenceBackend::Result concurrent_result_a;
      gufo::server::InferenceBackend::Result concurrent_result_b;
      std::exception_ptr failure_a;
      std::exception_ptr failure_b;
      std::jthread waiter_a([&] {
        try {
          concurrent_result_a = pending_a->Wait([&](std::string_view) {
            const std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back('a');
            return true;
          });
        } catch (...) {
          failure_a = std::current_exception();
        }
      });
      std::jthread waiter_b([&] {
        try {
          concurrent_result_b = pending_b->Wait([&](std::string_view) {
            const std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back('b');
            return true;
          });
        } catch (...) {
          failure_b = std::current_exception();
        }
      });
      waiter_a.join();
      waiter_b.join();
      if (failure_a != nullptr) {
        std::rethrow_exception(failure_a);
      }
      if (failure_b != nullptr) {
        std::rethrow_exception(failure_b);
      }

      Expect(concurrent_result_a.tokens == direct_a,
             "concurrent DeepSeek request A differs from isolated execution");
      Expect(concurrent_result_b.tokens == direct_b,
             "concurrent DeepSeek request B differs from isolated execution");
      Expect(concurrent_result_a.requested_logical_concurrency == 2 &&
                 concurrent_result_b.requested_logical_concurrency == 2,
             "DeepSeek scheduler reports two logical sessions");
      Expect(concurrent_result_a.physical_execution_width == 2 &&
                 concurrent_result_b.physical_execution_width == 2 &&
                 concurrent_result_a.execution_plan == "batched-w2" &&
                 concurrent_result_b.execution_plan == "batched-w2",
             "DeepSeek reports native two-session execution");
      const auto first_a = std::find(events.begin(), events.end(), 'a');
      const auto first_b = std::find(events.begin(), events.end(), 'b');
      const auto last_a = std::find(events.rbegin(), events.rend(), 'a').base();
      const auto last_b = std::find(events.rbegin(), events.rend(), 'b').base();
      Expect(first_a != events.end() && first_b != events.end(),
             "both DeepSeek sessions stream output");
      Expect(first_b < last_a && first_a < last_b,
             "both DeepSeek sessions make progress before the other completes");

      std::size_t cancellation_checks = 0;
      const auto cancelled =
          backend.complete(raw_prompt, 4, 0.0F, [&cancellation_checks] {
            ++cancellation_checks;
            return cancellation_checks > 2;
          });
      Expect(cancelled.cancelled, "cancellation status");
      Expect(cancelled.completion_tokens <= 1, "cancellation token bound");

      const auto recovered = backend.complete(raw_prompt, 2, 0.0F);
      Expect(recovered.tokens == direct_raw,
             "session reuse after cancellation");
      Expect(!recovered.cache_hit,
             "cancelled DeepSeek state must not remain cached");
    }
    const char* support_path =
        std::getenv("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
    if (support_path == nullptr || *support_path == '\0') {
      std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL is not set\n";
      return 77;
    }
    CheckDsparkServing(model_path, support_path);
    std::cout << "DeepSeek V4 Flash HTTP parity test passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
