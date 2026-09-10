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
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace {

using Model = gufo::models::deepseek_v4_flash::Model;
using ModelOptions = gufo::models::deepseek_v4_flash::ModelOptions;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<gufo::tokenization::TokenId> GenerateDirect(
    const std::shared_ptr<Model>& model, std::span<const int> prompt,
    std::size_t max_tokens,
    const gufo::sampling::SamplingConfig& sampling = {}) {
  std::string error;
  auto session = model->CreateSession(512, &error);
  Expect(session != nullptr, error);
  Expect(session->Sync(prompt, &error), error);
  const std::vector<gufo::sampling::TokenId> history(prompt.begin(),
                                                     prompt.end());
  gufo::sampling::SamplerState sampler(sampling, history);

  std::vector<gufo::tokenization::TokenId> result;
  result.reserve(max_tokens);
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

void CheckDsparkServing(const char* model_path, const char* support_path) {
  using namespace gufo::server;
  std::string error;
  auto model = Model::Load(
      model_path,
      ModelOptions{.max_context = 4096, .dspark_model_path = support_path},
      &error);
  Expect(model != nullptr, error);
  const TextSpeculativeConfig speculative{
      .backend = TextSpeculativeBackend::kDSpark,
      .draft_model_path = support_path};
  const ChatRequest prompt({{gufo::tokenization::ChatRole::kUser,
                             "Continue the pattern with twenty terms: red, "
                             "blue, blue, red, blue, blue,",
                             "", ""}});
  {
    InferenceBackend backend;
    Expect(backend.load(model, &error, 4096, 1, {}, {}, speculative), error);
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

  // Model-owned subgroups must survive mixed greedy and seeded sampling.
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
      Expect(i % 2 == 0 ? result.draft_tokens > 0 : result.draft_tokens == 0,
             "greedy requests retain DSpark and sampled requests use target "
             "sampling");
      const std::size_t expected_width = concurrency / 2;
      Expect(result.physical_execution_width == expected_width &&
                 result.execution_plan ==
                     (expected_width == 1 ? "serial-fallback" : "batched-w2"),
             "mixed sampling reports actual subgroup width");
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
    const auto result = backend.chat(prompt, 8, 0.0F);
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

int main() {
  try {
    const char* model_path = std::getenv("GUFO_DEEPSEEK_V4_FLASH_MODEL");
    if (model_path == nullptr || model_path[0] == '\0') {
      std::cout << "SKIP: GUFO_DEEPSEEK_V4_FLASH_MODEL is not set\n";
      return 77;
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
      const auto direct_chat =
          GenerateDirect(model, model->EncodeChat(direct_messages), 2);
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
      const auto direct_continuation = GenerateDirect(
          model, model->EncodeChat(continued_direct_messages), 2);
      const auto direct_fork =
          GenerateDirect(model, model->EncodeChat(forked_direct_messages), 2);

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
                 http_continuation.cached_prompt_tokens ==
                     http_fork.cached_prompt_tokens,
             "DeepSeek branches report the same shared root");
      Expect(http_continuation.cached_prompt_tokens <
                     http_continuation.prompt_tokens &&
                 http_fork.cached_prompt_tokens < http_fork.prompt_tokens,
             "DeepSeek branches prefill only their suffixes");
      for (const auto* result : {&http_continuation, &http_fork}) {
        Expect(
            result->cache_restore_bytes > 0 && result->cache_snapshot_bytes > 0,
            "DeepSeek branches account snapshot copy bytes");
        Expect(
            result->cache_restore_ms > 0.0 && result->cache_snapshot_ms > 0.0,
            "DeepSeek branches account snapshot copy time");
        Expect(result->cache_shared_bytes == 0,
               "full-copy DeepSeek snapshots do not claim shared bytes");
      }
      Expect(http_continuation.cache_restore_bytes ==
                 http_fork.cache_restore_bytes,
             "DeepSeek branches restore the same root payload");
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
      const auto direct_a =
          GenerateDirect(model, model->EncodeChat(direct_a_messages), 4);
      const auto direct_b =
          GenerateDirect(model, model->EncodeChat(direct_b_messages), 4);

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
