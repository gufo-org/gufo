#include "src/cli/serve/openai_chat.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/json.hpp"

namespace {

using namespace std::chrono_literals;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

class FakeBackend final : public gufo::server::TextGenerationBackend {
public:
  [[nodiscard]] std::string model_id() const override { return "test-model"; }
  [[nodiscard]] bool ready() const override { return true; }
  [[nodiscard]] SamplingDefaults sampling_defaults() const override {
    return defaults;
  }
  [[nodiscard]] gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning_defaults_value;
  }
  [[nodiscard]] InitialOutputState initial_output_state(
      const gufo::server::ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kReasoning
               : InitialOutputState::kContent;
  }

  Result complete(std::string_view, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck&, const TokenCallback&) override {
    return {};
  }

  Result chat(const gufo::server::ChatRequest& request, std::size_t max_tokens,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled,
              const TokenCallback& on_token) override {
    ++chat_calls;
    last_request = request;
    last_max_tokens = max_tokens;
    last_temperature = sampling.temperature;
    last_sampling = sampling;

    Result result;
    result.prompt_tokens = 7;
    result.cached_prompt_tokens = 5;
    result.prefill_tokens = 2;
    result.prefill_chunks = 1;
    result.queue_depth_at_submit = 3;
    result.client_queue_depth_at_submit = 1;
    result.resident_requests_at_admission = 2;
    result.requested_logical_concurrency = 4;
    result.physical_execution_width = 2;
    result.queue_ms = 1.25;
    result.prefill_ms = 2.5;
    result.decode_ms = 4.0;
    result.ttft_ms = 3.75;
    result.mean_inter_token_ms = 2.0;
    result.max_inter_token_ms = 2.5;
    result.execution_plan = "serial-fallback";
    result.cache_hit = true;
    for (const std::string& piece : pieces) {
      if ((is_cancelled && is_cancelled()) || (on_token && !on_token(piece))) {
        result.cancelled = true;
        result.finish_reason = FinishReason::kCancelled;
        return result;
      }
      result.text += piece;
      result.tokens.push_back(
          static_cast<gufo::tokenization::TokenId>(result.tokens.size()));
      if (block_after_first_piece &&
          result.tokens.size() == static_cast<std::size_t>(1)) {
        {
          const std::lock_guard<std::mutex> lock(mutex);
          first_piece_emitted = true;
        }
        condition.notify_all();
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return released; });
      }
    }
    result.completion_tokens = result.tokens.size();
    result.finish_reason = finish_reason;
    completed.store(true);
    return result;
  }

  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t max_tokens,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, bool stream_output) override {
    if (reject_on_start.has_value()) {
      throw gufo::server::TextGenerationError(*reject_on_start,
                                              "injected admission rejection");
    }
    return TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                             is_cancelled, stream_output);
  }

  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }

  bool WaitForFirstPiece() {
    std::unique_lock<std::mutex> lock(mutex);
    return condition.wait_for(lock, 2s, [&] { return first_piece_emitted; });
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    condition.notify_all();
  }

  std::vector<std::string> pieces;
  FinishReason finish_reason{FinishReason::kStop};
  bool block_after_first_piece{false};
  std::atomic<bool> completed{false};
  std::atomic<int> chat_calls{0};
  gufo::server::ChatRequest last_request;
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults_value;
  std::size_t last_max_tokens{0};
  float last_temperature{0.0F};
  gufo::sampling::SamplingConfig last_sampling;
  std::optional<gufo::server::TextGenerationErrorCode> reject_on_start;

private:
  std::mutex mutex;
  std::condition_variable condition;
  bool first_piece_emitted{false};
  bool released{false};
};

gufo::server::HttpRequest Request(
    std::string body,
    std::vector<std::pair<std::string, std::string>> headers = {}) {
  return {
      .method = "POST",
      .path = "/v1/chat/completions",
      .query = {},
      .body = std::move(body),
      .headers = std::move(headers),
      .is_cancelled = {},
  };
}

void TestStreamingIsLive() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.finish_reason =
      gufo::server::TextGenerationBackend::FinishReason::kLength;
  backend.block_after_first_piece = true;

  auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":2,
        "stream":true,
        "stream_options":{"include_usage":true}
      })"),
                                                 backend);
  Expect(response.status == 200, "Streaming request is accepted");
  Expect(static_cast<bool>(response.streaming_body),
         "Streaming request returns a streaming body");

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::string output;
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view chunk) {
      {
        const std::lock_guard<std::mutex> lock(output_mutex);
        output.append(chunk);
      }
      output_condition.notify_all();
      return true;
    });
  });

  Expect(backend.WaitForFirstPiece(), "Backend emits the first token");
  {
    std::unique_lock<std::mutex> lock(output_mutex);
    Expect(output_condition.wait_for(
               lock, 2s,
               [&] {
                 return output.find(R"("content":"Hel")") != std::string::npos;
               }),
           "First content delta is written promptly");
    Expect(!backend.completed.load(),
           "First content delta arrives before generation completes");
  }

  backend.Release();
  writer.join();

  Expect(output.find(R"("finish_reason":"length")") != std::string::npos,
         "Token limit is reported as finish_reason length");
  Expect(output.find(R"("prompt_tokens":7)") != std::string::npos,
         "Usage is emitted when requested");
  Expect(output.find(R"("cached_tokens":5)") != std::string::npos,
         "Usage reports transparently reused prompt tokens");
  Expect(output.find(R"("prefill_tokens":2)") != std::string::npos,
         "Usage reports actual prefill work");
  Expect(output.find(R"("prefill_ms":2.5)") != std::string::npos,
         "Usage reports server prefill time");
  Expect(output.find(R"("decode_ms":4)") != std::string::npos,
         "Usage reports server decode time");
  Expect(
      output.find(R"("requested_logical_concurrency":4)") != std::string::npos,
      "Usage reports configured logical concurrency");
  Expect(
      output.find(R"("execution_plan":"serial-fallback")") != std::string::npos,
      "Usage reports the executed serving plan");
  Expect(output.ends_with("data: [DONE]\n\n"),
         "Stream terminates with the OpenAI DONE sentinel");
}

void TestToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
      "</parameter>\n</function>\n</tool_call>",
  };

  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"weather in Rome"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"get_weather",
            "description":"Get weather",
            "parameters":{
              "type":"object",
              "properties":{"city":{"type":"string"}},
              "required":["city"]
            }
          }
        }],
        "tool_choice":"required",
        "stream":false
      })"),
                                                       backend);

  Expect(response.status == 200, "Tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Tool generation reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"get_weather")") != std::string::npos,
         "Tool name is translated to OpenAI format");
  Expect(response.body.find(R"(\"city\":\"Rome\")") != std::string::npos,
         "Tool arguments are translated to JSON");
  Expect(backend.last_request.tools.size() == 1,
         "Tool schema reaches the model backend");
  Expect(backend.last_request.tool_choice ==
             gufo::server::ChatRequest::ToolChoice::kRequired,
         "Required tool choice reaches the model backend");
}

void TestDeepSeekToolCallsAreStructured() {
  FakeBackend backend;
  backend.pieces = {
      "<｜DSML｜tool_calls｜>\n"
      "<｜DS｜invoke name=\"read\">\n"
      "<｜DS｜parameter name=\"path\" string=\"true\">"
      "/etc/hostname</｜DS｜parameter>\n"
      "</｜DS｜invoke>\n"
      "</｜DSML｜tool_calls｜>",
  };

  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"read the hostname"}],
        "tools":[{
          "type":"function",
          "function":{
            "name":"read",
            "description":"Read a file",
            "parameters":{
              "type":"object",
              "properties":{"path":{"type":"string"}},
              "required":["path"]
            }
          }
        }],
        "stream":false
      })"),
                                                       backend);

  Expect(response.status == 200, "DeepSeek tool request is accepted");
  Expect(response.body.find(R"("finish_reason":"tool_calls")") !=
             std::string::npos,
         "Hybrid DeepSeek syntax reports tool_calls finish reason");
  Expect(response.body.find(R"("name":"read")") != std::string::npos,
         "Hybrid DeepSeek tool name is translated");
  Expect(
      response.body.find(R"(\"path\":\"/etc/hostname\")") != std::string::npos,
      "Hybrid DeepSeek tool arguments are translated");
}

void TestBackendSamplingDefaults() {
  FakeBackend backend;
  backend.defaults = {
      .max_tokens = 37,
      .sampling = {.temperature = 0.25F},
  };

  const auto default_response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                               backend);
  Expect(default_response.status == 200, "Defaulted request is accepted");
  Expect(backend.last_max_tokens == 37,
         "Backend max-token default reaches generation");
  Expect(backend.last_temperature > 0.24F && backend.last_temperature < 0.26F,
         "Backend temperature default reaches generation");

  const auto override_response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "max_tokens":11,
        "temperature":0
      })"),
                                                                backend);
  Expect(override_response.status == 200, "Sampling override is accepted");
  Expect(backend.last_max_tokens == 11,
         "Explicit max tokens override the backend default");
  Expect(backend.last_temperature == 0.0F,
         "Explicit temperature overrides the backend default");
}

void TestAllSamplingControlsReachBackend() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "temperature":0.8,
        "top_k":40,
        "top_p":0.9,
        "min_p":0.05,
        "min_keep":3,
        "seed":123,
        "repeat_penalty":1.1,
        "repeat_last_n":32,
        "frequency_penalty":0.25,
        "presence_penalty":0.5
      })"),
                                                       backend);

  Expect(response.status == 200, "Complete sampling request is accepted");
  const auto& sampling = backend.last_sampling;
  Expect(sampling.temperature > 0.79F && sampling.temperature < 0.81F,
         "temperature reaches backend");
  Expect(sampling.top_k == 40, "top-k reaches backend");
  Expect(sampling.top_p > 0.89F && sampling.top_p < 0.91F,
         "top-p reaches backend");
  Expect(sampling.min_p > 0.04F && sampling.min_p < 0.06F,
         "min-p reaches backend");
  Expect(sampling.min_keep == 3, "min-keep reaches backend");
  Expect(sampling.seed == 123, "seed reaches backend");
  Expect(sampling.repeat_penalty > 1.09F && sampling.repeat_penalty < 1.11F,
         "repeat penalty reaches backend");
  Expect(sampling.repeat_last_n == 32, "repeat window reaches backend");
  Expect(
      sampling.frequency_penalty > 0.24F && sampling.frequency_penalty < 0.26F,
      "frequency penalty reaches backend");
  Expect(sampling.presence_penalty > 0.49F && sampling.presence_penalty < 0.51F,
         "presence penalty reaches backend");
}

void TestUnsupportedSamplingControlsAreRejected() {
  FakeBackend backend;
  for (const auto* field :
       {"draft_temperature", "temperature_draft", "draft_top_k", "draft_top_p",
        "draft_min_p", "draft_seed", "draft_policy", "samplers", "typical_p",
        "mirostat", "dynatemp_range"}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
    auto body = gufo::server::json::parse(request.body);
    body[field] = 0.8;
    request.body = body.dump();
    const auto response = gufo::server::HandleOpenAiChat(request, backend);
    Expect(response.status == 400 &&
               response.body.find("unsupported_sampling") != std::string::npos,
           "unsupported sampling must not be silently ignored");
  }
}

void TestAssistantReasoningContentReachesBackend() {
  FakeBackend backend;
  backend.pieces = {"Blue"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[
          {"role":"user","content":"Name one color."},
          {
            "role":"assistant",
            "reasoning_content":"I should answer concisely.",
            "content":"Red"
          },
          {"role":"user","content":"Name another."}
        ],
        "max_tokens":1
      })"),
                                                       backend);

  Expect(response.status == 200, "Assistant reasoning history is accepted");
  Expect(backend.last_request.messages.size() == 3,
         "Complete reasoning history reaches the backend");
  Expect(
      backend.last_request.messages[1].thought == "I should answer concisely.",
      "Assistant reasoning_content reaches the model template");
  Expect(backend.last_request.messages[1].content == "Red",
         "Assistant visible content remains separate from reasoning");
}

void TestPiReasoningControlsAndOutputFraming() {
  FakeBackend backend;
  backend.pieces = {"I should verify this.", "</think>\n\n", "Forty-two."};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"What is six times seven?"}],
        "reasoning_effort":"high",
        "chat_template_kwargs":{
          "enable_thinking":true,
          "reasoning_effort":"high",
          "preserve_thinking":false
        }
      })"),
                                                       backend);

  Expect(response.status == 200, "Pi reasoning request is accepted");
  Expect(backend.last_request.reasoning.enabled == true,
         "Pi enable_thinking reaches the backend");
  Expect(backend.last_request.reasoning.effort == gufo::ReasoningEffort::kHigh,
         "Pi reasoning effort reaches the backend");
  Expect(backend.last_request.reasoning.preserve_thinking == false,
         "Pi preservation control reaches the backend");
  Expect(response.body.find(R"("reasoning_content":"I should verify this.")") !=
             std::string::npos,
         "Prompt-opened reasoning is returned separately");
  Expect(response.body.find(R"("content":"Forty-two.")") != std::string::npos,
         "Visible answer excludes reasoning");
}

void TestPiNativeDeepSeekThinkingObject() {
  FakeBackend backend;
  backend.pieces = {"Check.", "</think>", "Done."};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Check this."}],
        "thinking":{"type":"enabled"},
        "reasoning_effort":"high"
      })"),
                                                       backend);
  Expect(response.status == 200,
         "Pi native DeepSeek thinking object is accepted");
  Expect(backend.last_request.reasoning.enabled == true,
         "Pi DeepSeek thinking.type enables reasoning");
  Expect(backend.last_request.reasoning.effort == gufo::ReasoningEffort::kHigh,
         "Pi DeepSeek reasoning effort reaches the backend");

  const auto disabled = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Answer directly."}],
        "thinking":{"type":"disabled"}
      })"),
                                                       backend);
  Expect(disabled.status == 200,
         "Pi native DeepSeek disabled thinking object is accepted");
  Expect(backend.last_request.reasoning.enabled == false,
         "Pi DeepSeek thinking.type disables reasoning");
}

void TestStreamingPromptOpenedReasoning() {
  FakeBackend backend;
  backend.pieces = {"Check", " carefully", "</thi", "nk>\n\n", "Done"};
  auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"Check it."}],
        "reasoning_effort":"xhigh",
        "stream":true
      })"),
                                                 backend);
  Expect(response.status == 200, "Streaming reasoning request is accepted");
  std::string output;
  response.streaming_body([&](std::string_view chunk) {
    output.append(chunk);
    return true;
  });
  Expect(output.find(R"("reasoning_content":"Check")") != std::string::npos,
         "Streaming reasoning uses reasoning_content deltas");
  Expect(output.find(R"("content":"Done")") != std::string::npos,
         "Streaming answer switches to content after think end");
  Expect(output.find(R"("content":"Check")") == std::string::npos,
         "Reasoning is never exposed as visible content");
}

void TestConflictingReasoningControlsAreRejected() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "reasoning_effort":"high",
        "chat_template_kwargs":{"enable_thinking":false}
      })"),
                                                       backend);
  Expect(response.status == 400, "Conflicting reasoning controls are rejected");
  Expect(response.body.find("invalid_reasoning") != std::string::npos,
         "Reasoning conflict has a stable error code");
  Expect(backend.chat_calls.load() == 0,
         "Invalid reasoning request never reaches generation");
}

void TestWrongModelIsRejected() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"wrong-model",
        "messages":[{"role":"user","content":"hello"}]
      })"),
                                                       backend);

  Expect(response.status == 404, "Unknown model is rejected");
  Expect(response.body.find("model_not_found") != std::string::npos,
         "Unknown model returns a stable error code");
  Expect(backend.chat_calls.load() == 0,
         "Rejected request never reaches the backend");
}

void TestClientIdentityReachesBackend() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response =
      gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}]
      })",
                                             {{"X-Client-ID", "pi-agent-2"}}),
                                     backend);

  Expect(response.status == 200, "identified request is accepted");
  Expect(backend.last_request.client_id == "pi-agent-2",
         "validated client identity reaches scheduling backend");
}

void TestInvalidClientIdentityIsRejected() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(
      Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}]
      })",
              {{"X-Client-ID", "contains spaces"}}),
      backend);

  Expect(response.status == 400, "invalid client identity is rejected");
  Expect(response.body.find("invalid_client_id") != std::string::npos,
         "invalid client identity returns a stable code");
  Expect(backend.chat_calls.load() == 0,
         "invalid identity never reaches generation");
}

void TestStreamingOverloadIsRejectedBeforeHeaders() {
  FakeBackend backend;
  backend.reject_on_start = gufo::server::TextGenerationErrorCode::kQueueFull;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
        "model":"test-model",
        "messages":[{"role":"user","content":"hello"}],
        "stream":true
      })"),
                                                       backend);

  Expect(response.status == 429,
         "streaming overload is rejected synchronously");
  Expect(!response.streaming_body,
         "overloaded stream does not commit successful SSE headers");
  Expect(response.body.find("queue_full") != std::string::npos,
         "overload response carries a stable retry code");
  bool retry_after = false;
  for (const auto& [name, value] : response.headers) {
    retry_after = retry_after || (name == "Retry-After" && value == "1");
  }
  Expect(retry_after, "retryable overload advertises Retry-After");
}

}  // namespace

int main() {
  TestStreamingIsLive();
  TestBackendSamplingDefaults();
  TestAllSamplingControlsReachBackend();
  TestUnsupportedSamplingControlsAreRejected();
  TestAssistantReasoningContentReachesBackend();
  TestPiReasoningControlsAndOutputFraming();
  TestPiNativeDeepSeekThinkingObject();
  TestStreamingPromptOpenedReasoning();
  TestConflictingReasoningControlsAreRejected();
  TestToolCallsAreStructured();
  TestDeepSeekToolCallsAreStructured();
  TestWrongModelIsRejected();
  TestClientIdentityReachesBackend();
  TestInvalidClientIdentityIsRejected();
  TestStreamingOverloadIsRejectedBeforeHeaders();
  std::cout << "All OpenAI chat protocol tests passed\n";
  return 0;
}
