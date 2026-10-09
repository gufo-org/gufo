#include "src/cli/serve/openai_chat.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/models/qwen/control_tokens.hpp"

using gufo::tokenization::kImEnd;

namespace {

using namespace std::chrono_literals;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

/// Streaming splits deltas differently from the buffered path, so content that
/// differs only in surrounding whitespace is the same response.
std::string Trimmed(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

/// Reports fixed prompt progress before delegating token generation.
class ProgressRequest final
    : public gufo::server::TextGenerationBackend::GenerationRequest {
public:
  using Backend = gufo::server::TextGenerationBackend;

  ProgressRequest(std::shared_ptr<GenerationRequest> inner,
                  std::vector<Backend::PromptProgress> progress,
                  std::optional<gufo::sampling::JsonConstraint::ToolFormat>
                      tool_format = {})
      : inner_(std::move(inner)),
        progress_(std::move(progress)),
        tool_format_(tool_format) {}

  Backend::Result Wait(const Backend::TokenCallback& on_token,
                       const Backend::ProgressCallback& on_progress,
                       const Backend::StartCallback& on_start) override {
    for (const auto& value : progress_) {
      if (on_progress && !on_progress(value)) {
        inner_->Cancel();
        break;
      }
    }
    return inner_->Wait(on_token, on_progress, on_start);
  }
  void Cancel() noexcept override { inner_->Cancel(); }
  std::optional<gufo::sampling::JsonConstraint::ToolFormat> ToolFormat()
      const override {
    return tool_format_;
  }

private:
  std::shared_ptr<GenerationRequest> inner_;
  std::vector<Backend::PromptProgress> progress_;
  const std::optional<gufo::sampling::JsonConstraint::ToolFormat> tool_format_;
};

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
    if (initial_output_state_override)
      return *initial_output_state_override;
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kReasoning
               : InitialOutputState::kContent;
  }

  Result complete(std::string_view, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck&, const TokenCallback&,
                  std::string_view, const std::vector<std::string>&) override {
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
    result.reasoning_tokens = reasoning_tokens;
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
    result.cache_hit = cache_miss_reason.empty();
    result.cache_miss_reason = cache_miss_reason;
    if (!result.cache_hit) {
      result.cached_prompt_tokens = 0;
      result.prefill_tokens = result.prompt_tokens;
      result.cache_common_prefix_tokens = 2;
      result.cache_checkpoint_tokens = 5;
    }
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
    result.stop_sequence = stop_sequence;
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
    return std::make_shared<ProgressRequest>(
        TextGenerationBackend::start_chat(request, max_tokens, sampling,
                                          is_cancelled, stream_output),
        progress, tool_format);
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
  std::vector<PromptProgress> progress;
  std::optional<gufo::sampling::JsonConstraint::ToolFormat> tool_format;
  std::string cache_miss_reason;
  FinishReason finish_reason{FinishReason::kStop};
  std::string stop_sequence;
  bool block_after_first_piece{false};
  std::atomic<bool> completed{false};
  std::atomic<int> chat_calls{0};
  gufo::server::ChatRequest last_request;
  SamplingDefaults defaults;
  gufo::ReasoningOptions reasoning_defaults_value;
  std::optional<InitialOutputState> initial_output_state_override;
  std::size_t last_max_tokens{0};
  std::size_t reasoning_tokens{0};
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
  backend.pieces = {"Hel<tool_call>", "lo"};
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
                 return output.find(R"("content":"Hel<tool_call>")") !=
                        std::string::npos;
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
  Expect(output.find(R"("prompt_n":2)") != std::string::npos,
         "Timings exclude cached tokens");
  Expect(output.find(R"("prompt_tokens_per_second":800)") != std::string::npos,
         "Usage throughput counts only tokens actually prefilled");
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
  Expect(response.stream_log &&
             response.stream_log->details.find("cached_tokens=5") !=
                 std::string::npos &&
             response.stream_log->details.find("finish=length") !=
                 std::string::npos,
         "Streaming completion retains request diagnostics");
}

void TestLessThanProseStreamsBeforeCompletion() {
  FakeBackend backend;
  backend.pieces = {"3 < 5 and x < y", " are comparisons."};
  backend.block_after_first_piece = true;
  auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model","messages":[{"role":"user","content":"explain"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
    "stream":true,"reasoning_effort":"none"
  })"),
                                                 backend);
  std::mutex mutex;
  std::condition_variable ready;
  std::string content;
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view part) {
      if (part == "data: [DONE]\n\n")
        return true;
      const auto event = gufo::json::parse(part.substr(6));
      {
        const std::lock_guard lock(mutex);
        for (const auto& choice : event.find("choices")->items()) {
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        }
      }
      ready.notify_all();
      return true;
    });
  });
  Expect(backend.WaitForFirstPiece(), "backend reaches the comparison prose");
  bool arrived;
  {
    std::unique_lock lock(mutex);
    arrived =
        ready.wait_for(lock, 2s, [&] { return content == backend.pieces[0]; });
  }
  const auto completed = backend.completed.load();
  backend.Release();
  writer.join();
  Expect(arrived && !completed,
         "all comparison prose streams while generation is still blocked");
  Expect(content == "3 < 5 and x < y are comparisons.",
         "comparison content stays exact");
}

void TestCachePromptOption() {
  for (bool stream : {false, true}) {
    for (const auto value : {"true", "false", "null", "0", "\"false\""}) {
      FakeBackend backend;
      auto response = gufo::server::HandleOpenAiChat(
          Request(
              std::string(
                  R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":)") +
              (stream ? "true" : "false") + R"(,"cache_prompt":)" + value +
              "}"),
          backend);
      const bool valid = std::string_view(value) == "true" ||
                         std::string_view(value) == "false";
      Expect(response.status == (valid ? 200 : 400),
             "cache_prompt accepts only JSON booleans");
      if (valid) {
        if (response.streaming_body)
          response.streaming_body([](std::string_view) { return true; });
        Expect(backend.last_request.cache_prompt ==
                   (std::string_view(value) == "true"),
               "cache_prompt reaches both buffered and streaming backends");
      }
    }
  }
}

void TestStreamingWithoutUsage() {
  for (const auto* options :
       {"", R"(,"stream_options":{"include_usage":false})"}) {
    FakeBackend backend;
    backend.pieces = {"ok"};
    auto response = gufo::server::HandleOpenAiChat(
        Request(
            std::string(
                R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true)") +
            options + "}"),
        backend);
    Expect(response.status == 200 && response.streaming_body,
           "Stream without usage is accepted");
    std::string output;
    response.streaming_body([&](std::string_view chunk) {
      output += chunk;
      return true;
    });
    Expect(output.find(R"("usage":)") == std::string::npos &&
               output.ends_with("data: [DONE]\n\n"),
           "Usage chunk is opt-in");
    Expect(
        output.find(R"("prompt_per_second":800)") != std::string::npos &&
            output.find(R"("cache_n":5)") != std::string::npos,
        "Terminal timings survive omitted or disabled usage on cached turns");
    Expect(response.stream_log &&
               response.stream_log->details.find("generated_tokens=1") !=
                   std::string::npos,
           "Request diagnostics do not depend on client usage preference");
  }
}

void TestUtf8Output() {
  const auto check = [](std::vector<std::string> pieces,
                        const std::string& expected, bool reasoning) {
    FakeBackend backend;
    backend.pieces = std::move(pieces);
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"hello"}]
    })");
    body["chat_template_kwargs"]["enable_thinking"] = reasoning;
    const auto field = reasoning ? "reasoning_content" : "content";
    const auto complete =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(complete.status == 200, "Unicode completion succeeds");
    const auto parsed = gufo::json::parse(complete.body);
    Expect(
        parsed.find("choices")->items().front().find("message")->member_str(
            field) == expected,
        "Non-streaming UTF-8 preserves scalars and replaces malformed bytes");
    body["stream"] = true;
    const auto stream =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(stream.status == 200 && stream.streaming_body,
           "Unicode stream succeeds");
    std::string text;
    stream.streaming_body([&](std::string_view chunk) {
      if (chunk == "data: [DONE]\n\n")
        return true;
      Expect(chunk.starts_with("data: "), "SSE has a data prefix");
      const auto event = gufo::json::parse(chunk.substr(6));
      for (const auto& choice : event.find("choices")->items()) {
        const auto* delta = choice.find("delta");
        if (delta)
          text += delta->member_str(field);
      }
      Expect(expected.starts_with(text) &&
                 (text.size() == expected.size() ||
                  (static_cast<unsigned char>(expected[text.size()]) & 0xC0) !=
                      0x80),
             "Every SSE event ends at a complete Unicode scalar");
      return true;
    });
    Expect(text == expected, "Streaming and non-streaming UTF-8 agree");
  };
  const std::string valid = "Aé中┌😀Z";
  const std::string replacement = "\xEF\xBF\xBD";
  for (const bool reasoning : {false, true}) {
    for (std::size_t split = 1; split < valid.size(); ++split)
      check({valid.substr(0, split), valid.substr(split)}, valid, reasoning);
    std::vector<std::string> bytes;
    for (char byte : valid)
      bytes.emplace_back(1, byte);
    check(bytes, valid, reasoning);
    check({"\xE2\x82", "X"}, replacement + "X", reasoning);
    check({"\xF0", "\x9F"}, replacement, reasoning);
    check({"\x80", "ok"}, replacement + "ok", reasoning);
    check({"\xC0\xAF"}, replacement + replacement, reasoning);
    check({"\xED", "\xA0\x80"}, replacement + replacement + replacement,
          reasoning);
    check({"\xF4\x90\x80\x80"},
          replacement + replacement + replacement + replacement, reasoning);
  }
}

void TestCachedPrefillMetrics() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})"),
      backend);
  Expect(response.status == 200, "Cached non-streaming response succeeds");
  const auto body = gufo::json::parse(response.body);
  const auto* usage = body.find("usage");
  const auto* timings = body.find("timings");
  Expect(body.find("metrics") == nullptr,
         "One timing schema prevents proxies from subtracting cached tokens "
         "twice");
  Expect(usage && usage->member_size("prompt_tokens") == 7,
         "Token usage includes cached tokens");
  Expect(timings && timings->member_size("prompt_n") == 2 &&
             timings->member_double("prompt_per_second") == 800 &&
             timings->member_double("prompt_per_token_ms") == 1.25,
         "Non-streaming timings report executed prefill work");

  gufo::server::TextGenerationBackend::Result cached;
  cached.prompt_tokens = cached.cached_prompt_tokens = 1024;
  cached.prefill_ms = 0.01;
  Expect(gufo::server::PrefillTokensPerSecond(cached) == 0,
         "Full cache hits cannot report artificial prefill throughput");
  cached.prefill_tokens = 10;
  cached.prefill_ms = 0;
  Expect(gufo::server::PrefillTokensPerSecond(cached) == 0,
         "Untimed work does not divide by zero");

  backend.cache_miss_reason = "prefix_changed";
  const auto miss = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"changed"}]})"),
      backend);
  const auto miss_body = gufo::json::parse(miss.body);
  const auto* miss_usage = miss_body.find("usage");
  const auto* metrics = miss_usage ? miss_usage->find("gufo") : nullptr;
  Expect(metrics &&
             metrics->member_str("cache_miss_reason") == "prefix_changed" &&
             metrics->member_size("cache_common_prefix_tokens") == 2 &&
             metrics->member_size("cache_checkpoint_tokens") == 5,
         "Usage explains cache misses without exposing prompt text");
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

void TestToolParameterCompatibility() {
  using gufo::json::Value;
  const std::pair<const char*, const char*> cases[] = {
      {R"({"name":"f"})", "{}"},
      {R"({"name":"f","parameters":null})", "{}"},
      {R"({"name":"f","parameters":{}})", "{}"},
      {R"({"name":"f","parametersJsonSchema":null})", "{}"},
      {R"({"name":"f","parametersJsonSchema":{"type":"object"}})",
       R"({"type":"object"})"},
      {R"({"name":"f","parameters":null,"parametersJsonSchema":{"type":"object"}})",
       R"({"type":"object"})"},
      {R"({"name":"f","parameters":{},"parametersJsonSchema":"ignored"})",
       "{}"},
  };
  for (bool flat : {false, true}) {
    for (bool stream : {false, true}) {
      for (const auto& [function_json, expected_parameters] : cases) {
        FakeBackend backend;
        backend.pieces = {
            "<tool_call>\n<function=f>\n</function>\n</tool_call>"};
        auto body = gufo::json::parse(R"({
          "model":"test-model","messages":[{"role":"user","content":"call f"}],
          "tool_choice":"required","tools":[]
        })");
        auto definition = Value::object();
        definition["type"] = "function";
        auto function = gufo::json::parse(function_json);
        if (flat) {
          for (const auto& [key, value] : function.members())
            definition.append_member(key, value);
        } else {
          definition["function"] = function;
        }
        body["tools"].push_back(std::move(definition));
        body["stream"] = stream;
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 200, "Compatible tool schema is accepted");
        std::string output = response.body;
        if (stream) {
          Expect(static_cast<bool>(response.streaming_body),
                 "Compatible tool request supports streaming");
          response.streaming_body([&](std::string_view chunk) {
            output += chunk;
            return true;
          });
        }
        Expect(output.find(R"("name":"f")") != std::string::npos &&
                   output.find(R"("finish_reason":"tool_calls")") !=
                       std::string::npos,
               "No-argument function returns a structured tool call");
        Expect(backend.last_request.tools.size() == 1,
               "Normalized tool reaches the backend");
        const auto& tool = backend.last_request.tools.front();
        Expect(tool.parameters_json == expected_parameters,
               "Missing/null schemas normalize and parameters take precedence");
        auto expected_function = Value::object();
        expected_function["name"] = "f";
        expected_function["parameters"] =
            gufo::json::parse(expected_parameters);
        auto expected_definition = Value::object();
        expected_definition["type"] = "function";
        expected_definition["function"] = expected_function;
        Expect(tool.definition_json == expected_definition.dump(),
               "Templates receive the normalized nested definition");
      }
    }
  }
}

void TestInvalidToolsFailBeforeGeneration() {
  const char* invalid[] = {
      "null",
      "42",
      R"({"function":{"name":"f"}})",
      R"({"type":42,"function":{"name":"f"}})",
      R"({"type":"custom","custom":{"name":"shell"}})",
      R"({"type":"function","function":null,"name":"f"})",
      R"({"type":"function","function":[],"name":"f"})",
      R"({"type":"function","function":{}})",
      R"({"type":"function","name":""})",
      R"({"type":"function","name":42})",
      R"({"type":"function","name":"f","parameters":"bad"})",
      R"({"type":"function","name":"f","parameters":[]})",
      R"({"type":"function","name":"f","parameters":false})",
      R"({"type":"function","name":"f","parametersJsonSchema":[]})",
  };
  for (bool stream : {false, true}) {
    for (bool valid_first : {false, true}) {
      for (const char* entry : invalid) {
        FakeBackend backend;
        auto body = gufo::json::parse(R"({
          "model":"test-model","messages":[{"role":"user","content":"use tools"}],
          "tools":[]
        })");
        if (valid_first)
          body["tools"].push_back(gufo::json::parse(
              R"({"type":"function","function":{"name":"valid","parameters":{}}})"));
        body["tools"].push_back(gufo::json::parse(entry));
        body["stream"] = stream;
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 400 &&
                   response.body.find("invalid_tools") != std::string::npos &&
                   !response.streaming_body && backend.chat_calls == 0,
               "Invalid tools fail before generation, including mixed lists");
      }
    }
  }
}

void TestResponsesClientCompatTolerances() {
  // Hosted tool types (Responses-only) are skipped, not rejected. Namespaces
  // group client-executed functions and flatten to the function list, while a
  // malformed function still fails the controls.
  {
    auto body = gufo::json::parse(R"({
      "tools":[
        {"type":"function","name":"exec","parameters":{"type":"object",
          "properties":{"cmd":{"type":"string"}},"required":["cmd"]}},
        {"type":"web_search","external_web_access":false},
        {"type":"namespace","name":"agents","tools":[{"type":"function",
          "name":"spawn"}]},
        {"type":"code_interpreter"}]})");
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses skips hosted tool types without failing");
    Expect(chat.tools.size() == 2 && chat.tools[0].name == "exec" &&
               chat.tools[1].name == "spawn",
           "Function tools survive, including those nested in namespaces");
  }
  {
    // A namespace is a client-side grouping, not a hosted tool: its functions
    // must reach the model with their schema and strictness intact.
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[{"type":"namespace","name":"crm",
                 "description":"Local customer tools","tools":[{"type":"function",
                 "name":"lookup","parameters":{"type":"object","properties":{},
                 "required":[],"additionalProperties":false},"strict":true}]}]})"),
               &chat) &&
               chat.tools.size() == 1 && chat.tools[0].name == "lookup" &&
               chat.tools[0].definition_json.find("\"strict\":true") !=
                   std::string::npos,
           "Namespace function tools are flattened with their definitions");
    // Rejected parses leave the request untouched, so every case parses into
    // fresh state instead of inheriting tools from a previous parse.
    gufo::server::ChatRequest ambiguous;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[
                 {"type":"function","name":"lookup"},
                 {"type":"namespace","name":"crm","tools":[{"type":"function",
                  "name":"lookup"}]}]})"),
               &ambiguous)
               .has_value(),
           "Namespace routing is rejected when function names are ambiguous");
    gufo::server::ChatRequest malformed;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[{"type":"namespace",
                 "name":"crm"}]})"),
               &malformed)
               .has_value(),
           "A namespace without a tools array is rejected");
  }
  {
    // The shared 128-function cap counts flattened functions at append time,
    // so both orderings around a full namespace reject the 129th function.
    std::string nested = R"({"type":"namespace","name":"crm","tools":[)";
    for (int i = 0; i < 128; ++i) {
      if (i != 0)
        nested += ",";
      nested += R"({"type":"function","name":"f)" + std::to_string(i) + "\"}";
    }
    nested += "]}";
    const std::string outside = R"({"type":"function","name":"outside"})";
    gufo::server::ChatRequest accepted;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"tools":[)" + nested + "]}"), &accepted) &&
               accepted.tools.size() == 128,
           "Exactly 128 functions flattened from a namespace are accepted");
    for (const bool outside_first : {false, true}) {
      const auto body = outside_first
                            ? R"({"tools":[)" + outside + "," + nested + "]}"
                            : R"({"tools":[)" + nested + "," + outside + "]}";
      gufo::server::ChatRequest overflow;
      Expect(gufo::server::ParseOpenAiResponseControls(gufo::json::parse(body),
                                                       &overflow)
                 .has_value(),
             "Both boundary orderings reject the 129th flattened function");
    }
  }
  {
    gufo::server::ChatRequest chat;
    Expect(
        gufo::server::ParseOpenAiResponseControls(
            gufo::json::parse(
                R"({"tools":[{"type":"function","function":null,"name":"f"}]})"),
            &chat)
            .has_value(),
        "Responses still rejects a malformed function tool");
  }

  // reasoning.summary is accepted and ignored; effort still applies; a
  // genuinely unknown reasoning member still fails.
  {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(
                   R"({"reasoning":{"effort":"low","summary":"auto"}})"),
               &chat) &&
               chat.reasoning.enabled == true &&
               chat.reasoning.effort == gufo::ReasoningEffort::kLow,
           "reasoning.summary is accepted while effort still applies");
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"reasoning":{"effort":"low","bogus":1}})"),
               &chat)
               .has_value(),
           "Unknown reasoning members are still rejected");
  }

  // text.verbosity is accepted and leaves the response format unset;
  // text.format still applies beside it; an unknown text member still fails.
  {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"verbosity":"low"}})"), &chat) &&
               !chat.response_format,
           "text.verbosity is accepted without forcing a response format");
    gufo::server::ChatRequest formatted;
    Expect(!gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"format":{"type":"json_object"},
                 "verbosity":"low"}})"),
               &formatted) &&
               formatted.response_format,
           "text.format still applies alongside verbosity");
    gufo::server::ChatRequest rejected;
    Expect(gufo::server::ParseOpenAiResponseControls(
               gufo::json::parse(R"({"text":{"bogus":1}})"), &rejected)
               .has_value(),
           "Unknown text members are still rejected");
  }

  // Codex routes calls by namespace and name: a flattened function call must
  // carry its namespace, while a top-level function call carries none.
  for (const bool stream : {false, true}) {
    auto body = gufo::json::parse(R"({"input":"go","tools":[
      {"type":"function","name":"exec","strict":false,
       "parameters":{"type":"object","properties":{}}},
      {"type":"namespace","name":"multi_agent_v1","tools":[
        {"type":"function","name":"close_agent","strict":false,
         "parameters":{"type":"object","properties":{}}}]}]})");
    body["stream"] = stream;
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts a namespace beside a function");
    FakeBackend backend;
    backend.pieces = {
        "<tool_call>{\"name\":\"close_agent\",\"arguments\":{}}</tool_call>"
        "<tool_call>{\"name\":\"exec\",\"arguments\":{}}</tool_call>"};
    auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 256, {}, stream);
    Expect(response.status == 200, "namespaced tool request succeeds");
    std::vector<gufo::json::Value> results;
    if (stream) {
      response.streaming_body([&](std::string_view chunk) {
        auto pos = chunk.find("data: ");
        if (pos != std::string_view::npos &&
            !chunk.substr(pos + 6).starts_with("[DONE]")) {
          auto event = gufo::json::parse(chunk.substr(pos + 6));
          if (event.member_str("type") == "response.completed")
            results.push_back(*event.find("response"));
        }
        return true;
      });
    } else {
      results.push_back(gufo::json::parse(response.body));
    }
    std::map<std::string, std::string> namespaces;
    for (const auto& result : results)
      for (const auto& item : result.find("output")->items())
        if (item.member_str("type") == "function_call")
          namespaces[item.member_str("name")] =
              item.contains("namespace") ? item.member_str("namespace") : "-";
    Expect(namespaces.size() == 2 &&
               namespaces["close_agent"] == "multi_agent_v1" &&
               namespaces["exec"] == "-",
           "function calls echo only their own namespace");
  }
}

void TestMidConversationSystemMessagesKeepOrder() {
  // Codex sends developer messages mid-conversation. The adapter keeps them in
  // place; each model template decides how to render them.
  FakeBackend backend;
  const auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[
      {"role":"user","content":"first"},
      {"role":"assistant","content":"noted"},
      {"role":"developer","content":"compacted state"},
      {"role":"user","content":"second"}]})");
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(response.status == 200 && backend.last_request.messages.size() == 4 &&
             backend.last_request.messages[0].content == "first" &&
             backend.last_request.messages[1].content == "noted" &&
             backend.last_request.messages[2].role ==
                 gufo::tokenization::ChatRole::kDeveloper &&
             backend.last_request.messages[2].content == "compacted state" &&
             backend.last_request.messages[3].content == "second",
         "Mid-conversation developer messages reach the backend in place");
}

void TestToolNameCharacters() {
  const auto declare = [](const std::string& name) {
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use it"}],
      "tools":[]
    })");
    auto tool = gufo::json::parse(
        R"({"type":"function","function":{"name":"PLACEHOLDER",
            "parameters":{"type":"object","properties":{}}}})");
    tool["function"]["name"] = name;
    body["tools"].push_back(std::move(tool));
    return body.dump();
  };
  const auto replay = [](const std::string& name) {
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use it"}]
    })");
    auto assistant =
        gufo::json::parse(R"({"role":"assistant","tool_calls":[]})");
    auto call = gufo::json::parse(
        R"({"id":"call_1","type":"function","function":{"name":"PLACEHOLDER",
            "arguments":"{\"path\":\"a.txt\"}"}})");
    call["function"]["name"] = name;
    assistant["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(assistant));
    body["messages"].push_back(gufo::json::parse(
        R"({"role":"tool","tool_call_id":"call_1","content":"done"})"));
    return body.dump();
  };

  // Agent harnesses name bridged tools after their server. Nothing between
  // the request and either renderer treats these characters as structure.
  const std::vector<std::string> renderable{
      "read_file",   "read-file", "readFile",
      "server:tool", "fs/read",   "github.create_issue",
      "tool@v1",     "a.b.c~d+e", std::string(64, 'n')};
  for (const std::string& name : renderable) {
    FakeBackend declared;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(declare(name)), declared);
    Expect(response.status == 200 && declared.last_request.tools.size() == 1 &&
               declared.last_request.tools.front().name == name,
           "A renderable tool name reaches the backend unchanged");

    FakeBackend replayed;
    Expect(gufo::server::HandleOpenAiChat(Request(replay(name)), replayed)
                       .status == 200 &&
               replayed.chat_calls == 1,
           "The same name is accepted when a message replays a call");
  }

  // New declarations retain their name restrictions. Historical calls are
  // records from an earlier turn and must not strand the conversation (#357).
  const std::vector<std::string> unrenderable{"bad>name",
                                              "bad<name",
                                              "bad\"name",
                                              "bad\\name",
                                              "bad name",
                                              "bad\nname",
                                              "bad\x7F"
                                              "name",
                                              "outil_traçage",
                                              "…",
                                              "reаd",
                                              "​read",
                                              std::string(65, 'n')};
  for (const std::string& name : unrenderable) {
    FakeBackend declared;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(declare(name)), declared);
    Expect(response.status == 400 &&
               response.body.find("invalid_tools") != std::string::npos &&
               declared.chat_calls == 0,
           "An unrenderable declared name fails with invalid_tools");

    FakeBackend replayed;
    const auto replay_response =
        gufo::server::HandleOpenAiChat(Request(replay(name)), replayed);
    Expect(replay_response.status == 200 && replayed.chat_calls == 1 &&
               replayed.last_request.messages[1].tool_calls[0].name == name &&
               replayed.last_request.messages[1].tool_calls[0].id == "call_1" &&
               replayed.last_request.messages[2].tool_call_id == "call_1",
           "Historical names and result pairing survive unchanged");

    auto item = gufo::json::parse(
        R"({"type":"function_call","call_id":"call_1","name":"placeholder",
            "arguments":"{\"path\":\"a.txt\"}"})");
    item["name"] = name;
    gufo::tokenization::ChatMessage message;
    gufo::core::ImageReadBudget budget;
    std::string error;
    Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                    &error) &&
               message.tool_calls.size() == 1 &&
               message.tool_calls[0].name == name &&
               message.tool_calls[0].id == "call_1" &&
               message.tool_calls[0].arguments[0].value == "a.txt",
           "Responses preserves the same historical names and arguments");
  }
}

// Chat templates render typed arguments with Jinja tojson, so the model
// generates that spelling. Replayed history must render it the same way, or
// the next turn re-prefills the call it generated.
void TestHistoricalTypedArgumentsUseTojson() {
  const auto item = gufo::json::parse(
      R"({"type":"function_call","call_id":"call_1","name":"edit",
          "arguments":"{\"path\":\"a.txt\",\"edits\":[{\"oldText\":\"a\",\"newText\":\"b\"}],\"n\":2}"})");
  gufo::tokenization::ChatMessage message;
  gufo::core::ImageReadBudget budget;
  std::string error;
  Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                  &error) &&
             message.tool_calls.size() == 1 &&
             message.tool_calls[0].arguments.size() == 3 &&
             message.tool_calls[0].arguments[0].value == "a.txt" &&
             message.tool_calls[0].arguments[0].is_string &&
             message.tool_calls[0].arguments[1].value ==
                 R"([{"oldText": "a", "newText": "b"}])" &&
             !message.tool_calls[0].arguments[1].is_string &&
             message.tool_calls[0].arguments[2].value == "2",
         "Typed historical arguments render as the template's tojson");
}

void TestMalformedHistoricalFunctions() {
  for (const auto source :
       {R"({"arguments":"{}"})", R"({"name":"","arguments":"{}"})",
        R"({"name":null,"arguments":"{}"})", R"({"name":42,"arguments":"{}"})",
        R"({"name":"read"})", R"({"name":"read","arguments":{}})",
        R"({"name":"read","arguments":null})",
        R"({"name":"read","arguments":"[]"})",
        R"({"name":"read","arguments":"["})",
        R"({"name":"read","arguments":"{\"x\":\"raw\nnewline\"}"})",
        R"({"name":"read\u0000file","arguments":"{}"})"}) {
    const auto function = gufo::json::parse(source);
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"continue"}]
    })");
    auto assistant =
        gufo::json::parse(R"({"role":"assistant","tool_calls":[]})");
    auto call = gufo::json::parse(R"({"id":"call_1","type":"function"})");
    call["function"] = function;
    assistant["tool_calls"].push_back(std::move(call));
    body["messages"].push_back(std::move(assistant));
    FakeBackend backend;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 400 && backend.chat_calls == 0 &&
               response.body.find("invalid_messages") != std::string::npos,
           "Malformed history is rejected before model work");
    auto item = function;
    item["type"] = "function_call";
    item["call_id"] = "call_1";
    gufo::tokenization::ChatMessage message;
    gufo::core::ImageReadBudget budget;
    std::string error;
    Expect(!gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                     &error) &&
               !error.empty(),
           "Responses rejects malformed history with an explanation");
  }
}

void TestQwenToolBoundariesAndSchema() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model", "messages":[{"role":"user","content":"use f"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"},
      "count":{"type":"integer"},"flag":{"type":"boolean"},
      "limit":{"type":["integer","null"]},"tags":{"type":"array"}}}}}]
  })");
  const std::string good =
      "<tool_call><function=f><parameter=text>42</parameter>"
      "<parameter=count>42</parameter></function></tool_call>";
  struct Case {
    std::string text;
    std::size_t calls;
    std::string argument;
  };
  for (const auto& item :
       {Case{good, 1, R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>literal </tool_call> "
             "and </function></parameter></function></tool_call>",
             1, R"({"text":"literal </tool_call> and </function>"})"},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>oops</parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>42</function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>ok</parameter>"
             "<parameter=count>42</function></tool_call>" +
                 good,
             1, R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>unclosed" + good, 1,
             R"({"text":"42","count":42})"},
        Case{"<tool_call><function=f><parameter=text>literal </think>"
             "</parameter></function></tool_call>",
             1, R"({"text":"literal </think>"})"},
        // A pipe-wrapped spelling that is not a vocabulary token is argument
        // data, not framing: nothing in the output made it a control token.
        Case{"<tool_call><function=f><parameter=text>\n<|not_a_vocab_entry|>\n"
             "</parameter></function></tool_call>",
             1, R"({"text":"<|not_a_vocab_entry|>"})"},
        // One newline on each side of a value is template framing; a file's
        // final newline, indentation and an empty value survive.
        Case{"<tool_call>\n<function=f>\n<parameter=text>\n  line 1\n"
             "line 2\n\n</parameter>\n<parameter=count>\n42\n</parameter>\n"
             "</function>\n</tool_call>",
             1, R"({"text":"  line 1\nline 2\n","count":42})"},
        Case{"<tool_call>\n<function=f>\n<parameter=text>\n\n</parameter>\n"
             "</function>\n</tool_call>",
             1, R"({"text":""})"},
        Case{"<tool_call>\r\n<function=f>\r\n<parameter=text>\r\n\r\nx\r\n"
             "</parameter>\r\n</function>\r\n</tool_call>",
             1, R"({"text":"\r\nx"})"},
        // Python literals where the schema wants JSON; a string keeps them.
        Case{"<tool_call><function=f><parameter=flag>\nTrue\n</parameter>"
             "<parameter=limit>None</parameter><parameter=tags>"
             "[False, \"None\", \"a \\\" True\"]</parameter>"
             "<parameter=text>True</parameter></function></tool_call>",
             1,
             R"({"flag":true,"limit":null,"tags":[false,"None","a \" True"],)"
             R"("text":"True"})"},
        Case{"<tool_call><function=f><parameter=flag>Yes</parameter>"
             "</function></tool_call>",
             0, ""},
        // A repeated parameter with the same value is dropped; a conflicting
        // repeat still rejects the call.
        Case{"<tool_call>\n<function=f>\n<parameter=text>\na\n</parameter>\n"
             "<parameter=count>42</parameter><parameter=flag>True</parameter>"
             "<parameter=text>a</parameter><parameter=count>\n42\n"
             "</parameter><parameter=flag>true</parameter>\n</function>\n"
             "</tool_call>",
             1, R"({"text":"a","count":42,"flag":true})"},
        Case{"<tool_call><function=f><parameter=text>a</parameter>"
             "<parameter=text>b</parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call><function=f><parameter=text>a</parameter>"
             "<parameter=text>a </parameter></function></tool_call>",
             0, ""},
        Case{"<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":"
             "\"literal </tool_call>\"}}</tool_call>",
             1, R"({"text":"literal </tool_call>"})"}}) {
    for (bool reasoning : {false, true}) {
      for (bool stream : {false, true}) {
        auto body = schema;
        body["stream"] = stream;
        body["chat_template_kwargs"] = Value::object();
        body["chat_template_kwargs"]["enable_thinking"] = reasoning;
        FakeBackend backend;
        const auto text =
            (reasoning ? "Considering. </think>" : "") + item.text;
        // Split every marker and argument across token callbacks.
        for (char c : text)
          backend.pieces.emplace_back(1, c);
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(response.status == 200, "tool boundary request succeeds");
        std::vector<Value> calls;
        if (!stream) {
          const auto output = gufo::json::parse(response.body);
          const auto& message =
              *output.find("choices")->items()[0].find("message");
          if (const auto* found = message.find("tool_calls"))
            calls.assign(found->items().begin(), found->items().end());
        } else {
          std::string output;
          response.streaming_body([&](std::string_view part) {
            output += part;
            return true;
          });
          std::size_t cursor = 0;
          while ((cursor = output.find("data: ", cursor)) !=
                 std::string::npos) {
            const auto begin = cursor + 6;
            cursor = output.find('\n', begin);
            const auto payload = output.substr(begin, cursor - begin);
            if (payload == "[DONE]")
              break;
            const auto event = gufo::json::parse(payload);
            const auto* choices = event.find("choices");
            if (!choices || choices->empty())
              continue;
            const auto* delta = choices->items()[0].find("delta");
            if (const auto* found = delta ? delta->find("tool_calls") : nullptr)
              calls.insert(calls.end(), found->items().begin(),
                           found->items().end());
            if (item.calls && delta)
              Expect(delta->member_str("reasoning_content").find('<') ==
                         std::string::npos,
                     "tool markers do not leak into reasoning deltas");
          }
        }
        if (calls.size() != item.calls)
          std::cerr << "Tool input: " << item.text
                    << "\nExpected calls: " << item.calls
                    << ", actual: " << calls.size() << '\n';
        Expect(calls.size() == item.calls, "only complete tool calls emitted");
        if (!calls.empty()) {
          if (calls.back().find("function")->member_str("arguments") !=
              item.argument)
            std::cerr << "Tool input: " << item.text
                      << "\nExpected: " << item.argument << "\nActual: "
                      << calls.back().find("function")->member_str("arguments")
                      << '\n';
          Expect(calls.back().find("function")->member_str("arguments") ==
                     item.argument,
                 "tool argument values and schema types preserved");
        }
      }
    }
  }
}

void TestToolChoiceEnforcement() {
  for (bool stream : {false, true}) {
    for (const auto* text :
         {"<tool_call><function=f></function></tool_call>",
          "<｜DSML｜tool_calls｜><｜DSML｜invoke "
          "name=\"f\"></｜DSML｜invoke></｜DSML｜tool_calls｜>"}) {
      for (const auto* choice : {"auto", "none", "required"}) {
        for (const auto* declared : {"", "f", "other"}) {
          auto body = gufo::json::parse(
              R"({"model":"test-model","messages":[{"role":"user","content":"use a tool"}]})");
          body["stream"] = stream;
          body["tool_choice"] = choice;
          if (*declared) {
            auto tool = gufo::json::parse(
                R"({"type":"function","function":{"name":"f","parameters":{"type":"object"}}})");
            tool["function"]["name"] = declared;
            body["tools"] = gufo::json::Value::array();
            body["tools"].push_back(std::move(tool));
          }
          FakeBackend backend;
          backend.pieces = {text};
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          const bool required = std::string_view(choice) == "required";
          if (required && !*declared) {
            Expect(response.status == 400, "required needs declared tools");
            continue;
          }
          std::string output = response.body;
          if (response.streaming_body)
            response.streaming_body([&](std::string_view part) {
              output += part;
              return true;
            });
          const bool allowed = std::string_view(declared) == "f" &&
                               std::string_view(choice) != "none";
          Expect(
              (output.find("\"tool_calls\":") != std::string::npos) == allowed,
              "only declared, enabled tool calls can enter API output");
          if (required && !allowed)
            Expect(
                output.find("tool_choice_unsatisfied") != std::string::npos &&
                    (stream || response.status == 502),
                "required cannot silently return text");
        }
      }
    }
  }
  FakeBackend backend;
  backend.pieces = {"ordinary text"};
  auto body = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"call f"}],"tool_choice":"required","tools":[{"type":"function","function":{"name":"f","parameters":{}}}]})");
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             502,
         "ordinary text cannot fulfill required tool choice");
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

void TestDeepSeekRepeatedToolParameters() {
  const auto call = [](std::string_view repeat) {
    return "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"read\">"
           "<｜DSML｜parameter name=\"path\" string=\"true\">/a"
           "</｜DSML｜parameter><｜DSML｜parameter name=\"path\" "
           "string=\"true\">" +
           std::string(repeat) +
           "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls｜>";
  };
  for (const auto& [repeat, accepted] :
       {std::pair{"/a", true}, std::pair{"\n/a\n", false},
        std::pair{"/b", false}}) {
    FakeBackend backend;
    backend.pieces = {call(repeat)};
    const auto response = gufo::server::HandleOpenAiChat(
        Request(
            R"({"model":"test-model","messages":[{"role":"user","content":"read"}],)"
            R"("tools":[{"type":"function","function":{"name":"read",)"
            R"("parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]})"),
        backend);
    Expect(response.status == 200, "DeepSeek repeated parameter request");
    Expect((response.body.find(R"("arguments":"{\"path\":\"/a\"}")") !=
            std::string::npos) == accepted,
           "identical DeepSeek repeats are dropped, conflicts rejected");
    Expect((response.body.find(R"("finish_reason":"tool_calls")") !=
            std::string::npos) == accepted,
           "a conflicting DeepSeek repeat is not a tool call");
  }
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

void TestModelSamplingDefaults() {
  using gufo::sampling::TextModelPreset;
  FakeBackend backend;
  backend.defaults.model = TextModelPreset::kQwen38;
  backend.defaults.supplied = {};
  auto constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  constraint->grammar = gufo::sampling::JsonConstraint::Object();
  constraint->vocabulary =
      std::make_shared<gufo::sampling::ConstraintVocabulary>(
          2, [](std::uint32_t id) {
            return gufo::sampling::ConstraintVocabulary::Piece{
                .text = id == 0 ? "{}" : "", .stop = id == 1};
          });
  backend.defaults.sampling.constraint = constraint;
  auto send = [&](std::string fields) {
    const auto response = gufo::server::HandleOpenAiChat(
        Request("{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hello\"}]" +
                fields + "}"),
        backend);
    Expect(response.status == 200, "Model default request accepted");
    Expect(backend.last_sampling.constraint ==
               backend.defaults.sampling.constraint,
           "Model presets preserve output constraints");
    return backend.last_sampling;
  };
  auto config = send("");
  Expect(config.temperature == 1.0F && config.top_p == 0.95F &&
             config.top_k == 20 && config.presence_penalty == 0.0F,
         "Qwen defaults follow thinking-on template");
  config = send(R"(,"chat_template_kwargs":{"enable_thinking":false})");
  Expect(config.temperature == 0.7F && config.top_p == 0.8F &&
             config.presence_penalty == 1.5F,
         "Request thinking-off selects its model preset");
  backend.reasoning_defaults_value.enabled = false;
  config = send(R"(,"temperature":null,"top_p":null,"presence_penalty":null)");
  Expect(config.temperature == 0.7F && config.presence_penalty == 1.5F,
         "Nulls inherit effective server thinking preset");
  config = send(R"(,"reasoning_effort":"high")");
  Expect(config.temperature == 1.0F && config.presence_penalty == 0.0F,
         "Effort enables thinking before sampling resolves");
  backend.defaults.sampling.temperature = 0.2F;
  backend.defaults.sampling.top_k = 0;
  backend.defaults.supplied.temperature = true;
  backend.defaults.supplied.top_k = true;
  config = send("");
  Expect(config.temperature == 0.2F && config.top_k == 0 &&
             config.top_p == 0.8F && config.presence_penalty == 1.5F,
         "Partial server override preserves other model defaults");
  config = send(R"(,"temperature":0,"presence_penalty":0,"top_k":40)");
  Expect(config.temperature == 0.0F && config.presence_penalty == 0.0F &&
             config.top_k == 40,
         "Explicit request zero overrides server and model presets");
  backend.defaults.model = TextModelPreset::kDeepSeekV4Flash;
  backend.defaults.supplied = {};
  for (const bool thinking : {false, true}) {
    backend.reasoning_defaults_value.enabled = thinking;
    config = send("");
    Expect(config.temperature == 1.0F && config.top_p == 0.95F &&
               config.top_k == 0 && config.min_p == 0.0F &&
               config.presence_penalty == 0.0F &&
               config.frequency_penalty == 0.0F &&
               config.repeat_penalty == 1.0F,
           "DeepSeek agentic defaults do not depend on reasoning");
  }
  backend.defaults.supplied = gufo::sampling::SamplingOverrides::All();
  config = send("");
  Expect(config.constraint == backend.defaults.sampling.constraint,
         "Explicit public-API configuration preserves its output constraint");
}

void TestCompleteToolDefinitionsReachTemplate() {
  FakeBackend backend;
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "messages":[{"role":"user","content":"Emit a value."}],
    "tools":[{"type":"function","function":{"parameters":{
      "type":"object","additionalProperties":false},"strict":true,
      "description":"","name":"emit"},
      "vendor":{"version":2}}]
  })"),
                                                       backend);
  Expect(response.status == 200 && backend.last_request.tools.size() == 1,
         "complete function definition reaches the backend");
  const auto& tool = backend.last_request.tools.front();
  Expect(
      tool.definition_json ==
          R"({"type":"function","function":{"parameters":{"type":"object","additionalProperties":false},"strict":true,"description":"","name":"emit"},"vendor":{"version":2}})",
      "Valid nested definitions retain every field and its original order");
  gufo::tokenization::ChatTemplateOptions options;
  options.enable_thinking = false;
  const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
      backend.last_request.messages, backend.last_request.tools, options);
  Expect(
      rendered.has_value() &&
          rendered->find(
              R"({"type": "function", "function": {"parameters": {"type": "object", "additionalProperties": false}, "strict": true, "description": "", "name": "emit"}, "vendor": {"version": 2}})") !=
              std::string::npos,
      "Existing valid tool JSON is unchanged in the Qwen prompt");
}

void TestFlatToolFieldsReachTemplate() {
  for (const char* strict : {"true", "false", "null"}) {
    FakeBackend backend;
    auto body = gufo::json::parse(R"({
      "model":"test-model","messages":[{"role":"user","content":"use f"}],
      "tools":[]
    })");
    auto function = gufo::json::parse(R"({
      "name":"f","description":"","parameters":{"type":"object","additionalProperties":false},
      "strict":null,"vendor":{"version":2}
    })");
    function["strict"] = gufo::json::parse(strict);
    auto flat = function;
    flat["type"] = "function";
    body["tools"].push_back(flat);
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 && backend.last_request.tools.size() == 1,
           "Flat function with additional fields is accepted");
    const auto flat_tools = backend.last_request.tools;
    const auto definition = gufo::json::parse(flat_tools[0].definition_json);
    const auto* nested = definition.find("function");
    Expect(nested && nested->dump() == function.dump() &&
               !definition.contains("strict") && !definition.contains("vendor"),
           "All flat function fields survive DS4's nested function extraction");
    gufo::tokenization::ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto flat_prompt = gufo::tokenization::QwenChatTemplate::Render(
        backend.last_request.messages, flat_tools, options);
    auto canonical = gufo::json::Value::object();
    canonical["type"] = "function";
    canonical["function"] = function;
    body["tools"] = gufo::json::Value::array();
    body["tools"].push_back(canonical);
    const auto nested_response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(nested_response.status == 200, "Nested equivalent is accepted");
    const auto nested_prompt = gufo::tokenization::QwenChatTemplate::Render(
        backend.last_request.messages, backend.last_request.tools, options);
    Expect(flat_prompt.has_value() && nested_prompt == flat_prompt,
           "Flat and nested tools produce identical Qwen prompts");
  }
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
  for (const auto& [field, value] :
       {std::pair{"temperature", 2.01}, std::pair{"presence_penalty", 2.01},
        std::pair{"frequency_penalty", -2.01}}) {
    for (const bool stream : {false, true}) {
      auto request = Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
      auto body = gufo::json::parse(request.body);
      body[field] = value;
      body["stream"] = stream;
      request.body = body.dump();
      const auto response = gufo::server::HandleOpenAiChat(request, backend);
      Expect(response.status == 400 &&
                 response.body.find(std::string("invalid_") + field) !=
                     std::string::npos,
             "out-of-range OpenAI sampling fails before streaming");
    }
  }
  for (const double choices : {0.0, 1.4, 2.0, 1e100}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
    auto body = gufo::json::parse(request.body);
    body["n"] = choices;
    request.body = body.dump();
    Expect(gufo::server::HandleOpenAiChat(request, backend).status == 400,
           "n must be exactly one, without truncation or overflow");
  }
  for (const auto* field : {"draft_temperature",  "temperature_draft",
                            "draft_top_k",        "draft_top_p",
                            "draft_min_p",        "draft_seed",
                            "draft_policy",       "samplers",
                            "typical_p",          "tfs_z",
                            "mirostat",           "mirostat_eta",
                            "mirostat_tau",       "dynatemp_range",
                            "dynatemp_exponent",  "xtc_probability",
                            "xtc_threshold",      "dry_multiplier",
                            "dry_base",           "dry_allowed_length",
                            "dry_penalty_last_n", "dry_sequence_breakers",
                            "top_n_sigma",        "logit_bias"}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
    auto body = gufo::json::parse(request.body);
    body[field] = 0.8;
    request.body = body.dump();
    const auto response = gufo::server::HandleOpenAiChat(request, backend);
    Expect(response.status == 400 &&
               response.body.find("unsupported_sampling") != std::string::npos,
           "unsupported sampling must not be silently ignored");
  }
}

void TestSamplingRanges() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  for (const auto* field : {"temperature", "top_p", "min_p",
                            "frequency_penalty", "presence_penalty"}) {
    const bool penalty = std::string_view(field).ends_with("penalty");
    const double minimum = penalty ? -2 : 0;
    const double maximum =
        penalty || std::string_view(field) == "temperature" ? 2 : 1;
    for (const auto value :
         {minimum - 1e-8, minimum, maximum, maximum + 1e-8}) {
      auto request = Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
      auto body = gufo::json::parse(request.body);
      body[field] = value;
      request.body = body.dump();
      const auto response = gufo::server::HandleOpenAiChat(request, backend);
      Expect(
          response.status == (value < minimum || value > maximum ? 400 : 200),
          "sampling range is inclusive and checked before float rounding");
    }
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

  for (const auto effort : {"low", "medium", "xhigh"}) {
    const auto native = gufo::server::HandleOpenAiChat(
        Request("{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hello\"}],\"chat_template_kwargs\":{"
                "\"reasoning_effort\":\"" +
                std::string(effort) + "\"}}"),
        backend);
    Expect(
        native.status == 200 && backend.last_request.reasoning.enabled == true,
        "Native effort alone enables thinking");
  }
  backend.pieces = {"Done."};
  const auto disabled = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low"}})"),
      backend);
  Expect(
      disabled.status == 200 && backend.last_request.reasoning.enabled == false,
      "Template enable_thinking=false suppresses the configured effort");
  Expect(disabled.body.find(R"("content":"Done.")") != std::string::npos,
         "Explicit thinking-off produces visible content");
  const auto vision_ids = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"add_vision_id":true}})"),
      backend);
  Expect(vision_ids.status == 200 && backend.last_request.add_vision_id,
         "Official add_vision_id option reaches the model formatter");
  const auto invalid_ids = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
                  "chat_template_kwargs":{"add_vision_id":"true"}})"),
      backend);
  Expect(invalid_ids.status == 400, "add_vision_id requires a boolean");
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

void TestInitialOutputPhases() {
  using gufo::json::Value;
  struct Case {
    std::string text;
    bool automatic{false};
  };
  for (const auto& item :
       {Case{"<think>literal example</think>"},
        Case{" \n<think>literal example</think>\nanswer"},
        Case{"<think>unfinished example"},
        Case{"<think>Check carefully.</think>\nAnswer.", true}}) {
    for (const bool stream : {false, true}) {
      FakeBackend backend;
      // Disabled thinking starts in the content phase. Tags requested as
      // literal output remain data. Automatic detection still recognizes an
      // initial reasoning block, including across token boundaries.
      if (item.automatic)
        backend.initial_output_state_override =
            FakeBackend::InitialOutputState::kAuto;
      for (const char byte : item.text)
        backend.pieces.emplace_back(1, byte);
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[]
      })");
      if (!item.automatic)
        body["reasoning_effort"] = "none";
      auto message = Value::object();
      message["role"] = "user";
      message["content"] =
          item.automatic
              ? "Answer carefully."
              : "Copy this XML exactly, without code fences: " + item.text;
      body["messages"].push_back(std::move(message));
      body["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "output-phase request succeeds");
      std::string content, reasoning;
      const auto collect = [&](const Value& message) {
        content += message.member_str("content");
        reasoning += message.member_str("reasoning_content");
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]")) {
            const auto event = gufo::json::parse(payload);
            for (const auto& choice : event.find("choices")->items())
              collect(*choice.find("delta"));
          }
          return true;
        });
      } else {
        const auto output = gufo::json::parse(response.body);
        collect(*output.find("choices")->items()[0].find("message"));
      }
      Expect(content == (item.automatic ? "Answer." : item.text) &&
                 reasoning == (item.automatic ? "Check carefully." : ""),
             "initial output phase controls reasoning tag interpretation");
    }
  }
}

void TestReasoningAnswerSeparator() {
  using gufo::json::Value;
  for (const bool thinking : {false, true})
    for (const bool stream : {false, true})
      for (const bool split : {false, true})
        for (const int mode : {0, 1, 2})
          for (const std::string separator :
               {"", "\n\n", "\r\n\r\n", " \t\n\n\f\v"})
            for (const bool truncated : {false, true}) {
              const std::string answer =
                  truncated   ? ""
                  : mode == 2 ? R"({"answer":9,"text":"\n\nliteral"})"
                              : "9\n\nNext paragraph.\n    Indented line.";
              const std::string raw =
                  (thinking ? "Check</think>" : "") + separator + answer;
              FakeBackend backend;
              if (truncated)
                backend.finish_reason = FakeBackend::FinishReason::kLength;
              backend.tool_format =
                  gufo::sampling::JsonConstraint::ToolFormat::kQwen;
              if (split) {
                for (const char byte : raw)
                  backend.pieces.emplace_back(1, byte);
              } else {
                backend.pieces = {raw};
              }
              auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"4+5?"}],
            "temperature":0})");
              body["stream"] = stream;
              body["reasoning_effort"] = thinking ? "low" : "none";
              if (mode == 1) {
                body["tools"] =
                    gufo::json::parse(R"([{"type":"function","function":{
              "name":"unused","parameters":{"type":"object","properties":{},
              "additionalProperties":false}}}])");
                body["tool_choice"] = "auto";
              } else if (mode == 2) {
                body["response_format"] = gufo::json::parse(R"({
              "type":"json_schema","json_schema":{"name":"answer","strict":true,
              "schema":{"type":"object","properties":{"answer":{"type":"integer"},
              "text":{"type":"string"}},
              "required":["answer","text"],"additionalProperties":false}}})");
              }
              const auto response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
              Expect(response.status == 200,
                     "reasoning separator request succeeds");
              std::string content;
              if (stream) {
                response.streaming_body([&](std::string_view piece) {
                  const auto payload = piece.substr(piece.find("data: ") + 6);
                  if (!payload.starts_with("[DONE]")) {
                    const auto event = gufo::json::parse(payload);
                    for (const auto& choice : event.find("choices")->items())
                      if (const auto* delta = choice.find("delta"))
                        content += delta->member_str("content");
                  }
                  return true;
                });
              } else {
                const auto parsed = gufo::json::parse(response.body);
                content = parsed.find("choices")
                              ->items()[0]
                              .find("message")
                              ->member_str("content");
              }
              const auto expected = thinking ? answer : separator + answer;
              if (content != expected)
                std::cerr << "separator thinking=" << thinking
                          << " stream=" << stream << " split=" << split
                          << " mode=" << mode << " truncated=" << truncated
                          << " content=" << Value(content).dump() << "\n";
              Expect(content == expected,
                     "only the reasoning separator is removed; answer "
                     "whitespace survives");
            }
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
  for (const auto* header : {"pi-agent-2", "contains spaces", "rotated"}) {
    auto request = Request(
        R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})",
        {{"X-Client-ID", header}});
    request.client_id = "192.0.2.7";
    const auto response = gufo::server::HandleOpenAiChat(request, backend);
    Expect(response.status == 200 &&
               backend.last_request.client_id == request.client_id,
           "untrusted headers cannot change the transport identity");
  }
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

void TestImagePartsRetainOrderAndIdentity() {
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "stop":["END"],
    "messages":[{"role":"user","content":[
      {"type":"text","text":"left"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"auto"}},
      {"type":"text","text":"right"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,BAUG"}}
    ]}]
  })"),
                                                       backend);
  Expect(response.status == 200, "image content parts reach the backend");
  Expect(backend.last_request.stop_sequences == std::vector<std::string>{"END"},
         "image requests preserve explicit stop sequences");
  const auto& message = backend.last_request.messages.front();
  Expect(message.content == "leftright" && message.images.size() == 2,
         "images do not become text placeholders before model preparation");
  Expect(message.images[0].offset == 4 && message.images[1].offset == 9,
         "image placement relative to text is preserved");
  Expect(*message.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}) &&
             *message.images[1].bytes == std::vector<std::uint8_t>({4, 5, 6}),
         "each image retains its own decoded transport bytes");
  for (
      const auto* body :
      {R"({"messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]})",
       R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID","detail":"low"}}]}]})",
       R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///tmp/image.png"}}]}]})"}) {
    Expect(gufo::server::HandleOpenAiChat(Request(body), backend).status == 400,
           "unsupported image role, policy and URL are rejected");
  }
}

void TestImageCountAndByteBudget() {
  const auto message = gufo::json::parse(R"({"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]})");
  for (const unsigned count : {17u, 257u}) {
    for (const bool split : {false, true}) {
      FakeBackend backend;
      auto body = gufo::json::Value::object();
      body["model"] = "test-model";
      body["messages"] = gufo::json::Value::array();
      if (split) {
        for (unsigned i = 0; i < count; ++i)
          body["messages"].push_back(message);
      } else {
        auto combined = message;
        for (unsigned i = 1; i < count; ++i)
          combined["content"].push_back(message.find("content")->items()[0]);
        body["messages"].push_back(std::move(combined));
      }
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      std::size_t observed = 0;
      for (const auto& parsed : backend.last_request.messages)
        observed += parsed.images.size();
      Expect(response.status == 200 && observed == count,
             "all image parts reach the model within the byte budget");
    }
  }
  const auto response_message = gufo::json::parse(R"({
    "role":"user","content":[{"type":"input_image",
    "image_url":"data:image/png;base64,AQID"}]})");
  gufo::core::ImageReadBudget budget;
  budget.remaining_bytes = 17 * 3;
  std::string error;
  for (unsigned i = 0; i < 17; ++i) {
    gufo::tokenization::ChatMessage parsed;
    Expect(gufo::server::ParseOpenAiResponseMessage(response_message, &parsed,
                                                    budget, &error) &&
               parsed.images.size() == 1,
           "Responses accepts more than sixteen images across messages");
  }
  gufo::tokenization::ChatMessage parsed;
  Expect(!gufo::server::ParseOpenAiResponseMessage(response_message, &parsed,
                                                   budget, &error) &&
             error.find("byte") != std::string::npos,
         "aggregate byte protection still covers every message");
}

void TestToolImagesRetainOrderAndIdentity() {
  using gufo::json::parse;
  using gufo::tokenization::ChatRole;
  FakeBackend backend;
  backend.pieces = {"ok"};
  const auto response = gufo::server::HandleOpenAiChat(Request(R"({
    "model":"test-model",
    "messages":[{"role":"tool","tool_call_id":"call_picture","content":[
      {"type":"text","text":"left "},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},
      {"type":"text","text":" right"},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,BAUG"}}
    ]}]
  })"),
                                                       backend);
  Expect(response.status == 200, "Chat tool images reach the backend");
  const auto& chat = backend.last_request.messages.front();
  Expect(chat.role == ChatRole::kTool && chat.tool_call_id == "call_picture" &&
             chat.content == "left  right" && chat.images.size() == 2 &&
             chat.images[0].offset == 5 && chat.images[1].offset == 11,
         "Chat images stay attached to the tool result at their text offsets");

  for (const auto* kind : {"function_call_output", "custom_tool_call_output"}) {
    for (const bool text : {false, true}) {
      auto item = parse(R"({"call_id":"call_picture","output":[
        {"type":"input_image","image_url":"data:image/png;base64,AQID"}
      ]})");
      item["type"] = kind;
      if (text)
        item["output"] = parse(R"([
          {"type":"input_text","text":"left "},
          {"type":"input_image","image_url":"data:image/png;base64,AQID"},
          {"type":"input_text","text":" right"}
        ])");
      gufo::core::ImageReadBudget budget;
      budget.remaining_bytes = 3;
      gufo::tokenization::ChatMessage message;
      std::string error;
      Expect(gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                      &error),
             "Responses accepts image-bearing tool outputs: " + error);
      Expect(
          message.role == ChatRole::kTool &&
              message.tool_call_id == "call_picture" &&
              message.content == (text ? "left  right" : "") &&
              message.images.size() == 1 &&
              message.images[0].offset == (text ? 5 : 0) &&
              *message.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}),
          "Responses retains call association, image bytes and order");
      gufo::tokenization::ChatMessage overflow;
      Expect(!gufo::server::ParseOpenAiResponseMessage(item, &overflow, budget,
                                                       &error),
             "tool output images consume the shared request byte budget");
    }
    auto literal = parse(R"({"call_id":"literal",
      "output":"data:image/png;base64,AQID"})");
    literal["type"] = kind;
    gufo::core::ImageReadBudget budget;
    budget.remaining_bytes = 0;
    gufo::tokenization::ChatMessage message;
    std::string error;
    Expect(gufo::server::ParseOpenAiResponseMessage(literal, &message, budget,
                                                    &error) &&
               message.images.empty() &&
               message.content == "data:image/png;base64,AQID",
           "string tool output stays literal text even when it spells an image "
           "URL");
  }
  for (const auto* role : {"assistant", "system", "developer"}) {
    auto message = parse(R"({"content":[
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}
    ]})");
    message["role"] = role;
    auto body = gufo::json::Value::object();
    body["model"] = "test-model";
    body["messages"] = gufo::json::Value::array();
    body["messages"].push_back(std::move(message));
    Expect(
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
            400,
        "images remain rejected on non-observation roles");
  }
  for (const auto* image :
       {R"({"url":17})",
        R"({"url":"data:image/png;base64,AQID","detail":"high"})",
        R"({"url":"file:///tmp/image.png"})",
        R"({"url":"data:image/png;base64,A==="})"}) {
    auto part = parse(R"({"type":"image_url"})");
    part["image_url"] = parse(image);
    auto message = parse(R"({"role":"tool","content":[]})");
    message["content"].push_back(std::move(part));
    auto body = parse(R"({"messages":[]})");
    body["model"] = "test-model";
    body["messages"].push_back(std::move(message));
    Expect(
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
            400,
        "tool images preserve URL, encoding and resolution validation");
  }
}

void TestStopSequencesAndDefaultFields() {
  using gufo::json::Value;
  const auto base = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"hello"}]})");
  for (const auto* stop : {"null", "[]", "\"END\"", "[\"END\",\"終\"]"}) {
    auto body = base;
    body["stop"] = gufo::json::parse(stop);
    FakeBackend backend;
    backend.pieces = {"hello"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "supported stop forms are accepted");
    const auto& expected = body["stop"];
    const std::size_t count = expected.is_null()     ? 0
                              : expected.is_string() ? 1
                                                     : expected.items().size();
    Expect(backend.last_request.stop_sequences.size() == count,
           "stop sequences reach the backend");
    if (count)
      Expect(backend.last_request.stop_sequences.front() == "END",
             "stop string bytes are preserved");
  }
  for (const bool stream : {false, true}) {
    for (const auto* stop :
         {"true", "123", "{}", "\"\"", "[\"\"]", "[\"END\",null]",
          "[\"1\",\"2\",\"3\",\"4\",\"5\"]"}) {
      auto body = base;
      body["stream"] = stream;
      body["stop"] = gufo::json::parse(stop);
      FakeBackend backend;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 400 && backend.chat_calls == 0 &&
                 !response.streaming_body,
             "invalid stops fail before generation or streaming headers");
    }
  }
  auto body = base;
  body["stop"] = std::string(4097, 's');
  FakeBackend backend;
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             400,
         "oversized stop sequences are bounded");
  backend.defaults.max_tokens = 37;
  backend.defaults.sampling.temperature = 0.7F;
  backend.defaults.sampling.top_p = 0.9F;
  backend.defaults.sampling.seed = 123;
  backend.defaults.sampling.frequency_penalty = 0.5F;
  backend.defaults.sampling.presence_penalty = 0.3F;
  body = base;
  for (const auto* field :
       {"tools", "tool_choice", "stream", "stream_options", "n", "max_tokens",
        "max_completion_tokens", "logprobs", "top_logprobs", "response_format",
        "modalities", "audio", "temperature", "top_p", "seed", "logit_bias",
        "frequency_penalty", "presence_penalty", "reasoning_effort"})
    body[field] = Value();
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "nullable API defaults do not enable unsupported features");
  Expect(backend.last_max_tokens == backend.defaults.max_tokens,
         "null token limits retain the configured default");
  Expect(backend.last_sampling.temperature ==
                 backend.defaults.sampling.temperature &&
             backend.last_sampling.top_p == backend.defaults.sampling.top_p &&
             backend.last_sampling.seed == backend.defaults.sampling.seed &&
             backend.last_sampling.frequency_penalty ==
                 backend.defaults.sampling.frequency_penalty &&
             backend.last_sampling.presence_penalty ==
                 backend.defaults.sampling.presence_penalty,
         "null sampling fields retain server defaults");
  body["logprobs"] = false;
  body["logit_bias"] = Value::object();
  body["response_format"] = gufo::json::parse(R"({"type":"text"})");
  body["modalities"] = gufo::json::parse(R"(["text"])");
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "explicit text-only and disabled logprobs defaults work");
  for (const auto* request :
       {R"({"logprobs":true})", R"({"top_logprobs":3})",
        R"({"modalities":["text","audio"]})", R"({"audio":{}})",
        R"({"response_format":{"type":"text","unexpected":true}})"}) {
    auto invalid = base;
    const auto fields = gufo::json::parse(request);
    for (const auto& [key, value] : fields.members())
      invalid[key] = value;
    const auto before = backend.chat_calls.load();
    Expect(gufo::server::HandleOpenAiChat(Request(invalid.dump()), backend)
                       .status == 400 &&
               backend.chat_calls == before,
           "actual unsupported feature requests remain explicit errors");
  }
}

void TestStructuredResponseFormat() {
  FakeBackend backend;
  backend.reasoning_defaults_value.enabled = true;
  auto body = gufo::json::parse(
      R"({"model":"test-model", "messages":[{"role":"user","content":"Return a value"}]})");
  for (
      const char* format :
      {R"({"type":"json_object"})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}})"}) {
    body["response_format"] = gufo::json::parse(format);
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 && backend.last_request.response_format,
           "compiled format reaches backend");
    Expect(backend.last_request.reasoning.enabled == true,
           "structured output retains the model reasoning default");
    for (
        const char* incompatible :
        {R"({"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})"}) {
      auto request = body;
      const auto fields = gufo::json::parse(incompatible);
      for (const auto& [key, value] : fields.members())
        request[key] = value;
      const auto accepted =
          gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
      Expect(accepted.status == 200 && backend.last_request.tools.size() == 1,
             "structured requests keep their active tools");
    }
  }
  body["response_format"]["json_schema"]["description"] =
      "Use the requested labels.";
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
                 200 &&
             backend.last_request.response_format_description ==
                 "Use the requested labels.",
         "format description reaches the model request");
  const std::string literal =
      R"({"text":"<think>literal</think> <tool_call>literal</tool_call> ┌"})";
  backend.pieces = {literal.substr(0, literal.size() - 4),
                    literal.substr(literal.size() - 4, 1),
                    literal.substr(literal.size() - 3)};
  body["response_format"] = gufo::json::parse(R"({"type":"json_object"})");
  body["reasoning_effort"] = "none";
  const auto buffered =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(gufo::json::parse(buffered.body)
                 .find("choices")
                 ->items()[0]
                 .find("message")
                 ->member_str("content") == literal,
         "JSON strings do not activate output protocol markers");
  body["stream"] = true;
  const auto streamed =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  std::string content;
  streamed.streaming_body([&](std::string_view chunk) {
    if (chunk == "data: [DONE]\n\n")
      return true;
    const auto event = gufo::json::parse(chunk.substr(6));
    for (const auto& choice : event.find("choices")->items()) {
      const auto* delta = choice.find("delta");
      if (delta) {
        Expect(!delta->contains("reasoning_content") &&
                   !delta->contains("tool_calls"),
               "JSON literals remain ordinary content in SSE");
        content += delta->member_str("content");
      }
    }
    return true;
  });
  Expect(content == literal,
         "streaming retains JSON bytes and UTF-8 boundaries");
  for (const bool stream : {false, true}) {
    body["stream"] = stream;
    body["reasoning_effort"] = "high";
    backend.pieces = {"Consider <tool_call> as text.", "</thi", "nk>", literal};
    const auto reply =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(reply.status == 200, "thinking and structured output coexist");
    std::string answer, reasoning;
    if (stream) {
      reply.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items()) {
          if (const auto* delta = choice.find("delta")) {
            answer += delta->member_str("content");
            reasoning += delta->member_str("reasoning_content");
          }
        }
        return true;
      });
    } else {
      const auto response = gufo::json::parse(reply.body);
      const auto* message =
          response.find("choices")->items()[0].find("message");
      answer = message->member_str("content");
      reasoning = message->member_str("reasoning_content");
    }
    Expect(answer == literal && reasoning == "Consider <tool_call> as text.",
           "only the reasoning delimiter changes structured-output phase");
  }
  body["stop"] = "END";
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             200,
         "structured requests accept explicit stops");
  body = gufo::json::parse(
      R"({"model":"test-model","messages":[{"role":"user","content":"value"}]})");
  for (
      const char* invalid :
      {R"({"type":"unknown"})",
       R"({"type":"json_schema","json_schema":{"name":"bad name","schema":{}}})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","strict":1,"schema":{}}})",
       R"({"type":"json_schema","json_schema":{"name":"Reply","schema":{"type":"object","properties":{},"additionalProperties":false,"not":{}}}})"}) {
    body["response_format"] = gufo::json::parse(invalid);
    body["stream"] = true;
    const auto calls = backend.chat_calls.load();
    const auto rejected =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(
        rejected.status == 400 && !rejected.streaming_body &&
            backend.chat_calls == calls &&
            rejected.body.find("invalid_response_format") != std::string::npos,
        "invalid schema fails before stream headers or admission");
  }
}

void TestExplicitStopOutputFraming() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool stream : {false, true}) {
    for (const bool thinking : {false, true}) {
      FakeBackend backend;
      // The scheduler already removed the stop marker. Test protocol framing
      // independently, including a required tool interrupted before its call.
      backend.pieces = {"safe"};
      backend.finish_reason = Finish::kStopSequence;
      backend.stop_sequence = "END";
      auto body = gufo::json::parse(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],
              "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
              "tool_choice":"required","stop":"END"})");
      body["stream"] = stream;
      body["chat_template_kwargs"] = gufo::json::Value::object();
      body["chat_template_kwargs"]["enable_thinking"] = thinking;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "explicit stop completes successfully");
      std::string output = response.body;
      if (response.streaming_body)
        response.streaming_body([&](std::string_view piece) {
          output += piece;
          return true;
        });
      Expect(output.find("\"finish_reason\":\"stop\"") != std::string::npos &&
                 output.find("tool_choice_unsatisfied") == std::string::npos &&
                 output.find("END") == std::string::npos,
             "explicit stop wins over required tools and hides its marker");
      Expect(
          output.find(thinking ? "\"reasoning_content\":\"safe\""
                               : "\"content\":\"safe\"") != std::string::npos,
          "stopped text retains reasoning/content framing");
    }
  }
}

void TestStructuredToolTruncation() {
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{}}</tool_call>)";
  for (const bool stream : {false, true}) {
    for (std::size_t length = 0; length <= call.size(); ++length) {
      FakeBackend backend;
      backend.pieces = {call.substr(0, length)};
      backend.finish_reason =
          gufo::server::TextGenerationBackend::FinishReason::kLength;
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "response_format":{"type":"json_object"},"reasoning_effort":"none",
        "tool_choice":"required",
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]
      })");
      body["stream"] = stream;
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200,
             "truncated constrained tool is a length result");
      if (stream) {
        std::string output;
        response.streaming_body([&](std::string_view chunk) {
          output += chunk;
          return true;
        });
        Expect(
            output.find("\"finish_reason\":\"length\"") != std::string::npos &&
                output.find("\"error\"") == std::string::npos &&
                output.find("<tool") == std::string::npos,
            "partial tool delimiters are not exposed as structured content");
      } else {
        const auto output = gufo::json::parse(response.body);
        const auto& choice = output.find("choices")->items()[0];
        Expect(choice.member_str("finish_reason") == "length" &&
                   choice.find("message")->member_str("content").empty() &&
                   choice.find("message")->contains("tool_calls") ==
                       (length == call.size()),
               "buffered truncated tools retain the length contract");
      }
    }
  }
}

void TestStrictToolSchema() {
  auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"call f"}],
    "reasoning_effort":"none","parallel_tool_calls":false,
    "tools":[{"type":"function","function":{"name":"f","strict":true,
      "parameters":{"type":"object","properties":{"text":{"type":"string"}},
        "required":["text"],"additionalProperties":false}}}]
  })");
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{"text":"<think>x</think></tool_call>"}}</tool_call>)";
  for (const bool stream : {false, true}) {
    FakeBackend backend;
    backend.pieces = {"Calling f. "};
    for (char byte : call)
      backend.pieces.emplace_back(1, byte);
    backend.pieces.emplace_back(" After f.");
    body["stream"] = stream;
    body["tool_choice"] =
        gufo::json::parse(R"({"type":"function","function":{"name":"f"}})");
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string output = response.body;
    std::string content;
    const auto collect = [&](const gufo::json::Value& event) {
      for (const auto& choice : event.find("choices")->items()) {
        const auto* message = choice.find(stream ? "delta" : "message");
        if (message)
          content += message->member_str("content");
      }
    };
    if (response.streaming_body)
      response.streaming_body([&](std::string_view chunk) {
        output += chunk;
        if (chunk != "data: [DONE]\n\n")
          collect(gufo::json::parse(chunk.substr(6)));
        return true;
      });
    else
      collect(gufo::json::parse(response.body));
    Expect(response.status == 200 && backend.last_request.constrained_tools &&
               !backend.last_request.response_format &&
               !backend.last_request.parallel_tool_calls &&
               backend.last_request.tool_choice ==
                   gufo::server::ChatRequest::ToolChoice::kRequired,
           "strict tool schemas and forced choice reach the runner "
           "independently of response_format");
    Expect(
        output.find("\"finish_reason\":\"tool_calls\"") != std::string::npos &&
            content == "Calling f.  After f." &&
            output.find("<think>x</think></tool_call>") != std::string::npos &&
            output.find("reasoning_content") == std::string::npos,
        "strict tool payload markers remain argument data in buffered and "
        "streaming responses");
  }
  body["tool_choice"] = "auto";
  body["parallel_tool_calls"] = true;
  for (const bool stream : {false, true}) {
    FakeBackend backend;
    const auto raw = " Before " + call + " between " + call + " after <";
    for (char byte : raw)
      backend.pieces.emplace_back(1, byte);
    body["stream"] = stream;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string content;
    if (stream) {
      response.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        return true;
      });
    } else {
      content = gufo::json::parse(response.body)
                    .find("choices")
                    ->items()[0]
                    .find("message")
                    ->member_str("content");
    }
    Expect(content == " Before  between  after <",
           "parallel strict calls preserve surrounding prose and whitespace");
  }
  body["parallel_tool_calls"] = false;
  for (
      const char* invalid :
      {R"({"tool_choice":{"type":"function","function":{"name":"absent"}}})",
       R"({"parallel_tool_calls":"false"})",
       R"({"tools":[{"type":"function","function":{"name":"bad<name","parameters":{}}}]})",
       R"({"tools":[{"type":"function","function":{"name":"f"}},{"type":"function","function":{"name":"f"}}]})",
       R"({"tools":[{"type":"function","function":{"name":"bad","strict":true,"parameters":{"type":"object"}}}]})"}) {
    FakeBackend backend;
    auto request = body;
    const auto fields = gufo::json::parse(invalid);
    for (const auto& [key, value] : fields.members())
      request[key] = value;
    Expect(gufo::server::HandleOpenAiChat(Request(request.dump()), backend)
                       .status == 400 &&
               backend.chat_calls == 0,
           "invalid tool constraints fail before generation");
  }
  body["tool_choice"] = "auto";
  for (const bool stream : {false, true}) {
    FakeBackend prose;
    prose.pieces = {"A literal ", "<"};
    body["stream"] = stream;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), prose);
    std::string content;
    if (response.streaming_body)
      response.streaming_body([&](std::string_view chunk) {
        if (chunk == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(chunk.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            content += delta->member_str("content");
        return true;
      });
    else
      content = gufo::json::parse(response.body)
                    .find("choices")
                    ->items()[0]
                    .find("message")
                    ->member_str("content");
    Expect(
        response.status == 200 && content == "A literal <",
        "automatic strict tools preserve ordinary text ending with a partial "
        "marker");
  }
  for (std::size_t length = 1; length < call.size(); ++length) {
    FakeBackend backend;
    backend.pieces = {"Calling f. ", call.substr(0, length)};
    backend.finish_reason =
        gufo::server::TextGenerationBackend::FinishReason::kLength;
    body["stream"] = false;
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200 &&
               response.body.find("\"finish_reason\":\"length\"") !=
                   std::string::npos &&
               response.body.find("\"tool_calls\"") == std::string::npos,
           "incomplete strict calls never become API tool calls");
  }
  FakeBackend no_arguments;
  no_arguments.pieces = {
      R"(<tool_call>{"name":"f","arguments":{}}</tool_call>)"};
  body["tools"] = gufo::json::parse(
      R"([{"type":"function","function":{"name":"f","strict":true}}])");
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), no_arguments);
  Expect(response.status == 200 && no_arguments.last_request.constrained_tools,
         "omitted parameters define a strict function without arguments");
  body["tools"] = gufo::json::parse(
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object"},"strict":false}}])");
  body["parallel_tool_calls"] = false;
  FakeBackend non_strict;
  non_strict.pieces = no_arguments.pieces;
  Expect(
      gufo::server::HandleOpenAiChat(Request(body.dump()), non_strict).status ==
              200 &&
          non_strict.last_request.constrained_tools,
      "parallel_tool_calls false also constrains non-strict tools");
}

void TestStopInsideToolArguments() {
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const bool stream : {false, true}) {
    for (const bool thinking : {false, true}) {
      for (const auto* partial :
           {"<tool_call><function=f><parameter=text>partial-",
            "<｜DSML｜tool_calls｜><｜DSML｜invoke name=\"f\">"
            "<｜DSML｜parameter name=\"text\" string=\"true\">partial-"}) {
        for (const bool earlier_call : {false, true}) {
          FakeBackend backend;
          backend.finish_reason = Finish::kStopSequence;
          backend.stop_sequence = "STOP";
          const std::string raw =
              std::string(earlier_call
                              ? "<tool_call><function=f></function></tool_call>"
                              : "") +
              partial;
          backend.pieces = {"safe"};
          // Constrained reasoning must close before a tool can start.
          if (thinking)
            backend.pieces.emplace_back("</think>");
          // Exercise splits inside XML delimiters and multibyte DSML tags.
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          auto body = gufo::json::parse(
              R"({"model":"test-model","messages":[{"role":"user","content":"call f"}],
                  "tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
                  "tool_choice":"required","stop":"STOP"})");
          body["stream"] = stream;
          body["chat_template_kwargs"] = gufo::json::Value::object();
          body["chat_template_kwargs"]["enable_thinking"] = thinking;
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200,
                 "a stop inside a tool argument succeeds");
          std::string output = response.body;
          if (response.streaming_body)
            response.streaming_body([&](std::string_view piece) {
              output += piece;
              return true;
            });
          Expect(output.find("partial-") == std::string::npos &&
                     output.find("<tool_call>") == std::string::npos &&
                     output.find("DSML") == std::string::npos,
                 "unfinished tool markup is never emitted as ordinary content");
          Expect((output.find("\"tool_calls\":") != std::string::npos) ==
                     earlier_call,
                 "only calls completed before the stop can be emitted");
          Expect(
              output.find("\"finish_reason\":\"stop\"") != std::string::npos &&
                  output.find(thinking ? "\"reasoning_content\":\"safe\""
                                       : "\"content\":\"safe\"") !=
                      std::string::npos,
              "preceding text and reasoning survive interrupted calls");
        }
      }
    }
  }
}

void TestNativeToolImplicitReasoningEnd() {
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const bool qwen = format == Format::kQwen;
    const std::string call =
        qwen ? "<tool_call>\n<function=read>\n<parameter=path>\nfile.txt\n"
               "</parameter>\n</function>\n</tool_call>"
             : "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n"
               "<｜DSML｜parameter name=\"path\" string=\"true\">file.txt"
               "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    for (const bool responses : {false, true})
      for (const bool stream : {false, true})
        for (const bool quoted : {false, true})
          for (const bool interrupted : {false, true}) {
            const std::string thought = quoted ? "Example:\n```xml\n" + call +
                                                     "\n```\nRead the file.\n\n"
                                               : "Read the file.\n\n";
            const std::string text =
                thought + (quoted ? "</think>" : "") +
                (interrupted ? call.substr(0, call.find("file.txt") + 4)
                             : call);
            // llama.cpp treats the two newlines before a DeepSeek call as
            // TC_SEPARATOR, including when the model omitted </think>.
            const std::string expected_thought =
                qwen || quoted ? thought
                               : thought.substr(0, thought.size() - 2);
            auto body = gufo::json::parse(R"({
              "model":"test-model","messages":[{"role":"user","content":"read the file"}],
              "reasoning_effort":"low","tool_choice":"required","parallel_tool_calls":false,
              "tools":[{"type":"function","function":{"name":"read","parameters":{
                "type":"object","properties":{"path":{"type":"string"}},
                "required":["path"],"additionalProperties":false}}}]})");
            body["stream"] = stream;
            FakeBackend backend;
            backend.tool_format = format;
            backend.finish_reason =
                interrupted ? Finish::kLength : Finish::kStop;
            for (const char byte : text)
              backend.pieces.emplace_back(1, byte);
            gufo::server::HttpResponse response;
            if (responses) {
              auto flat = *body["tools"].items()[0].find("function");
              flat["type"] = "function";
              body["tools"] = gufo::json::Value::array();
              body["tools"].push_back(std::move(flat));
              gufo::server::ChatRequest chat;
              chat.reasoning.enabled = true;
              Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                     "implicit tool fixture has valid Responses controls");
              response = gufo::server::CreateOpenAiResponse(
                  Request(body.dump()), backend, chat, 512, {}, stream);
            } else {
              response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
            }
            Expect(response.status == 200,
                   "implicit native tool request succeeds");
            gufo::json::Value result;
            std::string streamed_reasoning, streamed_content;
            bool streamed_tool = false;
            if (!stream)
              result = gufo::json::parse(response.body);
            else
              response.streaming_body([&](std::string_view piece) {
                if (piece == "data: [DONE]\n\n")
                  return true;
                const auto event =
                    gufo::json::parse(piece.substr(piece.find("data: ") + 6));
                if (responses) {
                  if (event.member_str("type") ==
                      "response.reasoning_summary_text.delta")
                    streamed_reasoning += event.member_str("delta");
                  if (event.member_str("type") == "response.output_text.delta")
                    streamed_content += event.member_str("delta");
                  if (event.member_str("type") == "response.completed" ||
                      event.member_str("type") == "response.incomplete")
                    result = *event.find("response");
                } else if (const auto* choices = event.find("choices")) {
                  for (const auto& choice : choices->items())
                    if (const auto* delta = choice.find("delta")) {
                      streamed_reasoning +=
                          delta->member_str("reasoning_content");
                      streamed_content += delta->member_str("content");
                      streamed_tool |= delta->contains("tool_calls");
                    }
                }
                return true;
              });
            std::string reasoning, content;
            std::size_t calls = 0;
            if (responses) {
              for (const auto& item : result.find("output")->items()) {
                calls += item.member_str("type") == "function_call";
                for (const auto* field : {"summary", "content"})
                  if (const auto* parts = item.find(field))
                    for (const auto& part : parts->items())
                      (std::string_view(field) == "summary" ? reasoning
                                                            : content) +=
                          part.member_str("text");
              }
            } else if (!stream) {
              const auto& message =
                  *result.find("choices")->items()[0].find("message");
              reasoning = message.member_str("reasoning_content");
              content = message.member_str("content");
              if (const auto* tools = message.find("tool_calls"))
                calls = tools->size();
            } else {
              reasoning = streamed_reasoning;
              content = streamed_content;
              calls = streamed_tool;
            }
            Expect(calls == (interrupted ? 0u : 1u),
                   "only a complete unquoted native call is invoked");
            Expect(content.empty() && reasoning == expected_thought,
                   "implicit call markup is not leaked into either output "
                   "channel");
            if (stream)
              Expect(streamed_content.empty() &&
                         streamed_reasoning == expected_thought,
                     "implicit reasoning boundary is correct during streaming");
          }
  }
}

void TestToolMarkersInsideConstrainedReasoning() {
  using gufo::json::Value;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const std::string reasoning =
      "Confirmed garbage on the docstring line. The broken line is:\n"
      "`    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{`\n"
      "A quoted <tool_call> is file data, not the end of reasoning.\n"
      "Also <｜DSML｜tool_calls> is quoted data. Preserve oldText exactly.";
  auto arguments = Value::object();
  arguments["path"] = "tests/unit/shared/test_vpc_block_public_access.py";
  arguments["edits"] = Value::array();
  auto replacement = Value::object();
  replacement["oldText"] =
      "    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"}}]}}</tool_call>\n\n<tool_call>('tool', '{";
  replacement["newText"] =
      "    \"\"\"The one BPA resource; find_resources maps logical id to a "
      "Mapping.\"\"\"";
  arguments["edits"].push_back(std::move(replacement));
  const std::string call =
      "<tool_call>\n<function=edit>\n<parameter=path>\n" +
      arguments.member_str("path") + "\n</parameter>\n<parameter=edits>\n" +
      arguments["edits"].dump() + "\n</parameter>\n</function>\n</tool_call>";
  const std::string prose = "\nVerify the file, then run mypy and tests.";
  // The reasoning separator is framing; newlines inside the edit and the
  // prose following its call remain data in both APIs and transports.
  const std::string complete = "</think>\n\n" + call + prose;
  for (const bool responses : {false, true})
    for (const bool stream : {false, true})
      for (const bool bytewise : {false, true})
        for (const bool strict : {false, true})
          for (const std::string& tail :
               {complete, std::string{}, std::string{"</thi"}}) {
            const bool truncated = tail != complete;
            const std::string expected_reasoning =
                reasoning + (truncated ? tail : "");
            const std::string expected_content = truncated ? "" : prose;
            auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"fix the docstring"}],
            "chat_template_kwargs":{"enable_thinking":true},
            "parallel_tool_calls":false,"tool_choice":"auto",
            "tools":[{"type":"function","function":{"name":"edit",
              "parameters":{"type":"object","properties":{
                "path":{"type":"string"},"edits":{"type":"array","items":{
                  "type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},
                  "required":["oldText","newText"],"additionalProperties":false}}},
                "required":["path","edits"],"additionalProperties":false}}}]
          })");
            body["stream"] = stream;
            auto tool = body["tools"].items()[0];
            tool["function"]["strict"] = strict;
            body["tools"] = Value::array();
            body["tools"].push_back(std::move(tool));
            FakeBackend backend;
            backend.finish_reason = truncated ? Finish::kLength : Finish::kStop;
            const std::string raw = reasoning + tail;
            if (bytewise)
              for (char byte : raw)
                backend.pieces.emplace_back(1, byte);
            else
              backend.pieces = {raw};
            gufo::server::HttpResponse response;
            if (responses) {
              auto flat = *body["tools"].items()[0].find("function");
              flat["type"] = "function";
              body["tools"] = Value::array();
              body["tools"].push_back(std::move(flat));
              gufo::server::ChatRequest chat;
              chat.reasoning.enabled = true;
              Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                     "reasoning marker fixture has valid Responses controls");
              response = gufo::server::CreateOpenAiResponse(
                  Request(body.dump()), backend, chat, 4096, {}, stream);
            } else {
              response =
                  gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
            }
            Expect(response.status == 200, "reasoning marker fixture succeeds");
            std::vector<Value> events;
            if (stream)
              response.streaming_body([&](std::string_view chunk) {
                const auto pos = chunk.find("data: ");
                if (pos != std::string_view::npos &&
                    !chunk.substr(pos + 6).starts_with("[DONE]"))
                  events.push_back(gufo::json::parse(chunk.substr(pos + 6)));
                return true;
              });
            else
              events.push_back(gufo::json::parse(response.body));
            std::string content, thought, emitted_arguments;
            std::string streamed_content, streamed_thought;
            std::size_t calls = 0;
            bool terminal = false;
            for (const auto& event : events) {
              if (responses) {
                if (event.member_str("type") == "response.output_text.delta")
                  streamed_content += event.member_str("delta");
                if (event.member_str("type") ==
                    "response.reasoning_summary_text.delta")
                  streamed_thought += event.member_str("delta");
                // Terminal Responses items are authoritative in both modes.
                const auto* result = stream ? event.find("response") : &event;
                if (result && result->member_str("status") ==
                                  (truncated ? "incomplete" : "completed")) {
                  terminal = true;
                  for (const auto& item : result->find("output")->items()) {
                    if (item.member_str("type") == "function_call") {
                      ++calls;
                      emitted_arguments = item.member_str("arguments");
                    } else if (item.member_str("type") == "reasoning") {
                      for (const auto& part : item.find("summary")->items())
                        thought += part.member_str("text");
                    } else if (item.member_str("type") == "message") {
                      for (const auto& part : item.find("content")->items())
                        content += part.member_str("text");
                    }
                  }
                }
              } else if (const auto* choices = event.find("choices")) {
                for (const auto& choice : choices->items()) {
                  if (!choice.member_str("finish_reason").empty()) {
                    terminal = true;
                    Expect(choice.member_str("finish_reason") ==
                               (truncated ? "length" : "tool_calls"),
                           "reasoning limits retain the correct finish reason");
                  }
                  const auto* message =
                      choice.find(stream ? "delta" : "message");
                  if (!message)
                    continue;
                  thought += message->member_str("reasoning_content");
                  content += message->member_str("content");
                  if (const auto* tools = message->find("tool_calls"))
                    for (const auto& tool : tools->items()) {
                      ++calls;
                      emitted_arguments +=
                          tool.find("function")->member_str("arguments");
                    }
                }
              }
            }
            Expect(terminal, "reasoning marker fixture has a terminal result");
            Expect(
                thought == expected_reasoning,
                "literal tool openers cannot terminate constrained reasoning");
            Expect(
                content == expected_content,
                "reasoning text and closing think tags never leak as content");
            Expect(truncated
                       ? calls == 0 && emitted_arguments.empty()
                       : calls == 1 && emitted_arguments == arguments.dump(),
                   "the edit preserves exact file data without protocol "
                   "pollution");
            if (responses && stream)
              Expect(
                  streamed_content == expected_content &&
                      streamed_thought == expected_reasoning,
                  "Responses deltas agree with the buffered phase boundaries");
          }
}

void TestToolMarkersWhenToolsDisabled() {
  using gufo::json::Value;
  const std::string thought =
      "Quoted <tool_call> and <｜DSML｜tool_calls> are reasoning data.";
  const std::string answer =
      "Literal <tool_call><function=f></function></tool_call> end.";
  for (const bool responses : {false, true})
    for (const bool stream : {false, true})
      for (const bool declared : {false, true})
        for (const int phase : {0, 1, 2}) {
          auto body = gufo::json::parse(R"({
            "model":"test-model","messages":[{"role":"user","content":"Explain syntax"}],
            "tool_choice":"none"})");
          body["stream"] = stream;
          body["chat_template_kwargs"]["enable_thinking"] = phase != 0;
          if (declared)
            body["tools"] =
                gufo::json::parse(R"([{"type":"function","function":{
              "name":"f","parameters":{"type":"object","properties":{},"additionalProperties":false}}}])");
          const bool incomplete = phase == 2;
          const auto expected_thought = phase ? thought : std::string{};
          const auto expected_answer = incomplete ? std::string{} : answer;
          const auto raw = expected_thought + (phase == 1 ? "</think>" : "") +
                           expected_answer;
          FakeBackend backend;
          backend.finish_reason =
              incomplete
                  ? gufo::server::TextGenerationBackend::FinishReason::kLength
                  : gufo::server::TextGenerationBackend::FinishReason::kStop;
          for (char byte : raw)
            backend.pieces.emplace_back(1, byte);
          gufo::server::HttpResponse response;
          if (responses) {
            if (declared) {
              auto tool = *body["tools"].items()[0].find("function");
              tool["type"] = "function";
              body["tools"] = Value::array();
              body["tools"].push_back(std::move(tool));
            }
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = phase != 0;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "disabled tools have valid Responses controls");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 4096, {}, stream);
          } else
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          std::string text, reasoning;
          const auto inspect = [&](const Value& event) {
            if (responses) {
              if (stream) {
                if (event.member_str("type") == "response.output_text.delta")
                  text += event.member_str("delta");
                if (event.member_str("type") ==
                    "response.reasoning_summary_text.delta")
                  reasoning += event.member_str("delta");
              } else
                for (const auto& item : event.find("output")->items()) {
                  Expect(item.member_str("type") != "function_call",
                         "disabled tools never become API calls");
                  for (const auto* field : {"content", "summary"})
                    if (const auto* parts = item.find(field))
                      for (const auto& part : parts->items())
                        (std::string_view(field) == "content" ? text
                                                              : reasoning) +=
                            part.member_str("text");
                }
            } else
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                if (!message)
                  continue;
                Expect(!message->contains("tool_calls"),
                       "disabled tools never become API calls");
                text += message->member_str("content");
                reasoning += message->member_str("reasoning_content");
              }
          };
          Expect(response.status == 200,
                 "disabled tool marker request succeeds");
          if (stream)
            response.streaming_body([&](std::string_view chunk) {
              const auto pos = chunk.find("data: ");
              if (pos != std::string_view::npos &&
                  !chunk.substr(pos + 6).starts_with("[DONE]"))
                inspect(gufo::json::parse(chunk.substr(pos + 6)));
              return true;
            });
          else
            inspect(gufo::json::parse(response.body));
          Expect(text == expected_answer && reasoning == expected_thought,
                 "disabled tool markers preserve content and reasoning phases");
        }
}

void TestNativeToolTransports() {
  using gufo::json::Value;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const std::string literal =
      " <think>literal</think><tool_call></tool_call>"
      "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":\"nested\"}}</"
      "tool_call>"
      "<｜DSML｜tool_calls><｜DSML｜invoke name=\"f\"></｜DSML｜invoke>"
      "</｜DSML｜tool_calls>"
      "</function></｜DSML｜invoke> \"42\" \\\nπ🦉\n \r";
  auto args = Value::object();
  args["text"] = literal;
  const std::vector<std::string> calls{
      "<tool_call>\n<function=f>\n<parameter=text>\n" + literal +
          "\n</parameter>\n</function>\n</tool_call>",
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
      "<｜DSML｜parameter name=\"text\" string=\"true\">" +
          literal +
          "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
          "</｜DSML｜tool_calls>",
      "<tool_call>{\"name\":\"f\",\"arguments\":" + args.dump() +
          "}</tool_call>"};
  for (const auto& call : calls)
    for (bool thinking : {false, true})
      for (bool stream : {false, true})
        for (bool parallel : {false, true})
          for (bool responses : {false, true})
            for (bool constrained : {false, true}) {
              auto body = gufo::json::parse(R"({
              "model":"test-model","messages":[{"role":"user","content":"call f"}],
              "tools":[{"type":"function","function":{"name":"f","strict":false,
                "parameters":{"type":"object","properties":{"text":{"type":"string"}},
                "required":["text"],"additionalProperties":false}}}],
              "tool_choice":"required"})");
              body["stream"] = stream;
              body["parallel_tool_calls"] = parallel;
              body["tool_choice"] = constrained ? "required" : "auto";
              body["chat_template_kwargs"]["enable_thinking"] = thinking;
              FakeBackend backend;
              if (constrained)
                backend.tool_format =
                    call.starts_with("<｜DSML｜tool_calls>")
                        ? gufo::sampling::JsonConstraint::ToolFormat::kDeepSeek
                        : gufo::sampling::JsonConstraint::ToolFormat::kQwen;
              const auto raw =
                  (thinking ? "Reasoning.</think>" : "") + call + " \n";
              for (char byte : raw)
                backend.pieces.emplace_back(1, byte);
              gufo::server::HttpResponse response;
              if (responses) {
                // Exercise the actual flat Responses function shape.
                auto flat = *body["tools"].items()[0].find("function");
                flat["type"] = "function";
                body["tools"] = Value::array();
                body["tools"].push_back(std::move(flat));
                gufo::server::ChatRequest chat;
                chat.reasoning.enabled = thinking;
                Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                       "Responses shares tool controls");
                response = gufo::server::CreateOpenAiResponse(
                    Request(body.dump()), backend, chat, 256, {}, stream);
              } else {
                response = gufo::server::HandleOpenAiChat(Request(body.dump()),
                                                          backend);
              }
              Expect(response.status == 200, "native tool request succeeds");
              std::vector<Value> output;
              if (stream) {
                response.streaming_body([&](std::string_view chunk) {
                  auto pos = chunk.find("data: ");
                  if (pos != std::string_view::npos &&
                      !chunk.substr(pos + 6).starts_with("[DONE]"))
                    output.push_back(gufo::json::parse(chunk.substr(pos + 6)));
                  return true;
                });
              } else {
                output.push_back(gufo::json::parse(response.body));
              }
              std::string arguments, reasoning;
              std::size_t call_count = 0;
              for (const auto& event : output) {
                if (responses) {
                  if (event.member_str("type") ==
                      "response.function_call_arguments.done")
                    Expect(event.member_str("name") == "f" &&
                               event.member_str("arguments") == args.dump(),
                           "Responses argument completion identifies the "
                           "function");
                  const auto* result = stream ? event.find("response") : &event;
                  if (result && result->member_str("status") == "completed")
                    for (const auto& item : result->find("output")->items())
                      if (item.member_str("type") == "function_call") {
                        ++call_count;
                        arguments = item.member_str("arguments");
                      }
                } else if (const auto* choices = event.find("choices")) {
                  for (const auto& choice : choices->items()) {
                    const auto* message =
                        choice.find(stream ? "delta" : "message");
                    if (!message)
                      continue;
                    reasoning += message->member_str("reasoning_content");
                    if (const auto* tools = message->find("tool_calls"))
                      for (const auto& tool : tools->items()) {
                        ++call_count;
                        arguments +=
                            tool.find("function")->member_str("arguments");
                      }
                  }
                }
              }
              Expect(call_count == 1 && arguments == args.dump(),
                     "native and JSON arguments survive both API transports");
              Expect(
                  backend.last_request.constrained_tools,
                  "automatic non-strict tools also bind decoding constraints");
              Expect(reasoning.find("literal") == std::string::npos,
                     "argument tags never enter reasoning");
            }
  for (const auto& call : calls)
    for (const auto finish : {Finish::kLength, Finish::kStopSequence})
      for (std::size_t length = 0; length < call.size(); ++length) {
        FakeBackend backend;
        backend.pieces = {call.substr(0, length)};
        backend.finish_reason = finish;
        auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "tool_choice":"required","tools":[{"type":"function","function":{
          "name":"f","parameters":{}}}]})");
        const auto result =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        Expect(result.status == 200, "tool truncation terminates successfully");
        const auto output = gufo::json::parse(result.body);
        const auto* emitted =
            output.find("choices")->items()[0].find("message")->find(
                "tool_calls");
        // A DSML invocation may finish before its outer list closes. Never
        // recover a nested literal call from a truncated argument.
        if (emitted)
          Expect(emitted->size() == 1 &&
                     emitted->items()[0]
                             .find("function")
                             ->member_str("arguments") == args.dump(),
                 "every native truncation preserves complete arguments only");
      }
}

void TestJsonToolStringOwnership() {
  using gufo::json::Value;
  using Constraint = gufo::sampling::JsonConstraint;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const auto schema = gufo::json::parse(R"({
    "type":"object","properties":{"content":{"type":"string","pattern":".*"}},
    "required":["content"],"additionalProperties":false
  })");
  // Native runners keep raw text for a pattern, as llama.cpp; the JSON
  // envelope below is the syntax of runners without a native format.
  Expect(Constraint::ToolParameters(schema, false,
                                    Constraint::ToolFormat::kQwen) != nullptr,
         "string patterns keep native tool framing");
  const auto grammar = Constraint::WithTools(
      nullptr, {{"write", Constraint::Compile(schema, false)}}, false, true);
  auto request = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"write"}],
    "tools":[]
  })");
  auto tool = Value::object();
  tool["type"] = "function";
  tool["function"]["name"] = "write";
  tool["function"]["parameters"] = schema;
  request["tools"].push_back(std::move(tool));
  // Complete @aarononeal's write example from #383 as JSON string data.
  std::string literal =
      "<tool_call>\n<function=write>\n<parameter=content>\nAAAA\n"
      "</parameter>\n</function>\n</tool_call>";
  std::erase(literal, '\n');
  for (const auto* prefix : {"", "quoted \"text\" and \\ "}) {
    Value arguments = Value::object();
    arguments["content"] = std::string(prefix) + literal;
    const std::string complete =
        "<tool_call>{\"name\":\"write\",\"arguments\":" + arguments.dump() +
        "}</tool_call>";
    const auto truncated =
        complete.substr(0, complete.find(literal) + literal.size());
    for (const bool partial : {false, true}) {
      const auto& raw = partial ? truncated : complete;
      auto state = grammar->Start();
      for (const unsigned char byte : raw)
        state = grammar->Advance(state, byte);
      Expect(!state.empty() && grammar->Complete(state) == !partial,
             "literal XML is an admitted JSON call or an unfinished prefix");
      for (const auto finish : {Finish::kLength, Finish::kStopSequence}) {
        for (const bool stream : {false, true}) {
          auto body = request;
          body["stream"] = stream;
          FakeBackend backend;
          backend.finish_reason = finish;
          if (finish == Finish::kStopSequence)
            backend.stop_sequence = "STOP";
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200,
                 "JSON tool ownership request succeeds");
          std::vector<Value> calls;
          std::string content;
          const auto inspect = [&](const Value& event) {
            if (const auto* choices = event.find("choices"))
              for (const auto& choice : choices->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                if (!message)
                  continue;
                content += message->member_str("content");
                if (const auto* found = message->find("tool_calls"))
                  calls.insert(calls.end(), found->items().begin(),
                               found->items().end());
              }
          };
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              const auto pos = chunk.find("data: ");
              if (pos != std::string_view::npos &&
                  !chunk.substr(pos + 6).starts_with("[DONE]"))
                inspect(gufo::json::parse(chunk.substr(pos + 6)));
              return true;
            });
          } else {
            inspect(gufo::json::parse(response.body));
          }
          if (partial && !calls.empty())
            std::cerr << "Truncated JSON produced: " << calls.front().dump()
                      << '\n';
          Expect(calls.size() == (partial ? 0 : 1),
                 "an unfinished JSON string cannot invoke its literal XML");
          Expect(content.empty(), "JSON tool framing stays out of content");
          if (!partial)
            Expect(calls.front().find("function")->member_str("arguments") ==
                       arguments.dump(),
                   "a complete JSON call preserves its literal XML argument");
        }
      }
    }
    for (const int rejected_shape : {0, 1, 2}) {
      auto rejected = Value::object();
      rejected["name"] = rejected_shape == 0 ? "undeclared" : "write";
      rejected["arguments"] = arguments;
      if (rejected_shape == 1)
        rejected["name"] = "";
      if (rejected_shape == 2) {
        rejected["arguments"] = Value::array();
        rejected["arguments"].push_back(std::string(prefix) + literal);
      }
      FakeBackend backend;
      backend.pieces = {"<tool_call>" + rejected.dump() + "</tool_call>" +
                        complete};
      const auto response =
          gufo::server::HandleOpenAiChat(Request(request.dump()), backend);
      Expect(response.status == 200,
             "rejected JSON calls permit later recovery");
      const auto output = gufo::json::parse(response.body);
      const auto* calls =
          output.find("choices")->items()[0].find("message")->find(
              "tool_calls");
      Expect(
          calls && calls->size() == 1 &&
              calls->items()[0].find("function")->member_str("arguments") ==
                  arguments.dump(),
          "rejected JSON envelopes cannot invoke quoted XML during recovery");
    }
  }
}

void TestNativeToolDialectSelection() {
  using gufo::json::Value;
  using gufo::sampling::JsonConstraint;
  using Format = JsonConstraint::ToolFormat;
  const auto native_call = [](Format format, std::string_view path) {
    if (format == Format::kDeepSeek)
      return "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n"
             "<｜DSML｜parameter name=\"path\" string=\"true\">" +
             std::string(path) +
             "</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
             "</｜DSML｜tool_calls>";
    return "<tool_call>\n<function=read>\n<parameter=path>\n" +
           std::string(path) + "\n</parameter>\n</function>\n</tool_call>";
  };
  // The JSON fallback leaves text before <tool_call> unconstrained. A native
  // DeepSeek call written there remains a call, as before this change.
  {
    auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string","pattern":"^[a-z.]+$"}},
    "required":["path"],"additionalProperties":false})");
    // A native runner keeps DSML for this schema; the JSON-envelope runner
    // below is the remaining fallback syntax.
    Expect(JsonConstraint::ToolParameters(schema, false, Format::kDeepSeek) !=
               nullptr,
           "patterned string keeps native tool framing");
    auto body = gufo::json::parse(R"({
    "model":"test-model","reasoning_effort":"none",
    "messages":[{"role":"user","content":"Read fixture.xml."}],
    "tool_choice":"auto","tools":[{"type":"function","function":{
      "name":"read","strict":false}}]})");
    auto tool = body["tools"].items()[0];
    tool["function"]["parameters"] = schema;
    body["tools"] = Value::array();
    body["tools"].push_back(std::move(tool));
    for (const bool stream : {false, true}) {
      body["stream"] = stream;
      FakeBackend backend;
      backend.tool_format = Format::kJson;
      backend.pieces = {native_call(Format::kDeepSeek, "fixture.xml")};
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "JSON fallback request succeeds");
      std::string content;
      std::vector<std::string> arguments;
      const auto collect = [&](const Value& event) {
        for (const auto& choice : event.find("choices")->items()) {
          const auto* message = choice.find(stream ? "delta" : "message");
          content += message->member_str("content");
          if (const auto* calls = message->find("tool_calls"))
            for (const auto& item : calls->items())
              arguments.push_back(
                  item.find("function")->member_str("arguments"));
        }
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]"))
            collect(gufo::json::parse(payload));
          return true;
        });
      } else {
        collect(gufo::json::parse(response.body));
      }
      Expect(arguments == std::vector<std::string>{R"({"path":"fixture.xml"})"},
             "a native call written under the JSON fallback is returned");
      Expect(content.find("DSML") == std::string::npos,
             "the recognized native call is not visible content");
    }
  }
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string"}},
    "required":["path"],"additionalProperties":false})");
    const auto parameters =
        JsonConstraint::ToolParameters(schema, false, format);
    Expect(parameters != nullptr, "file tool has a supported wire format");
    const auto grammar = JsonConstraint::WithTools(
        nullptr, {{"read", parameters}}, false, true, format);
    const std::string call = native_call(format, "fixture.xml");
    const auto foreign =
        format == Format::kDeepSeek ? Format::kQwen : Format::kDeepSeek;
    // As in llama.cpp, calls end the output in both formats.
    const std::string suffix;
    for (const std::string& prefix :
         {std::string{"The XML root is `<tool_calls></tool_calls>`.\n"},
          std::string{"The foreign wrapper is `"} +
              (foreign == Format::kQwen ? "<tool_call>"
                                        : "<｜DSML｜tool_calls>") +
              "`.\n",
          "Example syntax:\n" + native_call(foreign, "example.xml") +
              "\nNow inspect the fixture.\n"}) {
      const auto raw = prefix + call + suffix;
      auto state = grammar->Start();
      for (const unsigned char byte : raw)
        state = grammar->Advance(state, byte);
      Expect(grammar->Complete(state),
             "foreign visible literals and the actual call satisfy the "
             "production tool grammar");
      for (const bool responses : {false, true}) {
        for (const bool stream : {false, true}) {
          auto body = gufo::json::parse(R"({
          "model":"test-model","reasoning_effort":"none",
          "messages":[{"role":"user","content":"Inspect fixture.xml and its tool_calls root element."}],
          "tool_choice":"auto","tools":[{"type":"function","function":{
            "name":"read","strict":false}}]})");
          auto tool = body["tools"].items()[0];
          tool["function"]["parameters"] = schema;
          body["tools"] = Value::array();
          body["tools"].push_back(std::move(tool));
          body["stream"] = stream;
          FakeBackend backend;
          backend.tool_format = format;
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
          gufo::server::HttpResponse response;
          if (responses) {
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = false;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "Responses accepts the file tool");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 256, {}, stream);
          } else {
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          }
          Expect(response.status == 200, "native file tool request succeeds");
          // A later backend configuration must not change an admitted stream's
          // format. The fake snapshots this metadata on its GenerationRequest.
          backend.tool_format.reset();
          std::string content;
          std::vector<std::string> arguments;
          const auto collect = [&](const Value& event) {
            if (responses) {
              const auto* result = stream ? event.find("response") : &event;
              if (!result || result->member_str("status") != "completed")
                return;
              for (const auto& item : result->find("output")->items()) {
                if (item.member_str("type") == "function_call")
                  arguments.push_back(item.member_str("arguments"));
                else if (item.member_str("type") == "message")
                  for (const auto& part : item.find("content")->items())
                    content += part.member_str("text");
              }
            } else {
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                content += message->member_str("content");
                if (const auto* calls = message->find("tool_calls"))
                  for (const auto& item : calls->items())
                    arguments.push_back(
                        item.find("function")->member_str("arguments"));
              }
            }
          };
          if (stream) {
            response.streaming_body([&](std::string_view chunk) {
              const auto payload = chunk.substr(chunk.find("data: ") + 6);
              if (!payload.starts_with("[DONE]"))
                collect(gufo::json::parse(payload));
              return true;
            });
          } else {
            collect(gufo::json::parse(response.body));
          }
          Expect(arguments ==
                     std::vector<std::string>{R"({"path":"fixture.xml"})"},
                 "only the actual selected-dialect call reaches the client");
          Expect(content == prefix + suffix,
                 "foreign tool syntax remains literal visible content");
        }
      }
    }
  }
  // DeepSeek's template writes "\n\n" before its call block. That separator
  // is framing, as in llama.cpp's parser; returned as content, the replayed
  // history would carry it twice and miss the generated continuation.
  for (const std::string prose : {"", "Reading the fixture."}) {
    for (const bool stream : {false, true}) {
      auto body = gufo::json::parse(R"({
      "model":"test-model","reasoning_effort":"none",
      "messages":[{"role":"user","content":"Read fixture.xml."}],
      "tool_choice":"auto","tools":[{"type":"function","function":{
        "name":"read","parameters":{"type":"object",
        "properties":{"path":{"type":"string"}},"required":["path"]}}}]})");
      body["stream"] = stream;
      FakeBackend backend;
      backend.tool_format = Format::kDeepSeek;
      for (const char byte :
           prose + "\n\n" + native_call(Format::kDeepSeek, "fixture.xml"))
        backend.pieces.emplace_back(1, byte);
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      std::string content;
      std::size_t calls = 0;
      const auto collect = [&](const Value& event) {
        for (const auto& choice : event.find("choices")->items()) {
          const auto* message = choice.find(stream ? "delta" : "message");
          content += message->member_str("content");
          if (const auto* found = message->find("tool_calls"))
            calls += found->size();
        }
      };
      if (stream) {
        response.streaming_body([&](std::string_view chunk) {
          const auto payload = chunk.substr(chunk.find("data: ") + 6);
          if (!payload.starts_with("[DONE]"))
            collect(gufo::json::parse(payload));
          return true;
        });
      } else {
        collect(gufo::json::parse(response.body));
      }
      Expect(response.status == 200 && calls == 1 && content == prose,
             "the separator before a DeepSeek call block is not content");
    }
  }
}

void TestNativeToolTextOutsideEnvelopes() {
  using gufo::json::Value;
  using gufo::sampling::JsonConstraint;
  using Format = JsonConstraint::ToolFormat;
  using Finish = gufo::server::TextGenerationBackend::FinishReason;
  const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"path":{"type":"string"}},
    "required":["path"],"additionalProperties":false})");
  const auto invoke = [](std::string_view path) {
    return "\n<｜DSML｜invoke name=\"read\">\n"
           "<｜DSML｜parameter name=\"path\" string=\"true\">" +
           std::string(path) + "</｜DSML｜parameter>\n</｜DSML｜invoke>";
  };
  const auto envelope = [](const std::string& calls) {
    return "<｜DSML｜tool_calls>" + calls + "\n</｜DSML｜tool_calls>";
  };
  const std::string suffix = "\nNo edits are requested.";
  const auto first = envelope(invoke("fixture.txt"));
  const auto legacy =
      "<tool_calls>" + invoke("example.txt") + "\n</tool_calls>";
  const std::string literal = "Keep </｜DSML｜tool_calls> as argument data.";
  struct Fixture {
    Format format;
    std::string raw;
    std::string text;
    std::vector<std::string> paths;
    bool complete{true};
    bool canonical{true};
    Finish finish{Finish::kStop};
    bool required{false};
    std::optional<Format> admitted_format{};
  };
  const std::vector<Fixture> fixtures{
      {Format::kQwen,
       "<tool_call>\n<function=read>\n<parameter=path>\nfixture.txt\n"
       "</parameter>\n</function>\n</tool_call>" +
           suffix,
       suffix,
       {"fixture.txt"}},
      {Format::kJson,
       R"(<tool_call>{"name":"read","arguments":{"path":"fixture.txt"}}</tool_call>)" +
           suffix,
       suffix,
       {"fixture.txt"}},
      {Format::kDeepSeek, first + suffix, suffix, {"fixture.txt"}},
      // Native framing may be part of the admitted opener, not its tag name.
      {.format = Format::kDeepSeek,
       .raw = "\n\n" + first + suffix,
       .text = suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      {Format::kDeepSeek,
       "Before.\n" + envelope(invoke("fixture.txt") + invoke("second.txt")) +
           suffix,
       "Before.\n" + suffix,
       {"fixture.txt", "second.txt"}},
      {Format::kDeepSeek,
       first + "\nBetween.\n" + envelope(invoke("second.txt")) + suffix,
       "\nBetween.\n" + suffix,
       {"fixture.txt", "second.txt"}},
      {Format::kDeepSeek,
       envelope(invoke(literal)) + suffix,
       suffix,
       {literal}},
      // Interrupted output may keep a complete invocation even before its
      // outer list closes, but must not expose any unfinished markup.
      {Format::kDeepSeek,
       first + "\nBetween.\n<｜DSML｜tool_calls>" + invoke("second.txt"),
       "\nBetween.\n",
       {"fixture.txt", "second.txt"},
       false},
      {Format::kDeepSeek,
       first + "\nBetween.\n<｜DSML｜tool_calls>\n"
               "<｜DSML｜invoke name=\"read\">\n"
               "<｜DSML｜parameter name=\"path\" string=\"true\">partial",
       "\nBetween.\n",
       {"fixture.txt"},
       false},
      // Existing compatibility spelling is accepted by extraction, rather
      // than generated by the canonical native grammar.
      {Format::kDeepSeek,
       "<｜DSML｜tool_calls｜>" + invoke("fixture.txt") +
           "\n</｜DSML｜tool_calls｜>" + suffix,
       suffix,
       {"fixture.txt"},
       true,
       false},
      // A partial marker is legal ordinary text in auto mode. Required tools
      // instead hold it back when a length limit or explicit stop interrupts.
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n<｜DSML｜tool_calls",
       .paths = {"fixture.txt"},
       .finish = Finish::kLength},
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n",
       .paths = {"fixture.txt"},
       .finish = Finish::kLength,
       .required = true},
      {.format = Format::kDeepSeek,
       .raw = first + "\nBetween.\n<｜DSML｜tool_calls",
       .text = "\nBetween.\n",
       .paths = {"fixture.txt"},
       .finish = Finish::kStopSequence,
       .required = true},
      // Only an outer envelope starts DSML calls. Bare invocation examples
      // outside it are ordinary text, even before another real envelope.
      {Format::kDeepSeek,
       first + invoke("example.txt") + suffix,
       invoke("example.txt") + suffix,
       {"fixture.txt"}},
      {Format::kDeepSeek,
       first + invoke("example.txt") + envelope(invoke("second.txt")) + suffix,
       invoke("example.txt") + suffix,
       {"fixture.txt", "second.txt"}},
      {.format = Format::kDeepSeek,
       .raw = first + suffix,
       .text = suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      // A known native grammar owns only its canonical envelope. Foreign
      // wrapper examples remain prose, including native-looking invocations.
      {.format = Format::kDeepSeek,
       .raw = first + "\n<tool_calls></tool_calls>" + suffix,
       .text = "\n<tool_calls></tool_calls>" + suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      {.format = Format::kDeepSeek,
       .raw = first + legacy + suffix,
       .text = legacy + suffix,
       .paths = {"fixture.txt"},
       .admitted_format = Format::kDeepSeek},
      // Unknown metadata and JSON fallback deliberately retain legacy native
      // recognition. Both wrapped calls still belong to the tool output.
      {.format = Format::kDeepSeek,
       .raw = first + legacy + suffix,
       .text = suffix,
       .paths = {"fixture.txt", "example.txt"}},
      {.format = Format::kJson,
       .raw = first + legacy + suffix,
       .text = suffix,
       .paths = {"fixture.txt", "example.txt"},
       .admitted_format = Format::kJson},
  };
  for (const auto& fixture : fixtures) {
    auto parameters_schema = schema;
    if (fixture.format == Format::kJson) {
      parameters_schema["enum"] = gufo::json::parse(
          R"([{"path":"fixture.txt"},{"path":"example.txt"}])");
      // A native runner keeps DSML for finite objects too; this fixture
      // covers runners whose only call syntax is the JSON envelope.
      Expect(JsonConstraint::ToolParameters(parameters_schema, false,
                                            Format::kDeepSeek) != nullptr,
             "finite object parameters keep native tool framing");
    }
    if (fixture.canonical) {
      const auto parameters = JsonConstraint::ToolParameters(
          parameters_schema, false, fixture.format);
      Expect(parameters != nullptr, "native tool schema is supported");
      const auto grammar =
          JsonConstraint::WithTools(nullptr, {{"read", parameters}},
                                    fixture.required, true, fixture.format);
      // As in llama.cpp, native calls end the output (Qwen may add the
      // whitespace llama.cpp's `space` admits). Text after them is reachable
      // only without that grammar, e.g. the JSON envelope runners use.
      std::string_view admitted = fixture.raw;
      if (fixture.format == Format::kDeepSeek) {
        constexpr std::string_view kClose = "\n</｜DSML｜tool_calls>";
        admitted = admitted.substr(0, admitted.find(kClose) + kClose.size());
      } else if (fixture.format == Format::kQwen) {
        constexpr std::string_view kClose = "</tool_call>";
        auto end = admitted.find(kClose) + kClose.size();
        while (end < admitted.size() && admitted[end] == '\n')
          ++end;
        admitted = admitted.substr(0, end);
      }
      auto state = grammar->Start();
      for (const unsigned char byte : admitted) {
        state = grammar->Advance(state, byte);
        Expect(!state.empty(), "fixture is admitted by the production grammar");
      }
      Expect(grammar->Complete(state) ==
                 (fixture.complete || admitted.size() < fixture.raw.size()),
             "only complete fixture may terminate normally");
      if (admitted.size() < fixture.raw.size())
        Expect(grammar
                   ->Advance(state, static_cast<unsigned char>(
                                        fixture.raw[admitted.size()]))
                   .empty(),
               "native output ends after its calls");
    }
    for (const bool responses : {false, true})
      for (const bool stream : {false, true})
        for (const bool fragmented : {false, true}) {
          auto body = gufo::json::parse(R"({"model":"test-model",
            "reasoning_effort":"none","tool_choice":"auto",
            "messages":[{"role":"user","content":"Read the fixtures."}],
            "tools":[{"type":"function","function":{"name":"read",
              "strict":false}}]})");
          auto tool = body["tools"].items()[0];
          tool["function"]["parameters"] = parameters_schema;
          body["tools"] = Value::array();
          body["tools"].push_back(responses ? *tool.find("function") : tool);
          if (responses) {
            auto flat = body["tools"].items()[0];
            flat["type"] = "function";
            body["tools"] = Value::array();
            body["tools"].push_back(std::move(flat));
          }
          body["stream"] = stream;
          body["tool_choice"] = fixture.required ? "required" : "auto";
          FakeBackend backend;
          backend.tool_format = fixture.admitted_format;
          backend.finish_reason =
              fixture.complete ? fixture.finish : Finish::kLength;
          if (fragmented)
            for (const char byte : fixture.raw)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces = {fixture.raw};
          gufo::server::HttpResponse response;
          if (responses) {
            gufo::server::ChatRequest chat;
            chat.reasoning.enabled = false;
            Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
                   "Responses accepts tool controls");
            response = gufo::server::CreateOpenAiResponse(
                Request(body.dump()), backend, chat, 512, {}, stream);
          } else {
            response =
                gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          }
          Expect(response.status == 200, "native tool request succeeds");
          backend.tool_format.reset();
          std::string text, deltas;
          std::vector<std::string> paths;
          const auto call = [&](const Value& function) {
            Expect(function.member_str("name") == "read",
                   "only the declared intended call is returned");
            const auto arguments =
                gufo::json::parse(function.member_str("arguments"));
            Expect(arguments.size() == 1, "call has exactly its path argument");
            paths.push_back(arguments.member_str("path"));
          };
          const auto collect = [&](const Value& event) {
            if (responses) {
              if (event.member_str("type") == "response.output_text.delta")
                deltas += event.member_str("delta");
              const auto* result = stream ? event.find("response") : &event;
              if (!result ||
                  (stream && event.member_str("type") != "response.completed" &&
                   event.member_str("type") != "response.incomplete"))
                return;
              for (const auto& item : result->find("output")->items())
                if (item.member_str("type") == "function_call")
                  call(item);
                else if (item.member_str("type") == "message")
                  for (const auto& part : item.find("content")->items())
                    text += part.member_str("text");
            } else {
              for (const auto& choice : event.find("choices")->items()) {
                const auto* message = choice.find(stream ? "delta" : "message");
                text += message->member_str("content");
                if (const auto* calls = message->find("tool_calls"))
                  for (const auto& tool : calls->items())
                    call(*tool.find("function"));
              }
            }
          };
          if (stream)
            response.streaming_body([&](std::string_view chunk) {
              const auto data = chunk.find("data: ");
              Expect(data != std::string_view::npos, "stream uses SSE framing");
              const auto payload = chunk.substr(data + 6);
              if (!payload.starts_with("[DONE]"))
                collect(gufo::json::parse(payload));
              return true;
            });
          else
            collect(gufo::json::parse(response.body));
          Expect(backend.last_request.constrained_tools && backend.completed,
                 "tool request uses constrained generation");
          Expect(paths == fixture.paths,
                 "complete calls and literal arguments retain ownership");
          Expect(text == fixture.text,
                 "ordinary text outside complete tool envelopes survives");
          if (responses && stream)
            Expect(deltas == text,
                   "Responses text deltas agree with the final output");
        }
  }
}

void TestNativeReferencedArgumentTypes() {
  using gufo::json::Value;
  const auto arguments = gufo::json::parse(
      R"({"n":42,"b":true,"a":[1],"o":{"x":2},"s":"42","z":null})");
  const auto schema = gufo::json::parse(R"({
    "type":"object","properties":{
      "n":{"$ref":"#/$defs/number~1type~0"},
      "b":{"$ref":"#/$defs/boolean"},"a":{"$ref":"#/$defs/array"},
      "o":{"$ref":"#/$defs/object"},"s":{"$ref":"#/$defs/string"},
      "z":{"$ref":"#/$defs/null"}},
    "required":["n","b","a","o","s","z"],"additionalProperties":false,
    "$defs":{"number/type~":{"type":"integer"},"boolean":{"type":"boolean"},
      "array":{"type":"array","items":{"type":"integer"}},
      "object":{"type":"object","properties":{"x":{"type":"integer"}},
                "required":["x"],"additionalProperties":false},
      "string":{"type":"string"},"null":{"type":"null"}}})");
  for (const bool root_reference : {false, true})
    for (const bool responses : {false, true}) {
      auto parameters = schema;
      if (root_reference) {
        parameters = Value::object();
        parameters["$ref"] = "#/$defs/Arguments";
        parameters["$defs"] = *schema.find("$defs");
        auto object = Value::object();
        for (const auto& [key, value] : schema.members())
          if (key != "$defs")
            object[key] = value;
        parameters["$defs"]["Arguments"] = std::move(object);
      }
      auto body = gufo::json::parse(R"({
        "model":"test-model","messages":[{"role":"user","content":"call f"}],
        "tools":[{"type":"function","function":{"name":"f","strict":true}}],
        "tool_choice":"required"})");
      auto tool = body["tools"].items()[0];
      tool["function"]["parameters"] = parameters;
      body["tools"] = Value::array();
      body["tools"].push_back(std::move(tool));
      FakeBackend backend;
      std::string call = "<tool_call>\n<function=f>\n";
      for (const auto& [name, value] : arguments.members())
        call += "<parameter=" + name + ">\n" +
                (value.is_string() ? value.str() : value.dump()) +
                "\n</parameter>\n";
      backend.pieces = {call + "</function>\n</tool_call>"};
      gufo::server::HttpResponse response;
      if (responses) {
        gufo::server::ChatRequest chat;
        Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
               "referenced Responses tool schema is accepted");
        response = gufo::server::CreateOpenAiResponse(
            Request(body.dump()), backend, chat, 256, {}, false);
      } else {
        response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      }
      Expect(response.status == 200, "referenced native tool succeeds");
      const auto output = gufo::json::parse(response.body);
      const auto encoded =
          responses
              ? output.find("output")->items().back().member_str("arguments")
              : output.find("choices")
                    ->items()[0]
                    .find("message")
                    ->find("tool_calls")
                    ->items()[0]
                    .find("function")
                    ->member_str("arguments");
      Expect(encoded == arguments.dump(),
             "native argument types follow property and root references");
    }
}

// Schemas that used to force a JSON envelope now stay native (#438). Their raw
// values are typed as llama.cpp's qwen3-coder parser types them: a schema that
// admits other JSON types tries them first, an exact string stays text.
void TestNativeArgumentTypingMatrix() {
  struct Case {
    const char* schema;
    const char* raw;
    const char* expected;
  };
  const std::vector<Case> cases{
      {R"({"type":"string","pattern":"^[0-9]+$"})", "42", R"("42")"},
      {R"({"type":"string","format":"uri"})", "https://a.b",
       R"("https://a.b")"},
      {R"({"type":"string","default":"x","examples":["y"]})", "true",
       R"("true")"},
      {R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "42", "42"},
      {R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "forty",
       R"("forty")"},
      {R"({"allOf":[{"type":"string"},{"minLength":1}]})", "42", R"("42")"},
      {R"({"allOf":[{"type":"integer"},{"minimum":1}]})", "42", "42"},
      {R"({"const":"42"})", "42", R"("42")"},
      {R"({"enum":["true","false"]})", "true", R"("true")"},
      {R"({"enum":["null","42"]})", "null", R"("null")"},
      {R"({"anyOf":[{"const":"42"},{"type":"null"}]})", "42", R"("42")"},
      {R"({"enum":["42",17]})", "17", "17"},
      {R"({"type":"boolean"})", "True", "true"},
      {R"({"minLength":1})", "42", R"("42")"},
      {R"({"pattern":".*"})", "true", R"("true")"},
      {R"({"not":{"type":"null"}})", "plain", R"("plain")"},
      {R"({"type":["string","null"]})", "null", "null"},
      {R"({"anyOf":[{"type":"integer"},{"type":"null"}]})", "null", "null"},
      {R"({"type":"integer","exclusiveMinimum":0,"maximum":9007199254740991})",
       "240000", "240000"},
      {R"({"type":"object","properties":{"x":{"type":"integer"}}})",
       R"({"x": 1})", R"({"x":1})"},
      {R"({"type":"array","items":{"type":"object"}})", R"([{"k": "v"}])",
       R"([{"k":"v"}])"},
      // A recursive schema the grammar cannot enforce keeps its JSON value.
      {R"({"$ref":"#"})", "5", "5"},
      {R"({"type":"string"})", "line 1\n<b>\"q\"</b>",
       R"("line 1\n<b>\"q\"</b>")"},
      {R"({"type":"string"})", "line\n</parameter> is literal\nend",
       R"("line\n</parameter> is literal\nend")"},
      {R"({"type":"string"})",
       "line\n</parameter> <parameter=other>literal\nend",
       R"("line\n</parameter> <parameter=other>literal\nend")"},
      {R"({"type":"string"})", "line\n</parameter> </function>literal\nend",
       R"("line\n</parameter> </function>literal\nend")"},
  };
  for (const auto& item : cases) {
    for (
        const auto* root :
        {R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"})",
         R"({"type":"object","additionalProperties":true})"}) {
      for (const bool stream : {false, true}) {
        auto parameters = gufo::json::parse(root);
        parameters["properties"]["v"] = gufo::json::parse(item.schema);
        parameters["required"] = gufo::json::parse(R"(["v"])");
        auto body = gufo::json::parse(R"({"model":"test-model",
          "messages":[{"role":"user","content":"call record"}],
          "tools":[],"tool_choice":"required"})");
        auto tool = gufo::json::Value::object();
        tool["type"] = "function";
        tool["function"]["name"] = "record";
        tool["function"]["parameters"] = parameters;
        body["tools"].push_back(std::move(tool));
        body["stream"] = stream;
        FakeBackend backend;
        backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
        backend.pieces = {std::string("<tool_call>\n<function=record>\n"
                                      "<parameter=v>\n") +
                          item.raw +
                          "\n</parameter>\n</function>\n</tool_call>"};
        const auto response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
        if (response.status != 200)
          std::cerr << "native schema=" << item.schema << " raw=" << item.raw
                    << " status=" << response.status
                    << " body=" << response.body << '\n';
        Expect(response.status == 200, "native typed call succeeds");
        std::string arguments;
        std::string content;
        if (stream) {
          response.streaming_body([&](std::string_view part) {
            if (part == "data: [DONE]\n\n")
              return true;
            const auto event = gufo::json::parse(part.substr(6));
            for (const auto& choice : event.find("choices")->items())
              if (const auto* delta = choice.find("delta")) {
                content += delta->member_str("content");
                if (const auto* calls = delta->find("tool_calls"))
                  for (const auto& call : calls->items())
                    arguments += call.find("function")->member_str("arguments");
              }
            return true;
          });
        } else {
          const auto output = gufo::json::parse(response.body);
          const auto& message =
              *output.find("choices")->items()[0].find("message");
          content = message.member_str("content");
          if (const auto* calls = message.find("tool_calls"))
            arguments =
                calls->items()[0].find("function")->member_str("arguments");
        }
        const auto expected = std::string(R"({"v":)") + item.expected + "}";
        if (arguments.empty() || gufo::json::parse(arguments).dump() !=
                                     gufo::json::parse(expected).dump())
          std::cerr << "typing schema=" << item.schema << " raw=" << item.raw
                    << " stream=" << stream << " got=" << arguments << '\n';
        Expect(!arguments.empty() && gufo::json::parse(arguments).dump() ==
                                         gufo::json::parse(expected).dump(),
               "native arguments are typed as llama.cpp types them");
        Expect(content.empty(), "a native call leaves no visible framing");
      }
    }
  }
  // Parseable JSON and Python-literal recovery must both respect a known
  // argument type. A malformed call must not invoke a tool with another type.
  for (const auto& [type, raw] :
       std::vector<std::pair<std::string, std::string>>{{"integer", "true"},
                                                        {"integer", "True"},
                                                        {"boolean", "1"},
                                                        {"array", "{}"},
                                                        {"object", "null"}}) {
    auto body = gufo::json::parse(R"({"model":"test-model",
      "messages":[{"role":"user","content":"call record"}],
      "tools":[{"type":"function","function":{"name":"record",
        "parameters":{"type":"object","properties":{"v":{"type":"integer"}},
        "required":["v"],"additionalProperties":false}}}]})");
    auto tool = body["tools"].items()[0];
    tool["function"]["parameters"]["properties"]["v"]["type"] = type;
    body["tools"] = gufo::json::Value::array();
    body["tools"].push_back(std::move(tool));
    FakeBackend backend;
    backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
    backend.pieces = {"<tool_call>\n<function=record>\n<parameter=v>\n" + raw +
                      "\n</parameter>\n</function>\n</tool_call>"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "invalid native output is handled safely");
    const auto output = gufo::json::parse(response.body);
    const auto& message = *output.find("choices")->items()[0].find("message");
    Expect(!message.find("tool_calls"),
           "a value with the wrong declared type is not invoked");
  }
  // A declared name with surrounding spaces remains exact. llama.cpp's
  // JSON mapper trims it despite matching the exact key in its PEG grammar.
  for (const bool stream : {false, true}) {
    auto body = gufo::json::parse(R"({"model":"test-model",
      "messages":[{"role":"user","content":"call record"}],
      "tools":[{"type":"function","function":{"name":"record","strict":true,
        "parameters":{"type":"object","properties":{" value ":{"type":"string"}},
        "required":[" value "],"additionalProperties":false}}}],
      "tool_choice":"required"})");
    body["stream"] = stream;
    FakeBackend backend;
    backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
    backend.pieces = {
        "<tool_call>\n<function=record>\n<parameter= value >\nx\n</parameter>\n"
        "</function>\n</tool_call>"};
    const auto response =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    Expect(response.status == 200, "spaced parameter call succeeds");
    std::string arguments;
    if (stream) {
      response.streaming_body([&](std::string_view part) {
        if (part == "data: [DONE]\n\n")
          return true;
        const auto event = gufo::json::parse(part.substr(6));
        for (const auto& choice : event.find("choices")->items())
          if (const auto* delta = choice.find("delta"))
            if (const auto* calls = delta->find("tool_calls"))
              for (const auto& call : calls->items())
                arguments += call.find("function")->member_str("arguments");
        return true;
      });
    } else {
      const auto output = gufo::json::parse(response.body);
      arguments = output.find("choices")
                      ->items()[0]
                      .find("message")
                      ->find("tool_calls")
                      ->items()[0]
                      .find("function")
                      ->member_str("arguments");
    }
    Expect(arguments == R"({" value ":"x"})",
           "a declared spaced name keeps its exact spelling");
  }
}

void TestNativeUnionToolTypes() {
  // Non-strict unions stay in Qwen's native syntax (#383). The parser tries
  // their typed alternatives first, as llama.cpp's qwen3-coder parser does,
  // while plain strings keep quotes and literal JSON text.
  auto body = gufo::json::parse(R"({"model":"test-model",
    "messages":[{"role":"user","content":"call record"}],
    "tools":[{"type":"function","function":{"name":"record","parameters":{
      "type":"object","properties":{
        "content":{"type":"string"},
        "provider":{"anyOf":[{"type":"string","const":"brave"},
                             {"type":"string","const":"exa"}]},
        "args":{"anyOf":[{"type":"string"},
                         {"type":"object","additionalProperties":true}]},
        "label":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "note":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "limit":{"anyOf":[{"type":"integer"},{"type":"null"}]}},
      "required":["content"]}}}],
    "tool_choice":"auto"})");
  FakeBackend backend;
  backend.pieces = {
      "<tool_call>\n<function=record>\n"
      "<parameter=content>\n#include \"a.h\"\n42\n</parameter>\n"
      "<parameter=provider>\nexa\n</parameter>\n"
      "<parameter=args>\n{\"key\": 1}\n</parameter>\n"
      "<parameter=label>\nnull\n</parameter>\n"
      "<parameter=note>\nplain text\n</parameter>\n"
      "<parameter=limit>\n3\n</parameter>\n"
      "</function>\n</tool_call>"};
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(response.status == 200, "native union call succeeds");
  const auto output = gufo::json::parse(response.body);
  const auto arguments = gufo::json::parse(output.find("choices")
                                               ->items()[0]
                                               .find("message")
                                               ->find("tool_calls")
                                               ->items()[0]
                                               .find("function")
                                               ->member_str("arguments"));
  Expect(arguments.dump() ==
             gufo::json::parse(R"({"content":"#include \"a.h\"\n42",
               "provider":"exa","args":{"key":1},"label":null,
               "note":"plain text","limit":3})")
                 .dump(),
         "native union arguments try typed alternatives before text");
}

void TestWildcardToolTypes() {
  using gufo::json::Value;
  using Constraint = gufo::sampling::JsonConstraint;
  using Format = Constraint::ToolFormat;
  const auto schema = gufo::json::parse(R"({"type":"object",
    "properties":{"value":{"type":"string"}},"required":["value"],
    "patternProperties":{"^x_":{"type":"integer"}}})");
  const auto arguments = gufo::json::parse(
      R"({"value":"alpha","x_n":1,"x_b":true,"x_z":null,"x_a":[1],"x_o":{"n":1},"x_s":"1"})");
  for (const auto format : {Format::kQwen, Format::kDeepSeek}) {
    const auto parameters = Constraint::ToolParameters(schema, false, format);
    Expect(parameters != nullptr, "wildcard tools keep native framing");
    const auto grammar = Constraint::WithTools(
        nullptr, {{"record", parameters}}, true, false, format);
    // Qwen tags carry no type flag, so as in llama.cpp only declared
    // parameters are generated; DeepSeek's flag keeps typed wildcards.
    const auto expected = format == Format::kQwen
                              ? gufo::json::parse(R"({"value":"alpha"})")
                              : arguments;
    std::string call;
    if (format == Format::kQwen) {
      std::string extras;
      for (const auto& [name, value] : arguments.members())
        if (name != "value")
          extras += "<parameter=" + name + ">\n" +
                    (value.is_string() ? value.str() : value.dump()) +
                    "\n</parameter>\n";
      const std::string head =
          "<tool_call>\n<function=record>\n<parameter=value>\nalpha\n"
          "</parameter>\n";
      const std::string tail = "</function>\n</tool_call>";
      auto rejected = grammar->Start();
      for (unsigned char byte : head + extras + tail)
        rejected = grammar->Advance(rejected, byte);
      Expect(rejected.empty() || !grammar->Complete(rejected),
             "undeclared Qwen wildcards are not generated");
      call = head + tail;
    } else {
      call = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"record\">\n";
      for (const auto& [name, value] : arguments.members())
        call += "<｜DSML｜parameter name=\"" + name + "\" string=\"" +
                (value.is_string() ? "true" : "false") + "\">" +
                (value.is_string() ? value.str() : value.dump()) +
                "</｜DSML｜parameter>\n";
      call += "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    }
    auto state = grammar->Start();
    for (unsigned char byte : call)
      state = grammar->Advance(state, byte);
    Expect(grammar->Complete(state), "round-trip call is grammar-admissible");
    for (bool responses : {false, true}) {
      auto body = gufo::json::parse(R"({"model":"test-model",
        "messages":[{"role":"user","content":"call record"}],
        "tools":[{"type":"function","function":{"name":"record","strict":false}}],
        "tool_choice":"auto"})");
      auto tool = body["tools"].items()[0];
      tool["function"]["parameters"] = schema;
      body["tools"] = Value::array();
      body["tools"].push_back(std::move(tool));
      FakeBackend backend;
      backend.pieces = {call};
      gufo::server::HttpResponse response;
      if (responses) {
        gufo::server::ChatRequest chat;
        Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
               "wildcard Responses controls are accepted");
        response = gufo::server::CreateOpenAiResponse(
            Request(body.dump()), backend, chat, 256, {}, false);
      } else {
        response =
            gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      }
      Expect(response.status == 200, "wildcard call succeeds");
      const auto output = gufo::json::parse(response.body);
      const auto encoded =
          responses
              ? output.find("output")->items().back().member_str("arguments")
              : output.find("choices")
                    ->items()[0]
                    .find("message")
                    ->find("tool_calls")
                    ->items()[0]
                    .find("function")
                    ->member_str("arguments");
      Expect(encoded == expected.dump(),
             "wildcard JSON types survive transport");
    }
  }
}

void TestToolMetadataAndFraming() {
  using gufo::json::Value;
  auto body = gufo::json::parse(R"({
    "model":"test-model","messages":[{"role":"user","content":"call f"}],
    "tools":[{"type":"function","function":{"name":"f","strict":true,
      "parameters":{"type":"object","properties":{"text":{"type":"string"}},
        "required":["text"],"additionalProperties":false}}}],
    "tool_choice":{"type":"function","name":"f"},
    "parallel_tool_calls":null})");
  for (const bool stream : {false, true}) {
    gufo::server::ChatRequest chat;
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts nullable parallel_tool_calls");
    Expect(chat.forced_tool_name == "f" && !chat.parallel_tool_calls,
           "named tools retain their identity and single-call semantics");
    FakeBackend backend;
    backend.pieces = {
        "<tool_call>\n<function=f>\n<parameter=text>\r\nliteral\r\r\n"
        "</parameter>\n</function>\n</tool_call>"};
    const auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 256, {}, stream);
    const auto inspect = [&](const Value& result) {
      const auto* choice = result.find("tool_choice");
      Expect(choice && choice->member_str("type") == "function" &&
                 choice->member_str("name") == "f",
             "Responses echoes named function choices, including lifecycle "
             "events");
      if (result.member_str("status") == "completed") {
        const auto arguments = gufo::json::parse(
            result.find("output")->items().back().member_str("arguments"));
        Expect(arguments.member_str("text") == "literal\r",
               "CRLF framing retains a literal trailing carriage return");
      }
    };
    if (stream)
      response.streaming_body([&](std::string_view chunk) {
        const auto event =
            gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
        if (const auto* result = event.find("response"))
          inspect(*result);
        return true;
      });
    else
      inspect(gufo::json::parse(response.body));
  }
  body["tool_choice"] = "auto";
  gufo::server::ChatRequest chat;
  Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat) &&
             chat.parallel_tool_calls && chat.forced_tool_name.empty(),
         "null Responses parallel control retains default");
  chat.forced_tool_name = "f";
  Expect(!gufo::server::ParseOpenAiResponseControls(
             gufo::json::parse(R"({"tool_choice":"none"})"), &chat) &&
             chat.forced_tool_name.empty() &&
             chat.tool_choice == gufo::server::ChatRequest::ToolChoice::kNone,
         "an explicit choice clears inherited named-tool metadata");
  FakeBackend backend;
  Expect(gufo::server::HandleOpenAiChat(Request(body.dump()), backend).status ==
             400,
         "Chat Completions parallel control still requires a boolean");
}

void TestResponsesOutput() {
  using Backend = gufo::server::TextGenerationBackend;
  for (const bool reasoning : {false, true}) {
    for (const bool limited : {false, true}) {
      FakeBackend backend;
      backend.pieces =
          reasoning ? std::vector<std::string>{"Check ", "\xE2\x94",
                                               "\x8C</think>", "\nAnswer"}
                    : std::vector<std::string>{"Answer ", "\xE2\x94", "\x8C"};
      backend.finish_reason = limited ? Backend::FinishReason::kLength
                                      : Backend::FinishReason::kStop;
      backend.reasoning_tokens = reasoning ? 3 : 0;
      gufo::server::ChatRequest chat;
      chat.reasoning.enabled = reasoning;
      const auto buffered = gufo::server::CreateOpenAiResponse(
          Request("{}"), backend, chat, 0, {}, false);
      const auto body = gufo::json::parse(buffered.body);
      const auto stream = gufo::server::CreateOpenAiResponse(
          Request("{}"), backend, chat, 0, {}, true);
      std::vector<gufo::json::Value> events;
      std::string text, thought;
      stream.streaming_body([&](std::string_view chunk) {
        const auto begin = chunk.find("\ndata: ");
        Expect(begin != std::string::npos, "Responses SSE has event and data");
        auto event = gufo::json::parse(chunk.substr(begin + 7));
        const auto type = event.member_str("type");
        Expect(chunk.starts_with("event: " + type + "\n"),
               "SSE name matches the semantic event type");
        Expect(event.member_size("sequence_number") == events.size(),
               "Responses events have consecutive sequence numbers");
        if (type == "response.output_text.delta")
          text += event.member_str("delta");
        if (type == "response.reasoning_summary_text.delta")
          thought += event.member_str("delta");
        if (const auto* response = event.find("response")) {
          Expect(response->find("parallel_tool_calls") &&
                     !response->find("parallel_tool_calls")->as_bool() &&
                     response->member_str("tool_choice") == "none" &&
                     response->find("tools")->items().empty(),
                 "Lifecycle events declare the supported tool policy");
        }
        events.push_back(std::move(event));
        return true;
      });
      Expect(events.front().member_str("type") == "response.created" &&
                 events[1].member_str("type") == "response.in_progress",
             "Responses starts with the lifecycle events");
      Expect(events.back().member_str("type") ==
                 (limited ? "response.incomplete" : "response.completed"),
             "Responses reports its terminal status");
      const auto& terminal = *events.back().find("response");
      Expect(text == (reasoning ? "Answer" : "Answer ┌") &&
                 thought == (reasoning ? "Check ┌" : ""),
             "Responses preserves UTF-8 and separates reasoning from text");
      const auto& items = terminal.find("output")->items();
      const auto& buffered_items = body.find("output")->items();
      Expect(items.size() == buffered_items.size(),
             "Buffered and streamed Responses have the same output items");
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto field =
            items[i].member_str("type") == "reasoning" ? "summary" : "content";
        Expect(items[i].find(field)->dump() ==
                   buffered_items[i].find(field)->dump(),
               "Buffered and streamed text/reasoning agree");
      }
      Expect(terminal.find("usage")
                     ->find("input_tokens_details")
                     ->member_size("cached_tokens") == 5,
             "Responses retains prompt-cache usage");
      for (const auto* response : {&body, &terminal}) {
        Expect(response->find("parallel_tool_calls") &&
                   !response->find("parallel_tool_calls")->as_bool() &&
                   response->member_str("tool_choice") == "none" &&
                   response->find("tools")->items().empty(),
               "Responses includes SDK-required tool fields");
        const auto& usage = *response->find("usage");
        Expect(usage.find("input_tokens_details")
                           ->member_size("cache_write_tokens") == 2 &&
                   usage.find("output_tokens_details")
                           ->member_size("reasoning_tokens") ==
                       backend.reasoning_tokens &&
                   usage.member_size("output_tokens") == backend.pieces.size(),
               "Responses includes new-cache and actual reasoning usage");
      }
    }
  }
}

void TestResponsesLiveAndCancellation() {
  FakeBackend backend;
  backend.pieces = {"<tool_call>first", "second"};
  backend.block_after_first_piece = true;
  auto response = gufo::server::CreateOpenAiResponse(Request("{}"), backend, {},
                                                     0, {}, true);
  std::atomic<bool> first{false};
  std::jthread writer([&] {
    response.streaming_body([&](std::string_view chunk) {
      if (chunk.find("event: response.output_text.delta") != std::string::npos)
        first = true;
      return true;
    });
  });
  Expect(backend.WaitForFirstPiece() && first && !backend.completed,
         "Responses sends text before generation completes");
  backend.Release();
  writer.join();

  backend.block_after_first_piece = false;
  backend.completed = false;
  auto cancelled = gufo::server::CreateOpenAiResponse(Request("{}"), backend,
                                                      {}, 0, {}, true);
  bool terminal = false;
  cancelled.streaming_body([&](std::string_view chunk) {
    terminal |= chunk.find("event: response.completed") != std::string::npos;
    return chunk.find("event: response.output_text.delta") == std::string::npos;
  });
  Expect(!terminal && !backend.completed,
         "Responses disconnect cancels generation without a completed event");
}

}  // namespace

void TestStreamingPromptProgress() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.progress = {{.total = 7, .cache = 5, .processed = 5, .time_ms = 0},
                      {.total = 7, .cache = 5, .processed = 7, .time_ms = 3}};
  const auto stream = [&](std::string_view options) {
    auto response = gufo::server::HandleOpenAiChat(
        Request(
            R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],)"
            R"("stream":true)" +
            std::string(options) + "}"),
        backend);
    Expect(response.status == 200 && response.streaming_body,
           "progress request streams");
    std::string output;
    response.streaming_body([&](std::string_view chunk) {
      output.append(chunk);
      return true;
    });
    return output;
  };

  const auto output = stream(R"(,"return_progress":true)");
  const auto first = output.find(
      R"("delta":{},"finish_reason":null}],)"
      R"("prompt_progress":{"total":7,"cache":5,"processed":5,"time_ms":0}})");
  const auto last = output.find(
      R"("delta":{},"finish_reason":null}],)"
      R"("prompt_progress":{"total":7,"cache":5,"processed":7,"time_ms":3}})");
  const auto content = output.find(R"("content":"Hel")");
  Expect(first != std::string::npos && last != std::string::npos &&
             first < last && last < content,
         "empty-delta prompt progress chunks precede streamed content");

  for (const auto* options :
       {"", R"(,"return_progress":false)", R"(,"return_progress":null)"}) {
    Expect(stream(options).find("prompt_progress") == std::string::npos,
           "prompt progress is opt-in");
  }

  const auto invalid = gufo::server::HandleOpenAiChat(
      Request(
          R"({"model":"test-model","messages":[{"role":"user","content":"hello"}],)"
          R"("stream":true,"return_progress":"yes"})"),
      backend);
  Expect(invalid.status == 400 &&
             invalid.body.find("invalid_return_progress") != std::string::npos,
         "return_progress must be a boolean");
}

void TestResponsesPromptProgress() {
  FakeBackend backend;
  backend.pieces = {"Hel", "lo"};
  backend.progress = {{.total = 7, .cache = 5, .processed = 5, .time_ms = 0},
                      {.total = 7, .cache = 5, .processed = 7, .time_ms = 3}};
  for (const auto* setting : {"true", "false", "null"}) {
    gufo::server::ChatRequest chat;
    const auto body = gufo::json::parse(std::string(R"({"return_progress":)") +
                                        setting + "}");
    Expect(!gufo::server::ParseOpenAiResponseControls(body, &chat),
           "Responses accepts boolean and null progress settings");
    const auto response = gufo::server::CreateOpenAiResponse(
        Request(body.dump()), backend, chat, 8, {}, true);
    std::size_t sequence = 0, progress_count = 0;
    bool content_seen = false;
    response.streaming_body([&](std::string_view chunk) {
      const auto offset = chunk.find("data: ");
      Expect(offset != std::string_view::npos, "Responses SSE contains data");
      const auto event = gufo::json::parse(chunk.substr(offset + 6));
      Expect(event.member_size("sequence_number") == sequence++,
             "progress preserves contiguous Responses sequence numbers");
      if (const auto* progress = event.find("prompt_progress")) {
        Expect(
            !content_seen &&
                event.member_str("type") == "response.in_progress" &&
                event.find("response")->member_str("status") == "in_progress",
            "Responses progress precedes content in an in_progress event");
        Expect(
            progress->member_size("processed") == (progress_count == 0 ? 5 : 7),
            "Responses progress preserves cached and processed counts");
        ++progress_count;
      }
      content_seen |= event.member_str("type") == "response.output_text.delta";
      return true;
    });
    Expect(content_seen && progress_count == (chat.return_progress ? 2 : 0),
           "Responses progress is opt-in and preserves content");
  }
  gufo::server::ChatRequest chat;
  Expect(gufo::server::ParseOpenAiResponseControls(
             gufo::json::parse(R"({"return_progress":1})"), &chat)
             .has_value(),
         "Responses rejects non-boolean progress settings");
}

// Issue #383: the model repeats the closing framing of a call it just made, and
// a client that stores assistant content replays that markup as history on
// every later turn. Issue #266: the same framing folded into an argument value
// or an unfinished call silently dropped. Both are parse-path defects, so they
// are covered here for buffered and streamed transports.
void TestToolClosingFraming() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model", "messages":[{"role":"user","content":"use f"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"}},
      "required":["text"],"additionalProperties":false}}}]
  })");
  const std::string call =
      "<tool_call><function=f><parameter=text>\n42\n</parameter></function></"
      "tool_call>";
  const std::string inline_call =
      "<tool_call><function=f><parameter=text>42</parameter></function></"
      "tool_call>";
  const std::string envelope =
      "<invoke name=\"f\"><parameter name=\"text\">x</parameter></invoke>";
  const std::string raw_xml =
      "<invoke name=\"documentation\"><parameter "
      "name=\"text\">x</parameter></invoke>";
  struct Case {
    std::string text;
    std::size_t calls;
    std::string argument;
    std::string content;
  };
  const std::vector<Case> qwen_cases{
      {"I'll update `config.py\n" + call, 1, R"({"text":"42"})",
       "I'll update `config.py\n"},
      {"Let`s write it.\n" + call, 1, R"({"text":"42"})", "Let`s write it.\n"},
      {"looks like `" + call + "` and no call.", 0, "",
       "looks like `" + call + "` and no call."},
      {"looks like ``" + call + "`` and no call.", 0, "",
       "looks like ``" + call + "`` and no call."},
      {"looks like `" + call, 1, R"({"text":"42"})", "looks like `"},
      {"looks like ``" + call, 1, R"({"text":"42"})", "looks like ``"},
      {"`example\n\n" + call, 1, R"({"text":"42"})", "`example\n\n"},
      {"looks like `literal <think>\n" + call + "`", 0, "",
       "looks like `literal <think>\n" + call + "`"},
      {call + "\n</invoke>\n</parameter>\n</function>\n", 1, R"({"text":"42"})",
       ""},
      {call + "\n</function>\n" + call, 2, R"({"text":"42"})", ""},
      {call + "\n" + envelope, 1, R"({"text":"42"})", ""},
      {call + "\n</invoke>\n<|im_end|>", 1, R"({"text":"42"})",
       std::string(kImEnd)},
      {"</invoke>\n" + call, 1, R"({"text":"42"})", "</invoke>\n"},
      {"</invoke>", 0, "", "</invoke>"},
      {"Text </parameter>", 0, "", "Text </parameter>"},
      {"The token is <|im_end|>", 0, "", "The token is <|im_end|>"},
      {"<parameter>x</parameter>", 0, "", "<parameter>x</parameter>"},
      {raw_xml, 0, "", raw_xml},
      {"Raw XML: " + raw_xml, 0, "", "Raw XML: " + raw_xml},
      {envelope, 0, "", ""},
      {"Planning.\n" + envelope, 0, "", "Planning."},
      {"<invoke name=\"f\"><parameter name=\"text\">x", 0, "", ""},
      {"Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</"
       "tool_call>",
       0, "",
       "Planning.\n<parameter=text>\nx\n</parameter>\n</function>\n</"
       "tool_call>"},
      {"<function=f><parameter=text>x</parameter></function>", 0, "",
       "<function=f><parameter=text>x</parameter></function>"},
      {"```xml\n" + envelope + "\n```", 0, "", "```xml\n" + envelope + "\n```"},
      {"`" + envelope + "` and prose", 0, "", "`" + envelope + "` and prose"},
      {"```python\nprint(1)\n" + call, 1, R"({"text":"42"})",
       "```python\nprint(1)\n"},
      {"```xml\n" + call + "\n```", 0, "", "```xml\n" + call + "\n```"},
      {"```xml\n" + call + "\n```\n" + call, 1, R"({"text":"42"})",
       "```xml\n" + call + "\n```\n"},
      {"`" + inline_call + "`", 0, "", "`" + inline_call + "`"},
      {"``" + inline_call + "``", 0, "", "``" + inline_call + "``"},
      {"```xml\n<tool_call><function=unknown></function></tool_call>", 0, "",
       "```xml\n<tool_call><function=unknown></function></tool_call>"},
      {"```xml\n<tool_call><function=f></function></tool_call>", 0, "",
       "```xml\n<tool_call><function=f></function></tool_call>"},
      {"```xml\n<tool_call><function=f><parameter=extra>x</parameter></"
       "function></tool_call>",
       0, "",
       "```xml\n<tool_call><function=f><parameter=extra>x</parameter></"
       "function></tool_call>"},
      {"<tool_call><function=f><parameter=text>\nEOS = "
       "\"<|im_end|>\"\n</parameter></function></tool_call>",
       1, R"({"text":"EOS = \"<|im_end|>\""})", ""},
      {"<tool_call><function=f><parameter=text>\n<|im_start|><|endoftext|>"
       "<|not_a_vocab_entry|>\n</parameter></function></tool_call>",
       1, R"({"text":"<|im_start|><|endoftext|><|not_a_vocab_entry|>"})", ""},
      {"<tool_call><function=f><parameter=text>\n</invoke>\n</parameter></"
       "function></tool_call>",
       1, R"({"text":"</invoke>"})", ""},
  };
  // As in llama.cpp, DeepSeek output is content outside its native block,
  // which ends the output: client envelopes and closers are never removed.
  const std::string dsml =
      "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n"
      "<｜DSML｜parameter name=\"text\" string=\"true\">42"
      "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
  const std::vector<Case> deepseek_cases{
      {envelope, 0, "", envelope},
      {"Planning.\n" + envelope, 0, "", "Planning.\n" + envelope},
      {"<invoke name=\"f\"><parameter name=\"text\">x", 0, "",
       "<invoke name=\"f\"><parameter name=\"text\">x"},
      {"</invoke>\n</parameter>", 0, "", "</invoke>\n</parameter>"},
      {call, 0, "", call},
      {"</invoke>" + dsml, 1, R"({"text":"42"})", "</invoke>"},
      {"Planning.\n" + envelope + dsml, 1, R"({"text":"42"})",
       "Planning.\n" + envelope},
  };
  using Format = gufo::sampling::JsonConstraint::ToolFormat;
  for (const auto& [format, cases] :
       {std::pair{Format::kQwen, &qwen_cases},
        std::pair{Format::kDeepSeek, &deepseek_cases}})
    for (const auto& item : *cases) {
      for (bool stream : {false, true}) {
        for (bool bytewise : {false, true}) {
          auto body = schema;
          body["stream"] = stream;
          body["tool_choice"] = "auto";
          FakeBackend backend;
          backend.tool_format = format;
          if (bytewise)
            for (const char byte : item.text)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces.push_back(item.text);
          const auto response =
              gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
          Expect(response.status == 200, "framing request succeeds");
          std::vector<Value> calls;
          std::string content;
          std::string finish;
          if (!stream) {
            const auto output = gufo::json::parse(response.body);
            const auto& choice = output.find("choices")->items()[0];
            const auto& message = *choice.find("message");
            content = message.member_str("content");
            finish = choice.member_str("finish_reason");
            if (const auto* found = message.find("tool_calls"))
              calls.assign(found->items().begin(), found->items().end());
          } else {
            response.streaming_body([&](std::string_view part) {
              if (part == "data: [DONE]\n\n")
                return true;
              const auto event = gufo::json::parse(part.substr(6));
              for (const auto& choice : event.find("choices")->items()) {
                finish += choice.member_str("finish_reason");
                if (const auto* delta = choice.find("delta")) {
                  content += delta->member_str("content");
                  if (const auto* found = delta->find("tool_calls"))
                    calls.insert(calls.end(), found->items().begin(),
                                 found->items().end());
                }
              }
              return true;
            });
          }
          if (calls.size() != item.calls || content != item.content) {
            std::cerr << "Framing input: " << item.text << "\nstream=" << stream
                      << " bytewise=" << bytewise
                      << "\nExpected content: " << item.content
                      << "\nActual content: " << content
                      << "\nExpected calls: " << item.calls
                      << ", actual: " << calls.size() << '\n';
          }
          Expect(calls.size() == item.calls,
                 "only complete declared calls are emitted");
          Expect(content == item.content,
                 "literal content and framing are distinguished using request "
                 "context");
          Expect(finish == (item.calls ? "tool_calls" : "stop"),
                 "finish reason agrees with parsed calls");
          if (!calls.empty())
            Expect(calls.back().find("function")->member_str("arguments") ==
                       item.argument,
                   "literal arguments survive exactly");
        }
      }
    }
}

void TestUnconstrainedQuoteFallbackChecksSchema() {
  const std::string prefix = "```python\nprint(1)\n";
  const std::string valid =
      "<tool_call>\n<function=f>\n<parameter=text>42</parameter>\n</"
      "function>\n</tool_call>";
  const std::string wrong =
      "<tool_call><function=f><parameter=text>43</parameter></function></"
      "tool_call>";
  const std::string missing = "<tool_call><function=f></function></tool_call>";
  struct Case {
    std::string text;
    bool call;
    std::string content;
  };
  for (const auto& item : std::vector<Case>{
           {prefix + valid, true, prefix},
           {prefix + wrong, false, prefix + wrong},
           {prefix + missing, false, prefix + missing},
           {"looks like `" + valid + "`", false, "looks like `" + valid + "`"},
           {"looks like ``" + valid + "``", false,
            "looks like ``" + valid + "``"},
           {"looks like `" + valid, true, "looks like `"},
           {"I'll update `config.py\n" + valid, true,
            "I'll update `config.py\n"},
           {"Let`s write it.\n" + valid, true, "Let`s write it.\n"},
           {"looks like ``" + valid, true, "looks like ``"},
           {"looks like `" + wrong, false, "looks like `" + wrong},
           {"looks like `" + missing, false, "looks like `" + missing},
           {"looks like `" + wrong + "\n\n", false,
            "looks like `" + wrong + "\n\n"},
       }) {
    for (const bool constrained : {false, true}) {
      for (const bool stream : {false, true}) {
        for (const bool bytewise : {false, true}) {
          gufo::server::ChatRequest chat;
          chat.reasoning.enabled = false;
          Expect(
              !gufo::server::ParseOpenAiResponseControls(
                  gufo::json::parse(
                      R"({"tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"text":{"type":"string","const":"42"}},"required":["text"],"additionalProperties":false}}]})"),
                  &chat),
              "fallback fixture has valid tool controls");
          chat.constrained_tools = constrained;
          FakeBackend backend;
          backend.tool_format =
              gufo::sampling::JsonConstraint::ToolFormat::kQwen;
          if (bytewise)
            for (const char byte : item.text)
              backend.pieces.emplace_back(1, byte);
          else
            backend.pieces.push_back(item.text);
          const auto response = gufo::server::CreateOpenAiResponse(
              Request("{}"), backend, chat, 512, {}, stream);
          gufo::json::Value output;
          std::string streamed;
          if (!stream)
            output = gufo::json::parse(response.body);
          else
            response.streaming_body([&](std::string_view chunk) {
              const auto offset = chunk.find("data: ");
              const auto event = gufo::json::parse(chunk.substr(offset + 6));
              if (event.member_str("type") == "response.output_text.delta")
                streamed += event.member_str("delta");
              if (event.member_str("type") == "response.completed")
                output = *event.find("response");
              return true;
            });
          std::string content;
          std::size_t calls = 0;
          for (const auto& part : output.find("output")->items()) {
            if (part.member_str("type") == "function_call") {
              ++calls;
              Expect(part.member_str("name") == "f" &&
                         part.member_str("arguments") == R"({"text":"42"})",
                     "fallback emits exactly the declared schema-valid call");
            }
            if (part.member_str("type") == "message") {
              for (const auto& text : part.find("content")->items())
                content += text.member_str("text");
            }
          }
          Expect(calls == (item.call ? 1u : 0u),
                 "both parser paths apply unfinished-quote schema checks and "
                 "protect "
                 "inline quotations");
          Expect(content == item.content,
                 "quoted documentation and invalid fence examples stay "
                 "byte-exact");
          if (stream)
            Expect(streamed == content,
                   "streamed documentation agrees with the final response");
        }
      }
    }
  }
}

void TestUnfinishedInlineReasoningFallback() {
  const std::string prefix = "Check `foo then\n";
  const std::string valid =
      "<tool_call><function=f><parameter=text>42</parameter></function></"
      "tool_call>";
  const std::string wrong =
      "<tool_call><function=f><parameter=text>43</parameter></function></"
      "tool_call>";
  const std::string missing = "<tool_call><function=f></function></tool_call>";
  struct Case {
    std::string text;
    bool call;
    std::string reasoning;
    std::string content;
  };
  for (const auto& item : std::vector<Case>{
           {prefix + valid, true, prefix, ""},
           {prefix + valid + "</think>", true, prefix, ""},
           {prefix + wrong, false, prefix + wrong, ""},
           {prefix + missing, false, prefix + missing, ""},
           {prefix + wrong + "</think>Answer", false, prefix + wrong, "Answer"},
           {prefix + valid + "`</think>Answer", false, prefix + valid + "`",
            "Answer"},
           {prefix + valid + "`</think>" + valid, true, prefix + valid + "`",
            ""},
       }) {
    for (const bool stream : {false, true})
      for (const bool bytewise : {false, true}) {
        gufo::server::ChatRequest chat;
        chat.reasoning.enabled = true;
        Expect(
            !gufo::server::ParseOpenAiResponseControls(
                gufo::json::parse(
                    R"({"tools":[{"type":"function","name":"f","parameters":{"type":"object","properties":{"text":{"type":"string","const":"42"}},"required":["text"],"additionalProperties":false}}]})"),
                &chat),
            "reasoning fallback fixture has valid controls");
        // Exercise the legacy implicit transition before </think>. Constrained
        // reasoning deliberately requires the explicit phase delimiter.
        chat.constrained_tools = false;
        FakeBackend backend;
        backend.tool_format = gufo::sampling::JsonConstraint::ToolFormat::kQwen;
        const auto& raw = item.text;
        if (bytewise)
          for (const char byte : raw)
            backend.pieces.emplace_back(1, byte);
        else
          backend.pieces = {raw};
        const auto response = gufo::server::CreateOpenAiResponse(
            Request("{}"), backend, chat, 512, {}, stream);
        gufo::json::Value output;
        std::string streamed_content, streamed_reasoning;
        if (!stream)
          output = gufo::json::parse(response.body);
        else
          response.streaming_body([&](std::string_view chunk) {
            const auto event =
                gufo::json::parse(chunk.substr(chunk.find("data: ") + 6));
            if (event.member_str("type") == "response.output_text.delta")
              streamed_content += event.member_str("delta");
            if (event.member_str("type") ==
                "response.reasoning_summary_text.delta")
              streamed_reasoning += event.member_str("delta");
            if (event.member_str("type") == "response.completed")
              output = *event.find("response");
            return true;
          });
        std::string content, reasoning;
        std::size_t calls = 0;
        for (const auto& part : output.find("output")->items()) {
          if (part.member_str("type") == "function_call") {
            ++calls;
            Expect(part.member_str("name") == "f" &&
                       part.member_str("arguments") == R"({"text":"42"})",
                   "recovered reasoning call satisfies its complete schema");
          }
          for (const auto* field : {"content", "summary"})
            if (const auto* parts = part.find(field)) {
              for (const auto& text : parts->items())
                (std::string_view(field) == "summary" ? reasoning : content) +=
                    text.member_str("text");
            }
        }
        if (calls != (item.call ? 1u : 0u) || content != item.content ||
            Trimmed(reasoning) != Trimmed(item.reasoning)) {
          std::cerr << "Reasoning fallback: " << raw << "\nstream=" << stream
                    << " bytewise=" << bytewise << "\ncontent=" << content
                    << "\nreasoning=" << reasoning << "\ncalls=" << calls
                    << '\n';
        }
        Expect(calls == (item.call ? 1u : 0u),
               "an unfinished reasoning quote recovers only a declared "
               "schema-valid call");
        Expect(content == item.content &&
                   Trimmed(reasoning) == Trimmed(item.reasoning),
               "quoted or rejected examples remain in their original phase");
        if (stream)
          Expect(streamed_content == content &&
                     Trimmed(streamed_reasoning) == Trimmed(reasoning),
                 "held reasoning and content agree with final output");
      }
  }
}

// Issue #383 follow-up: a model answering *about* the dialect names its tags,
// and that prose must survive as content. Captured from the live session whose
// reply was truncated at the first inline mention (assistant row 103597) while
// a phantom `terminal` call carrying "..." reached the client and ran.
void TestProseAboutTheDialectIsNotACall() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model",
    "messages":[{"role":"user","content":"why was it a bug"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"}},
      "required":["text"]}}}]
  })");
  const std::string message =
      "Got it — no comment posted. Here's the explanation, just for you.\n"
      "\n"
      "## Why it was a bug\n"
      "\n"
      "**The setup.** The Qwen chat model was trained to signal tool calls "
      "with special markup: `<tool_call>`. The server's job is to act as a "
      "**translator**: it watches the raw token stream, converts that markup "
      "into the structured tool_calls field of the OpenAI API, and makes sure "
      "the *rest* of the markup never appears as text.\n"
      "\n"
      "1. **Closing tags escaped into content.** When the call parsed, gufo "
      "removed the opening envelope but let the tail — `</parameter>`, "
      "`</invoke>`, `</function>` — escape into content.\n"
      "\n"
      "The client repeats `<invoke name=\"terminal\">` in history, so the "
      "model mimics that dialect. Quoted, the shape is:\n"
      "\n"
      "```\n"
      "<invoke name=\"f\">\n"
      "<parameter name=\"text\">x</parameter>\n"
      "</invoke>\n"
      "```\n"
      "\n"
      "The same holds for `<|tool_call|>` and for `<function=f>`. Nothing "
      "above is a call being made, and none of it may be cut.";
  for (bool stream : {false, true}) {
    for (bool bytewise : {false, true}) {
      auto body = schema;
      body["stream"] = stream;
      body["tool_choice"] = "auto";
      FakeBackend backend;
      if (bytewise) {
        for (char byte : message)
          backend.pieces.emplace_back(1, byte);
      } else {
        backend.pieces.push_back(message);
      }
      const auto response =
          gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
      Expect(response.status == 200, "an answer about the dialect is served");
      std::vector<Value> calls;
      std::string content;
      if (!stream) {
        const auto output = gufo::json::parse(response.body);
        const auto& choice =
            *output.find("choices")->items()[0].find("message");
        if (const auto* text = choice.find("content");
            text && text->is_string())
          content = text->get_str();
        if (const auto* found = choice.find("tool_calls"))
          calls.assign(found->items().begin(), found->items().end());
      } else {
        std::string output;
        response.streaming_body([&](std::string_view part) {
          output += part;
          return true;
        });
        std::size_t cursor = 0;
        while ((cursor = output.find("data: ", cursor)) != std::string::npos) {
          const auto begin = cursor + 6;
          cursor = output.find('\n', begin);
          const auto payload = output.substr(begin, cursor - begin);
          if (payload == "[DONE]")
            break;
          const auto event = gufo::json::parse(payload);
          const auto* choices = event.find("choices");
          if (!choices || choices->empty())
            continue;
          const auto* delta = choices->items()[0].find("delta");
          if (!delta)
            continue;
          content += delta->member_str("content");
          if (const auto* found = delta->find("tool_calls"))
            calls.insert(calls.end(), found->items().begin(),
                         found->items().end());
        }
      }
      if (calls.empty() && Trimmed(content) != Trimmed(message))
        std::cerr << "Prose about the dialect, stream=" << stream
                  << " bytewise=" << bytewise
                  << "\nExpected content: " << message
                  << "\nActual content: " << content << '\n';
      Expect(calls.empty(), "prose that names the dialect is not a call");
      Expect(Trimmed(content) == Trimmed(message),
             "an answer about the dialect survives in full");
    }
  }
}

/// A bracket-dense line, in units that never split across pieces.
std::string AngleUnits(std::size_t units) {
  std::string text;
  text.reserve(units * 3);
  for (std::size_t i = 0; i < units; ++i)
    text += "<a>";
  return text;
}

/// Bracket-dense content is ordinary text: every '<' in it is one probe for the
/// parser, and a held or dropped run shows up as text that lost its brackets.
///
/// This is the deterministic form of the live case. Asked to repeat such a
/// line, a model runs away into a repetition loop and never stops, so the live
/// case reports a model runaway rather than a framing failure.
void TestBracketDenseContent() {
  constexpr std::size_t kUnits = 2000;
  constexpr std::size_t kUnitsPerPiece = 30;
  const std::string dense = AngleUnits(kUnits);
  // The tail is what the framing rules act on: a pipe-shaped spelling the
  // vocabulary does not own is prose, whether it is complete or still arriving.
  const std::vector<std::string> tails{"", "<|not_a_vocab|>",
                                       "<|not_a_vocab_entry"};
  for (const std::string& tail : tails) {
    FakeBackend backend;
    for (std::size_t offset = 0; offset < dense.size();
         offset += kUnitsPerPiece * 3)
      backend.pieces.push_back(dense.substr(offset, kUnitsPerPiece * 3));
    if (!tail.empty())
      backend.pieces.push_back(tail);

    auto body = gufo::json::parse(
        R"({"model":"test-model","messages":[{"role":"user","content":"echo it"}],)"
        R"("reasoning_effort":"none"})");
    const auto buffered =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    const auto parsed = gufo::json::parse(buffered.body);
    const auto* message = parsed.find("choices")->items()[0].find("message");
    Expect(message->member_str("content") == dense + tail,
           "bracket-dense prose and its tail survive verbatim");

    body["stream"] = true;
    const auto streamed =
        gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
    std::string content;
    streamed.streaming_body([&](std::string_view chunk) {
      if (chunk == "data: [DONE]\n\n")
        return true;
      const auto event = gufo::json::parse(chunk.substr(6));
      for (const auto& choice : event.find("choices")->items()) {
        const auto* delta = choice.find("delta");
        if (delta)
          content += delta->member_str("content");
      }
      return true;
    });
    Expect(content == dense + tail,
           "the streaming path retains bracket-dense prose and its tail");
  }
}

int main() {
  TestHistoricalTypedArgumentsUseTojson();
  TestStreamingPromptProgress();
  TestResponsesPromptProgress();
  TestToolMarkersInsideConstrainedReasoning();
  TestNativeToolImplicitReasoningEnd();
  TestToolMarkersWhenToolsDisabled();
  TestBracketDenseContent();
  TestStopSequencesAndDefaultFields();
  TestStructuredResponseFormat();
  TestStructuredToolTruncation();
  TestStrictToolSchema();
  TestExplicitStopOutputFraming();
  TestStopInsideToolArguments();
  TestResponsesOutput();
  TestNativeToolTransports();
  TestJsonToolStringOwnership();
  TestNativeToolDialectSelection();
  TestNativeToolTextOutsideEnvelopes();
  TestNativeReferencedArgumentTypes();
  TestWildcardToolTypes();
  TestNativeUnionToolTypes();
  TestNativeArgumentTypingMatrix();
  TestToolMetadataAndFraming();
  TestResponsesLiveAndCancellation();
  TestCachePromptOption();
  TestToolChoiceEnforcement();
  TestStreamingIsLive();
  TestLessThanProseStreamsBeforeCompletion();
  TestStreamingWithoutUsage();
  TestUtf8Output();
  TestCachedPrefillMetrics();
  TestBackendSamplingDefaults();
  TestModelSamplingDefaults();
  TestCompleteToolDefinitionsReachTemplate();
  TestFlatToolFieldsReachTemplate();
  TestAllSamplingControlsReachBackend();
  TestUnsupportedSamplingControlsAreRejected();
  TestSamplingRanges();
  TestAssistantReasoningContentReachesBackend();
  TestPiReasoningControlsAndOutputFraming();
  TestPiNativeDeepSeekThinkingObject();
  TestStreamingPromptOpenedReasoning();
  TestInitialOutputPhases();
  TestReasoningAnswerSeparator();
  TestConflictingReasoningControlsAreRejected();
  TestToolCallsAreStructured();
  TestToolParameterCompatibility();
  TestInvalidToolsFailBeforeGeneration();
  TestResponsesClientCompatTolerances();
  TestMidConversationSystemMessagesKeepOrder();
  TestToolNameCharacters();
  TestMalformedHistoricalFunctions();
  TestToolClosingFraming();
  TestUnconstrainedQuoteFallbackChecksSchema();
  TestUnfinishedInlineReasoningFallback();
  TestProseAboutTheDialectIsNotACall();
  TestQwenToolBoundariesAndSchema();
  TestDeepSeekToolCallsAreStructured();
  TestDeepSeekRepeatedToolParameters();
  TestWrongModelIsRejected();
  TestClientIdentityReachesBackend();
  TestStreamingOverloadIsRejectedBeforeHeaders();
  TestImagePartsRetainOrderAndIdentity();
  TestImageCountAndByteBudget();
  TestToolImagesRetainOrderAndIdentity();
  std::cout << "All OpenAI chat protocol tests passed\n";
  return 0;
}
