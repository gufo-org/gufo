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

#include "src/core/json.hpp"

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
  std::string cache_miss_reason;
  FinishReason finish_reason{FinishReason::kStop};
  std::string stop_sequence;
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

void TestQwenToolBoundariesAndSchema() {
  using gufo::json::Value;
  const auto schema = gufo::json::parse(R"({
    "model":"test-model", "messages":[{"role":"user","content":"use f"}],
    "tools":[{"type":"function","function":{"name":"f","parameters":{
      "type":"object","properties":{"text":{"type":"string"},
      "count":{"type":"integer"}}}}}]
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
        const auto text = (reasoning ? "Considering. " : "") + item.text;
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
        Expect(calls.size() == item.calls, "only complete tool calls emitted");
        if (!calls.empty())
          Expect(calls.back().find("function")->member_str("arguments") ==
                     item.argument,
                 "tool argument values and schema types preserved");
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
      "name":"f","description":"","parameters":{"type":"object"},
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

void TestAggregateImageLimit() {
  FakeBackend backend;
  auto body = gufo::json::Value::object();
  body["model"] = "test-model";
  body["messages"] = gufo::json::Value::array();
  const auto message = gufo::json::parse(R"({"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]})");
  for (unsigned i = 0; i < 17; ++i)
    body["messages"].push_back(message);
  const auto response =
      gufo::server::HandleOpenAiChat(Request(body.dump()), backend);
  Expect(response.status == 400,
         "image budget must cover all messages, not each separately");
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
        R"({"response_format":{"type":"json_object"}})",
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

}  // namespace

int main() {
  TestStopSequencesAndDefaultFields();
  TestExplicitStopOutputFraming();
  TestStopInsideToolArguments();
  TestCachePromptOption();
  TestToolChoiceEnforcement();
  TestStreamingIsLive();
  TestStreamingWithoutUsage();
  TestUtf8Output();
  TestCachedPrefillMetrics();
  TestBackendSamplingDefaults();
  TestCompleteToolDefinitionsReachTemplate();
  TestFlatToolFieldsReachTemplate();
  TestAllSamplingControlsReachBackend();
  TestUnsupportedSamplingControlsAreRejected();
  TestAssistantReasoningContentReachesBackend();
  TestPiReasoningControlsAndOutputFraming();
  TestPiNativeDeepSeekThinkingObject();
  TestStreamingPromptOpenedReasoning();
  TestConflictingReasoningControlsAreRejected();
  TestToolCallsAreStructured();
  TestToolParameterCompatibility();
  TestInvalidToolsFailBeforeGeneration();
  TestQwenToolBoundariesAndSchema();
  TestDeepSeekToolCallsAreStructured();
  TestWrongModelIsRejected();
  TestClientIdentityReachesBackend();
  TestStreamingOverloadIsRejectedBeforeHeaders();
  TestImagePartsRetainOrderAndIdentity();
  TestAggregateImageLimit();
  std::cout << "All OpenAI chat protocol tests passed\n";
  return 0;
}
