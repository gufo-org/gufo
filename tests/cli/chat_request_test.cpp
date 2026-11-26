// Checks DecodeChatRequest, the C++ caller of the Rust chat request decoder
// (src/core/chat_request), and the three fragment shims openai_chat_test
// calls (ParseOpenAiResponseControls, ParseAnthropicToolControls,
// ParseOpenAiResponseMessage). A fake environment serves fixed model names,
// fails chosen image URLs, rejects chosen schemas and logs every model check,
// image read, schema compile and compatibility check. Each case gives the
// expected request (the fields that differ from their defaults) or rejection
// (status, code and message) and the log, under three server settings. The
// C++ caller fills the response format, stop sequences, sampling and
// logprobs, runs the compatibility check where each API asks for it, answers
// an image read that runs out of memory with 400 and rethrows a hook's other
// exceptions. The crate's Rust tests cover the decoder's own steps and its
// C ABI; openai_chat_test and http_server_test the endpoints.
//
// One difference from the 0.35.0 C++ decoders is kept on purpose: a rejected
// fragment leaves its request or message unchanged, where C++ could have
// filled part of it (TestFragments).
#include "src/cli/serve/chat_request.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/openai_chat.hpp"
#include "src/core/json.hpp"

namespace {

using gufo::json::Value;
using gufo::server::ChatApi;
using gufo::server::ChatDecodeEnvironment;
using gufo::server::ChatRequest;
using gufo::server::DecodedChatRequest;
using gufo::server::HttpResponse;
namespace tokenization = gufo::tokenization;

int failures = 0;
// Set CHAT_REQUEST_PRINT to print every case's result.
const bool print = std::getenv("CHAT_REQUEST_PRINT") != nullptr;

std::string Printable(std::string_view text) {
  std::string out;
  for (const unsigned char c : text) {
    if ((c < 0x20 && c != '\n') || c >= 0x7F) {
      char escape[8];
      std::snprintf(escape, sizeof(escape), "\\x%02x", c);
      out += escape;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

void Check(std::string_view name, const std::string& actual,
           std::string_view expected) {
  if (print)
    std::printf("== %.*s\n%s\n", static_cast<int>(name.size()), name.data(),
                Printable(actual).c_str());
  if (actual == expected)
    return;
  ++failures;
  std::fprintf(stderr, "FAIL %.*s\n  expected:\n%.*s\n  actual:\n%s\n",
               static_cast<int>(name.size()), name.data(),
               static_cast<int>(expected.size()), expected.data(),
               Printable(actual).c_str());
}

// The server settings a decode runs under.
struct Setting {
  bool images = true;
  gufo::ReasoningOptions defaults;
  // Provider sampling defaults, one temperature per thinking mode.
  bool provider = false;
  bool require_model = true;
};

// One decode's environment and the calls it made.
struct Fixture {
  std::vector<std::string> log;
  ChatDecodeEnvironment environment;

  explicit Fixture(const Setting& setting) {
    environment.model_id = "m";
    environment.accepts_images = setting.images;
    environment.reasoning_defaults = setting.defaults;
    environment.sampling_defaults.max_tokens = 77;
    if (setting.provider) {
      gufo::sampling::TextSamplingDefaults values;
      values.thinking_unspecified.temperature = 0.25F;
      values.thinking_enabled.temperature = 0.5F;
      values.thinking_disabled.temperature = 0.75F;
      environment.sampling_defaults.provider = values;
      // The model supplies no control, so the provider's apply.
      environment.sampling_defaults.supplied = {};
    }
    environment.serves_model = [this](std::string_view name) {
      log.push_back("model:" + std::string(name));
      if (name == "throw")
        throw std::runtime_error("backend gone");
      return name == "m" || name == "alias";
    };
    environment.read_image = [this](std::string_view url,
                                    gufo::core::ImageReadBudget& budget) {
      log.push_back("image:" + std::string(url));
      if (!budget.accepts_images)
        throw std::invalid_argument("model does not support image input");
      if (url.find("fail") != std::string_view::npos)
        throw std::invalid_argument("image fetch failed");
      if (url.find("oom") != std::string_view::npos)
        throw std::bad_alloc();
      return std::vector<std::uint8_t>(url.begin(), url.end());
    };
    environment.compile_schema = [this](const Value& schema) {
      const auto text = schema.dump(Value::Numbers::kAsParsed);
      log.push_back("schema:" + text);
      if (text.find("reject") != std::string::npos)
        throw std::invalid_argument("schema rejected");
    };
    // Stands in for ReadCompatibilityOptions (http_server.cpp), which
    // http_server_test checks: it gets the thinking mode, sets the limit and
    // resolves the sampling defaults.
    environment.compatibility = [this](ChatApi api, const Value& body,
                                       std::optional<bool> thinking,
                                       std::size_t* max_tokens,
                                       gufo::sampling::SamplingConfig* sampling)
        -> std::optional<HttpResponse> {
      log.push_back(std::string("compatibility:") +
                    (api == ChatApi::kResponses ? "responses" : "messages") +
                    ":" + (thinking ? (*thinking ? "1" : "0") : "-"));
      if (body.find("reject_compatibility") != nullptr)
        return gufo::server::ChatRejection(400, "compatibility rejected",
                                           "invalid_request");
      *max_tokens = 33;
      *sampling = environment.sampling_defaults.Resolve(thinking);
      return std::nullopt;
    };
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
};

void Put(std::ostringstream& out, std::string_view name,
         std::string_view value) {
  if (!value.empty())
    out << ' ' << name << '=' << value;
}

std::string Describe(const tokenization::ChatMessage& message) {
  std::ostringstream out;
  const char* roles[] = {"system", "developer", "user", "assistant", "tool"};
  out << "  " << roles[static_cast<int>(message.role)];
  Put(out, "content", message.content);
  Put(out, "suffix", message.framing_suffix);
  Put(out, "name", message.name);
  Put(out, "thought", message.thought);
  Put(out, "call_id", message.tool_call_id);
  for (const auto& image : message.images) {
    out << " image@" << image.offset
        << (image.detail == tokenization::ChatMessage::ImageDetail::kLow
                ? "/low"
                : "")
        << '='
        << (image.bytes ? std::string(image.bytes->begin(), image.bytes->end())
                        : "<null>");
  }
  for (const auto& call : message.tool_calls) {
    out << " call(" << call.id << ' ' << call.name;
    for (const auto& argument : call.arguments)
      out << ' ' << argument.name << (argument.is_string ? "=s:" : "=j:")
          << argument.value;
    out << (call.partial_arguments ? " partial" : "") << ')';
  }
  return out.str();
}

std::string Optional(const std::optional<bool>& value) {
  return value ? (*value ? "1" : "0") : "-";
}

// The fields of `chat` that differ from a default request, one per line.
std::string Describe(const ChatRequest& chat) {
  std::ostringstream out;
  for (const auto& message : chat.messages)
    out << Describe(message) << '\n';
  for (const auto& tool : chat.tools) {
    out << "  tool " << tool.name;
    Put(out, "description", tool.description);
    Put(out, "parameters", tool.parameters_json);
    Put(out, "definition", tool.definition_json);
    out << '\n';
  }
  const ChatRequest defaults;
  if (chat.client_id != defaults.client_id)
    out << "  client=" << chat.client_id << '\n';
  if (chat.tool_choice != defaults.tool_choice)
    out << "  choice=" << static_cast<int>(chat.tool_choice) << '\n';
  if (!chat.forced_tool_name.empty())
    out << "  forced=" << chat.forced_tool_name << '\n';
  for (const auto& [name, space] : chat.tool_namespaces)
    out << "  namespace " << name << '=' << space << '\n';
  if (chat.constrained_tools)
    out << "  constrained\n";
  if (!chat.parallel_tool_calls)
    out << "  serial\n";
  const auto& reasoning = chat.reasoning;
  if (reasoning.enabled || reasoning.effort || reasoning.preserve_thinking ||
      reasoning.budget_tokens)
    out << "  reasoning enabled=" << Optional(reasoning.enabled) << " effort="
        << (reasoning.effort
                ? std::to_string(static_cast<int>(*reasoning.effort))
                : "-")
        << " preserve=" << Optional(reasoning.preserve_thinking) << " budget="
        << (reasoning.budget_tokens ? std::to_string(*reasoning.budget_tokens)
                                    : "-")
        << '\n';
  if (chat.add_vision_id)
    out << "  vision_id\n";
  if (!chat.cache_prompt)
    out << "  no cache_prompt\n";
  if (!chat.cache_persist)
    out << "  no cache_persist\n";
  if (chat.return_progress)
    out << "  progress\n";
  for (const auto& stop : chat.stop_sequences)
    out << "  stop " << stop << '\n';
  if (chat.response_format != nullptr)
    out << "  format " << chat.response_format_description << '\n';
  return out.str();
}

// Sampling fields that differ from a default SamplingConfig.
std::string Describe(const gufo::sampling::SamplingConfig& s) {
  const gufo::sampling::SamplingConfig d;
  std::ostringstream out;
  if (s.logprobs)
    out << " logprobs=" << *s.logprobs;
  if (s.temperature != d.temperature)
    out << " t=" << s.temperature;
  if (s.top_k != d.top_k)
    out << " k=" << s.top_k;
  if (s.top_p != d.top_p)
    out << " p=" << s.top_p;
  if (s.min_p != d.min_p)
    out << " min_p=" << s.min_p;
  if (s.seed != d.seed)
    out << " seed=" << s.seed;
  if (s.repeat_penalty != d.repeat_penalty)
    out << " rp=" << s.repeat_penalty;
  if (s.frequency_penalty != d.frequency_penalty)
    out << " fp=" << s.frequency_penalty;
  if (s.presence_penalty != d.presence_penalty)
    out << " pp=" << s.presence_penalty;
  if (s.constraint != nullptr)
    out << " constraint";
  if (s.reasoning_budget != nullptr)
    out << " reasoning_budget";
  return out.str();
}

std::string Describe(const HttpResponse& response) {
  return "rejected " + std::to_string(response.status) + " " + response.reason +
         " " + response.body;
}

std::string Describe(const DecodedChatRequest& decoded) {
  std::ostringstream out;
  out << "accepted model=" << (decoded.model ? *decoded.model : "-") << " max="
      << (decoded.max_tokens ? std::to_string(*decoded.max_tokens) : "-")
      << (decoded.stream ? " stream" : "")
      << (decoded.include_usage ? " usage" : "") << '\n'
      << Describe(decoded.chat) << "  sampling" << Describe(decoded.sampling);
  return out.str();
}

std::string Decode(ChatApi api, std::string_view body, const Setting& setting) {
  Fixture fixture(setting);
  DecodedChatRequest decoded;
  std::string result;
  try {
    const auto rejection = gufo::server::DecodeChatRequest(
        api, body, fixture.environment, setting.require_model, &decoded);
    result = rejection ? Describe(*rejection) : Describe(decoded);
  } catch (const std::exception& exception) {
    result = std::string("threw ") + exception.what();
  }
  result += "\nlog:";
  for (const auto& entry : fixture.log)
    result += " " + entry;
  return result;
}

struct Case {
  const char* name;
  ChatApi api;
  Setting setting;
  std::string body;
  std::string expected;
};

Setting ThinkingServer() {
  Setting setting;
  setting.images = false;
  setting.defaults.enabled = true;
  setting.defaults.effort = gufo::ReasoningEffort::kHigh;
  setting.defaults.preserve_thinking = false;
  setting.defaults.budget_tokens = 100;
  setting.provider = true;
  return setting;
}

Setting OptionalModelServer() {
  Setting setting;
  setting.defaults.enabled = false;
  setting.provider = true;
  setting.require_model = false;
  return setting;
}

std::string User(std::string_view content) {
  return R"({"model":"m","messages":[{"role":"user","content":)" +
         std::string(content) + "}]";
}

// Chat Completions.
const std::vector<Case>& ChatCases() {
  static const std::vector<Case> cases = {
      {"chat: every field",
       ChatApi::kChat,
       {},
       R"({"model":"alias","messages":[
           {"role":"system","content":"s"},
           {"role":"user","content":[{"type":"text","text":"a"},
             {"type":"image_url","image_url":{"url":"u1","detail":"low"}},
             {"type":"text","text":"b"},
             {"type":"image_url","image_url":{"url":"u2"}}]},
           {"role":"assistant","content":null,"reasoning_content":"why",
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"f",
              "arguments":"{\"a\":1.0,\"b\":\"x\",\"c\":[1e2,-0]}"}}]},
           {"role":"tool","content":"r","tool_call_id":"c1","name":"n"}],
         "tools":[{"type":"function","function":{"name":"f","description":"d",
           "strict":true,"parameters":{"type":"object","properties":{"a":{"type":"number","minimum":1.0}}}}},
           {"type":"function","function":{"name":"g"}}],
         "tool_choice":{"type":"function","function":{"name":"f"}},
         "chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low",
           "preserve_thinking":true,"add_vision_id":true},
         "reasoning_budget_tokens":5.0,
         "max_completion_tokens":null,"max_tokens":2,
         "stream":true,"stream_options":{"include_usage":true},
         "stop":["a","b"],
         "response_format":{"type":"json_schema","json_schema":{"name":"n","description":"fd","schema":{"type":"object"}}},
         "temperature":0.5,"top_p":0.9,"top_k":4,"seed":3,
         "logprobs":true,"top_logprobs":2,
         "cache_prompt":false,"cache_persist":false,"return_progress":true})",
       "accepted model=alias max=2 stream usage\n"
       "  system content=s\n"
       "  user content=ab image@1/low=u1 image@2=u2\n"
       "  assistant thought=why call(c1 f a=j:1.0 b=s:x c=j:[1e2, -0])\n"
       "  tool content=r name=n call_id=c1\n"
       "  tool f description=d "
       "parameters={\"type\":\"object\",\"properties\":{\"a\":{\"type\":"
       "\"number\",\"minimum\":1}}} "
       "definition={\"type\":\"function\",\"function\":{\"name\":\"f\","
       "\"description\":\"d\",\"strict\":true,\"parameters\":{\"type\":"
       "\"object\",\"properties\":{\"a\":{\"type\":\"number\",\"minimum\":1}}}}"
       "}\n"
       "  choice=2\n"
       "  forced=f\n"
       "  constrained\n"
       "  serial\n"
       "  reasoning enabled=1 effort=1 preserve=1 budget=5\n"
       "  vision_id\n"
       "  no cache_prompt\n"
       "  no cache_persist\n"
       "  progress\n"
       "  stop a\n"
       "  stop b\n"
       "  format fd\n"
       "  sampling logprobs=2 t=0.5 k=4 p=0.9 seed=3\n"
       "log: model:alias image:u1 image:u2 "
       "schema:{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\","
       "\"minimum\":1.0}}}"},
      {"chat: optional model and defaults", ChatApi::kChat,
       OptionalModelServer(),
       R"({"messages":[{"role":"user","content":"hi"}]})",
       "accepted model=m max=-\n"
       "  user content=hi\n"
       "  reasoning enabled=0 effort=- preserve=- budget=-\n"
       "  sampling t=0.75\n"
       "log: model:m"},
      {"chat: thinking defaults", ChatApi::kChat, ThinkingServer(),
       R"({"model":"m","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}})",
       "accepted model=m max=-\n"
       "  user content=hi\n"
       "  reasoning enabled=0 effort=3 preserve=0 budget=100\n"
       "  sampling t=0.75\n"
       "log: model:m"},
      {"chat: namespace tools",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f"}]}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"only function tools "
       "are "
       "supported\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "tools\"}}\n"
       "log: model:m"},
      {"chat: parse error",
       ChatApi::kChat,
       {},
       "{",
       "rejected 400 Bad Request {\"error\":{\"message\":\"JSON parse error: "
       "expected object key "
       "string\",\"type\":\"invalid_request_error\",\"code\":\"parse_error\"}}"
       "\n"
       "log:"},
      {"chat: not an object",
       ChatApi::kChat,
       {},
       "[]",
       "rejected 400 Bad Request {\"error\":{\"message\":\"request body must "
       "be a JSON "
       "object\",\"type\":\"invalid_request_error\",\"code\":\"invalid_body\"}}"
       "\n"
       "log:"},
      {"chat: empty body",
       ChatApi::kChat,
       {},
       "",
       "rejected 400 Bad Request {\"error\":{\"message\":\"JSON parse error: "
       "unexpected end of "
       "input\",\"type\":\"invalid_request_error\",\"code\":\"parse_error\"}}\n"
       "log:"},
      {"chat: unknown model",
       ChatApi::kChat,
       {},
       R"({"model":"x","messages":[]})",
       "rejected 404 Not Found {\"error\":{\"message\":\"model 'x' is not "
       "served by this "
       "process\",\"type\":\"invalid_request_error\",\"code\":\"model_not_"
       "found\"}}\n"
       "log: model:x"},
      {"chat: model required",
       ChatApi::kChat,
       {},
       R"({"messages":[{"role":"user","content":"hi"}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'model' is "
       "required\",\"type\":\"invalid_request_error\",\"code\":\"missing_"
       "model\"}}\n"
       "log:"},
      {"chat: no messages",
       ChatApi::kChat,
       {},
       R"({"model":"m","messages":[]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'messages' must be "
       "a non-empty "
       "array\",\"type\":\"invalid_request_error\",\"code\":\"missing_"
       "messages\"}}\n"
       "log: model:m"},
      {"chat: unknown role",
       ChatApi::kChat,
       {},
       R"({"model":"m","messages":[{"role":"bot","content":"x"}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"message role must "
       "be system, developer, user, assistant, or "
       "tool\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "messages\"}}\n"
       "log: model:m"},
      {"chat: assistant image",
       ChatApi::kChat,
       {},
       R"({"model":"m","messages":[{"role":"assistant","content":[{"type":"image_url","image_url":{"url":"u"}}]}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"image_url requires "
       "a user or tool message and a string "
       "URL\",\"type\":\"invalid_request_error\",\"code\":\"invalid_messages\"}"
       "}\n"
       "log: model:m"},
      {"chat: image detail",
       ChatApi::kChat,
       {},
       User(
           R"([{"type":"image_url","image_url":{"url":"u","detail":"tiny"}}])") +
           "}",
       "rejected 400 Bad Request {\"error\":{\"message\":\"image_url.detail "
       "must be 'auto', 'low', 'high' or "
       "'original'\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "messages\"}}\n"
       "log: model:m"},
      {"chat: failed image",
       ChatApi::kChat,
       {},
       User(R"([{"type":"image_url","image_url":{"url":"fail"}}])") + "}",
       "rejected 400 Bad Request {\"error\":{\"message\":\"image fetch "
       "failed\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "messages\"}}\n"
       "log: model:m image:fail"},
      {"chat: image out of memory",
       ChatApi::kChat,
       {},
       User(R"([{"type":"image_url","image_url":{"url":"oom"}}])") + "}",
       // The message is the standard library's what(), which differs
       // between libstdc++ and the MSVC STL.
       std::string("rejected 400 Bad Request {\"error\":{\"message\":\"") +
           std::bad_alloc().what() +
           "\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
           "messages\"}}\n"
           "log: model:m image:oom"},
      {"chat: text-only model", ChatApi::kChat, ThinkingServer(),
       User(R"([{"type":"image_url","image_url":{"url":"u"}}])") + "}",
       "rejected 400 Bad Request {\"error\":{\"message\":\"model does not "
       "support image "
       "input\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "messages\"}}\n"
       "log: model:m image:u"},
      {"chat: rejected strict schema",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"reject":1}}}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"schema "
       "rejected\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "tools\"}}\n"
       "log: model:m schema:{\"reject\":1}"},
      {"chat: repeated tool",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"tools":[{"type":"function","function":{"name":"f"}},{"type":"function","function":{"name":"f"}}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"function names must "
       "be "
       "unique\",\"type\":\"invalid_request_error\",\"code\":\"invalid_tools\"}"
       "}\n"
       "log: model:m"},
      {"chat: tool name",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"tools":[{"type":"function","function":{"name":"bad name"}}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"function names "
       "require 1-64 printable ASCII characters other than spaces, '<', '>', "
       "'\\\"' and "
       "'\\\\'\",\"type\":\"invalid_request_error\",\"code\":\"invalid_tools\"}"
       "}\n"
       "log: model:m"},
      {"chat: historical call arguments",
       ChatApi::kChat,
       {},
       R"({"model":"m","messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"tool arguments must "
       "encode a JSON "
       "object\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "messages\"}}\n"
       "log: model:m"},
      {"chat: required without tools",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"tool_choice":"required"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'tool_choice' "
       "cannot be required without "
       "tools\",\"type\":\"invalid_request_error\",\"code\":\"invalid_tool_"
       "choice\"}}\n"
       "log: model:m"},
      {"chat: choice names no tool",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":{"type":"function","function":{"name":"missing"}}})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'tool_choice' must "
       "name a declared "
       "function\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "tools\"}}\n"
       "log: model:m"},
      {"chat: conflicting reasoning",
       ChatApi::kChat,
       {},
       User(R"("hi")") +
           R"(,"thinking":{"type":"disabled"},"reasoning_effort":"high"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"reasoning controls "
       "disagree about whether thinking is "
       "enabled\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "reasoning\"}}\n"
       "log: model:m"},
      {"chat: reasoning budget",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"reasoning_budget_tokens":2147483648})",
       "rejected 400 Bad Request "
       "{\"error\":{\"message\":\"'reasoning_budget_tokens' must be an integer "
       "between -1 and "
       "2147483647\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "reasoning\"}}\n"
       "log: model:m"},
      {"chat: max tokens",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"max_tokens":4294967296})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'max_tokens' must "
       "be a positive "
       "integer\",\"type\":\"invalid_request_error\",\"code\":\"invalid_max_"
       "tokens\"}}\n"
       "log: model:m"},
      {"chat: stream",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"stream":"yes"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'stream' must be a "
       "boolean\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "stream\"}}\n"
       "log: model:m"},
      {"chat: n",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"n":2})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"only n=1 is "
       "supported\",\"type\":\"invalid_request_error\",\"code\":\"unsupported_"
       "n\"}}\n"
       "log: model:m"},
      {"chat: audio",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"modalities":["audio"]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"request field "
       "'modalities' is not "
       "implemented\",\"type\":\"invalid_request_error\",\"code\":"
       "\"unsupported_field\"}}\n"
       "log: model:m"},
      {"chat: response format",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"response_format":{"type":"xml"}})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"response_format "
       "type must be text, json_object or "
       "json_schema\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "response_format\"}}\n"
       "log: model:m"},
      {"chat: stop",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"stop":[1]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"stop sequences must "
       "be "
       "strings\",\"type\":\"invalid_request_error\",\"code\":\"invalid_stop\"}"
       "}\n"
       "log: model:m"},
      {"chat: temperature",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"temperature":"hot"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'temperature' must "
       "be a finite "
       "number\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "temperature\"}}\n"
       "log: model:m"},
      {"chat: logprobs",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"logprobs":"x"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'logprobs' must be "
       "a "
       "boolean\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "logprobs\"}}\n"
       "log: model:m"},
      {"chat: mirostat",
       ChatApi::kChat,
       {},
       User(R"("hi")") + R"(,"mirostat":1})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"request field "
       "'mirostat' is not supported with a value that enables "
       "it\",\"type\":\"invalid_request_error\",\"code\":\"unsupported_"
       "sampling\"}}\n"
       "log: model:m"},
      {"chat: model check throws",
       ChatApi::kChat,
       {},
       R"({"model":"throw","messages":[]})",
       "threw backend gone\n"
       "log: model:throw"},
  };
  return cases;
}

// Responses and Messages, with the compatibility check.
const std::vector<Case>& CompatibilityCases() {
  static const std::vector<Case> cases = {
      {"responses: every field",
       ChatApi::kResponses,
       {},
       R"({"model":"m","instructions":"sys","input":[
           {"type":"reasoning","summary":[{"type":"summary_text","text":"t"}]},
           {"role":"developer","content":"d"},
           {"role":"assistant","content":[{"type":"output_text","text":"answer"}]},
           {"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1.0}"},
           {"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"l "},{"type":"input_image","image_url":"u3"}]},
           {"role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_image","image_url":"u1","detail":"low"}]}],
         "tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f"},{"type":"web_search"}]},
           {"type":"function","name":"g","strict":true}],
         "tool_choice":"required","reasoning":{"effort":"high","summary":"auto"},
         "text":{"format":{"type":"json_schema","name":"n","description":"d","schema":{"type":"object"}}},
         "include":["message.output_text.logprobs"],"top_logprobs":3,
         "stream":true,"store":false})",
       "accepted model=m max=33 stream\n"
       "  system content=sys\n"
       "  assistant content=answer thought=t call(c1 f a=j:1.0)\n"
       "  developer content=d\n"
       "  tool content=l  call_id=c1 image@2=u3\n"
       "  user content=a image@1/low=u1\n"
       "  tool f parameters={} "
       "definition={\"type\":\"function\",\"function\":{\"name\":\"f\","
       "\"parameters\":{}}}\n"
       "  tool g "
       "parameters={\"type\":\"object\",\"properties\":{},\"required\":[],"
       "\"additionalProperties\":false} "
       "definition={\"type\":\"function\",\"function\":{\"name\":\"g\","
       "\"strict\":true,\"parameters\":{\"type\":\"object\",\"properties\":{},"
       "\"required\":[],\"additionalProperties\":false}}}\n"
       "  choice=2\n"
       "  namespace f=ns\n"
       "  constrained\n"
       "  reasoning enabled=1 effort=3 preserve=- budget=-\n"
       "  format d\n"
       "  sampling logprobs=3\n"
       "log: "
       "schema:{\"type\":\"object\",\"properties\":{},\"required\":[],"
       "\"additionalProperties\":false} compatibility:responses:1 image:u3 "
       "image:u1"},
      {"responses: thinking defaults", ChatApi::kResponses, ThinkingServer(),
       R"({"input":"hello","reasoning":{"effort":"none"}})",
       "accepted model=- max=33\n"
       "  user content=hello\n"
       "  reasoning enabled=0 effort=3 preserve=0 budget=100\n"
       "  sampling t=0.75\n"
       "log: compatibility:responses:0"},
      {"responses: empty input",
       ChatApi::kResponses,
       {},
       R"({"input":""})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'input' must "
       "contain text, message items with text/images, reasoning items, "
       "function calls or function "
       "outputs\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:responses:-"},
      {"responses: failed image",
       ChatApi::kResponses,
       {},
       R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":"fail"}]}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"image fetch "
       "failed\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:responses:- image:fail"},
      {"responses: historical call arguments",
       ChatApi::kResponses,
       {},
       R"({"input":[{"type":"function_call","call_id":"c3","name":"f","arguments":"nope"}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"tool arguments are "
       "not valid JSON: JSON parse error: bad "
       "literal\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:responses:-"},
      {"responses: namespace name",
       ChatApi::kResponses,
       {},
       R"({"input":"x","tools":[{"type":"namespace","name":"","tools":[{"type":"function","name":"f"}]}]})",
       "accepted model=- max=33\n"
       "  user content=x\n"
       "  tool f parameters={} "
       "definition={\"type\":\"function\",\"function\":{\"name\":\"f\","
       "\"parameters\":{}}}\n"
       "  constrained\n"
       "  sampling\n"
       "log: compatibility:responses:-"},
      {"responses: compatibility rejection",
       ChatApi::kResponses,
       {},
       R"({"input":"x","reject_compatibility":1})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"compatibility "
       "rejected\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:responses:-"},
      {"responses: logprobs",
       ChatApi::kResponses,
       {},
       R"({"input":"x","include":"x"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'include' must be "
       "an array of "
       "strings\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "include\"}}\n"
       "log: compatibility:responses:-"},
      {"responses: text format",
       ChatApi::kResponses,
       {},
       R"({"input":"x","text":{"format":{"type":"json_schema","name":"bad name","schema":{}}}})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"json_schema.name "
       "must contain 1\xe2"
       "\x80"
       "\x93"
       "64 letters, digits, '_' or "
       "'-'\",\"type\":\"invalid_request_error\",\"code\":\"invalid_response_"
       "format\"}}\n"
       "log:"},
      {"messages: every field",
       ChatApi::kMessages,
       {},
       R"({"model":"m","system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],
         "thinking":{"type":"enabled","budget_tokens":9},
         "output_config":{"effort":"high"},
         "tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}],
         "tool_choice":{"type":"any","disable_parallel_tool_use":true},
         "stop_sequences":["x"],
         "messages":[
           {"role":"user","content":[{"type":"text","text":"a"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},{"type":"image","source":{"type":"url","url":"u1"}}]},
           {"role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"tool_use","id":"c1","name":"f","input":{"a":1.0,"b":[1e2]}}]},
           {"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"c1","is_error":true,"content":[{"type":"text","text":"x"},{"type":"image","source":{"type":"url","url":"u3"}}]},{"type":"text","text":"after"}]}]})",
       "accepted model=m max=33\n"
       "  system content=ab\n"
       "  user content=a image@1=data:image/png;base64,AQID image@1=u1\n"
       "  assistant thought=t call(c1 f a=j:1 b=j:[100])\n"
       "  user content=before\n"
       "  tool content=x call_id=c1 image@1=u3\n"
       "  user content=after\n"
       "  tool f description=d "
       "parameters={\"type\":\"object\",\"properties\":{}} "
       "definition={\"type\":\"function\",\"function\":{\"name\":\"f\","
       "\"description\":\"d\",\"parameters\":{\"type\":\"object\","
       "\"properties\":{}}}}\n"
       "  choice=2\n"
       "  constrained\n"
       "  serial\n"
       "  reasoning enabled=1 effort=3 preserve=- budget=9\n"
       "  stop x\n"
       "  sampling\n"
       "log: compatibility:messages:1 image:data:image/png;base64,AQID "
       "image:u1 image:u3"},
      {"messages: thinking defaults", ChatApi::kMessages, ThinkingServer(),
       R"({"model":"m","messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive"}})",
       "accepted model=m max=33\n"
       "  user content=hi\n"
       "  reasoning enabled=1 effort=3 preserve=0 budget=100\n"
       "  sampling t=0.5\n"
       "log: compatibility:messages:1"},
      {"messages: thinking type",
       ChatApi::kMessages,
       {},
       R"({"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"sometimes"}})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'thinking.type' "
       "must be enabled, adaptive, or "
       "disabled\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log:"},
      {"messages: choice names no tool",
       ChatApi::kMessages,
       {},
       R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":{}}],"tool_choice":{"type":"tool","name":"missing"}})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'tool_choice' must "
       "name a declared "
       "function\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "tools\"}}\n"
       "log: compatibility:messages:-"},
      {"messages: tool use without a name",
       ChatApi::kMessages,
       {},
       R"({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"c2","input":{}}]}]})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"historical function "
       "calls require a non-empty name and string "
       "arguments\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:messages:-"},
      {"messages: stop sequences",
       ChatApi::kMessages,
       {},
       R"({"messages":[{"role":"user","content":"hi"}],"stop_sequences":"x"})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'stop_sequences' "
       "must be an array of "
       "strings\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:messages:-"},
      {"messages: compatibility rejection",
       ChatApi::kMessages,
       {},
       R"({"messages":[{"role":"user","content":"hi"}],"reject_compatibility":1})",
       "rejected 400 Bad Request {\"error\":{\"message\":\"compatibility "
       "rejected\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:messages:-"},
      {"responses: not an object",
       ChatApi::kResponses,
       {},
       "1",
       "rejected 400 Bad Request {\"error\":{\"message\":\"'input' must "
       "contain text, message items with text/images, reasoning items, "
       "function calls or function "
       "outputs\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
       "request\"}}\n"
       "log: compatibility:responses:-"},
      {"messages: parse error",
       ChatApi::kMessages,
       {},
       "{",
       "rejected 400 Bad Request {\"error\":{\"message\":\"JSON parse error: "
       "expected object key "
       "string\",\"type\":\"invalid_request_error\",\"code\":\"parse_error\"}}"
       "\n"
       "log:"},
  };
  return cases;
}

void TestDecoding() {
  for (const auto* cases : {&ChatCases(), &CompatibilityCases()})
    for (const Case& c : *cases)
      Check(c.name, Decode(c.api, c.body, c.setting), c.expected);
}

// A request with tools and choices the fragments start from.
ChatRequest Seed() {
  ChatRequest chat;
  tokenization::ChatTool tool;
  tool.name = "f";
  chat.tools.push_back(tool);
  chat.tool_choice = ChatRequest::ToolChoice::kRequired;
  chat.forced_tool_name = "f";
  chat.reasoning.enabled = true;
  chat.reasoning.effort = gufo::ReasoningEffort::kLow;
  return chat;
}

std::string Fragment(bool anthropic, std::string_view text) {
  ChatRequest chat = Seed();
  const Value body = gufo::json::parse(text);
  const auto rejection =
      anthropic ? gufo::server::ParseAnthropicToolControls(body, &chat)
                : gufo::server::ParseOpenAiResponseControls(body, &chat);
  return (rejection ? Describe(*rejection) + "\n" : std::string("accepted\n")) +
         Describe(chat);
}

std::string Item(std::string_view text) {
  tokenization::ChatMessage message;
  message.content = "unchanged";
  gufo::core::ImageReadBudget budget;
  std::string error;
  const bool accepted = gufo::server::ParseOpenAiResponseMessage(
      gufo::json::parse(text), &message, budget, &error);
  return (accepted ? "accepted" : "rejected " + error) + "\n" +
         Describe(message);
}

// The shims through the production image reader and schema compiler. A
// rejected fragment leaves its request or message as it was.
void TestFragments() {
  Check(
      "response controls",
      Fragment(
          false,
          R"({"tool_choice":{"type":"function","name":"f"},"reasoning":{"effort":"high"},"parallel_tool_calls":false})"),
      "accepted\n"
      "  tool f parameters={}\n"
      "  choice=2\n"
      "  forced=f\n"
      "  constrained\n"
      "  serial\n"
      "  reasoning enabled=1 effort=3 preserve=- budget=-\n");
  Check(
      "response controls: rejected",
      Fragment(
          false,
          R"({"reasoning":{"effort":"high"},"tool_choice":{"type":"function","name":"missing"}})"),
      "rejected 400 Bad Request {\"error\":{\"message\":\"'tool_choice' must "
      "name a declared "
      "function\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
      "tools\"}}\n"
      "  tool f parameters={}\n"
      "  choice=2\n"
      "  forced=f\n"
      "  reasoning enabled=1 effort=1 preserve=- budget=-\n");
  Check(
      "anthropic tools",
      Fragment(
          true,
          R"({"tools":[{"name":"g","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"g"}})"),
      "accepted\n"
      "  tool g parameters={\"type\":\"object\"} "
      "definition={\"type\":\"function\",\"function\":{\"name\":\"g\","
      "\"parameters\":{\"type\":\"object\"}}}\n"
      "  choice=2\n"
      "  forced=g\n"
      "  constrained\n"
      "  serial\n"
      "  reasoning enabled=1 effort=1 preserve=- budget=-\n");
  Check(
      "anthropic tools: rejected",
      Fragment(
          true,
          R"({"tools":[{"name":"g"},{"name":"bad name"}],"tool_choice":{"type":"auto"}})"),
      "rejected 400 Bad Request {\"error\":{\"message\":\"only custom tools "
      "with an input_schema are "
      "supported\",\"type\":\"invalid_request_error\",\"code\":\"invalid_"
      "tools\"}}\n"
      "  tool f parameters={}\n"
      "  choice=2\n"
      "  forced=f\n"
      "  reasoning enabled=1 effort=1 preserve=- budget=-\n");
  Check(
      "response item",
      Item(
          R"({"type":"function_call","call_id":"c","name":"f","arguments":"{\"b\":[1,{\"c\":2.50}]}"})"),
      "accepted\n"
      "  assistant call(c f b=j:[1, {\"c\": 2.50}])");
  Check("response item: rejected",
        Item(R"({"type":"function_call_output","call_id":"c1","output":5})"),
        "rejected function output requires a string or content array\n"
        "  user content=unchanged");
}

}  // namespace

int main() {
  try {
    TestDecoding();
    TestFragments();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d chat request checks failed\n", failures);
    return 1;
  }
  std::printf("chat request checks passed\n");
  return 0;
}
