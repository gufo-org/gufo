// Compares the Rust chat request decoder (src/core/chat_request, through
// src/cli/serve/chat_request.cpp) with frozen copies of the C++ decoders it
// replaced (tests/cli/chat_request_reference.cpp). Both run through the same
// fake environment, which serves fixed model names, logs every model check,
// image read, schema compile and compatibility check, fails chosen image
// URLs and rejects chosen schemas. Generated grids of Chat Completions,
// Responses and Messages bodies, seeded mutations of them and raw byte edits
// must give the same request, rejection (status, code and message) and log.
// The fragment shims (ParseOpenAiResponseControls,
// ParseAnthropicToolControls, ParseOpenAiResponseMessage) are compared too.
// The only difference allowed: a rejected fragment leaves its request or
// message unchanged, where the C++ one could have filled part of it.
#include "src/cli/serve/chat_request.hpp"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/openai_chat.hpp"
#include "src/core/json.hpp"
#include "tests/cli/chat_request_reference.hpp"

namespace {

using gufo::json::Value;
using gufo::server::ChatApi;
using gufo::server::ChatDecodeEnvironment;
using gufo::server::DecodedChatRequest;
using gufo::server::HttpResponse;
namespace reference = gufo::server::chat_request_reference;
namespace tokenization = gufo::tokenization;

int failures = 0;
int cases = 0;
// Accepted bodies per API, so a grid that only exercises rejections shows.
int accepted[3] = {};

std::string Printable(std::string_view text, std::size_t limit = 600) {
  std::string out;
  for (const unsigned char c : text.substr(0, limit)) {
    if (c < 0x20 || c >= 0x7F) {
      char escape[8];
      std::snprintf(escape, sizeof(escape), "\\x%02x", c);
      out += escape;
    } else {
      out += static_cast<char>(c);
    }
  }
  if (text.size() > limit)
    out += "...";
  return out;
}

// One decode's environment and the calls it made.
struct Fixture {
  std::vector<std::string> log;
  ChatDecodeEnvironment environment;

  Fixture(bool images, gufo::ReasoningOptions defaults, bool provider) {
    environment.model_id = "m";
    environment.accepts_images = images;
    environment.reasoning_defaults = defaults;
    environment.sampling_defaults.max_tokens = 77;
    if (provider) {
      gufo::sampling::TextSamplingDefaults values;
      values.thinking_unspecified.temperature = 0.25F;
      values.thinking_enabled.temperature = 0.5F;
      values.thinking_disabled.temperature = 0.75F;
      environment.sampling_defaults.provider = values;
    }
    environment.serves_model = [this](std::string_view name) {
      log.push_back("model:" + std::string(name));
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
    environment.compatibility =
        [this](ChatApi api, const Value& body, std::optional<bool> thinking,
               std::size_t* max_tokens,
               gufo::sampling::SamplingConfig* sampling) {
          log.push_back("compatibility:" +
                        std::string(thinking ? (*thinking ? "1" : "0") : "-"));
          return reference::Compatibility(api, body, thinking, max_tokens,
                                          sampling, environment);
        };
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
};

void Put(std::ostringstream& out, std::string_view name,
         std::string_view value) {
  out << name << '=' << value.size() << ':' << value << ';';
}

std::string Describe(const tokenization::ChatMessage& message) {
  std::ostringstream out;
  out << "role=" << static_cast<int>(message.role) << ';';
  Put(out, "content", message.content);
  Put(out, "suffix", message.framing_suffix);
  Put(out, "name", message.name);
  Put(out, "thought", message.thought);
  Put(out, "call_id", message.tool_call_id);
  for (const auto& image : message.images) {
    out << "image@" << image.offset << '/' << static_cast<int>(image.detail);
    Put(out, "",
        image.bytes ? std::string(image.bytes->begin(), image.bytes->end())
                    : "<null>");
  }
  for (const auto& call : message.tool_calls) {
    Put(out, "call", call.id);
    Put(out, "fn", call.name);
    for (const auto& argument : call.arguments) {
      Put(out, "arg", argument.name);
      Put(out, argument.is_string ? "s" : "j", argument.value);
    }
    out << (call.partial_arguments ? "partial;" : "");
  }
  return out.str();
}

std::string Describe(const gufo::server::ChatRequest& chat) {
  std::ostringstream out;
  for (const auto& message : chat.messages)
    out << "[" << Describe(message) << "]\n";
  for (const auto& tool : chat.tools) {
    Put(out, "tool", tool.name);
    Put(out, "description", tool.description);
    Put(out, "parameters", tool.parameters_json);
    Put(out, "definition", tool.definition_json);
    out << '\n';
  }
  Put(out, "client", chat.client_id);
  out << "choice=" << static_cast<int>(chat.tool_choice) << ';';
  Put(out, "forced", chat.forced_tool_name);
  for (const auto& [name, space] : chat.tool_namespaces) {
    Put(out, "ns", name);
    Put(out, "in", space);
  }
  const auto& reasoning = chat.reasoning;
  out << "constrained=" << chat.constrained_tools
      << ";parallel=" << chat.parallel_tool_calls << ";enabled="
      << (reasoning.enabled ? std::to_string(*reasoning.enabled) : "-")
      << ";effort="
      << (reasoning.effort ? std::to_string(static_cast<int>(*reasoning.effort))
                           : "-")
      << ";preserve="
      << (reasoning.preserve_thinking
              ? std::to_string(*reasoning.preserve_thinking)
              : "-")
      << ";budget="
      << (reasoning.budget_tokens ? std::to_string(*reasoning.budget_tokens)
                                  : "-")
      << ";vision=" << chat.add_vision_id << ";cache=" << chat.cache_prompt
      << chat.cache_persist << ";progress=" << chat.return_progress << ';';
  for (const auto& stop : chat.stop_sequences)
    Put(out, "stop", stop);
  out << "format=" << (chat.response_format != nullptr) << ';';
  Put(out, "format_description", chat.response_format_description);
  return out.str();
}

std::string Describe(const gufo::sampling::SamplingConfig& s) {
  std::ostringstream out;
  out << "logprobs=" << (s.logprobs ? std::to_string(*s.logprobs) : "-")
      << ";t=" << s.temperature << ";k=" << s.top_k << ";p=" << s.top_p
      << ";min_p=" << s.min_p << ";keep=" << s.min_keep << ";seed=" << s.seed
      << ";rp=" << s.repeat_penalty << ";rn=" << s.repeat_last_n
      << ";fp=" << s.frequency_penalty << ";pp=" << s.presence_penalty
      << ";constraint=" << (s.constraint != nullptr)
      << ";budget=" << (s.reasoning_budget != nullptr);
  return out.str();
}

std::string Describe(const HttpResponse& response) {
  return "rejected " + std::to_string(response.status) + " " + response.reason +
         " " + response.body;
}

std::string Describe(const std::optional<HttpResponse>& rejection,
                     const DecodedChatRequest& decoded) {
  if (rejection)
    return Describe(*rejection);
  std::ostringstream out;
  out << "accepted\n" << Describe(decoded.chat) << '\n';
  Put(out, "model", decoded.model ? *decoded.model : "<none>");
  out << "max="
      << (decoded.max_tokens ? std::to_string(*decoded.max_tokens) : "-")
      << ";stream=" << decoded.stream << ";usage=" << decoded.include_usage
      << '\n'
      << Describe(decoded.sampling);
  return out.str();
}

std::string Run(Fixture& fixture, bool rust, ChatApi api, std::string_view body,
                bool require_model) {
  DecodedChatRequest decoded;
  std::string result;
  try {
    const auto rejection =
        rust ? gufo::server::DecodeChatRequest(api, body, fixture.environment,
                                               require_model, &decoded)
             : reference::Decode(api, body, fixture.environment, require_model,
                                 &decoded);
    result = Describe(rejection, decoded);
  } catch (const std::exception& exception) {
    result = std::string("threw ") + exception.what();
  }
  result += "\nlog:";
  for (const auto& entry : fixture.log)
    result += "\n  " + entry;
  return result;
}

void Report(std::string_view what, std::string_view input,
            const std::string& old_result, const std::string& new_result) {
  ++failures;
  if (failures > 10)
    return;
  std::fprintf(stderr, "FAIL %.*s\n  body: %s\n  C++:  %s\n  Rust: %s\n",
               static_cast<int>(what.size()), what.data(),
               Printable(input).c_str(), Printable(old_result, 2000).c_str(),
               Printable(new_result, 2000).c_str());
}

struct Setting {
  bool images;
  gufo::ReasoningOptions defaults;
  bool provider;
  bool require_model;
};

const std::vector<Setting>& Settings() {
  static const std::vector<Setting> settings = [] {
    gufo::ReasoningOptions thinking;
    thinking.enabled = true;
    thinking.effort = gufo::ReasoningEffort::kHigh;
    thinking.preserve_thinking = false;
    thinking.budget_tokens = 100;
    gufo::ReasoningOptions plain;
    plain.enabled = false;
    return std::vector<Setting>{
        {true, {}, false, true},
        {false, thinking, true, true},
        {true, plain, true, false},
    };
  }();
  return settings;
}

void Compare(ChatApi api, std::string_view body, const Setting& setting) {
  ++cases;
  Fixture old_fixture(setting.images, setting.defaults, setting.provider);
  Fixture new_fixture(setting.images, setting.defaults, setting.provider);
  const auto old_result =
      Run(old_fixture, false, api, body, setting.require_model);
  const auto new_result =
      Run(new_fixture, true, api, body, setting.require_model);
  if (old_result.starts_with("accepted"))
    ++accepted[static_cast<int>(api)];
  if (old_result != new_result) {
    const char* names[] = {"chat", "responses", "messages"};
    Report(names[static_cast<int>(api)], body, old_result, new_result);
  }
}

// A chat with tools and choices the fragments start from.
gufo::server::ChatRequest SeedChat(int variant) {
  gufo::server::ChatRequest chat;
  if (variant == 1) {
    gufo::tokenization::ChatTool tool;
    tool.name = "f";
    chat.tools.push_back(tool);
    chat.tool_choice = gufo::server::ChatRequest::ToolChoice::kRequired;
    chat.forced_tool_name = "f";
    chat.reasoning.enabled = true;
    chat.reasoning.effort = gufo::ReasoningEffort::kLow;
  } else if (variant == 2) {
    chat.tool_choice = gufo::server::ChatRequest::ToolChoice::kRequired;
    chat.reasoning.budget_tokens = 9;
  }
  return chat;
}

std::string Fragment(bool rust, bool anthropic, const Value& body,
                     int seed_variant, const ChatDecodeEnvironment& env) {
  auto chat = SeedChat(seed_variant);
  try {
    std::optional<HttpResponse> rejection;
    if (anthropic)
      rejection = rust ? gufo::server::ParseAnthropicToolControls(body, &chat)
                       : reference::AnthropicToolControls(body, &chat, env);
    else
      rejection = rust ? gufo::server::ParseOpenAiResponseControls(body, &chat)
                       : reference::ResponseControls(body, &chat, env);
    // A rejected C++ fragment may have filled part of the request.
    return rejection ? Describe(*rejection) : Describe(chat);
  } catch (const std::exception& exception) {
    return std::string("threw ") + exception.what();
  }
}

std::string Item(bool rust, const Value& item,
                 const ChatDecodeEnvironment& env) {
  tokenization::ChatMessage message;
  gufo::core::ImageReadBudget budget;
  std::string error;
  try {
    const bool accepted =
        rust ? gufo::server::ParseOpenAiResponseMessage(item, &message, budget,
                                                        &error)
             : reference::ResponseItem(item, &message, budget, &error, env);
    // A rejected C++ item may have filled part of the message.
    return accepted ? "accepted " + Describe(message) : "rejected " + error;
  } catch (const std::exception& exception) {
    return std::string("threw ") + exception.what();
  }
}

// The production hooks the fragment shims use.
const ChatDecodeEnvironment& RealEnvironment() {
  static const ChatDecodeEnvironment environment = [] {
    ChatDecodeEnvironment hooks;
    hooks.read_image = [](std::string_view url,
                          gufo::core::ImageReadBudget& budget) {
      return gufo::core::ReadImageUrl(url, budget);
    };
    hooks.compile_schema = [](const Value& schema) {
      (void)gufo::sampling::JsonConstraint::Compile(schema, true);
    };
    return hooks;
  }();
  return environment;
}

void CompareFragments(ChatApi api, std::string_view text) {
  Value body;
  try {
    body = gufo::json::parse(text);
  } catch (const std::exception&) {
    return;
  }
  const auto& env = RealEnvironment();
  if (api != ChatApi::kChat) {
    const bool anthropic = api == ChatApi::kMessages;
    for (int seed = 0; seed < 3; ++seed) {
      ++cases;
      const auto old_result = Fragment(false, anthropic, body, seed, env);
      const auto new_result = Fragment(true, anthropic, body, seed, env);
      if (old_result != new_result)
        Report(anthropic ? "anthropic tools" : "response controls", text,
               old_result, new_result);
    }
  }
  if (api == ChatApi::kResponses) {
    if (const auto* input = body.find("input"); input && input->is_array()) {
      for (const auto& item : input->items()) {
        ++cases;
        const auto old_result = Item(false, item, env);
        const auto new_result = Item(true, item, env);
        if (old_result != new_result)
          Report("response item", item.dump(), old_result, new_result);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Generated bodies: each dimension is a list of member spellings ("" leaves
// the member out); a body takes one spelling per dimension.

// A dimension's spellings and the indices of the ones a valid body may use.
struct Dimension {
  std::vector<std::string> spellings;
  std::vector<std::size_t> valid;
};

std::string Join(const std::vector<std::string>& members) {
  std::string out = "{";
  for (const auto& member : members) {
    if (member.empty())
      continue;
    if (out.size() > 1)
      out += ',';
    out += member;
  }
  return out + "}";
}

const Dimension kChatContents{
    {
        R"("content":null)",
        R"("content":"hello")",
        R"("content":5)",
        R"("content":[])",
        R"("content":[{"type":"text","text":"a"},{"text":"b"}])",
        R"("content":[{"type":"input_text","text":"a"}])",
        R"("content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u1"}},{"type":"text","text":"b"}])",
        R"("content":[{"type":"image_url","image_url":{"url":"u2","detail":"low"}}])",
        R"("content":[{"type":"image_url","image_url":{"url":"u3","detail":"high"}}])",
        R"("content":[{"type":"image_url","image_url":{"url":"u4","detail":"tiny"}}])",
        R"("content":[{"type":"image_url","image_url":{"url":"fail"}}])",
        R"("content":[{"type":"image_url","image_url":{"url":"oom"}}])",
        R"("content":[{"type":"image_url","image_url":"u5"}])",
        R"("content":[{"type":"text","text":1}])",
        R"("content":[{"type":"audio"}])",
        R"("content":["text"])",
    },
    {0, 1, 3, 4, 5, 6, 7, 8}};

const Dimension kRoles{
    {R"("role":"user")", R"("role":"system")", R"("role":"developer")",
     R"("role":"assistant")", R"("role":"tool")", R"("role":"bot")",
     R"("role":7)", ""},
    {0, 1, 2, 3, 4}};

const Dimension kChatExtras{
    {
        "",
        R"("name":"n","tool_call_id":"c")",
        R"("reasoning_content":"why")",
        R"("reasoning_content":null)",
        R"("reasoning_content":3)",
        R"("tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1.0,\"b\":\"x\",\"c\":[1e2,-0]}"}}])",
        R"("tool_calls":[{"function":{"name":"f","arguments":"{}"}},{"id":"c2","function":{"name":"g","arguments":"{\"n\":9007199254740993}"}}])",
        R"("tool_calls":[{"type":"custom","function":{}}])",
        R"("tool_calls":[{"function":"f"}])",
        R"("tool_calls":[{"function":{"name":"","arguments":"{}"}}])",
        R"("tool_calls":[{"function":{"name":"f","arguments":"[1]"}}])",
        R"("tool_calls":[{"function":{"name":"f","arguments":"{\"a\":"}}])",
        R"("tool_calls":[{"function":{"name":"f\u0000g","arguments":"{}"}}])",
        R"("tool_calls":{})",
        R"("tool_calls":null)",
    },
    {0, 1, 2, 3, 5, 6}};

const Dimension kChatTools{
    {
        "",
        R"("tools":null)",
        R"("tools":{})",
        R"("tools":[1])",
        R"("tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"number","minimum":1.0}}}}}])",
        R"("tools":[{"type":"function","name":"f","strict":true,"parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}])",
        R"("tools":[{"type":"function","function":{"name":"f","strict":true}},{"type":"function","function":{"name":"g","strict":null,"parametersJsonSchema":{"type":"object"}}}])",
        R"("tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"reject":1}}}])",
        R"("tools":[{"type":"function","function":{"name":"f","strict":"yes"}}])",
        R"("tools":[{"type":"function","function":{"name":"f"}},{"type":"function","function":{"name":"f"}}])",
        R"("tools":[{"type":"function","function":{"name":"bad name"}}])",
        R"("tools":[{"type":"function","function":{"name":"f","parameters":[]}}])",
        R"("tools":[{"type":"function","function":"f"}])",
        R"("tools":[{"type":"web_search"}])",
        R"("tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f"}]}])",
        R"("tools":[{"type":"function","function":{"name":"f","parameters":null,"parametersJsonSchema":{"type":"object","x":1e2}},"extra":true}])",
    },
    {0, 1, 4, 5, 6, 15}};

const Dimension kChoices{
    {
        "",
        R"("tool_choice":null)",
        R"("tool_choice":"auto")",
        R"("tool_choice":"none")",
        R"("tool_choice":"required")",
        R"("tool_choice":"bogus")",
        R"("tool_choice":{"type":"function","function":{"name":"f"}})",
        R"("tool_choice":{"type":"function","name":"g"})",
        R"("tool_choice":{"type":"function","function":{"name":"missing"}})",
        R"("tool_choice":{"type":"custom"})",
        R"("tool_choice":3)",
    },
    {0, 1, 2, 3, 4, 6}};

const Dimension kParallel{
    {"", R"("parallel_tool_calls":false)", R"("parallel_tool_calls":null)",
     R"("parallel_tool_calls":1)"},
    {0, 1}};

const Dimension kThinking{{
                              "",
                              R"("thinking":{"type":"enabled"})",
                              R"("thinking":{"type":"disabled"})",
                              R"("thinking":{"type":"adaptive"})",
                              R"("thinking":null)",
                              R"("thinking":{})",
                          },
                          {0, 1, 2}};

const Dimension kEffort{{
                            "",
                            R"("reasoning_effort":"high")",
                            R"("reasoning_effort":"none")",
                            R"("reasoning_effort":"bogus")",
                            R"("reasoning_effort":1)",
                            R"("reasoning_effort":null)",
                        },
                        {0, 1, 2, 5}};

const Dimension kKwargs{
    {
        "",
        R"("chat_template_kwargs":{"enable_thinking":false})",
        R"("chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low","preserve_thinking":true})",
        R"("chat_template_kwargs":{"thinking_mode":"chat","reasoning_effort":"max"})",
        R"("chat_template_kwargs":{"thinking_mode":"thinking"})",
        R"("chat_template_kwargs":{"thinking_mode":"maybe"})",
        R"("chat_template_kwargs":{"thinking_mode":1})",
        R"("chat_template_kwargs":{"reasoning_effort":"medium"})",
        R"("chat_template_kwargs":{"reasoning_effort":"high"})",
        R"("chat_template_kwargs":{"preserve_thinking":"yes"})",
        R"("chat_template_kwargs":{"enable_thinking":"yes"})",
        R"("chat_template_kwargs":{"add_vision_id":true})",
        R"("chat_template_kwargs":{"add_vision_id":1})",
        R"("chat_template_kwargs":[])",
    },
    {0, 1, 2, 3, 4, 7, 11}};

const Dimension kBudgets{
    {
        "",
        R"("reasoning_budget_tokens":-1)",
        R"("reasoning_budget_tokens":0)",
        R"("reasoning_budget_tokens":5.0)",
        R"("reasoning_budget_tokens":2147483647)",
        R"("reasoning_budget_tokens":2147483648)",
        R"("reasoning_budget_tokens":1.5)",
        R"("reasoning_budget_tokens":"5")",
        R"("reasoning_budget_tokens":null,"thinking_budget_tokens":7)",
        R"("thinking_budget_tokens":-2)",
    },
    {0, 1, 2, 3, 4, 8}};

const Dimension kMaxTokens{{
                               "",
                               R"("max_tokens":1)",
                               R"("max_tokens":0)",
                               R"("max_tokens":1.5)",
                               R"("max_tokens":4294967295)",
                               R"("max_tokens":4294967296)",
                               R"("max_tokens":"3")",
                               R"("max_completion_tokens":null,"max_tokens":2)",
                               R"("max_completion_tokens":1e2)",
                               R"("max_completion_tokens":-0)",
                           },
                           {0, 1, 4, 7, 8}};

const Dimension kChatMisc{
    {
        "",
        R"("stream":true,"stream_options":{"include_usage":true})",
        R"("stream":"yes")",
        R"("stream":null,"stream_options":null)",
        R"("stream_options":{"include_usage":1})",
        R"("stream_options":[])",
        R"("return_progress":true)",
        R"("return_progress":"x")",
        R"("cache_prompt":false,"cache_persist":false)",
        R"("cache_prompt":1)",
        R"("cache_persist":"no")",
        R"("n":1.0)",
        R"("n":2)",
        R"("n":"1")",
        R"("ignore_eos":true)",
        R"("modalities":["text"])",
        R"("modalities":["audio"])",
        R"("audio":{})",
        R"("response_format":{"type":"json_object"})",
        R"("response_format":{"type":"json_schema","json_schema":{"name":"n","description":"d","schema":{"type":"object"}}})",
        R"("response_format":{"type":"xml"})",
        R"("response_format":null)",
        R"("stop":"x")",
        R"("stop":["a","b"])",
        R"("stop":[1])",
        R"("stop":5)",
        R"("temperature":0.5,"top_p":0.9,"top_k":4)",
        R"("temperature":3)",
        R"("temperature":"hot")",
        R"("logprobs":true,"top_logprobs":2)",
        R"("logprobs":"x")",
        R"("top_logprobs":2)",
        R"("seed":-1)",
        R"("logit_bias":{})",
        R"("logit_bias":{"1":2})",
        R"("draft_n":4)",
        R"("mirostat":1)",
        R"("mirostat":0)",
    },
    {0, 1, 3, 6, 8, 11, 15, 18, 19, 21, 22, 23, 26, 29, 32, 33, 36}};

const Dimension kModels{
    {R"("model":"m")", R"("model":"alias")", R"("model":"x")",
     R"("model":null)", R"("model":1)", "", R"("model":"")"},
    {0, 1}};

// Bodies alternate between any spelling and valid ones only, so the grids
// reach accepted requests as well as each rejection.
bool valid_only = false;

std::string Pick(std::mt19937_64& random, const Dimension& dimension) {
  const auto& all = dimension.spellings;
  if (valid_only)
    return all.at(dimension.valid[random() % dimension.valid.size()]);
  return all[random() % all.size()];
}

// A dimension member half the time, absent otherwise.
std::string Maybe(std::mt19937_64& random, const Dimension& dimension) {
  return random() % 2 ? Pick(random, dimension) : "";
}

std::string ChatMessage(std::mt19937_64& random) {
  if (!valid_only)
    return Join({Pick(random, kRoles), Pick(random, kChatContents),
                 Pick(random, kChatExtras)});
  // Images only on user and tool turns, history only on assistant ones.
  const std::string role = Pick(random, kRoles);
  const bool images = role.find("user") != std::string::npos ||
                      role.find("tool") != std::string::npos;
  const bool assistant = role.find("assistant") != std::string::npos;
  const std::size_t content = images ? random() % 8 : random() % 5;
  const std::size_t extras = assistant ? random() % 6 : random() % 2;
  return Join({role, kChatContents.spellings[kChatContents.valid[content]],
               kChatExtras.spellings[kChatExtras.valid[extras]]});
}

std::string ChatBody(std::mt19937_64& random) {
  std::string messages = R"("messages":[)";
  const int count = static_cast<int>(random() % 3) + (valid_only ? 1 : 0);
  for (int i = 0; i < count; ++i)
    messages += (i ? "," : "") + ChatMessage(random);
  messages += "]";
  if (!valid_only && random() % 20 == 0)
    messages = R"("messages":{})";
  if (!valid_only)
    return Join({Pick(random, kModels), messages, Maybe(random, kChatTools),
                 Maybe(random, kChoices), Maybe(random, kParallel),
                 Maybe(random, kThinking), Maybe(random, kEffort),
                 Maybe(random, kKwargs), Maybe(random, kBudgets),
                 Maybe(random, kMaxTokens), Maybe(random, kChatMisc),
                 Maybe(random, kChatMisc)});
  // A choice names a declared function only with tools that declare f, and
  // one reasoning control at a time cannot disagree with another.
  const std::string tools = Maybe(random, kChatTools);
  const bool declares = tools.find("\"f\"") != std::string::npos;
  const std::string choice =
      declares ? Maybe(random, kChoices)
               : kChoices.spellings[kChoices.valid[random() % 4]];
  const Dimension* reasoning[] = {&kThinking, &kEffort, &kKwargs};
  return Join({Pick(random, kModels), messages, tools, choice,
               Maybe(random, kParallel),
               Maybe(random, *reasoning[random() % 3]), Maybe(random, kBudgets),
               Maybe(random, kMaxTokens), Maybe(random, kChatMisc)});
}

const Dimension kResponseItems{
    {
        R"({"role":"user","content":"hi"})",
        R"({"role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_image","image_url":"u1","detail":"low"},{"type":"input_text","text":"b"}]})",
        R"({"type":"message","role":"user","content":[{"type":"input_image","image_url":"fail"}]})",
        R"({"role":"user","content":[{"type":"input_image","file_id":"x","image_url":"u"}]})",
        R"({"role":"user","content":[{"type":"input_image"}]})",
        R"({"role":"user","content":[{"type":"input_file"}]})",
        R"({"role":"user","content":[{"type":"text","text":"t"}]})",
        R"({"role":"assistant","content":[{"type":"output_text","text":"answer"}]})",
        R"({"role":"assistant","content":[{"type":"input_image","image_url":"u2"}]})",
        R"({"role":"system","content":"s"})",
        R"({"role":"developer","content":[{"type":"input_text","text":"d"}]})",
        R"({"role":"tool","content":"t"})",
        R"({"role":"user","content":"x","tool_calls":[]})",
        R"({"role":"user","type":"other","content":"x"})",
        R"({"role":"assistant","content":"x","reasoning_content":"r"})",
        R"({"type":"reasoning","summary":[{"type":"summary_text","text":"think"}]})",
        R"({"type":"reasoning","summary":[],"encrypted_content":null,"content":null})",
        R"({"type":"reasoning","summary":[],"encrypted_content":"opaque"})",
        R"({"type":"reasoning","summary":[{"type":"other","text":"x"}]})",
        R"({"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1.0}"})",
        R"({"type":"function_call","call_id":"c2","name":"g","arguments":"{\"b\":[1,{\"c\":2.50}]}"})",
        R"({"type":"function_call","call_id":"","name":"f","arguments":"{}"})",
        R"({"type":"function_call","call_id":"c3","name":"f","arguments":"nope"})",
        R"({"type":"function_call","call_id":"c4","name":"f\u0000x","arguments":"{}"})",
        R"({"type":"custom_tool_call","call_id":"c5","name":"apply","input":"line\n\"q\""})",
        R"({"type":"custom_tool_call","call_id":"c6","name":"apply"})",
        R"({"type":"function_call_output","call_id":"c1","output":"done"})",
        R"({"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"l "},{"type":"input_image","image_url":"u3"}]})",
        R"({"type":"function_call_output","call_id":"c1","output":5})",
        R"({"type":"custom_tool_call_output","call_id":"c5","output":"ok"})",
        R"("text item")",
    },
    {0, 1, 6, 7, 9, 10, 14, 15, 16, 19, 20, 24, 26, 27, 29}};

const Dimension kResponseTools{
    {
        "",
        R"("tools":[{"type":"function","name":"f","parameters":{"type":"object"}}])",
        R"("tools":[{"type":"function","name":"f","strict":true}])",
        R"("tools":[{"type":"namespace","name":"ns","tools":[{"type":"function","name":"f"},{"type":"web_search"}]},{"type":"function","name":"g"}])",
        R"("tools":[{"type":"namespace","name":"","tools":[{"type":"function","name":"f"}]}])",
        R"("tools":[{"type":"namespace","name":"ns"}])",
        R"("tools":[{"type":"web_search"},{"type":"mcp"}])",
        R"("tools":[{"type":"function","name":"f"},{"type":"namespace","name":"n2","tools":[{"type":"function","name":"f"}]}])",
    },
    {0, 1, 2, 3, 4, 6}};

const Dimension kResponseReasoning{
    {
        "",
        R"("reasoning":{"effort":"high"})",
        R"("reasoning":{"effort":"none","summary":"auto"})",
        R"("reasoning":{"effort":null})",
        R"("reasoning":{"effort":5})",
        R"("reasoning":{"effort":"bogus"})",
        R"("reasoning":{"x\u0000y":1})",
        R"("reasoning":"high")",
        R"("reasoning":null)",
    },
    {0, 1, 2, 3}};

const Dimension kResponseText{
    {
        "",
        R"("text":{"format":{"type":"json_schema","name":"n","description":"d","schema":{"type":"object"}}})",
        R"("text":{"format":{"type":"text"},"verbosity":"low"})",
        R"("text":{"format":{"type":"json_schema","name":"bad name","schema":{}}})",
        R"("text":{"color":1})",
        R"("text":5)",
        R"("text":null)",
    },
    {0, 1, 2}};

const Dimension kResponseMisc{
    {
        "",
        R"("max_output_tokens":5)",
        R"("max_output_tokens":0)",
        R"("max_tokens":5)",
        R"("stream":true)",
        R"("stream":"yes")",
        R"("store":false)",
        R"("store":true)",
        R"("include":["message.output_text.logprobs"],"top_logprobs":3)",
        R"("include":"x")",
        R"("temperature":0.5)",
        R"("temperature":9)",
        R"("return_progress":true,"cache_prompt":false,"cache_persist":false)",
        R"("return_progress":1)",
        R"("reasoning_budget_tokens":3)",
        R"("parallel_tool_calls":null)",
        R"("parallel_tool_calls":"x")",
        R"("tool_choice":"required")",
        R"("tool_choice":{"type":"function","name":"f"})",
        R"("tool_choice":"none")",
        R"("previous_response_id":"r")",
        R"("stop":["x"])",
    },
    {0, 1, 4, 6, 8, 10, 12, 14, 15}};

std::string ResponsesBody(std::mt19937_64& random) {
  std::string input;
  switch (valid_only ? 0 + 3 * (random() % 2) : random() % 6) {
    case 0:
      input = R"("input":"hello")";
      break;
    case 1:
      input = R"("input":"")";
      break;
    case 2:
      input = R"("input":{})";
      break;
    default: {
      input = R"("input":[)";
      const int count = static_cast<int>(random() % 4);
      for (int i = 0; i < count; ++i)
        input += (i ? "," : "") + Pick(random, kResponseItems);
      input += "]";
    }
  }
  const Dimension instructions{
      {"", R"("instructions":"sys")", R"("instructions":1)"}, {0, 1}};
  const std::string extra = valid_only ? "" : Maybe(random, kResponseMisc);
  return Join({Maybe(random, kModels), Maybe(random, instructions), input,
               Maybe(random, kResponseTools), Maybe(random, kResponseReasoning),
               Maybe(random, kResponseText), Maybe(random, kResponseMisc),
               extra});
}

const Dimension kMessageTurns{
    {
        R"({"role":"user","content":"hi"})",
        R"({"role":"user","content":[{"type":"text","text":"a"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQID"}},{"type":"text","text":"b"}]})",
        R"({"role":"user","content":[{"type":"image","source":{"type":"url","url":"u1"}}]})",
        R"({"role":"user","content":[{"type":"image","source":{"type":"url","url":"fail"}}]})",
        R"({"role":"user","content":[{"type":"image","source":{"type":"file"}}]})",
        R"({"role":"user","content":[{"type":"image","source":{"type":"base64","data":"x"}}]})",
        R"({"role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"x"}]})",
        R"({"role":"assistant","content":[{"type":"thinking"}]})",
        R"({"role":"assistant","content":[{"type":"image","source":{"type":"url","url":"u2"}}]})",
        R"({"role":"assistant","content":[{"type":"thinking","thinking":"t"},{"type":"tool_use","id":"c1","name":"f","input":{"a":1.0,"b":[1e2]}}]})",
        R"({"role":"assistant","content":[{"type":"tool_use","id":"","name":"f","input":{}}]})",
        R"({"role":"assistant","content":[{"type":"tool_use","id":"c2","input":{}}]})",
        R"({"role":"assistant","content":[{"type":"tool_use","id":"c3","name":"f","input":[]}]})",
        R"({"role":"assistant","content":[{"type":"tool_use","id":"c4","name":"f","input":{}},{"type":"redacted"}]})",
        R"({"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"c1","content":"r"},{"type":"text","text":"after"}]})",
        R"({"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","is_error":true,"content":[{"type":"text","text":"x"},{"type":"image","source":{"type":"url","url":"u3"}}]}]})",
        R"({"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":5}]})",
        R"({"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","is_error":"x"}]})",
        R"({"role":"user","content":[{"type":"tool_result"}]})",
        R"({"role":"tool","content":[{"type":"tool_result","tool_use_id":"c1"}]})",
        R"({"role":"system","content":"s"})",
        R"({"role":"developer","content":[{"type":"input_text","text":"d"}]})",
        R"({"role":"user","content":[{"type":"output_text","text":"o"}]})",
        R"({"role":"user","content":null})",
        R"({"role":"bot","content":"x"})",
        R"({"role":"user","type":"other","content":"x"})",
        R"({"role":"user","content":"x","tool_calls":[]})",
        R"(7)",
    },
    {0, 1, 2, 6, 9, 14, 15, 20, 21, 22}};

const Dimension kMessagesMisc{
    {
        "",
        R"("system":"sys")",
        R"("system":[{"type":"text","text":"a"},{"type":"input_text","text":"b"}])",
        R"("system":[{"type":"image","source":{"type":"url","url":"u"}}])",
        R"("system":null)",
        R"("system":5)",
        R"("thinking":{"type":"enabled","budget_tokens":9})",
        R"("thinking":{"type":"adaptive","budget_tokens":9,"display":"omitted"})",
        R"("thinking":{"type":"disabled"})",
        R"("thinking":{"type":"enabled","budget_tokens":0})",
        R"("thinking":{"type":"enabled","display":"full"})",
        R"("thinking":{"type":"sometimes"})",
        R"("thinking":null)",
        R"("output_config":{"effort":"high"})",
        R"("output_config":{"effort":"minimal"})",
        R"("output_config":{"effort":null})",
        R"("output_config":{"format":1})",
        R"("output_config":7)",
        R"("tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{}}}])",
        R"("tools":[{"type":"custom","name":"f","input_schema":{"type":"object"}},{"name":"g","input_schema":{}}])",
        R"("tools":[{"name":"f"}])",
        R"("tools":[{"type":"bash_20250124","name":"bash"}])",
        R"("tools":{})",
        R"("tool_choice":{"type":"auto"})",
        R"("tool_choice":{"type":"any","disable_parallel_tool_use":true})",
        R"("tool_choice":{"type":"tool","name":"f","disable_parallel_tool_use":false})",
        R"("tool_choice":{"type":"tool","name":"missing"})",
        R"("tool_choice":{"type":"none"})",
        R"("tool_choice":{"type":"auto","disable_parallel_tool_use":"x"})",
        R"("tool_choice":"auto")",
        R"("stop_sequences":["x"])",
        R"("stop_sequences":"x")",
        R"("stream":true)",
        R"("stream":"x")",
        R"("temperature":0.5,"top_k":3)",
        R"("max_tokens":0)",
        R"("max_tokens":10)",
        R"("max_output_tokens":10)",
        R"("response_format":{})",
        R"("metadata":{})",
    },
    {0, 1, 2, 6, 7, 8, 13, 14, 15, 18, 19, 23, 24, 25, 27, 30, 32, 35, 36, 39}};

std::string MessagesBody(std::mt19937_64& random) {
  std::string messages = R"("messages":[)";
  const int count = static_cast<int>(random() % 4) + (valid_only ? 1 : 0);
  for (int i = 0; i < count; ++i)
    messages += (i ? "," : "") + Pick(random, kMessageTurns);
  messages += "]";
  // Valid bodies take fewer members, which rarely repeat one.
  const std::string extra = valid_only ? "" : Maybe(random, kMessagesMisc);
  return Join({Maybe(random, kModels), messages, Maybe(random, kMessagesMisc),
               Maybe(random, kMessagesMisc), Maybe(random, kMessagesMisc),
               extra});
}

// ---------------------------------------------------------------------------
// Mutations of a body: a value at a random path changes type, an object
// loses, renames or repeats a member, a number changes spelling, or a string
// or key gains a NUL or non-ASCII byte.

const char* const kNumbers[] = {"1.0",        "1e2", "-0",  "9007199254740993",
                                "4294967296", "-1",  "0.5", "2147483648"};
const char* const kValues[] = {"null",  "true", "false", "1",
                               "\"s\"", "[]",   "{}",    "\"fail\""};

std::size_t Count(const Value& value) {
  std::size_t count = 1;
  for (const auto& item : value.items())
    count += Count(item);
  for (const auto& [key, member] : value.members())
    count += Count(member);
  return count;
}

std::string Spice(std::mt19937_64& random, std::string text) {
  // A NUL, é, a zero-width space, a space or a framing character.
  static const char* const kInserts[] = {"", "\xC3\xA9", "\xE2\x80\x8B", " ",
                                         "<"};
  const std::size_t choice = random() % 5;
  const std::string insert =
      choice == 0 ? std::string(1, '\0') : std::string(kInserts[choice]);
  const std::size_t at = text.empty() ? 0 : random() % (text.size() + 1);
  text.insert(at, insert);
  return text;
}

Value Mutate(std::mt19937_64& random, const Value& value, std::size_t& target) {
  if (target-- == 0) {
    switch (random() % 4) {
      case 0:
        return gufo::json::parse(kValues[random() % std::size(kValues)]);
      case 1:
        if (value.is_object() && !value.empty()) {
          Value out = Value::object();
          const std::size_t pick = random() % value.size();
          // Drop, rename or repeat the member at `pick`.
          const std::size_t edit = random() % 3;
          std::size_t index = 0;
          for (const auto& [key, member] : value.members()) {
            const bool picked = index++ == pick;
            if (picked && edit == 0)
              continue;
            out.append_member(picked && edit == 1 ? Spice(random, key) : key,
                              member);
            if (picked && edit == 2)
              out.append_member(key, member);
          }
          return out;
        }
        break;
      case 2:
        if (value.is_number() || value.is_bool() || value.is_null())
          return gufo::json::parse(kNumbers[random() % std::size(kNumbers)]);
        break;
      default:
        if (value.is_string())
          return Value(Spice(random, value.str()));
        break;
    }
    return gufo::json::parse(kValues[random() % std::size(kValues)]);
  }
  if (value.is_array()) {
    Value out = Value::array();
    for (const auto& item : value.items())
      out.push_back(Mutate(random, item, target));
    return out;
  }
  if (value.is_object()) {
    Value out = Value::object();
    for (const auto& [key, member] : value.members())
      out.append_member(key, Mutate(random, member, target));
    return out;
  }
  return value;
}

std::string EditBytes(std::mt19937_64& random, std::string text) {
  static const char kBytes[] = {'"',  '{',    '}',    ',', '\\', ':',
                                '\0', '\xC3', '\xFF', '1', ']',  'n'};
  const std::size_t at = random() % (text.size() + 1);
  switch (random() % 3) {
    case 0:
      text.resize(at);
      break;
    case 1:
      if (at < text.size())
        text.erase(at, 1);
      break;
    default:
      text.insert(at, 1, kBytes[random() % sizeof(kBytes)]);
  }
  return text;
}

std::string MutateText(std::mt19937_64& random, const std::string& text) {
  Value value;
  try {
    value = gufo::json::parse(text);
  } catch (const std::exception&) {
    return EditBytes(random, text);  // A body that repeats a member.
  }
  std::size_t target = random() % Count(value);
  return Mutate(random, value, target).dump(Value::Numbers::kAsParsed);
}

void CompareAll(ChatApi api, const std::string& body, std::size_t index) {
  const auto& setting = Settings()[index % Settings().size()];
  Compare(api, body, setting);
  CompareFragments(api, body);
}

}  // namespace

int main() {
  std::mt19937_64 random(1351);
  constexpr int kBodies = 1500;
  constexpr int kMutations = 3;
  const std::pair<ChatApi, std::function<std::string(std::mt19937_64&)>>
      generators[] = {
          {ChatApi::kChat, ChatBody},
          {ChatApi::kResponses, ResponsesBody},
          {ChatApi::kMessages, MessagesBody},
      };
  std::size_t index = 0;
  for (const auto& [api, generate] : generators) {
    // Each body, its mutations and a byte edit; settings rotate.
    for (int body = 0; body < kBodies; ++body) {
      valid_only = body % 2 == 1;
      const std::string text = generate(random);
      CompareAll(api, text, index++);
      for (int mutation = 0; mutation < kMutations; ++mutation)
        CompareAll(api, MutateText(random, text), index++);
      CompareAll(api, EditBytes(random, text), index++);
    }
    // Bodies that are not objects.
    for (const char* text : {"[]", "1", "\"x\"", "null", "", "{"})
      for (std::size_t setting = 0; setting < Settings().size(); ++setting)
        CompareAll(api, text, setting);
  }
  std::printf(
      "%d cases, %d differences; accepted chat %d, responses %d, messages "
      "%d\n",
      cases, failures, accepted[0], accepted[1], accepted[2]);
  return failures == 0 ? 0 : 1;
}
