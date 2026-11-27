#include "src/cli/serve/chat_request.hpp"

#include <gufo/chat_request.h>

#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/logprobs.hpp"
#include "src/cli/serve/response_format.hpp"
#include "src/cli/serve/responses_logprobs.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/rust/callback.hpp"

namespace gufo::server {
namespace {

std::string_view Text(const std::uint8_t* data, std::size_t len) {
  return {reinterpret_cast<const char*>(data), len};
}

const std::uint8_t* Bytes(std::string_view text) {
  return reinterpret_cast<const std::uint8_t*>(text.data());
}

// Err's `const char*` message ends at the first NUL.
std::string UpToNul(const std::string& message) {
  return std::string(message.c_str());
}

// The C ABI names each value of these enums; a new value must be added there.
static_assert(static_cast<int>(tokenization::ChatRole::kSystem) ==
              GUFO_CHAT_ROLE_SYSTEM);
static_assert(static_cast<int>(tokenization::ChatRole::kDeveloper) ==
              GUFO_CHAT_ROLE_DEVELOPER);
static_assert(static_cast<int>(tokenization::ChatRole::kUser) ==
              GUFO_CHAT_ROLE_USER);
static_assert(static_cast<int>(tokenization::ChatRole::kAssistant) ==
              GUFO_CHAT_ROLE_ASSISTANT);
static_assert(static_cast<int>(tokenization::ChatRole::kTool) ==
              GUFO_CHAT_ROLE_TOOL);
static_assert(static_cast<int>(ReasoningEffort::kMinimal) ==
              GUFO_CHAT_EFFORT_MINIMAL);
static_assert(static_cast<int>(ReasoningEffort::kLow) == GUFO_CHAT_EFFORT_LOW);
static_assert(static_cast<int>(ReasoningEffort::kMedium) ==
              GUFO_CHAT_EFFORT_MEDIUM);
static_assert(static_cast<int>(ReasoningEffort::kHigh) ==
              GUFO_CHAT_EFFORT_HIGH);
static_assert(static_cast<int>(ReasoningEffort::kXHigh) ==
              GUFO_CHAT_EFFORT_XHIGH);
static_assert(static_cast<int>(ReasoningEffort::kMax) == GUFO_CHAT_EFFORT_MAX);
static_assert(static_cast<int>(ChatRequest::ToolChoice::kAuto) ==
              GUFO_CHAT_TOOL_CHOICE_AUTO);
static_assert(static_cast<int>(ChatRequest::ToolChoice::kNone) ==
              GUFO_CHAT_TOOL_CHOICE_NONE);
static_assert(static_cast<int>(ChatRequest::ToolChoice::kRequired) ==
              GUFO_CHAT_TOOL_CHOICE_REQUIRED);

// Each enum value as the C ABI names it; -Wswitch flags a value added to the
// C++ enum without one.
GufoChatEffort ToC(ReasoningEffort effort) {
  switch (effort) {
    case ReasoningEffort::kMinimal:
      return GUFO_CHAT_EFFORT_MINIMAL;
    case ReasoningEffort::kLow:
      return GUFO_CHAT_EFFORT_LOW;
    case ReasoningEffort::kMedium:
      return GUFO_CHAT_EFFORT_MEDIUM;
    case ReasoningEffort::kHigh:
      return GUFO_CHAT_EFFORT_HIGH;
    case ReasoningEffort::kXHigh:
      return GUFO_CHAT_EFFORT_XHIGH;
    case ReasoningEffort::kMax:
      return GUFO_CHAT_EFFORT_MAX;
  }
  throw std::logic_error("unknown reasoning effort");
}

GufoChatToolChoice ToC(ChatRequest::ToolChoice choice) {
  switch (choice) {
    case ChatRequest::ToolChoice::kAuto:
      return GUFO_CHAT_TOOL_CHOICE_AUTO;
    case ChatRequest::ToolChoice::kNone:
      return GUFO_CHAT_TOOL_CHOICE_NONE;
    case ChatRequest::ToolChoice::kRequired:
      return GUFO_CHAT_TOOL_CHOICE_REQUIRED;
  }
  throw std::logic_error("unknown tool choice");
}

ReasoningEffort EffortFrom(std::int64_t value) {
  switch (value) {
    case GUFO_CHAT_EFFORT_MINIMAL:
      return ReasoningEffort::kMinimal;
    case GUFO_CHAT_EFFORT_LOW:
      return ReasoningEffort::kLow;
    case GUFO_CHAT_EFFORT_MEDIUM:
      return ReasoningEffort::kMedium;
    case GUFO_CHAT_EFFORT_HIGH:
      return ReasoningEffort::kHigh;
    case GUFO_CHAT_EFFORT_XHIGH:
      return ReasoningEffort::kXHigh;
    case GUFO_CHAT_EFFORT_MAX:
      return ReasoningEffort::kMax;
    default:
      throw std::logic_error("unknown chat effort");
  }
}

ChatRequest::ToolChoice ToolChoiceFrom(std::int64_t value) {
  switch (value) {
    case GUFO_CHAT_TOOL_CHOICE_AUTO:
      return ChatRequest::ToolChoice::kAuto;
    case GUFO_CHAT_TOOL_CHOICE_NONE:
      return ChatRequest::ToolChoice::kNone;
    case GUFO_CHAT_TOOL_CHOICE_REQUIRED:
      return ChatRequest::ToolChoice::kRequired;
    default:
      throw std::logic_error("unknown chat tool choice");
  }
}

tokenization::ChatRole RoleFrom(std::int64_t value) {
  switch (value) {
    case GUFO_CHAT_ROLE_SYSTEM:
      return tokenization::ChatRole::kSystem;
    case GUFO_CHAT_ROLE_DEVELOPER:
      return tokenization::ChatRole::kDeveloper;
    case GUFO_CHAT_ROLE_USER:
      return tokenization::ChatRole::kUser;
    case GUFO_CHAT_ROLE_ASSISTANT:
      return tokenization::ChatRole::kAssistant;
    case GUFO_CHAT_ROLE_TOOL:
      return tokenization::ChatRole::kTool;
    default:
      throw std::logic_error("unknown chat role");
  }
}

std::optional<bool> Thinking(std::int32_t argument) {
  if (argument == GUFO_CHAT_UNSET)
    return std::nullopt;
  return argument != 0;
}

GufoChatReasoning ToC(const ReasoningOptions& options) {
  const auto flag = [](const std::optional<bool>& value) -> std::int32_t {
    return value ? static_cast<std::int32_t>(*value) : GUFO_CHAT_UNSET;
  };
  return {
      .enabled = flag(options.enabled),
      .effort = options.effort ? static_cast<std::int32_t>(ToC(*options.effort))
                               : GUFO_CHAT_UNSET,
      .preserve_thinking = flag(options.preserve_thinking),
      .has_budget = static_cast<std::uint8_t>(options.budget_tokens ? 1 : 0),
      .budget_tokens = options.budget_tokens.value_or(0),
  };
}

// One decode: the hooks, the request the helpers and the emitted steps
// build, and why it stopped.
struct Decoder {
  Decoder(const ChatDecodeEnvironment& environment_in,
          core::ImageReadBudget& budget_in)
      : environment(environment_in), budget(budget_in) {}

  const ChatDecodeEnvironment& environment;
  core::ImageReadBudget& budget;
  DecodedChatRequest pending;
  std::vector<std::shared_ptr<const std::vector<std::uint8_t>>> images;
  std::optional<HttpResponse> rejection;
  std::string rejection_message;
  std::string namespace_key;
  core::ffi::CallbackFailure failure;

  int ReadImage(std::string_view url, std::uint32_t* id,
                const GufoChatMessage* error) {
    try {
      auto bytes = std::make_shared<const std::vector<std::uint8_t>>(
          environment.read_image(url, budget));
      images.push_back(std::move(bytes));
    } catch (const std::exception& exception) {
      // std::bad_alloc too: an image read that runs out of memory answers
      // 400 with its message, as the C++ decoders did and as a decode that
      // runs out of memory does (see Run).
      const std::string_view message = exception.what();
      error->set(error->state, Bytes(message), message.size());
      return 1;
    }
    *id = static_cast<std::uint32_t>(images.size() - 1);
    return 0;
  }

  int CompileSchema(std::string_view text, const GufoChatMessage* error) {
    const json::Value schema = json::parse(text);
    try {
      environment.compile_schema(schema);
    } catch (const std::exception& exception) {
      const std::string_view message = exception.what();
      error->set(error->state, Bytes(message), message.size());
      return 1;
    }
    return 0;
  }

  int Reject(HttpResponse response) {
    rejection = std::move(response);
    return GUFO_CHAT_HOST_REJECTED;
  }

  // The last value a helper got; Chat's sampling and logprobs helpers read
  // the same body view.
  std::string parsed_text;
  json::Value parsed_value;

  const json::Value* Parse(std::string_view text) {
    if (text.empty())
      return nullptr;
    if (text != parsed_text) {
      parsed_value = json::parse(text);
      parsed_text = text;
    }
    return &parsed_value;
  }

  int Field(GufoChatHostField field, std::int32_t argument,
            std::string_view text) {
    const json::Value* found = Parse(text);
    auto& chat = pending.chat;
    switch (field) {
      case GUFO_CHAT_FIELD_RESPONSE_FORMAT:
        try {
          chat.response_format = ParseResponseFormat(found);
          if (chat.response_format) {
            if (const auto* specification = found->find("json_schema"))
              chat.response_format_description =
                  specification->member_str("description");
          }
        } catch (const std::exception& error) {
          return Reject(
              ChatRejection(400, error.what(), "invalid_response_format"));
        }
        return GUFO_CHAT_HOST_OK;
      case GUFO_CHAT_FIELD_STOP:
        if (const auto error = ParseStopSequences(
                found, StopSequenceFormat::kOpenAi, &chat.stop_sequences))
          return Reject(ChatRejection(400, *error, "invalid_stop"));
        return GUFO_CHAT_HOST_OK;
      case GUFO_CHAT_FIELD_SAMPLING: {
        sampling::SamplingConfig parsed;
        if (const auto error = ParseSamplingConfig(
                *found,
                environment.sampling_defaults.Resolve(Thinking(argument)),
                &parsed))
          return Reject(
              ChatRejection(400, error->message, error->code.c_str()));
        pending.sampling = parsed;
        return GUFO_CHAT_HOST_OK;
      }
      case GUFO_CHAT_FIELD_LOGPROBS:
        if (const auto error = ParseLogprobs(*found, true, &pending.sampling))
          return Reject(
              ChatRejection(400, error->message, error->code.c_str()));
        return GUFO_CHAT_HOST_OK;
      case GUFO_CHAT_FIELD_RESPONSES_OPTIONS:
      case GUFO_CHAT_FIELD_MESSAGES_OPTIONS: {
        if (!environment.compatibility)
          throw std::logic_error("chat decode needs a compatibility check");
        std::size_t max_tokens = 0;
        const auto api = field == GUFO_CHAT_FIELD_RESPONSES_OPTIONS
                             ? ChatApi::kResponses
                             : ChatApi::kMessages;
        if (auto error =
                environment.compatibility(api, *found, Thinking(argument),
                                          &max_tokens, &pending.sampling))
          return Reject(std::move(*error));
        pending.max_tokens = max_tokens;
        if (api == ChatApi::kResponses) {
          if (const auto error =
                  ParseResponsesLogprobs(*found, &pending.sampling))
            return Reject(ChatRejection(400, UpToNul(error->message),
                                        error->code.c_str()));
        }
        return GUFO_CHAT_HOST_OK;
      }
      case GUFO_CHAT_FIELD_TEXT_FORMAT:
        try {
          chat.response_format = ParseResponseFormat(found, true);
          if (chat.response_format)
            chat.response_format_description = found->member_str("description");
        } catch (const std::exception& error) {
          return Reject(
              ChatRejection(400, error.what(), "invalid_response_format"));
        }
        return GUFO_CHAT_HOST_OK;
      case GUFO_CHAT_FIELD_STOP_SEQUENCES:
        if (const auto error = ParseStopSequences(
                found, StopSequenceFormat::kAnthropic, &chat.stop_sequences))
          return Reject(ChatRejection(400, UpToNul(*error), "invalid_request"));
        return GUFO_CHAT_HOST_OK;
    }
    throw std::logic_error("unknown chat host field");
  }

  tokenization::ChatMessage& Message() {
    if (pending.chat.messages.empty())
      throw std::logic_error("chat step outside a message");
    return pending.chat.messages.back();
  }

  tokenization::ChatMessage::ToolCall& Call() {
    auto& calls = Message().tool_calls;
    if (calls.empty())
      throw std::logic_error("chat step outside a tool call");
    return calls.back();
  }

  tokenization::ChatTool& Tool() {
    if (pending.chat.tools.empty())
      throw std::logic_error("chat step outside a tool");
    return pending.chat.tools.back();
  }

  void Emit(const GufoChatEvent& event) {
    auto& chat = pending.chat;
    std::string text(Text(event.text, event.len));
    const bool on = event.integer != 0;
    const auto optional_flag = [&]() -> std::optional<bool> {
      return event.integer < 0 ? std::nullopt : std::optional(on);
    };
    switch (event.op) {
      case GUFO_CHAT_OP_MODEL:
        pending.model = std::move(text);
        return;
      case GUFO_CHAT_OP_MAX_TOKENS:
        pending.max_tokens = static_cast<std::size_t>(event.integer);
        return;
      case GUFO_CHAT_OP_STREAM:
        pending.stream = on;
        return;
      case GUFO_CHAT_OP_INCLUDE_USAGE:
        pending.include_usage = on;
        return;
      case GUFO_CHAT_OP_CACHE_PROMPT:
        chat.cache_prompt = on;
        return;
      case GUFO_CHAT_OP_CACHE_PERSIST:
        chat.cache_persist = on;
        return;
      case GUFO_CHAT_OP_RETURN_PROGRESS:
        chat.return_progress = on;
        return;
      case GUFO_CHAT_OP_ADD_VISION_ID:
        chat.add_vision_id = on;
        return;
      case GUFO_CHAT_OP_REASONING_ENABLED:
        chat.reasoning.enabled = optional_flag();
        return;
      case GUFO_CHAT_OP_REASONING_EFFORT:
        chat.reasoning.effort = event.integer < 0
                                    ? std::nullopt
                                    : std::optional(EffortFrom(event.integer));
        return;
      case GUFO_CHAT_OP_PRESERVE_THINKING:
        chat.reasoning.preserve_thinking = optional_flag();
        return;
      case GUFO_CHAT_OP_BUDGET_TOKENS:
        chat.reasoning.budget_tokens =
            event.integer < 0
                ? std::nullopt
                : std::optional(static_cast<std::size_t>(event.integer));
        return;
      case GUFO_CHAT_OP_TOOL_CHOICE:
        chat.tool_choice = ToolChoiceFrom(event.integer);
        return;
      case GUFO_CHAT_OP_FORCED_TOOL_NAME:
        chat.forced_tool_name = std::move(text);
        return;
      case GUFO_CHAT_OP_PARALLEL_TOOL_CALLS:
        chat.parallel_tool_calls = on;
        return;
      case GUFO_CHAT_OP_CONSTRAINED_TOOLS:
        chat.constrained_tools = on;
        return;
      case GUFO_CHAT_OP_NAMESPACE_KEY:
        namespace_key = std::move(text);
        return;
      case GUFO_CHAT_OP_NAMESPACE_VALUE:
        chat.tool_namespaces[namespace_key] = std::move(text);
        return;
      case GUFO_CHAT_OP_TOOL: {
        tokenization::ChatTool tool;
        tool.name = std::move(text);
        chat.tools.push_back(std::move(tool));
        return;
      }
      case GUFO_CHAT_OP_TOOL_DESCRIPTION:
        Tool().description = std::move(text);
        return;
      case GUFO_CHAT_OP_TOOL_PARAMETERS:
        Tool().parameters_json = std::move(text);
        return;
      case GUFO_CHAT_OP_TOOL_DEFINITION:
        Tool().definition_json = std::move(text);
        return;
      case GUFO_CHAT_OP_SELECT_TOOL: {
        auto tool = chat.tools.at(event.index);
        chat.tools = {std::move(tool)};
        return;
      }
      case GUFO_CHAT_OP_MESSAGE:
        chat.messages.emplace_back().role = RoleFrom(event.integer);
        return;
      case GUFO_CHAT_OP_CONTENT:
        Message().content = std::move(text);
        return;
      case GUFO_CHAT_OP_NAME:
        Message().name = std::move(text);
        return;
      case GUFO_CHAT_OP_THOUGHT:
        Message().thought = std::move(text);
        return;
      case GUFO_CHAT_OP_TOOL_CALL_ID:
        Message().tool_call_id = std::move(text);
        return;
      case GUFO_CHAT_OP_IMAGE:
      case GUFO_CHAT_OP_IMAGE_LOW:
        Message().images.push_back(
            {static_cast<std::size_t>(event.integer), images.at(event.index),
             event.op == GUFO_CHAT_OP_IMAGE_LOW
                 ? tokenization::ChatMessage::ImageDetail::kLow
                 : tokenization::ChatMessage::ImageDetail::kAuto});
        return;
      case GUFO_CHAT_OP_CALL: {
        tokenization::ChatMessage::ToolCall call;
        call.id = std::move(text);
        Message().tool_calls.push_back(std::move(call));
        return;
      }
      case GUFO_CHAT_OP_CALL_NAME:
        Call().name = std::move(text);
        return;
      case GUFO_CHAT_OP_ARGUMENT: {
        tokenization::ChatMessage::ToolArgument argument;
        argument.name = std::move(text);
        Call().arguments.push_back(std::move(argument));
        return;
      }
      case GUFO_CHAT_OP_ARGUMENT_VALUE: {
        auto& arguments = Call().arguments;
        if (arguments.empty())
          throw std::logic_error("chat argument value without a name");
        arguments.back().value = std::move(text);
        arguments.back().is_string = on;
        return;
      }
      case GUFO_CHAT_OP_PART:
        Message().parts.push_back(
            {.thought = event.index != 0,
             .end = static_cast<std::size_t>(event.integer)});
        return;
    }
    throw std::logic_error("unknown chat builder step");
  }
};

Decoder& Of(void* context) {
  return *static_cast<Decoder*>(context);
}

constexpr GufoChatHost kHost{
    .serves_model = [](void* context, const std::uint8_t* name,
                       std::size_t len) noexcept -> int {
      auto& decoder = Of(context);
      return decoder.failure.Guard(-1, [&] {
        return decoder.environment.serves_model(Text(name, len)) ? 1 : 0;
      });
    },
    .read_image = [](void* context, const std::uint8_t* url, std::size_t len,
                     std::uint32_t* image_id,
                     const GufoChatMessage* error) noexcept -> int {
      auto& decoder = Of(context);
      return decoder.failure.Guard(-1, [&] {
        return decoder.ReadImage(Text(url, len), image_id, error);
      });
    },
    .compile_schema = [](void* context, const std::uint8_t* text,
                         std::size_t len,
                         const GufoChatMessage* error) noexcept -> int {
      auto& decoder = Of(context);
      return decoder.failure.Guard(
          -1, [&] { return decoder.CompileSchema(Text(text, len), error); });
    },
    .host_field = [](void* context, GufoChatHostField field,
                     std::int32_t argument, const std::uint8_t* text,
                     std::size_t len) noexcept -> int {
      auto& decoder = Of(context);
      return decoder.failure.Guard(
          -1, [&] { return decoder.Field(field, argument, Text(text, len)); });
    },
    .emit = [](void* context, const GufoChatEvent* event) noexcept -> int {
      auto& decoder = Of(context);
      return decoder.failure.Guard(1, [&] {
        decoder.Emit(*event);
        return 0;
      });
    },
    .reject =
        [](void* context, const GufoChatRejection* rejection) noexcept {
          auto& decoder = Of(context);
          decoder.failure.Guard([&] {
            decoder.rejection_message =
                Text(rejection->message, rejection->message_len);
            decoder.rejection =
                ChatRejection(rejection->status, decoder.rejection_message,
                              Text(rejection->code, rejection->code_len));
          });
        },
};

// Runs the Rust decoder; returns whether it accepted the body.
bool Run(GufoChatApi api, std::string_view body,
         const GufoChatDefaults& defaults, Decoder& decoder) {
  const auto status = gufo_chat_decode(api, Bytes(body), body.size(), &defaults,
                                       &kHost, &decoder);
  decoder.failure.Rethrow();
  switch (status) {
    case GUFO_CHAT_ACCEPTED:
      return true;
    case GUFO_CHAT_REJECTED:
      if (!decoder.rejection)
        throw std::logic_error("chat decoder rejected without a response");
      return false;
    case GUFO_CHAT_NO_MEMORY: {
      // The C++ decoders caught std::bad_alloc with the other exceptions of
      // json::parse and answered 400 with its message; keep that answer.
      decoder.rejection_message = std::bad_alloc().what();
      decoder.rejection =
          ChatRejection(400, decoder.rejection_message, "parse_error");
      return false;
    }
    case GUFO_CHAT_STOPPED:
    case GUFO_CHAT_INTERNAL:
      break;
  }
  throw std::logic_error("chat decoder failed");
}

// The hooks the fragments use: the image reader and schema compiler.
const ChatDecodeEnvironment& FragmentEnvironment() {
  static const ChatDecodeEnvironment environment = [] {
    ChatDecodeEnvironment hooks;
    hooks.read_image = [](std::string_view url, core::ImageReadBudget& budget) {
      return core::ReadImageUrl(url, budget);
    };
    hooks.compile_schema = [](const json::Value& schema) {
      (void)sampling::JsonConstraint::Compile(schema, true);
    };
    return hooks;
  }();
  return environment;
}

// Decodes a fragment of the request `chat` holds.
std::optional<HttpResponse> DecodeFragment(GufoChatApi api,
                                           const json::Value& body,
                                           ChatRequest* chat) {
  core::ImageReadBudget budget;
  Decoder decoder(FragmentEnvironment(), budget);
  decoder.pending.chat = *chat;
  std::vector<GufoChatBytes> names;
  names.reserve(chat->tools.size());
  for (const auto& tool : chat->tools)
    names.push_back({Bytes(tool.name), tool.name.size()});
  const GufoChatDefaults defaults{
      .model_id = {},
      .require_model = 1,
      .reasoning = ToC(chat->reasoning),
      .tool_names = names.data(),
      .tool_name_count = names.size(),
      .tool_choice = ToC(chat->tool_choice),
  };
  if (!Run(api, body.dump(json::Value::Numbers::kAsParsed), defaults, decoder))
    return std::move(decoder.rejection);
  *chat = std::move(decoder.pending.chat);
  return std::nullopt;
}

}  // namespace

HttpResponse ChatRejection(int status, std::string message,
                           std::string_view code) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::string(code);
  response["error"] = std::move(error);
  HttpResponse rejection;
  rejection.status = status;
  rejection.reason = status == 404 ? "Not Found" : "Bad Request";
  rejection.body = response.dump();
  return rejection;
}

ChatDecodeEnvironment ChatEnvironment(const TextGenerationBackend& backend) {
  return {
      .model_id = backend.model_id(),
      .accepts_images = backend.supports_images(),
      .reasoning_defaults = backend.reasoning_defaults(),
      .sampling_defaults = backend.sampling_defaults(),
      .serves_model =
          [&backend](std::string_view name) {
            return backend.serves_model(name);
          },
      .read_image =
          [](std::string_view url, core::ImageReadBudget& budget) {
            return core::ReadImageUrl(url, budget);
          },
      .compile_schema =
          [](const json::Value& schema) {
            (void)sampling::JsonConstraint::Compile(schema, true);
          },
      .compatibility = {},
  };
}

std::optional<HttpResponse> DecodeChatRequest(
    ChatApi api, std::string_view body,
    const ChatDecodeEnvironment& environment, bool require_model,
    DecodedChatRequest* output) {
  core::ImageReadBudget budget;
  budget.accepts_images = environment.accepts_images;
  Decoder decoder(environment, budget);
  const GufoChatDefaults defaults{
      .model_id = {Bytes(environment.model_id), environment.model_id.size()},
      .require_model = static_cast<std::uint8_t>(require_model ? 1 : 0),
      .reasoning = ToC(environment.reasoning_defaults),
      .tool_names = nullptr,
      .tool_name_count = 0,
      .tool_choice = GUFO_CHAT_TOOL_CHOICE_AUTO,
  };
  const GufoChatApi raw = api == ChatApi::kChat        ? GUFO_CHAT_API_CHAT
                          : api == ChatApi::kResponses ? GUFO_CHAT_API_RESPONSES
                                                       : GUFO_CHAT_API_MESSAGES;
  if (!Run(raw, body, defaults, decoder))
    return std::move(decoder.rejection);
  *output = std::move(decoder.pending);
  return std::nullopt;
}

std::optional<HttpResponse> DecodeResponseControls(const json::Value& body,
                                                   ChatRequest* chat) {
  return DecodeFragment(GUFO_CHAT_FRAGMENT_RESPONSE_CONTROLS, body, chat);
}

std::optional<HttpResponse> DecodeAnthropicToolControls(const json::Value& body,
                                                        ChatRequest* chat) {
  return DecodeFragment(GUFO_CHAT_FRAGMENT_ANTHROPIC_TOOLS, body, chat);
}

bool DecodeResponseItem(const json::Value& item,
                        tokenization::ChatMessage* message,
                        core::ImageReadBudget& budget, std::string* error) {
  Decoder decoder(FragmentEnvironment(), budget);
  const GufoChatDefaults defaults{
      .model_id = {},
      .require_model = 1,
      .reasoning = ToC(ReasoningOptions{}),
      .tool_names = nullptr,
      .tool_name_count = 0,
      .tool_choice = GUFO_CHAT_TOOL_CHOICE_AUTO,
  };
  if (!Run(GUFO_CHAT_FRAGMENT_RESPONSE_ITEM,
           item.dump(json::Value::Numbers::kAsParsed), defaults, decoder)) {
    *error = std::move(decoder.rejection_message);
    return false;
  }
  if (decoder.pending.chat.messages.size() != 1)
    throw std::logic_error("chat decoder gave no Responses message");
  *message = std::move(decoder.pending.chat.messages.front());
  return true;
}

}  // namespace gufo::server
