// The model's own spelling of tool calls and reasoning, for Gufo's grammars.
//
// A Loom package's chat program reads the model's native output (its tool-call
// syntax, its reasoning markers) and the runner publishes it as the canonical
// markup Gufo parses. Grammars constrain the model's tokens, so on the
// host-sampling path they must be written in the native spelling. The package
// names it in its llm_config.json (chat.call_format, chat.think_open /
// think_end, ...), which model.json lists so loom-fetch delivers it. This file
// maps those to Gufo's JsonConstraint::ToolSyntax and to the reasoning markers
// a grammar uses; a package without tool calls has no syntax, and its tool
// requests stay unconstrained.
#ifndef GUFO_LOOM_DIALECT_HPP_
#define GUFO_LOOM_DIALECT_HPP_

#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>
#include <optional>
#include <string>

namespace gufo::loom {

struct NativeDialect {
  // The package declared its dialect.
  bool known{false};
  std::string call_format;
  // The native call syntax; nullopt when Gufo's grammars cannot spell it.
  std::optional<sampling::JsonConstraint::ToolSyntax> tools;
  // The raw markers around reasoning; an empty open means the package does
  // not reason (no reasoning phase in grammars).
  std::string reasoning_open;
  std::string reasoning_close;
  // Framing the model writes between the end of reasoning and the next
  // channel: OpenAI harmony starts a new assistant message, and a JSON answer
  // lives in its `final` channel.
  std::string after_reasoning;
  std::string answer_channel;
  // Glimmer's channel format: the reasoning markers above are the ones the
  // output carries after the generation prompt.
  bool atem{false};
  // A lazy grammar cannot tell this format's calls from its other output
  // where it starts, so automatic tool use stays on the device (the package
  // parses the calls); required and named calls are still constrained.
  bool auto_on_device{false};
  // The model reasons whatever the request says (chat.thinking "always"), so
  // its output always starts in reasoning, prompt-opened or not.
  bool always_reasons{false};
};

// `chat` is the "chat" object of the package's llm_config.json.
inline NativeDialect NativeDialectOf(const json::Value& chat) {
  using Syntax = sampling::JsonConstraint::ToolSyntax;
  NativeDialect dialect;
  dialect.known = true;
  dialect.call_format = chat.member_str("call_format");
  const auto& format = dialect.call_format;
  if (format == "qwen_xml") {
    // <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
    dialect.tools = sampling::JsonConstraint::Syntax(
        sampling::JsonConstraint::ToolFormat::kQwen);
  } else if (format == "hermes") {
    // <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>
    dialect.tools = Syntax{
        .open = "<tool_call>",
        .name_prefix = "\n{\"name\": \"",
        .name_suffix = "\", \"arguments\": ",
        .close = "}\n</tool_call>",
        .name_delimiters = "\"",
    };
  } else if (format == "glm") {
    // <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
    // as the package's template (and so its history) spells it: no line breaks.
    dialect.tools = Syntax{
        .open = "<tool_call>",
        .close = "</tool_call>",
        .name_delimiters = "<",
        .tagged =
            sampling::JsonConstraint::TaggedArguments{
                .parameter_open = "<arg_key>",
                .raw_value_open = "</arg_key><arg_value>",
                .typed_value_open = "</arg_key><arg_value>",
                .value_close = "</arg_value>",
            },
    };
  } else if (format == "gemma4") {
    // <|tool_call>call:NAME{key:<|"|>v<|"|>,n:3}<tool_call|> (keys sorted,
    // bare)
    dialect.tools = Syntax{
        .open = "<|tool_call>",
        .name_prefix = "call:",
        .close = "<tool_call|>",
        .name_delimiters = "{",
        .values = {.string_open = "<|\"|>",
                   .string_close = "<|\"|>",
                   .bare_keys = true,
                   .whitespace = false,
                   .sorted_keys = true},
    };
  } else if (format == "harmony") {
    // After the analysis channel (and its <|end|>), a new assistant message:
    //   <|start|>assistant<|channel|>commentary to=functions.NAME
    //   <|constrain|>json<|message|>{...}<|call|>
    // A model that skips analysis starts at <|channel|>, as the prompt ends in
    // <|start|>assistant, so <|start|>assistant belongs to the reasoning close.
    dialect.tools = Syntax{
        .open = "<|channel|>commentary",
        .name_prefix = " to=functions.",
        .name_suffix = " <|constrain|>json<|message|>",
        .close = "<|call|>",
        .name_delimiters = " ",
    };
    dialect.after_reasoning = "<|start|>assistant";
    // gpt-oss may also write the recipient before the channel (the order its
    // template renders history in); the package's parser reads either.
    dialect.auto_on_device = true;
    dialect.answer_channel = "<|channel|>final<|message|>";
  } else if (format == "json") {
    // Llama 3.x: {"name": "NAME", "parameters": {...}}
    // (chat.call_arguments_key)
    dialect.tools = Syntax{
        .open = "{\"name\": \"",
        .name_suffix = "\", \"" +
                       chat.member_str("call_arguments_key", "arguments") +
                       "\": ",
        .close = "}",
        .name_delimiters = "\"",
    };
  } else if (format == "pythonic") {
    // LFM2: <|tool_call_start|>[NAME(k='v', n=3), OTHER()]<|tool_call_end|>
    // One list holds every call (chat.calls_open ... chat.calls_close); the
    // arguments are Python keyword arguments (see ValueSyntax::keyword_root).
    dialect.tools = Syntax{
        .open = chat.member_str("calls_open", "<|tool_call_start|>["),
        .name_suffix = "(",
        .close = ")",
        .call_separator = chat.member_str("call_separator", ", "),
        .block_close = chat.member_str("calls_close", "]<|tool_call_end|>"),
        .name_delimiters = "( ",
        .values = {.string_open = "'",
                   .string_close = "'",
                   .keyword_root = true},
    };
  } else if (format == "inkling2") {
    // Inkling 2: each call is its own model message that names the function,
    // then names it again in compact JSON with its arguments under "args":
    //   <|message_model|>NAME<|content_invoke_tool_json|>
    //   {"name":"NAME","args":{...}}<|end_message|>
    dialect.tools = Syntax{
        .open = "<|message_model|>",
        .name_suffix =
            chat.member_str("call_marker", "<|content_invoke_tool_json|>"),
        .repeat_prefix = "{\"name\":\"",
        .repeat_suffix = "\",\"args\":",
        .close = "}<|end_message|>",
        .name_delimiters = "<\"",
        .values = {.whitespace = false},
    };
    // Every Inkling message, text and thinking included, starts with
    // <|message_model|>.
    dialect.auto_on_device = true;
  } else if (format == "atem") {
    // Glimmer's channel format. After the generation prompt
    // (<|start|>assistant) a call is addressed to its function and written as
    // an XML block that names the function again; the next call opens a new
    // message:
    //   to=NAME<|message|><atem:function_calls>\n<atem:invoke name="NAME">\n
    //   <atem:parameter
    //   name="K">V</atem:parameter>\n</atem:invoke>\n</atem:function_calls>
    //   <|eom|><|start|>assistant to=NEXT<|message|>...
    // Reasoning is `to=self<|message|>...` and ends where the next message
    // header starts, so the grammar's reasoning close is the header's start.
    dialect.tools = Syntax{
        .open = " to=",
        .name_suffix = "<|message|>",
        .repeat_prefix = "<atem:function_calls>\n<atem:invoke name=\"",
        .repeat_suffix = "\">\n",
        .close = "</atem:invoke>\n</atem:function_calls>",
        .call_separator = "<|eom|><|start|>assistant",
        .name_delimiters = "<\" ",
        .tagged =
            sampling::JsonConstraint::TaggedArguments{
                .parameter_open = "<atem:parameter name=\"",
                .raw_value_open = "\">",
                .typed_value_open = "\">",
                .value_close = "</atem:parameter>\n",
            },
    };
    dialect.atem = true;
    // The call header (" to=NAME") starts like the answer's (" to=user").
    dialect.auto_on_device = true;
  }
  // none (no tool calls): no descriptor.
  dialect.always_reasons = chat.member_str("thinking") == "always";
  const std::string open = chat.member_str("think_open");
  if (chat.member_str("thinking", "never") != "never" && !open.empty()) {
    dialect.reasoning_open = open;
    dialect.reasoning_close = chat.member_str("think_end", "</think>");
    if (dialect.atem) {
      // The prompt ends in "<|start|>assistant": the model's output continues
      // with the recipient. Reasoning is addressed to self and ends at the next
      // message header, which a tool call or the answer continues.
      dialect.reasoning_open = " to=self<|message|>";
      dialect.reasoning_close = "<|start|>assistant";
      dialect.answer_channel = " to=user<|message|>";
    }
  }
  return dialect;
}

}  // namespace gufo::loom

#endif  // GUFO_LOOM_DIALECT_HPP_
