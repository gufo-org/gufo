#include "src/models/nemotron35_bolt/chat_template.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

#include "src/core/json.hpp"
#include "src/models/nemotron35_bolt/tokenizer.hpp"

namespace gufo::models::nemotron35_bolt {
namespace {

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>\n";
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kEmptyThink = "<think></think>";

constexpr std::string_view kToolInstructions =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
    "following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the "
    "value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- "
    "Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call "
    "in natural language BEFORE the function call, but NOT after\n- If there "
    "is no function call available, answer the question like normal with "
    "your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

/// Python's repr of a float: shortest round-trip digits, positional between
/// 1e-4 and 1e16, otherwise exponent form with a signed two-digit exponent.
std::string PythonFloat(double value) {
  if (std::isnan(value))
    return "nan";
  if (std::isinf(value))
    return value < 0 ? "-inf" : "inf";
  char buffer[64];
  const auto r = std::to_chars(buffer, buffer + sizeof(buffer), value,
                               std::chars_format::scientific);
  std::string sci(buffer, r.ptr);
  const auto e_pos = sci.find('e');
  const int exponent = std::stoi(sci.substr(e_pos + 1));
  std::string mantissa = sci.substr(0, e_pos);
  const bool negative = mantissa.front() == '-';
  if (negative)
    mantissa.erase(0, 1);
  std::string digits;
  for (const char c : mantissa)
    if (c != '.')
      digits.push_back(c);
  std::string out = negative ? "-" : "";
  if (exponent < -4 || exponent >= 16) {
    out += digits.substr(0, 1);
    if (digits.size() > 1)
      out += "." + digits.substr(1);
    char exp[16];
    std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+',
                  std::abs(exponent));
    return out + exp;
  }
  const int point = exponent + 1;  // Digits before the decimal point.
  if (point <= 0) {
    out += "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
  } else if (static_cast<std::size_t>(point) >= digits.size()) {
    out += digits + std::string(point - digits.size(), '0') + ".0";
  } else {
    out += digits.substr(0, point) + "." + digits.substr(point);
  }
  return out;
}

std::string PythonNumber(double value) {
  constexpr double kExactIntegers = 9007199254740992.0;  // 2^53
  if (std::trunc(value) == value && std::fabs(value) <= kExactIntegers &&
      !(value == 0.0 && std::signbit(value))) {
    return std::to_string(static_cast<long long>(value));
  }
  return PythonFloat(value);
}

std::string PythonQuote(const std::string& s) {
  const bool has_single = s.find('\'') != std::string::npos;
  const bool has_double = s.find('"') != std::string::npos;
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (const char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (ch == '\\') {
      out += "\\\\";
    } else if (ch == quote) {
      out += '\\';
      out += ch;
    } else if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (ch == '\t') {
      out += "\\t";
    } else if (c < 0x20 || c == 0x7F) {
      char hex[8];
      std::snprintf(hex, sizeof(hex), "\\x%02x", c);
      out += hex;
    } else {
      out += ch;
    }
  }
  out += quote;
  return out;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::string PythonRepr(const json::Value& v) {
  switch (v.type()) {
    case json::Value::Type::kNull:
      return "None";
    case json::Value::Type::kBool:
      return v.as_bool() ? "True" : "False";
    case json::Value::Type::kNumber:
      return PythonNumber(v.as_double());
    case json::Value::Type::kString:
      return PythonQuote(v.str());
    case json::Value::Type::kArray: {
      std::string out = "[";
      bool first = true;
      for (const auto& item : v.items()) {
        if (!first)
          out += ", ";
        first = false;
        out += PythonRepr(item);
      }
      return out + "]";
    }
    case json::Value::Type::kObject: {
      std::string out = "{";
      bool first = true;
      for (const auto& [key, item] : v.members()) {
        if (!first)
          out += ", ";
        first = false;
        out += PythonQuote(key) + ": " + PythonRepr(item);
      }
      return out + "}";
    }
  }
  return {};
}

/// Jinja's `| string` filter: Python str().
std::string PythonStr(const json::Value& v) {
  return v.is_string() ? v.str() : PythonRepr(v);
}

bool IsContainer(const json::Value& v) {
  return v.is_object() || v.is_array();
}

void RenderExtraKeys(std::string& out, const json::Value& dict,
                     std::initializer_list<std::string_view> handled) {
  if (!dict.is_object())
    return;
  for (const auto& [key, value] : dict.members()) {
    bool skip = false;
    for (const auto h : handled)
      skip = skip || key == h;
    if (skip)
      continue;
    out += "\n<" + key + ">";
    out += IsContainer(value) ? value.tojson() : PythonStr(value);
    out += "</" + key + ">";
  }
}

/// The function object the template iterates: `tool.function` when present.
json::Value ToolFunction(const ChatTool& tool) {
  if (!tool.definition_json.empty()) {
    json::Value definition = json::parse(tool.definition_json);
    if (const auto* function = definition.find("function"))
      return *function;
    return definition;
  }
  json::Value function = json::Value::object();
  function["name"] = tool.name;
  if (!tool.description.empty())
    function["description"] = tool.description;
  function["parameters"] = json::parse(
      tool.parameters_json.empty() ? std::string_view("{}")
                                   : std::string_view(tool.parameters_json));
  return function;
}

void RenderTools(std::string& out, std::span<const ChatTool> tools) {
  out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
  for (const ChatTool& t : tools) {
    const json::Value tool = ToolFunction(t);
    const json::Value* name = tool.find("name");
    out += "\n<function>\n<name>";
    out += name != nullptr ? PythonStr(*name) : "";
    out += "</name>";
    if (const auto* description = tool.find("description")) {
      out += "\n<description>";
      out += PythonStrip(PythonStr(*description));
      out += "</description>";
    }
    out += "\n<parameters>";
    const json::Value* parameters = tool.find("parameters");
    if (parameters != nullptr && parameters->is_object()) {
      const json::Value* properties = parameters->find("properties");
      if (properties != nullptr && properties->is_object()) {
        for (const auto& [param, fields] : properties->members()) {
          out += "\n<parameter>\n<name>" + param + "</name>";
          if (const auto* type = fields.find("type"))
            out += "\n<type>" + PythonStr(*type) + "</type>";
          if (const auto* description = fields.find("description")) {
            out += "\n<description>";
            out += PythonStrip(PythonStr(*description));
            out += "</description>";
          }
          if (const auto* values = fields.find("enum"))
            out += "\n<enum>" + values->tojson() + "</enum>";
          RenderExtraKeys(out, fields, {"name", "type", "description", "enum"});
          out += "\n</parameter>";
        }
      }
    }
    if (parameters != nullptr)
      RenderExtraKeys(out, *parameters, {"type", "properties", "required"});
    if (parameters != nullptr) {
      if (const auto* required = parameters->find("required"))
        out += "\n<required>" + required->tojson() + "</required>";
    }
    out += "\n</parameters>";
    RenderExtraKeys(out, tool, {"type", "name", "description", "parameters"});
    out += "\n</function>";
  }
  out += kToolInstructions;
}

/// The template's `args_value | string` for scalars; containers keep tojson.
std::string ArgumentText(const ChatToolArgument& argument) {
  if (argument.is_string)
    return argument.value;
  const json::Value value = json::parse(argument.value);
  return IsContainer(value) ? value.tojson() : PythonStr(value);
}

std::string_view AfterLast(std::string_view text, std::string_view token) {
  const auto pos = text.rfind(token);
  return pos == std::string_view::npos ? text : text.substr(pos + token.size());
}

std::string_view BeforeFirst(std::string_view text, std::string_view token) {
  const auto pos = text.find(token);
  return pos == std::string_view::npos ? text : text.substr(0, pos);
}

bool Contains(std::string_view text, std::string_view token) {
  return text.find(token) != std::string_view::npos;
}

/// Rendered text with the ranges of client text in it.
struct Rendered {
  std::string text;
  std::vector<ContentSpan> spans;

  void Framing(std::string_view s) { text += s; }
  void Content(std::string_view s) {
    if (!s.empty())
      spans.push_back({text.size(), s.size()});
    text += s;
  }
  /// Appends `slice`, a view into `source.text`, with its client ranges.
  void Append(const Rendered& source, std::string_view slice) {
    const auto begin =
        static_cast<std::size_t>(slice.data() - source.text.data());
    const auto end = begin + slice.size();
    for (const ContentSpan& span : source.spans) {
      const auto from = std::max(span.offset, begin);
      const auto to = std::min(span.offset + span.size, end);
      if (from < to)
        spans.push_back({text.size() + from - begin, to - from});
    }
    text += slice;
  }
};

void RenderAssistant(Rendered& out, const ChatMessage& m, bool truncate) {
  Rendered content;
  if (!PythonStrip(m.reasoning).empty()) {
    content.Framing(kThinkOpen);
    content.Framing("\n");
    content.Content(m.reasoning);
    content.Framing(kThinkClose);
  } else if (!Contains(m.content, kThinkOpen) &&
             !Contains(m.content, kThinkClose)) {
    content.Framing(kEmptyThink);
  }
  content.Content(m.content);
  const std::string_view text = content.text;
  out.Framing(kImStart);
  out.Framing("assistant\n");
  if (!m.tool_calls.empty()) {
    if (!PythonStrip(text).empty()) {
      if (!truncate) {
        out.Append(content, PythonStrip(text));
        out.Framing("\n");
      } else {
        std::string_view c = text;
        if (Contains(c, kThinkClose))
          c = AfterLast(c, kThinkClose);
        else if (Contains(c, kThinkOpen))
          c = BeforeFirst(c, kThinkOpen);
        out.Framing(kEmptyThink);
        out.Append(content, c);
        out.Framing("\n");
      }
    } else {
      out.Framing(kEmptyThink);
    }
    for (const ChatToolCall& call : m.tool_calls) {
      out.Framing("<tool_call>\n<function=" + call.name + ">\n");
      for (const ChatToolArgument& argument : call.arguments) {
        out.Framing("<parameter=" + argument.name + ">\n");
        out.Content(ArgumentText(argument));
        out.Framing("\n</parameter>\n");
      }
      out.Framing("</function>\n</tool_call>\n");
    }
    out.Framing(kImEnd);
    return;
  }
  if (!truncate) {
    out.Append(content, PythonStrip(text));
    out.Framing(kImEnd);
    return;
  }
  Rendered c;
  if (Contains(text, kThinkOpen) && Contains(text, kThinkClose)) {
    c.Framing(kEmptyThink);
    c.Append(content, AfterLast(text, kThinkClose));
  } else {
    c.Append(content, text);
  }
  out.Append(c, PythonStrip(c.text));
  out.Framing(kImEnd);
}

const char* RoleName(ChatRole role) {
  switch (role) {
    case ChatRole::kSystem:
      return "system";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

}  // namespace

std::string_view PythonStrip(std::string_view value) {
  const auto whitespace = [](UChar32 cp) {
    // Python str.strip also strips the four ASCII information separators.
    return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
  };
  std::int32_t first = 0;
  auto last = static_cast<std::int32_t>(value.size());
  while (first < last) {
    auto next = first;
    UChar32 cp;
    U8_NEXT(value.data(), next, last, cp);
    if (!whitespace(cp))
      break;
    first = next;
  }
  while (first < last) {
    auto previous = last;
    UChar32 cp;
    U8_PREV(value.data(), first, previous, cp);
    if (!whitespace(cp))
      break;
    last = previous;
  }
  return value.substr(first, last - first);
}

ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning) {
  ChatOptions options;
  options.enable_thinking = reasoning.enabled.value_or(true);
  if (reasoning.effort.has_value()) {
    switch (*reasoning.effort) {
      case ReasoningEffort::kMinimal:
      case ReasoningEffort::kLow:
      case ReasoningEffort::kMedium:
        options.medium_effort = true;
        break;
      default:
        break;
    }
  }
  if (reasoning.preserve_thinking.has_value())
    options.truncate_history_thinking = !*reasoning.preserve_thinking;
  return options;
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatOptions& options,
                       std::vector<ContentSpan>* content_spans) {
  Rendered out;
  std::string system_message;
  std::span<const ChatMessage> loop = messages;
  if (!messages.empty() && messages.front().role == ChatRole::kSystem) {
    system_message = messages.front().content;
    loop = messages.subspan(1);
  }
  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < loop.size(); ++i) {
    if (loop[i].role == ChatRole::kUser)
      last_user = static_cast<std::ptrdiff_t>(i);
  }

  // The template always defines system_message, so every prompt opens with a
  // (possibly empty) system turn.
  out.Framing(kImStart);
  out.Framing("system\n");
  out.Content(system_message);
  if (!tools.empty()) {
    if (!system_message.empty())
      out.Framing("\n\n");
    RenderTools(out.text, tools);
  }
  out.Framing(kImEnd);

  for (std::size_t i = 0; i < loop.size(); ++i) {
    const ChatMessage& m = loop[i];
    const bool truncate = options.truncate_history_thinking &&
                          static_cast<std::ptrdiff_t>(i) < last_user;
    switch (m.role) {
      case ChatRole::kAssistant:
        RenderAssistant(out, m, truncate);
        break;
      case ChatRole::kUser:
      case ChatRole::kSystem:
        out.Framing(kImStart);
        out.Framing(RoleName(m.role));
        out.Framing("\n");
        out.Content(m.content);
        if (m.role == ChatRole::kUser &&
            static_cast<std::ptrdiff_t>(i) == last_user &&
            options.medium_effort)
          out.Framing("\n\n{reasoning effort: efficient}");
        out.Framing(kImEnd);
        break;
      case ChatRole::kTool: {
        if (i > 0 && loop[i - 1].role != ChatRole::kTool) {
          out.Framing(kImStart);
          out.Framing("user\n");
        }
        out.Framing("<tool_response>\n");
        out.Content(m.content);
        out.Framing("\n</tool_response>\n");
        const bool last = i + 1 == loop.size();
        if (last || loop[i + 1].role != ChatRole::kTool)
          out.Framing(kImEnd);
        break;
      }
    }
  }
  if (options.add_generation_prompt) {
    out.Framing(kImStart);
    out.Framing(options.enable_thinking ? "assistant\n<think>\n"
                                        : "assistant\n<think></think>");
  }
  if (content_spans != nullptr)
    *content_spans = std::move(out.spans);
  return std::move(out.text);
}

std::vector<std::uint32_t> TokenizeChat(
    const Tokenizer& tokenizer, std::string_view rendered,
    std::span<const ContentSpan> content_spans) {
  std::vector<TokenId> tokens;
  const auto append = [&](std::string_view text, bool parse_control) {
    const auto ids = tokenizer.Encode(text, parse_control);
    tokens.insert(tokens.end(), ids.begin(), ids.end());
  };
  std::size_t cursor = 0;
  for (const ContentSpan& span : content_spans) {
    const auto content = rendered.substr(span.offset, span.size);
    if (!tokenizer.SpellsControl(content))
      continue;
    append(rendered.substr(cursor, span.offset - cursor), true);
    append(content, false);
    cursor = span.offset + span.size;
  }
  append(rendered.substr(cursor), true);
  return tokens;
}

}  // namespace gufo::models::nemotron35_bolt
