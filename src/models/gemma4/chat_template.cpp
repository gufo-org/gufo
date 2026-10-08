#include "src/models/gemma4/chat_template.hpp"

#include <unicode/locid.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::models::gemma4 {
namespace {

using json::Value;
using tokenization::ChatMessage;
using tokenization::ChatRole;
using tokenization::ChatTool;

constexpr std::string_view kQ = kStringQuote;

// ---------------------------------------------------------------------------
// Python/Jinja value semantics over gufo::json. A null pointer is Jinja's
// Undefined; a JSON null is Python's None.
// ---------------------------------------------------------------------------

const Value* Get(const Value* object, const std::string& key) {
  return object != nullptr && object->is_object() ? object->find(key) : nullptr;
}

bool Truthy(const Value* v) {
  if (v == nullptr || v->is_null()) {
    return false;
  }
  if (v->is_bool()) {
    return v->as_bool();
  }
  if (v->is_number()) {
    return v->as_double() != 0.0;
  }
  if (v->is_string()) {
    return !v->str().empty();
  }
  return !v->empty();
}

/// Python's repr/str of a float ('short' style): shortest round-trip digits,
/// scientific notation when the decimal exponent is below -4 or at least 16.
std::string PyFloat(double x) {
  if (std::isnan(x)) {
    return "nan";
  }
  if (std::isinf(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  std::array<char, 64> buf{};
  const auto [end, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), x,
                                       std::chars_format::scientific);
  (void)ec;
  std::string sci(buf.data(), end);  // [-]d[.ddd]e[+-]xx
  const bool negative = sci.front() == '-';
  if (negative) {
    sci.erase(0, 1);
  }
  const std::size_t e = sci.find('e');
  std::string digits = sci.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  const int exponent = std::stoi(sci.substr(e + 1));
  std::string out = negative ? "-" : "";
  if (exponent < -4 || exponent >= 16) {
    out += digits.substr(0, 1);
    if (digits.size() > 1) {
      out += "." + digits.substr(1);
    }
    char exp[16];
    std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+',
                  std::abs(exponent));
    return out + exp;
  }
  if (exponent < 0) {
    return out + "0." +
           std::string(static_cast<std::size_t>(-exponent - 1), '0') + digits;
  }
  const auto int_digits = static_cast<std::size_t>(exponent) + 1;
  if (digits.size() <= int_digits) {
    return out + digits + std::string(int_digits - digits.size(), '0') + ".0";
  }
  return out + digits.substr(0, int_digits) + "." + digits.substr(int_digits);
}

std::string PyNumber(double x) {
  // gufo::json keeps numbers as doubles; integral values print as ints.
  if (std::isfinite(x) && std::floor(x) == x && std::fabs(x) < 9.0e15) {
    return std::to_string(static_cast<long long>(x));
  }
  return PyFloat(x);
}

std::string PyStrRepr(const std::string& s) {
  const bool has_single = s.find('\'') != std::string::npos;
  const bool has_double = s.find('"') != std::string::npos;
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (unsigned char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == static_cast<unsigned char>(quote)) {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else if (c < 0x20 || c == 0x7f) {
      char hex[8];
      std::snprintf(hex, sizeof(hex), "\\x%02x", c);
      out += hex;
    } else {
      out += static_cast<char>(c);
    }
  }
  out += quote;
  return out;
}

std::string PyStr(const Value& v);

/// Python repr(), used for containers and their members.
std::string PyRepr(const Value& v) {
  if (v.is_string()) {
    return PyStrRepr(v.str());
  }
  if (v.is_array()) {
    std::string out = "[";
    bool first = true;
    for (const auto& item : v.items()) {
      out += first ? "" : ", ";
      first = false;
      out += PyRepr(item);
    }
    return out + "]";
  }
  if (v.is_object()) {
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : v.members()) {
      out += first ? "" : ", ";
      first = false;
      out += PyStrRepr(key) + ": " + PyRepr(value);
    }
    return out + "}";
  }
  return PyStr(v);
}

/// Python str(), which Jinja uses for `{{ value }}`.
std::string PyStr(const Value& v) {
  if (v.is_null()) {
    return "None";
  }
  if (v.is_bool()) {
    return v.as_bool() ? "True" : "False";
  }
  if (v.is_number()) {
    return PyNumber(v.as_double());
  }
  if (v.is_string()) {
    return v.str();
  }
  return PyRepr(v);
}

/// `{{ x }}` of a possibly undefined value.
std::string Text(const Value* v) {
  return v == nullptr ? "" : PyStr(*v);
}

std::string ToUpper(const std::string& s) {
  std::string out;
  icu::UnicodeString::fromUTF8(s)
      .toUpper(icu::Locale::getRoot())
      .toUTF8String(out);
  return out;
}

std::string ToLower(const std::string& s) {
  std::string out;
  icu::UnicodeString::fromUTF8(s)
      .toLower(icu::Locale::getRoot())
      .toUTF8String(out);
  return out;
}

/// Jinja `| upper` of a possibly undefined value.
std::string Upper(const Value* v) {
  return ToUpper(Text(v));
}

/// Jinja `dictsort`: stable, case-insensitive key order.
std::vector<std::pair<const std::string*, const Value*>> DictSort(
    const Value& object) {
  std::vector<std::pair<std::string, std::size_t>> keys;
  std::vector<std::pair<const std::string*, const Value*>> items;
  for (const auto& [key, value] : object.members()) {
    keys.emplace_back(ToLower(key), items.size());
    items.emplace_back(&key, &value);
  }
  std::stable_sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  std::vector<std::pair<const std::string*, const Value*>> sorted;
  sorted.reserve(items.size());
  for (const auto& key : keys) {
    sorted.push_back(items[key.second]);
  }
  return sorted;
}

/// Iteration of a Jinja value: list items, mapping keys or string chars.
std::vector<Value> Iterate(const Value& v) {
  std::vector<Value> out;
  if (v.is_array()) {
    out = v.items();
  } else if (v.is_object()) {
    for (const auto& [key, value] : v.members()) {
      (void)value;
      out.emplace_back(key);
    }
  } else if (v.is_string()) {
    const std::string& s = v.str();
    for (std::size_t i = 0; i < s.size();) {
      std::size_t n = 1;
      const auto c = static_cast<unsigned char>(s[i]);
      if (c >= 0xF0) {
        n = 4;
      } else if (c >= 0xE0) {
        n = 3;
      } else if (c >= 0xC0) {
        n = 2;
      }
      out.emplace_back(s.substr(i, n));
      i += n;
    }
  }
  return out;
}

bool IsPythonSpace(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

/// Decodes the UTF-8 character starting at `i`; invalid bytes are not
/// whitespace, so their exact value does not matter here.
std::uint32_t CodePointAt(std::string_view s, std::size_t i, std::size_t* n) {
  const auto c = static_cast<unsigned char>(s[i]);
  std::size_t len = c < 0x80    ? 1
                    : c >= 0xF0 ? 4
                    : c >= 0xE0 ? 3
                    : c >= 0xC0 ? 2
                                : 1;
  if (i + len > s.size()) {
    len = 1;
  }
  std::uint32_t cp = len == 1 ? c : c & (0xFFU >> (len + 1));
  for (std::size_t k = 1; k < len; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
  }
  *n = len;
  return cp;
}

/// Python str.strip() with no arguments.
std::string PyTrim(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t n = 0;
    if (!IsPythonSpace(CodePointAt(s, begin, &n))) {
      break;
    }
    begin += n;
  }
  std::size_t end = s.size();
  while (end > begin) {
    std::size_t start = end - 1;
    while (start > begin &&
           (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
      --start;
    }
    std::size_t n = 0;
    if (!IsPythonSpace(CodePointAt(s, start, &n)) || start + n != end) {
      break;
    }
    end = start;
  }
  return std::string(s.substr(begin, end - begin));
}

std::string StripThinking(std::string_view text) {
  std::string result;
  std::size_t start = 0;
  while (true) {
    const std::size_t split = text.find(kThoughtEnd, start);
    const std::string_view part = text.substr(
        start, split == std::string_view::npos ? std::string_view::npos
                                               : split - start);
    const std::size_t open = part.find("<|channel>");
    result += open == std::string_view::npos ? part : part.substr(0, open);
    if (split == std::string_view::npos) {
      break;
    }
    start = split + kThoughtEnd.size();
  }
  return PyTrim(result);
}

// ---------------------------------------------------------------------------
// Template macros.
// ---------------------------------------------------------------------------

void FormatArgument(const Value& v, bool escape_keys, std::string* out) {
  if (v.is_null()) {
    *out += "null";
  } else if (v.is_string()) {
    *out += kQ;
    *out += v.str();
    *out += kQ;
  } else if (v.is_bool()) {
    *out += v.as_bool() ? "true" : "false";
  } else if (v.is_object()) {
    *out += '{';
    bool first = true;
    for (const auto& [key, value] : DictSort(v)) {
      *out += first ? "" : ",";
      first = false;
      if (escape_keys) {
        *out += kQ;
        *out += *key;
        *out += kQ;
      } else {
        *out += *key;
      }
      *out += ':';
      FormatArgument(*value, escape_keys, out);
    }
    *out += '}';
  } else if (v.is_array()) {
    *out += '[';
    bool first = true;
    for (const auto& item : v.items()) {
      *out += first ? "" : ",";
      first = false;
      FormatArgument(item, escape_keys, out);
    }
    *out += ']';
  } else {
    *out += PyStr(v);
  }
}

void QuotedList(const Value& list, std::string* out) {
  bool first = true;
  for (const auto& item : Iterate(list)) {
    *out += first ? "" : ",";
    first = false;
    *out += kQ;
    *out += PyStr(item);
    *out += kQ;
  }
}

// The template shows a string's allowed values only through `enum` and drops
// `const`. A single-value enum is the same constraint in the form it shows;
// without it Gemma 4 does not know the required value (gufo tool grammar).
Value ShowConstants(const Value& schema) {
  if (!schema.is_object()) {
    return schema;
  }
  Value shown = Value::object();
  for (const auto& [key, value] : schema.members()) {
    if (key == "properties" && value.is_object()) {
      Value properties = Value::object();
      for (const auto& [name, property] : value.members()) {
        properties.append_member(name, ShowConstants(property));
      }
      shown.append_member(key, std::move(properties));
    } else if (key == "items" || key == "parameters") {
      shown.append_member(key, ShowConstants(value));
    } else {
      shown.append_member(key, value);
    }
  }
  const Value* constant = schema.find("const");
  if (constant != nullptr && constant->is_string() &&
      !schema.contains("enum") && Upper(schema.find("type")) == "STRING") {
    Value values = Value::array();
    values.push_back(*constant);
    shown["enum"] = std::move(values);
  }
  return shown;
}

void FormatParameters(const Value& properties, bool filter_keys,
                      std::string* out) {
  static constexpr std::array<std::string_view, 5> kStandardKeys = {
      "description", "type", "properties", "required", "nullable"};
  bool found_first = false;
  for (const auto& [key_ptr, value] : DictSort(properties)) {
    const std::string& key = *key_ptr;
    if (filter_keys && std::find(kStandardKeys.begin(), kStandardKeys.end(),
                                 key) != kStandardKeys.end()) {
      continue;
    }
    bool add_comma = false;
    const auto comma = [&] {
      if (add_comma) {
        *out += ',';
      } else {
        add_comma = true;
      }
    };
    *out += found_first ? "," : "";
    found_first = true;
    *out += key + ":{";
    if (const Value* description = Get(value, "description");
        Truthy(description)) {
      *out += "description:";
      *out += kQ;
      *out += PyStr(*description);
      *out += kQ;
      add_comma = true;
    }
    const std::string type = Upper(Get(value, "type"));
    if (type == "STRING") {
      if (const Value* values = Get(value, "enum"); Truthy(values)) {
        comma();
        *out += "enum:";
        FormatArgument(*values, true, out);
      }
    } else if (type == "ARRAY") {
      const Value* items = Get(value, "items");
      if (items != nullptr && items->is_object() && Truthy(items)) {
        comma();
        *out += "items:{";
        bool items_first = false;
        for (const auto& [item_key, item_value] : DictSort(*items)) {
          if (item_value->is_null()) {
            continue;
          }
          *out += items_first ? "," : "";
          items_first = true;
          if (*item_key == "properties") {
            *out += "properties:{";
            if (item_value->is_object()) {
              FormatParameters(*item_value, false, out);
            }
            *out += '}';
          } else if (*item_key == "required") {
            *out += "required:[";
            QuotedList(*item_value, out);
            *out += ']';
          } else if (*item_key == "type") {
            *out += "type:";
            if (item_value->is_string()) {
              FormatArgument(Value(ToUpper(item_value->str())), true, out);
            } else {
              Value upper = Value::array();
              for (const auto& t : Iterate(*item_value)) {
                upper.push_back(Value(ToUpper(PyStr(t))));
              }
              FormatArgument(upper, true, out);
            }
          } else {
            *out += *item_key + ":";
            FormatArgument(*item_value, true, out);
          }
        }
        *out += '}';
      }
    }
    if (Truthy(Get(value, "nullable"))) {
      comma();
      *out += "nullable:true";
    }
    if (type == "OBJECT") {
      const Value* nested = Get(value, "properties");
      if (nested != nullptr && nested->is_object()) {
        comma();
        *out += "properties:{";
        FormatParameters(*nested, false, out);
        *out += '}';
      } else if (value->is_object()) {
        comma();
        *out += "properties:{";
        FormatParameters(*value, true, out);
        *out += '}';
      }
      if (const Value* required = Get(value, "required"); Truthy(required)) {
        comma();
        *out += "required:[";
        QuotedList(*required, out);
        *out += ']';
      }
    }
    comma();
    *out += "type:";
    *out += kQ;
    *out += type;
    *out += kQ;
    *out += '}';
  }
}

std::string FormatFunctionDeclaration(const Value& tool) {
  const Value* function = Get(&tool, "function");
  std::string out = "declaration:" + Text(Get(function, "name")) +
                    "{description:" + std::string(kQ) +
                    Text(Get(function, "description")) + std::string(kQ);
  if (const Value* params = Get(function, "parameters"); Truthy(params)) {
    out += ",parameters:{";
    if (const Value* properties = Get(params, "properties");
        Truthy(properties) && properties->is_object()) {
      out += "properties:{";
      FormatParameters(*properties, false, &out);
      out += "},";
    }
    if (const Value* required = Get(params, "required"); Truthy(required)) {
      out += "required:[";
      QuotedList(*required, &out);
      out += "],";
    }
    if (const Value* type = Get(params, "type"); Truthy(type)) {
      out += "type:" + std::string(kQ) + Upper(type) + std::string(kQ) + "}";
    }
  }
  if (function != nullptr && function->is_object() &&
      function->contains("response")) {
    const Value* response = function->find("response");
    out += ",response:{";
    if (const Value* description = Get(response, "description");
        Truthy(description)) {
      out += "description:" + std::string(kQ) + PyStr(*description) +
             std::string(kQ) + ",";
    }
    if (Upper(Get(response, "type")) == "OBJECT") {
      out += "type:" + std::string(kQ) + "OBJECT" + std::string(kQ) + "}";
    }
  }
  out += '}';
  return out;
}

std::string ToolResponseBlock(const std::string& name, const Value& response) {
  std::string out = "<|tool_response>";
  if (response.is_object()) {
    out += "response:" + name + "{";
    bool first = true;
    for (const auto& [key, value] : DictSort(response)) {
      out += first ? "" : ",";
      first = false;
      out += *key + ":";
      FormatArgument(*value, false, &out);
    }
    out += '}';
  } else {
    out += "response:" + name + "{value:";
    FormatArgument(response, false, &out);
    out += '}';
  }
  return out + "<tool_response|>";
}

std::string_view RoleName(ChatRole role) {
  switch (role) {
    case ChatRole::kSystem:
      return "system";
    case ChatRole::kDeveloper:
      return "developer";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

Value ArgumentsObject(const ChatMessage::ToolCall& call) {
  Value object = Value::object();
  for (const auto& argument : call.arguments) {
    if (argument.is_string) {
      object[argument.name] = Value(argument.value);
      continue;
    }
    try {
      object[argument.name] = json::parse(argument.value);
    } catch (const std::exception&) {
      object[argument.name] = Value(argument.value);
    }
  }
  return object;
}

enum class MessageType : std::uint8_t {
  kNone,
  kThink,
  kTool,
  kToolCall,
  kToolResponse,
};

}  // namespace

ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning) {
  ChatOptions options;
  if (reasoning.enabled.has_value()) {
    options.enable_thinking = *reasoning.enabled;
  } else if (reasoning.effort.has_value()) {
    options.enable_thinking = *reasoning.effort != ReasoningEffort::kMinimal;
  }
  options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
  return options;
}

bool ChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg) {
  const auto source = reader.GetMetadataString("tokenizer.chat_template");
  if (!source) {
    if (error_msg != nullptr) {
      *error_msg = "gemma4 GGUF has no tokenizer.chat_template";
    }
    return false;
  }
  const std::string digest = crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(source->data()), source->size()));
  if (digest != UnslothTemplateSha256()) {
    if (error_msg != nullptr) {
      *error_msg = "unqualified gemma4 chat template (sha256 " + digest + ")";
    }
    return false;
  }
  return true;
}

std::optional<RenderedPrompt> ChatTemplate::Render(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatOptions& options, std::string* error_msg,
    std::vector<std::size_t>* image_offsets) {
  const auto fail = [&](std::string message) -> std::optional<RenderedPrompt> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return std::nullopt;
  };
  if (image_offsets != nullptr) {
    image_offsets->clear();
  }
  for (const auto& message : messages) {
    if (message.images.empty()) {
      continue;
    }
    if (image_offsets == nullptr) {
      return fail("gemma4 image input requires a vision sidecar");
    }
    std::size_t previous = 0;
    for (const auto& image : message.images) {
      if (message.role != ChatRole::kUser || image.bytes == nullptr ||
          image.offset < previous || image.offset > message.content.size()) {
        return fail("invalid user image content");
      }
      previous = image.offset;
    }
  }
  // "call:NAME{" and "declaration:NAME{" end a function name at its first
  // brace, so a name holding one cannot be framed.
  constexpr std::string_view kBraceName =
      "gemma4 function names cannot contain '{'";
  for (const auto& message : messages) {
    if (message.name.find('{') != std::string::npos ||
        std::ranges::any_of(message.tool_calls, [](const auto& call) {
          return call.name.find('{') != std::string::npos;
        })) {
      return fail(std::string(kBraceName));
    }
  }
  std::vector<Value> tool_data;
  for (const auto& tool : tools) {
    if (tool.name.find('{') != std::string::npos) {
      return fail(std::string(kBraceName));
    }
    try {
      tool_data.push_back(json::parse(tool.definition_json.empty()
                                          ? std::string("{}")
                                          : tool.definition_json));
    } catch (const std::exception& e) {
      return fail(std::string("invalid tool definition: ") + e.what());
    }
    if (!tool_data.back().is_object()) {
      return fail("tool definitions must be objects");
    }
    if (const Value* function = tool_data.back().find("function");
        function != nullptr && function->is_object()) {
      tool_data.back()["function"] = ShowConstants(*function);
    }
  }

  std::string out = "<bos>";
  MessageType prev_type = MessageType::kNone;
  std::string_view prev_non_tool_role;
  std::span<const ChatMessage> loop = messages;
  const bool first_is_system =
      !messages.empty() && (messages[0].role == ChatRole::kSystem ||
                            messages[0].role == ChatRole::kDeveloper);
  if (options.enable_thinking || !tools.empty() || first_is_system) {
    out += "<|turn>system\n";
    if (options.enable_thinking) {
      out += "<|think|>\n";
      prev_type = MessageType::kThink;
    }
    if (first_is_system) {
      // Server-authored instructions follow the client's system text.
      out += PyTrim(messages[0].content + messages[0].framing_suffix);
      loop = messages.subspan(1);
    }
    for (const auto& tool : tool_data) {
      out += "<|tool>" + PyTrim(FormatFunctionDeclaration(tool)) + "<tool|>";
      prev_type = MessageType::kTool;
    }
    out += "<turn|>\n";
  }

  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < loop.size(); ++i) {
    if (loop[i].role == ChatRole::kUser) {
      last_user = static_cast<std::ptrdiff_t>(i);
    }
  }

  // Per-turn context an agent replaces on every request follows a tool
  // result or another user turn, never an assistant reply (that is ordinary
  // chat, which the next request keeps).
  const bool final_user_turn =
      last_user >= 1 &&
      static_cast<std::size_t>(last_user) + 1 == loop.size() &&
      loop.back().images.empty() &&
      (loop[static_cast<std::size_t>(last_user) - 1].role == ChatRole::kTool ||
       loop[static_cast<std::size_t>(last_user) - 1].role == ChatRole::kUser);
  bool seen_assistant = false;
  std::optional<std::size_t> final_user_start;
  std::size_t turn_end = 0;

  for (std::size_t index = 0; index < loop.size(); ++index) {
    const ChatMessage& message = loop[index];
    if (message.role == ChatRole::kTool) {
      continue;
    }
    if (message.role == ChatRole::kAssistant) {
      seen_assistant = true;
    } else if (final_user_turn && seen_assistant &&
               static_cast<std::ptrdiff_t>(index) == last_user) {
      final_user_start =
          loop[index - 1].role == ChatRole::kTool ? turn_end : out.size();
    }
    prev_type = MessageType::kNone;
    const std::string_view raw_role = RoleName(message.role);
    const std::string_view role =
        message.role == ChatRole::kAssistant ? "model" : raw_role;
    const bool continue_turn =
        role == "model" && prev_non_tool_role == "assistant";
    if (!continue_turn) {
      out += "<|turn>";
      out += role;
      out += '\n';
    }

    const bool thinking_gate =
        static_cast<std::ptrdiff_t>(index) > last_user ||
        (options.preserve_thinking && !message.tool_calls.empty());
    if (!message.thought.empty() && thinking_gate) {
      out += std::string(kThoughtStart) + message.thought +
             std::string(kThoughtEnd);
    }

    if (!message.tool_calls.empty()) {
      for (const auto& call : message.tool_calls) {
        out += std::string(kToolCallStart) + "call:" + call.name + "{";
        const Value arguments = ArgumentsObject(call);
        bool first = true;
        for (const auto& [key, value] : DictSort(arguments)) {
          out += first ? "" : ",";
          first = false;
          out += *key + ":";
          FormatArgument(*value, false, &out);
        }
        out += "}" + std::string(kToolCallEnd);
      }
      prev_type = MessageType::kToolCall;
    }

    bool tool_responses = false;
    if (!message.tool_calls.empty()) {
      for (std::size_t k = index + 1;
           k < loop.size() && loop[k].role == ChatRole::kTool; ++k) {
        const ChatMessage& follow = loop[k];
        std::string name = follow.name.empty() ? "unknown" : follow.name;
        for (const auto& call : message.tool_calls) {
          if (call.id == follow.tool_call_id) {
            name = call.name;
          }
        }
        out += ToolResponseBlock(name, Value(follow.content));
        tool_responses = true;
        prev_type = MessageType::kToolResponse;
      }
    }

    std::string content;
    if (!message.images.empty()) {
      // A content-part list: every text part is trimmed on its own and each
      // image renders as a placeholder. Adjacent flattened text parts form
      // one segment.
      std::size_t cursor = 0;
      for (const auto& image : message.images) {
        content += PyTrim(std::string_view(message.content)
                              .substr(cursor, image.offset - cursor));
        image_offsets->push_back(out.size() + content.size());
        content += kImagePlaceholder;
        cursor = image.offset;
      }
      content += PyTrim(std::string_view(message.content).substr(cursor));
    } else {
      content = role == "model"
                    ? StripThinking(message.content)
                    : PyTrim(message.content + message.framing_suffix);
    }
    out += content;
    const bool has_content = !PyTrim(content).empty();

    std::string_view next_role;
    bool next_found = false;
    for (std::size_t j = index + 1; j < loop.size(); ++j) {
      if (loop[j].role != ChatRole::kTool) {
        next_role = RoleName(loop[j].role);
        next_found = true;
        break;
      }
    }
    const bool continues_into_next =
        role == "model" && next_role == "assistant" &&
        (message.tool_calls.empty() || tool_responses);
    turn_end = out.size();
    if (prev_type == MessageType::kToolCall && !tool_responses) {
      out += "<|tool_response>";
    } else if (continues_into_next) {
    } else if (!(tool_responses && !has_content && !next_found)) {
      out += "<turn|>\n";
    }
    prev_non_tool_role = raw_role;
  }

  RenderedPrompt rendered;
  rendered.generation_prompt_offset = out.size();
  rendered.stable_prefix_offset = final_user_start.value_or(out.size());
  if (options.add_generation_prompt) {
    if (prev_type != MessageType::kToolResponse &&
        prev_type != MessageType::kToolCall) {
      out += "<|turn>model\n";
      if (!options.enable_thinking) {
        out += std::string(kThoughtStart) + std::string(kThoughtEnd);
      }
    } else if (prev_type == MessageType::kToolResponse &&
               options.enable_thinking) {
      out += kThoughtStart;
    }
  }
  rendered.text = std::move(out);
  return rendered;
}

}  // namespace gufo::models::gemma4
