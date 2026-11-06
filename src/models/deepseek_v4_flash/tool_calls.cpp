#include "src/models/deepseek_v4_flash/tool_calls.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <gufo/ext/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::models::deepseek_v4_flash {

namespace {

// The API surface trims framing with ASCII whitespace; the scanner keeps the
// same reading so parser decisions do not depend on Unicode table updates.
std::string_view TrimAscii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0)
    value.remove_prefix(1);
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0)
    value.remove_suffix(1);
  return value;
}

std::optional<std::string> Attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const std::size_t start = tag.find(prefix);
  if (start == std::string_view::npos)
    return std::nullopt;
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = tag.find('"', value_start);
  if (end == std::string_view::npos)
    return std::nullopt;
  return std::string(tag.substr(value_start, end - value_start));
}

// Hold a possible closing delimiter until its next byte disambiguates it.
std::string_view BeforePartialDelimiter(std::string_view value,
                                        std::string_view delimiter) {
  for (auto size = std::min(value.size(), delimiter.size()); size > 0; --size)
    if (delimiter.starts_with(value.substr(value.size() - size)))
      return value.substr(0, value.size() - size);
  return value;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace

void ScanDeepSeekToolCalls(
    std::string_view text, std::span<const std::string_view> markers,
    std::vector<ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  std::size_t cursor = 0;
  std::size_t envelope_begin = std::string_view::npos;
  std::string envelope_end;
  while (cursor < text.size()) {
    std::size_t invoke_start = std::string_view::npos;
    std::size_t syntax = 0;
    for (std::size_t index = 0; index < DeepSeekToolSyntax::kInvokeOpens.size();
         ++index) {
      const std::size_t position =
          text.find(DeepSeekToolSyntax::kInvokeOpens[index], cursor);
      if (position < invoke_start) {
        invoke_start = position;
        syntax = index;
      }
    }
    if (spans) {
      // Examine envelope boundaries only outside the parameter/invocation
      // ranges consumed below: an outer closing tag can be argument data.
      if (envelope_begin == std::string_view::npos) {
        std::string_view opening;
        std::size_t begin = std::string_view::npos;
        for (const auto marker : markers) {
          const auto position = text.find(marker, cursor);
          if (position < begin) {
            begin = position;
            opening = marker;
          }
        }
        // Bare invocation examples outside an outer envelope are prose.
        if (begin == std::string_view::npos)
          break;
        envelope_begin = begin;
        envelope_end =
            "</" + std::string(opening.substr(opening.find('<') + 1));
        cursor = begin + opening.size();
        continue;
      }
      if (const auto end = text.find(envelope_end, cursor);
          end < invoke_start) {
        cursor = end + envelope_end.size();
        spans->emplace_back(envelope_begin, cursor);
        envelope_begin = std::string_view::npos;
        continue;
      }
    }
    if (invoke_start == std::string_view::npos) {
      break;
    }
    const std::size_t tag_end = text.find('>', invoke_start);
    if (tag_end == std::string_view::npos) {
      break;
    }
    ScannedToolCall call;
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    bool valid = !call.name.empty();
    std::size_t protected_end = tag_end + 1;
    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < text.size()) {
      auto remainder = TrimAscii(text.substr(parameter_cursor));
      parameter_cursor =
          static_cast<std::size_t>(remainder.data() - text.data());
      if (!remainder.starts_with(DeepSeekToolSyntax::kParameterOpens[syntax])) {
        break;
      }
      const std::size_t parameter_start = parameter_cursor;
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      std::size_t parameter_end = text.find(
          DeepSeekToolSyntax::kParameterCloses[syntax], parameter_tag_end);
      if (parameter_tag_end == std::string_view::npos ||
          parameter_end == std::string_view::npos) {
        if (partial && parameter_tag_end != std::string_view::npos) {
          const auto tag = text.substr(parameter_start,
                                       parameter_tag_end - parameter_start + 1);
          const auto name = Attribute(tag, "name");
          if (name && !name->empty() &&
              Attribute(tag, "string").value_or("true") == "true")
            call.unfinished = ScannedToolArgument{
                .name = std::string(*name),
                .value = std::string(BeforePartialDelimiter(
                    text.substr(parameter_tag_end + 1),
                    DeepSeekToolSyntax::kParameterCloses[syntax])),
                .is_string = true};
        }
        protected_end = text.size();
        valid = false;
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      if (parameter_name && !parameter_name->empty() &&
          (!string_value || *string_value == "true" ||
           *string_value == "false")) {
        const bool is_string = string_value.value_or("true") == "true";
        // JSON strings may contain DSML delimiters. Find the end of the JSON
        // value before interpreting the enclosing markup.
        while (
            !is_string && parameter_end != std::string_view::npos &&
            !TryParseJson(TrimAscii(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))))
          parameter_end = text.find(
              DeepSeekToolSyntax::kParameterCloses[syntax],
              parameter_end +
                  DeepSeekToolSyntax::kParameterCloses[syntax].size());
        if (parameter_end == std::string_view::npos) {
          protected_end = text.size();
          valid = false;
          break;
        }
        const auto raw = text.substr(parameter_tag_end + 1,
                                     parameter_end - parameter_tag_end - 1);
        ScannedToolArgument argument{
            .name = std::string(*parameter_name),
            .value = std::string(is_string ? raw : TrimAscii(raw)),
            .is_string = is_string,
        };
        // As for Qwen calls: drop an identical repeat and reject a
        // conflicting one instead of letting the last value win.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(std::move(argument));
        } else if (previous->value != argument.value ||
                   previous->is_string != argument.is_string) {
          valid = false;
        }
      } else {
        valid = false;
      }
      parameter_cursor =
          parameter_end + DeepSeekToolSyntax::kParameterCloses[syntax].size();
      protected_end = parameter_cursor;
    }
    const bool complete =
        text.substr(parameter_cursor)
            .starts_with(DeepSeekToolSyntax::kInvokeCloses[syntax]);
    call.valid = valid;
    call.complete = complete;
    if ((valid && complete) || (partial && !call.name.empty())) {
      calls->push_back(std::move(call));
      if (!(valid && complete)) {
        break;
      }
    }
    cursor = complete ? parameter_cursor +
                            DeepSeekToolSyntax::kInvokeCloses[syntax].size()
                      : protected_end;
  }
}

}  // namespace gufo::models::deepseek_v4_flash
