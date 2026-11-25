// Tool-call dialect of the text the runner streams (see runner.cpp, Piece()).
//
// The model writes a call as
// <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME",
// "args":{...}}<|end_message|>; the name is written twice. Gufo v0.35.0's
// grammar description (ToolSyntax: open + name_prefix + NAME + name_suffix +
// arguments + close) can express the name only once, so the runner does not
// stream the header and the first name: the call reaches the host as
// <|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>,
// which the description covers exactly (open = the invoke token, name_prefix =
// {"name":", name_suffix =
// ","args":, close = }<|end_message|>). The scanner below rebuilds the
// canonical text and reuses the parser of src/text (tool_dialect.cpp).
#pragma once

#include <gufo/ext/text.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/inkling2/text/gufo_adapter.hpp"
#include "src/models/inkling2/text/tool_dialect.hpp"

namespace gufo::models::inkling2 {

namespace stream_dialect {
namespace d = ::inkling2::text::dialect;

inline constexpr std::string_view kNamePrefix = "{\"name\":\"";
/// The complete call header of the streamed dialect (syntax open +
/// name_prefix).
inline constexpr std::string_view kReasoningHeader =
    "<|content_invoke_tool_json|>{\"name\":\"";

/// The JSON string starting after the opening quote at `from`: its decoded end
/// position (just past the closing quote) or npos when the text ends first.
inline std::size_t NameEnd(std::string_view text, std::size_t from) {
  for (std::size_t i = from; i < text.size(); ++i) {
    if (text[i] == '\\')
      ++i;
    else if (text[i] == '"')
      return i + 1;
  }
  return std::string_view::npos;
}

inline void Scan(std::string_view text, std::span<const std::string_view>,
                 std::vector<ext::ScannedToolCall>* calls,
                 std::vector<std::pair<std::size_t, std::size_t>>* spans,
                 bool partial) {
  // canonical text with "<|message_model|>NAME" before each invoke token
  std::string canon;
  struct Ins {
    std::size_t canon_at, orig_at, len;
  };
  std::vector<Ins> ins;
  std::size_t at = 0;
  while (true) {
    const auto m = text.find(d::kInvokeJson, at);
    if (m == std::string_view::npos)
      break;
    canon.append(text.substr(at, m - at));
    const auto name_at = m + d::kInvokeJson.size();
    std::string header;
    if (text.substr(name_at).starts_with(kNamePrefix)) {
      const auto s = name_at + kNamePrefix.size();
      const auto e = NameEnd(text, s);
      if (e != std::string_view::npos) {
        header =
            std::string(d::kHeader) + std::string(text.substr(s, e - 1 - s));
      }
    }
    ins.push_back({canon.size(), m, header.size()});
    canon += header;
    canon.append(d::kInvokeJson);
    at = name_at;
  }
  canon.append(text.substr(at));
  std::vector<ext::ScannedToolCall> found;
  std::vector<std::pair<std::size_t, std::size_t>> cspans;
  ::inkling2::text::gufo_adapter::ScanToolCalls(canon, {}, &found, &cspans,
                                                partial);
  for (auto& c : found)
    calls->push_back(std::move(c));
  const auto back = [&](std::size_t c) {
    std::size_t sub = 0;
    for (const auto& i : ins) {
      if (c >= i.canon_at + i.len)
        sub += i.len;
      else if (c >= i.canon_at)
        return i.orig_at;
    }
    return c - sub;
  };
  for (auto& s : cspans)
    spans->emplace_back(back(s.first), back(s.second));
}
}  // namespace stream_dialect

inline ext::ToolDialect MakeStreamToolDialect() {
  namespace d = ::inkling2::text::dialect;
  ext::ToolDialect t;
  t.id = std::string(d::kId);
  t.markers = {d::kInvokeJson};
  t.closers = {d::kEndMessage};
  // A call header written inside a reasoning phase ends it (see runner.cpp,
  // InitialOutputState: a required call with reasoning off is generated in a
  // phase the model's own header and function name tokens, which have no
  // streamed text, fill).
  t.reasoning_markers = {stream_dialect::kReasoningHeader};
  t.scan_tool_calls = &stream_dialect::Scan;
  auto& s = t.syntax.emplace();
  s.open = std::string(d::kInvokeJson);
  s.name_prefix = std::string(stream_dialect::kNamePrefix);
  s.name_suffix = "\",\"args\":";
  s.close = "}" + std::string(d::kEndMessage);
  s.name_delimiters = "<\"\\";
  s.values.whitespace = false;  // the template and the model write compact JSON
  // The template declares tools and re-renders call arguments with sorted keys,
  // and the model writes them so: in schema order it could not go back to a
  // property declared before one it already wrote.
  s.values.sorted_keys = true;
  s.reasoning_header = true;
  return t;
}

}  // namespace gufo::models::inkling2
