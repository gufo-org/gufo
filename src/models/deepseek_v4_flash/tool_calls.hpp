#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_TOOL_CALLS_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_TOOL_CALLS_HPP_

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"

namespace gufo::models::deepseek_v4_flash {

/// The literals that frame a DeepSeek tool invocation. AppendToolCalls renders
/// with the canonical member of each spelling family, and
/// ScanDeepSeekToolCalls admits the whole family the checkpoint writes, so the
/// wire format cannot change in one direction only (#290).
struct DeepSeekToolSyntax {
  static constexpr std::string_view kCallsOpen = "<｜DSML｜tool_calls>";
  /// The template's separator before the block, as in llama.cpp's
  /// TC_SEPARATOR + FC_START.
  static constexpr std::string_view kCallsOpenWithSeparator =
      "\n\n<｜DSML｜tool_calls>";
  /// The canonical block opener followed by the renderer's first
  /// invocation header, as admitted by the native constrained grammar.
  static constexpr std::string_view kNativeCallOpen =
      "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"";
  static constexpr std::string_view kNativeCallOpenWithSeparator =
      "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"";
  static constexpr std::string_view kCallsClose = "</｜DSML｜tool_calls>";
  static constexpr std::array<std::string_view, 4> kInvokeOpens{
      "<｜DSML｜invoke", "<DSML｜invoke", "<｜DS｜invoke", "<DS｜invoke"};
  static constexpr std::array<std::string_view, 4> kInvokeCloses{
      "</｜DSML｜invoke>", "</DSML｜invoke>", "</｜DS｜invoke>",
      "</DS｜invoke>"};
  static constexpr std::array<std::string_view, 4> kParameterOpens{
      "<｜DSML｜parameter", "<DSML｜parameter", "<｜DS｜parameter",
      "<DS｜parameter"};
  static constexpr std::array<std::string_view, 4> kParameterCloses{
      "</｜DSML｜parameter>", "</DSML｜parameter>", "</｜DS｜parameter>",
      "</DS｜parameter>"};
  /// The canonical attribute framing the renderer writes.
  static constexpr std::string_view kNameAttribute = "name=\"";
  static constexpr std::string_view kStringAttribute = "string=\"";
};

/// One invocation exactly as the model wrote it, before the API surface
/// applies a declared schema.
using ScannedToolCall = ext::ScannedToolCall;
using ScannedToolArgument = ext::ScannedToolArgument;

/// Scans DeepSeek invocations and their enclosing call blocks. `markers` are
/// the dialect openers the request grammar admits; `spans` receives the byte
/// ranges of closed outer blocks. Schema interpretation and OpenAI shaping
/// stay with the API surface.
void ScanDeepSeekToolCalls(
    std::string_view text, std::span<const std::string_view> markers,
    std::vector<ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr,
    bool partial = false);

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_TOOL_CALLS_HPP_
