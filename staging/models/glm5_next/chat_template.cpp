#include "staging/models/glm5_next/chat_template.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <unicode/uchar.h>

#include "src/core/json.hpp"

namespace gufo::models::glm5_next {
namespace {

[[nodiscard]] std::string RoleToken(std::string_view name) {
  return std::string(kCtrlOpen) + std::string(name) + std::string(kCtrlClose);
}

void ReplaceAll(std::string* text, std::string_view from,
                std::string_view to) {
  if (from.empty()) {
    return;
  }
  std::size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

[[nodiscard]] bool IsPythonStripSpace(char32_t cp) noexcept {
  if (cp >= 0x09 && cp <= 0x0D) {
    return true;
  }
  if (cp < 0x80) {
    return cp == ' ';
  }
  return u_isUWhiteSpace(static_cast<UChar32>(cp)) != 0;
}

/// The reference template renders assistant text through `content.strip()`,
/// Python whitespace semantics (Unicode `str.isspace`).
void StripPython(std::string* text) {
  std::size_t begin = 0;
  while (begin < text->size()) {
    const auto first = static_cast<unsigned char>((*text)[begin]);
    std::size_t len = 1;
    char32_t cp = first;
    if ((first & 0xE0U) == 0xC0U && begin + 1 < text->size()) {
      cp = ((first & 0x1FU) << 6) |
           (static_cast<unsigned char>((*text)[begin + 1]) & 0x3FU);
      len = 2;
    } else if ((first & 0xF0U) == 0xE0U && begin + 2 < text->size()) {
      cp = ((first & 0x0FU) << 12) |
           ((static_cast<unsigned char>((*text)[begin + 1]) & 0x3FU) << 6) |
           (static_cast<unsigned char>((*text)[begin + 2]) & 0x3FU);
      len = 3;
    } else if ((first & 0xF8U) == 0xF0U && begin + 3 < text->size()) {
      cp = ((first & 0x07U) << 18) |
           ((static_cast<unsigned char>((*text)[begin + 1]) & 0x3FU) << 12) |
           ((static_cast<unsigned char>((*text)[begin + 2]) & 0x3FU) << 6) |
           (static_cast<unsigned char>((*text)[begin + 3]) & 0x3FU);
      len = 4;
    }
    if (!IsPythonStripSpace(cp)) {
      break;
    }
    begin += len;
  }
  std::size_t end = text->size();
  while (end > begin) {
    // Scan backward to the lead byte of the final codepoint.
    std::size_t start = end - 1;
    while (start > begin &&
           (static_cast<unsigned char>((*text)[start]) & 0xC0U) == 0x80U) {
      --start;
    }
    const auto first = static_cast<unsigned char>((*text)[start]);
    std::size_t len = 1;
    char32_t cp = first;
    if ((first & 0xE0U) == 0xC0U && start + 1 < end) {
      cp = ((first & 0x1FU) << 6) |
           (static_cast<unsigned char>((*text)[start + 1]) & 0x3FU);
      len = 2;
    } else if ((first & 0xF0U) == 0xE0U && start + 2 < end) {
      cp = ((first & 0x0FU) << 12) |
           ((static_cast<unsigned char>((*text)[start + 1]) & 0x3FU) << 6) |
           (static_cast<unsigned char>((*text)[start + 2]) & 0x3FU);
      len = 3;
    } else if ((first & 0xF8U) == 0xF0U && start + 3 < end) {
      cp = ((first & 0x07U) << 18) |
           ((static_cast<unsigned char>((*text)[start + 1]) & 0x3FU) << 12) |
           ((static_cast<unsigned char>((*text)[start + 2]) & 0x3FU) << 6) |
           (static_cast<unsigned char>((*text)[start + 3]) & 0x3FU);
      len = 4;
    }
    if (start + len != end || !IsPythonStripSpace(cp)) {
      break;
    }
    end = start;
  }
  if (begin != 0 || end != text->size()) {
    text->erase(0, begin);
    text->resize(end - begin);
  }
}

void AppendToolCall(std::string* out,
                    const tokenization::ChatMessage::ToolCall& call) {
  out->append(kToolCallOpenTag);
  out->append(call.name);
  // The reference per-argument mini-loop opens after a newline text node
  // inside the call block, present even for argument-less calls.
  out->push_back('\n');
  for (const auto& argument : call.arguments) {
    out->append(kArgKeyOpenTag);
    out->append(argument.name);
    out->append(kArgKeyCloseTag);
    out->append(kArgValueOpenTag);
    // Strings pass through; non-strings arrive as their JSON text already
    // normalized by the server, so the bytes go verbatim (reference template
    // re-serializes, which is identical for canonical JSON input).
    out->append(argument.value);
    out->append(kArgValueCloseTag);
  }
  out->append(kToolCallCloseTag);
}

/// Python truthiness for the template's `defer_loading` guard: absent, null,
/// false, 0 and "" keep the tool; anything else defers it.
[[nodiscard]] bool Deferred(const json::Value* defer) {
  if (defer == nullptr || defer->is_null()) {
    return false;
  }
  if (defer->is_bool()) {
    return defer->as_bool();
  }
  if (defer->is_number()) {
    return defer->as_double() != 0.0;
  }
  if (defer->is_string()) {
    return !defer->str().empty();
  }
  return true;
}

/// The reference `tool_to_json`: the tool object (unwrapped from a
/// `function` member) minus `defer_loading` and `strict`, serialized with
/// Jinja `tojson` (Python `json.dumps(ensure_ascii=False)`) in member order.
[[nodiscard]] json::Value ToolDefinitionJson(
    const tokenization::ChatTool& tool) {
  json::Value definition;
  if (!tool.definition_json.empty()) {
    definition = json::parse(tool.definition_json);
  } else {
    definition = json::Value::object();
    definition.append_member("name", json::Value(tool.name));
    if (!tool.description.empty()) {
      definition.append_member("description", json::Value(tool.description));
    }
    definition.append_member("parameters", json::parse(tool.parameters_json));
  }
  if (const auto* function = definition.find("function")) {
    definition = *function;
  }
  return definition;
}

/// The reference tools preamble: a dedicated system block listing each tool
/// schema between `<tools>` tags, emitted once before the message loop.
void AppendToolsPreamble(
    std::string* prompt, std::span<const tokenization::ChatTool> tools) {
  prompt->append(RoleToken("system"));
  prompt->append(
      "\n# Tools\n\nYou may call one or more functions to assist with the "
      "user query.\n\nYou are provided with function signatures within "
      "<tools></tools> XML tags:\n<tools>\n");
  for (const auto& tool : tools) {
    // Each iteration contributes its own leading newline; a deferred tool
    // contributes only that newline.
    prompt->push_back('\n');
    const auto raw = ToolDefinitionJson(tool);
    if (Deferred(raw.find("defer_loading"))) {
      continue;
    }
    auto emitted = json::Value::object();
    for (const auto& member : raw.members()) {
      if (member.first == "defer_loading" || member.first == "strict") {
        continue;
      }
      emitted.append_member(member.first, member.second);
    }
    prompt->append(emitted.tojson());
    prompt->append("\n\n");
  }
  prompt->append(
      "\n</tools>\n\nFor each function call, output the function name and "
      "arguments within the following XML format:\n<tool_call>"
      "{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>"
      "{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>"
      "{arg-value-2}</arg_value>...</tool_call>");
}

}  // namespace

void HideEosLiterals(std::string* text) {
  ReplaceAll(text, kGlmEosSpelling, kEosPlaceholder);
}

std::optional<std::string> RenderChat(
    std::span<const tokenization::ChatMessage> messages,
    bool add_generation_prompt,
    std::span<const tokenization::ChatTool> tools,
    const ChatRenderOptions& options) {
  std::string prompt;
  prompt.append("\x5B" "gMASK" "\x5D");
  prompt.append("\x3C" "sop" "\x3E");

  // Reference-template header: the effort line is always present because the
  // evaluated effort can only be low/high/max, never None.
  std::string label = "Max";
  if (options.reasoning_effort.has_value()) {
    if (*options.reasoning_effort == "low") {
      label = "Low";
    } else if (*options.reasoning_effort == "high") {
      label = "High";
    }
  }
  prompt.append(RoleToken("system"));
  prompt.append("Reasoning Effort: ");
  prompt.append(label);

  if (!tools.empty()) {
    AppendToolsPreamble(&prompt, tools);
  }

  std::size_t last_user_index = 0;
  bool have_user = false;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (messages[i].role == tokenization::ChatRole::kUser) {
      last_user_index = i;
      have_user = true;
    }
  }

  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& message = messages[i];
    switch (message.role) {
      case tokenization::ChatRole::kDeveloper:
        // The reference template has no developer branch: skipped.
        break;
      case tokenization::ChatRole::kSystem: {
        std::string text = message.content;
        HideEosLiterals(&text);
        prompt.append(RoleToken("system"));
        prompt.append(text);
        break;
      }
      case tokenization::ChatRole::kUser: {
        std::string text = message.content;
        HideEosLiterals(&text);
        prompt.append(RoleToken("user"));
        // Attached images frame the text with one vision triple per image,
        // matching the reference template's image content parts.
        for (std::size_t image = 0; image < message.images.size(); ++image) {
          prompt.append(kBeginOfImageTag);
          prompt.append(kImageTag);
          prompt.append(kEndOfImageTag);
        }
        prompt.append(text);
        break;
      }
      case tokenization::ChatRole::kAssistant: {
        prompt.append(RoleToken("assistant"));
        // reasoning_content: explicit first, else split from a literal
        // close-think boundary like the reference template. Defined state
        // is scoped per message here; the reference persists the Jinja
        // frame variable across turns (a template bug we do not replicate).
        std::string text = message.content;
        HideEosLiterals(&text);
        bool reasoning_defined = !message.thought.empty();
        std::string reasoning;
        if (reasoning_defined) {
          reasoning = message.thought;
          HideEosLiterals(&reasoning);
        } else if (const std::size_t close = text.find(kThinkCloseTag);
                   close != std::string::npos) {
          const std::string_view head(text.data(), close);
          std::size_t start = 0;
          if (const std::size_t open = head.rfind(kThinkOpenTag);
              open != std::string_view::npos) {
            start = open + kThinkOpenTag.size();
          }
          reasoning.assign(head.substr(start));
          text.erase(0, close + kThinkCloseTag.size());
          reasoning_defined = true;
        }
        // The reference guards with loop.index0 > last_user_index, whose
        // sentinel is -1: without any user turn no assistant is cleared.
        const bool index_after_user =
            !have_user || i > last_user_index;
        const bool keep = reasoning_defined &&
                          (!options.clear_thinking || index_after_user);
        prompt.append(kThinkOpenTag);
        if (keep) {
          prompt.append(reasoning);
        }
        prompt.append(kThinkCloseTag);
        StripPython(&text);
        if (!text.empty()) {
          prompt.append(text);
        }
        for (const auto& call : message.tool_calls) {
          // Reference-template artifacts of the non-dash-wrapped if/call
          // blocks.
          if (&call == &message.tool_calls.front()) {
            prompt.push_back('\n');
          }
          AppendToolCall(&prompt, call);
        }
        if (!message.tool_calls.empty()) {
          // The call-list closing tag is not dash-wrapped either, so one
          // newline trails the whole section.
          prompt.push_back('\n');
        }
        break;
      }
      case tokenization::ChatRole::kTool: {
        if (i > 0 && messages[i - 1].role == tokenization::ChatRole::kTool) {
          break;  // The run opened before this message.
        }
        std::size_t block_end = i;
        while (block_end + 1 < messages.size() &&
               messages[block_end + 1].role ==
                   tokenization::ChatRole::kTool) {
          ++block_end;
        }
        prompt.append(RoleToken("observation"));

        const tokenization::ChatMessage* prev =
            i > 0 ? &messages[i - 1] : nullptr;
        bool can_sort =
            prev != nullptr && !prev->tool_calls.empty() &&
            !prev->tool_calls[0].id.empty();
        if (can_sort) {
          // Every call id must be unique and matched by exactly one tool
          // answer with that id in the run; otherwise preserve order.
          for (const auto& call : prev->tool_calls) {
            if (call.id.empty()) {
              can_sort = false;
              break;
            }
            std::size_t matched = 0;
            for (std::size_t k = i; k <= block_end; ++k) {
              if (messages[k].tool_call_id == call.id) {
                ++matched;
              }
            }
            if (matched != 1) {
              can_sort = false;
              break;
            }
          }
          if (can_sort) {
            // The reference rejects runs whose call list carries duplicate
            // ids; per-id answer matching alone would accept them.
            for (std::size_t a = 0; a < prev->tool_calls.size() && can_sort;
                 ++a) {
              for (std::size_t b = a + 1; b < prev->tool_calls.size(); ++b) {
                if (prev->tool_calls[a].id == prev->tool_calls[b].id) {
                  can_sort = false;
                  break;
                }
              }
            }
          }
        }
        const auto emit_one = [&](const tokenization::ChatMessage& m) {
          std::string text = m.content;
          HideEosLiterals(&text);
          // The reference observation opener and every response tag are
          // emitted inline; each emitted response carries no extra newline.
          prompt.append(kToolResponseOpenTag);
          prompt.append(text);
          prompt.append(kToolResponseCloseTag);
        };
        if (can_sort) {
          for (const auto& call : prev->tool_calls) {
            for (std::size_t k = i; k <= block_end; ++k) {
              if (messages[k].tool_call_id == call.id) {
                emit_one(messages[k]);
                break;
              }
            }
          }
        } else {
          for (std::size_t k = i; k <= block_end; ++k) {
            emit_one(messages[k]);
          }
        }
        break;
      }
    }
  }

  if (add_generation_prompt) {
    prompt.append(RoleToken("assistant"));
    prompt.append(kThinkOpenTag);
  }
  return prompt;
}

}  // namespace gufo::models::glm5_next
