// Glue between the Gufo extension API (gufo/ext/text.hpp) and the text
// front-end, which uses only Gufo's core (GGUF reader, JSON parser).
// Header-only: the runner includes it. Compile-checked by
// tests/text/gufo_adapter_check.cpp.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"
#include "src/models/inkling2/text/chat_template.hpp"
#include "src/models/inkling2/text/json.hpp"
#include "src/models/inkling2/text/tokenizer.hpp"
#include "src/models/inkling2/text/tool_dialect.hpp"

namespace inkling2::text::gufo_adapter {

namespace g = ::gufo::ext;

inline std::string RoleName(g::ChatRole role) {
  switch (role) {
    case g::ChatRole::kSystem:
    case g::ChatRole::kDeveloper:
      return "system";
    case g::ChatRole::kUser:
      return "user";
    case g::ChatRole::kAssistant:
      return "assistant";
    case g::ChatRole::kTool:
      return "tool";
  }
  return "user";
}

/// A tool call's arguments as one JSON object text.
inline std::string ArgumentsJson(const g::ChatMessage::ToolCall& call) {
  json::Value obj = json::Value::Object();
  for (const auto& a : call.arguments) {
    if (a.is_string) {
      obj.Set(a.name, json::Value::String(a.value));
    } else {
      json::Value v;
      obj.Set(a.name, json::Parse(a.value, &v) ? std::move(v)
                                               : json::Value::String(a.value));
    }
  }
  return json::Dump(obj);
}

/// Whether `m.parts` cover its content and thought in order: ends grow, each
/// kind's last end is its string's size, and a non-empty string has parts.
inline bool PartsCover(const g::ChatMessage& m) {
  std::size_t text = 0, thought = 0;
  bool any_text = false, any_thought = false;
  for (const auto& part : m.parts) {
    auto& at = part.thought ? thought : text;
    const auto size = part.thought ? m.thought.size() : m.content.size();
    if (part.end < at || part.end > size)
      return false;
    at = part.end;
    (part.thought ? any_thought : any_text) = true;
  }
  return text == m.content.size() && thought == m.thought.size() &&
         (any_text || m.content.empty()) && (any_thought || m.thought.empty());
}

/// A message whose parts keep their order: thoughts and text in place, and
/// images at their content offsets.
inline void OrderedParts(const g::ChatMessage& m, Message* r) {
  r->content_kind = Message::Content::kParts;
  std::size_t text = 0, thought = 0, image = 0;
  const auto images_to = [&](std::size_t offset) {
    for (; image < m.images.size() && m.images[image].offset <= offset; ++image)
      r->parts.push_back({ContentPart::Kind::kImage, {}});
  };
  for (const auto& part : m.parts) {
    if (part.thought) {
      r->parts.push_back({ContentPart::Kind::kThinking,
                          m.thought.substr(thought, part.end - thought)});
      thought = part.end;
      continue;
    }
    images_to(text);
    r->parts.push_back(
        {ContentPart::Kind::kText, m.content.substr(text, part.end - text)});
    text = part.end;
  }
  images_to(m.content.size());
  if (!m.framing_suffix.empty()) {
    if (!r->parts.empty() && r->parts.back().kind == ContentPart::Kind::kText)
      r->parts.back().text += m.framing_suffix;
    else
      r->parts.push_back({ContentPart::Kind::kText, m.framing_suffix});
  }
}

inline std::vector<Message> ToMessages(const g::ChatRequest& request) {
  std::vector<Message> out;
  out.reserve(request.messages.size());
  for (const auto& m : request.messages) {
    Message r;
    r.role = RoleName(m.role);
    const std::string text = m.content + m.framing_suffix;
    const bool ordered = !m.parts.empty() && PartsCover(m);
    if (ordered) {
      OrderedParts(m, &r);
    } else if (m.images.empty()) {
      r.content_kind = Message::Content::kString;
      r.content = text;
    } else {
      // Images sit at byte offsets of the content: text before, the image, text
      // after.
      r.content_kind = Message::Content::kParts;
      std::size_t cursor = 0;
      for (const auto& img : m.images) {
        const std::size_t at = std::min(img.offset, m.content.size());
        if (at > cursor)
          r.parts.push_back(
              {ContentPart::Kind::kText, text.substr(cursor, at - cursor)});
        r.parts.push_back({ContentPart::Kind::kImage, {}});
        cursor = std::max(cursor, at);
      }
      if (cursor < text.size())
        r.parts.push_back({ContentPart::Kind::kText, text.substr(cursor)});
    }
    r.name = m.name;
    // Ordered parts carry the thoughts in their place.
    if (!ordered)
      r.reasoning_content = m.thought;
    r.tool_call_id = m.tool_call_id;
    for (const auto& c : m.tool_calls)
      r.tool_calls.push_back({c.id, c.name, ArgumentsJson(c)});
    out.push_back(std::move(r));
  }
  return out;
}

inline std::vector<Tool> ToTools(const g::ChatRequest& request) {
  std::vector<Tool> out;
  if (request.tool_choice == g::ChatRequest::ToolChoice::kNone)
    return out;
  for (const auto& t : request.tools) {
    if (!t.definition_json.empty()) {
      out.push_back({t.definition_json});
      continue;
    }
    json::Value fn = json::Value::Object();
    fn.Set("name", json::Value::String(t.name));
    fn.Set("description", json::Value::String(t.description));
    json::Value params;
    fn.Set("parameters", json::Parse(t.parameters_json, &params)
                             ? std::move(params)
                             : json::Value::Object());
    json::Value tool = json::Value::Object();
    tool.Set("type", json::Value::String("function"));
    tool.Set("function", std::move(fn));
    out.push_back({json::Dump(tool)});
  }
  return out;
}

inline std::optional<ReasoningLevel> LevelOf(
    const std::optional<::gufo::ReasoningEffort>& e) {
  if (!e)
    return std::nullopt;
  switch (*e) {
    case ::gufo::ReasoningEffort::kMinimal:
      return ReasoningLevel::kMinimal;
    case ::gufo::ReasoningEffort::kLow:
      return ReasoningLevel::kLow;
    case ::gufo::ReasoningEffort::kMedium:
      return ReasoningLevel::kMedium;
    case ::gufo::ReasoningEffort::kHigh:
      return ReasoningLevel::kHigh;
    case ::gufo::ReasoningEffort::kXHigh:
      return ReasoningLevel::kXHigh;
    case ::gufo::ReasoningEffort::kMax:
      return ReasoningLevel::kMax;
  }
  return std::nullopt;
}

/// Tools are in play and the request insists on a call.
inline bool CallRequired(const g::ChatRequest& request) {
  return !request.tools.empty() &&
         request.tool_choice == g::ChatRequest::ToolChoice::kRequired;
}

inline RenderOptions ToRenderOptions(const g::ChatRequest& request) {
  RenderOptions o;
  o.effort_name = EffortNameFor(request.reasoning.enabled,
                                LevelOf(request.reasoning.effort));
  o.preserve_thinking = request.reasoning.preserve_thinking;
  // Gufo's "reasoning off" (enabled=false, effort none) renders the
  // template's level 0, which does not stop the model from thinking (the model
  // card: the minimum "does not fully disable reasoning"; it opens a thinking
  // block at level 0 as at every other level). As in llama.cpp, the model
  // then decides: the prompt ends with the assistant header it writes first
  // anyway (log-probability ~0), so its first generated token is the thinking
  // opener, the answer's or a call's, and a reasoning budget starts counting
  // at the opener.
  o.generation_prefix = RenderOptions::Prefix::kHeader;
  return o;
}

/// Tokens at the end of the prompt that open the assistant's reply ("mutable
/// assistant framing" for the cache): the next turn re-renders them as part of
/// the assistant message, so a checkpoint is only reusable up to just before
/// them. The template itself has no generation prompt (zero).
inline std::size_t GenerationFramingTokens(const g::ChatRequest& request) {
  switch (ToRenderOptions(request).generation_prefix) {
    case RenderOptions::Prefix::kNone:
      return 0;
    case RenderOptions::Prefix::kHeader:
      return 1;
    case RenderOptions::Prefix::kHeaderText:
      return 2;
    case RenderOptions::Prefix::kEmptyThought:
      return 3;
  }
  return 0;
}

/// RenderAndTokenize for TextModelRunner. Throws std::invalid_argument with the
/// template's message when the template would raise.
inline std::optional<std::vector<g::TextRunnerToken>> RenderAndTokenize(
    const Tokenizer& tok, const g::ChatRequest& request,
    const std::vector<std::size_t>* image_counts = nullptr,
    PromptTokens* tokens_out = nullptr) {
  PromptTokens pt;
  std::string err;
  RenderOptions options = ToRenderOptions(request);
  if (image_counts != nullptr)
    options.image_counts = *image_counts;
  if (!text::RenderAndTokenize(tok, ToMessages(request), ToTools(request),
                               options, &pt, &err))
    throw std::invalid_argument(err);
  std::vector<g::TextRunnerToken> ids(pt.ids.begin(), pt.ids.end());
  if (tokens_out != nullptr)
    *tokens_out = std::move(pt);
  return ids;
}

// ---- reasoning and tool dialect ----

inline g::ReasoningTags BudgetTags() {
  return {dialect::kBudgetStart,
          dialect::kBudgetForcedEnd,
          {dialect::kBudgetForcedEnd}};
}
inline void ScanToolCalls(
    std::string_view text, std::span<const std::string_view>,
    std::vector<g::ScannedToolCall>* calls,
    std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  std::vector<dialect::Call> found;
  dialect::ScanToolCalls(text, partial, &found, spans);
  for (auto& c : found) {
    g::ScannedToolCall s;
    s.name = std::move(c.name);
    for (auto& a : c.arguments)
      s.arguments.push_back(
          {std::move(a.name), std::move(a.value), a.is_string});
    if (c.unfinished)
      s.unfinished =
          g::ScannedToolArgument{c.unfinished->name, c.unfinished->value, true};
    s.valid = c.valid;
    s.complete = c.complete;
    calls->push_back(std::move(s));
  }
}

}  // namespace inkling2::text::gufo_adapter
