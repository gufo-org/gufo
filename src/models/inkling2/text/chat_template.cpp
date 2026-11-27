#include "src/models/inkling2/text/chat_template.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace inkling2::text {

namespace {

constexpr const char* kMsgUser = "<|message_user|>";
constexpr const char* kMsgModel = "<|message_model|>";
constexpr const char* kMsgSystem = "<|message_system|>";
constexpr const char* kMsgTool = "<|message_tool|>";
constexpr const char* kText = "<|content_text|>";
constexpr const char* kThinking = "<|content_thinking|>";
constexpr const char* kImage = "<|content_image|>";
constexpr const char* kImagePlaceholder = "<|unused_200054|>";
constexpr const char* kInvokeJson = "<|content_invoke_tool_json|>";
constexpr const char* kXml = "<|content_xml|>";
constexpr const char* kEnd = "<|end_message|>";
constexpr const char* kEndSampling = "<|content_model_end_sampling|>";

class Builder {
public:
  explicit Builder(Rendered* r) : r_(r) {}
  void Control(const char* s) { r_->items.push_back({true, s}); }
  void Text(std::string s) { r_->items.push_back({false, std::move(s)}); }
  void Placeholder() {
    r_->image_items.push_back(r_->items.size());
    Control(kImagePlaceholder);
  }

private:
  Rendered* r_;
};

const char* RoleToken(const std::string& role) {
  if (role == "user")
    return kMsgUser;
  if (role == "assistant")
    return kMsgModel;
  if (role == "system")
    return kMsgSystem;
  if (role == "tool")
    return kMsgTool;
  return nullptr;
}

bool EffortValue(const RenderOptions& o, double* num, std::string* error) {
  if (o.effort_name) {
    std::string key = *o.effort_name;
    const auto is_ws = [](char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
             c == '\v';
    };
    while (!key.empty() && is_ws(key.front()))
      key.erase(key.begin());
    while (!key.empty() && is_ws(key.back()))
      key.pop_back();
    static const std::pair<const char*, double> kMap[] = {
        {"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7},
        {"high", 0.9}, {"xhigh", 0.99},  {"max", 0.99}};
    for (const auto& [k, v] : kMap) {
      if (key == k) {
        *num = v;
        return true;
      }
    }
    *error = "Unknown reasoning_effort: " + *o.effort_name;
    return false;
  }
  *num = o.effort_number ? *o.effort_number : 0.9;
  if (*num < 0.0 || *num > 0.99 || std::isnan(*num)) {
    *error = "reasoning_effort must be in [0.0, 0.99]";
    return false;
  }
  return true;
}

// The reference renderer (tml-renderers) quantizes the effort to integer
// milli-effort, then to hundredths, rounding halves to even both times, and
// writes the shortest decimal: 0.125 -> "0.12", 0.0015 -> "0". The named
// levels are unchanged.
std::string EffortText(double num) {
  const double milli = std::nearbyint(num * 1000.0);
  const double hundredths = std::nearbyint(milli / 1000.0 * 100.0) / 100.0;
  if (hundredths == 0.0)
    return "0";
  if (hundredths == 1.0)
    return "1";
  return json::PythonFloatRepr(hundredths);
}

void EmitEffort(Builder& b, double num) {
  b.Control(kMsgSystem);
  b.Control(kText);
  b.Text("Thinking effort level: " + EffortText(num));
  b.Control(kEnd);
}

const json::Value* Defined(const json::Value& obj, std::string_view key) {
  return obj.Find(key);
}

bool BuildToolSpecs(const std::vector<Tool>& tools, json::Value* specs,
                    std::string* error) {
  *specs = json::Value::Array();
  for (const auto& t : tools) {
    json::Value tool;
    std::string perr;
    if (!json::Parse(t.definition_json, &tool, &perr) || !tool.IsObject()) {
      *error = "tool definition is not a JSON object: " + perr;
      return false;
    }
    const json::Value* fnp = Defined(tool, "function");
    const json::Value& fn = fnp != nullptr ? *fnp : tool;
    if (!fn.IsObject()) {
      *error = "tool function must be an object";
      return false;
    }
    const json::Value* desc = Defined(fn, "description");
    const json::Value* name = Defined(fn, "name");
    const json::Value* params = Defined(fn, "parameters");
    const json::Value* type = Defined(tool, "type");
    if (name == nullptr) {
      *error = "tool function has no name";
      return false;
    }
    json::Value spec = json::Value::Object();
    spec.Set("description", desc != nullptr && desc->Truthy()
                                ? *desc
                                : json::Value::String(""));
    spec.Set("name", *name);
    spec.Set("parameters", params != nullptr && params->Truthy()
                               ? *params
                               : json::Value::Object());
    spec.Set("type", type != nullptr && type->Truthy()
                         ? *type
                         : json::Value::String("function"));
    specs->Push(std::move(spec));
  }
  return true;
}

}  // namespace

std::string Rendered::Text() const {
  std::string s;
  for (const auto& it : items)
    s += it.text;
  return s;
}

std::optional<std::string> EffortNameFor(std::optional<bool> enabled,
                                         std::optional<ReasoningLevel> level) {
  if (enabled && !*enabled)
    return "none";
  if (!level)
    return std::nullopt;
  switch (*level) {
    case ReasoningLevel::kMinimal:
      return "minimal";
    case ReasoningLevel::kLow:
      return "low";
    case ReasoningLevel::kMedium:
      return "medium";
    case ReasoningLevel::kHigh:
      return "high";
    case ReasoningLevel::kXHigh:
      return "xhigh";
    case ReasoningLevel::kMax:
      return "max";
  }
  return std::nullopt;
}

bool RenderChat(const std::vector<Message>& messages,
                const std::vector<Tool>& tools, const RenderOptions& options,
                Rendered* out, std::string* error) {
  std::string scratch;
  if (error == nullptr)
    error = &scratch;
  out->items.clear();
  out->image_items.clear();
  Builder b(out);

  double effort = 0;
  if (!EffortValue(options, &effort, error))
    return false;

  if (!tools.empty()) {
    json::Value specs;
    if (!BuildToolSpecs(tools, &specs, error))
      return false;
    b.Control(kMsgSystem);
    b.Text("tool_declare");
    b.Control(kXml);
    json::DumpOptions d;
    d.sort_keys = true;
    d.item_separator = ",";
    d.key_separator = ":";
    d.floats = json::FloatStyle::kTml;
    b.Text(json::Dump(specs, d));
    b.Control(kEnd);
  }

  std::size_t last_user = messages.size();
  for (std::size_t i = 0; i < messages.size(); ++i)
    if (messages[i].role == "user")
      last_user = i;
  const bool drop_old_thinking =
      options.preserve_thinking && !*options.preserve_thinking;

  bool effort_emitted = false;
  json::DumpOptions compact;
  compact.item_separator = ",";
  compact.key_separator = ":";
  compact.floats = json::FloatStyle::kTmlUnsigned;

  std::size_t image_index = 0;
  for (std::size_t mi = 0; mi < messages.size(); ++mi) {
    const Message& m = messages[mi];
    const char* role_token = RoleToken(m.role);
    if (role_token == nullptr) {
      *error = "Unknown message role: " + m.role;
      return false;
    }
    if (!effort_emitted && m.role != "system") {
      EmitEffort(b, effort);
      effort_emitted = true;
    }

    // header = role token + optional name (tool: name or the name of the call
    // it answers)
    std::string header_name;
    if (m.role == "tool") {
      if (!m.name.empty()) {
        header_name = m.name;
      } else if (!m.tool_call_id.empty()) {
        for (const auto& prev : messages)
          if (prev.role == "assistant")
            for (const auto& tc : prev.tool_calls)
              if (tc.id == m.tool_call_id)
                header_name += tc.name;  // the template appends every match
      }
    } else if (m.role == "assistant" && !m.name.empty()) {
      header_name = m.name;
    }
    const auto header = [&] {
      b.Control(role_token);
      if (!header_name.empty())
        b.Text(header_name);
    };

    const bool assistant = m.role == "assistant";
    const bool has_calls = assistant && !m.tool_calls.empty();
    const bool drop_thoughts = drop_old_thinking && mi < last_user;
    const bool show_thought =
        assistant && !m.reasoning_content.empty() && !drop_thoughts;
    const bool has_thought =
        show_thought;  // a dropped thought behaves as if absent
    // Thoughts among the parts, in their place (a turn written as blocks).
    const bool part_thought =
        assistant && !drop_thoughts &&
        std::any_of(m.parts.begin(), m.parts.end(), [](const ContentPart& p) {
          return p.kind == ContentPart::Kind::kThinking;
        });

    if (!show_thought && !part_thought && assistant &&
        options.empty_thought_for_assistants) {
      header();
      b.Control(kThinking);
      b.Control(kEnd);
    }
    if (show_thought) {
      header();
      b.Control(kThinking);
      b.Text(m.reasoning_content);
      b.Control(kEnd);
    }

    if (m.content_kind == Message::Content::kString) {
      if (!m.content.empty() || !(assistant && (has_calls || has_thought))) {
        header();
        b.Control(kText);
        b.Text(m.content);
        b.Control(kEnd);
      }
    } else if (m.content_kind == Message::Content::kParts) {
      for (const auto& part : m.parts) {
        switch (part.kind) {
          case ContentPart::Kind::kText:
            header();
            b.Control(kText);
            b.Text(part.text);
            b.Control(kEnd);
            break;
          case ContentPart::Kind::kThinking:
            if (!assistant) {
              *error =
                  "Thinking content is only supported for assistant "
                  "messages";
              return false;
            }
            if (drop_thoughts)
              break;
            header();
            b.Control(kThinking);
            b.Text(part.text);
            b.Control(kEnd);
            break;
          case ContentPart::Kind::kImage: {
            if (m.role != "user" && m.role != "tool") {
              *error =
                  "Image content is only supported for user and tool messages";
              return false;
            }
            header();
            b.Control(kImage);
            const std::size_t want = image_index < options.image_counts.size()
                                         ? options.image_counts[image_index]
                                         : options.image_placeholders;
            ++image_index;
            for (std::size_t k = 0; k < std::max<std::size_t>(1, want); ++k)
              b.Placeholder();
            b.Control(kEnd);
            break;
          }
          case ContentPart::Kind::kAudio:
            *error = "Inkling 2 does not accept audio input";
            return false;
        }
      }
    }

    if (has_calls) {
      for (const auto& tc : m.tool_calls) {
        if (!m.name.empty() && m.name != tc.name) {
          *error = "assistant name " + m.name +
                   " does not match its tool call " + tc.name;
          return false;
        }
        json::Value args = json::Value::Object();
        if (!tc.arguments_json.empty()) {
          std::string perr;
          json::Value parsed;
          if (!json::Parse(tc.arguments_json, &parsed, &perr)) {
            *error =
                "tool call arguments must be a parsed object, not a JSON "
                "string; canonicalize upstream";
            return false;
          }
          if (parsed.Truthy()) {
            if (!parsed.IsObject()) {
              *error = "tool call arguments must be an object";
              return false;
            }
            args = std::move(parsed);
          }
        }
        b.Control(kMsgModel);
        b.Text(tc.name);
        b.Control(kInvokeJson);
        std::string body =
            "{\"name\":" + json::QuoteString(tc.name) + ",\"args\":{";
        std::vector<const json::Value::Member*> ms;
        for (const auto& mem : args.members())
          ms.push_back(&mem);
        std::stable_sort(
            ms.begin(), ms.end(),
            [](const auto* x, const auto* y) { return x->first < y->first; });
        for (std::size_t k = 0; k < ms.size(); ++k) {
          body += json::QuoteString(ms[k]->first) + ":" +
                  json::Dump(ms[k]->second, compact);
          if (k + 1 < ms.size())
            body += ",";
        }
        body += "}}";
        b.Text(std::move(body));
        b.Control(kEnd);
      }
    }

    if (assistant)
      b.Control(kEndSampling);
  }

  if (!effort_emitted)
    EmitEffort(b, effort);

  switch (options.generation_prefix) {
    case RenderOptions::Prefix::kNone:
      break;
    case RenderOptions::Prefix::kHeader:
      b.Control(kMsgModel);
      break;
    case RenderOptions::Prefix::kHeaderText:
      b.Control(kMsgModel);
      b.Control(kText);
      break;
    case RenderOptions::Prefix::kEmptyThought:
      b.Control(kMsgModel);
      b.Control(kThinking);
      b.Control(kEnd);
      break;
  }
  return true;
}

bool RenderAndTokenize(const Tokenizer& tokenizer,
                       const std::vector<Message>& messages,
                       const std::vector<Tool>& tools,
                       const RenderOptions& options, PromptTokens* out,
                       std::string* error) {
  Rendered r;
  if (!RenderChat(messages, tools, options, &r, error))
    return false;
  out->ids.clear();
  out->image_positions.clear();
  std::size_t next_image = 0;
  for (std::size_t i = 0; i < r.items.size(); ++i) {
    const auto& it = r.items[i];
    if (it.control) {
      const auto id = tokenizer.FindSpecial(it.text);
      if (!id) {
        if (error != nullptr)
          *error = "vocabulary has no control token " + it.text;
        return false;
      }
      if (next_image < r.image_items.size() && r.image_items[next_image] == i) {
        out->image_positions.push_back(out->ids.size());
        ++next_image;
      }
      out->ids.push_back(*id);
    } else {
      tokenizer.EncodeOrdinary(it.text, &out->ids);
    }
  }
  return true;
}

}  // namespace inkling2::text
