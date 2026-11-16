#include "src/ext/text.hpp"

#include <algorithm>
#include <charconv>
#include <deque>
#include <stdexcept>
#include <utility>

namespace gufo::ext {

std::optional<TextRunnerRamCacheOptions> ParseRamCacheOption(
    std::string_view text, std::string* error) {
  if (text == "off")
    return TextRunnerRamCacheOptions{.enabled = false};
  std::size_t bytes = 0;
  const auto end = text.data() + text.size();
  const auto parsed = std::from_chars(text.data(), end, bytes);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != end) {
    if (error)
      *error =
          "--cache-ram-bytes must be off or a byte count (0 = auto), "
          "not '" +
          std::string(text) + "'";
    return std::nullopt;
  }
  return TextRunnerRamCacheOptions{.capacity_bytes = bytes};
}

void TextModelRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  for (const auto& advance : advances) {
    try {
      Advance(advance.state.get(), advance.token);
    } catch (...) {
      if (advance.failure != nullptr) {
        *advance.failure = std::current_exception();
      } else {
        throw;
      }
    }
  }
}

TextDecodeStep TextModelRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  TextDecodeStep step;
  if (max_tokens == 0) {
    return step;
  }
  auto selection = SelectNext(state, sampler);
  step.stop = selection.stop;
  if (selection.stop) {
    step.terminal_logprobs = std::move(selection.logprobs);
    return step;
  }
  const auto token = selection.token;
  step.selections.push_back(std::move(selection));
  Advance(state, token);
  return step;
}

std::vector<TextDecodeStep> TextModelRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  std::vector<TextDecodeStep> steps;
  steps.reserve(decodes.size());
  for (const auto& decode : decodes) {
    try {
      steps.push_back(DecodeStep(decode.state.get(), decode.max_tokens,
                                 decode.sampler.get()));
    } catch (...) {
      TextDecodeStep failed;
      failed.failure = std::current_exception();
      steps.push_back(std::move(failed));
    }
  }
  return steps;
}

std::shared_ptr<const sampling::TokenConstraint>
TextModelRunner::BindConstraint(
    std::shared_ptr<const sampling::JsonConstraint> grammar) const {
  const std::lock_guard lock(constraint_mutex_);
  for (const auto& binding : constraints_)
    if (binding->grammar == grammar)
      return binding;
  if (!constraint_vocabulary_)
    constraint_vocabulary_ = BuildConstraintVocabulary();
  if (!constraint_vocabulary_)
    throw std::invalid_argument("model does not support structured output");
  auto binding = std::make_shared<sampling::TokenConstraint>();
  binding->grammar = std::move(grammar);
  binding->vocabulary = constraint_vocabulary_;
  if (constraints_.size() >= 4)
    constraints_.erase(constraints_.begin());
  constraints_.push_back(binding);
  return binding;
}

std::size_t TextModelRunner::PrepareConstraintVocabulary() const {
  const std::lock_guard lock(constraint_mutex_);
  if (!constraint_vocabulary_)
    constraint_vocabulary_ = BuildConstraintVocabulary();
  return constraint_vocabulary_ ? constraint_vocabulary_->size() : 0;
}

ReasoningMarkup TextModelRunner::ReasoningDelimiters() const {
  const auto tags = ReasoningBudgetTags();
  const auto close = tags.ends.empty() ? tags.forced_end : tags.ends.front();
  // A runner that declares no budget tags keeps the host's classic framing.
  if (close.empty())
    return {"<think>", "</think>"};
  return {std::string(tags.start), std::string(close)};
}

std::shared_ptr<const sampling::ReasoningBudget>
TextModelRunner::BuildReasoningBudget(std::size_t tokens,
                                      std::span<const TextRunnerToken> prompt,
                                      ext::InitialOutputState initial) const {
  const auto tags = ReasoningBudgetTags();
  const auto start = TokenizeControl(tags.start);
  auto forced = TokenizeControl(tags.forced_end);
  if (start.size() != 1 || forced.empty())
    return nullptr;
  auto budget = std::make_shared<sampling::ReasoningBudget>();
  budget->tokens = tokens;
  budget->forced.assign(forced.begin(), forced.end());
  for (const auto end : tags.ends) {
    const auto token = TokenizeControl(end);
    if (token.size() != 1)
      return nullptr;
    budget->ends.push_back(token.front());
  }
  // An automatic output opens its own block; an opener in the prompt belongs
  // to an earlier turn.
  if (initial == ext::InitialOutputState::kAuto) {
    budget->opener = start.front();
  } else {
    const auto opened =
        std::ranges::find(prompt.rbegin(), prompt.rend(), start.front());
    if (opened != prompt.rend())
      budget->prompt.assign(opened.base(), prompt.end());
  }
  const std::lock_guard lock(constraint_mutex_);
  if (!partial_utf8_ && VocabularySize() != 0) {
    auto partial = std::make_shared<std::vector<bool>>(VocabularySize());
    auto blank = std::make_shared<std::vector<bool>>(VocabularySize());
    for (TextRunnerToken token = 0; token < partial->size(); ++token) {
      const auto text = Decode(std::span(&token, 1));
      bool complete = text.empty();
      for (std::size_t i = 1; i <= std::min<std::size_t>(4, text.size()); ++i) {
        const auto byte = static_cast<unsigned char>(text[text.size() - i]);
        if ((byte & 0xC0U) != 0x80U) {
          complete = i >= (byte >= 0xF0U   ? 4U
                           : byte >= 0xE0U ? 3U
                           : byte >= 0xC0U ? 2U
                                           : 1U);
          break;
        }
      }
      (*partial)[token] = !complete;
      (*blank)[token] = !text.empty() &&
                        text.find_first_not_of(" \t\r\n") == std::string::npos;
    }
    partial_utf8_ = std::move(partial);
    blank_tokens_ = std::move(blank);
  }
  budget->partial_utf8 = partial_utf8_;
  budget->blank = blank_tokens_;
  return budget;
}

// ---------------------------------------------------------------------------
// Provider & Tool Registries
// ---------------------------------------------------------------------------

static std::vector<TextProvider> g_text_providers;
// A deque keeps registered dialects at stable addresses.
static std::deque<ToolDialect> g_tool_dialects;

void RegisterTextProvider(TextProvider provider) {
  for (auto& existing : g_text_providers) {
    if (existing.id == provider.id) {
      existing = std::move(provider);
      return;
    }
  }
  g_text_providers.push_back(std::move(provider));
}

const TextProvider* FindTextProvider(std::string_view id) {
  for (const auto& provider : g_text_providers) {
    if (provider.id == id) {
      return &provider;
    }
  }
  return nullptr;
}

const TextProvider* ProbeTextProvider(std::string_view model_root) {
  for (const auto& provider : g_text_providers) {
    if (provider.probe && provider.probe(model_root)) {
      return &provider;
    }
  }
  return nullptr;
}

const TextProvider* ProbeTextProvider(const core::GgufReader& reader) {
  const bool unsized = reader.FirstUnsizedTensor() != nullptr;
  for (const auto& provider : g_text_providers) {
    if (unsized && !provider.accepts_unsized_tensors)
      continue;
    if (provider.probe_gguf && provider.probe_gguf(reader)) {
      return &provider;
    }
  }
  return nullptr;
}

std::string_view BuiltinToolDialectId(
    sampling::JsonConstraint::ToolFormat format) {
  using Format = sampling::JsonConstraint::ToolFormat;
  switch (format) {
    case Format::kQwen:
      return "qwen";
    case Format::kDeepSeek:
      return "deepseek";
    case Format::kJson:
      break;
  }
  return "json";
}

std::string TextModelRunner::ToolDialectId() const {
  return std::string(BuiltinToolDialectId(ToolFormat()));
}

void RegisterToolDialect(ToolDialect dialect) {
  using Format = sampling::JsonConstraint::ToolFormat;
  if (dialect.id.empty())
    dialect.id = std::string(BuiltinToolDialectId(dialect.format));
  const auto fail = [&](std::string_view reason) {
    throw std::invalid_argument("tool dialect '" + dialect.id + "' " +
                                std::string(reason));
  };
  if (dialect.id == BuiltinToolDialectId(Format::kJson) ||
      dialect.id == BuiltinToolDialectId(Format::kQwen))
    fail("is built into the host");
  if (dialect.markers.empty() || dialect.scan_tool_calls == nullptr)
    fail("needs markers and scan_tool_calls");
  if (std::ranges::any_of(dialect.markers, [](std::string_view marker) {
        return marker.find('<') == std::string_view::npos;
      }))
    fail("has a marker without '<'");
  if (dialect.syntax
          ? std::ranges::find(dialect.markers,
                              std::string_view(dialect.syntax->open)) ==
                dialect.markers.end()
          : dialect.format == Format::kJson)
    fail("needs a syntax opened by one of its markers, or a native format");
  for (auto& existing : g_tool_dialects) {
    if (existing.id == dialect.id) {
      existing = std::move(dialect);
      return;
    }
  }
  g_tool_dialects.push_back(std::move(dialect));
}

const ToolDialect* FindToolDialect(std::string_view id) {
  for (const auto& dialect : g_tool_dialects) {
    if (dialect.id == id) {
      return &dialect;
    }
  }
  return nullptr;
}

const ToolDialect* FindToolDialect(
    sampling::JsonConstraint::ToolFormat format) {
  return FindToolDialect(BuiltinToolDialectId(format));
}

std::vector<const ToolDialect*> AllToolDialects() {
  std::vector<const ToolDialect*> result;
  result.reserve(g_tool_dialects.size());
  for (const auto& dialect : g_tool_dialects) {
    result.push_back(&dialect);
  }
  return result;
}

}  // namespace gufo::ext
