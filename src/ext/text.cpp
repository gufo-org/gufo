#include "src/ext/text.hpp"

#include <algorithm>
#include <charconv>
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

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::Snapshot(
    const TextRunnerState&) const {
  throw std::logic_error("runner does not support snapshots");
}

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::SnapshotForPersistence(
    const TextRunnerState& state) const {
  return Snapshot(state);
}

std::size_t TextModelRunner::SnapshotPayloadBytes(
    const TextRunnerState&) const {
  return 0;
}

void TextModelRunner::RestoreOrFork(TextRunnerState&,
                                    const TextRunnerSnapshot&) const {
  throw std::logic_error("runner does not support snapshots");
}

std::size_t TextModelRunner::PersistentSnapshotPayloadBytes(
    const TextRunnerSnapshot&) const {
  return 0;
}

std::size_t TextModelRunner::SerializePersistentSnapshot(
    const TextRunnerSnapshot&, std::span<std::uint8_t>) const {
  return 0;
}

void TextModelRunner::RestorePersistentSnapshot(
    TextRunnerState&, std::span<const std::uint8_t>) const {
  throw std::logic_error("runner does not support persistent snapshots");
}

std::unique_ptr<ComponentCacheResources>
TextModelRunner::CreateComponentCacheResources(cache::ResourceLedger&,
                                               std::size_t) const {
  return nullptr;
}

std::unique_ptr<TextRunnerState> TextModelRunner::BindComponentState(
    cache::Adapter&, cache::Slot&) const {
  throw std::logic_error("runner does not support component cache");
}

void TextModelRunner::ReconcileComponentState(TextRunnerState&,
                                              std::size_t) const {}

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

std::shared_ptr<const sampling::ReasoningBudget>
TextModelRunner::BuildReasoningBudget(
    std::size_t tokens, std::span<const TextRunnerToken> prompt) const {
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
  const auto opened =
      std::ranges::find(prompt.rbegin(), prompt.rend(), start.front());
  if (opened != prompt.rend())
    budget->prompt.assign(opened.base(), prompt.end());
  const std::lock_guard lock(constraint_mutex_);
  if (!partial_utf8_ && VocabularySize() != 0) {
    auto partial = std::make_shared<std::vector<bool>>(VocabularySize());
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
    }
    partial_utf8_ = std::move(partial);
  }
  budget->partial_utf8 = partial_utf8_;
  return budget;
}

void TextModelRunner::StreamPersistentSnapshot(
    const TextRunnerSnapshot& snapshot, const SnapshotSink& sink) const {
  std::vector<std::uint8_t> payload(PersistentSnapshotPayloadBytes(snapshot));
  if (SerializePersistentSnapshot(snapshot, payload) != payload.size()) {
    throw std::runtime_error(
        "persistent snapshot serialized byte count mismatch");
  }
  sink(payload);
}

// ---------------------------------------------------------------------------
// Provider & Tool Registries
// ---------------------------------------------------------------------------

static std::vector<TextProvider> g_text_providers;
static std::vector<ToolDialect> g_tool_dialects;

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
  for (const auto& provider : g_text_providers) {
    if (provider.probe_gguf && provider.probe_gguf(reader)) {
      return &provider;
    }
  }
  return nullptr;
}

void RegisterToolDialect(ToolDialect dialect) {
  for (auto& existing : g_tool_dialects) {
    if (existing.format == dialect.format) {
      existing = std::move(dialect);
      return;
    }
  }
  g_tool_dialects.push_back(std::move(dialect));
}

const ToolDialect* FindToolDialect(
    sampling::JsonConstraint::ToolFormat format) {
  for (const auto& dialect : g_tool_dialects) {
    if (dialect.format == format) {
      return &dialect;
    }
  }
  return nullptr;
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
