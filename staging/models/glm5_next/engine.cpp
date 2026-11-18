#include "staging/models/glm5_next/engine.hpp"

#include <algorithm>
#include <utility>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/kernels/rocm/device_model.hpp"
#include "staging/models/glm5_next/kernels/rocm/executor.hpp"
#include "staging/models/glm5_next/weights.hpp"

namespace gufo::models::glm5_next {
namespace {

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.max_context == 0) {
    AssignError(error_msg, "max_context must be positive");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = GlmTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (m->device_ == nullptr) {
    return nullptr;
  }
  m->executor_ = rocm::Executor::Create(*m->device_, error_msg);
  if (m->executor_ == nullptr) {
    return nullptr;
  }
  return m;
}

std::shared_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  if (!live_session_.expired()) {
    AssignError(error_msg,
                "the GLM-5 device state is shared by all sessions");
    return nullptr;
  }
  std::unique_ptr<rocm::Session> state =
      executor_->CreateSession(max_context, error_msg);
  if (state == nullptr) {
    return nullptr;
  }
  auto session = std::shared_ptr<Session>(
      new Session(shared_from_this(), std::move(state), max_context));
  live_session_ = session;
  return session;
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const std::vector<TokenId> ids = tokenizer_->Encode(text);
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<TokenId> ids;
  ids.reserve(tokens.size());
  for (const std::int32_t t : tokens) {
    ids.push_back(static_cast<TokenId>(t));
  }
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

bool Model::HasMtp() const noexcept {
  return executor_ != nullptr && executor_->HasMtp();
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("GLM-5.3-Flash"));
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_ == nullptr ? 0 : device_->resident_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return executor_ == nullptr ? 0 : executor_->SessionBytes(context);
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> state,
                 std::uint32_t max_context)
    : model_(std::move(model)),
      state_(std::move(state)),
      max_context_(max_context) {
  draft_logits_.resize(model_->VocabSize());
  verify_rows_.resize(rocm::Executor::kMaxVerifyRows * model_->VocabSize());
}

Session::~Session() = default;

void Session::SyncLogits() {
  logits_.resize(model_->VocabSize());
  if (hipMemcpy(logits_.data(), model_->executor_->logits(),
                logits_.size() * sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    valid_ = false;
  }
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.size() > max_context_) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (prompt.empty()) {
    Reset();
    return true;
  }
  logits_.assign(model_->VocabSize(), 0.0f);
  const auto common = std::mismatch(tokens_.begin(), tokens_.end(),
                                    prompt.begin(), prompt.end());
  if (common.first != tokens_.end()) {
    state_->Reset();
    tokens_.clear();
    model_->executor_->Prefill(*state_, prompt.data(), prompt.size());
    tokens_.assign(prompt.begin(), prompt.end());
    SyncLogits();
    return valid_;
  }
  const std::span<const std::int32_t> tail(
      prompt.begin() +
          static_cast<std::ptrdiff_t>(common.first - tokens_.begin()),
      prompt.end());
  if (!tail.empty()) {
    model_->executor_->Prefill(*state_, tail.data(), tail.size());
    tokens_.insert(tokens_.end(), tail.begin(), tail.end());
    SyncLogits();
  }
  return valid_;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (tokens_.size() + 1 > max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  model_->executor_->Step(*state_, token);
  tokens_.push_back(token);
  SyncLogits();
  return valid_;
}

std::uint32_t Session::Position() const noexcept {
  return static_cast<std::uint32_t>(std::max<std::int64_t>(
      state_->position() + 1, 0));
}

void Session::Reset() {
  state_->Reset();
  tokens_.clear();
  logits_.clear();
  valid_ = true;
}

void Session::SetDraftLimits(std::uint32_t min_drafts,
                             std::uint32_t max_drafts) {
  const std::uint32_t cap = rocm::Executor::kMaxVerifyRows - 1U;
  // The shared CLI defaults (min 1, max >= the verify-row cap) mean "no user
  // preference". Measured optima for this nextn block on gfx1151 are a single
  // draft: k=1 emits seed + (draft or correction) every round and served
  // greedy decode at 13.8 t/s with 87.5 % acceptance, versus 12.1 t/s for the
  // open-ended streak ladder (docs/models/glm-5.3-flash/BENCHMARKS.md).
  // Explicit max in (1, cap) still opts into the ramp.
  if (min_drafts <= 1U && (max_drafts == 0U || max_drafts >= cap)) {
    max_drafts = 1U;
  }
  draft_max_ = std::min(max_drafts == 0U ? 1U : max_drafts, cap);
  draft_min_ = std::min(std::max(min_drafts, 1U), draft_max_);
  draft_k_ = draft_min_;
  draft_streak_ = 0U;
}

void Session::SetMtpEnabled(bool enabled) {
  mtp_enabled_ = enabled;
  model_->executor_->SetMtpPrefill(enabled);
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler,
                         DecodeResult* result, std::string* error_msg,
                         bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };

  // Speculative decoding: with the nextn block and an unpenalized sampler,
  // chain up to k drafts through the draft block, verify all k + 1 rows in
  // one batched trunk pass, and keep the longest prefix the trunk agrees
  // with. The verify pass reads the trunk weights once for the whole block,
  // so a hit costs ~1.2 tokens' time. Greedy samplers take the exact-match
  // path; temperature sampling uses the standard accept-and-residual rule.
  std::size_t k = 0;
  if (model_->HasMtp() && mtp_enabled_ && max_tokens >= 2 &&
      !sampler.config().penalties_enabled() &&
      tokens_.size() + 2 <= static_cast<std::size_t>(max_context_)) {
    k = std::min<std::size_t>(draft_k_, max_tokens - 1);
    k = std::min<std::size_t>(
        k, static_cast<std::size_t>(max_context_) - tokens_.size() - 1);
    k = std::min<std::size_t>(k, rocm::Executor::kMaxVerifyRows - 1U);
  }
  if (k == 0) {
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
    if (is_stop(token)) {
      result->stop = true;
      return true;
    }
    valid_ = false;
    model_->executor_->Step(*state_, token);
    tokens_.push_back(token);
    (void)hipMemcpy(logits_.data(), model_->executor_->logits(),
                    logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    valid_ = true;
    return true;
  }

  const std::size_t vocab = draft_logits_.size();
  const bool sampled = !sampler.config().can_use_unmodified_argmax();
  const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(token)) {
    result->stop = true;
    return true;
  }

  // Draft chain: the first draft conditions on the trunk hidden state,
  // every later one on the previous draft's own output hidden.
  valid_ = false;
  model_->executor_->MtpStep(*state_, token, model_->executor_->last_hidden());
  std::int32_t drafts[rocm::Executor::kMaxVerifyRows - 1U];
  // Random sampling keeps each draft distribution so the accept step can
  // weigh the target against it and draw the residual on a rejection.
  std::vector<std::vector<sampling::TokenId>> draft_ids(k);
  std::vector<std::vector<float>> draft_probs(k);
  std::vector<double> draft_p(k, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    if (i > 0) {
      model_->executor_->MtpDraft(*state_, drafts[i - 1]);
    }
    (void)hipMemcpy(draft_logits_.data(), model_->executor_->mtp_logits(),
                    vocab * sizeof(float), hipMemcpyDeviceToHost);
    if (!sampled) {
      drafts[i] = static_cast<std::int32_t>(
          sampler.SampleGreedy(std::span<const float>(draft_logits_)));
      continue;
    }
    const auto q = sampler.Distribution(std::span<const float>(draft_logits_));
    draft_ids[i].reserve(q.entries().size());
    draft_probs[i].reserve(q.entries().size());
    for (const auto& entry : q.entries()) {
      draft_ids[i].push_back(entry.token);
      draft_probs[i].push_back(static_cast<float>(entry.value));
    }
    drafts[i] =
        static_cast<std::int32_t>(q.Sample(sampler.mutable_rng_state()));
    draft_p[i] = q.probability(static_cast<sampling::TokenId>(drafts[i]));
  }
  result->drafted += k;
  model_->executor_->Verify(*state_, token, drafts,
                            static_cast<std::uint32_t>(k));
  (void)hipMemcpy(verify_rows_.data(), model_->executor_->verify_logits(),
                  (k + 1) * vocab * sizeof(float), hipMemcpyDeviceToHost);
  const auto row = [&](std::size_t r) {
    return std::span<const float>(verify_rows_.data() + r * vocab, vocab);
  };

  // Keep the longest draft prefix the target reproduces. Greedy compares
  // argmaxes; random sampling accepts draft d with probability min(1,
  // p(d)/q(d)) and, on the first rejection, replaces d with a draw from
  // the residual (p - q)+.
  std::size_t accepted = 0;
  std::int32_t correction = -1;
  if (!sampled) {
    while (accepted < k && static_cast<std::int32_t>(sampler.SampleGreedy(
                               row(accepted))) == drafts[accepted]) {
      ++accepted;
    }
    if (accepted < k) {
      correction =
          static_cast<std::int32_t>(sampler.SampleGreedy(row(accepted)));
    }
  } else {
    for (; accepted < k; ++accepted) {
      const auto p = sampler.Distribution(row(accepted));
      if (sampler.Uniform() * draft_p[accepted] <
          p.probability(static_cast<sampling::TokenId>(drafts[accepted]))) {
        continue;
      }
      correction = static_cast<std::int32_t>(
          p.SampleResidual(draft_ids[accepted], draft_probs[accepted],
                           sampler.mutable_rng_state()));
      break;
    }
  }
  bool stopped = false;
  for (std::size_t i = 0; i < accepted; ++i) {
    if (is_stop(drafts[i])) {
      accepted = i + 1;
      stopped = true;
      break;
    }
  }

  // Commit. A full accept advances the draft cache over the last draft
  // against its own target hidden row; a partial one rewinds the trunk and
  // the draft cache to the committed prefix.
  std::vector<sampling::TokenId> committed;
  committed.reserve(k + 2);
  committed.push_back(static_cast<sampling::TokenId>(token));
  for (std::size_t i = 0; i < accepted; ++i) {
    committed.push_back(static_cast<sampling::TokenId>(drafts[i]));
  }
  if (accepted == k) {
    model_->executor_->MtpAdvance(
        *state_, drafts[k - 1],
        model_->executor_->verify_hidden() +
            (k - 1) * model_->config().hidden_size);
    logits_.assign(row(k).begin(), row(k).end());
    if (!stopped) {
      ++draft_streak_;
      if (draft_streak_ >= 2U) {
        draft_k_ = std::min(draft_max_, draft_k_ + 1U);
        draft_streak_ = 0U;
      }
    }
  } else {
    model_->executor_->RollbackVerify(*state_,
                                      static_cast<std::uint32_t>(accepted + 1U));
    draft_k_ = std::max(draft_min_, draft_k_ - 1U);
    draft_streak_ = 0U;
  }
  result->accepted += accepted;
  for (const auto id : committed) {
    tokens_.push_back(static_cast<std::int32_t>(id));
    result->tokens.push_back(static_cast<std::int32_t>(id));
  }
  sampler.Accept(std::span<const sampling::TokenId>(committed));
  if (!stopped && correction >= 0) {
    // The target's own token at the first rejected row continues the
    // sequence; feeding it is the round's only extra cost.
    model_->executor_->Step(*state_, correction);
    tokens_.push_back(correction);
    result->tokens.push_back(correction);
    sampler.Accept(static_cast<sampling::TokenId>(correction));
    (void)hipMemcpy(logits_.data(), model_->executor_->logits(),
                    vocab * sizeof(float), hipMemcpyDeviceToHost);
    result->stop = is_stop(correction);
  } else if (stopped) {
    result->stop = true;
  }
  valid_ = true;
  return true;
}

}  // namespace gufo::models::glm5_next
