#include "staging/models/qwen36_a3b/engine.hpp"

#include <algorithm>
#include <cstring>

#include "src/core/gguf_reader.hpp"
#include "staging/models/qwen36_a3b/kernels/rocm/device_model.hpp"
#include "staging/models/qwen36_a3b/kernels/rocm/executor.hpp"
#include "staging/models/qwen36_a3b/weights.hpp"

namespace gufo::models::qwen36_a3b {
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
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  // The MTP draft block is optional: artifacts without it load and decode
  // single-token, artifacts with it enable draft-verify speculation.
  const auto mtp = MtpWeights::Bind(*m->reader_, c);
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, mtp.has_value() ? &*mtp : nullptr, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, options.max_context, error_msg,
                             options.attn_window, options.attn_sink);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == EosToken() ||
         token == static_cast<std::int32_t>(tokenizer_->GetPadTokenId());
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("Qwen3.6-35B-A3B"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return executor_->SessionBytes(context);
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      executor_(model_->executor_.get()) {
  logits_.resize(model_->VocabSize());
  draft_logits_.resize(model_->VocabSize());
  verify_rows_.resize(rocm::Executor::kMaxVerifyRows * model_->VocabSize());
}

void Session::SetDraftLimits(std::uint32_t min_drafts,
                             std::uint32_t max_drafts) {
  const std::uint32_t cap = rocm::Executor::kMaxVerifyRows - 1U;
  draft_max_ = std::min(max_drafts == 0U ? 1U : max_drafts, cap);
  draft_min_ = std::min(std::max(min_drafts, 1U), draft_max_);
  draft_k_ = draft_max_;
}

void Session::SetMtpEnabled(bool enabled) {
  session_->SetMtpEnabled(enabled);
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const auto identity =
      prompt ? prompt->cache_identity : std::vector<std::uint8_t>{};
  if (!tokens_.empty() && identity != image_identity_)
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(std::move(prompt), model_->vision_, nullptr);
  image_identity_ = identity;
  valid_ = was_valid;
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}

std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  valid_ = true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  // Feed the new tail through the batched prefill path: one GEMM-driven pass
  // over the whole span instead of a per-token GEMV step, so prompt processing
  // runs at prefill throughput. The executor's position already equals the
  // common prefix length (reset to 0 above, or a pure extension), so Prefill
  // continues from there and leaves the final token's logits.
  if (!executor_->Prefill(*session_, prompt.data() + common,
                          static_cast<std::uint32_t>(prompt.size() - common),
                          error_msg)) {
    return false;
  }
  for (std::size_t i = common; i < prompt.size(); ++i) {
    tokens_.push_back(prompt[i]);
  }
  (void)hipMemcpy(logits_.data(), executor_->logits(),
                  logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  valid_ = true;
  return true;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  if (!executor_->Step(*session_, token, error_msg)) {
    return false;
  }
  tokens_.push_back(token);
  (void)hipMemcpy(logits_.data(), executor_->logits(),
                  logits_.size() * sizeof(float), hipMemcpyDeviceToHost);
  valid_ = true;
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
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

  // Speculative decoding: with an MTP block and an unpenalized sampler,
  // chain up to k drafts through the draft block, verify all k + 1 rows in
  // one batched trunk pass, and keep the longest prefix the trunk agrees
  // with. The verify pass reads the trunk weights once for the whole block,
  // so a hit costs ~1.2 tokens' time. Greedy samplers take the exact-match
  // path; temperature sampling uses the standard accept-and-residual rule.
  std::size_t k = 0;
  if (model_->HasMtp() && session_->mtp_enabled() && max_tokens >= 2 &&
      !sampler.config().penalties_enabled() &&
      tokens_.size() + 2 <= ContextSize()) {
    k = std::min<std::size_t>(draft_k_, max_tokens - 1);
    k = std::min<std::size_t>(k, ContextSize() - tokens_.size() - 1);
    k = std::min<std::size_t>(k, rocm::Executor::kMaxVerifyRows - 1U);
  }
  if (k == 0) {
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
    if (is_stop(token)) {
      result->stop = true;
      return true;
    }
    valid_ = false;
    if (!executor_->Step(*session_, token, error_msg)) {
      return false;
    }
    tokens_.push_back(token);
    (void)hipMemcpy(logits_.data(), executor_->logits(),
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
  if (!executor_->MtpStep(*session_, token, error_msg)) {
    return false;
  }
  std::int32_t drafts[rocm::Executor::kMaxVerifyRows - 1U];
  // Random sampling keeps each draft distribution so the accept step can
  // weigh the target against it and draw the residual on a rejection.
  std::vector<std::vector<sampling::TokenId>> draft_ids(k);
  std::vector<std::vector<float>> draft_probs(k);
  std::vector<double> draft_p(k, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    if (i > 0 && !executor_->MtpDraft(*session_, drafts[i - 1], error_msg)) {
      return false;
    }
    (void)hipMemcpy(draft_logits_.data(), executor_->mtp_logits(),
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
  if (!executor_->Verify(*session_, token, drafts,
                         static_cast<std::uint32_t>(k), error_msg)) {
    return false;
  }
  (void)hipMemcpy(verify_rows_.data(), executor_->verify_logits(),
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
  // A stop token truncates the accepted prefix to the content before it:
  // like the serial path, the engine never applies an end-of-turn token, so
  // the rollback keeps exactly the committed content rows and Position()
  // equals the prompt plus `result->tokens`.
  bool stopped = false;
  for (std::size_t i = 0; i < accepted; ++i) {
    if (is_stop(drafts[i])) {
      accepted = i;
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
    if (!executor_->MtpAdvance(
            *session_, drafts[k - 1],
            executor_->verify_hidden() + (k - 1) * model_->config().hidden_size,
            error_msg)) {
      return false;
    }
    logits_.assign(row(k).begin(), row(k).end());
    if (!stopped) {
      draft_k_ = std::min(draft_max_, draft_k_ + 1U);
    }
  } else {
    executor_->RollbackVerify(*session_,
                              static_cast<std::uint32_t>(accepted + 1U));
    if (accepted == 0) {
      draft_k_ = std::max(draft_min_, draft_k_ - 1U);
    }
  }
  result->accepted += accepted;
  for (const auto id : committed) {
    tokens_.push_back(static_cast<std::int32_t>(id));
    result->tokens.push_back(static_cast<std::int32_t>(id));
  }
  sampler.Accept(std::span<const sampling::TokenId>(committed));
  if (!stopped && correction >= 0 && is_stop(correction)) {
    // The target ends the turn at the rejected row; it is a stop
    // signal, never content, and it must not reach the trunk.
    stopped = true;
  }
  if (!stopped && correction >= 0) {
    // The target's own token at the first rejected row continues the
    // sequence; feeding it is the round's only extra cost.
    if (!executor_->Step(*session_, correction, error_msg)) {
      return false;
    }
    tokens_.push_back(correction);
    result->tokens.push_back(correction);
    sampler.Accept(static_cast<sampling::TokenId>(correction));
    (void)hipMemcpy(logits_.data(), executor_->logits(), vocab * sizeof(float),
                    hipMemcpyDeviceToHost);
  }
  result->stop = stopped;
  valid_ = true;
  return true;
}

}  // namespace gufo::models::qwen36_a3b