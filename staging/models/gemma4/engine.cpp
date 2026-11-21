#include "staging/models/gemma4/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
#include "staging/models/gemma4/kernels/rocm/device_model.hpp"
#include "staging/models/gemma4/kernels/rocm/executor.hpp"
#include "staging/models/gemma4/tokenizer.hpp"
#include "staging/models/gemma4/vision/encoder.hpp"
#include "staging/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {
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
  if (options.prefill_chunk == 0) {
    AssignError(error_msg, "prefill_chunk must be positive");
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
  if (c.is_draft) {
    AssignError(error_msg, "this artifact is a draft, not the trunk");
    return nullptr;
  }
  if (options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::FromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, options.prefill_chunk, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  if (!options.draft_path.empty()) {
    m->draft_reader_ =
        core::GgufReader::OpenFile(options.draft_path, error_msg);
    if (!m->draft_reader_) {
      return nullptr;
    }
    auto draft_weights = DraftWeights::Bind(*m->draft_reader_, c, error_msg);
    if (!draft_weights) {
      return nullptr;
    }
    m->draft_weights_ =
        std::make_unique<DraftWeights>(std::move(*draft_weights));
    m->draft_device_ = rocm::DeviceDraft::Upload(*m->draft_weights_,
                                                 *m->draft_reader_, error_msg);
    if (!m->draft_device_) {
      return nullptr;
    }
    if (!m->executor_->AttachDraft(*m->draft_device_, error_msg)) {
      return nullptr;
    }
  }
  if (!options.vision_model_path.empty()) {
    m->vision_ = vision::Encoder::Open(options.vision_model_path, error_msg);
    if (!m->vision_) {
      return nullptr;
    }
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
  return tokenizer_->Encode(text);
}

std::vector<std::int32_t> Model::EncodeChat(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options) const {
  const std::string rendered = RenderChat(messages, tools, options);
  return tokenizer_->Encode(rendered, /*add_special=*/false,
                            /*parse_special=*/true);
}

vision::PreparedPrompt Model::EncodeChatVision(
    std::span<const ChatMessage> messages, std::span<const ChatTool> tools,
    const ChatTemplateOptions& options, const std::vector<core::Image>& images,
    std::string* error_msg) const {
  if (vision_ == nullptr) {
    AssignError(error_msg, "no vision tower loaded");
    return {};
  }
  const std::string rendered = RenderChat(messages, tools, options);
  return vision::BuildPrompt(*tokenizer_, rendered, images, {}, error_msg);
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  return tokenizer_->Decode(tokens);
}

std::string Model::TokenText(std::int32_t token) const {
  // Stop tokens are end-of-generation markers, never visible text: the EOT is
  // the assistant-turn terminator GGUF metadata resolves to EOS. Suppress it
  // here so no decoding path surfaces the raw turn marker, including streamed
  // speculative drafts that emit `TokenText` per accepted token.
  if (token == tokenizer_->EosId()) {
    return {};
  }
  return tokenizer_->TokenToPiece(token);
}

std::int32_t Model::EosToken() const noexcept {
  return tokenizer_->EosId();
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return token == tokenizer_->EosId();
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::string Model::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("Gemma-4-26B-A4B"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return executor_->SessionBytes(context);
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      executor_(model_->executor_.get()) {
  logits_.resize(model_->VocabSize());
  draft_logits_.resize(model_->VocabSize());
  drafts_.resize(rocm::Executor::kMaxVerifyRows - 1U);
  verify_rows_.resize(static_cast<std::size_t>(rocm::Executor::kMaxVerifyRows) *
                      model_->VocabSize());
}

Session::~Session() {
  if (vision_dev_ != nullptr) {
    (void)hipFree(vision_dev_);
    vision_dev_ = nullptr;
    vision_dev_cap_ = 0;
  }
}

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

void Session::SetMtpEnabled(bool enabled) {
  mtp_enabled_ = enabled && model_->HasMtp();
}

void Session::SetDraftLimits(std::uint32_t min_drafts,
                             std::uint32_t max_drafts) {
  const std::uint32_t cap = rocm::Executor::kMaxVerifyRows - 1U;
  if (max_drafts == 0U || max_drafts > cap) {
    max_drafts = cap;
  }
  if (min_drafts == 0U) {
    min_drafts = 1U;
  }
  if (min_drafts > max_drafts) {
    min_drafts = max_drafts;
  }
  draft_min_ = min_drafts;
  draft_max_ = max_drafts;
  draft_k_ = std::min(std::max(draft_k_, min_drafts), max_drafts);
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
  if (!valid_) {
    Reset();
  }
  // The KV cache is position-major and rewinds by rewriting rows, so a pure
  // extension only feeds the new tail; any divergence restarts the session.
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
  // over the whole span instead of a per-token GEMV step. The executor's
  // position already equals the common prefix length, so Prefill continues
  // from there and leaves the final token's logits.
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

bool Session::Sync(std::span<const std::int32_t> prompt,
                   const std::vector<vision::VisionSlot>& images,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (images.empty()) {
    return Sync(prompt, error_msg);
  }
  vision::Encoder* encoder = model_->vision_.get();
  if (encoder == nullptr) {
    AssignError(error_msg, "no vision tower loaded");
    return false;
  }
  const std::size_t hidden = encoder->config().projection_dim;

  // Grow the session's device staging buffer to hold every image's rows.
  std::size_t total_rows = 0;
  for (const vision::VisionSlot& slot : images) {
    total_rows += slot.count;
  }
  const std::size_t need_bytes = total_rows * hidden * sizeof(float);
  if (need_bytes > vision_dev_cap_) {
    if (vision_dev_ != nullptr) {
      (void)hipFree(vision_dev_);
      vision_dev_ = nullptr;
      vision_dev_cap_ = 0;
    }
    if (hipMalloc(&vision_dev_, need_bytes) != hipSuccess) {
      AssignError(error_msg, "failed to allocate vision staging buffer");
      return false;
    }
    vision_dev_cap_ = need_bytes;
  }

  // Run the tower over each slot and upload its rows into a contiguous device
  // region; the executor splices them over the token embeddings at the slot
  // offsets during prefill.
  std::vector<rocm::Executor::VisionImage> staged;
  staged.reserve(images.size());
  std::vector<float> rows;
  std::size_t row_cursor = 0;
  for (const vision::VisionSlot& slot : images) {
    if (!encoder->Encode(slot.pixels.data(), slot.nx, slot.ny, rows,
                         error_msg)) {
      return false;
    }
    if (rows.size() != static_cast<std::size_t>(slot.count) * hidden) {
      AssignError(error_msg, "vision encoder row count mismatch");
      return false;
    }
    float* dst = vision_dev_ + row_cursor * hidden;
    if (hipMemcpy(dst, rows.data(), rows.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      AssignError(error_msg, "failed to upload image embeddings");
      return false;
    }
    staged.push_back({dst, slot.offset, slot.count});
    row_cursor += slot.count;
  }

  // Image rows cannot be reused from a text-only prefix, so reprocess the
  // whole sequence with the staged embeddings spliced in.
  Reset();
  valid_ = false;
  executor_->SetVision(std::move(staged));
  if (!executor_->Prefill(*session_, prompt.data(),
                          static_cast<std::uint32_t>(prompt.size()),
                          error_msg)) {
    return false;
  }
  tokens_.assign(prompt.begin(), prompt.end());
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

  // Speculative decoding: with a draft sidecar and an unpenalized sampler,
  // chain up to k drafts through the MTP block (fixed position, trunk KV read
  // only), verify all k + 1 rows in one batched trunk pass, and keep the
  // longest prefix the trunk agrees with. The verify pass reads the trunk
  // weights once for the whole block. Greedy samplers take the exact-match
  // path; temperature sampling uses the standard accept-and-residual rule.
  std::size_t k = 0;
  if (model_->HasMtp() && mtp_enabled_ && max_tokens >= 2 &&
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

  const std::size_t vocab = logits_.size();
  const bool sampled = !sampler.config().can_use_unmodified_argmax();
  const auto token = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(token)) {
    result->stop = true;
    return true;
  }

  // Draft chain: the first draft conditions on the trunk hidden that produced
  // the current logits; every later one on the previous draft's own output
  // hidden. All drafts share the trunk position because the MTP block keeps no
  // KV of its own and reads the trunk cache read-only.
  valid_ = false;
  if (!executor_->DraftStep(*session_, token, executor_->h_out(), error_msg)) {
    return false;
  }
  std::vector<std::vector<sampling::TokenId>> draft_ids(k);
  std::vector<std::vector<float>> draft_probs(k);
  std::vector<double> draft_p(k, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    if (i > 0 && !executor_->DraftStep(*session_, drafts_[i - 1],
                                       executor_->draft_h_next(), error_msg)) {
      return false;
    }
    (void)hipMemcpy(draft_logits_.data(), executor_->draft_logits(),
                    vocab * sizeof(float), hipMemcpyDeviceToHost);
    if (!sampled) {
      drafts_[i] = static_cast<std::int32_t>(
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
    drafts_[i] =
        static_cast<std::int32_t>(q.Sample(sampler.mutable_rng_state()));
    draft_p[i] = q.probability(static_cast<sampling::TokenId>(drafts_[i]));
  }
  result->drafted += k;
  if (!executor_->Verify(*session_, token, drafts_.data(),
                         static_cast<std::uint32_t>(k), error_msg)) {
    return false;
  }
  (void)hipMemcpy(verify_rows_.data(), executor_->verify_logits(),
                  (k + 1) * vocab * sizeof(float), hipMemcpyDeviceToHost);
  const auto row = [&](std::size_t r) {
    return std::span<const float>(verify_rows_.data() + r * vocab, vocab);
  };

  // Keep the longest draft prefix the trunk reproduces. Greedy compares
  // argmaxes; random sampling accepts draft d with probability min(1, p(d)/
  // q(d)) and, on the first rejection, replaces d with a draw from the
  // residual (p - q)+.
  std::size_t accepted = 0;
  std::int32_t correction = -1;
  if (!sampled) {
    while (accepted < k && static_cast<std::int32_t>(sampler.SampleGreedy(
                               row(accepted))) == drafts_[accepted]) {
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
          p.probability(static_cast<sampling::TokenId>(drafts_[accepted]))) {
        continue;
      }
      correction = static_cast<std::int32_t>(
          p.SampleResidual(draft_ids[accepted], draft_probs[accepted],
                           sampler.mutable_rng_state()));
      break;
    }
  }
  // A stop token truncates the accepted prefix to the content before it:
  // like SelectNext, the engine never applies an end-of-turn token, so the
  // rollback below keeps exactly the committed content rows and Position()
  // equals the prompt plus `result->tokens`.
  bool stopped = false;
  for (std::size_t i = 0; i < accepted; ++i) {
    if (is_stop(drafts_[i])) {
      accepted = i;
      stopped = true;
      break;
    }
  }

  // Commit. A full accept leaves h_out() at the last verified row (Verify set
  // it), so the next round's first draft conditions on it directly; a partial
  // one rewinds the trunk to the committed prefix and h_out() to its last row.
  std::vector<sampling::TokenId> committed;
  committed.reserve(k + 2);
  committed.push_back(static_cast<sampling::TokenId>(token));
  for (std::size_t i = 0; i < accepted; ++i) {
    committed.push_back(static_cast<sampling::TokenId>(drafts_[i]));
  }
  if (accepted == k) {
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
    // The target ends the turn at the rejected row; it is a stop signal,
    // never content, and it must not reach the trunk.
    stopped = true;
  }
  if (!stopped && correction >= 0) {
    // The trunk's own token at the first rejected row continues the sequence;
    // feeding it is the round's only extra cost.
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

}  // namespace gufo::models::gemma4