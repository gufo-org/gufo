#include "src/cli/serve/tp_executor.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gufo::server {
namespace {

constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
/// Recorded in place of a result when a call throws.
constexpr std::uint64_t kFailureMarker = 0xdead'c0de'0000'0001ULL;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

[[nodiscard]] std::string_view OpName(TpInstructionOp op) {
  switch (op) {
    case TpInstructionOp::kInvalidate:
      return "reset";
    case TpInstructionOp::kPrefill:
      return "prefill";
    case TpInstructionOp::kAdvance:
      return "advance";
    case TpInstructionOp::kDecode:
      return "decode";
    case TpInstructionOp::kSnapshot:
      return "snapshot";
    case TpInstructionOp::kDrop:
      return "drop";
    case TpInstructionOp::kRestore:
      return "restore";
    case TpInstructionOp::kReuse:
      return "reuse";
    case TpInstructionOp::kCancelPrepare:
      return "cancel-prepare";
    case TpInstructionOp::kAdvanceBatch:
      return "batched advance";
    case TpInstructionOp::kEnd:
      return "end";
    case TpInstructionOp::kNone:
      break;
  }
  return "unknown";
}

[[nodiscard]] std::string Hex(std::uint64_t value) {
  char text[19];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(value));
  return text;
}

[[nodiscard]] std::uint32_t ToWire(std::size_t value, const char* what) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string("TP instruction ") + what +
                              " exceeds the protocol range");
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::int32_t TokenToWire(TextRunnerToken token) {
  if (token >
      static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 token exceeds the instruction range");
  }
  return static_cast<std::int32_t>(token);
}

/// A batch member is recorded in its request exactly like the advance it
/// replaces, so both ranks agree on a request's calls however they are batched.
[[nodiscard]] TpInstruction MemberAdvance(const TpBatchMember& member) {
  return {.op = TpInstructionOp::kAdvance,
          .state = member.state,
          .token = member.token};
}

/// Forward calls exchange partials, so they run inside a collective scope.
[[nodiscard]] bool RunsForward(TpInstructionOp op) {
  return op == TpInstructionOp::kPrefill || op == TpInstructionOp::kAdvance ||
         op == TpInstructionOp::kDecode || op == TpInstructionOp::kAdvanceBatch;
}

}  // namespace

void TpExecutionDigest::Add(std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value_ ^= (value >> shift) & 0xffU;
    value_ *= kFnvPrime;
  }
}

void TpExecutionDigest::Add(std::span<const TextRunnerToken> tokens) noexcept {
  Add(tokens.size());
  for (const auto token : tokens) {
    Add(token);
  }
}

void DigestTpCall(TpExecutionDigest& digest, const TpInstruction& instruction,
                  std::span<const TextRunnerToken> prefill_prompt) {
  digest.Add(static_cast<std::uint64_t>(instruction.op));
  digest.Add(instruction.state);
  digest.Add(instruction.snapshot_id);
  digest.Add(static_cast<std::uint32_t>(instruction.token));
  digest.Add(instruction.offset);
  digest.Add(instruction.count);
  digest.Add(instruction.prompt_size);
  digest.Add(instruction.rng);
  digest.Add(static_cast<std::uint32_t>(instruction.pending));
  if (instruction.op == TpInstructionOp::kPrefill) {
    digest.Add(prefill_prompt);
  }
}

void DigestTpPrefill(TpExecutionDigest& digest, const TextPrefillStep& step,
                     std::size_t checkpoint) {
  digest.Add(step.consumed_tokens);
  digest.Add(step.decode_ready ? 1U : 0U);
  digest.Add(checkpoint);
}

void DigestTpAdvance(TpExecutionDigest& digest, std::size_t checkpoint) {
  digest.Add(checkpoint);
}

void DigestTpDecode(TpExecutionDigest& digest, const TextDecodeStep& step,
                    std::size_t checkpoint) {
  digest.Add(step.selections.size());
  for (const auto& selection : step.selections) {
    digest.Add(selection.token);
    digest.Add(selection.stop ? 1U : 0U);
  }
  digest.Add(step.stop ? 1U : 0U);
  digest.Add(step.draft_tokens);
  digest.Add(step.draft_accepted_tokens);
  digest.Add(checkpoint);
}

void DigestTpFailure(TpExecutionDigest& digest) {
  digest.Add(kFailureMarker);
}

bool TpCacheAcknowledged(TpInstructionOp op) noexcept {
  return op == TpInstructionOp::kSnapshot || op == TpInstructionOp::kRestore ||
         op == TpInstructionOp::kReuse || op == TpInstructionOp::kCancelPrepare;
}

void TpControlInstructionSink::Synchronize(std::uint64_t sequence,
                                           TpInstruction instruction,
                                           const std::function<void()>& local) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  std::string error;
  if (!broker_->RegisterInstruction(sequence, instruction.index, &error)) {
    throw std::runtime_error(error);
  }
  if (!channel_->SendCommand({.sequence = sequence,
                              .kind = TpControlCommandKind::kInstruction,
                              .instruction = instruction},
                             &error)) {
    broker_->FailAll("TP cache instruction send failed: " + error);
    throw std::runtime_error(error);
  }
  ++next_index_;
  std::exception_ptr local_failure;
  try {
    local();
  } catch (...) {
    local_failure = std::current_exception();
  }
  // Drain even after a local failure: the next instruction must not overtake
  // a worker copy, and its acknowledgement must not become an orphan.
  const bool ok = broker_->WaitForInstruction(&error);
  if (local_failure)
    std::rethrow_exception(local_failure);
  if (!ok)
    throw std::runtime_error("TP cache worker failed: " + error);
}

bool TpControlInstructionSink::Send(std::uint64_t sequence,
                                    TpInstruction instruction,
                                    std::uint64_t* index, std::string* error) {
  const std::lock_guard<std::mutex> lock(mutex_);
  instruction.index = next_index_;
  const TpControlCommand command{
      .sequence = sequence,
      .kind = TpControlCommandKind::kInstruction,
      .instruction = std::move(instruction),
  };
  if (!channel_->SendCommand(command, error)) {
    return false;
  }
  if (index != nullptr) {
    *index = next_index_;
  }
  ++next_index_;
  return true;
}

/// A rank-0 state paired with the id that names rank 1's corresponding state.
class TpMirroredRunner::State final : public TextRunnerState {
public:
  State(const TpMirroredRunner* owner, std::unique_ptr<TextRunnerState> inner,
        std::uint32_t id)
      : owner_(owner), inner_(std::move(inner)), id_(id) {}

  void Invalidate() noexcept override {
    owner_->SendInvalidate(id_);
    inner_->Invalidate();
  }
  /// Deliberately not forwarded; see `TpMirroredRunner`.
  void SetCancellationCheck(const CancellationCheck&) override {}
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return inner_->MeasuredResources();
  }

  [[nodiscard]] TextRunnerState& inner() const noexcept { return *inner_; }
  [[nodiscard]] std::uint32_t id() const noexcept { return id_; }

private:
  const TpMirroredRunner* owner_;
  std::unique_ptr<TextRunnerState> inner_;
  std::uint32_t id_;
};

/// Binds a forward's collective scope while the forward runs. A scope that
/// cannot be bound or released leaves the pair's collectives untrustworthy, so
/// it fails the channel as well as the call.
class TpMirroredRunner::CallScope {
public:
  CallScope(const TpMirroredRunner& owner, std::uint64_t index)
      : owner_(owner), index_(index) {
    if (owner_.scope_ == nullptr) {
      return;
    }
    std::string error;
    if (!owner_.scope_->Begin(index_, &error)) {
      Fail("TP collective scope bind failed: " + error);
    }
    bound_ = true;
  }
  CallScope(const CallScope&) = delete;
  CallScope& operator=(const CallScope&) = delete;

  /// Releases the scope once the forward returned.
  void End() {
    if (!bound_) {
      return;
    }
    bound_ = false;
    std::string error;
    if (!owner_.scope_->End(index_, &error)) {
      Fail("TP collective scope release failed: " + error);
    }
  }

  ~CallScope() {
    if (!bound_) {
      return;
    }
    try {
      std::string error;
      if (!owner_.scope_->End(index_, &error)) {
        const std::lock_guard<std::mutex> lock(owner_.mutex_);
        if (owner_.failure_.empty()) {
          owner_.failure_ = "TP collective scope release failed: " + error;
        }
      }
    } catch (...) {
    }
  }

private:
  [[noreturn]] void Fail(const std::string& message) const {
    {
      const std::lock_guard<std::mutex> lock(owner_.mutex_);
      if (owner_.failure_.empty()) {
        owner_.failure_ = message;
      }
    }
    throw std::runtime_error(message);
  }

  const TpMirroredRunner& owner_;
  std::uint64_t index_;
  bool bound_{false};
};

TpMirroredRunner::TpMirroredRunner(std::shared_ptr<TextModelRunner> inner,
                                   std::shared_ptr<TpInstructionSink> sink,
                                   std::size_t snapshot_budget,
                                   std::shared_ptr<TpCallScope> scope)
    : inner_(std::move(inner)),
      sink_(std::move(sink)),
      snapshot_budget_(snapshot_budget),
      scope_(std::move(scope)) {
  if (inner_ == nullptr || sink_ == nullptr) {
    throw std::invalid_argument("TP mirrored runner needs a runner and a sink");
  }
  const auto capabilities = inner_->Descriptor().capabilities;
  multi_token_decode_ = capabilities.multi_token_decode;
}

void TpMirroredRunner::BeginRequest(std::uint64_t sequence) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (sequence == 0 || requests_.contains(sequence) ||
      unsettled_.contains(sequence) || rejected_.contains(sequence)) {
    throw std::logic_error("TP request is unnamed or already open");
  }
  requests_.emplace(sequence, Request{});
  unsettled_.insert(sequence);
}

bool TpMirroredRunner::EndRequest(std::uint64_t sequence, std::string* error) {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = requests_.find(sequence);
  if (found == requests_.end()) {
    SetError(error, "TP request " + std::to_string(sequence) + " is not open");
    return false;
  }
  const Request request = found->second;
  requests_.erase(found);
  std::erase_if(leases_,
                [&](const auto& lease) { return lease.second == sequence; });
  if (!failure_.empty()) {
    SetError(error, "TP instruction channel failed: " + failure_);
    return false;
  }
  if (request.count > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "TP request made too many model calls to report");
    return false;
  }
  const TpInstruction end{.op = TpInstructionOp::kEnd,
                          .count = static_cast<std::uint32_t>(request.count),
                          .digest = request.digest.value()};
  std::string send_error;
  if (!sink_->Send(sequence, end, nullptr, &send_error)) {
    failure_ = send_error.empty() ? "send failed" : send_error;
    SetError(error, "TP request end send failed: " + failure_);
    return false;
  }
  return true;
}

void TpMirroredRunner::Settle(std::uint64_t sequence, bool agreed) {
  const std::lock_guard<std::mutex> lock(mutex_);
  unsettled_.erase(sequence);
  if (!agreed) {
    rejected_.insert(sequence);
  }
}

bool TpMirroredRunner::Settled(std::uint64_t producer) const {
  return producer == 0 ||
         (!unsettled_.contains(producer) && !rejected_.contains(producer));
}

bool TpMirroredRunner::CanReuse(const TextRunnerState& state) const {
  const auto id = Mirrored(state).id();
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = producers_.find(id);
  return found == producers_.end() || Settled(found->second);
}

TpMirroredRunner::State& TpMirroredRunner::Mirrored(TextRunnerState& state) {
  auto* mirrored = dynamic_cast<State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

const TpMirroredRunner::State& TpMirroredRunner::Mirrored(
    const TextRunnerState& state) {
  const auto* mirrored = dynamic_cast<const State*>(&state);
  if (mirrored == nullptr) {
    throw std::logic_error("text runner state is not a TP2 mirrored state");
  }
  return *mirrored;
}

TpMirroredRunner::Sent TpMirroredRunner::Send(
    std::uint32_t state, const TpInstruction& instruction,
    std::span<const TextRunnerToken> prefill_prompt) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  const auto lease = leases_.find(state);
  if (lease == leases_.end()) {
    throw std::logic_error("TP2 model call outside a request");
  }
  Sent sent{.sequence = lease->second};
  auto& request = requests_.at(sent.sequence);
  DigestTpCall(request.digest, instruction, prefill_prompt);
  ++request.count;
  producers_[state] = sent.sequence;
  std::string error;
  if (!sink_->Send(sent.sequence, instruction, &sent.index, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
  return sent;
}

TpMirroredRunner::Sent TpMirroredRunner::SendBatch(
    TpInstruction& instruction) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!failure_.empty()) {
    throw std::runtime_error("TP instruction channel failed: " + failure_);
  }
  for (auto& member : instruction.batch) {
    const auto lease = leases_.find(member.state);
    if (lease == leases_.end()) {
      throw std::logic_error("TP2 model call outside a request");
    }
    member.sequence = lease->second;
  }
  // Recorded only once every member has a request, so a refused batch leaves
  // no trace in any of them.
  for (const auto& member : instruction.batch) {
    auto& request = requests_.at(member.sequence);
    DigestTpCall(request.digest, MemberAdvance(member));
    ++request.count;
    producers_[member.state] = member.sequence;
  }
  Sent sent;
  std::string error;
  if (!sink_->Send(0, instruction, &sent.index, &error)) {
    failure_ = error.empty() ? "send failed" : error;
    throw std::runtime_error("TP instruction send failed: " + failure_);
  }
  return sent;
}

void TpMirroredRunner::SendInvalidate(std::uint32_t state) const noexcept {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  try {
    lock.lock();
    if (!failure_.empty()) {
      return;
    }
    const TpInstruction instruction{.op = TpInstructionOp::kInvalidate,
                                    .state = state};
    // A reset of a leased state belongs to its request; the cache also resets
    // states between requests.
    std::uint64_t sequence = 0;
    if (const auto lease = leases_.find(state); lease != leases_.end()) {
      sequence = lease->second;
      auto& request = requests_.at(sequence);
      DigestTpCall(request.digest, instruction);
      ++request.count;
    }
    producers_.erase(state);
    std::string error;
    if (!sink_->Send(sequence, instruction, nullptr, &error)) {
      failure_ = error.empty() ? "reset send failed" : error;
    }
  } catch (...) {
    if (lock.owns_lock()) {
      try {
        failure_ = "reset send failed";
      } catch (...) {
      }
    }
  }
}

void TpMirroredRunner::Record(
    std::uint64_t sequence,
    const std::function<void(TpExecutionDigest&)>& add) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = requests_.find(sequence);
  if (found != requests_.end()) {
    add(found->second.digest);
  }
}

TextRunnerDescriptor TpMirroredRunner::Descriptor() const {
  auto descriptor = inner_->Descriptor();
  auto& capabilities = descriptor.capabilities;
  capabilities.batched_multi_token_decode = false;
  capabilities.batched_multi_token_decode_max_width = 0;
  descriptor.persistence.reset();
  return descriptor;
}

TextRunnerResourceClaim TpMirroredRunner::ResourceClaim() const {
  auto claim = inner_->ResourceClaim();
  claim.retained_snapshot_capacity_bytes = std::min(
      claim.retained_snapshot_capacity_bytes.value_or(0), snapshot_budget_);
  return claim;
}

std::vector<TextExecutionPlan> TpMirroredRunner::SupportedPlans() const {
  // Batched advances are mirrored as one instruction. Batched multi-token
  // decoding is not (see `Descriptor`), so the scheduler advances a batch of
  // multi-token requests one token each.
  return inner_->SupportedPlans();
}

std::vector<TextRunnerToken> TpMirroredRunner::Tokenize(
    std::string_view text) const {
  return inner_->Tokenize(text);
}

std::optional<std::vector<TextRunnerToken>> TpMirroredRunner::RenderAndTokenize(
    const ChatRequest& request) const {
  return inner_->RenderAndTokenize(request);
}

std::optional<TextPreparedPrompt> TpMirroredRunner::PreparePrompt(
    const ChatRequest& request) const {
  return inner_->PreparePrompt(request);
}

void TpMirroredRunner::SetPromptContext(
    TextRunnerState& state,
    std::shared_ptr<const TextPromptContext> context) const {
  auto& mirrored = Mirrored(state);
  const auto* request = dynamic_cast<const TpRequestContext*>(context.get());
  if (context != nullptr && request == nullptr) {
    throw std::invalid_argument("TP2 does not support prompt contexts");
  }
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (request == nullptr) {
      leases_.erase(mirrored.id());
    } else if (!requests_.contains(request->sequence)) {
      throw std::logic_error("TP request context names no open request");
    } else {
      leases_[mirrored.id()] = request->sequence;
    }
  }
  inner_->SetPromptContext(mirrored.inner(), nullptr);
}

TextGenerationBackend::InitialOutputState TpMirroredRunner::InitialOutputState(
    const ChatRequest& request) const {
  return inner_->InitialOutputState(request);
}

std::string TpMirroredRunner::Decode(
    std::span<const TextRunnerToken> tokens) const {
  return inner_->Decode(tokens);
}

std::unique_ptr<TextRunnerState> TpMirroredRunner::CreateState() const {
  auto inner = inner_->CreateState();
  if (inner == nullptr) {
    return nullptr;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  return std::make_unique<State>(this, std::move(inner), next_state_id_++);
}

std::optional<TextDecodeSelection> TpMirroredRunner::PreviewFirstToken(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->PreviewFirstToken(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::PreparePrefixReuse(
    TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  const auto sequence =
      CacheCall(mirrored.id(),
                {.op = TpInstructionOp::kReuse,
                 .state = mirrored.id(),
                 .prompt_size = ToWire(prefix.size(), "reuse prefix")},
                [&] { inner_->PreparePrefixReuse(mirrored.inner(), prefix); });
  Record(sequence, [&](TpExecutionDigest& digest) {
    digest.Add(inner_->CheckpointPosition(mirrored.inner()));
  });
}

void TpMirroredRunner::PrepareBatchExecution(TextRunnerState& state) const {
  inner_->PrepareBatchExecution(Mirrored(state).inner());
}

TextPrefillStep TpMirroredRunner::Prefill(
    TextRunnerState& state, std::span<const TextRunnerToken> prompt,
    std::size_t offset, std::size_t max_input_tokens) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  if (offset >= prompt.size() || max_input_tokens == 0) {
    // Nothing to prefill: runners reject the call before any model work, so
    // there is nothing for rank 1 to join.
    return inner_->Prefill(mirrored.inner(), prompt, offset, max_input_tokens);
  }
  // A budget beyond the remaining prompt means the same as the remaining
  // prompt, and fits the wire; both ranks prefill with the clamped value.
  const std::size_t count = std::min(max_input_tokens, prompt.size() - offset);
  const TpInstruction instruction{
      .op = TpInstructionOp::kPrefill,
      .state = mirrored.id(),
      .offset = ToWire(offset, "offset"),
      .count = ToWire(count, "budget"),
      .prompt_size = ToWire(prompt.size(), "prompt")};
  const auto sent = Send(mirrored.id(), instruction, prompt);
  TextPrefillStep step;
  try {
    CallScope scope(*this, sent.index);
    step = inner_->Prefill(mirrored.inner(), prompt, offset, count);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpPrefill(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

TextDecodeSelection TpMirroredRunner::SelectNext(
    TextRunnerState& state, sampling::SamplerState& sampler) const {
  return inner_->SelectNext(Mirrored(state).inner(), sampler);
}

void TpMirroredRunner::Advance(TextRunnerState& state,
                               TextRunnerToken token) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  const auto sent = Send(mirrored.id(), {.op = TpInstructionOp::kAdvance,
                                         .state = mirrored.id(),
                                         .token = TokenToWire(token)});
  try {
    CallScope scope(*this, sent.index);
    inner_->Advance(mirrored.inner(), token);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpAdvance(digest, inner_->CheckpointPosition(mirrored.inner()));
  });
}

TextDecodeStep TpMirroredRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  // A multi-token cycle is one instruction. It carries the sampler's draw
  // state, so rank 1, whose sampler otherwise matches, draws what rank 0 draws
  // and makes the same draft and acceptance decisions. A one-token budget runs
  // through the base implementation instead, which selects here, on rank 0
  // only, and mirrors the `Advance`; forwarding it to the wrapped runner would
  // let that `Advance` bypass this wrapper.
  if (max_tokens < 2 || !multi_token_decode_) {
    return TextModelRunner::DecodeStep(state, max_tokens, sampler);
  }
  auto& mirrored = Mirrored(state);
  const auto count = ToWire(max_tokens, "decode budget");
  const auto draw = sampler.SaveDrawState();
  if (draw.pending.has_value() &&
      *draw.pending > static_cast<sampling::TokenId>(
                          std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 pending draw exceeds the protocol range");
  }
  const auto sent = Send(
      mirrored.id(), {.op = TpInstructionOp::kDecode,
                      .state = mirrored.id(),
                      .count = count,
                      .rng = draw.rng,
                      .pending = draw.pending.has_value()
                                     ? static_cast<std::int32_t>(*draw.pending)
                                     : -1});
  TextDecodeStep step;
  try {
    CallScope scope(*this, sent.index);
    step = inner_->DecodeStep(mirrored.inner(), count, sampler);
    scope.End();
  } catch (...) {
    Record(sent.sequence, DigestTpFailure);
    throw;
  }
  Record(sent.sequence, [&](TpExecutionDigest& digest) {
    DigestTpDecode(digest, step, inner_->CheckpointPosition(mirrored.inner()));
  });
  return step;
}

std::vector<TextDecodeStep> TpMirroredRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  if (decodes.size() > 1) {
    throw std::invalid_argument("TP2 does not mirror batched decoding yet");
  }
  return TextModelRunner::DecodeBatch(decodes);
}

void TpMirroredRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  if (advances.size() < 2) {
    TextModelRunner::AdvanceBatch(advances);
    return;
  }
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  TpInstruction instruction{.op = TpInstructionOp::kAdvanceBatch};
  std::vector<std::exception_ptr> failures(advances.size());
  std::vector<TextRunnerAdvance> inner;
  inner.reserve(advances.size());
  for (std::size_t index = 0; index < advances.size(); ++index) {
    auto& mirrored = Mirrored(advances[index].state.get());
    instruction.batch.push_back(
        {.state = mirrored.id(), .token = TokenToWire(advances[index].token)});
    inner.push_back({.state = mirrored.inner(),
                     .token = advances[index].token,
                     .failure = &failures[index]});
  }
  const auto sent = SendBatch(instruction);
  try {
    CallScope scope(*this, sent.index);
    inner_->AdvanceBatch(inner);
    scope.End();
  } catch (...) {
    // The batch failed as a whole, including members it had advanced.
    const auto failure = std::current_exception();
    for (auto& member : failures) {
      if (!member) {
        member = failure;
      }
    }
  }
  std::exception_ptr unreported;
  for (std::size_t index = 0; index < advances.size(); ++index) {
    const auto sequence = instruction.batch[index].sequence;
    if (failures[index]) {
      Record(sequence, DigestTpFailure);
      if (advances[index].failure != nullptr) {
        *advances[index].failure = failures[index];
      } else if (!unreported) {
        unreported = failures[index];
      }
      continue;
    }
    Record(sequence, [&](TpExecutionDigest& digest) {
      DigestTpAdvance(digest,
                      inner_->CheckpointPosition(inner[index].state.get()));
    });
  }
  if (unreported) {
    std::rethrow_exception(unreported);
  }
}

std::size_t TpMirroredRunner::CheckpointPosition(
    const TextRunnerState& state) const {
  return inner_->CheckpointPosition(Mirrored(state).inner());
}

void TpMirroredRunner::PrepareCancellation(TextRunnerState& state) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  (void)CacheCall(
      mirrored.id(),
      {.op = TpInstructionOp::kCancelPrepare, .state = mirrored.id()},
      [&] { inner_->PrepareCancellation(mirrored.inner()); });
}

class TpMirroredRunner::SnapshotHandle final : public TextRunnerSnapshot {
public:
  SnapshotHandle(std::shared_ptr<const TpMirroredRunner> owner,
                 std::unique_ptr<TextRunnerSnapshot> inner, std::uint64_t id,
                 std::uint64_t producer)
      : owner(std::move(owner)),
        inner(std::move(inner)),
        id(id),
        producer(producer) {}
  ~SnapshotHandle() override { owner->Drop(id); }
  std::size_t PayloadBytes() const noexcept override {
    return inner->PayloadBytes();
  }
  std::shared_ptr<const TpMirroredRunner> owner;
  std::unique_ptr<TextRunnerSnapshot> inner;
  std::uint64_t id;
  /// The last request that changed the captured state, zero for none.
  std::uint64_t producer;
};

std::uint64_t TpMirroredRunner::CacheCall(
    std::uint32_t state, const TpInstruction& instruction,
    const std::function<void()>& local) const {
  std::uint64_t sequence = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto lease = leases_.find(state);
    if (lease == leases_.end() || !failure_.empty()) {
      throw std::logic_error("TP cache call outside a healthy request");
    }
    sequence = lease->second;
    auto& request = requests_.at(sequence);
    DigestTpCall(request.digest, instruction);
    ++request.count;
    if (instruction.op != TpInstructionOp::kSnapshot) {
      producers_[state] = sequence;
    }
  }
  try {
    sink_->Synchronize(sequence, instruction, local);
  } catch (...) {
    // A failed capture is a skipped snapshot on both ranks, not a failed call.
    if (instruction.op != TpInstructionOp::kSnapshot) {
      Record(sequence, DigestTpFailure);
    }
    throw;
  }
  return sequence;
}

void TpMirroredRunner::Drop(std::uint64_t id) const noexcept {
  try {
    const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty())
      return;
    // A drop only frees memory, so it belongs to no request.
    const TpInstruction instruction{.op = TpInstructionOp::kDrop,
                                    .snapshot_id = id};
    std::string error;
    if (!sink_->Send(0, instruction, nullptr, &error))
      failure_ = "snapshot drop failed: " + error;
  } catch (...) {
    // Match reset's deferred channel failure behavior; destructors must not
    // throw.
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      failure_ = "snapshot drop failed";
    } catch (...) {
    }
  }
}

std::size_t TpMirroredRunner::SnapshotPayloadBytes(
    const TextRunnerState& state) const {
  return inner_->SnapshotPayloadBytes(Mirrored(state).inner());
}

std::unique_ptr<TextRunnerSnapshot> TpMirroredRunner::Snapshot(
    const TextRunnerState& state) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const auto& mirrored = Mirrored(state);
  if (next_snapshot_id_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("TP snapshot ID exhausted");
  }
  const auto id = next_snapshot_id_++;
  std::uint64_t producer = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (const auto found = producers_.find(mirrored.id());
        found != producers_.end()) {
      producer = found->second;
    }
  }
  std::unique_ptr<TextRunnerSnapshot> snapshot;
  try {
    (void)CacheCall(
        mirrored.id(),
        {.op = TpInstructionOp::kSnapshot,
         .state = mirrored.id(),
         .snapshot_id = id},
        [&] {
          snapshot = inner_->Snapshot(mirrored.inner());
          if (!snapshot)
            throw std::runtime_error("snapshot capture returned null");
        });
    return std::make_unique<SnapshotHandle>(shared_from_this(),
                                            std::move(snapshot), id, producer);
  } catch (...) {
    Drop(id);
    throw;
  }
}

void TpMirroredRunner::RestoreOrFork(TextRunnerState& state,
                                     const TextRunnerSnapshot& snapshot) const {
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (!handle || handle->owner.get() != this)
    throw std::invalid_argument("foreign TP snapshot");
  auto& mirrored = Mirrored(state);
  const auto sequence = CacheCall(
      mirrored.id(),
      {.op = TpInstructionOp::kRestore,
       .state = mirrored.id(),
       .snapshot_id = handle->id},
      [&] { inner_->RestoreOrFork(mirrored.inner(), *handle->inner); });
  Record(sequence, [&](TpExecutionDigest& digest) {
    digest.Add(inner_->CheckpointPosition(mirrored.inner()));
  });
}

bool TpMirroredRunner::CanReuse(const TextRunnerSnapshot& snapshot) const {
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (handle == nullptr || handle->owner.get() != this) {
    return false;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  return Settled(handle->producer);
}

/// One open request on rank 1: what it needs to execute rank 0's calls for it
/// and to judge them.
struct TpExecutor::Request {
  Request(const TpControlCommand& begin, std::vector<TextRunnerToken> tokens,
          std::size_t states)
      : prompt(std::move(tokens)),
        greedy(begin.sampling.can_use_unmodified_argmax()),
        sampler(begin.sampling, prompt),
        own(states) {}

  void Note(std::string message) {
    if (failure.empty()) {
      failure = std::move(message);
    }
  }

  std::vector<TextRunnerToken> prompt;
  /// Under greedy decoding the logits are bit-identical on both ranks, so rank
  /// 1's own choice must equal every token rank 0 advances. Rank 0's token is
  /// what both ranks feed, so without this check a numerical divergence (the
  /// full-Q8 bug was one) would go unnoticed. The choice is made after each
  /// call while rank 0 is still selecting, so it costs rank 1 idle time only.
  bool greedy;
  /// Built as rank 0's pool builds the request's sampler, and fed the same
  /// accepted tokens, so a multi-token decode sees the same history; each
  /// `decode` instruction supplies rank 0's draw state.
  sampling::SamplerState sampler;
  std::vector<std::optional<OwnChoice>> own;
  TpExecutionDigest digest;
  std::uint64_t count{0};
  std::string failure;
};

TpExecutor::TpExecutor(std::shared_ptr<TextModelRunner> runner,
                       std::size_t state_count, std::size_t snapshot_budget,
                       std::shared_ptr<TpCallScope> scope)
    : runner_(std::move(runner)),
      scope_(std::move(scope)),
      snapshot_budget_(snapshot_budget) {
  if (runner_ == nullptr || state_count == 0) {
    throw std::invalid_argument("TP executor needs a runner and a state");
  }
  states_.reserve(state_count);
  for (std::size_t index = 0; index < state_count; ++index) {
    auto state = runner_->CreateState();
    if (state == nullptr) {
      throw std::runtime_error("TP executor state creation failed");
    }
    states_.push_back(std::move(state));
  }
}

TpExecutor::~TpExecutor() = default;

bool TpExecutor::TakeIndex(const TpControlCommand& command,
                           std::string* error) {
  if (command.instruction.index != next_index_) {
    SetError(error, "TP instruction " +
                        std::to_string(command.instruction.index) +
                        " arrived, expected " + std::to_string(next_index_));
    return false;
  }
  ++next_index_;
  return true;
}

TextRunnerState& TpExecutor::StateFor(std::uint32_t id) {
  if (id >= states_.size()) {
    throw std::out_of_range("TP instruction names state " + std::to_string(id) +
                            " of " + std::to_string(states_.size()));
  }
  return *states_[id];
}

TpExecutor::OwnChoice TpExecutor::ChooseGreedy(TextRunnerState& state) const {
  sampling::SamplerState greedy{sampling::SamplingConfig{}};
  const auto selection = runner_->SelectNext(state, greedy);
  return {.stop = selection.stop, .token = selection.token};
}

bool TpExecutor::BeginScope(std::uint64_t index, std::string* error) {
  std::string scope_error;
  if (scope_ != nullptr && !scope_->Begin(index, &scope_error)) {
    SetError(error, "TP collective scope bind failed: " + scope_error);
    return false;
  }
  return true;
}

bool TpExecutor::EndScope(std::uint64_t index, std::string* error) {
  std::string scope_error;
  if (scope_ != nullptr && !scope_->End(index, &scope_error)) {
    SetError(error, "TP collective scope release failed: " + scope_error);
    return false;
  }
  return true;
}

void TpExecutor::DropSnapshot(std::uint64_t id) {
  if (id == 0 || id > last_snapshot_id_)
    throw std::logic_error("unknown snapshot drop ID");
  const auto found = snapshots_.find(id);
  if (found != snapshots_.end()) {
    snapshot_bytes_ -= found->second->PayloadBytes();
    snapshots_.erase(found);
  }
}

bool TpExecutor::Execute(const TpControlCommand& command,
                         const Respond& respond, std::string* error) {
  if (command.kind == TpControlCommandKind::kSingle) {
    return Open(command, error);
  }
  if (!TakeIndex(command, error)) {
    return false;
  }
  const auto& instruction = command.instruction;
  if (instruction.op == TpInstructionOp::kAdvanceBatch) {
    return ExecuteBatch(instruction, error);
  }
  if (command.sequence == 0) {
    return ExecuteIdle(instruction, error);
  }
  const auto found = requests_.find(command.sequence);
  if (found == requests_.end()) {
    SetError(error, "TP instruction names request " +
                        std::to_string(command.sequence) +
                        ", which is not open");
    return false;
  }
  auto& request = *found->second;
  if (instruction.op != TpInstructionOp::kEnd) {
    return ExecuteCall(command.sequence, request, instruction, respond, error);
  }
  if (request.failure.empty() &&
      (instruction.count != request.count ||
       instruction.digest != request.digest.value())) {
    request.failure = "TP ranks ran different model calls: rank 0 made " +
                      std::to_string(instruction.count) + " with digest " +
                      Hex(instruction.digest) + ", rank 1 made " +
                      std::to_string(request.count) + " with digest " +
                      Hex(request.digest.value());
  }
  TpControlResponse response{.sequence = command.sequence,
                             .error = std::move(request.failure)};
  if (response.error.size() > (1U << 20)) {
    response.error.resize(1U << 20);
  }
  requests_.erase(found);
  return respond(response, error);
}

bool TpExecutor::Open(const TpControlCommand& begin, std::string* error) {
  if (begin.sequence == 0 || requests_.contains(begin.sequence)) {
    SetError(error, "TP request " + std::to_string(begin.sequence) +
                        " is unnamed or already open");
    return false;
  }
  std::vector<TextRunnerToken> prompt;
  prompt.reserve(begin.prompt_tokens.size());
  for (const auto token : begin.prompt_tokens) {
    if (token < 0) {
      SetError(error, "TP request prompt holds a negative token");
      return false;
    }
    prompt.push_back(static_cast<TextRunnerToken>(token));
  }
  requests_.emplace(
      begin.sequence,
      std::make_unique<Request>(begin, std::move(prompt), states_.size()));
  return true;
}

bool TpExecutor::ExecuteIdle(const TpInstruction& instruction,
                             std::string* error) {
  if (instruction.op != TpInstructionOp::kInvalidate &&
      instruction.op != TpInstructionOp::kDrop) {
    SetError(error, "TP worker received a model call outside a request");
    return false;
  }
  try {
    if (instruction.op == TpInstructionOp::kDrop) {
      DropSnapshot(instruction.snapshot_id);
    } else {
      StateFor(instruction.state).Invalidate();
    }
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
  return true;
}

bool TpExecutor::ExecuteCall(std::uint64_t sequence, Request& request,
                             const TpInstruction& instruction,
                             const Respond& respond, std::string* error) {
  ++request.count;
  const bool prompt_fits = instruction.prompt_size <= request.prompt.size();
  const auto prefill_prompt =
      instruction.op == TpInstructionOp::kPrefill && prompt_fits
          ? std::span<const TextRunnerToken>(request.prompt)
                .first(instruction.prompt_size)
          : std::span<const TextRunnerToken>{};
  DigestTpCall(request.digest, instruction, prefill_prompt);
  const bool forward = RunsForward(instruction.op);
  if (forward && !BeginScope(instruction.index, error)) {
    return false;
  }
  std::string call_error;
  try {
    auto& state = StateFor(instruction.state);
    auto& choice = request.own[instruction.state];
    switch (instruction.op) {
      case TpInstructionOp::kSnapshot: {
        if (instruction.snapshot_id <= last_snapshot_id_)
          throw std::logic_error("reused snapshot ID");
        last_snapshot_id_ = instruction.snapshot_id;
        const auto estimate = runner_->SnapshotPayloadBytes(state);
        if (estimate > snapshot_budget_ - snapshot_bytes_)
          throw std::runtime_error("worker snapshot budget exhausted");
        auto snapshot = runner_->Snapshot(state);
        if (!snapshot)
          throw std::runtime_error("snapshot capture returned null");
        const auto bytes = snapshot->PayloadBytes();
        if (bytes > snapshot_budget_ - snapshot_bytes_)
          throw std::runtime_error("worker snapshot exceeds reserved budget");
        snapshots_.emplace(instruction.snapshot_id, std::move(snapshot));
        snapshot_bytes_ += bytes;
        break;
      }
      case TpInstructionOp::kDrop:
        DropSnapshot(instruction.snapshot_id);
        break;
      case TpInstructionOp::kRestore: {
        choice.reset();
        const auto found = snapshots_.find(instruction.snapshot_id);
        if (found == snapshots_.end())
          throw std::logic_error("unknown snapshot ID");
        runner_->RestoreOrFork(state, *found->second);
        request.digest.Add(runner_->CheckpointPosition(state));
        break;
      }
      case TpInstructionOp::kReuse:
        choice.reset();
        if (!prompt_fits)
          throw std::invalid_argument("reuse exceeds request prompt");
        runner_->PreparePrefixReuse(
            state, std::span<const TextRunnerToken>(request.prompt)
                       .first(instruction.prompt_size));
        request.digest.Add(runner_->CheckpointPosition(state));
        if (request.greedy)
          choice = ChooseGreedy(state);
        break;
      case TpInstructionOp::kCancelPrepare:
        runner_->PrepareCancellation(state);
        break;
      case TpInstructionOp::kInvalidate:
        choice.reset();
        state.Invalidate();
        break;
      case TpInstructionOp::kPrefill: {
        if (!prompt_fits) {
          throw std::invalid_argument("prefill exceeds the request prompt");
        }
        choice.reset();
        const auto step = runner_->Prefill(
            state, prefill_prompt, instruction.offset, instruction.count);
        DigestTpPrefill(request.digest, step,
                        runner_->CheckpointPosition(state));
        if (request.greedy && step.decode_ready) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kAdvance: {
        const auto token = static_cast<TextRunnerToken>(instruction.token);
        if (request.greedy &&
            (!choice.has_value() || choice->stop || choice->token != token)) {
          request.Note("rank 1 " +
                       (choice.has_value()
                            ? (choice->stop ? std::string("stopped")
                                            : "selected token " +
                                                  std::to_string(choice->token))
                            : std::string("had no token")) +
                       " where rank 0 advanced token " + std::to_string(token) +
                       " at instruction " + std::to_string(instruction.index));
        }
        choice.reset();
        runner_->Advance(state, token);
        request.sampler.Accept(token);
        DigestTpAdvance(request.digest, runner_->CheckpointPosition(state));
        if (request.greedy) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kDecode: {
        choice.reset();
        request.sampler.RestoreDrawState(
            {.rng = instruction.rng,
             .pending =
                 instruction.pending >= 0
                     ? std::optional<sampling::TokenId>(
                           static_cast<sampling::TokenId>(instruction.pending))
                     : std::nullopt});
        const auto step =
            runner_->DecodeStep(state, instruction.count, request.sampler);
        for (const auto& selection : step.selections) {
          request.sampler.Accept(selection.token);
        }
        DigestTpDecode(request.digest, step,
                       runner_->CheckpointPosition(state));
        if (request.greedy) {
          choice = ChooseGreedy(state);
        }
        break;
      }
      case TpInstructionOp::kAdvanceBatch:
      case TpInstructionOp::kEnd:
      case TpInstructionOp::kNone:
        throw std::logic_error("TP instruction has no model call");
    }
  } catch (const std::exception& exception) {
    call_error = "rank 1 " + std::string(OpName(instruction.op)) +
                 " failed: " + exception.what();
    // A failed capture only means no snapshot: the acknowledgement makes
    // rank 0 skip it too and drop the ID, and neither state changed.
    if (instruction.op != TpInstructionOp::kSnapshot) {
      DigestTpFailure(request.digest);
      request.Note(call_error);
    }
  }
  if (forward && !EndScope(instruction.index, error)) {
    return false;
  }
  if (TpCacheAcknowledged(instruction.op)) {
    const TpControlResponse response{
        .instruction_index = instruction.index,
        .sequence = sequence,
        .error = call_error,
        .kind = TpControlResponseKind::kInstruction};
    return respond(response, error);
  }
  return true;
}

bool TpExecutor::ExecuteBatch(const TpInstruction& instruction,
                              std::string* error) {
  // Every member must name an open request and a state: a batch that does
  // not is a broken stream, not a failed call.
  std::vector<Request*> members;
  members.reserve(instruction.batch.size());
  for (const auto& member : instruction.batch) {
    const auto found = requests_.find(member.sequence);
    if (found == requests_.end() || member.state >= states_.size()) {
      SetError(error, "TP batch member names request " +
                          std::to_string(member.sequence) + " and state " +
                          std::to_string(member.state) +
                          ", which are not open");
      return false;
    }
    members.push_back(found->second.get());
  }
  std::vector<std::exception_ptr> failures(members.size());
  std::vector<TextRunnerAdvance> advances;
  advances.reserve(members.size());
  for (std::size_t index = 0; index < members.size(); ++index) {
    const auto& member = instruction.batch[index];
    auto& request = *members[index];
    ++request.count;
    DigestTpCall(request.digest, MemberAdvance(member));
    auto& choice = request.own[member.state];
    const auto token = static_cast<TextRunnerToken>(member.token);
    if (request.greedy &&
        (!choice.has_value() || choice->stop || choice->token != token)) {
      request.Note(
          "rank 1 " +
          (choice.has_value()
               ? (choice->stop
                      ? std::string("stopped")
                      : "selected token " + std::to_string(choice->token))
               : std::string("had no token")) +
          " where rank 0 advanced token " + std::to_string(token) +
          " in batched instruction " + std::to_string(instruction.index));
    }
    choice.reset();
    advances.push_back({.state = *states_[member.state],
                        .token = token,
                        .failure = &failures[index]});
  }
  if (!BeginScope(instruction.index, error)) {
    return false;
  }
  try {
    runner_->AdvanceBatch(advances);
  } catch (...) {
    // The batch failed as a whole, including members it had advanced.
    const auto failure = std::current_exception();
    for (auto& member : failures) {
      if (!member) {
        member = failure;
      }
    }
  }
  if (!EndScope(instruction.index, error)) {
    return false;
  }
  for (std::size_t index = 0; index < members.size(); ++index) {
    const auto& member = instruction.batch[index];
    auto& request = *members[index];
    auto& state = *states_[member.state];
    if (failures[index]) {
      DigestTpFailure(request.digest);
      std::string what = "unknown error";
      try {
        std::rethrow_exception(failures[index]);
      } catch (const std::exception& exception) {
        what = exception.what();
      } catch (...) {
      }
      request.Note("rank 1 batched advance failed: " + what);
      continue;
    }
    const auto token = static_cast<TextRunnerToken>(member.token);
    request.sampler.Accept(token);
    DigestTpAdvance(request.digest, runner_->CheckpointPosition(state));
    if (request.greedy) {
      try {
        request.own[member.state] = ChooseGreedy(state);
      } catch (const std::exception& exception) {
        request.Note(std::string("rank 1 greedy choice failed: ") +
                     exception.what());
      }
    }
  }
  return true;
}

}  // namespace gufo::server
