#include "src/cli/serve/tp_executor.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
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
    case TpInstructionOp::kPromptContext:
      return "prompt context";
    case TpInstructionOp::kPersist:
      return "persist";
    case TpInstructionOp::kRestoreDisk:
      return "disk restore";
    case TpInstructionOp::kDecodeBatch:
      return "batched decode";
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

/// A batch member is recorded in its request exactly like the advance or
/// decode it replaces, so both ranks agree on a request's calls however they
/// are batched.
[[nodiscard]] TpInstruction MemberCall(TpInstructionOp op,
                                       const TpBatchMember& member) {
  if (op == TpInstructionOp::kDecodeBatch) {
    return {.op = TpInstructionOp::kDecode,
            .state = member.state,
            .count = member.count,
            .rng = member.rng,
            .pending = member.pending};
  }
  return {.op = TpInstructionOp::kAdvance,
          .state = member.state,
          .token = member.token};
}

/// The sampler's draw state as a decode instruction carries it.
[[nodiscard]] std::int32_t PendingToWire(
    const sampling::SamplerState::DrawState& draw) {
  if (!draw.pending.has_value()) {
    return -1;
  }
  if (*draw.pending > static_cast<sampling::TokenId>(
                          std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("TP2 pending draw exceeds the protocol range");
  }
  return static_cast<std::int32_t>(*draw.pending);
}

[[nodiscard]] sampling::SamplerState::DrawState DrawFromWire(
    std::uint64_t rng, std::int32_t pending) {
  return {.rng = rng,
          .pending = pending >= 0 ? std::optional<sampling::TokenId>(
                                        static_cast<sampling::TokenId>(pending))
                                  : std::nullopt};
}

/// Both ranks record which prompt context a state was bound to, so a request
/// whose images reached rank 1 differently fails the agreement check.
void DigestTpContext(TpExecutionDigest& digest,
                     const std::shared_ptr<const TextPromptContext>& context) {
  const std::span<const std::uint8_t> identity =
      context != nullptr
          ? std::span<const std::uint8_t>(context->cache_identity)
          : std::span<const std::uint8_t>{};
  digest.Add(identity.size());
  for (const auto byte : identity) {
    digest.Add(byte);
  }
}

// Rank 0's half of a disk-cache payload starts with this header: magic,
// format, reserved, rank 1's file key, the writing process and the request
// that produced the snapshot.
constexpr std::array<std::uint8_t, 8> kPersistMagic = {'G', 'U', 'F', 'O',
                                                       'T', 'P', '2', 'R'};
constexpr std::uint32_t kPersistFormat = 1;
constexpr std::size_t kPersistHeaderBytes = 8 + 4 + 4 + 8 + 8 + 8;

void PutLe(std::uint8_t* out, std::uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i)
    out[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

std::uint64_t GetLe(const std::uint8_t* in, int bytes) {
  std::uint64_t value = 0;
  for (int i = 0; i < bytes; ++i)
    value |= std::uint64_t{in[i]} << (8 * i);
  return value;
}

std::uint64_t RandomNonzero() {
  std::random_device device;
  std::uint64_t value = 0;
  while (value == 0)
    value = (std::uint64_t{device()} << 32) ^ device();
  return value;
}

/// Forward calls exchange partials, so they run inside a collective scope.
[[nodiscard]] bool RunsForward(TpInstructionOp op) {
  return op == TpInstructionOp::kPrefill || op == TpInstructionOp::kAdvance ||
         op == TpInstructionOp::kDecode ||
         op == TpInstructionOp::kAdvanceBatch ||
         op == TpInstructionOp::kDecodeBatch;
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
  // A capture carries the request's call count and digest so far, which the
  // digest cannot include.
  digest.Add(instruction.op == TpInstructionOp::kSnapshot ? 0
                                                          : instruction.count);
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
         op == TpInstructionOp::kReuse ||
         op == TpInstructionOp::kCancelPrepare ||
         op == TpInstructionOp::kRestoreDisk;
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
  nonce_ = RandomNonzero();
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
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    unsettled_.erase(sequence);
    dependencies_.erase(sequence);
    if (!agreed) {
      rejected_.insert(sequence);
    }
  }
  settled_.notify_all();
}

bool TpMirroredRunner::AwaitDependencies(std::uint64_t sequence) const {
  // A dependency has released its state, so its verdict follows within one
  // round trip once rank 1 reaches its end.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    const auto found = dependencies_.find(sequence);
    if (found == dependencies_.end()) {
      return true;
    }
    bool pending = false;
    for (const auto producer : found->second) {
      if (rejected_.contains(producer)) {
        return false;
      }
      pending = pending || unsettled_.contains(producer);
    }
    if (!pending) {
      return true;
    }
    if (settled_.wait_until(lock, deadline) == std::cv_status::timeout) {
      return false;
    }
  }
}

std::string TpMirroredRunner::Failure() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return failure_;
}

bool TpMirroredRunner::CanReuse(const TextRunnerState& state) const {
  // A released state is reused before its producer's verdict arrives, as on
  // one host; the reusing request then depends on that verdict.
  const auto id = Mirrored(state).id();
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto found = producers_.find(id);
  return found == producers_.end() || !rejected_.contains(found->second);
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
    DigestTpCall(request.digest, MemberCall(instruction.op, member));
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
  if (descriptor.persistence.has_value()) {
    // Rank 0's files hold its half and name rank 1's: neither a one-host
    // server nor a lone rank can restore them.
    constexpr std::string_view kMarker = "tp2=rank0-mirror-v1\n";
    auto& identity = descriptor.persistence->compatibility_identity;
    identity.insert(identity.end(), kMarker.begin(), kMarker.end());
  }
  return descriptor;
}

TextRunnerResourceClaim TpMirroredRunner::ResourceClaim() const {
  auto claim = inner_->ResourceClaim();
  claim.retained_snapshot_capacity_bytes = std::min(
      claim.retained_snapshot_capacity_bytes.value_or(0), snapshot_budget_);
  return claim;
}

std::vector<TextExecutionPlan> TpMirroredRunner::SupportedPlans() const {
  // Batched advances and decodes are each mirrored as one instruction.
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
    throw std::invalid_argument("TP2 prompt context names no request");
  }
  const auto images = request != nullptr ? request->images : nullptr;
  // Binding a context can reset the model state, so rank 1 binds it too, at
  // the same point in the call order.
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty()) {
      throw std::runtime_error("TP instruction channel failed: " + failure_);
    }
    std::uint64_t sequence = 0;
    if (request == nullptr) {
      leases_.erase(mirrored.id());
    } else if (!requests_.contains(request->sequence)) {
      throw std::logic_error("TP request context names no open request");
    } else {
      sequence = request->sequence;
      leases_[mirrored.id()] = sequence;
      // The state may hold what a request rank 1 has not judged yet.
      const auto producer = producers_.find(mirrored.id());
      if (producer != producers_.end() && producer->second != sequence &&
          unsettled_.contains(producer->second)) {
        dependencies_[sequence].insert(producer->second);
      }
    }
    const TpInstruction instruction{.op = TpInstructionOp::kPromptContext,
                                    .state = mirrored.id()};
    if (sequence != 0) {
      auto& open = requests_.at(sequence);
      DigestTpCall(open.digest, instruction);
      DigestTpContext(open.digest, images);
      ++open.count;
    }
    std::string error;
    if (!sink_->Send(sequence, instruction, nullptr, &error)) {
      failure_ = error.empty() ? "send failed" : error;
      throw std::runtime_error("TP instruction send failed: " + failure_);
    }
  }
  inner_->SetPromptContext(mirrored.inner(), images);
}

std::vector<std::uint8_t> TpMirroredRunner::EncodePromptContext(
    const TextPromptContext& context) const {
  return inner_->EncodePromptContext(context);
}

std::shared_ptr<const TextPromptContext> TpMirroredRunner::DecodePromptContext(
    std::span<const std::uint8_t> bytes) const {
  return inner_->DecodePromptContext(bytes);
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
  const auto sent = Send(mirrored.id(), {.op = TpInstructionOp::kDecode,
                                         .state = mirrored.id(),
                                         .count = count,
                                         .rng = draw.rng,
                                         .pending = PendingToWire(draw)});
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
  if (decodes.size() < 2 || !multi_token_decode_) {
    return TextModelRunner::DecodeBatch(decodes);
  }
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  // One instruction carries every member's budget and draw state, like the
  // decode each replaces, and the batch's draft count: rank 0 chooses it from
  // its own cycle timings, which rank 1 cannot reproduce.
  TpInstruction instruction{.op = TpInstructionOp::kDecodeBatch};
  std::vector<TextRunnerDecode> inner;
  inner.reserve(decodes.size());
  for (const auto& decode : decodes) {
    auto& mirrored = Mirrored(decode.state.get());
    const auto draw = decode.sampler.get().SaveDrawState();
    instruction.batch.push_back(
        {.state = mirrored.id(),
         .count = ToWire(decode.max_tokens, "decode budget"),
         .rng = draw.rng,
         .pending = PendingToWire(draw)});
    inner.push_back({.state = mirrored.inner(),
                     .max_tokens = decode.max_tokens,
                     .sampler = decode.sampler});
  }
  const auto plan = inner_->PlanDecodeBatch(inner);
  instruction.batch_drafts =
      plan.has_value() ? ToWire(std::size_t{*plan} + 1, "draft count") : 0;
  const auto sent = SendBatch(instruction);
  std::vector<TextDecodeStep> steps;
  try {
    CallScope scope(*this, sent.index);
    steps = inner_->DecodeBatchPlanned(inner, plan);
    scope.End();
    if (steps.size() != decodes.size()) {
      throw std::runtime_error(
          "text runner returned an invalid decode batch size");
    }
  } catch (...) {
    // The batch failed as a whole, including members it had decoded.
    const auto failure = std::current_exception();
    steps.assign(decodes.size(), TextDecodeStep{});
    for (auto& step : steps) {
      step.failure = failure;
    }
  }
  for (std::size_t index = 0; index < decodes.size(); ++index) {
    const auto sequence = instruction.batch[index].sequence;
    if (steps[index].failure) {
      Record(sequence, DigestTpFailure);
      continue;
    }
    Record(sequence, [&](TpExecutionDigest& digest) {
      DigestTpDecode(digest, steps[index],
                     inner_->CheckpointPosition(inner[index].state.get()));
    });
  }
  return steps;
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
  TpInstruction sent;
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
    sent = instruction;
    if (instruction.op == TpInstructionOp::kSnapshot) {
      // Rank 1 compares the request's calls so far before it captures, so a
      // snapshot is known to hold the same state on both ranks.
      sent.count = ToWire(request.count, "count");
      sent.digest = request.digest.value();
    }
  }
  try {
    sink_->Synchronize(sequence, sent, local);
  } catch (...) {
    // A failed capture is a skipped snapshot on both ranks, and a failed disk
    // restore a cache miss (the cache resets the state), not a failed call.
    if (instruction.op != TpInstructionOp::kSnapshot &&
        instruction.op != TpInstructionOp::kRestoreDisk) {
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
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto lease = leases_.find(mirrored.id());
    if (lease != leases_.end() && handle->producer != 0 &&
        handle->producer != lease->second &&
        unsettled_.contains(handle->producer)) {
      dependencies_[lease->second].insert(handle->producer);
    }
  }
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
  // Rank 1 compared both ranks' calls before capturing, so a snapshot is
  // reusable as soon as it exists, until its producer is rejected; a request
  // that restores it earlier depends on that verdict.
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (handle == nullptr || handle->owner.get() != this) {
    return false;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  return !rejected_.contains(handle->producer);
}

std::size_t TpMirroredRunner::PersistentSnapshotPayloadBytes(
    const TextRunnerSnapshot& snapshot) const {
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (handle == nullptr || handle->owner.get() != this)
    throw std::invalid_argument("foreign TP snapshot");
  return kPersistHeaderBytes +
         inner_->PersistentSnapshotPayloadBytes(*handle->inner);
}

std::vector<std::uint8_t> TpMirroredRunner::BeginPersist(
    const TextRunnerSnapshot& snapshot) const {
  const auto* handle = dynamic_cast<const SnapshotHandle*>(&snapshot);
  if (handle == nullptr || handle->owner.get() != this)
    throw std::invalid_argument("foreign TP snapshot");
  const auto key = RandomNonzero();
  {
    const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty())
      throw std::runtime_error("TP instruction channel failed: " + failure_);
    // What rank 1 rejected never reaches the disk. A snapshot of a request
    // still running is written, and checked again if restored in this
    // process.
    if (rejected_.contains(handle->producer))
      throw std::runtime_error("TP snapshot of a rejected request");
    // Rank 1 persists its half in the background: the snapshot outlives the
    // write there, and a failed write only makes a later restore a miss.
    const TpInstruction instruction{.op = TpInstructionOp::kPersist,
                                    .snapshot_id = handle->id,
                                    .file_key = key};
    std::string error;
    if (!sink_->Send(0, instruction, nullptr, &error)) {
      failure_ = "snapshot persist send failed: " + error;
      throw std::runtime_error(failure_);
    }
  }
  std::vector<std::uint8_t> header(kPersistHeaderBytes);
  std::copy(kPersistMagic.begin(), kPersistMagic.end(), header.begin());
  PutLe(header.data() + 8, kPersistFormat, 4);
  PutLe(header.data() + 16, key, 8);
  PutLe(header.data() + 24, nonce_, 8);
  PutLe(header.data() + 32, handle->producer, 8);
  return header;
}

std::size_t TpMirroredRunner::SerializePersistentSnapshot(
    const TextRunnerSnapshot& snapshot,
    std::span<std::uint8_t> destination) const {
  const auto bytes = PersistentSnapshotPayloadBytes(snapshot);
  if (destination.size() < bytes)
    throw std::invalid_argument("TP persistent snapshot destination too small");
  const auto header = BeginPersist(snapshot);
  std::copy(header.begin(), header.end(), destination.begin());
  const auto& handle = dynamic_cast<const SnapshotHandle&>(snapshot);
  return header.size() + inner_->SerializePersistentSnapshot(
                             *handle.inner, destination.subspan(header.size()));
}

void TpMirroredRunner::StreamPersistentSnapshot(
    const TextRunnerSnapshot& snapshot, const SnapshotSink& sink) const {
  const auto header = BeginPersist(snapshot);
  sink(header);
  const auto& handle = dynamic_cast<const SnapshotHandle&>(snapshot);
  inner_->StreamPersistentSnapshot(*handle.inner, sink);
}

void TpMirroredRunner::RestorePersistentSnapshot(
    TextRunnerState& state, std::span<const std::uint8_t> payload) const {
  if (payload.size() < kPersistHeaderBytes ||
      !std::equal(kPersistMagic.begin(), kPersistMagic.end(),
                  payload.begin()) ||
      GetLe(payload.data() + 8, 4) != kPersistFormat)
    throw std::invalid_argument("not a TP2 disk-cache payload");
  const auto key = GetLe(payload.data() + 16, 8);
  const auto nonce = GetLe(payload.data() + 24, 8);
  const auto producer = GetLe(payload.data() + 32, 8);
  if (key == 0)
    throw std::invalid_argument("TP2 disk-cache payload names no file");
  const std::lock_guard<std::recursive_mutex> call_lock(call_mutex_);
  auto& mirrored = Mirrored(state);
  if (nonce == nonce_) {
    // Written by this process: as for a snapshot in memory, a rejected
    // producer makes the file a miss, and one without a verdict yet makes
    // the restoring request depend on it.
    const std::lock_guard<std::mutex> lock(mutex_);
    if (rejected_.contains(producer))
      throw std::runtime_error("TP snapshot of a request rank 1 rejected");
    const auto lease = leases_.find(mirrored.id());
    if (lease != leases_.end() && producer != 0 && producer != lease->second &&
        unsettled_.contains(producer)) {
      dependencies_[lease->second].insert(producer);
    }
  }
  (void)CacheCall(mirrored.id(),
                  {.op = TpInstructionOp::kRestoreDisk,
                   .state = mirrored.id(),
                   .file_key = key},
                  [&] {
                    inner_->RestorePersistentSnapshot(
                        mirrored.inner(), payload.subspan(kPersistHeaderBytes));
                  });
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
  /// The request's prompt context (its images), rebuilt from `kSingle`.
  std::shared_ptr<const TextPromptContext> images;
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
                       std::shared_ptr<TpCallScope> scope,
                       std::shared_ptr<TpDiskStore> disk)
    : runner_(std::move(runner)),
      scope_(std::move(scope)),
      disk_(std::move(disk)),
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
  if (instruction.op == TpInstructionOp::kAdvanceBatch ||
      instruction.op == TpInstructionOp::kDecodeBatch) {
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
  auto request =
      std::make_unique<Request>(begin, std::move(prompt), states_.size());
  if (!begin.prompt_context.empty()) {
    // A context rank 1 cannot rebuild fails the request, not the pair: its
    // calls still run, and the digest reports the difference.
    try {
      request->images = runner_->DecodePromptContext(begin.prompt_context);
    } catch (const std::exception& exception) {
      request->Note(std::string("rank 1 prompt context is invalid: ") +
                    exception.what());
    }
  }
  requests_.emplace(begin.sequence, std::move(request));
  return true;
}

bool TpExecutor::ExecuteIdle(const TpInstruction& instruction,
                             std::string* error) {
  if (instruction.op != TpInstructionOp::kInvalidate &&
      instruction.op != TpInstructionOp::kDrop &&
      instruction.op != TpInstructionOp::kPersist &&
      instruction.op != TpInstructionOp::kPromptContext) {
    SetError(error, "TP worker received a model call outside a request");
    return false;
  }
  try {
    if (instruction.op == TpInstructionOp::kDrop) {
      DropSnapshot(instruction.snapshot_id);
    } else if (instruction.op == TpInstructionOp::kPersist) {
      // Best effort, like rank 0's write: a snapshot rank 1 skipped or a
      // write it cannot queue only makes a later restore of the file a miss.
      const auto found = snapshots_.find(instruction.snapshot_id);
      if (disk_ != nullptr && found != snapshots_.end()) {
        (void)disk_->PersistAsync(instruction.file_key, runner_, found->second);
      }
    } else if (instruction.op == TpInstructionOp::kPromptContext) {
      runner_->SetPromptContext(StateFor(instruction.state), nullptr);
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
        // Rank 0 sent the request's calls so far: capture only a state both
        // ranks reached the same way, so the snapshot is reusable at once.
        if (instruction.count != request.count ||
            instruction.digest != request.digest.value() ||
            !request.failure.empty())
          throw std::runtime_error("the ranks diverged before this capture");
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
      case TpInstructionOp::kRestoreDisk:
        choice.reset();
        if (disk_ == nullptr)
          throw std::runtime_error("rank 1 has no disk cache");
        disk_->Restore(instruction.file_key, *runner_, state);
        break;
      case TpInstructionOp::kPromptContext:
        DigestTpContext(request.digest, request.images);
        runner_->SetPromptContext(state, request.images);
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
            DrawFromWire(instruction.rng, instruction.pending));
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
      case TpInstructionOp::kDecodeBatch:
      case TpInstructionOp::kPersist:
      case TpInstructionOp::kEnd:
      case TpInstructionOp::kNone:
        throw std::logic_error("TP instruction has no model call");
    }
  } catch (const std::exception& exception) {
    call_error = "rank 1 " + std::string(OpName(instruction.op)) +
                 " failed: " + exception.what();
    // A failed capture only means no snapshot: the acknowledgement makes
    // rank 0 skip it too and drop the ID, and neither state changed. A failed
    // disk restore is a cache miss: rank 0's cache resets the state on both
    // ranks and prefills instead.
    if (instruction.op != TpInstructionOp::kSnapshot &&
        instruction.op != TpInstructionOp::kRestoreDisk) {
      DigestTpFailure(request.digest);
      request.Note(call_error);
    }
  }
  if (forward && !EndScope(instruction.index, error)) {
    if (!call_error.empty() && error != nullptr) {
      *error = call_error + "; " + *error;
    }
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
  const bool decode = instruction.op == TpInstructionOp::kDecodeBatch;
  std::vector<std::exception_ptr> failures(members.size());
  std::vector<TextRunnerAdvance> advances;
  std::vector<TextRunnerDecode> decodes;
  if (decode) {
    decodes.reserve(members.size());
  } else {
    advances.reserve(members.size());
  }
  for (std::size_t index = 0; index < members.size(); ++index) {
    const auto& member = instruction.batch[index];
    auto& request = *members[index];
    ++request.count;
    DigestTpCall(request.digest, MemberCall(instruction.op, member));
    auto& choice = request.own[member.state];
    if (decode) {
      choice.reset();
      request.sampler.RestoreDrawState(
          DrawFromWire(member.rng, member.pending));
      decodes.push_back({.state = *states_[member.state],
                         .max_tokens = member.count,
                         .sampler = request.sampler});
      continue;
    }
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
  std::vector<TextDecodeStep> steps;
  try {
    if (decode) {
      // Rank 0's draft count, which it chose from its own timings.
      const auto plan =
          instruction.batch_drafts != 0
              ? std::optional<std::uint32_t>(instruction.batch_drafts - 1)
              : std::nullopt;
      steps = runner_->DecodeBatchPlanned(decodes, plan);
      if (steps.size() != members.size()) {
        throw std::runtime_error(
            "text runner returned an invalid decode batch size");
      }
      for (std::size_t index = 0; index < members.size(); ++index) {
        failures[index] = steps[index].failure;
      }
    } else {
      runner_->AdvanceBatch(advances);
    }
  } catch (...) {
    // The batch failed as a whole, including members it had run.
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
      request.Note("rank 1 " + std::string(OpName(instruction.op)) +
                   " failed: " + what);
      continue;
    }
    if (decode) {
      for (const auto& selection : steps[index].selections) {
        request.sampler.Accept(selection.token);
      }
      DigestTpDecode(request.digest, steps[index],
                     runner_->CheckpointPosition(state));
    } else {
      const auto token = static_cast<TextRunnerToken>(member.token);
      request.sampler.Accept(token);
      DigestTpAdvance(request.digest, runner_->CheckpointPosition(state));
    }
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
