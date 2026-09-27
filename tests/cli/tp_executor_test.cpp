// Drives the real scheduler and runner pool on rank 0 through
// `TpMirroredRunner`, and a `TpExecutor` on rank 1, over a loopback control
// channel. A toy model records every model call each rank makes; the pair is
// correct when both ranks record the same calls with the same results, for
// greedy, sampled, stopped, cancelled, multi-token, concurrent and batched
// requests, when rank 1 reports the divergences it is meant to catch, and when
// a request that reused what rank 1 then rejected fails too.

#include "src/cli/serve/tp_executor.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/tp_control.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationBackend;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TpCallScope;
using gufo::server::TpControlChannel;
using gufo::server::TpControlCommand;
using gufo::server::TpControlCommandKind;
using gufo::server::TpControlConfig;
using gufo::server::TpControlInstructionSink;
using gufo::server::TpControlResponse;
using gufo::server::TpExecutor;
using gufo::server::TpInstruction;
using gufo::server::TpInstructionOp;
using gufo::server::TpInstructionSink;
using gufo::server::TpMirroredRunner;
using gufo::server::TpRequestContext;
using FinishReason = TextGenerationBackend::FinishReason;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

std::uint16_t FreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
  const auto port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

constexpr std::size_t kVocab = 32;
constexpr TextRunnerToken kEos = 0;

/// One model call and what it produced, as a comparable line.
using CallLog = std::vector<std::string>;

struct ToyOptions {
  bool mtp{false};
  bool cache{false};
  bool fail_capture_once{false};
  bool fail_restore_once{false};
  std::size_t cache_budget{4096};
  std::function<void()> capture_hook;
  /// Runs before every prefill chunk.
  std::function<void()> prefill_hook;
  std::size_t prefill_chunk{3};
  /// Widest batched plan offered; one offers only serial steps.
  std::size_t width{1};
  /// Rank-1 faults: shift the greedy choice once, at this sequence length.
  std::optional<std::size_t> diverge_at;
  /// Rank-1 fault: report one extra draft in the next multi-token step.
  bool extra_draft_once{false};
  /// Rank-1 fault: refuse to rebuild a request's prompt context.
  bool fail_context_decode{false};
};

/// A toy "image": it shifts every logit of the sequence that sees it, and its
/// shade is its cache identity.
struct ToyContext final : gufo::server::TextPromptContext {
  explicit ToyContext(std::uint8_t shade) : shade(shade) {
    cache_identity = {shade};
  }
  std::uint8_t shade;
};

class ToySnapshot final : public gufo::server::TextRunnerSnapshot {
public:
  explicit ToySnapshot(std::vector<TextRunnerToken> tokens)
      : tokens(std::move(tokens)) {}
  std::size_t PayloadBytes() const noexcept override {
    return tokens.size() * sizeof(TextRunnerToken);
  }
  std::vector<TextRunnerToken> tokens;
};

class ToyRunner;

class ToyState final : public TextRunnerState {
public:
  ToyState(const ToyRunner* owner, std::size_t id) : owner_(owner), id_(id) {}
  void Invalidate() noexcept override;
  void SetCancellationCheck(const CancellationCheck&) override;

  std::vector<TextRunnerToken> tokens;
  /// The bound context's shade, zero without one.
  std::uint8_t shade{0};
  std::size_t id() const { return id_; }

private:
  const ToyRunner* owner_;
  std::size_t id_;
};

/// A deterministic model whose next token depends on the whole sequence, so
/// any divergence between the ranks changes later tokens too.
class ToyRunner final : public TextModelRunner {
public:
  explicit ToyRunner(ToyOptions options) : options_(options) {}

  CallLog Log() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return log_;
  }
  void Record(std::string line) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    log_.push_back(std::move(line));
  }
  std::size_t prefill_calls() const { return prefill_calls_.load(); }
  std::size_t cancellation_checks() const { return cancellation_checks_; }
  void CountCancellationCheck() const { ++cancellation_checks_; }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "toy",
            .state_abi = "toy-v1",
            .max_context = 256,
            .capabilities = TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = options_.cache,
                .fork = options_.cache,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = options_.mtp,
                .prefix_reuse = options_.cache,
            }};
  }
  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 8 * 64,
            .per_request_state_bytes = 64,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes =
                options_.cache ? options_.cache_budget : 0U,
            .requires_device_runtime_lock = true};
  }
  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= options_.width; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }
  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const auto token : tokens) {
      text += Piece(token);
    }
    return text;
  }
  static std::string Piece(TextRunnerToken token) {
    return " t" + std::to_string(token) + ";";
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<ToyState>(this, next_state_++);
  }
  /// Like Flash-Next's vision prompt: a different context resets a state that
  /// holds a sequence.
  void SetPromptContext(TextRunnerState& state,
                        std::shared_ptr<const gufo::server::TextPromptContext>
                            context) const override {
    const auto* toy = dynamic_cast<const ToyContext*>(context.get());
    if (context != nullptr && toy == nullptr) {
      throw std::invalid_argument("toy prompt context expected");
    }
    auto& target = Toy(state);
    const std::uint8_t shade = toy != nullptr ? toy->shade : 0;
    if (shade == 0 && target.shade == 0) {
      return;
    }
    if (!target.tokens.empty() && shade != target.shade) {
      target.tokens.clear();
    }
    target.shade = shade;
    Record("context s" + std::to_string(target.id()) + " " +
           std::to_string(shade));
  }
  [[nodiscard]] std::vector<std::uint8_t> EncodePromptContext(
      const gufo::server::TextPromptContext& context) const override {
    return {dynamic_cast<const ToyContext&>(context).shade};
  }
  [[nodiscard]] std::shared_ptr<const gufo::server::TextPromptContext>
  DecodePromptContext(std::span<const std::uint8_t> bytes) const override {
    if (options_.fail_context_decode || bytes.size() != 1) {
      throw std::invalid_argument("injected prompt context failure");
    }
    return std::make_shared<ToyContext>(bytes[0]);
  }
  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    Require(std::ranges::equal(Toy(state).tokens, prefix),
            "reused prefix equals state");
    Record("reuse " + std::to_string(prefix.size()));
  }
  std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return CheckpointPosition(state) * sizeof(TextRunnerToken);
  }
  std::unique_ptr<gufo::server::TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    if (options_.capture_hook)
      options_.capture_hook();
    if (options_.fail_capture_once) {
      options_.fail_capture_once = false;
      throw std::runtime_error("injected capture failure");
    }
    Record("snapshot " + std::to_string(CheckpointPosition(state)));
    return std::make_unique<ToySnapshot>(
        dynamic_cast<const ToyState&>(state).tokens);
  }
  void RestoreOrFork(
      TextRunnerState& state,
      const gufo::server::TextRunnerSnapshot& snapshot) const override {
    if (options_.fail_restore_once) {
      options_.fail_restore_once = false;
      throw std::runtime_error("injected restore failure");
    }
    Toy(state).tokens = dynamic_cast<const ToySnapshot&>(snapshot).tokens;
    Record("restore " + std::to_string(CheckpointPosition(state)));
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    if (options_.prefill_hook)
      options_.prefill_hook();
    auto& toy = Toy(state);
    if (offset != toy.tokens.size() || offset >= prompt.size()) {
      throw std::logic_error("toy prefill offset mismatch");
    }
    const auto consumed = std::min(
        {max_input_tokens, prompt.size() - offset, options_.prefill_chunk});
    toy.tokens.insert(toy.tokens.end(), prompt.begin() + offset,
                      prompt.begin() + offset + consumed);
    ++prefill_calls_;
    Record("prefill s" + std::to_string(toy.id()) + " @" +
           std::to_string(offset) + " +" + std::to_string(consumed) + "/" +
           std::to_string(prompt.size()));
    return {.consumed_tokens = consumed,
            .decode_ready = toy.tokens.size() == prompt.size()};
  }
  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    auto& toy = Toy(state);
    const auto logits = Logits(toy.tokens, toy.shade);
    const auto token = static_cast<TextRunnerToken>(sampler.Sample(logits));
    if (token == kEos) {
      return {.stop = true};
    }
    return {.stop = false, .token = token, .piece = Piece(token)};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& toy = Toy(state);
    toy.tokens.push_back(token);
    Record("advance s" + std::to_string(toy.id()) + " " +
           std::to_string(token));
  }
  void AdvanceBatch(std::span<const gufo::server::TextRunnerAdvance> advances)
      const override {
    if (advances.size() > 1) {
      Record("batch " + std::to_string(advances.size()));
    }
    TextModelRunner::AdvanceBatch(advances);
  }
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (!options_.mtp || max_tokens < 2) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& toy = Toy(state);
    TextDecodeStep step;
    std::string produced;
    // Like Flash-Next: draw on a working copy, leave a deferred draw for the
    // next call when sampling at random, and publish the draw state back.
    gufo::sampling::SamplerState working = sampler;
    for (std::size_t i = 0; i < std::min<std::size_t>(max_tokens, 3); ++i) {
      const auto token = static_cast<TextRunnerToken>(
          working.Sample(Logits(toy.tokens, toy.shade)));
      if (token == kEos) {
        step.stop = true;
        break;
      }
      toy.tokens.push_back(token);
      step.selections.push_back({.token = token, .piece = Piece(token)});
      produced += " " + std::to_string(token);
    }
    if (!step.stop && sampler.config().uses_random_sampling()) {
      working.DeferSample(working.Sample(Logits(toy.tokens, toy.shade)));
      produced += " deferred";
    }
    sampler.CopyDrawStateFrom(working);
    step.draft_tokens = 3;
    step.draft_accepted_tokens =
        step.selections.empty() ? 0 : step.selections.size() - 1;
    if (options_.extra_draft_once) {
      options_.extra_draft_once = false;
      ++step.draft_tokens;
    }
    Record("decode s" + std::to_string(toy.id()) + " max" +
           std::to_string(max_tokens) + produced);
    return step;
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return dynamic_cast<const ToyState&>(state).tokens.size();
  }

private:
  static ToyState& Toy(TextRunnerState& state) {
    return dynamic_cast<ToyState&>(state);
  }
  std::vector<float> Logits(const std::vector<TextRunnerToken>& tokens,
                            std::uint8_t shade) const {
    std::size_t sum = std::size_t{shade} * 5;
    for (const auto token : tokens) {
      sum += token;
    }
    // Never EOS unless the sequence is long, so budgets decide most lengths.
    std::size_t peak = (sum * 7 + tokens.size() * 3) % (kVocab - 1) + 1;
    if (tokens.size() >= 40) {
      peak = kEos;
    }
    if (options_.diverge_at == tokens.size()) {
      options_.diverge_at.reset();
      peak = peak % (kVocab - 1) + 1;
    }
    std::vector<float> logits(kVocab);
    for (std::size_t i = 0; i < kVocab; ++i) {
      logits[i] = -0.25F * static_cast<float>(i > peak ? i - peak : peak - i);
    }
    return logits;
  }

  mutable ToyOptions options_;
  mutable std::mutex mutex_;
  mutable CallLog log_;
  mutable std::size_t next_state_{0};
  mutable std::atomic<std::size_t> prefill_calls_{0};
  mutable std::atomic<std::size_t> cancellation_checks_{0};
};

void ToyState::Invalidate() noexcept {
  tokens.clear();
  try {
    owner_->Record("reset s" + std::to_string(id_));
  } catch (...) {
  }
}

void ToyState::SetCancellationCheck(const CancellationCheck&) {
  owner_->CountCancellationCheck();
}

/// Rank 0 with the real scheduler, rank 1 with an executor, and the channel
/// between them. Both ranks hold `sessions` states.
class Pair {
public:
  Pair(ToyOptions rank0_options, ToyOptions rank1_options,
       std::size_t sessions = 1)
      : inner0_(std::make_shared<ToyRunner>(rank0_options)),
        runner1_(std::make_shared<ToyRunner>(rank1_options)) {
    const auto port = FreePort();
    std::string server_error;
    std::string client_error;
    std::thread connector([&] {
      client_ = TpControlChannel::Connect("127.0.0.1", port, &client_error);
    });
    server_ = TpControlChannel::Listen(port, &server_error);
    connector.join();
    Require(server_ != nullptr && client_ != nullptr,
            "channel pair: " + server_error + client_error);
    const TpControlConfig rank0{
        .rank = 0,
        .world_size = 2,
        .max_context = 256,
        .auth_token = "executor-test",
        .sessions = static_cast<std::uint32_t>(sessions)};
    TpControlConfig rank1 = rank0;
    rank1.rank = 1;
    bool server_ok = false;
    bool client_ok = false;
    std::thread handshake(
        [&] { client_ok = client_->Handshake(rank1, &client_error); });
    server_ok = server_->Handshake(rank0, &server_error);
    handshake.join();
    Require(server_ok && client_ok,
            "handshake: " + server_error + client_error);

    broker_ =
        std::make_shared<gufo::server::TpResponseBroker>(server_, sessions + 8);
    sink_ = std::make_shared<TpControlInstructionSink>(server_, broker_);
    mirrored_ = std::make_shared<TpMirroredRunner>(inner0_, sink_);
    pool_ = std::make_shared<TextRunnerPool>(mirrored_, sessions);
    scheduler_ = std::make_shared<TextGenerationScheduler>(pool_);
    executor_ = std::make_unique<TpExecutor>(runner1_, sessions);
    worker_ = std::thread([this] { Work(); });
  }

  ~Pair() {
    scheduler_.reset();
    pool_.reset();
    broker_->FailAll("test finished");
    server_.reset();
    client_->Interrupt();
    worker_.join();
  }

  struct Outcome {
    TextGenerationBackend::Result result;
    std::string worker_error;
  };

  /// A request that finished on rank 0 but has not ended on rank 1 yet.
  struct Open {
    std::uint64_t sequence{0};
    Outcome outcome;
  };

  /// Runs one request as serving does. `submitted` runs once the scheduler
  /// has the request.
  Outcome Run(
      std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
      gufo::sampling::SamplingConfig sampling = {},
      std::vector<std::string> stop_sequences = {},
      TextGenerationScheduler::CancellationCheck is_cancelled = {},
      bool reuse = false, std::size_t prefix = 0,
      const std::function<void()>& submitted = {},
      std::shared_ptr<const gufo::server::TextPromptContext> images = {}) {
    return Close(RunOpen(std::move(prompt), max_tokens, std::move(sampling),
                         std::move(stop_sequences), std::move(is_cancelled),
                         reuse, prefix, submitted, std::move(images)));
  }

  /// Runs a request on rank 0 without ending it, as when the next request
  /// arrives before rank 1's verdict on this one.
  Open RunOpen(
      std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
      gufo::sampling::SamplingConfig sampling = {},
      std::vector<std::string> stop_sequences = {},
      TextGenerationScheduler::CancellationCheck is_cancelled = {},
      bool reuse = false, std::size_t prefix = 0,
      const std::function<void()>& submitted = {},
      std::shared_ptr<const gufo::server::TextPromptContext> images = {}) {
    const auto sequence = next_sequence_.fetch_add(1);
    TpControlCommand begin{.sequence = sequence,
                           .max_tokens = static_cast<std::uint32_t>(max_tokens),
                           .client_id = "executor-test",
                           .sampling = sampling};
    if (images != nullptr) {
      begin.prompt_context = mirrored_->EncodePromptContext(*images);
    }
    for (const auto token : prompt) {
      begin.prompt_tokens.push_back(static_cast<std::int32_t>(token));
    }
    std::string error;
    Require(broker_->RegisterPendingResponse(sequence, &error),
            "register: " + error);
    Require(server_->SendCommand(begin, &error), "begin: " + error);
    mirrored_->BeginRequest(sequence);
    TextRequestMetadata metadata;
    metadata.prompt_context =
        std::make_shared<TpRequestContext>(sequence, std::move(images));
    metadata.cache_prompt = reuse;
    metadata.cache_prefix_tokens = prefix;
    metadata.stop_sequences = std::move(stop_sequences);
    auto request = scheduler_->Submit(std::move(prompt), max_tokens, sampling,
                                      is_cancelled, reuse, std::move(metadata));
    if (submitted) {
      submitted();
    }
    Open open{.sequence = sequence};
    try {
      open.outcome.result = request.Wait();
    } catch (const std::exception& exception) {
      open.outcome.worker_error = exception.what();
    }
    return open;
  }

  /// Ends a request on rank 1 and settles it as serving does.
  Outcome Close(Open open) {
    const auto sequence = open.sequence;
    Outcome outcome = std::move(open.outcome);
    const std::string local_error = std::move(outcome.worker_error);
    std::string error;
    Require(mirrored_->EndRequest(sequence, &error), "end: " + error);
    TpControlResponse response;
    Require(broker_->WaitForResponse(sequence, &response, &error),
            "response: " + error);
    Require(response.sequence == sequence, "response sequence");
    const bool dependencies = mirrored_->AwaitDependencies(sequence);
    mirrored_->Settle(sequence, response.error.empty() && dependencies);
    outcome.worker_error = !response.error.empty() ? response.error
                           : !dependencies
                               ? std::string("reused a rejected state")
                               : local_error;
    return outcome;
  }

  /// A reset sent between requests, as the continuation cache may send one.
  void ResetBetweenRequests() {
    std::string error;
    Require(sink_->Send(0, {.op = TpInstructionOp::kInvalidate, .state = 0},
                        nullptr, &error),
            "idle reset: " + error);
  }

  void RequireSameCalls(const std::string& what) const {
    const auto rank0 = inner0_->Log();
    const auto rank1 = runner1_->Log();
    if (rank0 != rank1) {
      std::fprintf(stderr, "rank 0 calls:\n");
      for (const auto& line : rank0) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
      std::fprintf(stderr, "rank 1 calls:\n");
      for (const auto& line : rank1) {
        std::fprintf(stderr, "  %s\n", line.c_str());
      }
    }
    Require(rank0 == rank1, what + ": both ranks make the same model calls");
    Require(worker_failure_.empty(), what + ": worker " + worker_failure_);
  }

  std::size_t snapshots() const { return worker_snapshots_.load(); }
  std::size_t open_requests() const { return worker_open_.load(); }
  void ClearCache() {
    pool_->ClearCache();
    // An empty execution request fences idle drops without taking another
    // snapshot (cache_prompt=false only disables lookups, not capture).
    const auto sequence = next_sequence_.fetch_add(1);
    std::string error;
    Require(broker_->RegisterPendingResponse(sequence, &error), error);
    Require(server_->SendCommand(
                {.sequence = sequence, .max_tokens = 1, .prompt_tokens = {1}},
                &error),
            error);
    mirrored_->BeginRequest(sequence);
    Require(mirrored_->EndRequest(sequence, &error), error);
    TpControlResponse response;
    Require(broker_->WaitForResponse(sequence, &response, &error), error);
    Require(response.error.empty(), response.error);
    mirrored_->Settle(sequence, true);
  }

  const ToyRunner& rank0() const { return *inner0_; }
  const ToyRunner& rank1() const { return *runner1_; }

private:
  void Work() {
    const auto respond = [this](const TpControlResponse& response,
                                std::string* error) {
      // Published before a verdict leaves, so a test that has its verdict
      // sees the worker's tables as they were then.
      worker_snapshots_ = executor_->snapshot_count();
      worker_open_ = executor_->open_requests();
      return client_->SendResponse(response, error);
    };
    for (;;) {
      TpControlCommand command;
      std::string error;
      if (!client_->ReceiveCommand(&command, &error)) {
        return;
      }
      if (!executor_->Execute(command, respond, &error)) {
        worker_failure_ = error;
        return;
      }
    }
  }

  std::shared_ptr<ToyRunner> inner0_;
  std::shared_ptr<ToyRunner> runner1_;
  std::shared_ptr<TpControlChannel> server_;
  std::shared_ptr<TpControlChannel> client_;
  std::shared_ptr<gufo::server::TpResponseBroker> broker_;
  std::shared_ptr<TextRunnerPool> pool_;
  std::shared_ptr<TpControlInstructionSink> sink_;
  std::shared_ptr<TpMirroredRunner> mirrored_;
  std::shared_ptr<TextGenerationScheduler> scheduler_;
  std::unique_ptr<TpExecutor> executor_;
  std::thread worker_;
  std::string worker_failure_;
  std::atomic<std::size_t> worker_snapshots_{0};
  std::atomic<std::size_t> worker_open_{0};
  std::atomic<std::uint64_t> next_sequence_{1};
};

/// Records rank 0's instructions and runs cache calls at once, for tests of
/// the mirrored runner alone.
class CaptureSink final : public TpInstructionSink {
public:
  struct Sent {
    std::uint64_t sequence;
    TpInstruction instruction;
  };
  void Synchronize(std::uint64_t sequence, TpInstruction instruction,
                   const std::function<void()>& local) override {
    instruction.index = next_++;
    sent.push_back({sequence, std::move(instruction)});
    local();
  }
  bool Send(std::uint64_t sequence, TpInstruction instruction,
            std::uint64_t* index, std::string*) override {
    instruction.index = next_;
    if (index != nullptr) {
      *index = next_;
    }
    ++next_;
    sent.push_back({sequence, std::move(instruction)});
    return true;
  }
  std::vector<Sent> sent;

private:
  std::uint64_t next_{0};
};

/// Records which instruction indices had a collective scope bound.
class RecordingScope final : public TpCallScope {
public:
  bool Begin(std::uint64_t index, std::string*) override {
    Require(!bound_.has_value(), "one collective scope at a time");
    bound_ = index;
    began.push_back(index);
    return true;
  }
  bool End(std::uint64_t index, std::string*) override {
    Require(bound_ == index, "a scope ends as it began");
    bound_.reset();
    return true;
  }
  std::vector<std::uint64_t> began;

private:
  std::optional<std::uint64_t> bound_;
};

std::size_t Count(const CallLog& log, const std::string& prefix) {
  return static_cast<std::size_t>(std::count_if(
      log.begin(), log.end(),
      [&](const std::string& line) { return line.rfind(prefix, 0) == 0; }));
}

/// Holds the scheduler in its first prefill until `count` requests were
/// submitted, so they run concurrently however the threads are scheduled.
class SubmissionGate {
public:
  explicit SubmissionGate(std::size_t count) : count_(count) {}
  void Submitted() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++submitted_;
    }
    condition_.notify_all();
  }
  std::function<void()> Hook() {
    return [this] {
      if (passed_.exchange(true)) {
        return;
      }
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return submitted_ >= count_; });
    };
  }

private:
  const std::size_t count_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t submitted_{0};
  std::atomic<bool> passed_{false};
};

gufo::sampling::SamplingConfig Sampled() {
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.9F;
  config.seed = 42;
  // History matters only with a penalty, so this also checks that rank 1's
  // sampler accepts the same tokens as rank 0's.
  config.repeat_penalty = 1.3F;
  return config;
}

}  // namespace

int main() {
  const std::vector<TextRunnerToken> prompt{5, 9, 13, 17, 21, 3, 8};

  for (const bool mtp : {false, true}) {
    Pair pair({.mtp = mtp, .cache = true}, {.mtp = mtp, .cache = true});
    const auto sampling = mtp ? Sampled() : gufo::sampling::SamplingConfig{};
    const auto first = pair.Run(prompt, 6, sampling, {}, {}, true);
    Require(first.worker_error.empty(), "cached first: " + first.worker_error);
    const auto again = pair.Run(prompt, 6, sampling, {}, {}, true);
    Require(again.worker_error.empty(),
            "cached restore: " + again.worker_error);
    Require(first.result.tokens == again.result.tokens,
            "restore preserves outputs");
    Require(again.result.cached_prompt_tokens == prompt.size(),
            "identical replay restores the whole prompt, as on one host");
    pair.RequireSameCalls("snapshot reuse");
    auto branch = prompt;
    branch.push_back(11);
    const auto fork = pair.Run(branch, 4, sampling, {}, {}, true);
    Require(fork.worker_error.empty(), "branch: " + fork.worker_error);
    Require(fork.result.cached_prompt_tokens == prompt.size(),
            "older boundary restored");
    pair.RequireSameCalls("branch reuse");
    (void)pair.Run(prompt, 6, sampling, {}, {}, true);
    auto continuation = prompt;
    continuation.insert(continuation.end(), again.result.tokens.begin(),
                        again.result.tokens.end());
    continuation.push_back(7);
    const auto live = pair.Run(continuation, 4, sampling, {}, {}, true);
    Require(live.worker_error.empty(), "live: " + live.worker_error);
    Require(live.result.cached_prompt_tokens > prompt.size(),
            "live frontier reused");
    pair.RequireSameCalls("live reuse");
  }
  // A capture that fails on either rank only skips the snapshot, as on one
  // host: the request succeeds, and no rank keeps a copy the other lacks.
  for (const bool rank0_fails : {false, true}) {
    Pair pair({.cache = true, .fail_capture_once = rank0_fails},
              {.cache = true, .fail_capture_once = !rank0_fails});
    const auto first = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(first.worker_error.empty(),
            "failed capture is not a failed request: " + first.worker_error);
    Require(pair.snapshots() == 0, "failed capture leaves no worker copy");
    const auto next = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(next.worker_error.empty(),
            "request after failed capture: " + next.worker_error);
    Require(next.result.cached_prompt_tokens == 0,
            "a skipped snapshot is a miss");
    Require(next.result.tokens == first.result.tokens,
            "a skipped snapshot keeps outputs");
    Require(pair.snapshots() == 1, "next capture succeeds on both ranks");
  }
  {
    Pair pair({.cache = true}, {.cache = true, .fail_restore_once = true});
    const auto first = pair.Run(prompt, 6, {}, {}, {}, true);
    const auto failed = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(!failed.worker_error.empty(),
            "worker restore failure reaches request verdict");
    // The failed restore reset the state on both ranks and left the snapshot
    // an agreed request captured intact, so the next request reuses it.
    const auto recovered = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(recovered.worker_error.empty(),
            "next request recovers: " + recovered.worker_error);
    Require(recovered.result.cached_prompt_tokens == prompt.size() &&
                recovered.result.tokens == first.result.tokens,
            "a failed request keeps the snapshots it did not change");
  }

  {
    Pair pair({.cache = true, .cache_budget = 32},
              {.cache = true, .cache_budget = 32});
    for (int i = 0; i < 4; ++i) {
      auto distinct = prompt;
      distinct[0] += i;
      const auto run = pair.Run(distinct, 3, {}, {}, {}, true);
      Require(run.worker_error.empty(), "eviction: " + run.worker_error);
      Require(pair.snapshots() == 1, "eviction drops old worker copies");
    }
    pair.ClearCache();  // Drops outside an active request must work too.
    Require(pair.snapshots() == 0, "idle drops empty worker table");
  }
  {
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    std::atomic<bool> cancelled{false};
    ToyOptions held{.cache = true};
    held.capture_hook = [&] {
      entered.set_value();
      released.wait();
    };
    Pair pair(held, {.cache = true});
    auto pending = std::async(std::launch::async, [&] {
      return pair.Run(
          prompt, 6, {}, {}, [&] { return cancelled.load(); }, true);
    });
    Require(entered.get_future().wait_for(std::chrono::seconds(5)) ==
                std::future_status::ready,
            "asynchronous capture starts");
    Require(Count(pair.rank0().Log(), "advance") == 0,
            "capture freezes the state before advance");
    cancelled = true;
    release.set_value();
    const auto stopped = pending.get();
    Require(stopped.worker_error.empty(),
            "cancel during capture: " + stopped.worker_error);
    const auto next = pair.Run(prompt, 3, {}, {}, {}, true);
    Require(next.worker_error.empty() && next.result.cached_prompt_tokens > 0,
            "joined capture survives cancellation");
    pair.RequireSameCalls("capture cancellation reuse");
  }

  // Mutating the instruction stream must fail even though every surviving
  // frame is valid on its own and has a freshly contiguous transport index.
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto runner = std::make_shared<ToyRunner>(ToyOptions{.cache = true});
    TpExecutor executor(runner, 1);
    const TpControlCommand begin{
        .sequence = 1, .max_tokens = 1, .prompt_tokens = {5, 9}};
    std::vector<gufo::server::TpInstruction> calls{
        {.op = TpInstructionOp::kPrefill, .count = 2, .prompt_size = 2},
        {.op = TpInstructionOp::kSnapshot, .snapshot_id = 1},
        {.op = TpInstructionOp::kReuse, .prompt_size = 2},
        {.op = TpInstructionOp::kRestore, .snapshot_id = 1},
        {.op = TpInstructionOp::kDrop, .snapshot_id = 1}};
    gufo::server::TpExecutionDigest digest;
    for (std::size_t index = 0; index < calls.size(); ++index) {
      auto& call = calls[index];
      const std::vector<TextRunnerToken> prefix{5, 9};
      gufo::server::DigestTpCall(digest, call, prefix);
      if (call.op == TpInstructionOp::kSnapshot) {
        // As rank 0 sends it: the request's calls so far.
        call.count = static_cast<std::uint32_t>(index + 1);
        call.digest = digest.value();
      }
      if (call.op == TpInstructionOp::kPrefill)
        gufo::server::DigestTpPrefill(
            digest, {.consumed_tokens = 2, .decode_ready = true}, 2);
      if (call.op == TpInstructionOp::kRestore ||
          call.op == TpInstructionOp::kReuse)
        digest.Add(2);
    }
    if (mutation == 1)
      calls.pop_back();  // missing drop
    if (mutation == 2)
      calls.erase(calls.begin() + 2);  // missing reuse
    if (mutation == 3)
      calls[3].snapshot_id = 2;  // wrong restore
    calls.push_back(
        {.op = TpInstructionOp::kEnd, .count = 5, .digest = digest.value()});
    std::optional<std::string> outcome;
    const auto respond = [&](const TpControlResponse& response, std::string*) {
      if (response.kind == gufo::server::TpControlResponseKind::kSingle) {
        outcome = response.error;
      }
      return true;
    };
    std::string error;
    Require(executor.Execute(begin, respond, &error), error);
    for (std::size_t next = 0; next < calls.size(); ++next) {
      calls[next].index = next;
      Require(executor.Execute({.sequence = 1,
                                .kind = TpControlCommandKind::kInstruction,
                                .instruction = calls[next]},
                               respond, &error),
              error);
    }
    Require(outcome.has_value() && outcome->empty() == (mutation == 0),
            "instruction mutation is detected: " + outcome.value_or("none"));
    Require(executor.open_requests() == 0, "the verdict closes the request");
    if (mutation == 0)
      Require(executor.snapshot_count() == 0, "intact drop frees snapshot");
  }

  // Rank 1 refuses a stream that names requests it does not know.
  {
    auto runner = std::make_shared<ToyRunner>(ToyOptions{});
    TpExecutor executor(runner, 2);
    const auto respond = [](const TpControlResponse&, std::string*) {
      return true;
    };
    std::string error;
    Require(!executor.Execute({.sequence = 3,
                               .kind = TpControlCommandKind::kInstruction,
                               .instruction = {.op = TpInstructionOp::kPrefill,
                                               .count = 1,
                                               .prompt_size = 1}},
                              respond, &error) &&
                error.find("not open") != std::string::npos,
            "an instruction for an unknown request is a protocol failure: " +
                error);
    TpExecutor batch_executor(runner, 2);
    Require(batch_executor.Execute(
                {.sequence = 1, .max_tokens = 1, .prompt_tokens = {5}}, respond,
                &error),
            error);
    Require(
        !batch_executor.Execute(
            {.sequence = 0,
             .kind = TpControlCommandKind::kInstruction,
             .instruction = {.op = TpInstructionOp::kAdvanceBatch,
                             .batch = {{.sequence = 1, .state = 0},
                                       {.sequence = 2, .state = 1}}}},
            respond, &error) &&
            error.find("not open") != std::string::npos,
        "a batch member of an unknown request is a protocol failure: " + error);
  }

  // The mirrored runner alone: every call belongs to the request its state is
  // leased to, forwards bind their instruction's collective scope, a batch is
  // one instruction, and a request that reuses what another computed depends
  // on rank 1's verdict on it.
  {
    auto toy = std::make_shared<ToyRunner>(ToyOptions{.cache = true});
    auto sink = std::make_shared<CaptureSink>();
    auto scope = std::make_shared<RecordingScope>();
    auto runner = std::make_shared<TpMirroredRunner>(
        toy, sink, std::numeric_limits<std::size_t>::max(), scope);
    auto first = runner->CreateState();
    auto second = runner->CreateState();
    const std::vector<TextRunnerToken> short_prompt{4, 5, 6};
    bool refused = false;
    try {
      (void)runner->Prefill(*first, short_prompt, 0, 3);
    } catch (const std::logic_error&) {
      refused = true;
    }
    Require(refused && sink->sent.empty(),
            "a call on a state no request leases is refused unsent");

    runner->BeginRequest(7);
    runner->BeginRequest(8);
    runner->SetPromptContext(*first, std::make_shared<TpRequestContext>(7));
    runner->SetPromptContext(*second, std::make_shared<TpRequestContext>(8));
    (void)runner->Prefill(*first, short_prompt, 0, 3);
    (void)runner->Prefill(*second, short_prompt, 0, 3);
    // Leasing a state binds it on rank 1 too, before the request's calls.
    Require(
        sink->sent.size() == 4 &&
            sink->sent[0].instruction.op == TpInstructionOp::kPromptContext &&
            sink->sent[0].sequence == 7 &&
            sink->sent[1].instruction.op == TpInstructionOp::kPromptContext &&
            sink->sent[1].sequence == 8 && sink->sent[2].sequence == 7 &&
            sink->sent[3].sequence == 8,
        "each call names the request its state is leased to");
    const auto snapshot = runner->Snapshot(*first);
    Require(runner->CanReuse(*snapshot),
            "a snapshot is reusable as soon as it is captured");
    Require(sink->sent.back().instruction.op == TpInstructionOp::kSnapshot &&
                sink->sent.back().instruction.count == 3 &&
                sink->sent.back().instruction.digest != 0,
            "a capture carries the request's call count and digest");

    std::exception_ptr failures[2];
    const gufo::server::TextRunnerAdvance advances[] = {
        {.state = *first, .token = 9, .failure = &failures[0]},
        {.state = *second, .token = 10, .failure = &failures[1]}};
    runner->AdvanceBatch(advances);
    const auto& batch = sink->sent.back();
    Require(batch.sequence == 0 &&
                batch.instruction.op == TpInstructionOp::kAdvanceBatch &&
                batch.instruction.batch.size() == 2 &&
                batch.instruction.batch[0].sequence == 7 &&
                batch.instruction.batch[0].state == 0 &&
                batch.instruction.batch[0].token == 9 &&
                batch.instruction.batch[1].sequence == 8 &&
                batch.instruction.batch[1].state == 1 && !failures[0] &&
                !failures[1],
            "a batched advance is one instruction naming each member");
    Require(Count(toy->Log(), "batch 2") == 1,
            "rank 0 runs the batch as one call");
    std::vector<std::uint64_t> forwards;
    for (const auto& sent : sink->sent) {
      if (sent.instruction.op == TpInstructionOp::kPrefill ||
          sent.instruction.op == TpInstructionOp::kAdvanceBatch) {
        forwards.push_back(sent.instruction.index);
      }
    }
    Require(scope->began == forwards,
            "every forward binds its instruction's collective scope");

    const auto second_snapshot = runner->Snapshot(*second);
    std::string error;
    Require(runner->EndRequest(7, &error) && runner->EndRequest(8, &error),
            error);
    Require(!runner->EndRequest(7, &error), "a request ends once");
    Require(runner->CanReuse(*first) && runner->CanReuse(*second),
            "an ended request's state is reusable before its verdict");
    // Requests that reuse those states depend on the verdicts.
    runner->BeginRequest(9);
    runner->BeginRequest(10);
    runner->SetPromptContext(*first, std::make_shared<TpRequestContext>(9));
    runner->SetPromptContext(*second, std::make_shared<TpRequestContext>(10));
    runner->Settle(7, true);
    runner->Settle(8, false);
    Require(runner->AwaitDependencies(9),
            "a request that reused an agreed state stands");
    Require(!runner->AwaitDependencies(10),
            "a request that reused a rejected state fails too");
    Require(runner->CanReuse(*first) && runner->CanReuse(*snapshot),
            "an agreed request's state and snapshot are reusable");
    Require(!runner->CanReuse(*second) && !runner->CanReuse(*second_snapshot),
            "a rejected request's state and snapshot are not reused");
    second->Invalidate();
    Require(runner->CanReuse(*second), "a reset state is clean");
    Require(runner->EndRequest(9, &error) && runner->EndRequest(10, &error),
            error);
    bool unleased = false;
    try {
      runner->Advance(*first, 5);
    } catch (const std::logic_error&) {
      unleased = true;
    }
    Require(unleased, "an ended request releases its state");
  }

  // AR: greedy, sampled, stopped and cancelled requests on one pair.
  {
    Pair pair({}, {});
    const auto greedy = pair.Run(prompt, 6);
    Require(greedy.worker_error.empty(), "greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 6 &&
                greedy.result.finish_reason == FinishReason::kLength,
            "greedy request produces its budget");
    pair.RequireSameCalls("greedy");
    const auto calls = pair.rank0().Log();
    Require(Count(calls, "prefill") == 3,
            "a 7-token prompt prefills in three chunks of three");
    // The last token is published without advancing it.
    Require(Count(calls, "advance") == 5, "greedy advances all but the last");
    Require(pair.rank0().cancellation_checks() == 0,
            "the model session never receives a cancellation check");

    const auto again = pair.Run(prompt, 6);
    Require(again.worker_error.empty() &&
                again.result.tokens == greedy.result.tokens,
            "a repeated request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("repeated");

    const auto sampled = pair.Run(prompt, 8, Sampled());
    Require(sampled.worker_error.empty(), "sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 8, "sampled request completes");
    pair.RequireSameCalls("sampled");

    // Stop at a generated piece that does not occur earlier in the output.
    std::optional<std::size_t> stop_at;
    std::string text;
    for (std::size_t i = 0; i < greedy.result.tokens.size(); ++i) {
      const auto piece = ToyRunner::Piece(greedy.result.tokens[i]);
      if (i >= 2 && text.find(piece) == std::string::npos) {
        stop_at = i;
        break;
      }
      text += piece;
    }
    Require(stop_at.has_value(), "the greedy output has a usable stop piece");
    const auto stop_piece = ToyRunner::Piece(greedy.result.tokens[*stop_at]);
    const auto stopped = pair.Run(prompt, 6, {}, {stop_piece});
    Require(stopped.worker_error.empty(), "stop: " + stopped.worker_error);
    Require(stopped.result.finish_reason == FinishReason::kStopSequence &&
                stopped.result.stop_sequence == stop_piece,
            "the request ends at its stop sequence");
    pair.RequireSameCalls("stop sequence");

    // Cancel after the second prefill chunk of a long prompt.
    const auto before = pair.rank0().prefill_calls();
    const std::vector<TextRunnerToken> long_prompt{1, 2,  3,  4,  5,  6,  7, 8,
                                                   9, 10, 11, 12, 13, 14, 15};
    const auto cancelled = pair.Run(long_prompt, 6, {}, {}, [&pair, before] {
      return pair.rank0().prefill_calls() >= before + 2;
    });
    Require(cancelled.worker_error.empty(),
            "cancel: " + cancelled.worker_error);
    Require(cancelled.result.cancelled ||
                cancelled.result.finish_reason == FinishReason::kCancelled,
            "the request is cancelled");
    Require(pair.rank0().prefill_calls() < before + 5,
            "cancellation stops prefill at a chunk boundary");
    pair.RequireSameCalls("cancelled during prefill");

    // A reset between requests reaches rank 1 too.
    pair.ResetBetweenRequests();
    const auto after_reset = pair.Run(prompt, 3);
    Require(after_reset.worker_error.empty(),
            "after an idle reset: " + after_reset.worker_error);
    Require(Count(pair.rank1().Log(), "reset") ==
                Count(pair.rank0().Log(), "reset") + 1,
            "rank 1 executed the reset sent between requests");
  }

  // Multi-token decoding is one instruction per cycle, greedy or sampled; a
  // sampled cycle carries rank 0's draw state.
  {
    Pair pair({.mtp = true}, {.mtp = true});
    const auto greedy = pair.Run(prompt, 7);
    Require(greedy.worker_error.empty(), "MTP greedy: " + greedy.worker_error);
    Require(greedy.result.tokens.size() == 7, "MTP greedy completes");
    Require(Count(pair.rank0().Log(), "decode") > 0,
            "greedy MTP decodes in multi-token cycles");
    pair.RequireSameCalls("MTP greedy");

    const auto decodes = Count(pair.rank0().Log(), "decode");
    const auto sampled = pair.Run(prompt, 9, Sampled());
    Require(sampled.worker_error.empty(),
            "MTP sampled: " + sampled.worker_error);
    Require(sampled.result.tokens.size() == 9, "MTP sampled completes");
    Require(Count(pair.rank0().Log(), "decode") > decodes,
            "sampled MTP decodes in multi-token cycles too");
    pair.RequireSameCalls("MTP sampled");
    const auto again = pair.Run(prompt, 9, Sampled());
    Require(
        again.worker_error.empty() &&
            again.result.tokens == sampled.result.tokens,
        "a seeded sampled MTP request is reproducible: " + again.worker_error);
    pair.RequireSameCalls("MTP sampled again");

    // Without a seed each rank's sampler starts from its own random state, so
    // only rank 0's draw state, carried by each decode, keeps them together.
    auto unseeded = Sampled();
    unseeded.seed = -1;
    const auto random_draws = pair.Run(prompt, 9, unseeded);
    Require(random_draws.worker_error.empty(),
            "unseeded sampled MTP: " + random_draws.worker_error);
    pair.RequireSameCalls("MTP sampled without a seed");
  }

  // Rank 1's own greedy choice catches a numerical divergence.
  {
    Pair pair({}, {.diverge_at = prompt.size() + 2});
    const auto diverged = pair.Run(prompt, 6);
    Require(diverged.worker_error.find("selected token") != std::string::npos,
            "a rank-1 divergence fails the request: " + diverged.worker_error);
    // Rank 1 still advanced rank 0's tokens, so the pair stays usable.
    pair.RequireSameCalls("after divergence");
    const auto next = pair.Run(prompt, 6);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  // What a rejected request computed is never reused: the next request
  // recomputes it, and that agreed result is reusable again.
  {
    Pair pair({.cache = true},
              {.cache = true, .diverge_at = prompt.size() + 2});
    const auto diverged = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(diverged.worker_error.find("selected token") != std::string::npos,
            "a rank-1 divergence fails the request: " + diverged.worker_error);
    const auto recomputed = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(recomputed.worker_error.empty() &&
                recomputed.result.cached_prompt_tokens == 0,
            "a rejected request's snapshot is not reused: " +
                recomputed.worker_error);
    const auto reused = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(reused.worker_error.empty() &&
                reused.result.cached_prompt_tokens == prompt.size() &&
                reused.result.tokens == recomputed.result.tokens,
            "an agreed recomputation is reused: " + reused.worker_error);
    pair.RequireSameCalls("rejected continuation");
  }

  // A request reuses what the previous one left before rank 1's verdict on
  // it arrives, as on one host, and stands or falls with that verdict.
  for (const bool reject : {false, true}) {
    const std::string what =
        reject ? "reuse before a rejection" : "reuse before a verdict";
    ToyOptions rank1{.cache = true};
    if (reject) {
      rank1.diverge_at = prompt.size() + 2;
    }
    Pair pair({.cache = true}, rank1);
    auto first = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    auto second = pair.RunOpen(prompt, 6, {}, {}, {}, true);
    Require(second.outcome.result.cached_prompt_tokens == prompt.size(),
            what + ": the second request reuses the first one's prompt");
    const auto first_end = pair.Close(std::move(first));
    const auto second_end = pair.Close(std::move(second));
    Require(first_end.worker_error.empty() != reject,
            what + ": the first request's verdict: " + first_end.worker_error);
    Require(
        second_end.worker_error.empty() != reject,
        what + ": the second request follows it: " + second_end.worker_error);
  }

  // Concurrent requests: each call names its request, concurrent AR decoders
  // advance in one batched instruction, MTP decoders take turns, and every
  // request produces what it produces alone.
  const std::vector<TextRunnerToken> other{2, 4, 6, 8, 10, 12, 14};
  for (const bool mtp : {false, true}) {
    const auto sampling = mtp ? Sampled() : gufo::sampling::SamplingConfig{};
    std::vector<TextRunnerToken> alone[2];
    {
      Pair solo({.mtp = mtp}, {.mtp = mtp});
      alone[0] = solo.Run(prompt, 12, sampling).result.tokens;
      alone[1] = solo.Run(other, 12, sampling).result.tokens;
    }
    SubmissionGate gate(2);
    const std::size_t width = mtp ? 1 : 4;
    Pair pair({.mtp = mtp, .prefill_hook = gate.Hook(), .width = width},
              {.mtp = mtp, .width = width}, 2);
    const auto submitted = [&gate] { gate.Submitted(); };
    auto first = std::async(std::launch::async, [&] {
      return pair.Run(prompt, 12, sampling, {}, {}, false, 0, submitted);
    });
    auto second = std::async(std::launch::async, [&] {
      return pair.Run(other, 12, sampling, {}, {}, false, 0, submitted);
    });
    const auto a = first.get();
    const auto b = second.get();
    const std::string what = mtp ? "concurrent MTP" : "concurrent AR";
    Require(a.worker_error.empty() && b.worker_error.empty(),
            what + ": " + a.worker_error + b.worker_error);
    Require(a.result.tokens == alone[0] && b.result.tokens == alone[1],
            what + " requests produce what they produce alone");
    Require((Count(pair.rank0().Log(), "batch") > 0) == !mtp,
            what + ": AR decoders advance in batches, MTP ones in turns");
    pair.RequireSameCalls(what);
    Require(pair.open_requests() == 0, what + ": rank 1 closed both requests");
  }

  // The execution digest catches a result the ranks disagree on.
  {
    Pair pair({.mtp = true}, {.mtp = true, .extra_draft_once = true});
    const auto mismatch = pair.Run(prompt, 7);
    Require(mismatch.worker_error.find("different model calls") !=
                std::string::npos,
            "a result difference fails the request: " + mismatch.worker_error);
    const auto next = pair.Run(prompt, 7);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  // A request with images: rank 1 rebuilds its prompt context and binds it
  // wherever rank 0 does, and the cache keeps image and text states apart.
  for (const bool mtp : {false, true}) {
    const std::string what = mtp ? "MTP images" : "AR images";
    Pair pair({.mtp = mtp, .cache = true}, {.mtp = mtp, .cache = true});
    const auto image = std::make_shared<ToyContext>(7);
    const auto text = pair.Run(prompt, 6, {}, {}, {}, true);
    const auto seen = pair.Run(prompt, 6, {}, {}, {}, true, 0, {}, image);
    const auto again = pair.Run(prompt, 6, {}, {}, {}, true, 0, {}, image);
    const auto text_again = pair.Run(prompt, 6, {}, {}, {}, true);
    Require(text.worker_error.empty() && seen.worker_error.empty() &&
                again.worker_error.empty() && text_again.worker_error.empty(),
            what + ": " + text.worker_error + seen.worker_error +
                again.worker_error + text_again.worker_error);
    Require(seen.result.tokens != text.result.tokens,
            what + ": the image changes the output");
    Require(seen.result.cached_prompt_tokens == 0,
            what + ": an image request does not reuse a text state");
    Require(again.result.tokens == seen.result.tokens &&
                again.result.cached_prompt_tokens == prompt.size(),
            what + ": the same image reuses its prompt");
    Require(text_again.result.tokens == text.result.tokens,
            what + ": text after an image matches text alone");
    Require(Count(pair.rank0().Log(), "context") > 0,
            what + ": the image context reaches the model");
    pair.RequireSameCalls(what);
  }

  // A prompt context rank 1 cannot rebuild fails that request only.
  {
    Pair pair({}, {.fail_context_decode = true});
    const auto image = std::make_shared<ToyContext>(3);
    const auto refused = pair.Run(prompt, 4, {}, {}, {}, false, 0, {}, image);
    Require(refused.worker_error.find("prompt context") != std::string::npos,
            "an unreadable prompt context fails the request: " +
                refused.worker_error);
    const auto next = pair.Run(prompt, 4);
    Require(next.worker_error.empty(),
            "the next request succeeds: " + next.worker_error);
  }

  std::puts(
      "PASS: TP2 executor mirrors greedy, sampled, stopped, cancelled, "
      "multi-token, concurrent, batched and image requests, reports divergence "
      "and reuses only agreed continuations");
  return 0;
}
