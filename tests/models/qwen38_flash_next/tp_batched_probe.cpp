// TP=2 batched-decode probe: one program, two ranks, two hosts.
//
// qwen38_flash_next_tp_batched_probe --tp-rank 0 [options]
// qwen38_flash_next_tp_batched_probe --tp-rank 1 --tp-bootstrap-host HOST
//                                    [options]
//
// Both ranks load the model over one RDMA pair, prefill every member's prompt
// in turn and decode greedily. By default each decode step advances every
// running member in one `Session::EvaluateBatch`, the call concurrent serving
// makes for single-token decoders; `--serial` advances them one at a time with
// `Session::Evaluate`, as a lone decoder does. No control channel, scheduler
// or executor is involved, and each rank samples its own logits, so the ranks
// agree only if they compute the same logits. Each rank prints the member set
// and logit hashes of every step and each member's tokens and checksum; diff
// the two ranks' logs, and a batched run against a serial one.
//
// `--moe-input-hashes` hashes every MoE input during prefill, to find the
// first layer at which the ranks diverge. `--allreduce-bench N` loads no model
// and times the queued exchange the model's per-layer sums use.
//
// OPERATING RULES
//   * Start rank 0 first and rank 1 within 30 s: both RDMA bootstraps give up
//     after 30 s. Wrap both ranks in an external timeout.
//   * The ranks must also finish loading the model within 30 s of each other:
//     they meet at a zero-byte exchange before the prefill, which times out
//     like any other. A host that has not read the model recently loads it
//     from disk much more slowly, so warm its page cache first.
//   * The RDMA adapter maps fixed IOVA windows, so the ranks run on separate
//     hosts with no other RDMA process on either.
//   * Both ranks need the same --model, --context, --max-tokens, --width,
//     prompts and mode. Nothing checks that, so a mismatch shows up as logs
//     that differ or as a 30 s exchange timeout.
//
// EXIT REASONS
//   0  the program ran to completion; agreement is judged by diffing the logs
//   1  transport, model or execution failure
//   2  argument failure
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/verbs.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {

/// Members 0 and 1 take their prompts from options; the default width.
constexpr std::size_t kPromptOptions = 2;
/// `--width` bound; the Q8 decode GEMVs take at most eight rows
/// (MMVQ_MAX_BATCH_SIZE).
constexpr std::uint32_t kMaxWidth = 8;
/// Prompts for members beyond the first two, which keep
/// `--prompt-0`/`--prompt-1` and their defaults.
constexpr const char* kExtraPrompts[kMaxWidth - kPromptOptions] = {
    "Water boils at sea level at",
    "The author of Hamlet is",
    "The speed of light in a vacuum is about",
    "Photosynthesis converts sunlight into",
    "The tallest mountain on Earth is",
    "A binary search on a sorted array runs in",
};
/// TP=2 is fixed at two ranks (verbs.cpp).
constexpr std::uint32_t kWorldSize = 2;
/// Both ranks decode greedily, so each can check its own choice.
constexpr float kGreedyTemperature = 0.0F;
/// First id `--prompt-tokens-N` emits. The band is deliberately LOW: the head
/// of a byte-level BPE vocabulary is ordinary text bytes, so a synthetic prompt
/// stays well conditioned instead of probing arbitrary embedding rows, and the
/// walk starts at 1 rather than 0.
constexpr std::int32_t kSyntheticFirstToken = 1;
/// `--prompt-tokens-N` walks the low band with this stride. It is coprime with
/// `kSyntheticSpan` (512) so the sequence visits every id in the band before it
/// repeats, which keeps a long synthetic prompt from degenerating into a
/// trivially periodic one.
constexpr std::size_t kSyntheticStride = 5;
/// Width of the low band `--prompt-tokens-N` walks. Clamped to `vocab - 1` at
/// run time so the emitted ids are always inside the embedding.
constexpr std::size_t kSyntheticSpan = 512;
/// Member m's synthetic prompt starts this far into the walk, so the members
/// carry different prompts of equal length.
constexpr std::size_t kSyntheticMemberOffset = 11;
/// Monotonic origin for every `[batched-probe] t=<ms>ms` stamp.
/// `steady_clock` never steps, so a difference between two stamps is a real
/// elapsed duration rather than a wall-clock jump.
const std::chrono::steady_clock::time_point kProbeStart =
    std::chrono::steady_clock::now();

enum ExitReason {
  kOk = 0,
  kTransportFailure = 1,
  kInvalidArguments = 2,
};

/// Milliseconds elapsed since process start.
///
/// The RDMA collective timeout is 30 s and a model load is also tens of
/// seconds, so an unstamped lifecycle log cannot attribute a stall to either.
/// Every lifecycle line therefore carries this stamp, and the absence of a gap
/// between two stamps is itself the evidence that a phase was fast.
std::string ElapsedMs() {
  return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - kProbeStart)
                            .count());
}

void Say(const char* role, const std::string& line) {
  const std::string stamp = ElapsedMs();
  std::printf("[batched-probe %s t=%sms] %s\n", role, stamp.c_str(),
              line.c_str());
  std::fflush(stdout);
}

void Warn(const std::string& line) {
  const std::string stamp = ElapsedMs();
  std::fprintf(stderr, "[batched-probe t=%sms] %s\n", stamp.c_str(),
               line.c_str());
  std::fflush(stderr);
}

void Usage() {
  std::puts(R"(TP=2 batched-decode probe for Flash-Next.

Usage:
  qwen38_flash_next_tp_batched_probe --tp-rank 0 [options]
  qwen38_flash_next_tp_batched_probe --tp-rank 1 --tp-bootstrap-host HOST
                                     [options]

Both ranks prefill every member's prompt, then decode greedily: by default each
step advances every running member in one EvaluateBatch, with --serial one
member at a time with Evaluate. Each rank samples its own logits and prints,
per step, the member set and logit hashes, and per member the tokens and a
checksum. Diff the two ranks' logs, and a batched run against a serial one:
the lines after the time stamps must match.

Modes:
  --serial                Advance the members one at a time instead of in one
                          batch. Both modes print decode_ms and ms_per_step, so
                          a pair of runs measures what batching buys. A member
                          that stops early leaves a serial step with fewer
                          forwards: compare ms_per_forward across runs whose
                          members stop at different steps.
  --width N               Members, from 2 to 8 (default 2). Members beyond the
                          first two use fixed built-in prompts.
  --moe-input-hashes      Print a hash of every MoE input during prefill, to
                          find the first layer at which the ranks diverge.
  --allreduce-bench N     Load no model: time N exchanges each of 1, 2 and 8
                          decode rows and one 512-row prefill batch, and print
                          min/p50/mean/p99 microseconds per exchange. Each is
                          the queued all-reduce the model's per-layer sums use,
                          timed until the stream has drained.

Options (the same on both ranks unless noted):
  --tp-rank 0|1           This rank. Required.
  --tp-bootstrap-host HOST  Rank 0's address; required on rank 1. Rank 0
                          listens on all interfaces.
  --tp-bootstrap-port N   RDMA bootstrap port (default 18515)
  --tp-device N           HIP device index (default 0)
  --tp-gid-index N        RDMA GID index (default 0)
  --model PATH            First GGUF shard, required; a model without an MTP
                          sidecar
  --context N             Context (default 4096)
  --max-tokens N          Per-member token budget, nonzero (default 8)
  --prompt-0 TEXT         Member 0 prompt (default: a France question)
  --prompt-1 TEXT         Member 1 prompt (default: a planet question)
  --prompt-file-0 PATH    Read member 0's prompt from a file instead
  --prompt-file-1 PATH    Read member 1's prompt from a file instead
  --prompt-tokens-0 N     Synthesize member 0's prompt as exactly N token ids
                          instead of tokenizing text: ids walked from 1 with
                          stride 5 across the first 512 ids of the vocabulary,
                          member 1 starting 11 steps in, skipping stop tokens.
                          This is how a prompt longer than one prefill step is
                          exercised with an exact token count. Refused together
                          with --prompt-0 or --prompt-file-0.
  --prompt-tokens-1 N     The same for member 1.
  --help                  This text

Limits:
  * Start rank 0 first and rank 1 within 30 s: both RDMA bootstraps give up
    after 30 s. Wrap both ranks in an external timeout.
  * The ranks must also finish loading the model within 30 s of each other:
    they meet at a zero-byte exchange before the prefill, and "pre-prefill
    barrier failed: verbs all-reduce timed out" means one rank loaded more
    slowly. Warm the page cache first on a host that has not read the model
    recently, e.g. cat the model's shards to /dev/null.
  * Both containers need `--network host`; without it rank 0's bootstrap
    listener is unreachable, which surfaces only as a 30 s accept timeout.
  * The RDMA adapter maps fixed IOVA windows, so the ranks run on separate
    hosts with no other RDMA process on either.
  * Prompt plus budget must fit --context.

Exit reasons:
  0 the program ran to completion; agreement is judged by diffing the logs
  1 transport, model or execution failure   2 arguments)");
}

/// Owns the single bound collective scope and releases it on every exit path.
/// `EndOperation` is purely local (verbs.cpp), so this never blocks.
class OperationScope {
public:
  OperationScope(std::shared_ptr<q::rocm::Communicator> communicator,
                 std::uint64_t scope_id)
      : communicator_(std::move(communicator)), scope_id_(scope_id) {}

  ~OperationScope() {
    if (!bound_) {
      return;
    }
    try {
      std::string ignored;
      (void)communicator_->EndOperation(scope_id_, &ignored);
    } catch (...) {
    }
  }

  OperationScope(const OperationScope&) = delete;
  OperationScope& operator=(const OperationScope&) = delete;

  [[nodiscard]] bool Begin(std::string* error) {
    if (!communicator_->BeginOperation(scope_id_, error)) {
      return false;
    }
    bound_ = true;
    return true;
  }

  [[nodiscard]] bool End(std::string* error) {
    if (!bound_) {
      return true;
    }
    bound_ = false;
    return communicator_->EndOperation(scope_id_, error);
  }

private:
  std::shared_ptr<q::rocm::Communicator> communicator_;
  const std::uint64_t scope_id_;
  bool bound_{false};
};

/// Records the byte size of every exchange this rank issues while a recording
/// window is open.
///
/// `Model::Load` wires the executor's exchanges to this object
/// (`Executor::TwoRankAllReduce` and `TwoRankSplitReduce`), so the sizes
/// collected here are the byte sequence the peer sees; both ranks print theirs
/// after the prefill, and the two lines must match. Every method delegates, so
/// the wire behaviour is exactly the inner communicator's; the decorator adds
/// no exchange and drops none. No lock: the probe drives one model from one
/// thread.
class CollectiveTrace : public q::rocm::Communicator {
public:
  explicit CollectiveTrace(std::shared_ptr<q::rocm::Communicator> inner)
      : inner_(std::move(inner)) {}

  [[nodiscard]] std::uint32_t rank() const noexcept override {
    return inner_->rank();
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept override {
    return inner_->world_size();
  }
  [[nodiscard]] int device_index() const noexcept override {
    return inner_->device_index();
  }
  [[nodiscard]] bool BeginOperation(std::uint64_t scope_id,
                                    std::string* error) override {
    return inner_->BeginOperation(scope_id, error);
  }
  [[nodiscard]] bool EndOperation(std::uint64_t scope_id,
                                  std::string* error) override {
    return inner_->EndOperation(scope_id, error);
  }
  [[nodiscard]] const float* ExchangePartial(const float* data,
                                             std::size_t bytes,
                                             hipStream_t stream,
                                             std::string* error) override {
    if (recording_) {
      sizes_.push_back(bytes);
    }
    return inner_->ExchangePartial(data, bytes, stream, error);
  }
  [[nodiscard]] bool StartPartial(const float* data, std::size_t bytes,
                                  hipStream_t stream,
                                  std::string* error) override {
    if (recording_) {
      sizes_.push_back(bytes);
    }
    return inner_->StartPartial(data, bytes, stream, error);
  }
  [[nodiscard]] const float* FinishPartial(std::size_t bytes,
                                           std::string* error) override {
    return inner_->FinishPartial(bytes, error);
  }
  [[nodiscard]] bool QueuePartial(std::size_t bytes, QueuedPartial* partial,
                                  std::string* error) override {
    if (recording_) {
      sizes_.push_back(bytes);
    }
    return inner_->QueuePartial(bytes, partial, error);
  }
  [[nodiscard]] bool CheckQueued(std::string* error) override {
    return inner_->CheckQueued(error);
  }

  /// Starts a recording window. The prefill is the only window the probe opens.
  void BeginRecording() {
    sizes_.clear();
    recording_ = true;
  }
  /// Closes the window and returns the byte size of every exchange issued
  /// inside it, in order.
  [[nodiscard]] std::vector<std::size_t> EndRecording() {
    recording_ = false;
    return sizes_;
  }

private:
  std::shared_ptr<q::rocm::Communicator> inner_;
  std::vector<std::size_t> sizes_;
  bool recording_{false};
};

bool ParseUint(std::string_view text, std::uint32_t* value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool ParsePort(std::string_view text, std::uint16_t* value) {
  std::uint32_t parsed = 0;
  if (!ParseUint(text, &parsed) || parsed == 0 || parsed > 65535) {
    return false;
  }
  *value = static_cast<std::uint16_t>(parsed);
  return true;
}

bool ReadPromptFile(const std::string& path, std::string* prompt) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    return false;
  }
  *prompt = buffer.str();
  // Prompts are line oriented: a trailing newline is presentation, not a token.
  while (!prompt->empty() &&
         (prompt->back() == '\n' || prompt->back() == '\r')) {
    prompt->pop_back();
  }
  return true;
}

/// `65536B x48, 30464B x48`, the exchange byte sequence run-length encoded,
/// so a reader sees the prefill batches rather than a long integer list.
std::string ByteRunsText(const std::vector<std::size_t>& sizes) {
  std::string text;
  for (std::size_t index = 0; index < sizes.size();) {
    std::size_t run = index;
    while (run < sizes.size() && sizes[run] == sizes[index]) {
      ++run;
    }
    if (!text.empty()) {
      text += ", ";
    }
    text += std::to_string(sizes[index]) + "B x" + std::to_string(run - index);
    index = run;
  }
  return text.empty() ? "<none>" : text;
}

/// The synthetic prompt `--prompt-tokens-N` asks for: exactly `count` ids,
/// deterministic, and different for each member.
///
/// A text prompt's token count is a property of the vocabulary, so the same
/// `--prompt-0` can produce a different id count, and therefore a different
/// prefill, on any vocabulary change. The ids stay inside a low band of real
/// byte-level tokens so the embeddings are well conditioned, and an id the
/// model reports as a stop token is stepped over rather than dropped, because
/// the count is the contract.
std::vector<std::int32_t> SyntheticPrompt(const q::Model& model,
                                          std::size_t count,
                                          std::size_t member) {
  const std::size_t vocab = model.VocabSize();
  if (vocab <= static_cast<std::size_t>(kSyntheticFirstToken)) {
    // No usable id. An empty prompt is refused by the caller, which is the same
    // exit the tokenizer path reports for an empty prompt, so there is no
    // unrepresentable case left.
    return {};
  }
  // Clamped so the walk stays inside the embedding: the largest id it can
  // produce is `kSyntheticFirstToken + span - 1`, and `span <= vocab - 1`.
  const std::size_t span = std::min(kSyntheticSpan, vocab - 1);
  const std::size_t offset = member * kSyntheticMemberOffset;
  std::vector<std::int32_t> tokens;
  tokens.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::size_t id =
        kSyntheticFirstToken + ((index + offset) * kSyntheticStride) % span;
    // Step over a stop id rather than dropping the position: the count is the
    // contract. `id + 1 < vocab` keeps the walk from ever naming an id at or
    // past the end of the embedding.
    while (id + 1 < vocab && model.IsStopToken(static_cast<std::int32_t>(id))) {
      ++id;
    }
    tokens.push_back(static_cast<std::int32_t>(id));
  }
  return tokens;
}

/// One member's decode, kept for the report.
struct MemberTrace {
  std::vector<std::int32_t> tokens;
  std::size_t decode_forwards{0};
  bool stopped_on_token{false};
  bool stopped_on_budget{false};
};

/// `--moe-input-hashes`: hashes every MoE input while armed, so the ranks'
/// replicated hidden states can be compared layer by layer. Each hash drains
/// the stream and copies the input to the host, so it is armed only around
/// prefill.
struct MoeInputHashes {
  bool armed{false};
  std::vector<std::uint64_t> hashes;
  std::vector<std::size_t> sizes;
  std::vector<std::uint8_t> host;

  void Observe(const float* data, std::size_t bytes, hipStream_t stream) {
    if (!armed) {
      return;
    }
    host.resize(bytes);
    std::uint64_t hash = 0;
    if (hipStreamSynchronize(stream) == hipSuccess &&
        hipMemcpy(host.data(), data, bytes, hipMemcpyDeviceToHost) ==
            hipSuccess) {
      hash = 1469598103934665603ULL;
      for (const std::uint8_t byte : host) {
        hash ^= byte;
        hash *= 1099511628211ULL;
      }
    }
    hashes.push_back(hash);
    sizes.push_back(bytes);
  }
};

/// FNV-1a over the exact bytes of a logit row, so two ranks that compute the
/// same row bit for bit print the same value.
std::uint64_t LogitChecksum(std::span<const float> logits) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(logits.data());
  for (std::size_t index = 0; index < logits.size_bytes(); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

/// A 64-bit FNV-1a over raw token bytes, so a cross-rank comparison has one
/// token to diff instead of two id lists. Not a cryptographic digest: it only
/// has to detect an unintended difference.
std::uint64_t TokenChecksum(std::span<const std::int32_t> tokens) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::int32_t token : tokens) {
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<std::uint8_t>(
          (static_cast<std::uint32_t>(token) >> (8 * byte)) & 0xFFU);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::string TokenListText(std::span<const std::int32_t> tokens) {
  std::string text;
  for (const std::int32_t token : tokens) {
    if (!text.empty()) {
      text += ',';
    }
    text += std::to_string(token);
  }
  return text;
}

/// Tokenizes the members' prompts: members 0 and 1 from their
/// options, the rest from `kExtraPrompts`. Every prompt must be nonempty and
/// fit the context with the token budget.
bool BatchedPrompts(
    const q::Model& model, const std::string (&prompt)[kPromptOptions],
    const std::optional<std::uint32_t> (&prompt_tokens)[kPromptOptions],
    std::uint32_t width, std::uint32_t budget, std::uint32_t context,
    std::vector<std::vector<std::int32_t>>* out) {
  out->assign(width, {});
  for (std::size_t index = 0; index < width; ++index) {
    auto& tokens = (*out)[index];
    if (index < kPromptOptions) {
      tokens = prompt_tokens[index].has_value()
                   ? SyntheticPrompt(model, *prompt_tokens[index], index)
                   : model.Tokenize(prompt[index]);
    } else {
      tokens = model.Tokenize(kExtraPrompts[index - kPromptOptions]);
    }
    if (tokens.empty()) {
      Warn("member " + std::to_string(index) + " prompt tokenized empty");
      return false;
    }
    if (tokens.size() + budget > static_cast<std::size_t>(context)) {
      Warn("member " + std::to_string(index) + " prompt has " +
           std::to_string(tokens.size()) + " tokens, plus --max-tokens " +
           std::to_string(budget) + ", which exceeds the negotiated context " +
           std::to_string(context));
      return false;
    }
  }
  return true;
}

/// Prefills every member, then decodes all of them greedily on both ranks.
///
/// Both ranks run this identical program, so the log is the whole comparison:
/// each rank prints the member set and logit hashes of every step, and each
/// member's tokens and checksum. The member set is checked as well as the
/// tokens, because a member that stops early drops out of later steps, and if
/// the ranks ever disagreed about that, their exchanges would diverge.
///
/// `serial` is the baseline for the same program: each step advances the
/// running members one at a time instead of in one forward, so a batched and
/// a serial run differ only in batching. Both report the decode time spent in
/// the advances alone, excluding sampling and logging.
int RunBatched(const char* role, const std::shared_ptr<q::Model>& model,
               std::span<const std::vector<std::int32_t>> prompt,
               std::uint32_t budget, std::uint32_t context,
               const std::shared_ptr<CollectiveTrace>& collectives, bool serial,
               MoeInputHashes* moe_inputs, std::string* error) {
  // Both ranks bind the same scope id. Nothing on the wire negotiates it, so
  // it is a constant agreed by construction, and the log prints it so a
  // reader can confirm both sides bound the same value.
  constexpr std::uint64_t kBatchedScope = 1;

  const std::size_t width = prompt.size();
  std::vector<std::unique_ptr<q::Session>> session(width);
  for (std::size_t index = 0; index < width; ++index) {
    session[index] = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, context, error);
    if (!session[index]) {
      return kTransportFailure;
    }
  }

  OperationScope scope(collectives, kBatchedScope);
  if (!scope.Begin(error)) {
    Warn(std::string(role) + " could not bind batched scope " +
         std::to_string(kBatchedScope) + ": " + *error);
    return kTransportFailure;
  }
  Say(role,
      "batched scope bound: " + std::to_string(kBatchedScope) +
          ", prefill_capacity=" + std::to_string(model->PrefillCapacity()) +
          " budget=" + std::to_string(budget));

  // Prefill stays per member: it has no batch form in the runner or the
  // engine, so the program prefills member 0, then member 1, and so on. Only
  // the decode advance is batched.
  std::vector<MemberTrace> trace(width);
  // A zero-byte exchange is a barrier: the ranks load the model at different
  // speeds, and without it the first prefill exchange would bill the slower
  // load to the faster rank's prefill time.
  if (collectives->ExchangePartial(nullptr, 0, nullptr, error) == nullptr) {
    Warn(std::string(role) + " pre-prefill barrier failed: " + *error);
    return kTransportFailure;
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  if (moe_inputs != nullptr) {
    moe_inputs->armed = true;
  }
  collectives->BeginRecording();
  for (std::size_t index = 0; index < width; ++index) {
    if (!session[index]->Sync(prompt[index], error)) {
      (void)collectives->EndRecording();
      Warn(std::string(role) + " member " + std::to_string(index) +
           " batched prefill failed: " + *error);
      return kTransportFailure;
    }
  }
  if (moe_inputs != nullptr) {
    moe_inputs->armed = false;
    for (std::size_t index = 0; index < moe_inputs->hashes.size(); ++index) {
      Say(role, "moe-input " + std::to_string(index) +
                    " bytes=" + std::to_string(moe_inputs->sizes[index]) +
                    " hash=" + std::to_string(moe_inputs->hashes[index]));
    }
  }
  const auto prefill_elapsed = std::chrono::steady_clock::now() - prefill_start;
  const double prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_elapsed).count();
  std::string prompt_sizes;
  std::size_t prefill_tokens = 0;
  for (const auto& member_prompt : prompt) {
    prompt_sizes += (prompt_sizes.empty() ? "" : "/") +
                    std::to_string(member_prompt.size());
    prefill_tokens += member_prompt.size();
  }
  const auto prefill_sizes = collectives->EndRecording();
  Say(role, "prefill complete: " + std::to_string(prefill_sizes.size()) +
                " exchanges of [" + ByteRunsText(prefill_sizes) +
                "], prompt sizes " + prompt_sizes + " tokens");
  char prefill_timing[128];
  std::snprintf(prefill_timing, sizeof(prefill_timing),
                "prefill_ms=%.1f prefill_tokens=%zu prefill_tokens_per_s=%.1f",
                prefill_ms, prefill_tokens,
                prefill_ms <= 0.0 ? 0.0 : prefill_tokens * 1000.0 / prefill_ms);
  Say(role, std::string("prefill timing: ") + prefill_timing);

  const gufo::sampling::SamplingConfig sampling{
      .temperature = kGreedyTemperature,
  };
  std::vector<std::unique_ptr<gufo::sampling::SamplerState>> samplers;
  for (std::size_t index = 0; index < width; ++index) {
    samplers.push_back(
        std::make_unique<gufo::sampling::SamplerState>(sampling));
  }

  // A member stops at a stop token, or once it holds its budget without
  // evaluating the token that reached it, as the distributed runner does
  // (`final_token_advance_required` is false).
  std::chrono::steady_clock::duration decode_time{};
  std::size_t timed_steps = 0;
  std::size_t forwards = 0;
  std::size_t advanced_tokens = 0;
  for (std::size_t step = 0; step < budget; ++step) {
    std::vector<q::Session::AdvanceRequest> requests(width);
    std::vector<std::size_t> members(width);
    std::size_t rows = 0;
    std::string included;
    std::string logit_hashes;
    for (std::size_t index = 0; index < width; ++index) {
      if (!trace[index].stopped_on_token &&
          trace[index].tokens.size() >= budget) {
        trace[index].stopped_on_budget = true;
      }
      if (trace[index].stopped_on_token || trace[index].stopped_on_budget) {
        continue;
      }
      const auto logits = session[index]->Logits();
      if (logits.empty()) {
        *error = "session " + std::to_string(index) +
                 " has no logits to sample from";
        return kTransportFailure;
      }
      // The logits a rank samples from: ranks that agree bit for bit here
      // hold the same hidden state, so the first differing step localizes a
      // numerical divergence even before any sampled token differs.
      logit_hashes += (logit_hashes.empty() ? "" : ",") +
                      std::to_string(LogitChecksum(logits));
      const auto sampled =
          static_cast<std::int32_t>(samplers[index]->Sample(logits));
      if (model->IsStopToken(sampled)) {
        trace[index].stopped_on_token = true;
        continue;
      }
      trace[index].tokens.push_back(sampled);
      if (trace[index].tokens.size() >= budget) {
        trace[index].stopped_on_budget = true;
        continue;
      }
      requests[rows] = q::Session::AdvanceRequest{
          .session = session[index].get(),
          .token = sampled,
      };
      members[rows] = index;
      included += static_cast<char>('a' + index);
      ++rows;
    }
    // The per-step member set is the schedule. It must be identical on both
    // ranks, so it is logged rather than inferred.
    Say(role, "batched step " + std::to_string(step) +
                  " rows=" + std::to_string(rows) + " members=" +
                  (included.empty() ? std::string("none") : included) +
                  " logits=" + logit_hashes);
    if (rows == 0) {
      break;
    }
    const auto step_start = std::chrono::steady_clock::now();
    if (serial) {
      for (std::size_t row = 0; row < rows; ++row) {
        if (!requests[row].session->Evaluate(requests[row].token, error)) {
          Warn(std::string(role) + " serial advance failed at step " +
               std::to_string(step) + ": " + *error);
          return kTransportFailure;
        }
      }
    } else if (!q::Session::EvaluateBatch(std::span(requests).first(rows),
                                          error)) {
      Warn(std::string(role) + " batched advance failed at step " +
           std::to_string(step) + ": " + *error);
      return kTransportFailure;
    }
    decode_time += std::chrono::steady_clock::now() - step_start;
    ++timed_steps;
    forwards += serial ? rows : 1;
    advanced_tokens += rows;
    // Only the members in this step's advance took part in a forward.
    for (std::size_t row = 0; row < rows; ++row) {
      ++trace[members[row]].decode_forwards;
    }
  }
  const double decode_ms =
      std::chrono::duration<double, std::milli>(decode_time).count();
  char timing[160];
  std::snprintf(timing, sizeof(timing),
                "decode_ms=%.1f ms_per_step=%.2f ms_per_forward=%.2f "
                "tokens_per_s=%.1f",
                decode_ms, timed_steps == 0 ? 0.0 : decode_ms / timed_steps,
                forwards == 0 ? 0.0 : decode_ms / forwards,
                decode_ms <= 0.0 ? 0.0 : advanced_tokens * 1000.0 / decode_ms);
  Say(role, std::string(serial ? "serial" : "batched") +
                " decode: steps=" + std::to_string(timed_steps) +
                " forwards=" + std::to_string(forwards) + " advanced_tokens=" +
                std::to_string(advanced_tokens) + " " + timing);

  if (!scope.End(error)) {
    Warn(std::string(role) + " could not release batched scope: " + *error);
    return kTransportFailure;
  }
  for (std::size_t index = 0; index < width; ++index) {
    Say(role,
        "batched member " + std::to_string(index) +
            " tokens=" + TokenListText(trace[index].tokens) +
            " count=" + std::to_string(trace[index].tokens.size()) +
            " checksum=" + std::to_string(TokenChecksum(trace[index].tokens)) +
            " decode_forwards=" + std::to_string(trace[index].decode_forwards) +
            (trace[index].stopped_on_token ? " stop_token" : "") +
            (trace[index].stopped_on_budget ? " budget" : ""));
  }
  return kOk;
}

/// `--allreduce-bench N`: times N back-to-back exchanges per payload with no
/// model loaded, on both ranks in lockstep. The payloads are one, two and eight
/// Flash-Next decode rows and one 512-row prefill batch. Each exchange is the
/// queued all-reduce the model's per-layer sums use
/// (`Executor::TwoRankAllReduce`): the GPU stages its partial, a communicator
/// thread writes it and waits for the peer's, and the GPU adds it. It is timed
/// from queuing until the stream has drained, so nothing overlaps it; model
/// compute is absent by construction.
int RunAllReduceBench(
    const char* role,
    const std::shared_ptr<q::rocm::Communicator>& communicator,
    std::uint32_t device, std::uint32_t iterations) {
  // One Flash-Next hidden row: 2560 floats, the decode all-reduce unit.
  constexpr std::uint32_t kHidden = 2560;
  constexpr std::size_t kRowBytes = kHidden * sizeof(float);
  constexpr std::array<std::size_t, 4> kRows{1, 2, 8, 512};
  constexpr std::uint64_t kBenchScope = 1;
  if (hipSetDevice(static_cast<int>(device)) != hipSuccess) {
    Warn(std::string(role) + " allreduce bench could not select the device");
    return kTransportFailure;
  }
  float* data = nullptr;
  hipStream_t stream = nullptr;
  const std::size_t max_bytes = kRows.back() * kRowBytes;
  if (hipMalloc(reinterpret_cast<void**>(&data), max_bytes) != hipSuccess ||
      hipMemset(data, 0, max_bytes) != hipSuccess ||
      hipStreamCreate(&stream) != hipSuccess) {
    Warn(std::string(role) + " allreduce bench could not allocate buffers");
    if (data != nullptr) {
      (void)hipFree(data);
    }
    return kTransportFailure;
  }
  const auto release = [&] {
    (void)hipStreamDestroy(stream);
    (void)hipFree(data);
  };
  const auto all_reduce =
      q::rocm::Executor::TwoRankAllReduce(communicator, kHidden);
  const auto reduce_status =
      q::rocm::Executor::TwoRankReduceStatus(communicator);
  std::string error;
  if (!communicator->BeginOperation(kBenchScope, &error)) {
    Warn(std::string(role) + " allreduce bench scope bind failed: " + error);
    release();
    return kTransportFailure;
  }
  for (const std::size_t rows : kRows) {
    const std::size_t bytes = rows * kRowBytes;
    std::vector<double> micros;
    micros.reserve(iterations);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      if (!all_reduce(data, bytes, stream, &error)) {
        Warn(std::string(role) + " allreduce bench failed to queue " +
             std::to_string(bytes) + " B: " + error);
        release();
        return kTransportFailure;
      }
      if (hipStreamSynchronize(stream) != hipSuccess) {
        Warn(std::string(role) + " allreduce bench stream failed");
        release();
        return kTransportFailure;
      }
      if (!reduce_status(&error)) {
        Warn(std::string(role) + " allreduce bench failed at " +
             std::to_string(bytes) + " B: " + error);
        release();
        return kTransportFailure;
      }
      micros.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count());
    }
    std::sort(micros.begin(), micros.end());
    double total = 0.0;
    for (const double value : micros) {
      total += value;
    }
    char line[192];
    std::snprintf(line, sizeof(line),
                  "allreduce rows=%zu bytes=%zu n=%u min_us=%.1f p50_us=%.1f "
                  "mean_us=%.1f p99_us=%.1f",
                  rows, bytes, iterations, micros.front(),
                  micros[micros.size() / 2], total / micros.size(),
                  micros[(micros.size() * 99) / 100]);
    Say(role, line);
  }
  if (!communicator->EndOperation(kBenchScope, &error)) {
    Warn(std::string(role) + " allreduce bench scope release failed: " + error);
    release();
    return kTransportFailure;
  }
  release();
  return kOk;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::uint32_t> rank;
  std::string model_path;
  std::string prompt[kPromptOptions] = {"The capital of France is",
                                        "The largest planet in the solar "
                                        "system is"};
  std::string prompt_file[kPromptOptions];
  /// Whether `--prompt-N` was given for a member, so a default text prompt is
  /// not mistaken for a requested one when `--prompt-tokens-N` is also present.
  bool prompt_given[kPromptOptions] = {false, false};
  /// `--prompt-tokens-N`: synthesize this member's prompt as exactly N token
  /// ids instead of tokenizing text, so its length is known before the model
  /// is loaded.
  std::optional<std::uint32_t> prompt_tokens[kPromptOptions];
  std::uint32_t budget = 8;
  std::uint32_t context = 4096;
  std::uint32_t width = kPromptOptions;
  std::uint32_t device = 0;
  std::uint32_t gid = 0;
  /// `--serial`: the baseline, advancing the members one at a time.
  bool serial = false;
  /// `--allreduce-bench N`: time N exchanges per payload with no model.
  std::uint32_t allreduce_bench = 0;
  /// `--moe-input-hashes`: print a hash of every MoE input during prefill.
  bool moe_input_hashes = false;
  std::uint16_t bootstrap_port = 18515;
  std::string bootstrap_host;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (arg == "--help" || arg == "-h") {
      Usage();
      return kOk;
    } else if (arg == "--tp-rank") {
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed) || parsed >= kWorldSize) {
        Warn("--tp-rank must be 0 or 1");
        return kInvalidArguments;
      }
      rank = parsed;
    } else if (arg == "--model") {
      model_path = next();
    } else if (arg == "--max-tokens") {
      if (!ParseUint(next(), &budget)) {
        Warn("--max-tokens requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--prompt-0") {
      prompt[0] = next();
      prompt_given[0] = true;
    } else if (arg == "--prompt-1") {
      prompt[1] = next();
      prompt_given[1] = true;
    } else if (arg == "--prompt-file-0") {
      prompt_file[0] = next();
    } else if (arg == "--prompt-file-1") {
      prompt_file[1] = next();
    } else if (arg == "--prompt-tokens-0" || arg == "--prompt-tokens-1") {
      // One parse branch for both members: the value is a token COUNT, so the
      // name carries the member and nothing else in the argument is read.
      const std::size_t index = arg == "--prompt-tokens-0" ? 0 : 1;
      std::uint32_t parsed = 0;
      if (!ParseUint(next(), &parsed)) {
        // A missing, negative, trailing-garbage or non-numeric value all land
        // here. None of them is a count, and none may fall back to the text
        // prompt: that would silently prefill a length nobody asked for.
        Warn(arg + " requires an unsigned token count");
        return kInvalidArguments;
      }
      if (parsed == 0) {
        // Refused rather than clamped: `Session::Sync` refuses a zero-length
        // prompt, and a silent clamp would report a prefill of a length nobody
        // asked for.
        Warn(arg + " must be at least 1 token");
        return kInvalidArguments;
      }
      prompt_tokens[index] = parsed;
    } else if (arg == "--tp-bootstrap-port") {
      if (!ParsePort(next(), &bootstrap_port)) {
        Warn("--tp-bootstrap-port is out of range");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-bootstrap-host") {
      bootstrap_host = next();
    } else if (arg == "--tp-device") {
      if (!ParseUint(next(), &device)) {
        Warn("--tp-device requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--tp-gid-index") {
      if (!ParseUint(next(), &gid)) {
        Warn("--tp-gid-index requires an unsigned integer");
        return kInvalidArguments;
      }
    } else if (arg == "--serial") {
      serial = true;
    } else if (arg == "--width") {
      if (!ParseUint(next(), &width) || width < kPromptOptions ||
          width > kMaxWidth) {
        Warn("--width requires a member count from 2 to 8");
        return kInvalidArguments;
      }
    } else if (arg == "--moe-input-hashes") {
      moe_input_hashes = true;
    } else if (arg == "--allreduce-bench") {
      if (!ParseUint(next(), &allreduce_bench) || allreduce_bench == 0) {
        Warn("--allreduce-bench requires a positive iteration count");
        return kInvalidArguments;
      }
    } else if (arg == "--context") {
      if (!ParseUint(next(), &context)) {
        Warn("--context requires an unsigned integer");
        return kInvalidArguments;
      }
    } else {
      Warn("unknown argument: " + arg);
      return kInvalidArguments;
    }
  }

  if (!rank.has_value()) {
    Warn("--tp-rank 0|1 is required; see --help");
    return kInvalidArguments;
  }
  const std::string role = "rank" + std::to_string(*rank);
  if (*rank != 0 && bootstrap_host.empty()) {
    Warn("--tp-rank 1 requires --tp-bootstrap-host with rank 0's address");
    return kInvalidArguments;
  }
  if (device > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      gid > std::numeric_limits<std::uint8_t>::max()) {
    Warn("--tp-device or --tp-gid-index is out of range");
    return kInvalidArguments;
  }
  if (allreduce_bench != 0 &&
      (serial || moe_input_hashes || width != kPromptOptions)) {
    Warn(
        "--allreduce-bench loads no model, so it takes no --serial, --width "
        "or --moe-input-hashes");
    return kInvalidArguments;
  }
  if (allreduce_bench == 0) {
    for (std::size_t index = 0; index < kPromptOptions; ++index) {
      if (!prompt_file[index].empty() &&
          !ReadPromptFile(prompt_file[index], &prompt[index])) {
        Warn("prompt file for member " + std::to_string(index) +
             " is unreadable: " + prompt_file[index]);
        return kInvalidArguments;
      }
    }
    if (model_path.empty() || context == 0 || budget == 0) {
      Warn(
          "--model, a nonzero --context and a nonzero --max-tokens are "
          "required");
      return kInvalidArguments;
    }
    // Checked before the RDMA rendezvous and the model load, which a
    // budget that cannot fit the context would otherwise waste.
    if (budget > context) {
      Warn("--max-tokens " + std::to_string(budget) + " exceeds --context " +
           std::to_string(context));
      return kInvalidArguments;
    }
    for (std::size_t index = 0; index < kPromptOptions; ++index) {
      const std::string option = "--prompt-tokens-" + std::to_string(index);
      // Two sources for one member's prompt is an argument mistake, not a
      // precedence question: whichever won silently, the run would exercise a
      // prompt the operator did not ask for.
      if (prompt_tokens[index].has_value() &&
          (prompt_given[index] || !prompt_file[index].empty())) {
        Warn(option + " cannot be combined with --prompt-" +
             std::to_string(index) + " or --prompt-file-" +
             std::to_string(index) + "; a member has exactly one prompt");
        return kInvalidArguments;
      }
      // A synthetic prompt's length is known here, so it is checked before any
      // RDMA peer is touched. A text or file prompt is checked after the load,
      // where its token count first exists.
      if (prompt_tokens[index].has_value() &&
          static_cast<std::uint64_t>(*prompt_tokens[index]) + budget >
              context) {
        Warn("member " + std::to_string(index) + " " + option + " " +
             std::to_string(*prompt_tokens[index]) + " plus --max-tokens " +
             std::to_string(budget) + " exceeds --context " +
             std::to_string(context));
        return kInvalidArguments;
      }
      if (!prompt_tokens[index].has_value() && prompt[index].empty()) {
        Warn("member " + std::to_string(index) + " has an empty prompt");
        return kInvalidArguments;
      }
    }
  }

  // The RDMA pair forms first: both bootstraps block for up to 30 s, so the
  // model must not be mapped before the pair exists.
  std::string error;
  const q::rocm::IbrverbsConfig rdma{
      .rank = *rank,
      .world_size = kWorldSize,
      // An empty bootstrap host makes rank 0 the listener.
      .bootstrap_host = *rank == 0 ? std::string() : bootstrap_host,
      .bootstrap_port = bootstrap_port,
      .device_index = device,
      .gid_index = gid,
  };
  auto communicator = q::rocm::CreateIbrverbsCommunicator(rdma, &error);
  if (!communicator) {
    Warn("RDMA communicator failed: " + error);
    return kTransportFailure;
  }
  Say(role.c_str(), "RDMA peer established on bootstrap port " +
                        std::to_string(bootstrap_port));
  if (allreduce_bench != 0) {
    return RunAllReduceBench(role.c_str(), communicator, device,
                             allreduce_bench);
  }

  Say(role.c_str(), std::string(serial ? "serial" : "batched") + " decode of " +
                        std::to_string(width) + " members");
  auto collectives = std::make_shared<CollectiveTrace>(std::move(communicator));
  q::ModelOptions options{
      .max_context = context,
      .mtp_model_path = "",
      .max_draft_tokens = q::kMaxMtpDraftTokens,
      .vision_model_path = "",
      .decode_concurrency = 1,
      .tp_rank = *rank,
      .tp_world_size = kWorldSize,
      .hip_device = static_cast<int>(device),
      .communicator = collectives,
  };
  auto moe_inputs = std::make_shared<MoeInputHashes>();
  if (moe_input_hashes) {
    options.moe_observer = [moe_inputs](const float* data, std::size_t bytes,
                                        hipStream_t stream) {
      moe_inputs->Observe(data, bytes, stream);
    };
  }
  auto model = q::Model::Load(model_path, options, &error);
  if (!model) {
    Warn("model load failed: " + error);
    return kTransportFailure;
  }
  if (model->HasMtp()) {
    Warn("the model reports an MTP sidecar, which this probe refuses");
    return kInvalidArguments;
  }
  std::vector<std::vector<std::int32_t>> prompts;
  if (!BatchedPrompts(*model, prompt, prompt_tokens, width, budget, context,
                      &prompts)) {
    return kInvalidArguments;
  }
  return RunBatched(role.c_str(), model, prompts, budget, context, collectives,
                    serial, moe_input_hashes ? moe_inputs.get() : nullptr,
                    &error);
}
