// Inkling 2 Mini forward pass on HIP.
// As described in docs/internal/models/inkling-2-mini/FORWARD.md: 48 layer
// visits per token
// (24 physical layers, the core layers 6..17 applied three times), block
// attention residuals, loop message controllers, short convolutions,
// content-conditioned relative-bias attention with sinks, sigmoid-routed MoE
// and the softcapped LM head. F32 activations; weights stay in their file
// formats; KV is f16.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/state.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::mem {
class Region;
class SessionState;
class StateAccount;
}  // namespace gufo::mem

#include "src/models/inkling2/config.hpp"
#include "src/models/inkling2/dspark.hpp"
#include "src/models/inkling2/weights.hpp"

namespace gufo::models::inkling2 {

class Engine;

/// HIP refuses (and invalidates) a graph capture when another thread issues a
/// copy on the legacy stream while it is open. Graph capture takes this gate
/// exclusively; every device call made by the cache or vision threads without
/// the engine mutex takes it shared (`std::shared_lock` on this mutex). Gufo's
/// own cache-thread HIP calls (pinned host registration) cannot be gated; if
/// one of them invalidates a capture, the decode path falls back to eager
/// execution for that context bucket and width (a failed capture is not
/// retried).
std::shared_mutex& CaptureGate();
class SessionStateAccess;  // state_layout.hpp (cache layer)

// ---- session/cache layer <-> engine
// ------------------------------------------
/// Installed on a Session by the cache adapter. Called on the host, inside
/// Forward, before an ubatch writes rows [first, end) (absolute positions) of
/// the session's state, so the cache can preserve checkpoints that borrow them.
/// Must not call Engine::Forward on the same session.
class SessionHooks {
public:
  virtual ~SessionHooks() = default;
  virtual void BeforeWrite(std::size_t first, std::size_t end) = 0;
};

/// Layered prefill request for one Forward call: split the pass into
/// `groups` groups of visits of similar cost and call `yield` between them. The
/// engine mutex is released and the transient activations are spilled around
/// `yield`, so it may run Forward on OTHER sessions.
struct PrefillYield {
  std::size_t groups = 1;
  std::function<void()> yield;
  /// Polled between visits (about 40 ms apart in a 4,096-token pass); when it
  /// returns true (or throws) Forward waits for the queued work and throws
  /// PrefillCancelled. The session is then mid-pass and must be reset.
  std::function<bool()> cancelled;
};

/// Thrown by Engine::Forward when PrefillYield::cancelled reports true.
struct PrefillCancelled : std::runtime_error {
  PrefillCancelled() : std::runtime_error("generation cancelled") {}
};
// ---- end session/cache layer
// -------------------------------------------------------------------------------

// ---- image rows
// --------------------------------------------------------------------------
/// Input rows that replace the embedding (token_embd + token_embd_norm) of
/// tokens [first, first+count) of one Forward call: `device` points to
/// [count][2304] floats on the device, used as they are (the image projector's
/// output is not normed). The ids at those positions are placeholders and only
/// need to be valid. Rows must stay valid until Forward returns.
struct InputRows {
  std::size_t first = 0;
  std::size_t count = 0;
  const float* device = nullptr;
};
// ---- end image rows
// -------------------------------------------------------------------------------

/// Per-sequence state: 48 visit slots of KV (8 global, 40 sliding-window rings)
/// and 48 slots of short-convolution history. Owned device memory.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::size_t position() const { return pos_; }
  /// Logits (vocab_text floats, soft-capped) of the last evaluated token; empty
  /// before any.
  /// (Kept on the device after a greedy step and copied here when read.)
  [[nodiscard]] const std::vector<float>& logits() const;
  /// Whether there are logits (without copying them).
  [[nodiscard]] bool has_logits() const {
    return logits_stale_ || !logits_.empty();
  }
  [[nodiscard]] std::size_t allocated_bytes() const { return bytes_; }
  /// Charges the global-attention K/V pages (mem::SessionState); null when
  /// the session has no demand-mapped state.
  [[nodiscard]] std::shared_ptr<mem::StateAccount> account() const;
  /// Before the model writes rows [first, end): backs rows [0, end) (plus a key
  /// tile) of every growing K/V region and makes the pages of the written rows
  /// private where a checkpoint shares them. Throws when that fails.
  void BackRows(std::size_t first, std::size_t end);
  /// Backs rows [0, end) without touching shared pages (restore loads).
  void EnsureRows(std::size_t end);
  /// Before a restore of rows [0, rows): releases the pages checkpoints share,
  /// so the loads write the session's own pages.
  void PrepareRestore(std::size_t rows);
  /// Rows backed past a frontier: one prefill key tile.
  static constexpr std::size_t kKvMarginRows = 64;

private:
  friend class Engine;
  friend class SessionStateAccess;  // export/import of the per-visit
                                    // state
  Session() = default;
  SessionHooks* hooks_ = nullptr;
  std::size_t pos_ = 0;
  std::size_t verify_rows_ = 0;  // rows of an uncommitted verification
  std::size_t verify_row0_ = 0;  // its first row in the verification step
  Drafter::State draft_;         // DSpark context (Engine::AttachDrafter)
  std::size_t bytes_ = 0;
  mutable std::vector<float> logits_;
  // The logits of the last token on the device (Engine::Step and
  // CommitVerify without host rows); logits_ is refreshed from them when
  // logits_stale_.
  float* dlogits_ = nullptr;
  mutable bool logits_stale_ = false;
  std::size_t vocab_ = 0;
  struct Slot {
    void* k = nullptr;  // f16 [rows][1024]
    void* v = nullptr;
    float* conv = nullptr;  // [6656][3]
    bool global = false;
    std::size_t rows = 0;  // global: max_context; SWA: ring size
  };
  std::array<Slot, 48> slots_{};
  void* tbl_ = nullptr;  // device table of the 48 slots (fast::SlotEntry[48])
  /// The declared state (Engine::DeclareState), allocated by the core.
  std::unique_ptr<mem::SessionState> state_;
  /// Regions of the global-attention K and V rows, backed as rows are written.
  std::vector<std::shared_ptr<mem::Region>> regions_;
};

class Engine {
public:
  /// Binds and uploads the weights of `reader` (must stay alive as long as the
  /// Engine).
  [[nodiscard]] static std::unique_ptr<Engine> Create(
      std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
      std::string* error);
  ~Engine();

  [[nodiscard]] const Inkling2Config& config() const { return weights_.config; }
  [[nodiscard]] std::size_t weight_bytes() const { return weight_bytes_; }
  [[nodiscard]] std::size_t session_bytes() const;
  /// The part of session_bytes() in demand-mapped regions (0 without them).
  [[nodiscard]] std::size_t demand_mapped_bytes() const;
  /// Tokens of one prefill pass. A pass costs about 115 ms however few tokens
  /// it carries (every routed expert is dequantised once per visit), so wider
  /// passes are faster per token; the activations and every session's window
  /// rings grow with it. Prefill is bit-identical for any split.
  static constexpr std::uint32_t kUbatch = 4096;
  [[nodiscard]] std::uint32_t ubatch() const { return ubatch_; }
  [[nodiscard]] std::uint32_t max_context() const { return max_context_; }
  /// Rows of one sliding-window ring (window + ubatch); the cache layer
  /// sizes against it.
  [[nodiscard]] std::uint32_t ring_rows() const { return 1024 + ubatch_; }

  [[nodiscard]] std::unique_ptr<Session> NewSession();
  void Reset(Session& s);

  /// Receives the logits of `count` consecutive positions starting at `first`
  /// (index into the `tokens` span of the Forward call), `count x vocab_text`
  /// floats, soft-capped.
  using LogitsSink = std::function<void(std::size_t first, std::size_t count,
                                        const float* data)>;

  /// Maximum in-pass checkpoints accepted in a single Forward call.
  static constexpr std::size_t kMaxPassCheckpoints = 8;
  /// Captured private state at an in-pass checkpoint boundary `position` inside
  /// a Forward ubatch ending at `valid_end`. When `conv`/`draft_f` are null or
  /// `logits` is empty, the live session itself stands at `position`.
  struct PrefillCheckpoint {
    std::size_t position = 0;
    std::size_t valid_end = 0;
    const float* conv = nullptr;     // device [48][6656][3]
    std::span<const float> logits;   // host [vocab_text]
    const float* draft_f = nullptr;  // device [2304]
    std::size_t draft_first = 0;
    std::size_t draft_valid = 0;
    std::size_t draft_f_pos = static_cast<std::size_t>(-1);
  };
  using PrefillCheckpointConsumer =
      std::function<void(std::size_t index, const PrefillCheckpoint&)>;
  /// Whether `boundary` can be captured in-pass inside a Forward of `consumed`
  /// tokens starting at `offset` (all window rows for `boundary` remain intact
  /// at the end of its ubatch).
  [[nodiscard]] bool CanCaptureInPass(std::size_t offset, std::size_t consumed,
                                      std::size_t boundary) const;
  /// Temporary device bytes needed to capture one interior boundary in-pass.
  [[nodiscard]] std::size_t PrefillCheckpointScratchBytes() const;

  /// Evaluates prompt `tokens` at the session position (ubatched). Afterwards
  /// `s.logits()` holds the last token's logits. With `sink`, the logits of
  /// every token are also delivered. Throws std::runtime_error on failure or
  /// when the context is exhausted. With `yield`, groups >= 2 and a
  /// callback, the call yields between visit groups. With `rows`, the
  /// listed token positions take the given input rows instead of an embedding.
  /// Every call, however short, takes the prompt arithmetic, and a token's
  /// rows do not depend on how the prompt is split into calls: a restored
  /// prefix plus a tail equals one call over the whole prompt, bit for bit.
  void Forward(Session& s, std::span<const std::int32_t> tokens,
               const LogitsSink& sink = {}, const PrefillYield* yield = nullptr,
               std::span<const InputRows> rows = {},
               std::span<const std::size_t> checkpoints = {},
               const PrefillCheckpointConsumer& on_checkpoint = {});
  /// One decode step of `token` (the decode arithmetic, as Step); afterwards
  /// `s.logits()` holds its logits on the host.
  void Decode(Session& s, std::int32_t token);

  /// Waits for all queued device work (M2: the cache adapter copies state with
  /// hipMemcpy).
  void Synchronize();
  /// Allocates the area a prefill pass spills its live activations to while
  /// another session runs in its yield (13 x ubatch x 2304 F32). A host with
  /// several sessions calls it at load; otherwise the first such yield does.
  void ReserveYieldSpill();

  static constexpr std::size_t kMaxDecodeBatch = 8;
  /// Rows of one session's verification step: the anchor and its drafts
  /// (up to 63 copied ones).
  static constexpr std::size_t kMaxVerifyRows = 64;
  /// Rows of a verification step of several sessions, in all.
  static constexpr std::size_t kMaxBatchVerifyRows = 16;
  /// One decode step (one token each) for up to 8 distinct sessions under a
  /// single lock. The runner's `AdvanceBatch` calls it. M3: one graph launch
  /// reads the weights once for all tokens (GEMV -> small GEMM over the tokens,
  /// MoE per (token, expert) pair, attention per session); each session's
  /// `logits()` holds its result. `BeforeWrite` fires for every session. Falls
  /// back to running the sessions one after another when a layer has no fast
  /// kernels or INKLING2_NOBATCH is set. True when DecodeBatch can run several
  /// sessions as one step (otherwise it runs them one by one).
  [[nodiscard]] bool batch_capable() const;
  void DecodeBatch(std::span<Session* const> sessions,
                   std::span<const std::int32_t> tokens);
  /// Per-row parameters for GPU-side linear sampling (temperature <= 0 skips
  /// the row).
  struct LinearSample {
    float temperature = 0.f;
    float min_p = 0.f;
    double u = 0.0;
  };
  /// DecodeBatch (batch_capable() only) whose logits stay on the device
  /// (`logits()` reads them when asked): `argmax` (when non-null) receives
  /// each row's argmax, `sampled` (with `linear`) receives GPU linear samples
  /// for rows with temperature > 0, and the rows with host[i] set are also
  /// copied to `rows` ([sessions][vocab_text], valid until the next step).
  void DecodeBatchRows(std::span<Session* const> sessions,
                       std::span<const std::int32_t> tokens,
                       std::span<const bool> host, std::span<const float>* rows,
                       std::int32_t* argmax,
                       std::span<const LinearSample> linear = {},
                       std::int32_t* sampled = nullptr);

  /// Speculative verification. `Verify` evaluates up to VerifyRows() tokens
  /// at the session's next positions in one step; row i's logits (`tokens`
  /// rows x vocab_text, soft-capped) are bit for bit those a one-token decode
  /// of tokens[0..i] would give. The session does not advance until
  /// `CommitVerify(s, keep)` keeps the first `keep` rows (its convolution
  /// histories then equal those of decoding exactly them; the K/V of the
  /// other rows are past the frontier) and takes row keep - 1 of `logits`
  /// (Verify's output) as the session's logits.
  [[nodiscard]] std::size_t VerifyRows(const Session& s) const;
  /// `logits` (may be null) receives the rows (engine memory, valid until the
  /// next Verify), `argmax` (may be null) each row's first maximum (as the
  /// greedy sampler picks it) without copying the rows, and `sampled` (with
  /// `linear`) GPU linear samples without copying the rows.
  void Verify(Session& s, std::span<const std::int32_t> tokens,
              std::span<const float>* logits, std::int32_t* argmax = nullptr,
              std::span<const LinearSample> linear = {},
              std::int32_t* sampled = nullptr);
  /// Verification of several distinct sessions in one step: counts[g] rows
  /// of sessions[g] (each within VerifyRows of its session), tokens laid out
  /// session after session, at most kMaxVerifyRows in all; `logits` /
  /// `argmax` / `sampled` cover every row in that order. Each session is then
  /// committed with its own CommitVerify (logits: its rows' subspan) before the
  /// next step. Every row's result is the one a lone Verify of its session
  /// gives.
  void VerifyBatch(std::span<Session* const> sessions,
                   std::span<const std::size_t> counts,
                   std::span<const std::int32_t> tokens,
                   std::span<const float>* logits, std::int32_t* argmax,
                   std::span<const LinearSample> linear = {},
                   std::int32_t* sampled = nullptr);
  /// Copies rows [first, first + count) of the last Verify/DecodeBatchRows
  /// step's device logits into host memory on demand (e.g. when a constrained
  /// greedy argmax is rejected).
  void ReadVerifyRows(std::size_t first, std::size_t count,
                      std::span<const float>* out);
  /// Empty `logits`: row keep - 1 comes from the device.
  void CommitVerify(Session& s, std::size_t keep,
                    std::span<const float> logits);
  /// CommitVerify of several sessions of one verification step at once.
  void CommitVerifyBatch(std::span<Session* const> sessions,
                         std::span<const std::size_t> keeps,
                         std::span<const std::span<const float>> logits);

  /// DSpark drafting. AttachDrafter loads the draft GGUF; call it before the
  /// first NewSession (sessions then also hold the draft context, which every
  /// Forward, DecodeBatch and CommitVerify extends). DraftReady: whether
  /// Propose can run at the session position; Propose writes up to `n`
  /// greedy proposals that follow `anchor` (the token at the session
  /// position).
  [[nodiscard]] bool AttachDrafter(const std::string& path, std::string* error);
  [[nodiscard]] bool has_drafter() const;
  [[nodiscard]] bool DraftReady(const Session& s) const;
  int Propose(Session& s, std::int32_t anchor, int n, std::int32_t* out,
              float* confidence = nullptr, const Drafter::More& more = {},
              const Drafter::Coupled* coupled = nullptr);

  /// One decode step of `token` that leaves the logits on the device (the
  /// session copies them when read) and writes their argmax (the greedy
  /// sampler's choice): Forward of one token without the copy.
  /// Proposals of several distinct sessions in one drafter pass (up to
  /// Drafter::kMaxSessions); each one's drafts are those Propose gives it.
  struct ProposeRequest {
    Session* session = nullptr;
    std::int32_t anchor = 0;
    int n = 0;
    std::int32_t* out = nullptr;
    float* confidence = nullptr;
    Drafter::More more;
    const Drafter::Coupled* coupled = nullptr;
    int done = 0;  // out
  };
  void ProposeBatch(std::span<ProposeRequest> requests);
  void Step(Session& s, std::int32_t token, std::int32_t* argmax);

  /// Optional per-visit hidden-state tap for bisecting against llama.cpp:
  /// called with the visit index (0..47), a label ("l_out" or "ffn_inp") and
  /// the first token's [2304] row.
  using Tap = std::function<void(int visit, const char* what,
                                 const std::vector<float>& row)>;
  void SetTap(Tap tap) { tap_ = std::move(tap); }

private:
  Engine() = default;
  /// The buffers of one session: per visit K and V (global visits grow with
  /// the context, window visits are rings) and the convolution history.
  struct StateLayout {
    mem::StateDeclaration state{Session::kKvMarginRows};
    std::array<mem::StateBufferId, 48> k{}, v{}, conv{};
    mem::StateBufferId table{};
    std::array<mem::StateBufferId, 3> dk{}, dv{};  // with a drafter
    mem::StateBufferId df{};
    mem::StateBufferId logits{};
  };
  [[nodiscard]] StateLayout DeclareState() const;
  /// Measures whether the decode weight prefetch pays on this device and
  /// keeps it only then.
  void CalibrateDecodePrefetch();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Inkling2Weights weights_;
  std::shared_ptr<const core::GgufReader> reader_;
  std::size_t weight_bytes_ = 0;
  std::uint32_t max_context_ = 0;
  std::uint32_t ubatch_ = kUbatch;
  Tap tap_;
  std::mutex mutex_;
};

}  // namespace gufo::models::inkling2
