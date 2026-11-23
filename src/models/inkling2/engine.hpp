// Inkling 2 Mini forward pass on HIP (milestone M1: simple, correct, slow).
// Implements notes/B-forward-spec.md sections 2-3: 48 layer visits per token
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
#include <string>
#include <vector>

namespace gufo::mem {
class Region;
class SessionState;
class StateAccount;
}  // namespace gufo::mem

#include "src/models/inkling2/config.hpp"
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
class SessionStateAccess;  // state_layout.hpp (M2 cache layer)

// ---- M2 hook (ext-p): session/cache layer <-> engine
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

/// Layered prefill request for one Forward call (M2): split the pass into
/// `groups` groups of visits of similar cost and call `yield` between them. The
/// engine mutex is released and the transient activations are spilled around
/// `yield`, so it may run Forward on OTHER sessions.
struct PrefillYield {
  std::size_t groups = 1;
  std::function<void()> yield;
};
// ---- end M2 hook
// -------------------------------------------------------------------------------

// ---- M5 hook: image rows
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
// ---- end M5 hook
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
  [[nodiscard]] const std::vector<float>& logits() const { return logits_; }
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
  friend class SessionStateAccess;  // M2 hook: export/import of the per-visit
                                    // state
  Session() = default;
  SessionHooks* hooks_ = nullptr;  // M2 hook
  std::size_t pos_ = 0;
  std::size_t verify_rows_ = 0;  // rows of an uncommitted verification
  std::size_t bytes_ = 0;
  std::vector<float> logits_;
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
  [[nodiscard]] std::uint32_t ubatch() const { return ubatch_; }
  [[nodiscard]] std::uint32_t max_context() const { return max_context_; }
  /// Rows of one sliding-window ring (window + ubatch); the cache layer (M2)
  /// sizes against it.
  [[nodiscard]] std::uint32_t ring_rows() const { return 1024 + ubatch_; }

  [[nodiscard]] std::unique_ptr<Session> NewSession();
  void Reset(Session& s);

  /// Receives the logits of `count` consecutive positions starting at `first`
  /// (index into the `tokens` span of the Forward call), `count x vocab_text`
  /// floats, soft-capped.
  using LogitsSink = std::function<void(std::size_t first, std::size_t count,
                                        const float* data)>;

  /// Evaluates `tokens` at the session position (ubatched). Afterwards
  /// `s.logits()` holds the last token's logits. With `sink`, the logits of
  /// every token are also delivered. Throws std::runtime_error on failure or
  /// when the context is exhausted. With `yield` (M2), groups >= 2 and a
  /// callback, the call yields between visit groups. With `rows` (M5), the
  /// listed token positions take the given input rows instead of an embedding.
  void Forward(Session& s, std::span<const std::int32_t> tokens,
               const LogitsSink& sink = {}, const PrefillYield* yield = nullptr,
               std::span<const InputRows> rows = {});

  /// Waits for all queued device work (M2: the cache adapter copies state with
  /// hipMemcpy).
  void Synchronize();

  static constexpr std::size_t kMaxDecodeBatch = 4;
  /// One decode step (one token each) for up to 4 distinct sessions under a
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

  /// Speculative verification. `Verify` evaluates up to VerifyRows() tokens
  /// at the session's next positions in one step; row i's logits (`tokens`
  /// rows x vocab_text, soft-capped) are bit for bit those a one-token decode
  /// of tokens[0..i] would give. The session does not advance until
  /// `CommitVerify(s, keep)` keeps the first `keep` rows (its convolution
  /// histories then equal those of decoding exactly them; the K/V of the
  /// other rows are past the frontier) and takes row keep - 1 of `logits`
  /// (Verify's output) as the session's logits.
  [[nodiscard]] std::size_t VerifyRows(const Session& s) const;
  void Verify(Session& s, std::span<const std::int32_t> tokens,
              std::vector<float>* logits);
  void CommitVerify(Session& s, std::size_t keep,
                    std::span<const float> logits);

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
  };
  [[nodiscard]] StateLayout DeclareState() const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Inkling2Weights weights_;
  std::shared_ptr<const core::GgufReader> reader_;
  std::size_t weight_bytes_ = 0;
  std::uint32_t max_context_ = 0;
  std::uint32_t ubatch_ = 2048;
  Tap tap_;
  std::mutex mutex_;
};

}  // namespace gufo::models::inkling2
