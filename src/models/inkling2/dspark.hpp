// DSpark drafter of Inkling 2 Mini (Unsloth's dspark-Inkling-2-Mini GGUF,
// architecture `dflash`): the checkpoint's own 3-layer block drafter.
//
// The target's hidden states at four layer visits (the inputs of visits 5, 23
// and 41 and the final normed output) are mixed per drafter layer into
// context K/V rows ("injection", one row per committed token). A draft block
// [anchor, mask x 7] at the next positions attends to that context (two
// sliding-window layers, one global) and to itself (non-causal), and a gated
// RNN Markov head chains the proposals: row i's logits are the target's LM
// head applied to h_i + W_o(latent_i) + b_o, latent_i depending on the previous
// proposal. The drafter only proposes: Engine::Verify checks every proposal
// exactly, so its arithmetic is not part of the model's numerical contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

// The HIP stream type, declared as hip_runtime.h does (this header is also
// read by host-only code).
typedef struct ihipStream_t* hipStream_t;

namespace gufo::models::inkling2 {

class Drafter {
public:
  /// Target tensors the drafter borrows (device pointers).
  struct Target {
    const void* token_embd;        // Q8_0 [201024][2304]
    const float* token_embd_norm;  // [2304]
    const void* output;            // Q8_0 LM head
    int vocab;                     // text rows of the head
    float eps;
    float softcap;  // of the target's logits
  };
  /// Per-session state (buffers declared with the session). Context rows of
  /// positions [first, valid) are injected (after a restore the context
  /// starts again at the restored position, and drafts attend to what there
  /// is); `f` is the Markov context of position f_pos.
  struct State {
    void* k[3] = {};
    void* v[3] = {};
    std::size_t cap[3] = {};  // rows (window layers: rings)
    float* f = nullptr;       // [2304]
    std::size_t first = 0, valid = 0;
    std::size_t f_pos = static_cast<std::size_t>(-1);
  };
  static constexpr int kMaxDraft = 7;
  static constexpr std::size_t kWindow = 1024;
  /// Feature row of one token: [4][2304] floats.
  static constexpr std::size_t kFeatureFloats = 4 * 2304;

  [[nodiscard]] static std::unique_ptr<Drafter> Open(const std::string& path,
                                                     const Target& target,
                                                     std::string* error);
  ~Drafter();
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  [[nodiscard]] std::size_t weight_bytes() const { return weight_bytes_; }
  /// Whether layer l keeps only a window (a ring of `ring` rows).
  [[nodiscard]] static bool WindowLayer(int l) { return l != 1; }

  /// Optional in-pass capture of a row's Markov context vector (`dst`: device
  /// [2304] floats) during Inject.
  struct InjectCapture {
    int row = 0;
    float* dst = nullptr;
  };
  /// Injects the features [n][4][2304] of positions [pos0, pos0 + n). After a
  /// gap (pos0 > valid) the context restarts at pos0.
  void Inject(hipStream_t s, State& st, const float* feats, std::size_t pos0,
              int n, std::span<const InjectCapture> captures = {});
  /// Inject for several sessions' rows (kMaxRows in all) at once: every
  /// context weight is read once for all of them.
  static constexpr int kMaxRows = 16;
  struct Rows {
    State* st = nullptr;
    const float* feats = nullptr;  // [n][4][2304]
    std::size_t pos0 = 0;
    int n = 0;
  };
  void InjectRows(hipStream_t s, std::span<const Rows> rows);
  /// Whether Propose can run at `pos` (context and Markov state present).
  [[nodiscard]] static bool Ready(const State& st, std::size_t pos) {
    return pos > 0 && st.valid == pos && st.f_pos == pos - 1;
  }
  /// After proposal i (1-based) with the probabilities so far: whether to
  /// draft the next one.
  using More = std::function<bool(int, const float*)>;
  /// Coupled proposals for a request that samples with temperature only (the
  /// target sampler's linear draw): proposal i is drawn from the drafter's
  /// distribution with u[i - 1], the uniform the target's selection after
  /// proposal i - 1 will use. Null: greedy proposals.
  struct Coupled {
    float temperature = 1.f, min_p = 0.f;
    float u[kMaxDraft] = {};
  };
  /// Up to `n` (<= kMaxDraft) greedy proposals after `anchor` at `pos`, and
  /// optionally each one's draft probability (the softmax maximum over the
  /// proposal head). Returns how many were drafted.
  int Propose(hipStream_t s, State& st, std::int32_t anchor, std::size_t pos,
              int n, std::int32_t* out, float* confidence = nullptr,
              const More& more = {}, const Coupled* coupled = nullptr);
  /// Proposals of several sessions at once (up to kMaxSessions): each
  /// request's drafts are those a lone Propose gives, and every weight row is
  /// read once for all of them.
  static constexpr int kMaxSessions = 8;
  struct Request {
    State* st = nullptr;
    std::int32_t anchor = 0;
    std::size_t pos = 0;
    int n = 0;
    std::int32_t* out = nullptr;
    float* confidence = nullptr;
    const More* more = nullptr;
    const Coupled* coupled = nullptr;
    int done = 0;  // out: drafts written
  };
  void ProposeBatch(hipStream_t s, std::span<Request> requests);

private:
  Drafter() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::size_t weight_bytes_ = 0;
};

}  // namespace gufo::models::inkling2
