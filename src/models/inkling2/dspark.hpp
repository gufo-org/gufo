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
#include <memory>
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
  };
  /// Per-session state (buffers declared with the session). Context rows of
  /// positions [0, valid) are injected; `f` is the Markov context of position
  /// f_pos.
  struct State {
    void* k[3] = {};
    void* v[3] = {};
    std::size_t cap[3] = {};  // rows (window layers: rings)
    float* f = nullptr;       // [2304]
    std::size_t valid = 0;
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

  /// Injects the features [n][4][2304] of positions [pos0, pos0 + n). Only a
  /// continuation of the injected rows (pos0 <= valid) is kept.
  void Inject(hipStream_t s, State& st, const float* feats, std::size_t pos0,
              int n);
  /// Whether Propose can run at `pos` (context and Markov state present).
  [[nodiscard]] static bool Ready(const State& st, std::size_t pos) {
    return pos > 0 && st.valid == pos && st.f_pos == pos - 1;
  }
  /// Up to `n` (<= kMaxDraft) greedy proposals after `anchor` at `pos`, and
  /// optionally each one's draft probability (the softmax maximum over the
  /// proposal head).
  void Propose(hipStream_t s, State& st, std::int32_t anchor, std::size_t pos,
               int n, std::int32_t* out, float* confidence = nullptr);

private:
  Drafter() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::size_t weight_bytes_ = 0;
};

}  // namespace gufo::models::inkling2
