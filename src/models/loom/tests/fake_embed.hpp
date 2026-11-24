// A fake libloom_text_embed for testing the Loom runner without a GPU: a
// byte-level model whose logit rows are a scripted function of (last consumed
// token, position). It implements the C API the runner calls, including the
// ABI 2 logits readback, and records every epoch so tests can assert which
// ones left logits.
#ifndef GUFO_LOOM_FAKE_EMBED_HPP_
#define GUFO_LOOM_FAKE_EMBED_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fake {

inline constexpr int kBytes = 256;
inline constexpr int kEos = 256;
inline constexpr int kVocabulary = 257;  // 256 byte tokens and EOS
inline constexpr int kWidth = 384;       // padded to 128

// Scripts a row: fills `out` (kWidth floats, pre-filled with -3e38) for the
// state after consuming `last` at position `position`.
using Script = std::function<void(int last, int position, float* out)>;

struct Epoch {
  std::vector<std::uint32_t> rows;
  std::vector<int> tokens;
  std::vector<bool> exported;  // per row: its logit row was downloaded
  bool host_tokens{false};
};

struct State {
  Script script;
  std::vector<Epoch> epochs;
  bool declares_logits{true};
  // Verified tokens per step of the package's drafter; 0 without one.
  std::uint32_t draft_depth{0};
  // What the fake chat program renders for the placeholder tool call.
  std::string tool_call_text{
      "<tool_call>\n{\"name\": \"qprobe\", \"arguments\": {\"zz\": "
      "7}}\n</tool_call>"};
  std::string reasoning_close{"</think>"};
};

State& state();
void reset();
// The argmax token of a scripted row (what the device selects).
int Argmax(const float* row);

}  // namespace fake

#endif
