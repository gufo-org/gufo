#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace qfn = gufo::models::qwen38_flash_next;
namespace cache = gufo::cache;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

qfn::Config ProductionGeometry() {
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.context_length = 262144;
  c.hidden_size = 2560;
  c.hc_count = 4;
  c.vocab_size = 248320;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}

template<class F>
void Reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid geometry or frontier accepted");
}

void Check(bool mtp) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout layout(c, mtp, 100000, 8);
  Require(layout.Components().size() == (mtp ? 130 : 124),
          "state inventory omitted a component");
  Require(layout.PrivateBytes() == (mtp ? 133032096 : 131614880),
          "private byte claim differs from independent geometry");
  Require(layout.RawCapacity() == 2048, "raw ring reserve is not bounded");
  for (const auto target : {32768U, 100000U}) {
    const auto draft = mtp ? target - 1 : 0;
    const auto positions =
        layout.Positions(target, draft, target / 4, draft / 4);
    // Twelve target attention layers, thirteen with the predictor. Two
    // 512-element f16 KV rows each; one 128-element f16 pool per four tokens;
    // host token history has four bytes per target token.
    const auto expected =
        std::size_t{target} * 25348 +
        (mtp ? std::size_t{draft} * 2048 + std::size_t{draft / 4} * 256 : 0);
    Require(layout.RowBytes(target, draft, target / 4, draft / 4) == expected,
            "row byte claim used target position for the draft or pools");
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const auto& component = layout.Components()[i];
      Require(component.descriptor.id.value == i + 1,
              "component IDs are not unique and stable");
      if (component.descriptor.kind == cache::ComponentKind::kPrivateState)
        Require(positions[i].valid_rows == target,
                "private state does not attest the checkpoint boundary");
      if (component.part == qfn::ContinuationPart::kDraftKey)
        Require(positions[i].valid_rows == draft,
                "predictor KV lost its independent frontier");
      if (component.part == qfn::ContinuationPart::kDraftPooledKeys)
        Require(positions[i].valid_rows == draft / 4,
                "predictor pools are counted in tokens");
    }
  }
  // Before sparse attention starts, every raw row is private. Afterwards a
  // checkpoint can hold an unfinished pool and a lagging predictor together.
  (void)layout.Positions(2048, mtp ? 2047 : 0, 0, 0);
  (void)layout.Positions(2051, mtp ? 2049 : 0, 512, mtp ? 512 : 0);
  Reject([&] { (void)layout.Positions(2049, 0, 0, 0); });
  Reject([&] { (void)layout.Positions(100001, 0, 25000, 0); });
  Reject([&] { (void)layout.Positions(10, 11, 2, 2); });
  Reject([&] { (void)layout.Positions(10, 0, 3, 0); });
  Reject([&] { (void)layout.Positions(10, mtp ? 9 : 0, 2, 3); });
  if (!mtp)
    Reject([&] { (void)layout.Positions(10, 1, 2, 0); });
  const qfn::ContinuationLayout short_layout(c, mtp, 1000, 8);
  Require(short_layout.RawCapacity() == 1000,
          "short contexts reserve an unreachable raw frontier");
  auto no_ple = c;
  no_ple.ple_layer = -1;
  const qfn::ContinuationLayout no_ple_layout(no_ple, mtp, 100000, 8);
  Require(layout.PrivateBytes() - no_ple_layout.PrivateBytes() == 368640,
          "PLE history inventory is incorrect");
}
}  // namespace

int main() {
  try {
    Check(false);
    Check(true);
    const auto c = ProductionGeometry();
    Reject([&] { qfn::ContinuationLayout layout(c, false, 0, 8); });
    Reject([&] { qfn::ContinuationLayout layout(c, true, 100000, 0); });
    auto bad = c;
    bad.compress_ratio = 0;
    Reject([&] { qfn::ContinuationLayout layout(bad, false, 100000, 8); });
    std::cout << "Flash-Next continuation geometry: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
