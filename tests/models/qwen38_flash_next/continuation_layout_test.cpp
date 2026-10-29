#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <map>
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
    // 8-bit KV rows each (512 codes and 32 f16 scales); one 128-element f16
    // pool per four tokens; host token history has four bytes per target
    // token.
    const auto expected =
        std::size_t{target} * 14596 +
        (mtp ? std::size_t{draft} * 1152 + std::size_t{draft / 4} * 256 : 0);
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
void CheckExtendedContext(bool mtp) {
  const auto native = ProductionGeometry();
  Reject([&] {
    qfn::ContinuationLayout layout(native, mtp, native.context_length + 1, 8);
  });
  for (const auto context : {409600U, 655360U}) {
    auto scaled = native;
    scaled.rope_scaling = {.factor = context == 409600 ? 1.5625F : 2.5F,
                           .original_context = 262144};
    const qfn::ContinuationLayout layout(scaled, mtp, context, 8);
    Require(scaled.context_length == 262144,
            "continuation scaling rewrote the native context");
    Require(layout.Components().size() == (mtp ? 130 : 124),
            "extended context changed the state inventory");
    Require(layout.RawCapacity() == 2048,
            "extended context inflated the bounded raw ring");
    for (const auto target : {262145U, context}) {
      const auto draft = mtp ? target - 1 : 0;
      // Twelve layers of 1152 KV bytes per token and 256 pooled bytes per
      // complete four-token block, plus four history bytes per token.
      const auto expected =
          std::size_t{target} * 13828 + std::size_t{target / 4} * 3072 +
          (mtp ? std::size_t{draft} * 1152 + std::size_t{draft / 4} * 256 : 0);
      Require(layout.RowBytes(target, draft, target / 4, draft / 4) == expected,
              "extended frontier lost target or predictor rows");
    }
    Reject([&] {
      qfn::ContinuationLayout too_large(scaled, mtp, context + 1, 8);
    });
    Reject([&] { (void)layout.Positions(context + 1, 0, context / 4, 0); });
  }
}
void CheckBacking(bool mtp, std::uint32_t context, std::uint32_t concurrency,
                  std::size_t ram, std::size_t checkpoints,
                  std::size_t independent_rows) {
  const auto c = ProductionGeometry();
  const qfn::ContinuationLayout layout(c, mtp, context, 8);
  std::size_t available = 0, required = 0;
  std::size_t bytes = 0;
  for (const auto& allocation : layout.BackingPlan(ram, concurrency)) {
    Require(allocation.block_bytes % 4096 == 0 && allocation.blocks != 0,
            "invalid physical backing class");
    Require(allocation.metadata_bytes >= 4096 + allocation.blocks * 1024,
            "backing plan omitted allocator headroom");
    Require(allocation.block_bytes == 65536,
            "backing pages cannot serve all components");
    available += allocation.blocks;
    bytes +=
        allocation.block_bytes * allocation.blocks + allocation.metadata_bytes;
  }
  Require(bytes <= (ram - (4ULL << 20)) / 10 * 9,
          "physical backing exceeds the startup budget");
  for (const auto& component : layout.Components()) {
    const auto& d = component.descriptor;
    const auto payload = d.kind == cache::ComponentKind::kPrivateState
                             ? d.state_bytes
                             : d.rows_per_chunk * d.row_bytes;
    const auto pages = (payload + 65535) / 65536;
    if (d.kind == cache::ComponentKind::kPrivateState)
      required += pages * checkpoints;
    else {
      const bool pooled =
          component.part == qfn::ContinuationPart::kPooledKeys ||
          component.part == qfn::ContinuationPart::kDraftPooledKeys;
      const auto rows =
          pooled ? independent_rows / c.compress_ratio : independent_rows;
      required += pages * ((rows + d.rows_per_chunk - 1) / d.rows_per_chunk +
                           checkpoints);
    }
  }
  // Independent physical demands, including shared-class collisions and a
  // partial tail per checkpoint. A row-heavy pool must not strand most of its
  // budget while private state already refuses feasible histories.
  Require(available >= required,
          "feasible capture envelope does not fit committed pages");
  Require(layout.BackingPlan(1 << 20, concurrency).empty(),
          "tiny budgets should leave cache metadata headroom");
  Reject([&] { (void)layout.BackingPlan(ram, 0); });
  Reject([&] { (void)layout.BackingPlan(ram, 9); });
}
}  // namespace

int main() {
  try {
    Check(false);
    Check(true);
    for (const auto mtp : {false, true}) {
      CheckExtendedContext(mtp);
      CheckBacking(mtp, 32768, 2, 8ULL << 30, 32, 2 * 32768);
      CheckBacking(mtp, 32768, 2, 2ULL << 30, 8, 2 * 4096);
      CheckBacking(mtp, 4096, 8, 8ULL << 30, 8, 8 * 4096);
      CheckBacking(mtp, 133760, 1, 8ULL << 30, 8, 2 * 100000);
    }
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
