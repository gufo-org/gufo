#include "src/models/deepseek_v4_flash/continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ds4 = gufo::models::deepseek_v4_flash;
namespace cache = gufo::cache;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

ds4::ContinuationGeometry Production(std::uint32_t stages) {
  ds4::ContinuationGeometry geometry{.head_dim = 512,
                                     .indexer_head_dim = 128,
                                     .window = 128,
                                     .vocabulary = 129280,
                                     .draft_stages = stages};
  // Two window-only layers, then alternating overlapped and sparse layers.
  geometry.ratios = {0, 0};
  for (std::uint32_t layer = 2; layer < 43; ++layer)
    geometry.ratios.push_back(layer % 2 ? 128 : 4);
  return geometry;
}

std::size_t Count(const ds4::ContinuationLayout& layout,
                  ds4::ContinuationPart part) {
  return std::ranges::count(layout.Components(), part,
                            &ds4::ContinuationComponent::part);
}

void TestInventory() {
  const ds4::ContinuationLayout layout(Production(0), 131072);
  Require(Count(layout, ds4::ContinuationPart::kLayerState) == 43,
          "every layer keeps a sliding window");
  Require(Count(layout, ds4::ContinuationPart::kAttentionRows) == 41,
          "only compressed layers append attention rows");
  Require(Count(layout, ds4::ContinuationPart::kIndexerRows) == 21,
          "only overlapped layers append indexer rows");
  Require(Count(layout, ds4::ContinuationPart::kDraftWindow) == 0,
          "autoregressive sessions carry no draft window");
  std::uint32_t id = 0;
  for (const auto& component : layout.Components())
    Require(component.descriptor.id.value == ++id, "IDs are dense from one");
  const ds4::ContinuationLayout speculative(Production(3), 131072);
  Require(Count(speculative, ds4::ContinuationPart::kDraftWindow) == 3,
          "each DSpark stage has a draft window");
  Require(speculative.PrivateBytes() ==
              layout.PrivateBytes() + 3 * 128 * 512 * sizeof(float),
          "a draft window holds the sliding window of one stage");
}

void TestPrivateBytes() {
  // Overlapped compressors keep (2 * width) * (2 * ratio) floats per tensor.
  Require(ds4::CompressorStateBytes(4, 512) == 2 * 512 * 2 * 4 * sizeof(float),
          "overlapped attention frontier");
  Require(ds4::CompressorStateBytes(128, 512) == 512 * 128 * sizeof(float),
          "sparse attention frontier");
  ds4::ContinuationGeometry window = Production(0);
  window.ratios = {0};
  ds4::ContinuationGeometry sparse = window;
  sparse.ratios = {128};
  ds4::ContinuationGeometry overlapped = window;
  overlapped.ratios = {4};
  const auto base = ds4::ContinuationLayout(window, 4096).PrivateBytes();
  Require(ds4::ContinuationLayout(sparse, 4096).PrivateBytes() - base ==
              2 * ds4::CompressorStateBytes(128, 512),
          "sparse layers add two attention frontiers");
  Require(ds4::ContinuationLayout(overlapped, 4096).PrivateBytes() - base ==
              2 * ds4::CompressorStateBytes(4, 512) +
                  2 * ds4::CompressorStateBytes(4, 128),
          "overlapped layers add attention and indexer frontiers");
}

void TestPositions() {
  const ds4::ContinuationLayout layout(Production(1), 4096);
  const auto positions = layout.Positions(1031);
  Require(positions.size() == layout.Components().size(),
          "one position per component");
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto& component = layout.Components()[i];
    const auto ratio = layout.Geometry().ratios[component.index];
    switch (component.part) {
      case ds4::ContinuationPart::kAttentionRows:
      case ds4::ContinuationPart::kIndexerRows:
        Require(positions[i].valid_rows == 1031 / ratio,
                "compressed rows count only completed windows");
        break;
      default:
        Require(positions[i].valid_rows == 1031,
                "other components report the token frontier");
    }
  }
  bool rejected = false;
  try {
    (void)layout.Positions(4097);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "a frontier beyond context is rejected");
}

void TestRowBytes() {
  const ds4::ContinuationLayout layout(Production(0), 131072);
  const std::size_t tokens = 8192;
  const auto expected = tokens * sizeof(std::int32_t) +
                        21 * (tokens / 4) * (512 + 128) * sizeof(float) +
                        20 * (tokens / 128) * 512 * sizeof(float);
  Require(layout.RowBytes(tokens) == expected, "row bytes at a frontier");
}

void TestChunks() {
  const ds4::ContinuationLayout layout(Production(0), 131072);
  for (const auto& component : layout.Components()) {
    const auto& d = component.descriptor;
    if (d.kind != cache::ComponentKind::kAppendRows)
      Require(!d.rows_per_chunk && !d.row_bytes, "private state has no rows");
    else if (component.part != ds4::ContinuationPart::kTokens)
      Require(d.rows_per_chunk * d.row_bytes >= 65536,
              "a device-row chunk fills a backing page");
  }
}

void TestBacking() {
  const ds4::ContinuationLayout layout(Production(0), 4096);
  Require(layout.BackingPlan(1 << 20).empty(), "tiny budgets commit nothing");
  const auto plan = layout.BackingPlan(1ULL << 30);
  Require(plan.size() == 1 && plan[0].block_bytes == 65536,
          "one page class serves rows and private state");
  Require(plan[0].blocks * plan[0].block_bytes + plan[0].metadata_bytes <=
              1ULL << 30,
          "committed backing stays inside the budget");
}
}  // namespace

int main() {
  try {
    TestInventory();
    TestPrivateBytes();
    TestPositions();
    TestRowBytes();
    TestChunks();
    TestBacking();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
