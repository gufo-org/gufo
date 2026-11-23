// CPU checks of session state declarations (src/mem/state_declaration.hpp):
// the bytes a runner claims follow from the declared buffers, counted as
// sizes or as the 2 MiB pages a demand-mapped region reserves.
#include "src/mem/state_declaration.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "src/mem/account.hpp"
#include "src/mem/ledger.hpp"

namespace {

using gufo::mem::kPageBytes;
using gufo::mem::RegionPageBytes;
using gufo::mem::RegionShape;
using gufo::mem::RegionSpanBytes;
using gufo::mem::StateBytes;
using gufo::mem::StateDeclaration;
using gufo::mem::StateKind;

constexpr std::size_t kMiB = std::size_t{1} << 20;

template<typename F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void TestKindsAndIds() {
  StateDeclaration state(64);
  assert(state.margin_tokens() == 64);
  assert(!state.has_context());
  const auto fixed = state.Fixed(100);
  const auto whole = state.Whole({.row_bytes = 256, .rows = 10});
  assert(!state.has_context());
  const auto context = state.Context({.row_bytes = 1024, .rows = 4096}, 4);
  assert(state.has_context());
  assert(fixed == 0 && whole == 1 && context == 2);
  assert(state.buffers().size() == 3);
  assert(state.buffer(fixed).kind == StateKind::kFixed);
  assert(state.buffer(whole).kind == StateKind::kWhole);
  assert(state.buffer(context).kind == StateKind::kContext);
  assert(state.buffer(context).tokens_per_row == 4);
  assert(state.buffer(whole).tokens_per_row == 1);
  // A fixed buffer is one row of its bytes, at least one.
  assert(state.buffer(fixed).shape.row_bytes == 100);
  assert(state.buffer(fixed).shape.rows == 1);
  assert(state.Bytes(state.Fixed(0), StateBytes::kSize) == 1);
  // Zero tokens per row reads as one, as StateAccount does.
  const auto one = state.Context({.row_bytes = 8, .rows = 8}, 0);
  assert(state.buffer(one).tokens_per_row == 1);
}

void TestSizesAndPages() {
  StateDeclaration state;
  // 1 MiB + 256 bytes: one page counted as a size, two as pages.
  const RegionShape odd{.row_bytes = 256, .rows = 4097};
  const auto a = state.Context(odd);
  assert(state.Bytes(a, StateBytes::kSize) == 4097 * 256);
  assert(state.Bytes(a, StateBytes::kPages) == kPageBytes);
  // Segments back to back: their pages are the union, not one per segment.
  const RegionShape planes{.row_bytes = 4096, .rows = 384, .segments = 8};
  const auto b = state.Context(planes);
  assert(state.Bytes(b, StateBytes::kSize) == 8 * 384 * 4096);
  assert(state.Bytes(b, StateBytes::kPages) ==
         RegionPageBytes(planes, planes.rows));
  assert(state.Bytes(b, StateBytes::kPages) == 6 * kPageBytes);
  // Strided segments span every stride but the last's.
  const RegionShape strided{.row_bytes = 512,
                            .rows = 1024,
                            .segments = 3,
                            .segment_stride = 4 * kMiB};
  assert(RegionSpanBytes(strided) == 2 * 4 * kMiB + kMiB / 2);
  const auto c = state.Whole(strided);
  assert(state.Bytes(c, StateBytes::kPages) == 3 * kPageBytes);
  const auto d = state.Fixed(3 * kMiB + 1);
  assert(state.Bytes(d, StateBytes::kPages) == 3 * kMiB + 1);

  const std::size_t size = state.Bytes(a, StateBytes::kSize) +
                           state.Bytes(b, StateBytes::kSize) +
                           RegionSpanBytes(strided) + 3 * kMiB + 1;
  assert(state.Bytes(StateBytes::kSize) == size);
  const std::size_t pages =
      kPageBytes + 6 * kPageBytes + 3 * kPageBytes + 3 * kMiB + 1;
  assert(state.Bytes(StateBytes::kPages) == pages);
  // Only the buffers that grow with the context are demand-mapped.
  assert(state.ContextBytes(StateBytes::kSize) == 4097 * 256 + 8 * 384 * 4096);
  assert(state.ContextBytes(StateBytes::kPages) == 7 * kPageBytes);
}

void TestAccountAgreesWithPages() {
  // A full claim on an account of the declared kContext buffers charges
  // exactly ContextBytes(kPages): the claim and the ledger see one number.
  StateDeclaration state(16);
  const RegionShape k{.row_bytes = 1024, .rows = 5000};
  const RegionShape compressed{.row_bytes = 2048, .rows = 5000 / 4 + 2};
  (void)state.Context(k);
  (void)state.Context(compressed, 4);
  (void)state.Whole({.row_bytes = 256, .rows = 1251});
  (void)state.Fixed(4096);
  gufo::mem::StateAccount account(state.margin_tokens());
  for (const auto& b : state.buffers()) {
    if (b.kind == StateKind::kContext)
      account.AddRegion(b.shape, b.tokens_per_row, [](std::size_t) {});
  }
  assert(account.full_bytes() == state.ContextBytes(StateBytes::kPages));
}

void TestInvalidShapes() {
  StateDeclaration state;
  assert(Throws([&] { (void)state.Context({.row_bytes = 0, .rows = 1}); }));
  assert(Throws([&] { (void)state.Context({.row_bytes = 1, .rows = 0}); }));
  assert(Throws(
      [&] { (void)state.Whole({.row_bytes = 1, .rows = 1, .segments = 0}); }));
  assert(Throws([&] {
    (void)state.Context(
        {.row_bytes = 64, .rows = 64, .segments = 2, .segment_stride = 64});
  }));
  // With two bytes per row, these spans wrap to zero and to the stride.
  constexpr auto first_overflow =
      std::numeric_limits<std::size_t>::max() / 2 + 1;
  for (const auto rows : {first_overflow, first_overflow + 1}) {
    const RegionShape shape{
        .row_bytes = 2, .rows = rows, .segments = 2, .segment_stride = 2};
    assert(Throws([&] { (void)state.Context(shape); }));
    assert(Throws([&] { (void)state.Whole(shape); }));
  }
  assert(state.buffers().empty());

  const auto adjacent = state.Whole(
      {.row_bytes = 3, .rows = 2, .segments = 2, .segment_stride = 6});
  assert(state.Bytes(adjacent, StateBytes::kSize) == 12);
}

}  // namespace

int main() {
  TestKindsAndIds();
  TestSizesAndPages();
  TestAccountAgreesWithPages();
  TestInvalidShapes();
  std::puts("mem_state_declaration_test: ok");
  return 0;
}
