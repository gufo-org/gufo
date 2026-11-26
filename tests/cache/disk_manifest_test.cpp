// The Rust disk manifest codec and CRC-64 through src/cache/disk.hpp, with
// expected results: manifests with row, window and private-state components
// encode to the 0.35.0 bytes (pinned by length, header and checksum trailer)
// under format version 2 or 3, and decode to the same fields with capacities
// that match the metadata plan disk.cpp charges; each limit (2 Mi tokens,
// 64 KiB of input, 256 components, 65,536 payload references) is reached and
// exceeded; incoherent manifests and damaged bytes throw
// std::invalid_argument; checksums match recorded CRC-64/ECMA-182 values and
// chain through their seed. The crate's Rust tests cover validation, every
// truncation and bit flip, both checksum paths and the C ABI.
#include <gufo/cache_manifest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/disk.hpp"

using namespace gufo::cache;
namespace {
[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}
void Expect(bool condition, const std::string& message) {
  if (!condition)
    Fail(message);
}
// SplitMix64.
struct Random {
  std::uint64_t state;
  std::uint64_t Next() {
    auto z = state += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

DiskFileId File(std::uint8_t n) {
  DiskFileId id;
  id.fill(n);
  return id;
}
DiskPayload Payload(std::uint8_t n, std::uint64_t bytes) {
  return {File(n), bytes, 0x0101010101010101ULL * n};
}
// Chunks and tail for `rows` valid rows, with files numbered from `file`.
DiskComponent RowComponent(ComponentDescriptor d, Rows rows,
                           std::uint8_t file) {
  DiskComponent c{d, {d.id, rows}, {}, {}, {}};
  for (auto row = FirstChunkRow(d, rows); row + d.rows_per_chunk <= rows;
       row += d.rows_per_chunk)
    c.chunks.push_back(Payload(file++, d.row_bytes * d.rows_per_chunk));
  if (rows % d.rows_per_chunk)
    c.tail = Payload(file, d.row_bytes * (rows % d.rows_per_chunk));
  return c;
}
DiskComponent Private(std::uint32_t id, std::size_t bytes, Rows rows,
                      std::uint8_t file) {
  return {{{id}, 1, ComponentKind::kPrivateState, 0, 0, bytes, 0},
          {{id}, rows},
          {},
          {},
          Payload(file, bytes)};
}
DiskManifest Base(std::size_t tokens) {
  DiskManifest m;
  m.checkpoint = {5};
  m.lineage = {6};
  for (std::size_t i = 0; i < m.compatibility.size(); ++i)
    m.compatibility[i] = static_cast<std::uint8_t>(i + 1);
  m.purpose = CheckpointPurpose::kGenerated;
  m.rank = 2;
  m.input = {'i', 'd', 'e', 'n', 't', 'i', 't', 'y'};
  for (std::size_t i = 0; i < tokens; ++i)
    m.tokens.push_back(static_cast<Token>(i * 2654435761U));
  return m;
}
// Version 2: append rows with two chunks and a tail, and private state.
DiskManifest Plain() {
  auto m = Base(40);
  m.components.push_back(
      RowComponent({{1}, 1, ComponentKind::kAppendRows, 4, 16, 0, 0}, 40, 1));
  m.components.push_back(Private(2, 8, 40, 9));
  return m;
}
// Version 3: a window of 20 rows over 50 keeps chunks from row 16 on, beside
// append rows and private state of another layout version.
DiskManifest Windowed() {
  auto m = Base(50);
  m.purpose = CheckpointPurpose::kLearned;
  m.input.clear();
  m.components.push_back(
      RowComponent({{3}, 1, ComponentKind::kAppendRows, 2, 32, 0, 0}, 64, 1));
  m.components.push_back(
      RowComponent({{7}, 4, ComponentKind::kWindowRows, 3, 16, 0, 20}, 50, 4));
  m.components.push_back(Private(9, 24, 50, 9));
  m.components.back().descriptor.layout_version = 0xFFFFFFFF;
  return m;
}

std::uint64_t Trailer(std::span<const std::uint8_t> bytes) {
  std::uint64_t n{};
  for (unsigned i = 0; i < 8; ++i)
    n |= std::uint64_t(bytes[bytes.size() - 8 + i]) << (8 * i);
  return n;
}
std::vector<std::uint8_t> Signed(std::vector<std::uint8_t> body) {
  const auto crc = DiskChecksum(body);
  for (unsigned i = 0; i < 8; ++i)
    body.push_back(crc >> (8 * i));
  return body;
}
std::vector<std::uint8_t> Body(std::span<const std::uint8_t> bytes) {
  return {bytes.begin(), bytes.end() - 8};
}

bool Rejected(const std::function<void()>& call, const std::string& what) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (const std::exception& e) {
    Fail(what + ": unexpected exception: " + e.what());
  }
  return false;
}
void EncodeRejects(const DiskManifest& m, const std::string& what) {
  Expect(Rejected([&] { (void)EncodeManifest(m); }, what), what + ": encoded");
}
// Decoding rejects the bytes, and so does planning unless `structure` is
// false: a plan reads the counts and presence bytes alone, not the checksum,
// magic or version.
void DecodeRejects(std::span<const std::uint8_t> bytes, const std::string& what,
                   bool structure = true) {
  Expect(Rejected([&] { (void)DecodeManifest(bytes); }, what),
         what + ": decoded");
  GufoCacheManifestPlan plan{};
  Expect(gufo_cache_manifest_plan(bytes.data(), bytes.size(), &plan) ==
             (structure ? GUFO_CACHE_REJECTED : GUFO_CACHE_COMPLETE),
         what + ": plan differs");
}

// Equal fields, and capacities equal to the sizes, which the store's
// metadata accounting depends on.
void ExpectSame(const DiskManifest& a, const DiskManifest& b,
                const std::string& what) {
  Expect(a.checkpoint == b.checkpoint && a.lineage == b.lineage &&
             a.compatibility == b.compatibility && a.input == b.input &&
             a.tokens == b.tokens && a.purpose == b.purpose &&
             a.rank == b.rank && a.components.size() == b.components.size(),
         what + ": manifest fields differ");
  Expect(a.input.capacity() == a.input.size() &&
             a.tokens.capacity() == a.tokens.size() &&
             a.components.capacity() == a.components.size(),
         what + ": decoding reserved more than it holds");
  for (std::size_t i = 0; i < a.components.size(); ++i) {
    const auto &x = a.components[i], &y = b.components[i];
    const auto &d = x.descriptor, &e = y.descriptor;
    Expect(d.id == e.id && d.layout_version == e.layout_version &&
               d.kind == e.kind && d.row_bytes == e.row_bytes &&
               d.rows_per_chunk == e.rows_per_chunk &&
               d.state_bytes == e.state_bytes &&
               d.window_rows == e.window_rows && x.position == y.position &&
               x.chunks == y.chunks && x.chunks.capacity() == x.chunks.size() &&
               x.tail == y.tail && x.private_state == y.private_state,
           what + ": component " + std::to_string(i) + " differs");
  }
}

// Decodes to `m`, and plans the metadata and references decoding retains.
void RoundTrip(const DiskManifest& m, std::span<const std::uint8_t> bytes,
               const std::string& what) {
  ExpectSame(DecodeManifest(bytes), m, what);
  GufoCacheManifestPlan plan{};
  Expect(gufo_cache_manifest_plan(bytes.data(), bytes.size(), &plan) ==
             GUFO_CACHE_COMPLETE,
         what + ": plan failed");
  std::size_t chunks{}, references{};
  for (const auto& c : m.components) {
    chunks += c.chunks.size();
    references +=
        c.chunks.size() + c.tail.has_value() + c.private_state.has_value();
  }
  Expect(plan.input_bytes == m.input.size() && plan.tokens == m.tokens.size() &&
             plan.components == m.components.size() && plan.chunks == chunks &&
             plan.references == references,
         what + ": plan differs");
}

void Encodings() {
  // Length and checksum trailer of the 0.35.0 encoding.
  const struct {
    std::string_view what;
    DiskManifest manifest;
    std::uint8_t version;
    std::size_t size;
    std::uint64_t trailer;
  } cases[] = {
      {"plain", Plain(), 2, 475, 0x95049e139fb411deULL},
      {"windowed", Windowed(), 3, 626, 0x501343cb7386c6fbULL},
  };
  for (const auto& test : cases) {
    const std::string what(test.what);
    const auto bytes = EncodeManifest(test.manifest);
    const std::vector<std::uint8_t> header{
        'G', 'U', 'F', 'O', 'M', 'N', 'F', '2', test.version, 0, 0, 0};
    Expect(std::equal(header.begin(), header.end(), bytes.begin()),
           what + ": magic or version differs");
    // The checkpoint follows the version; the checksum covers every byte
    // before it.
    Expect(bytes[12] == 5 && bytes[20] == 6 && bytes[28] == 1,
           what + ": fixed fields differ");
    Expect(Trailer(bytes) == DiskChecksum(Body(bytes)),
           what + ": the trailer is not the checksum of the body");
    Expect(bytes.size() == test.size && Trailer(bytes) == test.trailer,
           what + ": encoding differs from 0.35.0: " +
               std::to_string(bytes.size()) + " bytes, trailer " +
               std::to_string(Trailer(bytes)));
    RoundTrip(test.manifest, bytes, what);

    // Damaged bytes; re-signed, the structure behind the checksum is checked.
    DecodeRejects(std::span(bytes).first(bytes.size() - 1),
                  what + " truncated");
    DecodeRejects(std::span(bytes).first(19), what + " under 20 bytes");
    auto flipped = bytes;
    flipped[40] ^= 4;
    DecodeRejects(flipped, what + " flipped", false);
    auto body = Body(bytes);
    body.push_back(0);
    DecodeRejects(Signed(body), what + " with a trailing byte");
    body = Body(bytes);
    body[0] = 'g';
    DecodeRejects(Signed(body), what + " magic", false);
    // Only manifests with window components use version 3, so each has one
    // encoding; version 4 is unknown.
    for (const std::uint8_t version :
         {std::uint8_t(5 - test.version), std::uint8_t{1}, std::uint8_t{4}}) {
      body = Body(bytes);
      body[8] = version;
      DecodeRejects(Signed(body),
                    what + " as version " + std::to_string(version), false);
    }
  }
  std::vector<std::uint8_t> huge(kMaxManifestBytes + 1);
  DecodeRejects(huge, "over the manifest size");
}

// Incoherent manifests, through the C++ structures.
void Incoherent() {
  const auto rejects = [](const std::function<void(DiskManifest&)>& change,
                          const std::string& what) {
    auto m = Windowed();
    change(m);
    EncodeRejects(m, what);
  };
  rejects([](auto& m) { m.checkpoint.value = 0; }, "no checkpoint");
  rejects([](auto& m) { m.lineage.value = 0; }, "no lineage");
  rejects([](auto& m) { m.purpose = static_cast<CheckpointPurpose>(4); },
          "unknown purpose");
  rejects([](auto& m) { m.components.clear(); }, "no components");
  rejects([](auto& m) { std::swap(m.components[0], m.components[1]); },
          "descending ids");
  rejects([](auto& m) { m.components[1].position.id = {3}; },
          "another position id");
  rejects([](auto& m) { m.components[0].descriptor.layout_version = 0; },
          "layout version 0");
  rejects(
      [](auto& m) {
        m.components[0].descriptor.kind = static_cast<ComponentKind>(3);
      },
      "unknown kind");
  rejects([](auto& m) { m.components[1].descriptor.window_rows = 0; },
          "a window of no rows");
  rejects([](auto& m) { m.components[0].descriptor.window_rows = 4; },
          "a window on append rows");
  rejects([](auto& m) { m.components[1].position.valid_rows += 1; },
          "rows without their tail");
  rejects([](auto& m) { m.components[1].chunks.pop_back(); }, "a lost chunk");
  rejects([](auto& m) { m.components[0].chunks[1].bytes += 1; },
          "a chunk of another size");
  rejects(
      [](auto& m) {
        m.components[0].chunks[1].file = m.components[0].chunks[0].file;
      },
      "a chunk file twice");
  rejects([](auto& m) { m.components[1].tail.reset(); }, "no tail");
  rejects([](auto& m) { m.components[1].tail->bytes += 1; },
          "a tail of another size");
  rejects([](auto& m) { m.components[2].private_state.reset(); },
          "no private state");
  rejects([](auto& m) { m.components[2].private_state->bytes += 1; },
          "private state of another size");
  rejects([](auto& m) { m.components[2].position.valid_rows -= 1; },
          "private state at another boundary");
  rejects(
      [](auto& m) {
        m.components[1].tail->file = m.components[2].private_state->file;
      },
      "a tail sharing the private state's file");
  rejects([](auto& m) { m.components[1].private_state = Payload(99, 3); },
          "private state on rows");
  rejects([](auto& m) { m.components[0].descriptor.row_bytes = UINT64_MAX; },
          "rows past 64 bits");
}

// Each limit, reached and exceeded.
void Limits() {
  auto m = Base(2 * 1024 * 1024);
  m.components.push_back(Private(1, 8, m.tokens.size(), 1));
  m.input.assign(65536, 7);
  const auto bytes = EncodeManifest(m);
  RoundTrip(m, bytes, "token and input limits");
  m.input.push_back(7);
  EncodeRejects(m, "input limit exceeded");
  m.input.pop_back();
  m.tokens.push_back(1);
  m.components[0].position.valid_rows += 1;
  EncodeRejects(m, "token limit exceeded");

  // 256 components holding 65,536 references.
  auto wide = Base(0);
  for (std::uint32_t i = 0; i < 256; ++i) {
    DiskComponent c{{{i + 1}, 1, ComponentKind::kAppendRows, 2, 1, 0, 0},
                    {{i + 1}, 256},
                    {},
                    {},
                    {}};
    for (unsigned j = 0; j < 256; ++j) {
      DiskFileId file{};
      file[0] = static_cast<std::uint8_t>(j);
      file[1] = static_cast<std::uint8_t>(j >> 8);
      c.chunks.push_back({file, 2, j});
    }
    wide.components.push_back(std::move(c));
  }
  const auto wide_bytes = EncodeManifest(wide);
  RoundTrip(wide, wide_bytes, "component and reference limits");
  // The same bytes with a 65,537th reference: a tail, a private state or one
  // more chunk of the last component, whose presence bytes end the body.
  const auto wide_body = Body(wide_bytes);
  const std::vector<std::uint8_t> extra(32, 7);
  for (const std::size_t flag : {2, 1}) {
    auto body = wide_body;
    body[body.size() - flag] = 1;
    body.insert(body.end() - flag + 1, extra.begin(), extra.end());
    DecodeRejects(Signed(std::move(body)), "reference limit exceeded");
  }
  {
    auto body = wide_body;
    const auto count = body.size() - 2 - 256 * 32 - 4;
    body[count] += 1;
    body.insert(body.end() - 2, extra.begin(), extra.end());
    DecodeRejects(Signed(std::move(body)), "reference limit exceeded");
  }
  // The last component's tail is reference 65,537.
  auto more = wide;
  auto& last = more.components.back();
  last.descriptor.rows_per_chunk = 2;
  last.position.valid_rows = 2 * 256 + 1;
  for (auto& chunk : last.chunks)
    chunk.bytes = 4;
  last.tail = Payload(0xEE, 2);
  EncodeRejects(more, "reference limit exceeded");
  // A 257th component.
  more = wide;
  more.components.back().chunks.pop_back();
  more.components.back().position.valid_rows -= 1;
  more.components.push_back(Private(1000, 8, 0, 0xEE));
  EncodeRejects(more, "component limit exceeded");
}

void Checksums() {
  // The catalogue check value of CRC-64/ECMA-182; a seed passes empty input.
  const std::string_view check = "123456789";
  Expect(DiskChecksum(
             std::span(reinterpret_cast<const std::uint8_t*>(check.data()),
                       check.size())) == 0x6c40df5f0b497347ULL,
         "check value differs");
  Expect(DiskChecksum({}, 0x1234) == 0x1234, "an empty checksum differs");
  Random r{9};
  std::vector<std::uint8_t> data((1u << 20) + 64);
  for (auto& b : data)
    b = r.Next();
  // Recorded from 0.35.0, around its table and folding paths.
  const struct {
    std::size_t offset, len;
    std::uint64_t seed, crc;
  } cases[] = {
      {0, 1, 0, 0x5f4c95afc5edc62eULL},
      {3, 7, 0, 0x24b7d07de77415c3ULL},
      {1, 63, 0, 0xfc55a0ab65108484ULL},
      {0, 64, 0x1234, 0x4a665d898d225567ULL},
      {5, 255, 0, 0x970895571cbddde3ULL},
      {7, 256, 0, 0x4cefa9471c2909e8ULL},
      {9, 1100, ~0ULL, 0x4bae464c4026c79fULL},
      {0, (1u << 20) + 64, 0, 0xc87739aa954ddffaULL},
  };
  for (const auto& c : cases) {
    const auto crc =
        DiskChecksum(std::span(data).subspan(c.offset, c.len), c.seed);
    Expect(crc == c.crc, "checksum of " + std::to_string(c.len) +
                             " bytes differs: " + std::to_string(crc));
  }
  // Pieces chain through the seed.
  const auto whole = DiskChecksum(data);
  for (unsigned i = 0; i < 20; ++i) {
    std::uint64_t crc = 0;
    for (std::size_t at = 0; at < data.size();) {
      const auto n =
          std::min<std::size_t>(1 + r.Next() % 70000, data.size() - at);
      crc = DiskChecksum(std::span(data).subspan(at, n), crc);
      at += n;
    }
    Expect(crc == whole, "chained checksum differs");
  }
}
}  // namespace

int main() {
  Checksums();
  Encodings();
  Incoherent();
  Limits();
  std::cout << "PASS: disk manifest expected results\n";
}
