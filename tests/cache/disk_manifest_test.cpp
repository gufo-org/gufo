// Compares the Rust disk manifest codec and CRC-64 behind src/cache/disk.hpp
// with the frozen C++ copy in disk_manifest_reference.cpp: generated
// manifests must encode to the same bytes, every truncation and deterministic
// mutation must decode to the same manifest or be rejected by both, and
// checksums over random lengths, offsets and seeds must agree on both paths.
#include <gufo/cache_manifest.h>

#include <chrono>
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
#include "tests/cache/disk_manifest_reference.hpp"

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
  std::uint64_t Below(std::uint64_t n) { return n ? Next() % n : 0; }
  bool Chance(unsigned percent) { return Below(100) < percent; }
};

bool Same(const DiskPayload& a, const DiskPayload& b) {
  return a == b;
}
bool Same(const std::optional<DiskPayload>& a,
          const std::optional<DiskPayload>& b) {
  return a.has_value() == b.has_value() && (!a || Same(*a, *b));
}
// Equal fields, and equal capacities, which the store's metadata accounting
// depends on.
bool Same(const DiskManifest& a, const DiskManifest& b) {
  if (a.checkpoint != b.checkpoint || a.lineage != b.lineage ||
      a.compatibility != b.compatibility || a.input != b.input ||
      a.input.capacity() != b.input.capacity() || a.tokens != b.tokens ||
      a.tokens.capacity() != b.tokens.capacity() || a.purpose != b.purpose ||
      a.rank != b.rank || a.components.size() != b.components.size() ||
      a.components.capacity() != b.components.capacity())
    return false;
  for (std::size_t i = 0; i < a.components.size(); ++i) {
    const auto &x = a.components[i], &y = b.components[i];
    const auto &d = x.descriptor, &e = y.descriptor;
    if (d.id != e.id || d.layout_version != e.layout_version ||
        d.kind != e.kind || d.row_bytes != e.row_bytes ||
        d.rows_per_chunk != e.rows_per_chunk ||
        d.state_bytes != e.state_bytes || d.window_rows != e.window_rows ||
        x.position != y.position || x.chunks != y.chunks ||
        x.chunks.capacity() != y.chunks.capacity() || !Same(x.tail, y.tail) ||
        !Same(x.private_state, y.private_state))
      return false;
  }
  return true;
}

// The outcome of one operation: its value, or that it was rejected.
template<class T>
std::optional<T> Outcome(const std::function<T()>& call,
                         const std::string& what) {
  try {
    return call();
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  } catch (const std::exception& e) {
    Fail(what + ": unexpected exception: " + e.what());
  }
}

struct Counts {
  std::size_t encodings{}, accepted{}, rejected{}, plans{};
} counts;

void CompareEncode(const DiskManifest& m, const std::string& what) {
  const auto expected = Outcome<std::vector<std::uint8_t>>(
      [&] { return reference::Encode(m); }, what + " reference encode");
  const auto actual = Outcome<std::vector<std::uint8_t>>(
      [&] { return EncodeManifest(m); }, what + " encode");
  Expect(expected.has_value() == actual.has_value(),
         what + ": encode accepted by one side only");
  Expect(!expected || *expected == *actual, what + ": encodings differ");
  ++counts.encodings;
}

// Decoding and the metadata plan of the same bytes.
void CompareDecode(std::span<const std::uint8_t> bytes,
                   const std::string& what) {
  const auto expected = Outcome<DiskManifest>(
      [&] { return reference::Decode(bytes); }, what + " reference decode");
  const auto actual = Outcome<DiskManifest>(
      [&] { return DecodeManifest(bytes); }, what + " decode");
  Expect(expected.has_value() == actual.has_value(),
         what + ": decode accepted by one side only");
  Expect(!expected || Same(*expected, *actual), what + ": manifests differ");
  ++(expected ? counts.accepted : counts.rejected);

  const auto plan = Outcome<reference::Plan>(
      [&] { return reference::PlanManifest(bytes); }, what + " plan");
  GufoCacheManifestPlan rust{};
  const auto status =
      gufo_cache_manifest_plan(bytes.data(), bytes.size(), &rust);
  Expect(status == GUFO_CACHE_COMPLETE || status == GUFO_CACHE_REJECTED,
         what + ": plan failed");
  Expect(plan.has_value() == (status == GUFO_CACHE_COMPLETE),
         what + ": plan accepted by one side only");
  if (plan) {
    // disk.cpp charges these sizes for the capacities decoding reserves.
    const auto retained = rust.input_bytes + rust.tokens * sizeof(Token) +
                          rust.components * sizeof(DiskComponent) +
                          rust.chunks * sizeof(DiskPayload);
    Expect(plan->retained == retained && plan->references == rust.references,
           what + ": plans differ");
    ++counts.plans;
  }
  if (expected) {
    const auto retained =
        expected->input.capacity() + expected->tokens.capacity() * 4 +
        expected->components.capacity() * sizeof(DiskComponent);
    std::size_t chunks{};
    for (const auto& c : expected->components)
      chunks += c.chunks.capacity();
    Expect(plan && plan->retained == retained + chunks * sizeof(DiskPayload),
           what + ": plan does not match the decoded capacities");
  }
}

DiskFileId File(Random& r) {
  DiskFileId id;
  for (auto& b : id)
    b = r.Next();
  return id;
}
DiskPayload Payload(Random& r, std::uint64_t bytes) {
  return {File(r), bytes, r.Chance(20) ? 0 : r.Next()};
}
// Row components of every geometry, window components and private state,
// within the format's limits.
DiskManifest Generate(Random& r, std::size_t tokens, std::size_t components) {
  DiskManifest m;
  m.checkpoint = {r.Next() | 1};
  m.lineage = {r.Chance(10) ? 1 : r.Next() | 1};
  for (auto& b : m.compatibility)
    b = r.Next();
  m.purpose = static_cast<CheckpointPurpose>(r.Below(4));
  m.rank = r.Chance(30) ? 0 : r.Next();
  m.input.resize(r.Chance(30) ? 0 : r.Below(r.Chance(5) ? 65537 : 200));
  for (auto& b : m.input)
    b = r.Next();
  m.tokens.resize(tokens);
  for (auto& t : m.tokens)
    t = r.Next();
  std::uint32_t id = r.Below(4);
  for (std::size_t i = 0; i < components; ++i) {
    id += 1 + r.Below(r.Chance(10) ? 1000000 : 3);
    DiskComponent c;
    auto& d = c.descriptor;
    d.id = {id};
    d.layout_version = r.Chance(80) ? 1 : 1 + r.Below(UINT32_MAX);
    const auto kind = r.Below(3);
    if (kind == 1) {
      d.kind = ComponentKind::kPrivateState;
      d.state_bytes = 1 + r.Below(r.Chance(10) ? UINT64_MAX - 1 : 4096);
      c.position = {d.id, tokens};
      c.private_state = Payload(r, d.state_bytes);
    } else {
      d.kind = kind ? ComponentKind::kWindowRows : ComponentKind::kAppendRows;
      d.rows_per_chunk = 1 + r.Below(r.Chance(20) ? 8 : 4096);
      d.row_bytes = 1 + r.Below(r.Chance(10) ? 1u << 20 : 64);
      if (kind)
        d.window_rows = 1 + r.Below(r.Chance(50) ? 64 : 20000);
      Rows rows = r.Chance(30) ? tokens : r.Below(r.Chance(50) ? 64 : 30000);
      // Keep the chunk count within the reference limit.
      while ((rows - FirstChunkRow(d, rows)) / d.rows_per_chunk > 2000)
        rows /= 2;
      c.position = {d.id, rows};
      const auto first = FirstChunkRow(d, rows);
      for (Rows row = first; row + d.rows_per_chunk <= rows;
           row += d.rows_per_chunk)
        c.chunks.push_back(Payload(r, d.row_bytes * d.rows_per_chunk));
      if (rows % d.rows_per_chunk)
        c.tail = Payload(r, d.row_bytes * (rows % d.rows_per_chunk));
    }
    m.components.push_back(std::move(c));
  }
  return m;
}

// One field of a valid manifest changed, often into an invalid one.
void Mutate(Random& r, DiskManifest& m) {
  auto& c = m.components[r.Below(m.components.size())];
  auto& d = c.descriptor;
  auto& other = m.components[r.Below(m.components.size())];
  const auto any = [&] {
    return r.Chance(50) ? r.Below(4) : r.Chance(50) ? r.Next() : UINT64_MAX;
  };
  switch (r.Below(28)) {
    case 0:
      m.checkpoint.value = 0;
      break;
    case 1:
      m.lineage.value = 0;
      break;
    case 2:
      m.purpose = static_cast<CheckpointPurpose>(r.Below(256));
      break;
    case 3:
      m.input.resize(65536 + r.Below(2));
      break;
    case 4:
      m.tokens.push_back(1);
      break;
    case 5:
      if (!m.tokens.empty())
        m.tokens.pop_back();
      break;
    case 6:
      d.id = other.descriptor.id;
      break;
    case 7:
      c.position.id.value = any();
      break;
    case 8:
      d.layout_version = 0;
      break;
    case 9:
      d.kind = static_cast<ComponentKind>(r.Below(256));
      break;
    case 10:
      d.row_bytes = any();
      break;
    case 11:
      d.rows_per_chunk = any();
      break;
    case 12:
      d.state_bytes = any();
      break;
    case 13:
      d.window_rows = any();
      break;
    case 14:
      c.position.valid_rows = any();
      break;
    case 15:
      c.position.valid_rows += r.Chance(50) ? 1 : -1;
      break;
    case 16:
      if (!c.chunks.empty())
        c.chunks.pop_back();
      break;
    case 17:
      c.chunks.push_back(c.chunks.empty() ? Payload(r, 1) : c.chunks[0]);
      break;
    case 18:
      if (!c.chunks.empty())
        c.chunks.back().bytes += 1;
      break;
    case 19:
      if (c.chunks.size() > 1)
        c.chunks.back().file = c.chunks.front().file;
      break;
    case 20:
      c.tail = r.Chance(50) ? std::nullopt : std::optional(Payload(r, any()));
      break;
    case 21:
      c.private_state = r.Chance(50) ? std::nullopt
                                     : std::optional(Payload(r, d.state_bytes));
      break;
    case 22:
      if (c.tail && other.private_state)
        c.tail->file = other.private_state->file;
      else if (c.tail && other.tail && &c != &other)
        c.tail->file = other.tail->file;
      break;
    case 23:
      m.components.clear();
      break;
    case 24:
      std::swap(c, other);
      break;
    case 25:
      m.components.push_back(m.components.back());
      break;
    case 26:
      if (c.tail)
        c.tail->bytes += 1;
      break;
    case 27:
      if (c.private_state)
        c.private_state->bytes += 1;
      break;
  }
}

std::vector<std::uint8_t> Signed(std::vector<std::uint8_t> body) {
  const auto crc = reference::Checksum(body);
  for (unsigned i = 0; i < 8; ++i)
    body.push_back(crc >> (8 * i));
  return body;
}

// Truncations, byte and bit changes, insertions and erasures, each also with
// a valid trailer so the structure behind the checksum is checked.
void Damage(Random& r, const std::vector<std::uint8_t>& bytes, unsigned n,
            bool every_truncation, const std::string& what) {
  const std::vector<std::uint8_t> body(bytes.begin(), bytes.end() - 8);
  const auto cuts = every_truncation ? bytes.size() : n;
  for (std::size_t i = 0; i < cuts; ++i) {
    const auto len = every_truncation ? i : r.Below(bytes.size());
    CompareDecode(std::span(bytes).first(len), what + " truncated");
    if (len <= body.size())
      CompareDecode(Signed({body.begin(), body.begin() + len}),
                    what + " signed truncation");
  }
  for (unsigned i = 0; i < n; ++i) {
    auto changed = body;
    // Bias towards the fixed fields and counts at the start.
    const auto at = r.Chance(40)
                        ? r.Below(std::min<std::size_t>(changed.size(), 120))
                        : r.Below(changed.size());
    switch (r.Below(5)) {
      case 0:
        changed[at] ^= 1u << r.Below(8);
        break;
      case 1:
        changed[at] = r.Next();
        break;
      case 2:
        changed[at] = r.Chance(50) ? 0 : 0xff;
        break;
      case 3:
        changed.insert(changed.begin() + at, r.Next());
        break;
      case 4:
        changed.erase(changed.begin() + at);
        break;
    }
    CompareDecode(Signed(changed), what + " signed mutation");
    changed.resize(changed.size() + 8);
    std::copy(bytes.end() - 8, bytes.end(), changed.end() - 8);
    CompareDecode(changed, what + " mutation");
  }
}

void Manifests() {
  Random r{1276};
  for (unsigned i = 0; i < 200; ++i) {
    const auto tokens = r.Chance(20) ? 0 : r.Below(r.Chance(5) ? 20000 : 300);
    const auto m = Generate(r, tokens, 1 + r.Below(r.Chance(10) ? 40 : 6));
    const auto what = "manifest " + std::to_string(i);
    CompareEncode(m, what);
    const auto bytes = reference::Encode(m);
    CompareDecode(bytes, what);
    // Each format version, re-signed: only manifests with window components
    // may use version 3.
    for (const std::uint8_t version : {2, 3}) {
      std::vector<std::uint8_t> body(bytes.begin(), bytes.end() - 8);
      body[8] = version;
      CompareDecode(Signed(std::move(body)), what + " version");
    }
    Damage(r, bytes, 100, bytes.size() < 1024, what);
    for (unsigned j = 0; j < 30; ++j) {
      auto mutated = m;
      for (auto k = 1 + r.Below(2); k > 0 && !mutated.components.empty(); --k)
        Mutate(r, mutated);
      CompareEncode(mutated, what + " mutated");
      if (const auto valid = Outcome<std::vector<std::uint8_t>>(
              [&] { return reference::Encode(mutated); }, what))
        CompareDecode(*valid, what + " mutated");
    }
  }
}

// Each limit, reached and exceeded.
void Limits() {
  Random r{64};
  auto m = Generate(r, 2 * 1024 * 1024, 1);
  m.components[0].descriptor = {{1}, 1, ComponentKind::kPrivateState, 0, 0, 8};
  m.components[0].position = {{1}, m.tokens.size()};
  m.components[0].chunks.clear();
  m.components[0].tail.reset();
  m.components[0].private_state = Payload(r, 8);
  m.input.assign(65536, 7);
  CompareEncode(m, "token and input limits");
  const auto bytes = reference::Encode(m);
  CompareDecode(bytes, "token and input limits");
  Damage(r, bytes, 3, false, "token and input limits");
  m.input.push_back(7);
  CompareEncode(m, "input limit exceeded");
  m.input.pop_back();
  m.tokens.push_back(1);
  m.components[0].position.valid_rows += 1;
  CompareEncode(m, "token limit exceeded");

  // 256 components holding 65,536 references.
  auto wide = Generate(r, 0, 0);
  for (std::uint32_t i = 0; i < 256; ++i) {
    DiskComponent c{{{i + 1}, 1, ComponentKind::kAppendRows, 2, 1, 0},
                    {{i + 1}, 256},
                    {},
                    {},
                    {}};
    for (unsigned j = 0; j < 256; ++j)
      c.chunks.push_back(Payload(r, 2));
    wide.components.push_back(std::move(c));
  }
  CompareEncode(wide, "component and reference limits");
  const auto wide_bytes = reference::Encode(wide);
  CompareDecode(wide_bytes, "component and reference limits");
  Damage(r, wide_bytes, 20, false, "component and reference limits");
  // The same bytes with a 65,537th reference: a tail, a private state or one
  // more chunk of the last component, whose presence bytes end the body.
  const std::vector<std::uint8_t> wide_body(wide_bytes.begin(),
                                            wide_bytes.end() - 8);
  const std::vector<std::uint8_t> extra(32, 7);
  for (const std::size_t flag : {2, 1}) {
    auto body = wide_body;
    body[body.size() - flag] = 1;
    body.insert(body.end() - flag + 1, extra.begin(), extra.end());
    CompareDecode(Signed(std::move(body)), "reference limit exceeded");
  }
  {
    auto body = wide_body;
    const auto count = body.size() - 2 - 256 * 32 - 4;
    body[count] += 1;
    body.insert(body.end() - 2, extra.begin(), extra.end());
    CompareDecode(Signed(std::move(body)), "reference limit exceeded");
  }
  // The last component's tail is reference 65,537.
  auto more = wide;
  auto& last = more.components.back();
  last.descriptor.rows_per_chunk = 2;
  last.position.valid_rows = 2 * 256 + 1;
  for (auto& chunk : last.chunks)
    chunk.bytes = 4;
  last.tail = Payload(r, 2);
  CompareEncode(more, "reference limit exceeded");
  more = wide;
  more.components.back().chunks.pop_back();
  more.components.back().position.valid_rows -= 1;
  more.components.push_back({{{1000}, 1, ComponentKind::kPrivateState, 0, 0, 8},
                             {{1000}, 0},
                             {},
                             {},
                             Payload(r, 8)});
  CompareEncode(more, "component limit exceeded");
}

void Checksums() {
  Random r{9};
  std::vector<std::uint8_t> data((1u << 20) + 64);
  for (auto& b : data)
    b = r.Next();
  std::size_t folded{};
  const auto compare = [&](std::size_t offset, std::size_t len,
                           std::uint64_t seed) {
    const auto bytes = std::span(data).subspan(offset, len);
    const auto expected = reference::ChecksumTableOnly(bytes, seed);
    Expect(reference::Checksum(bytes, seed) == expected,
           "reference paths differ at " + std::to_string(len));
    Expect(DiskChecksum(bytes, seed) == expected,
           "checksum differs at length " + std::to_string(len));
    if (len >= 64 && reference::CanFold()) {
      Expect(reference::ChecksumFoldOnly(bytes, seed) == expected,
             "reference fold differs at " + std::to_string(len));
      folded += len >= 256;
    }
  };
  for (std::size_t len = 0; len <= 1100; ++len)
    compare(r.Below(16), len, r.Chance(30) ? 0 : r.Next());
  for (unsigned i = 0; i < 300; ++i)
    compare(r.Below(64), r.Below(data.size() - 64),
            r.Chance(30) ? 0 : r.Next());
  compare(0, data.size(), 0);
  Expect(!reference::CanFold() || folded > 0, "the folding path never ran");
  // Pieces chain through the seed.
  for (unsigned i = 0; i < 20; ++i) {
    std::uint64_t crc = 0;
    for (std::size_t at = 0; at < data.size();) {
      const auto n =
          std::min<std::size_t>(1 + r.Below(70000), data.size() - at);
      crc = DiskChecksum(std::span(data).subspan(at, n), crc);
      at += n;
    }
    Expect(crc == reference::Checksum(data), "chained checksum differs");
  }
}
}  // namespace

int main() {
  const auto begin = std::chrono::steady_clock::now();
  Checksums();
  Manifests();
  Limits();
  Expect(counts.accepted > 1000 && counts.rejected > 50000,
         "too few accepted or rejected manifests");
  std::cout << "PASS: " << counts.encodings << " encodings, " << counts.accepted
            << " accepted and " << counts.rejected << " rejected decodings, "
            << counts.plans << " plans in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             begin)
                   .count()
            << " s\n";
}
