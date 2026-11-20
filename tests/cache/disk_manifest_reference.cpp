// A frozen copy of the disk manifest codec and CRC-64 of src/cache/disk.cpp
// in gufo 0.35.0, before they moved to Rust (src/core/cache_manifest).
// Only the names and namespace changed. disk_manifest_test compares the Rust
// codec with it; do not change it to follow the Rust code.
#include "tests/cache/disk_manifest_reference.hpp"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <set>
#include <stdexcept>

namespace gufo::cache::reference {
namespace {
constexpr std::array<std::uint8_t, 8> kMagic{'G', 'U', 'F', 'O',
                                             'M', 'N', 'F', '2'};
constexpr std::size_t kMaxTokens = std::size_t{2} * 1024 * 1024;
constexpr std::size_t kMaxComponents = 256, kMaxReferences = 65536;
constexpr std::size_t kMaxInputBytes = 65536;
[[noreturn]] void Invalid() {
  throw std::invalid_argument("invalid cache v2 manifest");
}
std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  if (b > UINT64_MAX - a)
    Invalid();
  return a + b;
}
std::uint64_t Multiply(std::uint64_t a, std::uint64_t b) {
  if (a && b > UINT64_MAX / a)
    Invalid();
  return a * b;
}
void Validate(const DiskManifest& m) {
  if (!m.checkpoint.value || !m.lineage.value || m.tokens.size() > kMaxTokens ||
      m.input.size() > kMaxInputBytes || m.components.empty() ||
      m.components.size() > kMaxComponents ||
      static_cast<unsigned>(m.purpose) >
          static_cast<unsigned>(CheckpointPurpose::kLearned))
    Invalid();
  std::set<DiskFileId> private_files;
  std::size_t references{};
  std::uint32_t previous{};
  bool first = true;
  for (const auto& c : m.components) {
    const auto& d = c.descriptor;
    const auto tail = c.tail, private_state = c.private_state;
    if (c.position.id != d.id || !d.layout_version ||
        (!first && d.id.value <= previous))
      Invalid();
    previous = d.id.value;
    first = false;
    references += c.chunks.size() + bool(tail) + bool(private_state);
    if (references > kMaxReferences)
      Invalid();
    if (HasRows(d.kind)) {
      if (!d.row_bytes || !d.rows_per_chunk || d.state_bytes || private_state ||
          (d.kind == ComponentKind::kWindowRows) != (d.window_rows != 0) ||
          c.chunks.size() != (c.position.valid_rows -
                              FirstChunkRow(d, c.position.valid_rows)) /
                                 d.rows_per_chunk)
        Invalid();
      (void)Multiply(d.row_bytes, c.position.valid_rows);
      auto full_bytes = Multiply(d.row_bytes, d.rows_per_chunk);
      std::set<DiskFileId> chunks;
      for (const auto& p : c.chunks)
        if (p.bytes != full_bytes || !chunks.insert(p.file).second)
          Invalid();
      auto remaining = c.position.valid_rows % d.rows_per_chunk;
      if (bool(tail) != bool(remaining) ||
          (tail.has_value() &&
           tail.value().bytes != Multiply(d.row_bytes, remaining)))
        Invalid();
    } else if (d.kind == ComponentKind::kPrivateState) {
      if (d.row_bytes || d.rows_per_chunk || !d.state_bytes || d.window_rows ||
          !c.chunks.empty() || tail || !private_state ||
          private_state->bytes != d.state_bytes ||
          c.position.valid_rows != m.tokens.size())
        Invalid();
    } else
      Invalid();
    for (auto p : {tail, private_state})
      if (p && !private_files.insert(p->file).second)
        Invalid();
  }
}
struct Writer {
  std::vector<std::uint8_t> bytes;
  void Number(std::uint64_t n, unsigned width) {
    if (bytes.size() + width > kMaxManifestBytes - 8)
      Invalid();
    for (unsigned i = 0; i < width; ++i) {
      bytes.push_back(n & 255);
      n >>= 8;
    }
  }
  void Blob(std::span<const std::uint8_t> b) {
    if (b.size() > kMaxManifestBytes - 8 - bytes.size())
      Invalid();
    bytes.insert(bytes.end(), b.begin(), b.end());
  }
  void Payload(const DiskPayload& p) {
    Blob(p.file);
    Number(p.bytes, 8);
    Number(p.checksum, 8);
  }
};
struct Reader {
  std::span<const std::uint8_t> bytes;
  std::uint64_t Number(unsigned width) {
    auto b = Blob(width);
    std::uint64_t n{};
    for (unsigned i = 0; i < width; ++i)
      n |= std::uint64_t(b[i]) << (8 * i);
    return n;
  }
  std::span<const std::uint8_t> Blob(std::size_t n) {
    if (n > bytes.size())
      Invalid();
    auto result = bytes.first(n);
    bytes = bytes.subspan(n);
    return result;
  }
  std::size_t Count(std::size_t max, std::size_t element_bytes) {
    auto n = Number(4);
    if (n > max || n > bytes.size() / element_bytes)
      Invalid();
    return n;
  }
  DiskPayload Payload() {
    DiskPayload p;
    auto id = Blob(p.file.size());
    std::copy(id.begin(), id.end(), p.file.begin());
    p.bytes = Number(8);
    p.checksum = Number(8);
    return p;
  }
};
Plan PlanMetadata(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 20 || bytes.size() > kMaxManifestBytes)
    Invalid();
  Reader r{bytes.first(bytes.size() - 8)};
  r.Blob(65);  // Fixed fields preceding the input length.
  Plan plan;
  auto input = r.Count(kMaxInputBytes, 1);
  r.Blob(input);
  plan.retained = input;
  auto tokens = r.Count(kMaxTokens, 4);
  r.Blob(tokens * sizeof(Token));
  plan.retained = Add(plan.retained, tokens * sizeof(Token));
  auto components = r.Count(kMaxComponents, 47);
  plan.retained = Add(plan.retained, components * sizeof(DiskComponent));
  for (std::size_t i = 0; i < components; ++i) {
    // The kind byte follows the ID and layout version; a window component
    // also carries its window size.
    if (r.Blob(41)[8] == static_cast<std::uint8_t>(ComponentKind::kWindowRows))
      r.Blob(8);
    auto chunks = r.Count(kMaxReferences - plan.references, 32);
    plan.references += chunks;
    r.Blob(chunks * 32);
    plan.retained = Add(plan.retained, chunks * sizeof(DiskPayload));
    for (unsigned j = 0; j < 2; ++j) {
      auto present = r.Number(1);
      if (present > 1)
        Invalid();
      if (present) {
        if (++plan.references > kMaxReferences)
          Invalid();
        r.Blob(32);
      }
    }
  }
  if (!r.bytes.empty())
    Invalid();
  return plan;
}
std::uint64_t ChecksumTable(std::span<const std::uint8_t> bytes,
                            std::uint64_t crc) {
  static constexpr auto table = [] {
    std::array<std::array<std::uint64_t, 256>, 8> t{};
    for (unsigned i = 0; i < 256; ++i) {
      auto n = std::uint64_t(i) << 56;
      for (unsigned j = 0; j < 8; ++j)
        n = (n << 1) ^ ((n >> 63) ? 0x42f0e1eba9ea3693ULL : 0);
      t[0][i] = n;
    }
    for (unsigned slice = 1; slice < t.size(); ++slice)
      for (unsigned i = 0; i < 256; ++i) {
        const auto previous = t[slice - 1][i];
        t[slice][i] = (previous << 8) ^ t[0][previous >> 56];
      }
    return t;
  }();
  // Preserve CRC-64/ECMA and streaming seeds while breaking the per-byte
  // dependency chain. memcpy permits arbitrary payload alignment.
  while (bytes.size() >= 8) {
    std::uint64_t word;
    std::memcpy(&word, bytes.data(), sizeof(word));
    if constexpr (std::endian::native == std::endian::little)
      word = __builtin_bswap64(word);
    crc ^= word;
    crc = table[7][crc >> 56] ^ table[6][(crc >> 48) & 255] ^
          table[5][(crc >> 40) & 255] ^ table[4][(crc >> 32) & 255] ^
          table[3][(crc >> 24) & 255] ^ table[2][(crc >> 16) & 255] ^
          table[1][(crc >> 8) & 255] ^ table[0][crc & 255];
    bytes = bytes.subspan(8);
  }
  for (auto b : bytes)
    crc = (crc << 8) ^ table[0][(crc >> 56) ^ b];
  return crc;
}
constexpr std::uint64_t CrcPower(unsigned exponent) {
  std::uint64_t result = 1;
  while (exponent--)
    result = (result << 1) ^ ((result >> 63) ? 0x42f0e1eba9ea3693ULL : 0);
  return result;
}
__attribute__((target("pclmul,ssse3"))) inline __m128i CrcLoad(
    const std::uint8_t* bytes) {
  // Treat the big-endian byte sequence as a polynomial with its low limb in
  // the low SIMD lane. Loads remain unaligned and never pass the input end.
  return _mm_shuffle_epi8(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes)),
      _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
}
__attribute__((target("pclmul,ssse3"))) inline __m128i CrcFold(__m128i previous,
                                                               __m128i powers,
                                                               __m128i next) {
  return _mm_xor_si128(
      _mm_xor_si128(_mm_clmulepi64_si128(previous, powers, 0x00),
                    _mm_clmulepi64_si128(previous, powers, 0x11)),
      next);
}
__attribute__((target("pclmul,ssse3"))) std::uint64_t ChecksumFold(
    std::span<const std::uint8_t> bytes, std::uint64_t crc) {
  // Fold in GF(2), modulo the same ECMA polynomial. A low limb advances by
  // x^128 (or x^512); its high limb advances another 64 bits. Four independent
  // lanes hide carry-less multiply latency. Their final merge restores byte
  // order before the unchanged table reduction and partial-byte suffix.
  const auto powers = _mm_set_epi64x(CrcPower(192), CrcPower(128));
  const auto wide_powers = _mm_set_epi64x(CrcPower(576), CrcPower(512));
  auto a = _mm_xor_si128(CrcLoad(bytes.data()), _mm_set_epi64x(crc, 0));
  auto b = CrcLoad(bytes.data() + 16);
  auto c = CrcLoad(bytes.data() + 32);
  auto d = CrcLoad(bytes.data() + 48);
  bytes = bytes.subspan(64);
  while (bytes.size() >= 64) {
    a = CrcFold(a, wide_powers, CrcLoad(bytes.data()));
    b = CrcFold(b, wide_powers, CrcLoad(bytes.data() + 16));
    c = CrcFold(c, wide_powers, CrcLoad(bytes.data() + 32));
    d = CrcFold(d, wide_powers, CrcLoad(bytes.data() + 48));
    bytes = bytes.subspan(64);
  }
  a = CrcFold(CrcFold(CrcFold(a, powers, b), powers, c), powers, d);
  while (bytes.size() >= 16) {
    a = CrcFold(a, powers, CrcLoad(bytes.data()));
    bytes = bytes.subspan(16);
  }
  std::array<std::uint8_t, 16> remainder;
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(remainder.data()),
      _mm_shuffle_epi8(a, _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                                        4, 3, 2, 1, 0)));
  return ChecksumTable(bytes, ChecksumTable(remainder, 0));
}
}  // namespace
Plan PlanManifest(std::span<const std::uint8_t> bytes) {
  return PlanMetadata(bytes);
}
std::uint64_t ChecksumTableOnly(std::span<const std::uint8_t> bytes,
                                std::uint64_t crc) {
  return ChecksumTable(bytes, crc);
}
bool CanFold() {
  return __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("ssse3");
}
std::uint64_t ChecksumFoldOnly(std::span<const std::uint8_t> bytes,
                               std::uint64_t crc) {
  if (!CanFold() || bytes.size() < 64)
    throw std::logic_error("the folding path needs PCLMUL and 64 bytes");
  return ChecksumFold(bytes, crc);
}
std::uint64_t Checksum(std::span<const std::uint8_t> bytes, std::uint64_t crc) {
  static const bool folding =
      __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("ssse3");
  if (folding && bytes.size() >= 256)
    return ChecksumFold(bytes, crc);
  return ChecksumTable(bytes, crc);
}
std::vector<std::uint8_t> Encode(const DiskManifest& m) {
  Validate(m);
  const bool windowed = std::ranges::any_of(m.components, [](const auto& c) {
    return c.descriptor.kind == ComponentKind::kWindowRows;
  });
  Writer w;
  w.Blob(kMagic);
  w.Number(windowed ? kDiskWindowFormatVersion : kDiskFormatVersion, 4);
  w.Number(m.checkpoint.value, 8);
  w.Number(m.lineage.value, 8);
  w.Blob(m.compatibility);
  w.Number(static_cast<unsigned>(m.purpose), 1);
  w.Number(m.rank, 4);
  w.Number(m.input.size(), 4);
  w.Blob(m.input);
  w.Number(m.tokens.size(), 4);
  for (auto t : m.tokens)
    w.Number(t, 4);
  w.Number(m.components.size(), 4);
  for (const auto& c : m.components) {
    const auto& d = c.descriptor;
    const auto tail = c.tail, private_state = c.private_state;
    w.Number(d.id.value, 4);
    w.Number(d.layout_version, 4);
    w.Number(static_cast<unsigned>(d.kind), 1);
    w.Number(d.row_bytes, 8);
    w.Number(d.rows_per_chunk, 8);
    w.Number(d.state_bytes, 8);
    w.Number(c.position.valid_rows, 8);
    if (d.kind == ComponentKind::kWindowRows)
      w.Number(d.window_rows, 8);
    w.Number(c.chunks.size(), 4);
    for (const auto& p : c.chunks)
      w.Payload(p);
    w.Number(bool(tail), 1);
    if (tail.has_value())
      w.Payload(tail.value());
    w.Number(bool(private_state), 1);
    if (private_state.has_value())
      w.Payload(private_state.value());
  }
  auto checksum = Checksum(w.bytes);
  // The checksum is the final eight bytes and excluded from its own input.
  for (unsigned i = 0; i < 8; ++i)
    w.bytes.push_back((checksum >> (8 * i)) & 255);
  return std::move(w.bytes);
}
DiskManifest Decode(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 20 || bytes.size() > kMaxManifestBytes)
    Invalid();
  Reader trailer{bytes.last(8)};
  if (Checksum(bytes.first(bytes.size() - 8)) != trailer.Number(8))
    Invalid();
  Reader r{bytes.first(bytes.size() - 8)};
  auto magic = r.Blob(kMagic.size());
  if (!std::equal(magic.begin(), magic.end(), kMagic.begin()))
    Invalid();
  const auto version = r.Number(4);
  if (version != kDiskFormatVersion && version != kDiskWindowFormatVersion)
    Invalid();
  DiskManifest m;
  m.checkpoint.value = r.Number(8);
  m.lineage.value = r.Number(8);
  auto identity = r.Blob(m.compatibility.size());
  std::copy(identity.begin(), identity.end(), m.compatibility.begin());
  m.purpose = static_cast<CheckpointPurpose>(r.Number(1));
  m.rank = r.Number(4);
  auto input = r.Blob(r.Count(kMaxInputBytes, 1));
  m.input.assign(input.begin(), input.end());
  auto tokens = r.Count(kMaxTokens, 4);
  m.tokens.reserve(tokens);
  for (std::size_t i = 0; i < tokens; ++i)
    m.tokens.push_back(r.Number(4));
  auto components = r.Count(kMaxComponents, 47);
  m.components.reserve(components);
  std::size_t references{};
  for (std::size_t i = 0; i < components; ++i) {
    DiskComponent c;
    auto& d = c.descriptor;
    d.id.value = r.Number(4);
    d.layout_version = r.Number(4);
    d.kind = static_cast<ComponentKind>(r.Number(1));
    d.row_bytes = r.Number(8);
    d.rows_per_chunk = r.Number(8);
    d.state_bytes = r.Number(8);
    c.position = {d.id, r.Number(8)};
    if (d.kind == ComponentKind::kWindowRows) {
      if (version != kDiskWindowFormatVersion)
        Invalid();
      d.window_rows = r.Number(8);
    }
    auto chunks = r.Count(kMaxReferences - references, 32);
    c.chunks.reserve(chunks);
    references += chunks;
    for (std::size_t j = 0; j < chunks; ++j)
      c.chunks.push_back(r.Payload());
    auto optional_payload = [&]() -> std::optional<DiskPayload> {
      auto present = r.Number(1);
      if (present > 1)
        Invalid();
      if (!present)
        return {};
      if (++references > kMaxReferences)
        Invalid();
      return r.Payload();
    };
    c.tail = optional_payload();
    c.private_state = optional_payload();
    m.components.push_back(std::move(c));
  }
  // Only manifests with window components use the window format version, so
  // every manifest has one canonical encoding.
  if (!r.bytes.empty() ||
      (version == kDiskWindowFormatVersion &&
       std::ranges::none_of(m.components, [](const auto& c) {
         return c.descriptor.kind == ComponentKind::kWindowRows;
       })))
    Invalid();
  Validate(m);
  return m;
}
}  // namespace gufo::cache::reference
