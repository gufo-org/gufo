#include "src/cache/disk.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include "src/core/platform/fs.hpp"

namespace gufo::cache {
namespace detail {
struct DiskAvailability {
  ResourceCharge metadata;
  std::atomic<bool> quarantined{false};
};
}  // namespace detail
namespace {
template<class F>
struct Finally {
  F action;
  ~Finally() { action(); }
};
DiskPublicationStats Difference(const DiskPublicationStats& after,
                                const DiskPublicationStats& before) {
  return {after.fsync_calls - before.fsync_calls,
          after.fsync_ns - before.fsync_ns,
          after.filesystem_ns - before.filesystem_ns,
          after.checksum_ns - before.checksum_ns,
          after.metadata_lock_ns - before.metadata_lock_ns,
          after.io_lock_ns - before.io_lock_ns};
}
class CostTimer {
public:
  explicit CostTimer(std::uint64_t& cost)
      : cost_(cost), start_(std::chrono::steady_clock::now()) {}
  ~CostTimer() {
    cost_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - start_)
                 .count();
  }

private:
  std::uint64_t& cost_;
  std::chrono::steady_clock::time_point start_;
};
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
struct MetadataPlan {
  std::size_t retained{}, references{};
};
MetadataPlan PlanMetadata(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 20 || bytes.size() > kMaxManifestBytes)
    Invalid();
  Reader r{bytes.first(bytes.size() - 8)};
  r.Blob(65);  // Fixed fields preceding the input length.
  MetadataPlan plan;
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
class Fd {
public:
  explicit Fd(platform::PlatformFile value = platform::kInvalidFile)
      : value_(value) {}
  ~Fd() { platform::CloseFile(value_); }
  Fd(Fd&& o) noexcept : value_(o.value_) { o.value_ = platform::kInvalidFile; }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd& operator=(Fd&&) = delete;
  platform::PlatformFile Get() const { return value_; }
  void Swap(Fd& other) { std::swap(value_, other.value_); }

private:
  platform::PlatformFile value_;
};
[[noreturn]] void IoError(const std::string& message,
                          const std::error_code& ec) {
  throw std::runtime_error(message + ": " + ec.message());
}
void SyncDirectoryFd(const platform::Directory& directory) {
  std::error_code ec;
  if (!platform::SyncDirectory(directory, ec))
    IoError("sync cache file/directory", ec);
}
platform::Directory DirectoryAt(const platform::Directory& parent,
                                const char* name) {
  std::error_code ec;
  if (!platform::MakeDirectoryAt(parent, name, ec))
    IoError("create cache directory", ec);
  auto directory = platform::OpenDirectoryAt(parent, name, ec);
  if (!directory.valid())
    IoError(std::string("open cache directory ") + name, ec);
  return directory;
}
template<typename F>
void Scan(const platform::Directory& directory, F visit) {
  std::error_code ec;
  if (!platform::ListDirectory(
          directory,
          [&](std::string_view name, const platform::FileInfo& info) {
            // ListDirectory reports every entry; callers expect regular
            // files only.
            if (info.regular)
              visit(std::string(name), info);
          },
          ec))
    IoError("scan cache directory", ec);
}
std::optional<DiskFileId> ParseName(const std::string& name) {
  if (name.size() != 36 || name.substr(32) != ".bin")
    return {};
  DiskFileId id;
  for (std::size_t i = 0; i < 16; ++i) {
    auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      return -1;
    };
    const int a = hex(name[2 * i]), b = hex(name[2 * i + 1]);
    if (a < 0 || b < 0)
      return {};
    id[i] = (a << 4) | b;
  }
  return id;
}
bool Regular(const platform::FileInfo& info) {
  return info.regular && info.links == 1;
}
bool Contains(const DiskManifest& manifest, bool private_file,
              const DiskPayload& payload) {
  for (const auto& c : manifest.components)
    if (private_file ? c.tail == payload || c.private_state == payload
                     : std::find(c.chunks.begin(), c.chunks.end(), payload) !=
                           c.chunks.end())
      return true;
  return false;
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
std::uint64_t DiskChecksum(std::span<const std::uint8_t> bytes,
                           std::uint64_t crc) {
  static const bool folding =
      __builtin_cpu_supports("pclmul") && __builtin_cpu_supports("ssse3");
  if (folding && bytes.size() >= 256)
    return ChecksumFold(bytes, crc);
  return ChecksumTable(bytes, crc);
}
std::vector<std::uint8_t> EncodeManifest(const DiskManifest& m) {
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
  auto checksum = DiskChecksum(w.bytes);
  // The checksum is the final eight bytes and excluded from its own input.
  for (unsigned i = 0; i < 8; ++i)
    w.bytes.push_back((checksum >> (8 * i)) & 255);
  return std::move(w.bytes);
}
DiskManifest DecodeManifest(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < 20 || bytes.size() > kMaxManifestBytes)
    Invalid();
  Reader trailer{bytes.last(8)};
  if (DiskChecksum(bytes.first(bytes.size() - 8)) != trailer.Number(8))
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
std::string DiskFileName(DiskFileId id) {
  constexpr char hex[] = "0123456789abcdef";
  std::string name;
  for (auto b : id) {
    name += hex[b >> 4];
    name += hex[b & 15];
  }
  return name + ".bin";
}
struct DiskStore::Impl {
  ResourceLedger* ledger;
  ResourceCharge metadata, entry_capacity;
  platform::Directory root;
  Fd lock;
  platform::Directory version, manifests, chunks, private_files, temporary;
  DiskStartupStats stats;
  std::vector<DurableEntry> entries;
  std::uint64_t budget_bytes;
  DiskPublicationStats publication_stats;
  bool recovery_required{};
  std::uint64_t next_epoch{1};
  DiskRetireResult Retire(CheckpointId, std::uint64_t);
  mutable std::timed_mutex mutex;
  mutable std::mutex index_mutex;
  mutable std::mutex stats_mutex;
  struct FileIdentity {
    platform::FileInfo info;
    std::uint64_t checksum;
    bool operator==(const FileIdentity& b) const = default;
  };
  struct VerifiedFile {
    ResourceCharge charge;
    FileIdentity identity;
  };
  std::mutex verification_mutex;
  // One entry per payload file whose bytes matched its checksum, until the
  // file is removed. A matching identity skips the next re-read.
  std::map<std::pair<bool, DiskFileId>, VerifiedFile> verified;
  bool VerificationCached(std::pair<bool, DiskFileId> key,
                          const FileIdentity& identity) {
    std::lock_guard guard(verification_mutex);
    auto it = verified.find(key);
    return it != verified.end() && it->second.identity == identity;
  }
  void RememberVerified(std::pair<bool, DiskFileId> key,
                        const FileIdentity& identity) {
    std::lock_guard guard(verification_mutex);
    auto it = verified.find(key);
    if (it != verified.end()) {
      it->second.identity = identity;
      return;
    }
    try {
      auto reservation = ledger->Reserve(ResourceCategory::kMetadata, 256);
      auto [inserted, added] =
          verified.emplace(key, VerifiedFile{{}, identity});
      try {
        inserted->second.charge = reservation.Convert();
      } catch (...) {
        verified.erase(inserted);
        throw;
      }
    } catch (const std::bad_alloc&) {
      // Verification caching is optional; the completed read remains valid.
    }
  }
  void ForgetVerified(std::pair<bool, DiskFileId> key) {
    std::unique_lock guard(verification_mutex);
    auto node = verified.extract(key);
    guard.unlock();
    // The charge returns to the ledger outside the verification lock.
    node = {};
  }
#ifdef GUFO_CACHE_TESTING
  std::function<void(DiskPublicationStep)> crash_hook;
#define DISK_STEP(step)                      \
  do {                                       \
    if (crash_hook)                          \
      crash_hook(DiskPublicationStep::step); \
  } while (false)
#else
#define DISK_STEP(step) \
  do {                  \
  } while (false)
#endif
  // Declared last: future joins before state/FDs/lock are destroyed.
  std::future<void> reclamation;

  Impl(ResourceLedger& l, const std::filesystem::path& path,
       std::uint64_t budget, ResourceReservation reservation)
      : ledger(&l),
        metadata(reservation.Convert()),
        root(OpenRoot(path)),
        lock(OpenLock(root, path)),
        version(DirectoryAt(root, "v2")),
        manifests(DirectoryAt(version, "manifests")),
        chunks(DirectoryAt(version, "chunks")),
        private_files(DirectoryAt(version, "private")),
        temporary(DirectoryAt(version, "tmp")),
        budget_bytes(budget) {
    // Persist the layout, including a newly created root and ancestors, before
    // any checkpoint can rely on these directory entries.
    SyncDirectoryFd(version);
    SyncDirectoryFd(root);
    std::error_code ec;
    if (!platform::SyncAncestors(root, ec))
      IoError("sync cache ancestors", ec);
    CleanupLegacy();
    std::size_t manifest_count{};
    for (const platform::Directory* directory :
         {&manifests, &chunks, &private_files})
      Scan(*directory,
           [&](const std::string& name, const platform::FileInfo& st) {
             if (ParseName(name)) {
               stats.managed_bytes = Add(stats.managed_bytes, st.size);
               if (directory == &manifests)
                 ++manifest_count;
             }
           });
    Scan(temporary, [&](const std::string& name, const platform::FileInfo& st) {
      if (ParseName(name))
        stats.managed_bytes = Add(stats.managed_bytes, st.size);
    });
    if (manifest_count) {
      auto entry_reservation =
          ledger->Reserve(ResourceCategory::kMetadata,
                          Multiply(manifest_count, sizeof(DurableEntry)));
      std::vector<DurableEntry> allocated;
      allocated.reserve(manifest_count);
      if (allocated.capacity() != manifest_count)
        throw std::logic_error("unexpected disk index capacity");
      entry_capacity = entry_reservation.Convert();
      entries.swap(allocated);
    }
    Scan(manifests, [&](const std::string& name, const platform::FileInfo& st) {
      auto id = ParseName(name);
      if (!id)
        return;
      try {
        auto entry = ReadManifest(name, st);
        auto& m = entry.manifest_;
        bool available = true;
        for (const auto& c : m.components) {
          for (const auto& p : c.chunks)
            available = Check(chunks, p) && available;
          for (auto p : {c.tail, c.private_state})
            if (p)
              available = Check(private_files, *p) && available;
        }
        if (!available)
          Invalid();
        entry.file_ = *id;
        entry.epoch_ = next_epoch++;
        entries.push_back(std::move(entry));
      } catch (const std::invalid_argument&) {
        ++stats.rejected_manifests;
      }
    });
    ValidateReferences();
    // Rejected manifests still own on-disk claims until durably reclaimed. A
    // new write could otherwise fill their missing dependencies and resurrect
    // conflicting checkpoint identities on the next restart.
    recovery_required = stats.rejected_manifests != 0;
    // A process abort can leave unsynced renames or deletions in any managed
    // namespace. Persist the discovered namespace before exposing entries or
    // spending deletion credits; no payload reads or orphan cleanup are needed.
    for (const platform::Directory* directory :
         {&manifests, &chunks, &private_files, &temporary})
      SyncDirectoryFd(*directory);
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.file_ < b.file_; });
  }
  ~Impl() {
    if (reclamation.valid())
      reclamation.wait();
    while (!verified.empty()) {
      auto node = verified.extract(verified.begin());
      auto charge = std::move(node.mapped().charge);
      node = {};
    }
  }
  void ValidateReferences(bool reject = true) {
    if (entries.empty())
      return;
    // Shared files must describe the same immutable lineage/range/layout.
    // Private bytes cannot be aliased across checkpoints. Reject every side
    // of a conflict, independent of filesystem enumeration order.
    struct Claim {
      CompatibilityDigest compatibility;
      std::vector<std::uint64_t> geometry;
      std::vector<std::size_t> entries;
      bool conflict{};
    };
    std::size_t references{};
    for (const auto& entry : entries)
      for (const auto& c : entry.manifest_.components)
        references = Add(
            references, c.chunks.size() + bool(c.tail) + bool(c.private_state));
    // Cover map records, geometry arrays, vector growth and tree links before
    // building the temporary cross-manifest reference catalog.
    auto working = ledger->Reserve(
        ResourceCategory::kMetadata,
        Add(Multiply(references, sizeof(Claim) + sizeof(DiskFileId) +
                                     24 * sizeof(std::uint64_t)),
            Multiply(entries.size(), 96)));
    std::map<std::pair<bool, DiskFileId>, Claim> claims;
    std::map<std::uint64_t, std::vector<std::size_t>> checkpoints;
    std::vector<bool> rejected(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto& m = entries[i].manifest_;
      checkpoints[m.checkpoint.value].push_back(i);
      auto claim = [&](bool is_private, const DiskPayload& p,
                       const ComponentDescriptor& d, std::uint64_t first) {
        const std::vector<std::uint64_t> geometry{
            m.lineage.value, d.id.value,       d.layout_version,
            d.row_bytes,     d.rows_per_chunk, first,
            p.bytes,         p.checksum,       d.window_rows};
        auto [it, inserted] = claims.try_emplace(
            {is_private, p.file}, Claim{m.compatibility, geometry, {}, false});
        auto& value = it->second;
        if (!inserted &&
            (is_private || value.compatibility != m.compatibility ||
             value.geometry != geometry))
          value.conflict = true;
        value.entries.push_back(i);
      };
      for (const auto& c : m.components) {
        Rows first = FirstChunkRow(c.descriptor, c.position.valid_rows);
        for (const auto& p : c.chunks) {
          claim(false, p, c.descriptor, first);
          first = Add(first, c.descriptor.rows_per_chunk);
        }
        for (auto p : {c.tail, c.private_state})
          if (p)
            claim(true, *p, c.descriptor, first);
      }
    }
    for (const auto& [key, claim] : claims)
      if (claim.conflict)
        for (auto i : claim.entries)
          rejected[i] = true;
    for (const auto& [id, owners] : checkpoints)
      if (owners.size() > 1)
        for (auto i : owners)
          rejected[i] = true;
    if (!reject &&
        std::find(rejected.begin(), rejected.end(), true) != rejected.end())
      Invalid();
    std::size_t i{};
    std::erase_if(entries, [&](const auto&) {
      const bool remove = rejected[i++];
      if (remove)
        ++stats.rejected_manifests;
      return remove;
    });
  }
  void Sync(platform::PlatformFile file) {
    auto begin = std::chrono::steady_clock::now();
    std::error_code ec;
    const bool synced = platform::SyncFile(file, ec);
    ++publication_stats.fsync_calls;
    publication_stats.fsync_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    if (!synced)
      IoError("sync cache file/directory", ec);
  }
  void Sync(const platform::Directory& directory) {
    auto begin = std::chrono::steady_clock::now();
    std::error_code ec;
    const bool synced = platform::SyncDirectory(directory, ec);
    ++publication_stats.fsync_calls;
    publication_stats.fsync_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    if (!synced)
      IoError("sync cache file/directory", ec);
  }
  void Remove(const platform::Directory& directory, const std::string& name) {
    platform::FileInfo st{};
    std::error_code ec;
    if (!platform::StatAt(directory, name, false, &st, ec)) {
      if (ec == std::errc::no_such_file_or_directory)
        return;
      IoError("stat cache removal", ec);
    }
    if (!Regular(st))
      return;
    if (&directory == &chunks || &directory == &private_files)
      if (auto id = ParseName(name))
        ForgetVerified({&directory == &private_files, *id});
    if (!platform::Unlink(directory, name, ec))
      IoError("remove cache file", ec);
    {
      std::lock_guard guard(stats_mutex);
      stats.managed_bytes -= st.size;
    }
  }
  bool Referenced(const platform::Directory& directory, DiskFileId id) const {
    for (const auto& entry : entries) {
      if (&directory == &manifests && entry.file_ == id)
        return true;
      for (const auto& c : entry.manifest_.components) {
        if (&directory == &chunks)
          for (const auto& p : c.chunks)
            if (p.file == id)
              return true;
        if (&directory == &private_files)
          for (auto p : {c.tail, c.private_state})
            if (p && p->file == id)
              return true;
      }
    }
    return false;
  }
  void RequireRecovered() const {
    if (recovery_required)
      throw std::runtime_error(
          "cache durability outcome uncertain; run orphan recovery before "
          "publication or retirement");
  }
  void Reclaim() {
    // Any incomplete cleanup must establish deletion durability before its
    // released byte capacity can fund another publication.
    recovery_required = true;
    // Even invalid manifests are removed and that removal made durable before
    // dependencies: a crash must never resurrect a manifest with deleted bytes.
    Scan(manifests, [&](const auto& name, const auto&) {
      auto id = ParseName(name);
      if (id && !Referenced(manifests, *id))
        Remove(manifests, name);
    });
    Sync(manifests);
    for (const platform::Directory* directory :
         {&chunks, &private_files, &temporary}) {
      Scan(*directory, [&](const auto& name, const auto&) {
        auto id = ParseName(name);
        if (id && !Referenced(*directory, *id))
          Remove(*directory, name);
      });
      Sync(*directory);
    }
    // A failed unlink syscall can leave its outcome uncertain. Reconcile the
    // conservative charge with the actual namespace after successful cleanup.
    std::uint64_t managed{};
    for (const platform::Directory* directory :
         {&manifests, &chunks, &private_files, &temporary})
      Scan(*directory, [&](const auto& name, const auto& st) {
        if (ParseName(name))
          managed = Add(managed, st.size);
      });
    {
      std::lock_guard guard(stats_mutex);
      stats.managed_bytes = managed;
    }
    recovery_required = false;
  }
  void Rename(const platform::Directory& target, const std::string& name) {
    CostTimer timer(publication_stats.filesystem_ns);
    std::error_code ec;
    if (platform::RenameNoReplace(temporary, name, target, name, ec) !=
        platform::RenameResult::kRenamed)
      IoError("publish cache file", ec);
  }
  void WriteFile(platform::PlatformFile fd,
                 std::span<const std::uint8_t> bytes) {
    while (!bytes.empty()) {
      std::size_t n;
      {
        CostTimer timer(publication_stats.filesystem_ns);
        std::error_code ec;
        n = platform::Write(fd, bytes.data(), bytes.size(), ec);
        if (n == 0) {
          if (ec == std::errc::interrupted)
            continue;
          IoError("write cache file", ec);
        }
      }
      {
        std::unique_lock guard(stats_mutex, std::defer_lock);
        {
          CostTimer timer(publication_stats.metadata_lock_ns);
          guard.lock();
        }
        stats.managed_bytes += n;
      }
      bytes = bytes.subspan(n);
    }
  }
  Fd Create(const std::string& name) {
    CostTimer timer(publication_stats.filesystem_ns);
    std::error_code ec;
    Fd fd(
        platform::OpenAt(temporary, name, platform::FileMode::kCreateNew, ec));
    if (!platform::FileValid(fd.Get()))
      IoError("create temporary cache file", ec);
    return fd;
  }
  bool Exists(const platform::Directory& directory, const std::string& name) {
    CostTimer timer(publication_stats.filesystem_ns);
    platform::FileInfo st{};
    std::error_code ec;
    if (platform::StatAt(directory, name, false, &st, ec))
      return true;
    if (ec == std::errc::no_such_file_or_directory)
      return false;
    IoError("stat publication destination", ec);
  }
  void Quarantine(bool private_file, DiskFileId file) {
    std::lock_guard guard(index_mutex);
    for (auto& entry : entries)
      for (const auto& component : entry.manifest_.components) {
        bool matches{};
        if (private_file) {
          for (const auto& payload : {component.tail, component.private_state})
            matches |= payload && payload->file == file;
        } else {
          for (const auto& payload : component.chunks)
            matches |= payload.file == file;
        }
        if (matches) {
          entry.available_ = false;
          entry.availability_->quarantined.store(true);
        }
      }
  }
  [[noreturn]] void DependencyFailure(bool private_file, DiskFileId file) {
    Quarantine(private_file, file);
    throw DiskDependencyError(private_file, file);
  }
  std::shared_ptr<detail::DiskAvailability> NewAvailability() {
    auto admission = ledger->Reserve(ResourceCategory::kMetadata,
                                     sizeof(detail::DiskAvailability) + 64);
    auto result = std::make_shared<detail::DiskAvailability>();
    result->metadata = admission.Convert();
    return result;
  }
  void Verify(const platform::Directory& directory, const DiskPayload& payload,
              std::span<std::uint8_t> staging = {},
              const DiskStagingAccess* access = nullptr) {
    // Reject a caller's conflicting claim without poisoning existing owners.
    // Only a failure against the store's attested checksum/length quarantines.
    {
      std::lock_guard guard(index_mutex);
      for (const auto& entry : entries)
        for (const auto& component : entry.manifest_.components) {
          if (&directory == &private_files) {
            for (const auto& existing :
                 {component.tail, component.private_state})
              if (existing && existing->file == payload.file &&
                  *existing != payload)
                Invalid();
          } else {
            for (const auto& existing : component.chunks)
              if (existing.file == payload.file && existing != payload)
                Invalid();
          }
        }
    }
    std::error_code ec;
    platform::PlatformFile descriptor;
    {
      CostTimer timer(publication_stats.filesystem_ns);
      descriptor = platform::OpenAt(directory, DiskFileName(payload.file),
                                    platform::FileMode::kRead, ec);
    }
    Fd fd(descriptor);
    platform::FileInfo st{};
    int status;
    {
      CostTimer timer(publication_stats.filesystem_ns);
      status = platform::Stat(fd.Get(), &st, ec) ? 0 : -1;
    }
    const bool private_file = &directory == &private_files;
    if (!platform::FileValid(fd.Get()) || status < 0 || !Regular(st) ||
        st.size != payload.bytes)
      DependencyFailure(private_file, payload.file);
    // Each publication inherits the chunks of earlier checkpoints. Re-read
    // only those whose identity changed since this store last verified them.
    const auto key = std::make_pair(private_file, payload.file);
    const FileIdentity identity{st, payload.checksum};
    if (VerificationCached(key, identity))
      return;
    ResourceReservation charge;
    std::vector<std::uint8_t> allocated;
    if (staging.empty()) {
      charge = ledger->Reserve(ResourceCategory::kMetadata, 65536);
      allocated.resize(65536);
      staging = allocated;
    }
    std::uint64_t remaining = payload.bytes, crc{};
    while (remaining) {
      Finally release{[&] {
        if (access)
          access->release();
      }};
      if (access)
        access->acquire();
      std::size_t n;
      {
        CostTimer timer(publication_stats.filesystem_ns);
        n = platform::Read(fd.Get(), staging.data(),
                           std::min<std::uint64_t>(remaining, staging.size()),
                           ec);
      }
      if (n == 0 && ec == std::errc::interrupted)
        continue;
      if (n == 0)
        DependencyFailure(&directory == &private_files, payload.file);
      {
        CostTimer timer(publication_stats.checksum_ns);
        crc = DiskChecksum(staging.first(n), crc);
      }
      remaining -= n;
    }
    platform::FileInfo after{};
    int final_status;
    {
      CostTimer timer(publication_stats.filesystem_ns);
      final_status = platform::Stat(fd.Get(), &after, ec) ? 0 : -1;
    }
    if (crc != payload.checksum || final_status < 0 || !Regular(after) ||
        !(st == after))
      DependencyFailure(private_file, payload.file);
    RememberVerified(key, identity);
    {
      // Verified; drop the pages this re-read brought in.
      CostTimer timer(publication_stats.filesystem_ns);
      platform::Advise(fd.Get(), platform::AdviseHint::kDontNeed, 0, 0);
    }
  }
  void Publish(DiskFileId file, const DiskManifest& m,
               std::span<const DiskWriteBuffer> buffers,
               std::span<const DiskWriteSource> sources = {},
               std::span<std::uint8_t> staging = {},
               const DiskStagingAccess* access = nullptr) {
    RequireRecovered();
    // Admission covers vector growth, encoding, validation and dependency map
    // before allocating them. Host buffers remain caller-owned.
    std::uint64_t estimate =
        Add(128, Add(m.input.size(), Multiply(m.tokens.size(), 4)));
    std::size_t references{};
    for (const auto& c : m.components) {
      estimate = Add(estimate, Add(128, Multiply(c.chunks.size(), 32)));
      references = Add(references,
                       c.chunks.size() + bool(c.tail) + bool(c.private_state));
    }
    auto working =
        ledger->Reserve(ResourceCategory::kMetadata,
                        Add(Multiply(estimate, 2), Multiply(references, 256)));
    auto bytes = EncodeManifest(m);
    auto plan = PlanMetadata(bytes);
    auto retained = ledger->Reserve(ResourceCategory::kMetadata, plan.retained);
    DurableEntry candidate;
    candidate.manifest_ = DecodeManifest(bytes);
    candidate.file_ = file;
    candidate.durable_ = false;
    candidate.availability_ = NewAvailability();
    auto pin_reservation =
        ledger->Reserve(ResourceCategory::kMetadata, sizeof(ResourceCharge));
    candidate.pin_ =
        std::make_shared<const ResourceCharge>(pin_reservation.Convert());
    candidate.metadata_ = retained.Convert();
    auto name = DiskFileName(file);
    if (Exists(manifests, name) || Exists(temporary, name))
      Invalid();
    std::map<std::pair<bool, DiskFileId>, DiskPayload> dependencies;
    for (const auto& c : m.components) {
      for (const auto& p : c.chunks)
        dependencies.emplace(std::make_pair(false, p.file), p);
      for (auto p : {c.tail, c.private_state})
        if (p)
          dependencies.emplace(std::make_pair(true, p->file), *p);
    }
    // Reject surplus/duplicate buffers; a reused chunk may have a supplied
    // buffer, but its existing immutable file is still independently verified.
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      auto& b = buffers[i];
      auto it = dependencies.find({b.private_file, b.file});
      if (it == dependencies.end() || b.bytes.size() != it->second.bytes ||
          DiskChecksum(b.bytes) != it->second.checksum)
        Invalid();
      for (std::size_t j = 0; j < i; ++j)
        if (buffers[j].file == b.file &&
            buffers[j].private_file == b.private_file)
          Invalid();
    }
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const auto& source = sources[i];
      auto it = dependencies.find({source.private_file, source.file});
      if (it == dependencies.end() || source.bytes != it->second.bytes ||
          !source.read || !source.alignment ||
          staging.size() < source.alignment || source.bytes % source.alignment)
        Invalid();
      for (std::size_t j = 0; j < i; ++j)
        if (sources[j].file == source.file &&
            sources[j].private_file == source.private_file)
          Invalid();
    }
    // Most writes fit even if every payload were new. Only inspect stored
    // ownership when that upper bound exceeds the remaining capacity.
    auto remaining = stats.managed_bytes > budget_bytes
                         ? 0
                         : budget_bytes - stats.managed_bytes;
    bool check_capacity = bytes.size() > remaining;
    if (!check_capacity) {
      remaining -= bytes.size();
      for (const auto& [key, payload] : dependencies) {
        if (payload.bytes > remaining) {
          check_capacity = true;
          break;
        }
        remaining -= payload.bytes;
      }
    }
    if (check_capacity) {
      std::uint64_t required = bytes.size();
      for (const auto& [key, payload] : dependencies)
        if (!Referenced(key.first ? private_files : chunks, payload.file))
          required = Add(required, payload.bytes);
      // Eviction can retry this publication many times. Refuse it before
      // reading and checksumming the shared prefix on every failed attempt.
      if (stats.managed_bytes > budget_bytes ||
          required > budget_bytes - stats.managed_bytes)
        throw DiskCapacityExhausted();
    }
    std::uint64_t additional = bytes.size();
    for (auto& [key, p] : dependencies) {
      const platform::Directory& dir = key.first ? private_files : chunks;
      if (Referenced(dir, p.file)) {
        // Streamed captures do not hash their immutable sources up front.
        // A queued continuation may reuse a chunk published by an earlier
        // job. Resolve its unspecified checksum from the store's attestation,
        // then verify the file against that attestation before reusing it.
        if (!p.checksum &&
            std::ranges::any_of(sources, [&](const auto& source) {
              return source.private_file == key.first && source.file == p.file;
            })) {
          std::lock_guard guard(index_mutex);
          const auto resolve = [&](const DiskPayload& existing) {
            if (existing.file != p.file)
              return;
            if (existing.bytes != p.bytes)
              Invalid();
            p.checksum = existing.checksum;
          };
          for (const auto& entry : entries)
            for (const auto& component : entry.manifest_.components) {
              if (key.first) {
                for (const auto& payload :
                     {component.tail, component.private_state})
                  if (payload)
                    resolve(*payload);
              } else {
                for (const auto& payload : component.chunks)
                  resolve(payload);
              }
            }
          for (auto& component : candidate.manifest_.components) {
            if (key.first) {
              for (auto* payload : {&component.tail, &component.private_state})
                if (*payload && (*payload)->file == p.file)
                  **payload = p;
            } else {
              for (auto& payload : component.chunks)
                if (payload.file == p.file)
                  payload = p;
            }
          }
        }
        Verify(dir, p, staging, access);
        continue;
      }
      if (Exists(dir, DiskFileName(p.file)) ||
          Exists(temporary, DiskFileName(p.file)))
        Invalid();
      if (p.file == file)
        Invalid();  // temporary namespace is shared
      auto it =
          std::find_if(buffers.begin(), buffers.end(), [&](const auto& b) {
            return b.private_file == key.first && b.file == p.file;
          });
      if (it == buffers.end() &&
          std::none_of(sources.begin(), sources.end(), [&](const auto& s) {
            return s.private_file == key.first && s.file == p.file;
          }))
        Invalid();
      additional = Add(additional, p.bytes);
    }
    // A directory opened above its budget admits nothing until retirement or
    // reclamation brings it back within the budget.
    if (stats.managed_bytes > budget_bytes ||
        additional > budget_bytes - stats.managed_bytes)
      throw DiskCapacityExhausted();
    std::unique_lock index_guard(index_mutex, std::defer_lock);
    {
      CostTimer timer(publication_stats.metadata_lock_ns);
      index_guard.lock();
    }
    for (const auto& entry : entries) {
      if (entry.available_)
        continue;
      for (const auto& component : entry.manifest_.components) {
        for (const auto& chunk : component.chunks)
          if (dependencies.contains({false, chunk.file}))
            throw DiskDependencyError(false, chunk.file);
        for (const auto& payload : {component.tail, component.private_state})
          if (payload && dependencies.contains({true, payload->file}))
            throw DiskDependencyError(true, payload->file);
      }
    }
    if (entries.size() == entries.capacity()) {
      auto reserve =
          ledger->Reserve(ResourceCategory::kMetadata,
                          Multiply(entries.size() + 1, sizeof(DurableEntry)));
      std::vector<DurableEntry> replacement;
      replacement.reserve(entries.size() + 1);
      auto charge = reserve.Convert();
      for (auto& e : entries)
        replacement.push_back(std::move(e));
      entries.swap(replacement);
      // Release the old allocation while its capacity charge is still live.
      std::vector<DurableEntry>().swap(replacement);
      entry_capacity = std::move(charge);
    }
    if (next_epoch == UINT64_MAX)
      throw std::overflow_error("disk publication epochs exhausted");
    candidate.epoch_ = next_epoch++;
    entries.push_back(std::move(candidate));
    bool manifest_attempted{};
    try {
      ValidateReferences(false);
      index_guard.unlock();
      // Candidate is invisible to pinned snapshots until committed. Reuse from
      // filesystem existence after validating claims, never from this
      // candidate.
      for (auto& [key, p] : dependencies) {
        const platform::Directory& dir = key.first ? private_files : chunks;
        auto payload_name = DiskFileName(p.file);
        if (Exists(dir, payload_name))
          continue;
        auto it =
            std::find_if(buffers.begin(), buffers.end(), [&](const auto& b) {
              return b.private_file == key.first && b.file == p.file;
            });
        auto fd = Create(payload_name);
        if (it != buffers.end()) {
          WriteFile(fd.Get(), it->bytes);
        } else {
          auto source =
              std::find_if(sources.begin(), sources.end(), [&](const auto& s) {
                return s.private_file == key.first && s.file == p.file;
              });
          std::uint64_t offset{}, crc{};
          const auto piece_bytes =
              staging.size() / source->alignment * source->alignment;
          while (offset < p.bytes) {
            Finally release{[&] {
              if (access)
                access->release();
            }};
            if (access)
              access->acquire();
            auto piece = staging.first(
                std::min<std::uint64_t>(piece_bytes, p.bytes - offset));
            source->read(offset, piece);
            {
              CostTimer timer(publication_stats.checksum_ns);
              crc = DiskChecksum(piece, crc);
            }
            WriteFile(fd.Get(), piece);
            offset += piece.size();
          }
          if (p.checksum && p.checksum != crc)
            Invalid();
          p.checksum = crc;
          for (auto& c : entries.back().manifest_.components) {
            if (!key.first) {
              for (auto& chunk : c.chunks)
                if (chunk.file == p.file)
                  chunk.checksum = crc;
            } else {
              for (auto* private_payload : {&c.tail, &c.private_state})
                if (*private_payload && (*private_payload)->file == p.file)
                  (*private_payload)->checksum = crc;
            }
          }
        }
        DISK_STEP(kPayloadWritten);
        Sync(fd.Get());
        {
          // The payload is durable; drop its clean pages so multi-GB snapshots
          // don't accumulate in the page cache.
          CostTimer timer(publication_stats.filesystem_ns);
          platform::Advise(fd.Get(), platform::AdviseHint::kDontNeed, 0, 0);
        }
        DISK_STEP(kPayloadSynced);
        Rename(dir, payload_name);
        DISK_STEP(kPayloadRenamed);
      }
      Sync(temporary);
      DISK_STEP(kTemporaryDirectorySynced);
      Sync(chunks);
      DISK_STEP(kChunkDirectorySynced);
      Sync(private_files);
      DISK_STEP(kDependenciesSynced);
      index_guard.lock();
      if (!entries.back().available_)
        throw std::runtime_error("checkpoint dependency quarantined");
      index_guard.unlock();
      if (!sources.empty()) {
        // Free the provisional encoding before allocating the final one.
        std::vector<std::uint8_t>().swap(bytes);
        bytes = EncodeManifest(entries.back().manifest_);
      }
      auto fd = Create(name);
      WriteFile(fd.Get(), bytes);
      DISK_STEP(kManifestWritten);
      Sync(fd.Get());
      DISK_STEP(kManifestSynced);
      manifest_attempted = true;
      Rename(manifests, name);
      DISK_STEP(kManifestRenamed);
      Sync(manifests);
      DISK_STEP(kManifestDirectorySynced);
      Sync(temporary);
      DISK_STEP(kFinalTemporaryDirectorySynced);
      index_guard.lock();
      if (!entries.back().available_)
        throw std::runtime_error("checkpoint dependency quarantined");
      entries.back().payload_verified_ = true;
      entries.back().durable_ = true;
      DISK_STEP(kIndexed);
    } catch (...) {
      if (!index_guard.owns_lock())
        index_guard.lock();
      if (manifest_attempted)
        recovery_required = true;
      entries.pop_back();
      // Conservatively retain every byte as an orphan, including an uncertain
      // manifest rename. Recovery removes manifests durably before payloads.
      throw;
    }
  }
  static platform::Directory OpenRoot(const std::filesystem::path& path) {
    // Do not collapse '..': an earlier component may be a symlink and the
    // kernel's configured-directory resolution must remain authoritative.
    auto root_path = path;
    while (root_path.has_relative_path() &&
           (root_path.filename().empty() || root_path.filename() == "."))
      root_path = root_path.parent_path();
    if (root_path.empty())
      root_path = ".";
    std::filesystem::create_directories(root_path);
    platform::Directory directory;
    std::error_code ec;
    if (!directory.Open(root_path, ec))
      IoError("open cache directory " + path.string(), ec);
    return directory;
  }
  static Fd OpenLock(const platform::Directory& root,
                     const std::filesystem::path& path) {
    std::error_code ec;
    Fd lock(
        platform::OpenAt(root, "LOCK", platform::FileMode::kOpenOrCreate, ec));
    platform::FileInfo st{};
    if (!platform::FileValid(lock.Get()) ||
        !platform::Stat(lock.Get(), &st, ec) || !Regular(st))
      throw std::runtime_error("invalid cache ownership lock in " +
                               path.string());
    if (!platform::TryLockExclusive(lock.Get(), ec))
      throw std::runtime_error(
          "cache directory ownership unavailable (another process may own "
          "it): " +
          path.string());
    return lock;
  }
  void CleanupLegacy() {
    Scan(root, [&](const std::string& name, const platform::FileInfo& st) {
      if (!Regular(st))
        return;
      bool managed = name.starts_with(".tmp-");
      if (!managed && name.ends_with(".kvc")) {
        std::error_code ec;
        const Fd file(
            platform::OpenAt(root, name, platform::FileMode::kRead, ec));
        platform::FileInfo current{};
        if (!platform::FileValid(file.Get()) ||
            !platform::Stat(file.Get(), &current, ec) || !Regular(current))
          return;
        std::array<char, 4> magic{};
        std::size_t offset{};
        while (offset < magic.size()) {
          std::error_code read_ec;
          const auto n = platform::Read(file.Get(), magic.data() + offset,
                                        magic.size() - offset, read_ec);
          if (n == 0 && read_ec == std::errc::interrupted)
            continue;
          if (n == 0 && read_ec)
            IoError("read legacy cache magic", read_ec);
          if (!n)
            break;
          offset += n;
          stats.legacy_probe_bytes_read += n;
        }
        managed = offset == magic.size() &&
                  magic == std::array<char, 4>{'G', 'U', 'F', 'O'};
      }
      if (managed) {
        std::error_code ec;
        if (!platform::Unlink(root, name, ec))
          IoError("remove legacy cache file", ec);
        stats.removed_legacy_bytes = Add(stats.removed_legacy_bytes, st.size);
      }
    });
  }
  DurableEntry ReadManifest(const std::string& name,
                            const platform::FileInfo& found) {
    if (!Regular(found) || found.size < 20 || found.size > kMaxManifestBytes)
      Invalid();
    std::error_code ec;
    const Fd file(
        platform::OpenAt(manifests, name, platform::FileMode::kRead, ec));
    platform::FileInfo st{};
    if (!platform::FileValid(file.Get()) ||
        !platform::Stat(file.Get(), &st, ec) || !Regular(st) ||
        st.size != found.size)
      Invalid();
    auto read_reservation =
        ledger->Reserve(ResourceCategory::kMetadata, st.size);
    std::vector<std::uint8_t> bytes(st.size);
    std::size_t offset{};
    while (offset < bytes.size()) {
      std::error_code read_ec;
      const auto n = platform::Read(file.Get(), bytes.data() + offset,
                                    bytes.size() - offset, read_ec);
      if (n == 0 && read_ec == std::errc::interrupted)
        continue;
      if (n == 0)
        Invalid();
      offset += n;
      stats.manifest_bytes_read += n;
    }
    const auto plan = PlanMetadata(bytes);
    auto retained = ledger->Reserve(ResourceCategory::kMetadata, plan.retained);
    auto validation =
        plan.references
            ? ledger->Reserve(ResourceCategory::kMetadata,
                              Multiply(plan.references,
                                       sizeof(DiskFileId) + 4 * sizeof(void*)))
            : ResourceReservation{};
    DurableEntry entry;
    entry.manifest_ = DecodeManifest(bytes);
    entry.availability_ = NewAvailability();
    // Decoding explicitly reserves exact capacities with the pinned library.
    std::size_t actual =
        entry.manifest_.input.capacity() +
        entry.manifest_.tokens.capacity() * sizeof(Token) +
        entry.manifest_.components.capacity() * sizeof(DiskComponent);
    for (const auto& c : entry.manifest_.components)
      actual = Add(actual, c.chunks.capacity() * sizeof(DiskPayload));
    if (actual != plan.retained)
      throw std::logic_error("unexpected disk manifest capacity");
    auto pin_reservation =
        ledger->Reserve(ResourceCategory::kMetadata, sizeof(ResourceCharge));
    entry.pin_ =
        std::make_shared<const ResourceCharge>(pin_reservation.Convert());
    if (retained)
      entry.metadata_ = retained.Convert();
    return entry;
  }
  bool Check(const platform::Directory& directory, const DiskPayload& p) {
    ++stats.dependency_stats;
    platform::FileInfo st{};
    std::error_code ec;
    return platform::StatAt(directory, DiskFileName(p.file), false, &st, ec) &&
           Regular(st) && st.size == p.bytes;
  }
};
DurableEntry& DurableEntry::operator=(DurableEntry&& other) noexcept {
  if (this != &other) {
    // Free the replaced vectors while their previous accounting is live.
    auto previous = std::move(metadata_);
    manifest_ = std::move(other.manifest_);
    file_ = other.file_;
    durable_ = other.durable_;
    payload_verified_ = other.payload_verified_;
    available_ = other.available_;
    availability_ = std::move(other.availability_);
    epoch_ = other.epoch_;
    metadata_ = std::move(other.metadata_);
    pin_ = std::move(other.pin_);
  }
  return *this;
}
DiskStore::DiskStore(ResourceLedger& ledger,
                     const std::filesystem::path& directory,
                     std::uint64_t budget)
    : impl_(std::make_unique<Impl>(
          ledger, directory, budget,
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)))) {}
DiskStore::~DiskStore() = default;
std::span<const DurableEntry> DiskStore::Entries() const {
  return impl_->entries;
}
DiskStartupStats DiskStore::Stats() const {
  std::lock_guard guard(impl_->stats_mutex);
  return impl_->stats;
}
DiskPublicationStats DiskStore::PublicationStats() const {
  std::lock_guard guard(impl_->mutex);
  return impl_->publication_stats;
}
void DiskStore::Publish(DiskFileId file, const DiskManifest& manifest,
                        std::span<const DiskWriteBuffer> buffers) {
  std::lock_guard guard(impl_->mutex);
  impl_->Publish(file, manifest, buffers);
}
DiskPublicationStats DiskStore::PublishStream(
    DiskFileId file, const DiskManifest& manifest,
    std::span<const DiskWriteSource> sources, std::span<std::uint8_t> staging,
    DiskPublicationStats* observation, const DiskStagingAccess* access,
    std::stop_token stop) {
  if (staging.empty())
    throw std::invalid_argument("empty streamed publication staging");
  if (access && (!access->acquire || !access->release))
    throw std::invalid_argument("incomplete disk staging lease");
  auto begin = std::chrono::steady_clock::now();
  std::unique_lock guard(impl_->mutex, std::defer_lock);
  Finally wait_observation{[&] {
    if (!guard.owns_lock() && observation)
      observation->io_lock_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - begin)
              .count();
  }};
  do {
    if (stop.stop_requested())
      throw std::runtime_error("cache publication cancelled");
  } while (!guard.try_lock_for(std::chrono::milliseconds(2)));
  const auto before = impl_->publication_stats;
  Finally observe{[&] {
    if (observation)
      *observation = Difference(impl_->publication_stats, before);
  }};
  impl_->publication_stats.io_lock_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count();
  if (stop.stop_requested())
    throw std::runtime_error("cache publication cancelled");
  impl_->Publish(file, manifest, {}, sources, staging, access);
  return Difference(impl_->publication_stats, before);
}
std::optional<DiskSnapshot> DiskStore::Open(CheckpointId id) const {
  std::lock_guard guard(impl_->index_mutex);
  for (const auto& entry : impl_->entries) {
    if (!entry.durable_ || !entry.available_ ||
        entry.manifest_.checkpoint != id)
      continue;
    auto reservation = impl_->ledger->Reserve(ResourceCategory::kMetadata,
                                              entry.metadata_.Info().bytes);
    DiskSnapshot snapshot;
    snapshot.manifest_ = entry.manifest_;
    snapshot.metadata_ = reservation.Convert();
    snapshot.pin_.token_ = entry.pin_;
    snapshot.pin_.epoch_ = entry.epoch_;
    snapshot.store_ = impl_.get();
    return snapshot;
  }
  return {};
}
void DiskStore::PrefetchPayload(const DiskSnapshot& snapshot, bool private_file,
                                const DiskPayload& payload,
                                std::uint64_t bytes) const {
  if (snapshot.store_ != impl_.get() || !snapshot.pin_ ||
      !Contains(snapshot.manifest_, private_file, payload))
    throw std::invalid_argument("invalid disk prefetch dependency");
  bytes = std::min(bytes, payload.bytes);
  if (!bytes)
    return;
  const auto& directory = private_file ? impl_->private_files : impl_->chunks;
  std::error_code ec;
  Fd file(platform::OpenAt(directory, DiskFileName(payload.file),
                           platform::FileMode::kRead, ec));
  platform::FileInfo status{};
  if (platform::FileValid(file.Get()) &&
      platform::Stat(file.Get(), &status, ec) && Regular(status))
    platform::Advise(file.Get(), platform::AdviseHint::kWillNeed, 0, bytes);
}
DiskReadStats DiskStore::ReadPayload(
    const DiskSnapshot& snapshot, bool private_file, const DiskPayload& payload,
    std::span<std::uint8_t> staging,
    const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>&
        consume,
    DiskReadStats* observation) {
  if (snapshot.store_ != impl_.get() || !snapshot.pin_ || staging.empty() ||
      !consume)
    throw std::invalid_argument("invalid disk read lease or staging");
  if (!Contains(snapshot.manifest_, private_file, payload))
    throw std::invalid_argument("payload outside disk snapshot");
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](auto start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                start)
        .count();
  };
  DiskReadStats stats;
  Finally observe{[&] {
    if (observation)
      *observation = stats;
  }};
  auto begin = Clock::now();
  const auto& directory = private_file ? impl_->private_files : impl_->chunks;
  std::error_code ec;
  Fd file(platform::OpenAt(directory, DiskFileName(payload.file),
                           platform::FileMode::kRead, ec));
  platform::FileInfo before{}, after{};
  if (!platform::FileValid(file.Get()) ||
      !platform::Stat(file.Get(), &before, ec) || !Regular(before) ||
      before.size != payload.bytes)
    impl_->DependencyFailure(private_file, payload.file);
  const Impl::FileIdentity identity{before, payload.checksum};
  stats.filesystem_ns += elapsed(begin);
  const auto key = std::make_pair(private_file, payload.file);
  stats.verification_cached = impl_->VerificationCached(key, identity);
  std::uint64_t offset{}, checksum{};
  while (offset < payload.bytes) {
    auto piece = staging.first(
        std::min<std::uint64_t>(staging.size(), payload.bytes - offset));
    begin = Clock::now();
    std::size_t filled{};
    while (filled < piece.size()) {
      const auto n = platform::Read(file.Get(), piece.data() + filled,
                                    piece.size() - filled, ec);
      if (n == 0 && ec == std::errc::interrupted)
        continue;
      if (n == 0)
        impl_->DependencyFailure(private_file, payload.file);
      filled += n;
    }
    stats.filesystem_ns += elapsed(begin);
    if (!stats.verification_cached) {
      begin = Clock::now();
      checksum = DiskChecksum(piece, checksum);
      stats.checksum_ns += elapsed(begin);
    }
    stats.bytes += piece.size();
    consume(offset, piece);
    offset += piece.size();
  }
  begin = Clock::now();
  if (!platform::Stat(file.Get(), &after, ec) || !Regular(after) ||
      identity.info != after ||
      (!stats.verification_cached && checksum != payload.checksum))
    impl_->DependencyFailure(private_file, payload.file);
  // The consumer holds the bytes now; drop the page-cache copy.
  platform::Advise(file.Get(), platform::AdviseHint::kDontNeed, 0, 0);
  stats.filesystem_ns += elapsed(begin);
  if (!stats.verification_cached)
    impl_->RememberVerified(key, identity);
  return stats;
}
DiskReadPin DiskStore::Pin(CheckpointId id, std::uint64_t epoch) const {
  std::lock_guard guard(impl_->index_mutex);
  DiskReadPin pin;
  for (const auto& entry : impl_->entries)
    if (entry.durable_ && entry.available_ &&
        entry.manifest_.checkpoint == id && (!epoch || epoch == entry.epoch_)) {
      pin.token_ = entry.pin_;
      pin.epoch_ = entry.epoch_;
      break;
    }
  return pin;
}
DiskRetireResult DiskStore::Impl::Retire(CheckpointId id, std::uint64_t epoch) {
  RequireRecovered();
  std::unique_lock index_guard(index_mutex);
  auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) {
    return e.manifest_.checkpoint == id;
  });
  if (it == entries.end() || (epoch && it->epoch_ != epoch))
    return DiskRetireResult::kMissing;
  if (it->pin_.use_count() > 1)
    return DiskRetireResult::kPinned;
  // Remove the index claim before attempting unlink: an I/O error can report
  // an uncertain outcome even when the directory entry has disappeared. Keep
  // dependencies untouched until the manifest-directory barrier succeeds.
  auto retired = std::move(*it);
  entries.erase(it);
  index_guard.unlock();
  try {
    Remove(manifests, DiskFileName(retired.file_));
    Sync(manifests);
    for (const auto& c : retired.manifest_.components) {
      for (const auto& p : c.chunks)
        if (!Referenced(chunks, p.file))
          Remove(chunks, DiskFileName(p.file));
      for (auto p : {c.tail, c.private_state})
        if (p && !Referenced(private_files, p->file))
          Remove(private_files, DiskFileName(p->file));
    }
    Sync(chunks);
    Sync(private_files);
    return DiskRetireResult::kRetired;
  } catch (...) {
    // Both checkpoint identities and freed payload capacity remain uncertain
    // until every deletion has crossed its directory durability barrier.
    recovery_required = true;
    throw;
  }
}

bool DiskStore::Retire(CheckpointId id) {
  std::lock_guard guard(impl_->mutex);
  return impl_->Retire(id, 0) == DiskRetireResult::kRetired;
}
DiskRetireResult DiskStore::TryRetire(CheckpointId id, std::uint64_t epoch) {
  std::unique_lock guard(impl_->mutex, std::try_to_lock);
  if (!guard.owns_lock())
    return DiskRetireResult::kBusy;
  return impl_->Retire(id, epoch);
}
void DiskStore::Invalidate(CheckpointId id, std::uint64_t epoch) {
  std::lock_guard guard(impl_->index_mutex);
  for (auto& entry : impl_->entries)
    if (entry.manifest_.checkpoint == id && entry.epoch_ == epoch) {
      entry.available_ = false;
      entry.availability_->quarantined.store(true);
    }
}
void DiskStore::InvalidateDependency(bool private_file, DiskFileId file) {
  impl_->Quarantine(private_file, file);
}
std::shared_ptr<const DiskDescription> DiskStore::Describe(CheckpointId id) {
  std::lock_guard guard(impl_->index_mutex);
  for (const auto& entry : impl_->entries) {
    if (!entry.durable_ || !entry.available_ ||
        entry.manifest_.checkpoint != id)
      continue;
    auto reservation = impl_->ledger->Reserve(
        ResourceCategory::kMetadata,
        Add(entry.metadata_.Info().bytes, sizeof(DiskDescription) + 64));
    auto description = std::make_shared<DiskDescription>();
    description->manifest_ = entry.manifest_;
    description->store_ = this;
    description->epoch_ = entry.epoch_;
    description->availability_ = entry.availability_;
    description->metadata_ = reservation.Convert();
    return description;
  }
  return {};
}
bool DiskDescription::Quarantined() const {
  return availability_->quarantined.load();
}
DiskReadPin DiskDescription::Pin() const {
  return store_->Pin(manifest_.checkpoint, epoch_);
}

void DiskStore::ReclaimOrphans() {
  // Scheduling/joining are caller-ordered, independent of worker store locking.
  WaitForReclamation();
  impl_->reclamation = std::async(std::launch::async, [state = impl_.get()] {
    std::lock_guard guard(state->mutex);
    state->Reclaim();
  });
}
void DiskStore::WaitForReclamation() {
  if (impl_->reclamation.valid())
    impl_->reclamation.get();
}
#ifdef GUFO_CACHE_TESTING
void DiskStore::SetCrashHook(std::function<void(DiskPublicationStep)> hook) {
  std::lock_guard guard(impl_->mutex);
  impl_->crash_hook = std::move(hook);
}
#endif
}  // namespace gufo::cache
