#include "src/cache/disk.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace gufo::cache {
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
    if (d.kind == ComponentKind::kAppendRows) {
      if (!d.row_bytes || !d.rows_per_chunk || d.state_bytes || private_state ||
          c.chunks.size() != c.position.valid_rows / d.rows_per_chunk)
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
      if (d.row_bytes || d.rows_per_chunk || !d.state_bytes ||
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
    r.Blob(41);
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
  explicit Fd(int value = -1) : value_(value) {}
  ~Fd() {
    if (value_ >= 0)
      close(value_);
  }
  Fd(Fd&& o) noexcept : value_(o.value_) { o.value_ = -1; }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd& operator=(Fd&&) = delete;
  int Get() const { return value_; }
  void Swap(Fd& other) { std::swap(value_, other.value_); }

private:
  int value_;
};
[[noreturn]] void IoError(const std::string& message) {
  throw std::runtime_error(message + ": " + std::strerror(errno));
}
void SyncFd(int fd) {
  int result;
  do {
    result = fsync(fd);
  } while (result < 0 && errno == EINTR);
  if (result < 0)
    IoError("sync cache file/directory");
}
Fd DirectoryAt(int parent, const char* name) {
  if (mkdirat(parent, name, 0700) < 0 && errno != EEXIST)
    IoError("create cache directory");
  const int fd =
      openat(parent, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0)
    IoError(std::string("open cache directory ") + name);
  return Fd(fd);
}
template<typename F>
void Scan(int fd, F visit) {
  const int copy = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (copy < 0)
    IoError("open cache scan");
  auto close_directory = [](DIR* d) { closedir(d); };
  const std::unique_ptr<DIR, decltype(close_directory)> dir(fdopendir(copy),
                                                            close_directory);
  if (!dir) {
    close(copy);
    IoError("scan cache directory");
  }
  for (;;) {
    errno = 0;
    auto* entry = readdir(dir.get());
    if (!entry) {
      if (errno)
        IoError("read cache directory");
      break;
    }
    const std::string name(entry->d_name);
    if (name == "." || name == "..")
      continue;
    struct stat st{};
    if (fstatat(fd, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) < 0)
      IoError("stat cache file");
    if (S_ISREG(st.st_mode) && st.st_size >= 0)
      visit(name, st);
  }
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
bool Regular(const struct stat& st) {
  return S_ISREG(st.st_mode) && st.st_size >= 0 && st.st_nlink == 1;
}
}  // namespace
std::uint64_t DiskChecksum(std::span<const std::uint8_t> bytes,
                           std::uint64_t crc) {
  static const auto table = [] {
    std::array<std::uint64_t, 256> t{};
    for (unsigned i = 0; i < 256; ++i) {
      auto n = std::uint64_t(i) << 56;
      for (unsigned j = 0; j < 8; ++j)
        n = (n << 1) ^ ((n >> 63) ? 0x42f0e1eba9ea3693ULL : 0);
      t[i] = n;
    }
    return t;
  }();
  for (auto b : bytes)
    crc = (crc << 8) ^ table[(crc >> 56) ^ b];
  return crc;
}
std::vector<std::uint8_t> EncodeManifest(const DiskManifest& m) {
  Validate(m);
  Writer w;
  w.Blob(kMagic);
  w.Number(kDiskFormatVersion, 4);
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
  if (!std::equal(magic.begin(), magic.end(), kMagic.begin()) ||
      r.Number(4) != kDiskFormatVersion)
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
  if (!r.bytes.empty())
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
  Fd root, lock, version, manifests, chunks, private_files, temporary;
  DiskStartupStats stats;
  std::vector<DurableEntry> entries;
  std::uint64_t budget_bytes;
  DiskPublicationStats publication_stats;
  bool recovery_required{};
  mutable std::timed_mutex mutex;
  mutable std::mutex index_mutex;
  mutable std::mutex stats_mutex;
  struct FileIdentity {
    dev_t device;
    ino_t inode;
    off_t size;
    timespec modified, changed;
    std::uint64_t checksum;
    bool operator==(const FileIdentity& b) const {
      return device == b.device && inode == b.inode && size == b.size &&
             modified.tv_sec == b.modified.tv_sec &&
             modified.tv_nsec == b.modified.tv_nsec &&
             changed.tv_sec == b.changed.tv_sec &&
             changed.tv_nsec == b.changed.tv_nsec && checksum == b.checksum;
    }
  };
  struct VerifiedFile {
    ResourceCharge charge;
    FileIdentity identity;
  };
  std::mutex verification_mutex;
  std::map<std::pair<bool, DiskFileId>, VerifiedFile> verified;
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
        lock(OpenLock(root.Get(), path)),
        version(DirectoryAt(root.Get(), "v2")),
        manifests(DirectoryAt(version.Get(), "manifests")),
        chunks(DirectoryAt(version.Get(), "chunks")),
        private_files(DirectoryAt(version.Get(), "private")),
        temporary(DirectoryAt(version.Get(), "tmp")),
        budget_bytes(budget) {
    // Persist the layout, including a newly created root and ancestors, before
    // any checkpoint can rely on these directory entries.
    for (int fd : {version.Get(), root.Get()})
      SyncFd(fd);
    Fd ancestor(openat(root.Get(), "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    for (;;) {
      if (ancestor.Get() < 0)
        IoError("open cache parent");
      SyncFd(ancestor.Get());
      Fd parent(
          openat(ancestor.Get(), "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
      struct stat a{}, b{};
      if (parent.Get() < 0 || fstat(ancestor.Get(), &a) < 0 ||
          fstat(parent.Get(), &b) < 0)
        IoError("stat cache parent");
      if (a.st_dev == b.st_dev && a.st_ino == b.st_ino)
        break;
      ancestor.Swap(parent);
    }
    CleanupLegacy();
    std::size_t manifest_count{};
    for (const int directory :
         {manifests.Get(), chunks.Get(), private_files.Get()})
      Scan(directory, [&](const std::string& name, const struct stat& st) {
        if (ParseName(name)) {
          stats.managed_bytes = Add(stats.managed_bytes, st.st_size);
          if (directory == manifests.Get())
            ++manifest_count;
        }
      });
    Scan(temporary.Get(), [&](const std::string& name, const struct stat& st) {
      if (ParseName(name))
        stats.managed_bytes = Add(stats.managed_bytes, st.st_size);
    });
    if (stats.managed_bytes > budget)
      throw std::runtime_error("cache directory exceeds disk budget: " +
                               path.string());
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
    Scan(manifests.Get(), [&](const std::string& name, const struct stat& st) {
      auto id = ParseName(name);
      if (!id)
        return;
      try {
        auto entry = ReadManifest(name, st);
        auto& m = entry.manifest_;
        bool available = true;
        for (const auto& c : m.components) {
          for (const auto& p : c.chunks)
            available = Check(chunks.Get(), p) && available;
          for (auto p : {c.tail, c.private_state})
            if (p)
              available = Check(private_files.Get(), *p) && available;
        }
        if (!available)
          Invalid();
        entry.file_ = *id;
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
    for (int fd :
         {manifests.Get(), chunks.Get(), private_files.Get(), temporary.Get()})
      SyncFd(fd);
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
            m.lineage.value,  d.id.value, d.layout_version, d.row_bytes,
            d.rows_per_chunk, first,      p.bytes,          p.checksum};
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
        Rows first{};
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
  void Sync(int fd) {
    auto begin = std::chrono::steady_clock::now();
    int result;
    do {
      result = fsync(fd);
    } while (result < 0 && errno == EINTR);
    ++publication_stats.fsync_calls;
    publication_stats.fsync_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin)
            .count();
    if (result < 0)
      IoError("sync cache file/directory");
  }
  void Remove(int directory, const std::string& name) {
    struct stat st{};
    if (fstatat(directory, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) < 0) {
      if (errno == ENOENT)
        return;
      IoError("stat cache removal");
    }
    if (!Regular(st))
      return;
    if (unlinkat(directory, name.c_str(), 0) < 0)
      IoError("remove cache file");
    {
      std::lock_guard guard(stats_mutex);
      stats.managed_bytes -= st.st_size;
    }
  }
  bool Referenced(int directory, DiskFileId id) const {
    for (const auto& entry : entries) {
      if (directory == manifests.Get() && entry.file_ == id)
        return true;
      for (const auto& c : entry.manifest_.components) {
        if (directory == chunks.Get())
          for (const auto& p : c.chunks)
            if (p.file == id)
              return true;
        if (directory == private_files.Get())
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
    Scan(manifests.Get(), [&](const auto& name, const auto&) {
      auto id = ParseName(name);
      if (id && !Referenced(manifests.Get(), *id))
        Remove(manifests.Get(), name);
    });
    Sync(manifests.Get());
    for (int directory : {chunks.Get(), private_files.Get(), temporary.Get()}) {
      Scan(directory, [&](const auto& name, const auto&) {
        auto id = ParseName(name);
        if (id && !Referenced(directory, *id))
          Remove(directory, name);
      });
      Sync(directory);
    }
    // A failed unlink syscall can leave its outcome uncertain. Reconcile the
    // conservative charge with the actual namespace after successful cleanup.
    std::uint64_t managed{};
    for (int directory :
         {manifests.Get(), chunks.Get(), private_files.Get(), temporary.Get()})
      Scan(directory, [&](const auto& name, const auto& st) {
        if (ParseName(name))
          managed = Add(managed, st.st_size);
      });
    {
      std::lock_guard guard(stats_mutex);
      stats.managed_bytes = managed;
    }
    recovery_required = false;
  }
  void Rename(int target, const std::string& name) {
    CostTimer timer(publication_stats.filesystem_ns);
    if (syscall(SYS_renameat2, temporary.Get(), name.c_str(), target,
                name.c_str(), RENAME_NOREPLACE) < 0)
      IoError("publish cache file");
  }
  void WriteFile(int fd, std::span<const std::uint8_t> bytes) {
    while (!bytes.empty()) {
      ssize_t n;
      {
        CostTimer timer(publication_stats.filesystem_ns);
        n = write(fd, bytes.data(), bytes.size());
      }
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        IoError("write cache file");
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
    Fd fd(openat(temporary.Get(), name.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (fd.Get() < 0)
      IoError("create temporary cache file");
    return fd;
  }
  bool Exists(int directory, const std::string& name) {
    CostTimer timer(publication_stats.filesystem_ns);
    struct stat st{};
    if (fstatat(directory, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0)
      return true;
    if (errno != ENOENT)
      IoError("stat publication destination");
    return false;
  }
  void Verify(int directory, const DiskPayload& payload,
              std::span<std::uint8_t> staging = {},
              const DiskStagingAccess* access = nullptr) {
    int descriptor;
    struct stat st{};
    {
      CostTimer timer(publication_stats.filesystem_ns);
      descriptor = openat(directory, DiskFileName(payload.file).c_str(),
                          O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    }
    Fd fd(descriptor);
    int status;
    {
      CostTimer timer(publication_stats.filesystem_ns);
      status = fstat(fd.Get(), &st);
    }
    if (fd.Get() < 0 || status < 0 || !Regular(st) ||
        std::uint64_t(st.st_size) != payload.bytes)
      Invalid();
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
      ssize_t n;
      {
        CostTimer timer(publication_stats.filesystem_ns);
        n = read(fd.Get(), staging.data(),
                 std::min<std::uint64_t>(remaining, staging.size()));
      }
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        Invalid();
      {
        CostTimer timer(publication_stats.checksum_ns);
        crc = DiskChecksum(staging.first(n), crc);
      }
      remaining -= n;
    }
    if (crc != payload.checksum)
      Invalid();
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
    auto pin_reservation =
        ledger->Reserve(ResourceCategory::kMetadata, sizeof(ResourceCharge));
    candidate.pin_ =
        std::make_shared<const ResourceCharge>(pin_reservation.Convert());
    candidate.metadata_ = retained.Convert();
    auto name = DiskFileName(file);
    if (Exists(manifests.Get(), name) || Exists(temporary.Get(), name))
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
    std::uint64_t additional = bytes.size();
    for (const auto& [key, p] : dependencies) {
      int dir = key.first ? private_files.Get() : chunks.Get();
      if (Referenced(dir, p.file)) {
        Verify(dir, p, staging, access);
        continue;
      }
      if (Exists(dir, DiskFileName(p.file)) ||
          Exists(temporary.Get(), DiskFileName(p.file)))
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
    if (additional > budget_bytes - stats.managed_bytes)
      throw ResourceExhausted();
    std::unique_lock index_guard(index_mutex, std::defer_lock);
    {
      CostTimer timer(publication_stats.metadata_lock_ns);
      index_guard.lock();
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
    entries.push_back(std::move(candidate));
    bool manifest_attempted{};
    try {
      ValidateReferences(false);
      index_guard.unlock();
      // Candidate is invisible to pinned snapshots until committed. Reuse from
      // filesystem existence after validating claims, never from this
      // candidate.
      for (auto& [key, p] : dependencies) {
        int dir = key.first ? private_files.Get() : chunks.Get();
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
        DISK_STEP(kPayloadSynced);
        Rename(dir, payload_name);
        DISK_STEP(kPayloadRenamed);
      }
      Sync(temporary.Get());
      DISK_STEP(kTemporaryDirectorySynced);
      Sync(chunks.Get());
      DISK_STEP(kChunkDirectorySynced);
      Sync(private_files.Get());
      DISK_STEP(kDependenciesSynced);
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
      Rename(manifests.Get(), name);
      DISK_STEP(kManifestRenamed);
      Sync(manifests.Get());
      DISK_STEP(kManifestDirectorySynced);
      Sync(temporary.Get());
      DISK_STEP(kFinalTemporaryDirectorySynced);
      index_guard.lock();
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
  static Fd OpenRoot(const std::filesystem::path& path) {
    // Do not collapse '..': an earlier component may be a symlink and the
    // kernel's configured-directory resolution must remain authoritative.
    auto root_path = path;
    while (root_path.has_relative_path() &&
           (root_path.filename().empty() || root_path.filename() == "."))
      root_path = root_path.parent_path();
    if (root_path.empty())
      root_path = ".";
    std::filesystem::create_directories(root_path);
    const int fd = open(root_path.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
      IoError("open cache directory " + path.string());
    return Fd(fd);
  }
  static Fd OpenLock(int root, const std::filesystem::path& path) {
    Fd lock(openat(root, "LOCK",
                   O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK,
                   0600));
    struct stat st{};
    if (lock.Get() < 0 || fstat(lock.Get(), &st) < 0 || !Regular(st))
      throw std::runtime_error("invalid cache ownership lock in " +
                               path.string());
    if (flock(lock.Get(), LOCK_EX | LOCK_NB) < 0)
      throw std::runtime_error(
          "cache directory ownership unavailable (another process may own "
          "it): " +
          path.string());
    return lock;
  }
  void CleanupLegacy() {
    Scan(root.Get(), [&](const std::string& name, const struct stat& st) {
      if (!Regular(st))
        return;
      bool managed = name.starts_with(".tmp-");
      if (!managed && name.ends_with(".kvc")) {
        const Fd file(openat(root.Get(), name.c_str(),
                             O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
        struct stat current{};
        if (file.Get() < 0 || fstat(file.Get(), &current) < 0 ||
            !Regular(current))
          return;
        std::array<char, 4> magic{};
        std::size_t offset{};
        while (offset < magic.size()) {
          auto n =
              read(file.Get(), magic.data() + offset, magic.size() - offset);
          if (n < 0 && errno == EINTR)
            continue;
          if (n < 0)
            IoError("read legacy cache magic");
          if (!n)
            break;
          offset += n;
          stats.legacy_probe_bytes_read += n;
        }
        managed = offset == magic.size() &&
                  magic == std::array<char, 4>{'G', 'U', 'F', 'O'};
      }
      if (managed) {
        if (unlinkat(root.Get(), name.c_str(), 0) < 0)
          IoError("remove legacy cache file");
        stats.removed_legacy_bytes =
            Add(stats.removed_legacy_bytes, st.st_size);
      }
    });
  }
  DurableEntry ReadManifest(const std::string& name, const struct stat& found) {
    if (!Regular(found) || found.st_size < 20 ||
        std::uint64_t(found.st_size) > kMaxManifestBytes)
      Invalid();
    const Fd file(openat(manifests.Get(), name.c_str(),
                         O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat st{};
    if (file.Get() < 0 || fstat(file.Get(), &st) < 0 || !Regular(st) ||
        st.st_size != found.st_size)
      Invalid();
    auto read_reservation =
        ledger->Reserve(ResourceCategory::kMetadata, st.st_size);
    std::vector<std::uint8_t> bytes(st.st_size);
    std::size_t offset{};
    while (offset < bytes.size()) {
      auto n = read(file.Get(), bytes.data() + offset, bytes.size() - offset);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
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
  bool Check(int directory, const DiskPayload& p) {
    ++stats.dependency_stats;
    struct stat st{};
    return fstatat(directory, DiskFileName(p.file).c_str(), &st,
                   AT_SYMLINK_NOFOLLOW) == 0 &&
           Regular(st) && std::uint64_t(st.st_size) == p.bytes;
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
    if (!entry.durable_ || entry.manifest_.checkpoint != id)
      continue;
    auto reservation = impl_->ledger->Reserve(ResourceCategory::kMetadata,
                                              entry.metadata_.Info().bytes);
    DiskSnapshot snapshot;
    snapshot.manifest_ = entry.manifest_;
    snapshot.metadata_ = reservation.Convert();
    snapshot.pin_.token_ = entry.pin_;
    snapshot.store_ = impl_.get();
    return snapshot;
  }
  return {};
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
  bool belongs{};
  for (const auto& c : snapshot.manifest_.components) {
    if (private_file)
      belongs |= c.tail == payload || c.private_state == payload;
    else
      belongs |= std::find(c.chunks.begin(), c.chunks.end(), payload) !=
                 c.chunks.end();
  }
  if (!belongs)
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
  const int directory =
      private_file ? impl_->private_files.Get() : impl_->chunks.Get();
  Fd file(openat(directory, DiskFileName(payload.file).c_str(),
                 O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
  struct stat before{}, after{};
  if (file.Get() < 0 || fstat(file.Get(), &before) < 0 || !Regular(before) ||
      std::uint64_t(before.st_size) != payload.bytes)
    Invalid();
  const Impl::FileIdentity identity{before.st_dev,  before.st_ino,
                                    before.st_size, before.st_mtim,
                                    before.st_ctim, payload.checksum};
  stats.filesystem_ns += elapsed(begin);
  const auto key = std::make_pair(private_file, payload.file);
  {
    std::lock_guard guard(impl_->verification_mutex);
    auto it = impl_->verified.find(key);
    stats.verification_cached =
        it != impl_->verified.end() && it->second.identity == identity;
  }
  std::uint64_t offset{}, checksum{};
  while (offset < payload.bytes) {
    auto piece = staging.first(
        std::min<std::uint64_t>(staging.size(), payload.bytes - offset));
    begin = Clock::now();
    std::size_t filled{};
    while (filled < piece.size()) {
      auto n = read(file.Get(), piece.data() + filled, piece.size() - filled);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        Invalid();
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
  if (fstat(file.Get(), &after) < 0 || !Regular(after) ||
      identity != Impl::FileIdentity{after.st_dev, after.st_ino, after.st_size,
                                     after.st_mtim, after.st_ctim,
                                     payload.checksum} ||
      (!stats.verification_cached && checksum != payload.checksum))
    Invalid();
  stats.filesystem_ns += elapsed(begin);
  if (!stats.verification_cached) {
    std::lock_guard guard(impl_->verification_mutex);
    auto it = impl_->verified.find(key);
    if (it != impl_->verified.end()) {
      it->second.identity = identity;
    } else if (impl_->verified.size() < 1024) {
      try {
        auto reservation =
            impl_->ledger->Reserve(ResourceCategory::kMetadata, 256);
        auto [inserted, added] =
            impl_->verified.emplace(key, Impl::VerifiedFile{{}, identity});
        try {
          inserted->second.charge = reservation.Convert();
        } catch (...) {
          impl_->verified.erase(inserted);
          throw;
        }
      } catch (const std::bad_alloc&) {
        // Verification caching is optional; the completed read remains valid.
      }
    }
  }
  return stats;
}
DiskReadPin DiskStore::Pin(CheckpointId id) const {
  std::lock_guard guard(impl_->index_mutex);
  DiskReadPin pin;
  for (const auto& entry : impl_->entries)
    if (entry.durable_ && entry.manifest_.checkpoint == id) {
      pin.token_ = entry.pin_;
      break;
    }
  return pin;
}
bool DiskStore::Retire(CheckpointId id) {
  std::lock_guard guard(impl_->mutex);
  impl_->RequireRecovered();
  std::unique_lock index_guard(impl_->index_mutex);
  auto& entries = impl_->entries;
  auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) {
    return e.manifest_.checkpoint == id;
  });
  if (it == entries.end() || it->pin_.use_count() > 1)
    return false;
  // Remove the index claim before attempting unlink: an I/O error can report
  // an uncertain outcome even when the directory entry has disappeared. Keep
  // dependencies untouched until the manifest-directory barrier succeeds.
  auto retired = std::move(*it);
  entries.erase(it);
  index_guard.unlock();
  try {
    impl_->Remove(impl_->manifests.Get(), DiskFileName(retired.file_));
    impl_->Sync(impl_->manifests.Get());
    for (const auto& c : retired.manifest_.components) {
      for (const auto& p : c.chunks)
        if (!impl_->Referenced(impl_->chunks.Get(), p.file))
          impl_->Remove(impl_->chunks.Get(), DiskFileName(p.file));
      for (auto p : {c.tail, c.private_state})
        if (p && !impl_->Referenced(impl_->private_files.Get(), p->file))
          impl_->Remove(impl_->private_files.Get(), DiskFileName(p->file));
    }
    impl_->Sync(impl_->chunks.Get());
    impl_->Sync(impl_->private_files.Get());
    return true;
  } catch (...) {
    // Both checkpoint identities and freed payload capacity remain uncertain
    // until every deletion has crossed its directory durability barrier.
    impl_->recovery_required = true;
    throw;
  }
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
