#include "src/cache/disk.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace gufo::cache {
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

private:
  int value_;
};
[[noreturn]] void IoError(const std::string& message) {
  throw std::runtime_error(message + ": " + std::strerror(errno));
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
        temporary(DirectoryAt(version.Get(), "tmp")) {
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
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.file_ < b.file_; });
  }
  void ValidateReferences() {
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
    std::size_t i{};
    std::erase_if(entries, [&](const auto&) {
      const bool remove = rejected[i++];
      if (remove)
        ++stats.rejected_manifests;
      return remove;
    });
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
const DiskStartupStats& DiskStore::Stats() const {
  return impl_->stats;
}
}  // namespace gufo::cache
