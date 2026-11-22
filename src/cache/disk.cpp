#include "src/cache/disk.hpp"

#include <gufo/cache_manifest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/rust/callback.hpp"

namespace gufo::cache {
namespace detail {
struct DiskAvailability {
  ResourceCharge metadata;
  std::atomic<bool> quarantined{false};
};
}  // namespace detail
namespace {
// New payloads are synchronized in batches by concurrent threads.
constexpr std::size_t kPayloadSyncBatch = 128;
constexpr std::size_t kPayloadSyncThreads = 32;
#ifdef GUFO_CACHE_TESTING
thread_local void (*reference_validation_hook)(bool) noexcept {};
#endif
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
// The C interface mirrors these definitions.
static_assert(GUFO_CACHE_APPEND_ROWS ==
              static_cast<int>(ComponentKind::kAppendRows));
static_assert(GUFO_CACHE_PRIVATE_STATE ==
              static_cast<int>(ComponentKind::kPrivateState));
static_assert(GUFO_CACHE_WINDOW_ROWS ==
              static_cast<int>(ComponentKind::kWindowRows));
static_assert(GUFO_CACHE_MAX_PURPOSE ==
              static_cast<int>(CheckpointPurpose::kLearned));
static_assert(GUFO_CACHE_FORMAT_VERSION == kDiskFormatVersion);
static_assert(GUFO_CACHE_WINDOW_FORMAT_VERSION == kDiskWindowFormatVersion);
static_assert(GUFO_CACHE_MAX_MANIFEST_BYTES == kMaxManifestBytes);
static_assert(std::is_same_v<Token, std::uint32_t> &&
              sizeof(DiskFileId) == sizeof(GufoCachePayload::file));
// Throws for a failed codec call: a rejected manifest as before, memory
// exhaustion as std::bad_alloc, and an exception of a callback as itself.
void Check(GufoCacheStatus status,
           const core::ffi::CallbackFailure* failure = nullptr) {
  if (failure)
    failure->Rethrow();
  switch (status) {
    case GUFO_CACHE_COMPLETE:
      return;
    case GUFO_CACHE_REJECTED:
      Invalid();
    case GUFO_CACHE_NO_MEMORY:
      throw std::bad_alloc();
    case GUFO_CACHE_STOPPED:
    case GUFO_CACHE_INTERNAL:
      break;
  }
  throw std::logic_error("disk manifest codec failed");
}
GufoCachePayload CodecPayload(const DiskPayload& p) {
  GufoCachePayload c{};
  std::copy(p.file.begin(), p.file.end(), c.file);
  c.bytes = p.bytes;
  c.checksum = p.checksum;
  return c;
}
DiskPayload FromCodec(const GufoCachePayload& c) {
  DiskPayload p;
  std::copy(std::begin(c.file), std::end(c.file), p.file.begin());
  p.bytes = c.bytes;
  p.checksum = c.checksum;
  return p;
}
std::optional<DiskPayload> OptionalPayload(int present,
                                           const GufoCachePayload& c) {
  if (!present)
    return {};
  return FromCodec(c);
}
struct MetadataPlan {
  std::size_t retained{}, references{};
};
// The metadata DecodeManifest retains, which reserves exactly these
// capacities, and the payload references it validates.
MetadataPlan PlanMetadata(std::span<const std::uint8_t> bytes) {
  GufoCacheManifestPlan plan{};
  Check(gufo_cache_manifest_plan(bytes.data(), bytes.size(), &plan));
  auto retained = Add(plan.input_bytes, plan.tokens * sizeof(Token));
  retained = Add(retained, plan.components * sizeof(DiskComponent));
  retained = Add(retained, plan.chunks * sizeof(DiskPayload));
  return {retained, plan.references};
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
// Synchronizes files on helper threads while the caller continues; Wait()
// helps finish them. File systems commit concurrent synchronizations together,
// while one at a time each small file waits for its own commit (#1180).
class ConcurrentSync {
public:
  ConcurrentSync(std::span<const platform::PlatformFile> files,
                 std::size_t threads)
      : files_(files), begin_(std::chrono::steady_clock::now()) {
    threads = std::min(threads, files.size());
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
      try {
        workers_.emplace_back([this] { Work(); });
      } catch (const std::system_error&) {
        break;  // Wait() completes the remaining files.
      }
    }
  }
  ConcurrentSync(const ConcurrentSync&) = delete;
  ConcurrentSync& operator=(const ConcurrentSync&) = delete;
  ~ConcurrentSync() { Stop(); }
  // Returns the first failure once every file has been attempted.
  std::error_code Wait() {
    Work();
    Stop();
    return failure_;
  }
  [[nodiscard]] std::size_t Files() const { return files_.size(); }
  [[nodiscard]] std::uint64_t ElapsedNs() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - begin_)
        .count();
  }

private:
  void Work() {
    for (std::size_t i; (i = next_.fetch_add(1)) < files_.size();) {
      std::error_code ec;
      if (!platform::SyncFile(files_[i], ec)) {
        std::lock_guard guard(failure_mutex_);
        if (!failure_)
          failure_ = ec ? ec : std::make_error_code(std::errc::io_error);
      }
    }
  }
  void Stop() {
    // Joining also orders the workers' failure writes before Wait() reads.
    workers_.clear();
    end_ = std::chrono::steady_clock::now();
  }
  std::span<const platform::PlatformFile> files_;
  std::chrono::steady_clock::time_point begin_, end_;
  std::atomic<std::size_t> next_{};
  std::mutex failure_mutex_;
  std::error_code failure_;
  std::vector<std::jthread> workers_;
};
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
}  // namespace
std::uint64_t DiskChecksum(std::span<const std::uint8_t> bytes,
                           std::uint64_t crc) {
  Check(gufo_cache_checksum(bytes.data(), bytes.size(), crc, &crc));
  return crc;
}
std::vector<std::uint8_t> EncodeManifest(const DiskManifest& m) {
  // Oversized manifests are rejected before their C view is built.
  if (m.components.size() > GUFO_CACHE_MAX_COMPONENTS)
    Invalid();
  std::size_t chunk_count{};
  for (const auto& c : m.components) {
    chunk_count += c.chunks.size();
    if (chunk_count > GUFO_CACHE_MAX_REFERENCES)
      Invalid();
  }
  std::vector<GufoCachePayload> chunks;
  chunks.reserve(chunk_count);
  std::vector<GufoCacheComponent> components;
  components.reserve(m.components.size());
  for (const auto& c : m.components) {
    const auto& d = c.descriptor;
    const auto first = chunks.size();
    for (const auto& p : c.chunks)
      chunks.push_back(CodecPayload(p));
    components.push_back({
        .id = d.id.value,
        .layout_version = d.layout_version,
        .kind = static_cast<std::uint8_t>(d.kind),
        .row_bytes = d.row_bytes,
        .rows_per_chunk = d.rows_per_chunk,
        .state_bytes = d.state_bytes,
        .window_rows = d.window_rows,
        .position_id = c.position.id.value,
        .valid_rows = c.position.valid_rows,
        .chunks = chunks.data() + first,
        .chunk_count = c.chunks.size(),
        .has_tail = c.tail.has_value(),
        .tail = c.tail ? CodecPayload(*c.tail) : GufoCachePayload{},
        .has_private_state = c.private_state.has_value(),
        .private_state = c.private_state ? CodecPayload(*c.private_state)
                                         : GufoCachePayload{},
    });
  }
  GufoCacheManifest view{
      .checkpoint = m.checkpoint.value,
      .lineage = m.lineage.value,
      .compatibility = {},
      .purpose = static_cast<std::uint8_t>(m.purpose),
      .rank = m.rank,
      .input = m.input.data(),
      .input_len = m.input.size(),
      .tokens = m.tokens.data(),
      .token_count = m.tokens.size(),
      .components = components.data(),
      .component_count = components.size(),
  };
  std::copy(m.compatibility.begin(), m.compatibility.end(), view.compatibility);
  struct Output {
    std::vector<std::uint8_t> bytes;
    core::ffi::CallbackFailure failure;
  } out;
  const auto status = gufo_cache_manifest_encode(
      &view,
      [](void* context, std::size_t len) noexcept -> std::uint8_t* {
        auto& out = *static_cast<Output*>(context);
        return out.failure.Guard<std::uint8_t*>(nullptr, [&] {
          out.bytes.resize(len);
          return out.bytes.data();
        });
      },
      &out);
  Check(status, &out.failure);
  return std::move(out.bytes);
}
DiskManifest DecodeManifest(std::span<const std::uint8_t> bytes) {
  // Reserves the exact capacities PlanMetadata accounts for.
  struct Decoder {
    DiskManifest m;
    core::ffi::CallbackFailure failure;
  } decoder;
  const GufoCacheManifestSink sink{
      .context = &decoder,
      .begin =
          [](void* context, const GufoCacheManifest* view) noexcept {
            auto& decoder = *static_cast<Decoder*>(context);
            return decoder.failure.Guard(1, [&] {
              auto& m = decoder.m;
              m.checkpoint.value = view->checkpoint;
              m.lineage.value = view->lineage;
              std::copy(std::begin(view->compatibility),
                        std::end(view->compatibility), m.compatibility.begin());
              m.purpose = static_cast<CheckpointPurpose>(view->purpose);
              m.rank = view->rank;
              m.input.assign(view->input, view->input + view->input_len);
              m.tokens.reserve(view->token_count);
              m.components.reserve(view->component_count);
              return 0;
            });
          },
      .tokens =
          [](void* context, const std::uint32_t* tokens,
             std::size_t count) noexcept {
            auto& decoder = *static_cast<Decoder*>(context);
            return decoder.failure.Guard(1, [&] {
              decoder.m.tokens.insert(decoder.m.tokens.end(), tokens,
                                      tokens + count);
              return 0;
            });
          },
      .component =
          [](void* context, const GufoCacheComponent* c) noexcept {
            auto& decoder = *static_cast<Decoder*>(context);
            return decoder.failure.Guard(1, [&] {
              DiskComponent component;
              auto& d = component.descriptor;
              d.id.value = c->id;
              d.layout_version = c->layout_version;
              d.kind = static_cast<ComponentKind>(c->kind);
              d.row_bytes = c->row_bytes;
              d.rows_per_chunk = c->rows_per_chunk;
              d.state_bytes = c->state_bytes;
              d.window_rows = c->window_rows;
              component.position = {{c->position_id}, c->valid_rows};
              component.chunks.reserve(c->chunk_count);
              component.tail = OptionalPayload(c->has_tail, c->tail);
              component.private_state =
                  OptionalPayload(c->has_private_state, c->private_state);
              decoder.m.components.push_back(std::move(component));
              return 0;
            });
          },
      .chunks =
          [](void* context, const GufoCachePayload* chunks,
             std::size_t count) noexcept {
            auto& decoder = *static_cast<Decoder*>(context);
            return decoder.failure.Guard(1, [&] {
              auto& target = decoder.m.components.back().chunks;
              for (std::size_t i = 0; i < count; ++i)
                target.push_back(FromCodec(chunks[i]));
              return 0;
            });
          },
  };
  Check(gufo_cache_manifest_decode(bytes.data(), bytes.size(), &sink),
        &decoder.failure);
  return std::move(decoder.m);
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
// An index slot also pays for its entry's shared manifest object: the
// DiskManifest header and the shared_ptr control block beside it.
constexpr std::size_t kEntrySlotBytes =
    sizeof(DurableEntry) + sizeof(DiskManifest) + 64;
struct DiskStore::Impl {
  ResourceLedger* ledger;
  ResourceCharge metadata, entry_capacity;
  platform::Directory root;
  Fd lock;
  platform::Directory version, manifests, chunks, private_files, temporary,
      uses;
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
        uses(DirectoryAt(version, "uses")),
        budget_bytes(budget) {
    // Persist the layout, including a newly created root and ancestors, before
    // any checkpoint can rely on these directory entries.
    SyncDirectoryFd(version);
    SyncDirectoryFd(root);
    std::error_code ec;
    if (!platform::SyncAncestors(root, ec))
      IoError("sync cache ancestors", ec);
    std::size_t manifest_count{};
    for (const platform::Directory* directory :
         {&manifests, &chunks, &private_files, &uses})
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
                          Multiply(manifest_count, kEntrySlotBytes));
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
  void ValidateReferences() {
    if (entries.empty())
      return;
#ifdef GUFO_CACHE_TESTING
    const auto hook = reference_validation_hook;
    if (hook)
      hook(true);
    const Finally leave_scope{[hook] {
      if (hook)
        hook(false);
    }};
#endif
    // Shared files must describe the same immutable lineage/range/layout.
    // Private bytes cannot be aliased across checkpoints. Reject every side
    // of a conflict, independent of filesystem enumeration order.
    struct Claim {
      const DiskManifest* manifest{};
      const ComponentDescriptor* descriptor{};
      const DiskPayload* payload{};
      Rows first{};
      std::uint64_t observed_bytes{};
      bool regular{}, conflict{};
    };
    using Key = std::pair<bool, DiskFileId>;
    std::size_t references{};
    for (const auto& entry : entries)
      for (const auto& c : entry.manifest_->components)
        references = Add(
            references, c.chunks.size() + bool(c.tail) + bool(c.private_state));
    stats.dependency_references = references;
    // One temporary node per distinct dependency: first its filesystem
    // observation, then its borrowed canonical claim. Do not index orphan
    // directory entries or retain another metadata cache after recovery.
    // The reservation covers worst-case distinct nodes, tree/allocator overhead
    // and the two compact arrays. It is temporary working capacity, never
    // evidence of retained physical backing.
    auto working = ledger->Reserve(
        ResourceCategory::kMetadata,
        Add(Add(Multiply(references,
                         sizeof(Claim) + sizeof(Key) + 8 * sizeof(void*)),
                Multiply(entries.size(), sizeof(std::size_t) + 1)),
            2 * 64));
    std::map<Key, Claim> claims;
    std::vector<std::size_t> checkpoints(entries.size());
    std::vector<std::uint8_t> rejected(entries.size());
    if (checkpoints.capacity() != entries.size() ||
        rejected.capacity() != entries.size())
      throw std::logic_error("unexpected disk validation capacity");
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto check = [&](bool is_private, const DiskPayload& p) {
        auto [it, inserted] = claims.try_emplace({is_private, p.file});
        auto& observed = it->second;
        if (inserted) {
          ++stats.dependency_stats;
          platform::FileInfo st{};
          std::error_code ec;
          observed.regular =
              platform::StatAt(is_private ? private_files : chunks,
                               DiskFileName(p.file), false, &st, ec) &&
              Regular(st);
          // Allocation pressure is not evidence of a corrupt dependency.
          // Abort recovery and relinquish ownership; do not discard owners.
          if (!observed.regular && ec == std::errc::not_enough_memory)
            throw std::bad_alloc();
          observed.observed_bytes = st.size;
        }
        // Compare each reference with the actual size, not with whether a
        // different owner's first claim passed. Bad claims cannot poison
        // later compatible owners of the same file.
        rejected[i] |= !observed.regular || observed.observed_bytes != p.bytes;
      };
      for (const auto& c : entries[i].manifest_->components) {
        for (const auto& p : c.chunks)
          check(false, p);
        for (const auto* p : {&c.tail, &c.private_state})
          if (*p)
            check(true, p->value());
      }
    }
    stats.distinct_dependency_files = claims.size();
    // Only complete manifests participate in geometry/alias and checkpoint-ID
    // conflicts, preserving the previous filter-before-conflict behavior.
    // Reuse observations only within this exclusive-ownership recovery. An
    // external writer ignoring LOCK is not made safe by repeated stat calls;
    // restore must still check its opened file, identity and payload checksum.
    std::size_t available{};
    const auto geometry = [](const Claim& c) {
      const auto& d = *c.descriptor;
      return std::array<std::uint64_t, 9>{c.manifest->lineage.value,
                                          d.id.value,
                                          d.layout_version,
                                          d.row_bytes,
                                          d.rows_per_chunk,
                                          c.first,
                                          c.payload->bytes,
                                          c.payload->checksum,
                                          d.window_rows};
    };
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto& m = *entries[i].manifest_;
      if (rejected[i])
        continue;
      checkpoints[available++] = i;
      const auto claim = [&](bool is_private, const DiskPayload& p,
                             const ComponentDescriptor& d, Rows first) {
        const Claim current{&m, &d, &p, first};
        auto& previous = claims.at({is_private, p.file});
        if (!previous.manifest)
          previous = current;
        else if (is_private ||
                 previous.manifest->compatibility != m.compatibility ||
                 geometry(previous) != geometry(current))
          previous.conflict = true;
      };
      for (const auto& c : m.components) {
        Rows first = FirstChunkRow(c.descriptor, c.position.valid_rows);
        for (const auto& p : c.chunks) {
          claim(false, p, c.descriptor, first);
          first = Add(first, c.descriptor.rows_per_chunk);
        }
        for (const auto* p : {&c.tail, &c.private_state})
          if (*p)
            claim(true, p->value(), c.descriptor, first);
      }
    }
    checkpoints.resize(available);
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (rejected[i])
        continue;
      const auto conflict = [&](bool is_private, const DiskPayload& p) {
        return claims.at({is_private, p.file}).conflict;
      };
      for (const auto& c : entries[i].manifest_->components) {
        for (const auto& p : c.chunks)
          rejected[i] |= conflict(false, p);
        for (const auto* p : {&c.tail, &c.private_state})
          if (*p)
            rejected[i] |= conflict(true, p->value());
      }
    }
    const auto checkpoint = [&](std::size_t i) {
      return entries[i].manifest_->checkpoint.value;
    };
    std::sort(checkpoints.begin(), checkpoints.end(),
              [&](auto a, auto b) { return checkpoint(a) < checkpoint(b); });
    for (std::size_t first = 0; first < checkpoints.size();) {
      auto end = first + 1;
      while (end < checkpoints.size() &&
             checkpoint(checkpoints[end]) == checkpoint(checkpoints[first]))
        ++end;
      if (end - first > 1)
        for (auto i = first; i < end; ++i)
          rejected[checkpoints[i]] = true;
      first = end;
    }
    std::size_t i{};
    std::erase_if(entries, [&](const auto&) {
      const bool remove = rejected[i++];
      if (remove)
        ++stats.rejected_manifests;
      return remove;
    });
  }
  // Applies ValidateReferences' rules to the publication candidate, the last
  // entry. Indexed entries were already validated against each other, so only
  // pairs involving the candidate can conflict. Its catalog is sized by the
  // candidate rather than by every reference on disk: a full catalog grows
  // with the tier and, charged while the RAM cache fills its share of the
  // ledger, would refuse every later publication (#1030).
  void ValidateCandidate() {
    const auto& candidate = *entries.back().manifest_;
    using Geometry = std::array<std::uint64_t, 9>;
    const auto each_reference = [](const DiskManifest& m, auto&& visit) {
      for (const auto& c : m.components) {
        const auto& d = c.descriptor;
        Rows first = FirstChunkRow(d, c.position.valid_rows);
        const auto geometry = [&](const DiskPayload& p) {
          return Geometry{m.lineage.value, d.id.value,       d.layout_version,
                          d.row_bytes,     d.rows_per_chunk, first,
                          p.bytes,         p.checksum,       d.window_rows};
        };
        for (const auto& p : c.chunks) {
          visit(false, p.file, geometry(p));
          first = Add(first, d.rows_per_chunk);
        }
        for (auto p : {c.tail, c.private_state})
          if (p)
            visit(true, p->file, geometry(*p));
      }
    };
    std::size_t references{};
    for (const auto& c : candidate.components)
      references = Add(references,
                       c.chunks.size() + bool(c.tail) + bool(c.private_state));
    using Claims = std::map<std::pair<bool, DiskFileId>, Geometry>;
    // Map records plus tree links and allocator overhead per reference.
    auto working =
        ledger->Reserve(ResourceCategory::kMetadata,
                        Multiply(references, sizeof(Claims::value_type) + 64));
    Claims claims;
    each_reference(candidate, [&](bool is_private, const DiskFileId& file,
                                  const Geometry& geometry) {
      auto [it, inserted] = claims.try_emplace({is_private, file}, geometry);
      if (!inserted && (is_private || it->second != geometry))
        Invalid();
    });
    for (std::size_t i = 0; i + 1 < entries.size(); ++i) {
      const auto& m = *entries[i].manifest_;
      if (m.checkpoint == candidate.checkpoint)
        Invalid();
      const bool compatible = m.compatibility == candidate.compatibility;
      each_reference(m, [&](bool is_private, const DiskFileId& file,
                            const Geometry& geometry) {
        auto it = claims.find({is_private, file});
        if (it != claims.end() &&
            (is_private || !compatible || it->second != geometry))
          Invalid();
      });
    }
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
  // Waits for a concurrent sync and records its wall time from the start.
  void Wait(ConcurrentSync& sync) {
    const auto failure = sync.Wait();
    publication_stats.fsync_calls += sync.Files();
    publication_stats.fsync_ns += sync.ElapsedNs();
    if (failure)
      IoError("sync cache file/directory", failure);
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
  bool Remove(const platform::Directory& directory, const std::string& name) {
    platform::FileInfo st{};
    std::error_code ec;
    if (!platform::StatAt(directory, name, false, &st, ec)) {
      if (ec == std::errc::no_such_file_or_directory)
        return false;
      IoError("stat cache removal", ec);
    }
    if (!Regular(st))
      return false;
    if (&directory == &chunks || &directory == &private_files)
      if (auto id = ParseName(name))
        ForgetVerified({&directory == &private_files, *id});
    if (!platform::Unlink(directory, name, ec))
      IoError("remove cache file", ec);
    {
      std::lock_guard guard(stats_mutex);
      stats.managed_bytes -= st.size;
    }
    return true;
  }
  bool Referenced(const platform::Directory& directory, DiskFileId id) const {
    for (const auto& entry : entries) {
      if ((&directory == &manifests || &directory == &uses) &&
          entry.file_ == id)
        return true;
      for (const auto& c : entry.manifest_->components) {
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
         {&chunks, &private_files, &temporary, &uses}) {
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
         {&manifests, &chunks, &private_files, &temporary, &uses})
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
      for (const auto& component : entry.manifest_->components) {
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
        for (const auto& component : entry.manifest_->components) {
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
    candidate.manifest_ = std::make_shared<DiskManifest>(DecodeManifest(bytes));
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
            for (const auto& component : entry.manifest_->components) {
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
          for (auto& component : candidate.manifest_->components) {
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
      for (const auto& component : entry.manifest_->components) {
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
                          Multiply(entries.size() + 1, kEntrySlotBytes));
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
      ValidateCandidate();
      index_guard.unlock();
      // Candidate is invisible to pinned snapshots until committed. Reuse from
      // filesystem existence after validating claims, never from this
      // candidate.
      // Each payload is renamed into place only after its own sync. A batch
      // syncs on helper threads while the next one is written.
      struct Written {
        Fd fd;
        const platform::Directory* directory;
        std::string name;
      };
      std::vector<Written> written, syncing;
      std::vector<platform::PlatformFile> syncing_files;
      written.reserve(kPayloadSyncBatch);
      syncing.reserve(kPayloadSyncBatch);
      syncing_files.reserve(kPayloadSyncBatch);
      // Declared after its files: destruction joins its threads first.
      std::optional<ConcurrentSync> sync;
      const auto finish_sync = [&] {
        if (!sync)
          return;
        Wait(*sync);
        sync.reset();
        for (auto& w : syncing) {
          {
            // The payload is durable; drop its clean pages so multi-GB
            // snapshots don't accumulate in the page cache.
            CostTimer timer(publication_stats.filesystem_ns);
            platform::Advise(w.fd.Get(), platform::AdviseHint::kDontNeed, 0, 0);
          }
          DISK_STEP(kPayloadSynced);
          Rename(*w.directory, w.name);
          DISK_STEP(kPayloadRenamed);
        }
        syncing.clear();
      };
      const auto start_sync = [&] {
        finish_sync();
        syncing.swap(written);
        syncing_files.clear();
        for (const auto& w : syncing)
          syncing_files.push_back(w.fd.Get());
        if (!syncing.empty())
          sync.emplace(syncing_files, kPayloadSyncThreads);
      };
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
          for (auto& c : entries.back().manifest_->components) {
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
        written.push_back({std::move(fd), &dir, std::move(payload_name)});
        if (written.size() == kPayloadSyncBatch)
          start_sync();
      }
      start_sync();
      finish_sync();
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
        bytes = EncodeManifest(*entries.back().manifest_);
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
    entry.manifest_ = std::make_shared<DiskManifest>(DecodeManifest(bytes));
    entry.availability_ = NewAvailability();
    // Decoding explicitly reserves exact capacities with the pinned library.
    std::size_t actual =
        entry.manifest_->input.capacity() +
        entry.manifest_->tokens.capacity() * sizeof(Token) +
        entry.manifest_->components.capacity() * sizeof(DiskComponent);
    for (const auto& c : entry.manifest_->components)
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
std::size_t DiskStore::OwnedManifestBytes() const {
  const std::lock_guard guard(impl_->index_mutex);
  std::size_t owned{};
  for (const auto& entry : impl_->entries) {
    const auto& manifest = *entry.manifest_;
    auto bytes = Add(manifest.input.size(),
                     Multiply(manifest.tokens.size(), sizeof(Token)));
    bytes =
        Add(bytes, Multiply(manifest.components.size(), sizeof(DiskComponent)));
    for (const auto& component : manifest.components)
      bytes =
          Add(bytes, Multiply(component.chunks.size(), sizeof(DiskPayload)));
    // Decode initialized these elements before converting their admission.
    // A converted estimate elsewhere in the ledger proves no such backing.
    if (!entry.metadata_ || bytes > entry.metadata_.Info().bytes)
      throw std::logic_error("manifest storage exceeds retained charge");
    if (bytes > std::numeric_limits<std::size_t>::max() - owned)
      throw std::overflow_error("owned manifest storage overflows");
    owned += bytes;
  }
  return owned;
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
        entry.manifest_->checkpoint != id)
      continue;
    DiskSnapshot snapshot;
    snapshot.manifest_ = entry.manifest_;
    snapshot.metadata_ = entry.metadata_;
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
      !Contains(*snapshot.manifest_, private_file, payload))
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
std::uint64_t DiskStore::WarmPayload(const DiskSnapshot& snapshot,
                                     bool private_file,
                                     const DiskPayload& payload,
                                     std::span<std::uint8_t> buffer,
                                     std::stop_token stop) const {
  if (snapshot.store_ != impl_.get() || !snapshot.pin_ || buffer.empty() ||
      !Contains(*snapshot.manifest_, private_file, payload))
    throw std::invalid_argument("invalid disk warm-up dependency");
  const auto& directory = private_file ? impl_->private_files : impl_->chunks;
  std::error_code ec;
  Fd file(platform::OpenAt(directory, DiskFileName(payload.file),
                           platform::FileMode::kRead, ec));
  if (!platform::FileValid(file.Get()))
    return 0;
  std::uint64_t read{};
  while (read < payload.bytes && !stop.stop_requested()) {
    const auto n = platform::Read(
        file.Get(), buffer.data(),
        std::min<std::uint64_t>(buffer.size(), payload.bytes - read), ec);
    if (n == 0 && ec == std::errc::interrupted)
      continue;
    if (n == 0)
      break;
    read += n;
  }
  return read;
}
DiskReadStats DiskStore::ReadPayload(
    const DiskSnapshot& snapshot, bool private_file, const DiskPayload& payload,
    std::span<std::uint8_t> staging,
    const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>&
        consume,
    DiskReadStats* observation) {
  if (staging.empty())
    throw std::invalid_argument("invalid disk read lease or staging");
  return ReadPayload(
      snapshot, private_file, payload,
      [staging](std::uint64_t) { return staging; }, consume, observation);
}
DiskReadStats DiskStore::ReadPayload(
    const DiskSnapshot& snapshot, bool private_file, const DiskPayload& payload,
    const std::function<std::span<std::uint8_t>(std::uint64_t)>& buffer,
    const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>&
        consume,
    DiskReadStats* observation) {
  if (snapshot.store_ != impl_.get() || !snapshot.pin_ || !buffer || !consume)
    throw std::invalid_argument("invalid disk read lease or staging");
  if (!Contains(*snapshot.manifest_, private_file, payload))
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
    const auto staging = buffer(offset);
    if (staging.empty())
      throw std::invalid_argument("invalid disk read staging");
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
DiskUseHistory DiskStore::UseHistory(CheckpointId id) const {
  std::lock_guard guard(impl_->mutex);
  const auto it = std::find_if(
      impl_->entries.begin(), impl_->entries.end(),
      [&](const auto& e) { return e.manifest_->checkpoint == id; });
  if (it == impl_->entries.end())
    return {};
  std::error_code ec;
  const Fd file(platform::OpenAt(impl_->uses, DiskFileName(it->file_),
                                 platform::FileMode::kRead, ec));
  platform::FileInfo info{};
  std::array<std::uint8_t, 40> bytes{};
  if (!platform::FileValid(file.Get()) ||
      !platform::Stat(file.Get(), &info, ec) || !Regular(info) ||
      (info.size != 16 && info.size != bytes.size()))
    return {};
  std::size_t offset = 0;
  while (offset < info.size) {
    const auto read = platform::Read(file.Get(), bytes.data() + offset,
                                     info.size - offset, ec);
    if (!read) {
      if (ec == std::errc::interrupted)
        continue;
      return {};
    }
    offset += read;
  }
  const auto word = [&](std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
      value |= std::uint64_t(bytes[offset + i]) << (8 * i);
    return value;
  };
  const auto payload = static_cast<std::size_t>(info.size) - 8;
  if (word(payload) != DiskChecksum(std::span(bytes).first(payload), id.value))
    return {};
  if (info.size == 16)
    return {word(0), 0, {}};
  if (word(16) > 1 || (word(16) && word(24) > word(8)))
    return {};
  return {word(0), word(8), word(16) ? std::optional(word(24)) : std::nullopt};
}
std::uint64_t DiskStore::LastUse(CheckpointId id) const {
  return UseHistory(id).last_used;
}
bool DiskStore::TryRecordUse(CheckpointId id, std::uint64_t used) {
  return TryRecordUse(id, DiskUseHistory{used, 0, {}});
}
bool DiskStore::TryRecordUse(CheckpointId id, const DiskUseHistory& history) {
  if (history.reused_at && *history.reused_at > history.admitted_bytes)
    return false;
  std::unique_lock guard(impl_->mutex, std::try_to_lock);
  if (!guard.owns_lock() || impl_->recovery_required)
    return false;
  std::unique_lock index_guard(impl_->index_mutex);
  const auto it = std::find_if(
      impl_->entries.begin(), impl_->entries.end(), [&](const auto& e) {
        return e.durable_ && e.available_ && e.manifest_->checkpoint == id;
      });
  if (it == impl_->entries.end())
    return false;
  const auto name = DiskFileName(it->file_);
  index_guard.unlock();
  platform::FileInfo prior{};
  std::error_code ec;
  const auto released =
      platform::StatAt(impl_->uses, name, false, &prior, ec) && Regular(prior)
          ? prior.size
          : 0;
  {
    const std::lock_guard stats_guard(impl_->stats_mutex);
    const auto after_removal = impl_->stats.managed_bytes -
                               std::min(impl_->stats.managed_bytes, released);
    if (after_removal > impl_->budget_bytes ||
        impl_->budget_bytes - after_removal < 40)
      return false;
  }
  std::array<std::uint8_t, 40> bytes{};
  const std::array<std::uint64_t, 4> words{
      history.last_used, history.admitted_bytes,
      history.reused_at.has_value() ? 1U : 0U, history.reused_at.value_or(0)};
  for (std::size_t word = 0; word < words.size(); ++word)
    for (unsigned i = 0; i < 8; ++i)
      bytes[word * 8 + i] = words[word] >> (8 * i);
  const auto checksum = DiskChecksum(std::span(bytes).first(32), id.value);
  for (unsigned i = 0; i < 8; ++i)
    bytes[32 + i] = checksum >> (8 * i);
  try {
    // Use history is advisory: unlinking an old record before replacement may
    // lose recency on a crash, but never changes restore eligibility.
    impl_->Remove(impl_->uses, name);
    impl_->Remove(impl_->temporary, name);
    auto file = impl_->Create(name);
    impl_->WriteFile(file.Get(), bytes);
    impl_->Sync(file.Get());
    impl_->Rename(impl_->uses, name);
    impl_->Sync(impl_->uses);
    return true;
  } catch (...) {
    // Account partial files as orphans; normal recovery reclaims them.
    return false;
  }
}
DiskReadPin DiskStore::Pin(CheckpointId id, std::uint64_t epoch) const {
  std::lock_guard guard(impl_->index_mutex);
  DiskReadPin pin;
  for (const auto& entry : impl_->entries)
    if (entry.durable_ && entry.available_ &&
        entry.manifest_->checkpoint == id &&
        (!epoch || epoch == entry.epoch_)) {
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
    return e.manifest_->checkpoint == id;
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
    for (const auto& c : retired.manifest_->components) {
      for (const auto& p : c.chunks)
        if (!Referenced(chunks, p.file))
          Remove(chunks, DiskFileName(p.file));
      for (auto p : {c.tail, c.private_state})
        if (p && !Referenced(private_files, p->file))
          Remove(private_files, DiskFileName(p->file));
    }
    if (Remove(uses, DiskFileName(retired.file_)))
      Sync(uses);
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
    if (entry.manifest_->checkpoint == id && entry.epoch_ == epoch) {
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
        entry.manifest_->checkpoint != id)
      continue;
    auto reservation = impl_->ledger->Reserve(ResourceCategory::kMetadata,
                                              sizeof(DiskDescription) + 64);
    auto description = std::make_shared<DiskDescription>();
    description->manifest_metadata_ = entry.metadata_;
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
  return store_->Pin(manifest_->checkpoint, epoch_);
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
void DiskStore::SetReferenceValidationHook(void (*hook)(bool) noexcept) {
  reference_validation_hook = hook;
}
void DiskStore::SetCrashHook(std::function<void(DiskPublicationStep)> hook) {
  std::lock_guard guard(impl_->mutex);
  impl_->crash_hook = std::move(hook);
}
#endif
}  // namespace gufo::cache
