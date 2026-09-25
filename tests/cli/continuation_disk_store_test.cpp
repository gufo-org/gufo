#include "src/cli/serve/continuation_disk_store.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::ContinuationDiskEvent;
using gufo::server::ContinuationDiskEventAction;
using gufo::server::ContinuationDiskEventReason;
using gufo::server::ContinuationDiskStore;
using gufo::server::ContinuationDiskStoreOptions;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPersistenceDescriptor;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;

constexpr std::size_t kFakePayloadBytes = 16;
constexpr std::size_t kDiskHeaderBytes = 96;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "gufo-continuation-disk-XXXXXX")
                              .string();
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create temporary directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

template<typename Integer>
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     Integer value) {
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename Integer>
Integer GetLittleEndian(std::span<const std::uint8_t> source,
                        std::size_t offset) {
  Integer result = 0;
  for (std::size_t byte = 0; byte < sizeof(Integer); ++byte) {
    result |= static_cast<Integer>(source[offset + byte]) << (byte * 8U);
  }
  return result;
}

class FakeState final : public TextRunnerState {
public:
  void Invalidate() noexcept override {
    value = 0;
    position = 0;
  }

  std::uint64_t value{0};
  std::uint64_t position{0};
};

class FakeSnapshot final : public TextRunnerSnapshot {
public:
  FakeSnapshot(std::uint64_t snapshot_value, std::uint64_t snapshot_position)
      : value(snapshot_value), position(snapshot_position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return kFakePayloadBytes;
  }

  std::uint64_t value;
  std::uint64_t position;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake state");
  }
  return *fake;
}

const FakeSnapshot& RequireFakeSnapshot(const TextRunnerSnapshot& snapshot) {
  const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
  if (fake == nullptr) {
    throw std::invalid_argument("unexpected fake snapshot");
  }
  return *fake;
}

class FakeRunner final : public TextModelRunner {
public:
  explicit FakeRunner(std::string identity, std::string model_id = "fake-model")
      : identity_(identity.begin(), identity.end()),
        model_id_(std::move(model_id)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_id_,
        .state_abi = "fake-state-v1",
        .max_context = 4096,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
            },
        .persistence =
            TextRunnerPersistenceDescriptor{
                .compatibility_identity = identity_,
                .payload_version = 7,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {};
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(text.size());
    for (const char value : text) {
      tokens.push_back(static_cast<unsigned char>(value));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken>) const override {
    return {};
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<FakeState>();
  }

  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState&,
                                        std::span<const TextRunnerToken>,
                                        std::size_t,
                                        std::size_t) const override {
    throw std::logic_error("fake persistence test does not prefill");
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState&, gufo::sampling::SamplerState&) const override {
    throw std::logic_error("fake persistence test does not decode");
  }

  void Advance(TextRunnerState&, TextRunnerToken) const override {
    throw std::logic_error("fake persistence test does not advance");
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState&) const override {
    return kFakePayloadBytes;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& fake = RequireFakeState(state);
    return std::make_unique<FakeSnapshot>(fake.value, fake.position);
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto& saved = RequireFakeSnapshot(snapshot);
    auto& destination = RequireFakeState(state);
    destination.value = saved.value;
    destination.position = saved.position;
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    (void)RequireFakeSnapshot(snapshot);
    return kFakePayloadBytes;
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto& saved = RequireFakeSnapshot(snapshot);
    if (destination.size() != kFakePayloadBytes) {
      throw std::invalid_argument("fake destination size mismatch");
    }
    PutLittleEndian<std::uint64_t>(destination, 0, saved.value);
    PutLittleEndian<std::uint64_t>(destination, 8, saved.position);
    return destination.size();
  }

  std::function<void()> before_stream;
  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    if (before_stream)
      before_stream();
    std::array<std::uint8_t, kFakePayloadBytes> bytes{};
    (void)SerializePersistentSnapshot(snapshot, bytes);
    sink(std::span(bytes).first(7));
    sink(std::span(bytes).subspan(7));
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != kFakePayloadBytes) {
      throw std::invalid_argument("fake payload size mismatch");
    }
    auto& destination = RequireFakeState(state);
    destination.value = GetLittleEndian<std::uint64_t>(payload, 0);
    destination.position = GetLittleEndian<std::uint64_t>(payload, 8);
  }

private:
  std::vector<std::uint8_t> identity_;
  std::string model_id_;
};

std::unique_ptr<TextRunnerSnapshot> MakeSnapshot(const FakeRunner& runner,
                                                 std::uint64_t value,
                                                 std::uint64_t position) {
  auto state = runner.CreateState();
  auto& fake = RequireFakeState(*state);
  fake.value = value;
  fake.position = position;
  return runner.Snapshot(*state);
}

ContinuationDiskStoreOptions StoreOptions(
    const std::filesystem::path& directory,
    std::size_t capacity_bytes = 1024U * 1024U,
    std::size_t staging_bytes = 1024U * 1024U) {
  return {
      .directory = directory,
      .capacity_bytes = capacity_bytes,
      .staging_capacity_bytes = staging_bytes,
  };
}

std::vector<std::filesystem::path> CacheFiles(
    const std::filesystem::path& directory) {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".kvc") {
      files.push_back(entry.path());
    }
  }
  return files;
}

std::size_t ExpectedFileBytes(std::size_t identity_bytes,
                              std::size_t token_count) {
  return kDiskHeaderBytes + identity_bytes +
         token_count * sizeof(std::uint32_t) + kFakePayloadBytes;
}

ContinuationDiskStore::SaveResult SaveTokens(
    ContinuationDiskStore& store, const FakeRunner& runner,
    std::initializer_list<TextRunnerToken> tokens,
    const TextRunnerSnapshot& snapshot) {
  const std::vector<TextRunnerToken> owned(tokens);
  return store.Save(runner, owned, snapshot);
}

ContinuationDiskStore::RestoreResult RestoreTokens(
    ContinuationDiskStore& store, const FakeRunner& runner,
    TextRunnerState& state, std::initializer_list<TextRunnerToken> tokens) {
  const std::vector<TextRunnerToken> owned(tokens);
  return store.RestoreLongestPrefix(runner, state, owned);
}

void TestSha256KnownVector() {
  const std::array<std::uint8_t, 3> input = {'a', 'b', 'c'};
  Expect(gufo::crypto::Sha256Hex(input) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "shared SHA-256 implementation matches the standard vector");

  TemporaryDirectory directory;
  const auto path = directory.path() / "artifact.gguf";
  {
    std::ofstream output(path, std::ios::binary);
    output.write("abc", 3);
  }
  Expect(gufo::crypto::Sha256FileHex(path) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "streamed artifact SHA-256 matches the in-memory implementation");
}

void TestRestartRestoreAndCompatibilityIdentity() {
  TemporaryDirectory directory;
  const FakeRunner writer("artifact-A", "fake-old-executable");
  auto snapshot = MakeSnapshot(writer, 0x123456789ABCDEF0ULL, 3);
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    const auto saved = SaveTokens(store, writer, {1, 2, 3}, *snapshot);
    Expect(saved.stored && saved.payload_bytes == kFakePayloadBytes,
           "first process atomically stores a persistent snapshot");
    Expect(
        store.entry_count() == 1 && store.retained_bytes() == saved.file_bytes,
        "store accounts exact retained file bytes");
  }

  const FakeRunner compatible_reader("artifact-A", "fake-new-executable");
  auto compatible_state = compatible_reader.CreateState();
  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  const auto restored = RestoreTokens(restarted, compatible_reader,
                                      *compatible_state, {1, 2, 3, 4, 5});
  Expect(restored.restored && restored.token_count == 3,
         "new executable restores the longest compatible saved prefix");
  Expect(RequireFakeState(*compatible_state).value == 0x123456789ABCDEF0ULL &&
             RequireFakeState(*compatible_state).position == 3,
         "restart restore reconstructs exact model-owned state");

  const FakeRunner incompatible_reader("artifact-B");
  auto incompatible_state = incompatible_reader.CreateState();
  Expect(!RestoreTokens(restarted, incompatible_reader, *incompatible_state,
                        {1, 2, 3, 4})
              .restored,
         "different compatibility identity is a deterministic cache miss");
}

void TestLongestPrefixAndForcedHashCollision() {
  TemporaryDirectory directory;
  const FakeRunner runner("collision-model");
  const auto forced_collision = [](std::span<const std::uint8_t>) {
    return std::string(64, '0');
  };
  ContinuationDiskStore store(StoreOptions(directory.path()), {},
                              forced_collision);

  auto short_snapshot = MakeSnapshot(runner, 11, 2);
  auto long_snapshot = MakeSnapshot(runner, 22, 4);
  auto unrelated_snapshot = MakeSnapshot(runner, 33, 3);
  Expect(SaveTokens(store, runner, {1, 2}, *short_snapshot).stored,
         "collision bucket stores first key");
  Expect(SaveTokens(store, runner, {1, 2, 3, 4}, *long_snapshot).stored,
         "collision bucket stores longer exact key");
  Expect(SaveTokens(store, runner, {9, 8, 7}, *unrelated_snapshot).stored,
         "collision bucket stores unrelated exact key");

  auto state = runner.CreateState();
  const auto restored = RestoreTokens(store, runner, *state, {1, 2, 3, 4, 5});
  Expect(restored.restored && restored.token_count == 4 &&
             RequireFakeState(*state).value == 22,
         "full token verification selects longest exact key after collision");

  auto unrelated_state = runner.CreateState();
  Expect(
      RestoreTokens(store, runner, *unrelated_state, {9, 8, 7, 6}).restored &&
          RequireFakeState(*unrelated_state).value == 33,
      "forced collision cannot return another prompt's state");

  auto miss_state = runner.CreateState();
  Expect(!RestoreTokens(store, runner, *miss_state, {5, 5, 5}).restored,
         "forced collision without exact tokens remains a miss");
}

void TestCorruptionBecomesDeterministicMissAndRemoval() {
  TemporaryDirectory directory;
  const FakeRunner runner("corruption-model");
  std::vector<ContinuationDiskEvent> events;
  ContinuationDiskStore store(
      StoreOptions(directory.path()),
      [&](const ContinuationDiskEvent& event) { events.push_back(event); });
  auto snapshot = MakeSnapshot(runner, 44, 3);
  Expect(SaveTokens(store, runner, {4, 4, 4}, *snapshot).stored,
         "corruption test stores an entry");
  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 1, "one published cache file exists");

  std::fstream stream(files.front(),
                      std::ios::binary | std::ios::in | std::ios::out);
  stream.seekg(-1, std::ios::end);
  char value = 0;
  stream.read(&value, 1);
  value ^= static_cast<char>(0x5A);
  stream.seekp(-1, std::ios::end);
  stream.write(&value, 1);
  stream.close();

  if (::geteuid() != 0) {
    Expect(::chmod(directory.path().c_str(), S_IRUSR | S_IXUSR) == 0,
           "cache directory can simulate an unlink failure");
    auto lookup = std::async(std::launch::async, [&] {
      auto state = runner.CreateState();
      return RestoreTokens(store, runner, *state, {4, 4, 4, 5});
    });
    const bool returned =
        lookup.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    Expect(::chmod(directory.path().c_str(), S_IRWXU) == 0,
           "cache directory permissions restored");
    const auto result = lookup.get();
    Expect(returned && !result.restored && store.entry_count() == 1,
           "unremovable corrupt entries miss without an infinite retry");
  }

  auto state = runner.CreateState();
  Expect(!RestoreTokens(store, runner, *state, {4, 4, 4, 5}).restored,
         "checksum-invalid entry becomes a cache miss");
  Expect(store.entry_count() == 0 && CacheFiles(directory.path()).empty(),
         "checksum-invalid entry is removed from index and disk");
  Expect(std::ranges::any_of(
             events,
             [](const ContinuationDiskEvent& event) {
               return event.reason ==
                      ContinuationDiskEventReason::kChecksumMismatch;
             }),
         "sanitized event reports checksum removal reason");
}

void TestByteAndStagingLimits() {
  TemporaryDirectory staging_directory;
  const FakeRunner runner("limits");
  const std::size_t file_bytes = ExpectedFileBytes(6, 2);
  auto snapshot = MakeSnapshot(runner, 1, 2);

  ContinuationDiskStore staging_limited(
      StoreOptions(staging_directory.path(), 4096, file_bytes - 1));
  Expect(!SaveTokens(staging_limited, runner, {1, 2}, *snapshot).stored &&
             staging_limited.entry_count() == 0,
         "full file image must fit the bounded staging allocation");

  TemporaryDirectory byte_directory;
  ContinuationDiskStore byte_limited(
      StoreOptions(byte_directory.path(), file_bytes - 1, 4096));
  Expect(!SaveTokens(byte_limited, runner, {1, 2}, *snapshot).stored &&
             byte_limited.entry_count() == 0,
         "single entry larger than total disk budget is skipped");
}

void TestAutomaticStagingAndAdmissionDiagnostics() {
  TemporaryDirectory default_directory;
  ContinuationDiskStore defaults({.directory = default_directory.path()});
  Expect(defaults.capacity_bytes() == std::size_t{8} * 1024U * 1024U * 1024U &&
             defaults.staging_capacity_bytes() > kDiskHeaderBytes &&
             defaults.staging_capacity_bytes() <= 1024U * 1024U * 1024U,
         "defaults bound RAM staging to 1 GiB independently of disk retention");

  TemporaryDirectory directory;
  std::vector<ContinuationDiskEvent> events;
  const FakeRunner runner("automatic");
  ContinuationDiskStore store(
      {.directory = directory.path(), .capacity_bytes = 1024U * 1024U},
      [&](const auto& event) { events.push_back(event); });
  const auto limit = store.staging_capacity_bytes();
  Expect(limit > kDiskHeaderBytes && limit <= store.capacity_bytes(),
         "automatic staging resolves to a usable bounded RAM budget");
  const auto payload = limit / 2;
  auto first = store.ReserveCapture(runner, 2, payload);
  Expect(first != nullptr, "automatic staging admits a snapshot");
  Expect(!store.CanSave(runner, 2, payload) &&
             !store.ReserveCapture(runner, 2, payload),
         "automatic staging still accounts for all in-flight captures");
  Expect(!events.empty() &&
             events.back().reason ==
                 ContinuationDiskEventReason::kStagingCapacity &&
             events.back().staging_capacity_bytes == limit &&
             events.back().staging_used_bytes > payload &&
             events.back().payload_bytes == payload,
         "preflight rejection reports staging usage and required size");
  first.reset();
  Expect(store.CanSave(runner, 2, payload),
         "released staging is reusable without allocating the whole budget");

  Expect(!store.CanSave(runner, 2, store.capacity_bytes()) &&
             events.back().reason == ContinuationDiskEventReason::kByteCapacity,
         "disk retention rejection is distinguished from staging pressure");

  TemporaryDirectory explicit_directory;
  ContinuationDiskStore explicit_limit(
      StoreOptions(explicit_directory.path(), 4096, 256),
      [&](const auto& event) { events.push_back(event); });
  Expect(explicit_limit.staging_capacity_bytes() == 256 &&
             !explicit_limit.CanSave(runner, 2, 512) &&
             events.back().reason ==
                 ContinuationDiskEventReason::kStagingCapacity &&
             events.back().staging_used_bytes == 0,
         "an explicit too-small staging limit remains enforced and visible");
}

void TestSmallerStagingPreservesExistingFiles() {
  TemporaryDirectory directory;
  const FakeRunner runner("staging-restart");
  const auto file_bytes = ExpectedFileBytes(15, 2);
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(
        SaveTokens(store, runner, {1, 2}, *MakeSnapshot(runner, 42, 2)).stored,
        "restart fixture stores");
  }
  {
    std::vector<ContinuationDiskEvent> events;
    ContinuationDiskStore store(
        StoreOptions(directory.path(), 4096, file_bytes - 1),
        [&](const auto& event) { events.push_back(event); });
    Expect(store.entry_count() == 1 && store.retained_bytes() == file_bytes &&
               CacheFiles(directory.path()).size() == 1,
           "a smaller RAM budget preserves and accounts for existing files");
    auto state = runner.CreateState();
    Expect(!RestoreTokens(store, runner, *state, {1, 2, 3}).restored,
           "unverified oversized files are not exposed to lookup");
    Expect(std::ranges::any_of(
               events,
               [&](const auto& event) {
                 return event.action == ContinuationDiskEventAction::kSkipped &&
                        event.reason ==
                            ContinuationDiskEventReason::kStagingCapacity &&
                        event.file_bytes == file_bytes;
               }),
           "oversized startup files report a staging skip, not corruption");
  }
  {
    ContinuationDiskStore store(
        {.directory = directory.path(), .capacity_bytes = 4096});
    auto state = runner.CreateState();
    Expect(RestoreTokens(store, runner, *state, {1, 2, 3}).restored &&
               RequireFakeState(*state).value == 42,
           "automatic staging validates and restores the preserved file");
  }
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path(), file_bytes - 1, file_bytes - 1));
    Expect(store.retained_bytes() == 0 && CacheFiles(directory.path()).empty(),
           "unindexed files remain subject to the disk retention budget");
  }
}

void TestLruEvictionUsesActualFileBytes() {
  TemporaryDirectory directory;
  const FakeRunner runner("12345678");
  const std::size_t entry_bytes = ExpectedFileBytes(8, 2);
  ContinuationDiskStore store(
      StoreOptions(directory.path(), entry_bytes * 2, 4096));

  auto first = MakeSnapshot(runner, 1, 2);
  auto second = MakeSnapshot(runner, 2, 2);
  auto third = MakeSnapshot(runner, 3, 2);
  Expect(SaveTokens(store, runner, {1, 1}, *first).file_bytes == entry_bytes,
         "store reports exact first file size");
  Expect(SaveTokens(store, runner, {2, 2}, *second).stored,
         "second entry fills exact two-entry budget");

  auto touched = runner.CreateState();
  Expect(RestoreTokens(store, runner, *touched, {1, 1, 9}).restored,
         "restore refreshes first entry's LRU position");
  Expect(SaveTokens(store, runner, {3, 3}, *third).stored,
         "third entry evicts one least-recently-used file");
  Expect(store.entry_count() == 2 && store.retained_bytes() == entry_bytes * 2,
         "retained bytes never exceed configured total budget");

  auto first_state = runner.CreateState();
  auto second_state = runner.CreateState();
  auto third_state = runner.CreateState();
  Expect(RestoreTokens(store, runner, *first_state, {1, 1, 8}).restored,
         "recently touched entry survives LRU pressure");
  Expect(!RestoreTokens(store, runner, *second_state, {2, 2, 8}).restored,
         "oldest entry is the deterministic LRU victim");
  Expect(RestoreTokens(store, runner, *third_state, {3, 3, 8}).restored,
         "new entry survives LRU pressure");
}

void TestAtomicPublicationAndPrivatePermissions() {
  TemporaryDirectory directory;
  const auto orphan = directory.path() / ".tmp-orphan";
  {
    std::ofstream output(orphan, std::ios::binary);
    output << "partial";
  }

  const FakeRunner runner("permissions");
  ContinuationDiskStore store(StoreOptions(directory.path()));
  Expect(!std::filesystem::exists(orphan),
         "startup removes an interrupted private temporary file");
  auto snapshot = MakeSnapshot(runner, 77, 2);
  Expect(SaveTokens(store, runner, {7, 7}, *snapshot).stored,
         "permission test stores one entry");

  const auto files = CacheFiles(directory.path());
  Expect(files.size() == 1, "atomic publication exposes one final file");
  for (const auto& entry :
       std::filesystem::directory_iterator(directory.path())) {
    Expect(!entry.path().filename().string().starts_with(".tmp-"),
           "successful publication leaves no temporary file");
  }
  struct stat status{};
  Expect(::lstat(files.front().c_str(), &status) == 0 &&
             S_ISREG(status.st_mode) &&
             (status.st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
             (status.st_mode & S_IRUSR) != 0 && (status.st_mode & S_IWUSR) != 0,
         "published cache file is private and owner read-write");
}

void TestStartupRejectsUnsafeAndInvalidFiles() {
  TemporaryDirectory directory;
  const FakeRunner runner("startup-valid");
  const FakeRunner incompatible("startup-incompatible");
  std::filesystem::path truncated_file;
  std::filesystem::path checksum_file;
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    const auto save_and_find_file =
        [&](const FakeRunner& entry_runner,
            std::initializer_list<TextRunnerToken> tokens,
            const TextRunnerSnapshot& snapshot) {
          const auto before = CacheFiles(directory.path());
          Expect(SaveTokens(store, entry_runner, tokens, snapshot).stored,
                 "startup fixture publishes one immutable entry");
          const auto after = CacheFiles(directory.path());
          const auto added = std::find_if(
              after.begin(), after.end(), [&](const auto& candidate) {
                return std::find(before.begin(), before.end(), candidate) ==
                       before.end();
              });
          Expect(added != after.end(),
                 "startup fixture identifies the newly published file");
          return *added;
        };
    auto first = MakeSnapshot(runner, 1, 2);
    auto second = MakeSnapshot(runner, 2, 2);
    auto third = MakeSnapshot(runner, 3, 2);
    auto other = MakeSnapshot(incompatible, 4, 2);
    truncated_file = save_and_find_file(runner, {1, 1}, *first);
    checksum_file = save_and_find_file(runner, {2, 2}, *second);
    (void)save_and_find_file(runner, {3, 3}, *third);
    (void)save_and_find_file(incompatible, {4, 4}, *other);
  }

  auto files = CacheFiles(directory.path());
  Expect(files.size() == 4, "startup fixture exposes four immutable files");
  std::filesystem::resize_file(truncated_file, 12);
  {
    std::fstream stream(checksum_file,
                        std::ios::binary | std::ios::in | std::ios::out);
    stream.seekg(-1, std::ios::end);
    char value = 0;
    stream.read(&value, 1);
    value ^= static_cast<char>(0x5A);
    stream.seekp(-1, std::ios::end);
    stream.write(&value, 1);
  }
  const auto target = directory.path() / "external-target";
  {
    std::ofstream output(target, std::ios::binary);
    output << "must survive";
  }
  const auto unsafe_link = directory.path() / "unsafe.kvc";
  std::filesystem::create_symlink(target.filename(), unsafe_link);

  std::vector<ContinuationDiskEvent> events;
  ContinuationDiskStore restarted(
      StoreOptions(directory.path()),
      [&](const ContinuationDiskEvent& event) { events.push_back(event); });
  Expect(
      restarted.entry_count() == 2 && CacheFiles(directory.path()).size() == 2,
      "startup indexes only checksum-valid regular entries");
  Expect(
      std::filesystem::exists(target) && !std::filesystem::exists(unsafe_link),
      "unsafe symlink is removed without following its target");
  Expect(
      std::ranges::any_of(events,
                          [](const ContinuationDiskEvent& event) {
                            return event.reason ==
                                   ContinuationDiskEventReason::kCorrupt;
                          }) &&
          std::ranges::any_of(
              events,
              [](const ContinuationDiskEvent& event) {
                return event.reason ==
                       ContinuationDiskEventReason::kChecksumMismatch;
              }) &&
          std::ranges::any_of(events,
                              [](const ContinuationDiskEvent& event) {
                                return event.reason ==
                                       ContinuationDiskEventReason::kUnsafeFile;
                              }),
      "startup reports sanitized invalid-file reasons");

  std::size_t compatible_hits = 0;
  for (const std::array<TextRunnerToken, 3> prompt :
       {std::array<TextRunnerToken, 3>{1, 1, 9},
        std::array<TextRunnerToken, 3>{2, 2, 9},
        std::array<TextRunnerToken, 3>{3, 3, 9}}) {
    auto state = runner.CreateState();
    compatible_hits +=
        restarted.RestoreLongestPrefix(runner, *state, prompt).restored ? 1 : 0;
  }
  Expect(compatible_hits == 1,
         "the remaining compatible startup entry restores exactly");
  auto incompatible_state = incompatible.CreateState();
  Expect(
      RestoreTokens(restarted, incompatible, *incompatible_state, {4, 4, 9})
          .restored,
      "a valid different compatibility identity remains independently usable");
}

void TestRootSymlinkIsRejected() {
  TemporaryDirectory directory;
  const auto target = directory.path() / "target";
  const auto link = directory.path() / "cache-link";
  std::filesystem::create_directory(target);
  std::filesystem::create_directory_symlink(target.filename(), link);
  bool rejected = false;
  try {
    ContinuationDiskStore store(StoreOptions(link));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "configured cache-directory symlinks are rejected");
}

void TestConcurrentCallersRemainBoundedAndExact() {
  TemporaryDirectory directory;
  const FakeRunner runner("concurrent");
  constexpr std::size_t kThreadCount = 8;
  ContinuationDiskStore store(
      StoreOptions(directory.path(), 1024U * 1024U, 4096));
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    workers.emplace_back([&, index] {
      auto snapshot = MakeSnapshot(runner, 100 + index, 2);
      const std::array<TextRunnerToken, 2> tokens = {
          static_cast<TextRunnerToken>(index),
          static_cast<TextRunnerToken>(index + 100)};
      Expect(store.Save(runner, tokens, *snapshot).stored,
             "concurrent save succeeds under serialized I/O bound");
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  Expect(store.entry_count() == kThreadCount,
         "all concurrent saves publish distinct exact entries");
  for (std::size_t index = 0; index < kThreadCount; ++index) {
    auto state = runner.CreateState();
    const std::array<TextRunnerToken, 3> prompt = {
        static_cast<TextRunnerToken>(index),
        static_cast<TextRunnerToken>(index + 100), 999};
    Expect(store.RestoreLongestPrefix(runner, *state, prompt).restored &&
               RequireFakeState(*state).value == 100 + index,
           "concurrent store preserves exact per-key payloads");
  }
}

void TestConcurrentStoreInstancesPublishSafely() {
  TemporaryDirectory directory;
  const FakeRunner runner("multi-process");
  {
    ContinuationDiskStore first(StoreOptions(directory.path()));
    ContinuationDiskStore second(StoreOptions(directory.path()));
    auto first_snapshot = MakeSnapshot(runner, 101, 2);
    auto second_snapshot = MakeSnapshot(runner, 202, 2);
    std::thread first_writer([&] {
      Expect(SaveTokens(first, runner, {1, 0}, *first_snapshot).stored,
             "first store instance publishes atomically");
    });
    std::thread second_writer([&] {
      Expect(SaveTokens(second, runner, {2, 0}, *second_snapshot).stored,
             "second store instance publishes atomically");
    });
    first_writer.join();
    second_writer.join();
  }

  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  Expect(restarted.entry_count() == 2,
         "restart indexes files from concurrent store instances");
  auto first_state = runner.CreateState();
  auto second_state = runner.CreateState();
  Expect(
      RestoreTokens(restarted, runner, *first_state, {1, 0, 9}).restored &&
          RequireFakeState(*first_state).value == 101 &&
          RestoreTokens(restarted, runner, *second_state, {2, 0, 9}).restored &&
          RequireFakeState(*second_state).value == 202,
      "concurrent publishers expose no partial or confused entry");
}

void TestSharedPrefixBoundariesAndExactDedup() {
  TemporaryDirectory directory;
  const FakeRunner runner("shared-prefix-model");
  ContinuationDiskStore store(StoreOptions(directory.path()), {});

  // Conversation A: shared system prefix {7, 7, 7} plus its own turn.
  auto first = MakeSnapshot(runner, 11, 5);
  Expect(SaveTokens(store, runner, {7, 7, 7, 1, 2}, *first).stored,
         "first conversation frontier is stored");

  // Conversation B shares the first three tokens only.
  const std::vector<TextRunnerToken> second_prompt = {7, 7, 7, 3, 4, 5};
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 1, 4) ==
             std::vector<std::size_t>{3},
         "common prefix with a stored entry becomes a boundary");
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 4, 4).empty(),
         "boundaries below the minimum length are dropped");
  const std::vector<TextRunnerToken> first_prompt = {7, 7, 7, 1, 2};
  Expect(store.SharedPrefixBoundaries(runner, first_prompt, 1, 4).empty(),
         "an exact stored prompt yields no boundary");

  // Persist the shared prefix; a third conversation restores it.
  auto shared = MakeSnapshot(runner, 77, 3);
  Expect(SaveTokens(store, runner, {7, 7, 7}, *shared).stored,
         "shared prefix snapshot is stored");
  Expect(store.SharedPrefixBoundaries(runner, second_prompt, 1, 4).empty(),
         "a stored shared prefix is no longer a boundary");
  auto state = runner.CreateState();
  const auto restored = RestoreTokens(store, runner, *state, {7, 7, 7, 9});
  Expect(restored.restored && restored.token_count == 3 &&
             RequireFakeState(*state).value == 77,
         "new conversation restores the shared prefix");

  // Saving the same tokens again touches instead of rewriting.
  const std::size_t retained_before = store.retained_bytes();
  const std::size_t entries_before = store.entry_count();
  auto duplicate = MakeSnapshot(runner, 78, 3);
  Expect(!SaveTokens(store, runner, {7, 7, 7}, *duplicate).stored,
         "exact duplicate save is skipped");
  Expect(store.retained_bytes() == retained_before &&
             store.entry_count() == entries_before,
         "exact duplicate save leaves the store unchanged");
  Expect(store.Touch(runner, std::vector<TextRunnerToken>{7, 7, 7}),
         "touch finds the exact entry");
  Expect(!store.Touch(runner, std::vector<TextRunnerToken>{7, 7}),
         "touch misses a non-stored prefix");

  // Longest boundaries win when more than max_boundaries are shared.
  auto deeper = MakeSnapshot(runner, 12, 5);
  Expect(SaveTokens(store, runner, {7, 7, 7, 3, 8}, *deeper).stored,
         "deeper entry is stored");
  const std::vector<TextRunnerToken> third_prompt = {7, 7, 7, 3, 4, 6};
  Expect(store.SharedPrefixBoundaries(runner, third_prompt, 1, 4) ==
             std::vector<std::size_t>{4},
         "only the unstored shared length is reported");
  Expect(store.SharedPrefixBoundaries(runner, third_prompt, 1, 1) ==
             std::vector<std::size_t>{4},
         "capping keeps the longest boundary");
}

void TestBoundedAsyncPersistenceDoesNotBlockLookup() {
  TemporaryDirectory directory;
  auto runner = std::make_shared<FakeRunner>("async-store");
  std::binary_semaphore entered(0), release(0);
  const auto bytes = ExpectedFileBytes(
      runner->Descriptor().persistence->compatibility_identity.size(), 2);
  {
    ContinuationDiskStore store(
        StoreOptions(directory.path(), 4096, bytes + sizeof(FakeSnapshot)));
    Expect(store
               .Save(*runner, std::vector<TextRunnerToken>{7, 8},
                     *MakeSnapshot(*runner, 99, 2))
               .stored,
           "an existing entry is available before the new write");
    runner->before_stream = [&] {
      entered.release();
      release.acquire();
    };
    auto snapshot = MakeSnapshot(*runner, 42, 2);
    Expect(store.CanSave(*runner, 2, snapshot->PayloadBytes()),
           "empty persistence queue admits the snapshot before capture");
    auto capture = store.ReserveCapture(*runner, 2, snapshot->PayloadBytes());
    Expect(capture != nullptr &&
               !store.CanSave(*runner, 2, snapshot->PayloadBytes()) &&
               !store.ReserveCapture(*runner, 2, snapshot->PayloadBytes()),
           "in-flight captures reserve staging before any allocation");
    capture.reset();
    Expect(store.CanSave(*runner, 2, snapshot->PayloadBytes()),
           "discarded captures release their reservation");
    capture = store.ReserveCapture(*runner, 2, snapshot->PayloadBytes());
    Expect(store.SaveAsync(runner, {1, 2}, std::move(snapshot), {},
                           std::move(capture)) == bytes,
           "immutable snapshot is queued without waiting for serialization");
    const bool started = entered.try_acquire_for(std::chrono::seconds(2));
    Expect(started, "persistence worker starts independently");
    Expect(
        !store.CanSave(*runner, 2, sizeof(FakeSnapshot)) &&
            store.SaveAsync(runner, {3, 4}, MakeSnapshot(*runner, 43, 2)) == 0,
        "active writes count against the bounded queue");
    auto lookup = std::async(std::launch::async, [&] {
      auto state = runner->CreateState();
      return RestoreTokens(store, *runner, *state, {7, 8, 9}).restored &&
             RequireFakeState(*state).value == 99 &&
             store.Touch(*runner, std::vector<TextRunnerToken>{7, 8}) &&
             !RestoreTokens(store, *runner, *state, {1, 2, 9}).restored &&
             store
                 .SharedPrefixBoundaries(
                     *runner, std::vector<TextRunnerToken>{1, 9}, 1, 4)
                 .empty() &&
             !store.Touch(*runner, std::vector<TextRunnerToken>{1, 2});
    });
    const bool ready =
        lookup.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    release.release();
    Expect(ready && lookup.get(),
           "existing entries remain readable while a writer is blocked");
    store.Flush();
    runner->before_stream = {};
    auto state = runner->CreateState();
    Expect(RestoreTokens(store, *runner, *state, {1, 2, 9}).restored &&
               RequireFakeState(*state).value == 42,
           "streamed chunks form a checksummed restorable snapshot");
    Expect(store.SaveAsync(runner, {3, 4}, MakeSnapshot(*runner, 43, 2)) != 0,
           "completed writes release queue admission");
  }
  ContinuationDiskStore restarted(StoreOptions(directory.path()));
  Expect(restarted.entry_count() == 3,
         "shutdown drains accepted persistence jobs");
}

void TestIndexedPrefixLookup() {
  TemporaryDirectory directory;
  FakeRunner runner("prefix-index");
  ContinuationDiskStore store(StoreOptions(directory.path()));
  // Many independent branches, with terminal prefixes inside compressed edges.
  for (TextRunnerToken branch = 0; branch < 96; ++branch) {
    Expect(SaveTokens(store, runner, {7, branch, 8, 9},
                      *MakeSnapshot(runner, branch, 4))
               .stored,
           "indexed branch stores");
  }
  Expect(
      SaveTokens(store, runner, {7, 45}, *MakeSnapshot(runner, 999, 2)).stored,
      "shorter prefix splits an existing compressed edge");
  auto state = runner.CreateState();
  Expect(RestoreTokens(store, runner, *state, {7, 45, 8, 9, 10}).token_count ==
                 4 &&
             RequireFakeState(*state).value == 45,
         "longest terminal on the matching path wins");
  Expect(RestoreTokens(store, runner, *state, {7, 45, 10}).token_count == 2 &&
             RequireFakeState(*state).value == 999,
         "a mismatched edge falls back to its terminal ancestor");
  Expect(store.SharedPrefixBoundaries(
             runner, std::vector<TextRunnerToken>{7, 46, 8, 10}, 1, 4) ==
             std::vector<std::size_t>({1, 3}),
         "index returns branch and partial-edge boundaries in prompt order");
}

}  // namespace

void TestImageIdentitySurvivesRestart() {
  TemporaryDirectory directory;
  FakeRunner runner("image-cache-test");
  const std::vector<TextRunnerToken> prompt{1, 248056, 3};
  const std::vector<std::uint8_t> image_a{1, 2}, image_b{1, 3};
  {
    ContinuationDiskStore store(StoreOptions(directory.path()));
    Expect(store
               .Save(runner, prompt, *MakeSnapshot(runner, 10, prompt.size()),
                     image_a)
               .stored,
           "first image is persisted");
    Expect(store
               .Save(runner, prompt, *MakeSnapshot(runner, 20, prompt.size()),
                     image_b)
               .stored,
           "second image has an independent disk entry");
  }
  ContinuationDiskStore store(StoreOptions(directory.path()));
  auto state = runner.CreateState();
  Expect(store.RestoreLongestPrefix(runner, *state, prompt, image_a).restored &&
             RequireFakeState(*state).value == 10,
         "restart restores first image exactly");
  Expect(store.RestoreLongestPrefix(runner, *state, prompt, image_b).restored &&
             RequireFakeState(*state).value == 20,
         "restart restores second image exactly");
  Expect(!store.RestoreLongestPrefix(runner, *state, prompt).restored,
         "text cannot restore a disk image continuation");
  Expect(!store.Touch(runner, prompt), "disk dedup respects image identity");
}

int main() {
  TestBoundedAsyncPersistenceDoesNotBlockLookup();
  TestIndexedPrefixLookup();
  TestImageIdentitySurvivesRestart();
  TestSharedPrefixBoundariesAndExactDedup();
  TestSha256KnownVector();
  TestRestartRestoreAndCompatibilityIdentity();
  TestLongestPrefixAndForcedHashCollision();
  TestCorruptionBecomesDeterministicMissAndRemoval();
  TestByteAndStagingLimits();
  TestAutomaticStagingAndAdmissionDiagnostics();
  TestSmallerStagingPreservesExistingFiles();
  TestLruEvictionUsesActualFileBytes();
  TestAtomicPublicationAndPrivatePermissions();
  TestStartupRejectsUnsafeAndInvalidFiles();
  TestRootSymlinkIsRejected();
  TestConcurrentCallersRemainBoundedAndExact();
  TestConcurrentStoreInstancesPublishSafely();
  std::cout << "continuation disk store tests passed\n";
  return 0;
}
