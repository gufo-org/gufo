#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <iostream>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/core/logging/logging.hpp"
#include "src/families/qwen/vision/device_input.hpp"
#include "tests/families/qwen/vision_memory_fixture.hpp"

namespace vision = gufo::models::qwen::vision;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template<class Error = std::runtime_error, class Function>
void Refused(Function&& function) {
  try {
    function();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error("expected refusal");
}
}  // namespace

namespace fake {
enum class AllocationKind { kDynamic, kTextBaseDescriptor };
struct Allocation {
  std::size_t bytes;
  std::shared_ptr<vision::AdmissionBudget> owner;
  AllocationKind kind;
};
std::mutex mutex;
std::unordered_map<void*, Allocation> allocations;
std::unordered_map<std::shared_ptr<vision::AdmissionBudget>, std::size_t> peaks;
// Binding selects the owner only at allocation. Every allocation remembers
// that owner for later checks/frees, even on a different thread or binding.
thread_local std::weak_ptr<vision::AdmissionBudget> budget;
std::atomic<std::size_t> encodes{0};
// Fail one selected API call after n successful calls of the same kind.
thread_local std::optional<std::size_t> fail_malloc, fail_copy, fail_sync;
thread_local bool fail_encode_after_allocation{false};
// Keep source pointers live until synchronization, including failed
// submissions. The saved bytes let the completion check that its sources are
// unchanged.
struct PendingCopy {
  void* destination;
  const void* source;
  std::vector<std::byte> expected;
};
thread_local bool delay_copies{};
thread_local std::vector<PendingCopy> pending_copies;
thread_local std::size_t synchronized_copies{};

bool Fail(std::optional<std::size_t>& remaining) {
  if (!remaining)
    return false;
  if (*remaining) {
    --*remaining;
    return false;
  }
  remaining.reset();
  return true;
}
std::size_t Bytes() {
  const std::lock_guard lock(mutex);
  std::size_t bytes = 0;
  for (const auto& [pointer, allocation] : allocations)
    bytes += allocation.bytes;
  return bytes;
}
// Caller holds mutex, including through allocation registration/removal, so
// simultaneous allocations cannot both inspect the same old aggregate.
std::size_t LiveBytes(const std::shared_ptr<vision::AdmissionBudget>& owner,
                      AllocationKind kind) {
  std::size_t bytes = 0;
  for (const auto& [pointer, allocation] : allocations)
    if (allocation.owner == owner && allocation.kind == kind)
      bytes = vision::CheckedAdd(bytes, allocation.bytes);
  return bytes;
}
void CheckClaim(const std::shared_ptr<vision::AdmissionBudget>& owner,
                std::size_t additional = 0) {
  if (owner)
    Require(owner->Outstanding() >=
                vision::CheckedAdd(LiveBytes(owner, AllocationKind::kDynamic),
                                   additional),
            "aggregate live dynamic allocations exceed owning budget claim");
}
std::size_t DynamicBytes(
    const std::shared_ptr<vision::AdmissionBudget>& owner) {
  const std::lock_guard lock(mutex);
  return LiveBytes(owner, AllocationKind::kDynamic);
}
std::size_t DescriptorBytes(
    const std::shared_ptr<vision::AdmissionBudget>& owner) {
  const std::lock_guard lock(mutex);
  return LiveBytes(owner, AllocationKind::kTextBaseDescriptor);
}
std::size_t PeakBytes(const std::shared_ptr<vision::AdmissionBudget>& owner) {
  const std::lock_guard lock(mutex);
  return peaks[owner];
}
void TrackHost(void* pointer, std::size_t bytes,
               const std::shared_ptr<vision::AdmissionBudget>& owner) {
  const std::lock_guard lock(mutex);
  CheckClaim(owner, bytes);
  Require(
      allocations
          .emplace(pointer, Allocation{bytes, owner, AllocationKind::kDynamic})
          .second,
      "prepared host allocation is registered once");
  peaks[owner] =
      std::max(peaks[owner], LiveBytes(owner, AllocationKind::kDynamic));
}
void UntrackHost(void* pointer) {
  const std::lock_guard lock(mutex);
  const auto found = allocations.find(pointer);
  Require(found != allocations.end(), "prepared host allocation is owned");
  CheckClaim(found->second.owner);
  allocations.erase(found);
}
hipError_t Allocate(void** pointer, std::size_t bytes, AllocationKind kind) {
  if (Fail(fail_malloc))
    return hipErrorMemoryAllocation;
  const auto owner = budget.lock();
  const std::lock_guard lock(mutex);
  CheckClaim(owner, kind == AllocationKind::kDynamic ? bytes : 0);
  auto storage = std::make_unique<std::byte[]>(bytes);
  allocations.emplace(storage.get(), Allocation{bytes, owner, kind});
  peaks[owner] =
      std::max(peaks[owner], LiveBytes(owner, AllocationKind::kDynamic));
  *pointer = storage.release();
  return hipSuccess;
}
}  // namespace fake

hipError_t hipMalloc(void** pointer, std::size_t bytes) {
  return fake::Allocate(pointer, bytes, fake::AllocationKind::kDynamic);
}
hipError_t hipMalloc(vision::DeviceRope** pointer, std::size_t bytes) {
  Require(bytes == sizeof(vision::DeviceRope), "invalid descriptor size");
  return fake::Allocate(reinterpret_cast<void**>(pointer), bytes,
                        fake::AllocationKind::kTextBaseDescriptor);
}
hipError_t hipFree(void* pointer) {
  if (!pointer)
    return hipSuccess;
  const std::lock_guard lock(fake::mutex);
  const auto found = fake::allocations.find(pointer);
  Require(found != fake::allocations.end(), "double or unowned device free");
  // Validate before removal, not after: releasing a reservation while buffers
  // remain live must fail even when this free would bring usage below it.
  fake::CheckClaim(found->second.owner);
  delete[] static_cast<std::byte*>(pointer);
  fake::allocations.erase(found);
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* destination, const void* source,
                          std::size_t bytes, hipMemcpyKind, hipStream_t) {
  // Model the difficult case: the descriptor may have changed even when the
  // copy reports failure. Recovery cannot trust only the previous host layout.
  if (fake::delay_copies) {
    const auto* data = static_cast<const std::byte*>(source);
    fake::pending_copies.push_back({destination, source, {data, data + bytes}});
  } else {
    std::memcpy(destination, source, bytes);
  }
  return fake::Fail(fake::fail_copy) ? hipErrorUnknown : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  for (const auto& copy : fake::pending_copies) {
    Require(
        std::memcmp(copy.source, copy.expected.data(), copy.expected.size()) ==
            0,
        "asynchronous copy retains its unchanged host source until completion");
    std::memcpy(copy.destination, copy.source, copy.expected.size());
    ++fake::synchronized_copies;
  }
  fake::pending_copies.clear();
  return fake::Fail(fake::fail_sync) ? hipErrorUnknown : hipSuccess;
}
const char* hipGetErrorString(hipError_t error) {
  return error == hipErrorMemoryAllocation ? "injected allocation failure"
                                           : "injected copy/sync failure";
}

namespace fake {
struct DeviceStorage {
  explicit DeviceStorage(std::size_t bytes) {
    if (hipMalloc(&data, bytes) != hipSuccess)
      throw std::bad_alloc();
  }
  ~DeviceStorage() { (void)hipFree(data); }
  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;
  void* data{nullptr};
};

// Independent fixture for the encoder's explicit allocation lifetimes. Small
// 32/64-pixel inputs keep real storage below 1 MiB. These simulate scratch and
// output coexistence; they do not compile or execute the real HIP encoder.
std::array<std::size_t, 9> EncoderScratchSizes(const gufo::core::Image& image) {
  const std::size_t patches = (image.width / 16) * (image.height / 16);
  const auto tile = std::min<std::size_t>(patches, patches == 4096 ? 64 : 128);
  return {image.pixels.size(),     patches * 1152 * 4,      patches * 4304 * 4,
          patches * 4304 * 2,      patches * 3 * 1152 * 2,  patches * 1152 * 2,
          16 * tile * patches * 4, 16 * tile * patches * 2, tile * 1152 * 4};
}
std::size_t EncoderScratchBytes(const gufo::core::Image& image) {
  std::size_t bytes = 0;
  for (const auto size : EncoderScratchSizes(image))
    bytes += size;
  return bytes;
}
}  // namespace fake

namespace gufo::models::qwen::vision {
// The real DeviceInput .cpp, including Inject's ownership loop, is compiled
// unchanged. Only the weight-dependent encoder and row-copy kernel are fake.
struct Encoder::Impl {
  std::uint32_t width;
};
Encoder::Encoder(const std::filesystem::path&, std::uint32_t width)
    : impl_(std::make_unique<Impl>(Impl{width})) {}
Encoder::~Encoder() = default;
std::uint32_t Encoder::OutputWidth() const noexcept {
  return impl_->width;
}
Encoder::Embedding::~Embedding() {
  (void)hipFree(data_);
}
std::shared_ptr<const Encoder::Embedding> Encoder::Encode(
    const core::Image& image, const Observer&,
    const CancellationCheck& cancelled) {
  ++fake::encodes;
  if (cancelled && cancelled())
    throw std::runtime_error("injected encoder cancellation");
  std::vector<std::unique_ptr<fake::DeviceStorage>> scratch;
  for (const auto bytes : fake::EncoderScratchSizes(image))
    scratch.push_back(std::make_unique<fake::DeviceStorage>(bytes));
  {
    // Pinned upload uses the same unified-memory budget but ends before the
    // output embedding is allocated. All nine device scratch buffers survive.
    fake::DeviceStorage pinned(
        std::min<std::size_t>(image.pixels.size(), 16U << 20));
  }
  auto result = std::shared_ptr<Embedding>(new Embedding);
  result->rows_ = image.width / 32 * (image.height / 32);
  result->width_ = impl_->width;
  const auto bytes =
      std::size_t{result->rows_} * result->width_ * sizeof(float);
  if (hipMalloc(&result->data_, bytes) != hipSuccess)
    throw std::bad_alloc();
  std::fill_n(result->data_, bytes / sizeof(float), 7.0F);
  if (fake::fail_encode_after_allocation) {
    fake::fail_encode_after_allocation = false;
    throw std::bad_alloc();
  }
  return result;
}
void DeviceInput::LaunchRows(float* hidden, const float* embedding,
                             std::uint32_t rows, std::uint32_t width,
                             std::uint32_t hc, hipStream_t) {
  for (std::size_t i = 0; i < std::size_t{rows} * width * hc; ++i)
    hidden[i] = embedding[(i / (width * hc)) * width + i % width];
}
}  // namespace gufo::models::qwen::vision

namespace {
constexpr std::uint32_t kWidth = 2560;
std::shared_ptr<vision::AdmissionBudget> RuntimeBudget(
    std::size_t capacity, std::size_t execution = 0) {
  auto owner = std::make_shared<std::weak_ptr<vision::AdmissionBudget>>();
  auto budget = std::make_shared<vision::AdmissionBudget>(
      [owner, capacity] {
        const auto committed = fake::DynamicBytes(owner->lock());
        return committed < capacity ? capacity - committed : 0;
      },
      execution);
  *owner = budget;
  return budget;
}
std::shared_ptr<vision::Prompt> Prompt(std::uint32_t offset = 2,
                                       std::uint32_t side = 32) {
  auto image = std::make_shared<gufo::core::Image>();
  image->width = side;
  image->height = side;
  image->pixels.resize(std::size_t{side} * side * 3);
  auto prompt = std::make_shared<vision::Prompt>();
  const vision::ImageGrid grid{offset, side / 32, side / 32};
  prompt->images.push_back({std::move(image), grid, {}});
  prompt->rope.images.push_back(grid);
  prompt->cache_identity = {static_cast<std::uint8_t>(offset),
                            static_cast<std::uint8_t>(side)};
  return prompt;
}
std::shared_ptr<vision::Prompt> AdmittedPrompt(
    const std::shared_ptr<vision::AdmissionBudget>& budget,
    std::uint32_t offset = 2, std::uint32_t side = 32) {
  // The production preparation owner reserves before decoding and aliases the
  // prepared image pointers. Use tiny already-sized pixels here and track their
  // real allocation alongside the simulated device allocations in that same
  // budget.
  struct Owner {
    gufo::cache::ResourceReservation admission;
    gufo::cache::ResourceCharge committed;
    std::shared_ptr<const gufo::core::Image> pixels;
    Owner(const std::shared_ptr<vision::AdmissionBudget>& budget,
          std::shared_ptr<const gufo::core::Image> image)
        : admission(budget->Reserve(image->pixels.capacity())),
          pixels(std::move(image)) {
      fake::TrackHost(const_cast<std::uint8_t*>(pixels->pixels.data()),
                      pixels->pixels.capacity(), budget);
      committed = admission.Convert();
    }
    ~Owner() {
      Require(pixels.use_count() == 1,
              "last image alias frees pixels before releasing its reservation");
      fake::UntrackHost(const_cast<std::uint8_t*>(pixels->pixels.data()));
    }
  };
  auto prompt = Prompt(offset, side);
  auto owner = std::make_shared<Owner>(budget, prompt->images.front().pixels);
  prompt->RetainPixels(owner, budget);
  return prompt;
}
std::size_t Peak(const vision::Prompt& prompt) {
  std::vector<vision::ImageDimensions> images;
  for (const auto& image : prompt.images)
    images.push_back({image.pixels->width, image.pixels->height});
  return vision::testing::RequestMemoryBound(prompt.rope.PrefixLength(), images,
                                             kWidth);
}
void Inject(vision::DeviceInput& input, std::uint32_t offset,
            std::uint32_t rows = 1) {
  std::vector<float> hidden(std::size_t{rows} * kWidth);
  input.Inject(hidden.data(), offset, rows, kWidth, 1, nullptr);
  Require(hidden.front() == 7.0F, "fake row launcher received encoded image");
}
void EmptyDescriptor(const vision::DeviceInput& input) {
  Require(input.rope() && input.rope()->prefix_length == 0 &&
              input.rope()->delta == 0,
          "text transition repairs and empties stable device descriptor");
  Require(input.Bytes() == sizeof(vision::DeviceRope),
          "text transition releases every dynamic device buffer");
}

void AggregateCoverageRegression() {
  auto owner = RuntimeBudget(128);
  auto other = RuntimeBudget(1024);
  fake::budget = owner;
  vision::AdmissionLease claim(owner);
  claim.Grow(96);
  void* first = nullptr;
  void* replacement = nullptr;
  Require(hipMalloc(&first, 64) == hipSuccess, "initial dynamic allocation");
  // Each 64-byte allocation fits the 96-byte claim by itself. Both together
  // do not: this is the regression the old one-allocation check missed.
  Refused([&] { (void)hipMalloc(&replacement, 64); });
  Require(!replacement && fake::DynamicBytes(owner) == 64,
          "aggregate undercoverage refuses replacement before allocation");
  claim.SetCommitted(64);
  claim.Grow(128);
  Require(hipMalloc(&replacement, 64) == hipSuccess &&
              fake::PeakBytes(owner) == 128,
          "funded replacement includes old and new allocations at the peak");
  // Allocation ownership survives rebinding; another budget's empty claim must
  // not invalidate the original budget's frees.
  fake::budget = other;
  (void)hipFree(first);
  (void)hipFree(replacement);
  auto other_claim = other->Reserve(1024);
  fake::budget = owner;
  claim.Clear();
  vision::DeviceRope* descriptor = nullptr;
  Require(hipMalloc(&descriptor, sizeof(*descriptor)) == hipSuccess,
          "typed stable descriptor is funded by text base with no image claim");
  Require(fake::DescriptorBytes(owner) == sizeof(*descriptor) &&
              !fake::DynamicBytes(owner),
          "descriptor and dynamic allocation totals are explicitly separate");
  Refused([&] { (void)hipMalloc(&first, sizeof(*descriptor)); });
  Require(!fake::DynamicBytes(owner),
          "descriptor-sized dynamic allocation cannot borrow another budget");
  auto exact = owner->Reserve(sizeof(*descriptor));
  Require(hipMalloc(&first, sizeof(*descriptor)) == hipSuccess,
          "descriptor-sized dynamic allocation requires its own budget");
  exact = {};
  Refused([&] { (void)hipFree(first); });
  Require(fake::DynamicBytes(owner) == sizeof(*descriptor),
          "premature reservation release is caught before physical free");
  exact = owner->Reserve(sizeof(*descriptor));
  (void)hipFree(first);
  (void)hipFree(descriptor);
  Require(!fake::Bytes(), "negative coverage checks clean up every allocation");
}

void RetentionAndTextTransitions() {
  const auto prompt = Prompt();
  auto budget = RuntimeBudget(Peak(*prompt));
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.PrepareRope(nullptr);
    Require(fake::DescriptorBytes(budget) == sizeof(vision::DeviceRope) &&
                !fake::DynamicBytes(budget) && !budget->Outstanding(),
            "text descriptor remains outside dynamic image accounting");
    const auto* descriptor = input.rope();
    input.Configure(prompt, encoder, nullptr);
    Require(
        budget->Outstanding() == Peak(*prompt) - prompt->rope.PrefixLength() *
                                                     3 * sizeof(std::int32_t),
        "completed layout work is released while all encoder work stays "
        "funded");
    Inject(input, 2);
    Require(fake::PeakBytes(budget) ==
                input.Bytes() - sizeof(vision::DeviceRope) +
                    fake::EncoderScratchBytes(*prompt->images[0].pixels),
            "encoder scratch and output coexist with live position storage");
    const auto encodes = fake::encodes.load();
    const auto bytes = input.Bytes();
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
    Require(input.Bytes() == bytes && fake::encodes == encodes,
            "identical image history reuses its retained embedding");
    Require(input.rope() == descriptor,
            "image requests preserve graph descriptor address");
    Refused<gufo::core::ResourceAdmissionError>(
        [&] { input.Configure(Prompt(20, 64), encoder, nullptr); });
    Require(input.PromptAttachment() == prompt && input.Bytes() == bytes,
            "oversized replacement leaves existing image state intact");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "text frees image memory and reservation");
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
  }
  Require(!budget->Outstanding() && !fake::Bytes(),
          "session destruction frees buffers before reservations");
  fake::budget.reset();
  // Default mode retains high-water positions on text reuse.
  {
    vision::DeviceInput conservative;
    conservative.Configure(prompt, encoder, nullptr);
    conservative.Configure(nullptr, encoder, nullptr);
    Require(conservative.Bytes() > sizeof(vision::DeviceRope),
            "conservative default allocation retention is unchanged");
  }
  Require(!fake::Bytes(), "default session destruction releases buffers");
}

void GrowthAndRestoredHistory() {
  auto budget = RuntimeBudget(2U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto small = Prompt();
  const auto large = Prompt(2, 64);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(small, encoder, nullptr);
    Inject(input, 2);
    const auto initial = budget->Outstanding();
    input.Configure(large, encoder, nullptr);
    Require(budget->Outstanding() == Peak(*large) - large->rope.PrefixLength() *
                                                        3 *
                                                        sizeof(std::int32_t),
            "completed replacement releases old buffers and layout work while "
            "funding encoding");
    Inject(input, 2, 4);
    Require(budget->Outstanding() > initial,
            "growth charges replacement peak before releasing old buffers");
    Require(
        fake::PeakBytes(budget) ==
            fake::DynamicBytes(budget) +
                fake::EncoderScratchBytes(*large->images[0].pixels),
        "grown encoder scratch coexists with replacement output and positions");
    const auto* rope = input.rope();
    Require(
        rope->prefix_length == 6 && rope->delta == -2 &&
            rope->positions[12] == 2 && rope->positions[13] == 3 &&
            rope->positions[14] == 2 && rope->positions[15] == 2 &&
            rope->positions[16] == 3 && rope->positions[17] == 3,
        "production position upload preserves independent 2x2 mRoPE oracle");
    const auto retained = budget->Outstanding();
    input.RestoreLayout(large->rope, nullptr);
    Require(budget->Outstanding() == retained,
            "RAM reuse of the same layout retains the complete image claim");
    input.RestoreLayout({}, nullptr);
    Require(budget->Outstanding() >= retained && input.Bytes() > sizeof(*rope),
            "restoring an earlier frontier does not uncharge retained images");
    input.RestoreLayout(large->rope, nullptr);
    const auto encodes = fake::encodes.load();
    Inject(input, 2, 4);
    Require(fake::encodes == encodes,
            "restored history preserves already encoded image ownership");
    input.Configure(nullptr, encoder, nullptr);
  }
  // A fresh slot pays the complete attached history before a cache restores
  // positions. RAM/disk component restoration both call this production path.
  {
    vision::DeviceInput restored;
    restored.SetMemoryAdmission(budget);
    restored.Configure(large, encoder, nullptr);
    restored.RestoreLayout(large->rope, nullptr);
    Require(budget->Outstanding() == Peak(*large) - large->rope.PrefixLength() *
                                                        3 *
                                                        sizeof(std::int32_t),
            "fresh checkpoint attachment charges all future image work");
    Inject(restored, 2, 4);
  }
  Require(!fake::Bytes() && !budget->Outstanding(),
          "restored sessions leave neither allocations nor reservations");

  // Two history images retain the first output while the second is encoded.
  // The last encoding peak therefore contains positions, BOTH embeddings and
  // transient scratch, rather than only the largest individual allocation.
  const auto history = Prompt();
  const vision::ImageGrid second_grid{3, 1, 1};
  history->images.push_back({history->images.front().pixels, second_grid, {}});
  history->rope.images.push_back(second_grid);
  {
    auto history_budget = RuntimeBudget(Peak(*history));
    fake::budget = history_budget;
    vision::DeviceInput input;
    input.SetMemoryAdmission(history_budget);
    input.Configure(history, encoder, nullptr);
    Inject(input, 2, 2);
    Require(fake::PeakBytes(history_budget) ==
                fake::DynamicBytes(history_budget) +
                    fake::EncoderScratchBytes(*history->images[0].pixels),
            "image history retains both embeddings during transient encoding");
    input.Configure(nullptr, encoder, nullptr);
    Require(
        !history_budget->Outstanding() && !fake::DynamicBytes(history_budget),
        "history reset releases aggregate dynamic state and its claim");
  }
  Require(!fake::Bytes(), "history descriptor cleanup");
}

void AllocationAndCopyFailures() {
  auto budget = RuntimeBudget(2U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto prompt = Prompt();
  for (unsigned failure = 0; failure != 6; ++failure) {
    {
      vision::DeviceInput input;
      input.SetMemoryAdmission(budget);
      if (failure < 2)
        fake::fail_malloc = failure;
      else if (failure < 4)
        fake::fail_copy = failure - 2;
      else
        fake::fail_sync = failure - 4;
      Refused([&] { input.Configure(prompt, encoder, nullptr); });
      Require(budget->Outstanding() == Peak(*prompt),
              "partial allocation/copy failure keeps complete reservation");
      input.Configure(nullptr, encoder, nullptr);
      EmptyDescriptor(input);
      Require(
          !budget->Outstanding(),
          "failed descriptor/position stages recover to text without leaks");
      input.Configure(prompt, encoder, nullptr);
      Inject(input, 2);
    }
    Require(!fake::Bytes() && !budget->Outstanding(),
            "failed stage does not corrupt the following image request");
  }
  // Fail before scratch, partway through it, at pinned upload, at output, and
  // after output allocation. Every partial working set stays covered until
  // its individual allocations are really released during unwinding.
  for (const unsigned stage : {0U, 4U, 9U, 10U, 11U}) {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(prompt, encoder, nullptr);
    if (stage == 11)
      fake::fail_encode_after_allocation = true;
    else
      fake::fail_malloc = stage;
    Refused<std::bad_alloc>([&] { Inject(input, 2); });
    Require(
        budget->Outstanding() == Peak(*prompt) - prompt->rope.PrefixLength() *
                                                     3 * sizeof(std::int32_t),
        "encoder failure cannot release a live position or pending encoder "
        "claim");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "encoder failures recover to text");
  }
  Require(!fake::Bytes(), "encoder failure destroys temporary embeddings");
}

void FailedLayoutCopiesDrainSources() {
  auto budget = RuntimeBudget(2U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto prompt = Prompt();
  // Fail the positions submission, the following descriptor submission, or
  // final synchronization. Sources must survive every attempted copy.
  for (unsigned failure = 0; failure != 3; ++failure) {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    fake::delay_copies = true;
    fake::synchronized_copies = 0;
    if (failure < 2)
      fake::fail_copy = failure;
    else
      fake::fail_sync = 1;  // Initial growth drain succeeds.
    Refused([&] { input.Configure(prompt, encoder, nullptr); });
    Require(fake::pending_copies.empty() &&
                fake::synchronized_copies == (failure == 0 ? 1U : 2U),
            "failed layout drains submitted copies before local sources die");
    Require(budget->Outstanding() >= fake::DynamicBytes(budget),
            "failed asynchronous layout retains its allocation claim");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "drained failure can reset to text");
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
    input.Configure(nullptr, encoder, nullptr);
    fake::delay_copies = false;
  }
  for (const bool fail_submission : {false, true}) {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    fake::delay_copies = true;
    fake::synchronized_copies = 0;
    if (fail_submission)
      fake::fail_copy = 0;
    else
      fake::fail_sync = 0;
    Refused([&] { input.PrepareRope(nullptr); });
    Require(fake::pending_copies.empty() && fake::synchronized_copies == 1,
            "failed stable descriptor upload drains its stack source");
    input.Configure(prompt, encoder, nullptr);
    Inject(input, 2);
    input.Configure(nullptr, encoder, nullptr);
    fake::delay_copies = false;
  }
  Require(!fake::Bytes() && !budget->Outstanding(),
          "drained upload failures release allocations and reservations");
}

void Cancellation() {
  auto budget = RuntimeBudget(1U << 20);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto prompt = Prompt();
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.SetCancellationCheck([] { return true; });
    Refused([&] { input.Configure(prompt, encoder, nullptr); });
    Require(!budget->Outstanding() && !fake::Bytes(),
            "cancelled admission allocates no device state");
    input.SetCancellationCheck({});
    input.Configure(prompt, encoder, nullptr);
    input.SetCancellationCheck([] { return true; });
    Refused([&] { Inject(input, 2); });
    Require(
        budget->Outstanding() == Peak(*prompt) - prompt->rope.PrefixLength() *
                                                     3 * sizeof(std::int32_t),
        "cancellation during encoding retains existing positions and retry "
        "capacity");
    input.SetCancellationCheck({});
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
  }
  Require(!budget->Outstanding() && !fake::Bytes(),
          "cancelled request cleanup");
}

void ConcurrentInputs() {
  const auto prompt = Prompt();
  auto budget = RuntimeBudget(4096 + Peak(*prompt), 4096);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  std::barrier start(3);
  std::latch attempted(2), release(1);
  std::atomic<unsigned> admitted{0}, refused{0};
  auto request = [&] {
    fake::budget = budget;
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    start.arrive_and_wait();
    try {
      input.Configure(prompt, encoder, nullptr);
      ++admitted;
    } catch (const gufo::core::ResourceAdmissionError&) {
      ++refused;
    }
    attempted.count_down();
    release.wait();
  };
  std::thread first(request), second(request);
  start.arrive_and_wait();
  attempted.wait();
  const auto settled =
      Peak(*prompt) - prompt->rope.PrefixLength() * 3 * sizeof(std::int32_t);
  Require(
      admitted == 1 && refused == 1 && budget->Outstanding() == settled,
      "actual concurrent Configure paths cannot oversubscribe shared capacity");
  release.count_down();
  first.join();
  second.join();
  Require(!fake::Bytes() && !budget->Outstanding(),
          "concurrent input lifetimes release exactly once");
}

void OnDemandRetentionAndOwnership() {
  constexpr std::size_t protected_bytes = 4096;
  const auto unowned = Prompt();
  const auto capacity = Peak(*unowned);
  auto budget = RuntimeBudget(protected_bytes + capacity, protected_bytes);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  auto prompt = AdmittedPrompt(budget);
  const auto pixel_bytes = prompt->images.front().pixels->pixels.size();
  // A value copy retains the pixel reservation after the original prompt
  // dies. A weak budget marker without pixel-owned tickets misses this case.
  prompt = std::make_shared<vision::Prompt>(*prompt);
  Require(budget->Outstanding() == pixel_bytes &&
              fake::DynamicBytes(budget) == pixel_bytes,
          "copied prepared prompt retains pixel ownership and accounting");
  const auto position_bytes =
      prompt->rope.PrefixLength() * 3 * sizeof(std::int32_t);
  const auto embedding_bytes = std::size_t{kWidth} * sizeof(float);
  const auto scratch = vision::ImageScratchBytes({32, 32});
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(prompt, encoder, nullptr);
    Require(
        budget->Outstanding() ==
            pixel_bytes + position_bytes + embedding_bytes + scratch,
        "on-demand drops completed host layout work, funds future encoding, "
        "and charges prepared pixels once");
    Inject(input, 2);
    const auto retained = pixel_bytes + position_bytes + embedding_bytes;
    Require(budget->Outstanding() == retained &&
                fake::DynamicBytes(budget) == retained,
            "on-demand releases encoder peak only after scratch is freed");
    Require(budget->PendingBytes() == 0,
            "materialized pixels, positions and embeddings are not deducted "
            "again from a fresh free-memory observation");
    // Spend every newly available byte. A same-layout image continuation must
    // reuse retained embeddings without trying to reacquire encoder scratch.
    auto competing = budget->Reserve(capacity - retained);
    const auto encodes = fake::encodes.load();
    input.Configure(prompt, encoder, nullptr);
    input.RestoreLayout(prompt->rope, nullptr);
    Inject(input, 2);
    Require(fake::encodes == encodes && budget->Outstanding() == capacity,
            "exact-full budget still permits already-funded image reuse");
    input.RestoreLayout({}, nullptr);
    Require(fake::DynamicBytes(budget) == retained,
            "restoring an earlier frontier retains actual image capacity");
    Refused<gufo::core::ResourceAdmissionError>(
        [&] { input.RestoreLayout(prompt->rope, nullptr); });
    Require(
        fake::DynamicBytes(budget) == retained,
        "restore refuses missing temporary position capacity before change");
    competing = {};
    input.RestoreLayout(prompt->rope, nullptr);
    Require(
        budget->Outstanding() == retained,
        "restore releases temporary host positions after synchronized upload");
    // The queued/request owner can disappear; the session attachment keeps
    // both prepared pixels and their independent reservation alive.
    prompt.reset();
    Require(budget->Outstanding() == retained,
            "retained prompt alias owns pixels after request completion");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding() && !fake::DynamicBytes(budget),
            "text reset frees both prompt-owned and device-owned allocations");
  }
  Require(!fake::Bytes(), "on-demand descriptor shutdown");
}

void CopiedPreparedPixels() {
  auto budget = RuntimeBudget((2U << 20) + 4096, 4096);
  {
    auto prompt = Prompt();
    const auto original = prompt->images.front().pixels;
    Refused<std::invalid_argument>([&] { prompt->RetainPixels({}, budget); });
    Require(
        prompt->images.front().pixels == original &&
            prompt->memory_admission.expired() && !budget->Outstanding(),
        "missing reservation owner cannot mark uncharged pixels as admitted");
  }
  {
    auto prompt = AdmittedPrompt(budget);
    auto image = prompt->images.front().pixels;
    const auto bytes = image->pixels.capacity();
    prompt.reset();
    Require(
        budget->Outstanding() == bytes && fake::DynamicBytes(budget) == bytes,
        "an individual image copy owns pixels and ticket without a prompt");
  }
  Require(!budget->Outstanding() && !fake::Bytes(),
          "last image pointer frees its physical pixels and reservation");
  {
    auto source = AdmittedPrompt(budget);
    auto destination = AdmittedPrompt(budget, 4, 64);
    const std::weak_ptr<const gufo::core::Image> replaced =
        destination->images.front().pixels;
    const auto bytes = source->images.front().pixels->pixels.capacity();
    *destination = *source;
    source.reset();
    Require(
        replaced.expired() && budget->Outstanding() == bytes &&
            fake::DynamicBytes(budget) == bytes,
        "prompt copy assignment releases replaced pixels before their ticket");
    vision::Prompt moved;
    moved = std::move(*destination);
    destination.reset();
    Require(
        budget->Outstanding() == bytes && fake::DynamicBytes(budget) == bytes,
        "prompt move assignment preserves the copied pixel ticket");
  }
  Require(
      !budget->Outstanding() && !fake::Bytes(),
      "assigned prepared prompts release each pixel allocation exactly once");
}

void OnDemandGrowthHistoryAndFailures() {
  constexpr std::size_t protected_bytes = 4096;
  auto budget = RuntimeBudget(protected_bytes + (2U << 20), protected_bytes);
  fake::budget = budget;
  auto encoder = std::make_shared<vision::Encoder>("unused", kWidth);
  const auto small = Prompt();
  const auto large = Prompt(2, 64);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(small, encoder, nullptr);
    Inject(input, 2);
    const auto old_retained = budget->Outstanding();
    input.Configure(large, encoder, nullptr);
    const auto large_position =
        large->rope.PrefixLength() * 3 * sizeof(std::int32_t);
    const auto large_pixels = large->images[0].pixels->pixels.size();
    const auto large_embedding = 4 * std::size_t{kWidth} * sizeof(float);
    Require(
        budget->Outstanding() == large_position + large_pixels +
                                     large_embedding +
                                     vision::ImageScratchBytes({64, 64}),
        "replacement releases old buffers and host-layout peak after mutation");
    Inject(input, 2, 4);
    Require(
        budget->Outstanding() ==
                large_position + large_pixels + large_embedding &&
            budget->Outstanding() > old_retained,
        "grown retained storage remains charged after encoder scratch ends");
    input.Configure(nullptr, encoder, nullptr);
  }
  // Repeated image pixels produce separate embedding rows in the history.
  // Encoding the first image cannot release work still needed by the second.
  const auto history = Prompt();
  const vision::ImageGrid second{3, 1, 1};
  history->images.push_back({history->images.front().pixels, second, {}});
  history->rope.images.push_back(second);
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(history, encoder, nullptr);
    const auto prepared = budget->Outstanding();
    Inject(input, 2);
    Require(
        budget->Outstanding() == prepared,
        "first image keeps the remaining history encoder obligation funded");
    Inject(input, 3);
    Require(
        budget->Outstanding() == prepared - vision::ImageScratchBytes({32, 32}),
        "last history encoding releases the serial encoder scratch bound");
    input.Configure(nullptr, encoder, nullptr);
  }
  // Production Configure/RestoreLayout are shared by RAM and disk attachment.
  // A fresh session must reserve the complete missing history again.
  {
    vision::DeviceInput restored;
    restored.SetMemoryAdmission(budget);
    restored.Configure(history, encoder, nullptr);
    restored.RestoreLayout(history->rope, nullptr);
    const auto before = budget->Outstanding();
    fake::fail_encode_after_allocation = true;
    Refused<std::bad_alloc>([&] { Inject(restored, 2); });
    Require(budget->Outstanding() == before &&
                fake::DynamicBytes(budget) ==
                    restored.Bytes() - sizeof(vision::DeviceRope),
            "failed encode frees temporaries but retains retry/history "
            "obligations");
    restored.SetCancellationCheck([] { return true; });
    Refused([&] { Inject(restored, 2); });
    Require(budget->Outstanding() == before,
            "cancelled encoding cannot uncharge retained restored history");
    restored.SetCancellationCheck({});
    Inject(restored, 2, 2);
    restored.Configure(nullptr, encoder, nullptr);
  }
  Require(!fake::Bytes() && !budget->Outstanding(),
          "on-demand history, restoration and failed encoding clean up");
  // A changed layout may fail after freeing its old position buffer. Its
  // peak stays funded until recovery rewrites the descriptor and releases it.
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    input.Configure(small, encoder, nullptr);
    Inject(input, 2);
    fake::fail_copy = 0;
    Refused([&] { input.Configure(large, encoder, nullptr); });
    Require(budget->Outstanding() >= fake::DynamicBytes(budget),
            "failed replacement keeps aggregate retained allocations covered");
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(),
            "failed replacement resets without a leaked peak");
  }
  Require(!fake::Bytes(), "on-demand failed replacement descriptor shutdown");
  {
    vision::DeviceInput input;
    input.SetMemoryAdmission(budget);
    fake::fail_malloc = 1;  // Descriptor succeeds; position allocation fails.
    try {
      input.Configure(small, encoder, nullptr);
      throw std::runtime_error("expected physical allocation refusal");
    } catch (const gufo::core::ResourceAdmissionError& error) {
      Require(std::string(error.what()).find("reason=allocation_failure") !=
                  std::string::npos,
              "physical HIP OOM uses the scheduler's resource error path");
    }
    input.Configure(nullptr, encoder, nullptr);
    EmptyDescriptor(input);
    Require(!budget->Outstanding(), "physical OOM releases claims on reset");
    input.Configure(small, encoder, nullptr);
    Inject(input, 2);
  }
  Require(!fake::Bytes() && !budget->Outstanding(),
          "a later request succeeds after physical allocation refusal");
}
}  // namespace

int main() {
  gufo::server::Logger::SetLevel(gufo::server::LogLevel::kError);
  AggregateCoverageRegression();
  RetentionAndTextTransitions();
  GrowthAndRestoredHistory();
  AllocationAndCopyFailures();
  FailedLayoutCopiesDrainSources();
  Cancellation();
  ConcurrentInputs();
  OnDemandRetentionAndOwnership();
  CopiedPreparedPixels();
  OnDemandGrowthHistoryAndFailures();
  std::cout << "vision production lifecycle checks passed (simulated HIP)\n";
}
