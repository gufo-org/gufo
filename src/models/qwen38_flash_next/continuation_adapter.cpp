#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
bool Draft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool Pool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t target{}, draft{}, blocks{}, draft_blocks{}, hidden_base{};
  std::uint32_t residual_valid{}, image_bytes{};
  NgramHistory ngram;
  MtpLengthState policy;
  std::array<std::uint8_t, 32> image{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->continuation_hooks_ = this;
    this->session->session_->continuation_hooks_ = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    session->continuation_hooks_ = nullptr;
    session->session_->continuation_hooks_ = nullptr;
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  void BeforeExecution() override {
    if (!IsValid() || pending)
      throw std::logic_error("Flash-Next slot is not ready for execution");
  }
  void BeforeWrite(bool draft, std::uint32_t first, std::uint32_t end,
                   std::uint32_t first_block,
                   std::uint32_t end_block) override {
    // Model execution temporarily disables its public validity flag. The
    // adapter's restore/failure status and outstanding transfers still gate
    // every native write, including graph replay and speculative rollback.
    if (restoring || failed || pending)
      throw std::logic_error("Flash-Next slot is not ready for mutation");
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows ||
          Draft(component.part) != draft)
        continue;
      const auto a = Pool(component.part) ? first_block : first;
      const auto b = Pool(component.part) ? end_block : end;
      if (a < b)
        guard.BeforeOverwrite(component.descriptor.id, a, b);
    }
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    if (restoring || pending)
      throw std::logic_error("cannot reset a busy Flash-Next slot");
    GuardAll();
  }
  void AfterReset() override {
    // Native Reset queues zero fills. Settle them before exposing an empty
    // slot to captures on independent streams.
    Check(hipStreamSynchronize(adapter.model_->executor_->stream()));
  }
  void BeforeRestore() override {
    BeforeExecution();
    GuardAll();
  }
  void BeforeRelease() noexcept override {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  void GuardAll() {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  struct CaptureDuringGuard {
    explicit CaptureDuringGuard(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~CaptureDuringGuard() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  bool restoring{false}, guarding{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      layout_(model_ ? model_->config() : Config{},
              mode == core::SessionMode::kSpeculative, context,
              model_ ? model_->executor_->max_speculative() : 0),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasMtp()))
    throw std::invalid_argument("invalid Flash-Next adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Private components attest the target checkpoint boundary.
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  put(model_->executor_->max_speculative());
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Flash-Next adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(mode_, context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flash-Next slot");
  const auto& native = *state.session->session_;
  return layout_.Positions(native.position_, native.mtp_.position,
                           native.blocks_, native.mtp_.blocks);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flash-Next prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    const auto remaining = end - first;
    const auto capacity = remaining <= model_->PrefillThroughCapacity()
                              ? model_->PrefillThroughCapacity()
                              : model_->PrefillCapacity();
    first += std::min<Rows>(remaining, capacity);
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Flash-Next component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& native = *state.session->session_;
  switch (component.part) {
    case ContinuationPart::kTokens:
      return state.session->tokens_.data();
    case ContinuationPart::kKey:
      return native.attention_[component.layer].k_cache;
    case ContinuationPart::kValue:
      return native.attention_[component.layer].v_cache;
    case ContinuationPart::kPooledKeys:
      return native.attention_[component.layer].block_k;
    case ContinuationPart::kDraftKey:
      return native.mtp_.k_cache;
    case ContinuationPart::kDraftValue:
      return native.mtp_.v_cache;
    case ContinuationPart::kDraftPooledKeys:
      return native.mtp_.block_k;
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Flash-Next restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Flash-Next frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = [&](ContinuationPart part) -> std::uint32_t {
    for (std::size_t i = 0; i < ordered.size(); ++i)
      if (layout_.Components()[i].part == part) {
        if (ordered[i].valid_rows > context_)
          throw std::invalid_argument("Flash-Next frontier exceeds context");
        return static_cast<std::uint32_t>(ordered[i].valid_rows);
      }
    return 0;
  };
  const auto target = frontier(ContinuationPart::kTokens);
  const auto draft = frontier(ContinuationPart::kDraftKey);
  const auto expected =
      layout_.Positions(target, draft, frontier(ContinuationPart::kPooledKeys),
                        frontier(ContinuationPart::kDraftPooledKeys));
  if (ordered != expected)
    throw std::invalid_argument("inconsistent Flash-Next component frontiers");
  // Allocate host capacity and coverage first; any refusal leaves execution
  // valid. Guard all ranges while the original state is still readable.
  state.session->tokens_.reserve(target);
  state.session->logits_.reserve(model_->VocabSize());
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  Check(hipStreamSynchronize(model_->executor_->stream()));
  state.session->session_->PreserveSnapshots(0, 0);
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
  state.session->valid_ = false;
  state.session->tokens_.resize(target);
  state.session->logits_.resize(model_->VocabSize());
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading) const {
  auto& session = *state.session;
  auto& native = *session.session_;
  Region result;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false,
                       std::size_t offset = 0) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(session.logits_.data(), bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(native.linear_[component.layer].conv_state, bytes);
      break;
    case ContinuationPart::kRecurrent:
      add(native.linear_[component.layer].state, bytes);
      break;
    case ContinuationPart::kPleHistory:
      add(native.ple_history_, bytes);
      break;
    case ContinuationPart::kDraftResidual:
      if (loading || native.mtp_.residual_valid)
        add(native.mtp_.h, bytes);
      break;
    case ContinuationPart::kKeptHidden:
      add(native.mtp_.target_hidden,
          loading ? bytes
                  : std::size_t{session.KeptHiddenRows()} *
                        model_->config().HcDim() * sizeof(float));
      break;
    case ContinuationPart::kRawKeys:
    case ContinuationPart::kDraftRawKeys: {
      const auto positions = loading ? state.expected : Positions(state);
      const bool draft = Draft(component.part);
      std::uint32_t position = 0, blocks = 0;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto part = layout_.Components()[i].part;
        if (part ==
            (draft ? ContinuationPart::kDraftKey : ContinuationPart::kTokens))
          position = static_cast<std::uint32_t>(positions[i].valid_rows);
        if (part == (draft ? ContinuationPart::kDraftPooledKeys
                           : ContinuationPart::kPooledKeys))
          blocks = static_cast<std::uint32_t>(positions[i].valid_rows);
      }
      const auto& c = model_->config();
      const auto begin = blocks * c.compress_ratio;
      const auto rows = position - begin;
      const auto physical = begin & (native.index_capacity_ - 1);
      const auto tail = std::min(rows, native.index_capacity_ - physical);
      const auto width = std::size_t{c.indexer_head_dim} * sizeof(float);
      auto* pointer = reinterpret_cast<std::byte*>(
          draft ? native.mtp_.index_k
                : native.attention_[component.layer].index_k);
      add(pointer + physical * width, tail * width);
      add(pointer, (rows - tail) * width, false, tail * width);
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flash-Next private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->session_->spec_tokens_)
    throw std::logic_error(
        "cannot capture an invalid or speculative Flash-Next frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flash-Next private piece");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  if (component.part == ContinuationPart::kMetadata) {
    auto& session = *state.session;
    auto& native = *session.session_;
    Metadata metadata{};
    metadata.target = native.position_;
    metadata.draft = native.mtp_.position;
    metadata.blocks = native.blocks_;
    metadata.draft_blocks = native.mtp_.blocks;
    metadata.hidden_base = session.hidden_base_;
    metadata.residual_valid = native.mtp_.residual_valid;
    metadata.ngram = native.ngram_;
    metadata.policy = session.draft_length_.State();
    const auto identity = session.ImageIdentity(native.position_);
    metadata.image_bytes = static_cast<std::uint32_t>(identity.size());
    if (!identity.empty()) {
      if (identity.size() != metadata.image.size())
        throw std::logic_error("invalid Flash-Next image identity");
      std::ranges::copy(identity, metadata.image.begin());
    }
    std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
  }
  auto region = PrivateRegion(state, component, false);
  if (component.part == ContinuationPart::kMetadata)
    region.pieces = {{host_metadata.data(), 0, host_metadata.size(), true}};
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        std::ranges::fill(bytes, std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hipMemcpyAsync(destination, source, end - first,
                                 hipMemcpyDeviceToHost, transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Flash-Next private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Flash-Next private piece");
    const auto region = PrivateRegion(state, component, true);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hipMemcpyAsync(destination, source, end - first,
                                   hipMemcpyHostToDevice, transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flash-Next slot");
  const auto& native = *state.session->session_;
  const auto rows =
      component.part == ContinuationPart::kTokens
          ? state.session->tokens_.size()
      : Pool(component.part)
          ? (Draft(component.part) ? native.mtp_.blocks : native.blocks_)
          : (Draft(component.part) ? native.mtp_.position : native.position_);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > rows || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flash-Next row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : static_cast<std::byte*>(RowData(state, component)) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (component.part == ContinuationPart::kTokens && !bytes.empty())
          std::memcpy(bytes.data(), source, bytes.size());
        else if (!bytes.empty())
          Check(hipMemcpyAsync(bytes.data(), source, bytes.size(),
                               hipMemcpyDeviceToHost, transfer.Native()));
      });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flash-Next row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : static_cast<std::byte*>(RowData(state, component)) + first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (component.part == ContinuationPart::kTokens && !bytes.empty())
            std::memcpy(destination, bytes.data(), bytes.size());
          else if (!bytes.empty())
            Check(hipMemcpyAsync(destination, bytes.data(), bytes.size(),
                                 hipMemcpyHostToDevice, transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Flash-Next validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  auto& session = *state.session;
  auto& native = *session.session_;
  auto policy = session.draft_length_;
  if (metadata.version != 1 || metadata.hidden_base > metadata.target ||
      metadata.target - metadata.hidden_base >
          model_->executor_->max_speculative() ||
      metadata.residual_valid > 1 ||
      (metadata.residual_valid && !metadata.draft) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32) ||
      !policy.Restore(metadata.policy))
    return refuse();
  try {
    if (layout_.Positions(metadata.target, metadata.draft, metadata.blocks,
                          metadata.draft_blocks) != state.expected)
      return refuse();
    const auto identity = session.ImageIdentity(metadata.target);
    if (metadata.image_bytes && (identity.size() != 32 ||
                                 !std::ranges::equal(identity, metadata.image)))
      return refuse();
    if (metadata.image_bytes || (session.image_prompt_ && identity.empty())) {
      // Preserve the attached request's future images as well as its matching
      // computed prefix, including checkpoints before the first image.
      native.RestoreVisionLayout(session.image_prompt_->rope,
                                 model_->executor_->stream());
    } else {
      native.ConfigureVision(nullptr, nullptr, model_->executor_->stream());
      session.image_prompt_.reset();
    }
    Check(hipStreamSynchronize(model_->executor_->stream()));
  } catch (...) {
    return refuse();
  }
  native.position_ = metadata.target;
  native.blocks_ = metadata.blocks;
  native.mtp_.position = metadata.draft;
  native.mtp_.blocks = metadata.draft_blocks;
  native.mtp_.residual_valid = metadata.residual_valid;
  native.ngram_ = metadata.ngram;
  native.spec_base_ = metadata.target;
  native.spec_tokens_ = 0;
  native.snapshot_lineage_.reset();
  session.hidden_base_ = metadata.hidden_base;
  session.draft_length_ = policy;
  session.draft_token_ = 0;
  session.stats_ = {};
  session.valid_ = true;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    resetting = &state;
    state.BeforeRelease();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::qwen38_flash_next
