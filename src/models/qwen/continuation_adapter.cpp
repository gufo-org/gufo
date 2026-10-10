#include "src/models/qwen/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gufo::models::qwen {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
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
struct Metadata {
  std::uint32_t version{1}, position{}, logits_valid{}, has_next{}, next{},
      image_bytes{};
  std::array<std::uint8_t, 32> image{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<hip::QwenGpuExecutor> executor)
      : adapter(adapter), guard(guard), executor(std::move(executor)) {
    this->executor->continuation_hooks_ = this;
    this->executor->vision_input_.continuation_hooks_ = this;
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
    executor->continuation_hooks_ = nullptr;
    executor->vision_input_.continuation_hooks_ = nullptr;
  }
  bool IsValid() const noexcept override {
    return guarding || (!restoring && !failed && !executor->reset_failure_);
  }
  void BeforeExecution() override {
    if (!IsValid() || pending)
      throw std::logic_error("Qwen slot is not ready for execution");
  }
  void BeforeWrite(std::uint32_t first, std::uint32_t end) override {
    BeforeExecution();
    if (first > end || end > adapter.context_)
      throw std::invalid_argument("Qwen mutation exceeds context");
    Readable readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows &&
          first < end)
        guard.BeforeOverwrite(component.descriptor.id, first, end);
  }
  void AfterWrite(std::uint32_t end, bool logits) noexcept override {
    position = end;
    logits_valid = logits;
  }
  void WriteFailed() noexcept override {
    failed = true;
    native_restoring = false;
  }
  void GuardAll() {
    Readable readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    if (restoring || pending)
      throw std::logic_error("cannot reset a busy Qwen slot");
    GuardAll();
    failed = true;
  }
  void AfterReset() override {
    Check(hipStreamSynchronize(executor->arena_.stream));
    position = 0;
    logits_valid = false;
    verification = false;
    failed = false;
    native_restoring = false;
  }
  void BeforeRestore() override {
    BeforeExecution();
    GuardAll();
    failed = true;
    native_restoring = true;
  }
  void AfterRestore(std::uint32_t target) noexcept override {
    position = target;
    logits_valid = false;
    verification = false;
    failed = false;
    native_restoring = false;
  }
  void AfterSaveState() noexcept override { verification = true; }
  void AfterFinishVerification() noexcept override { verification = false; }
  void AfterLogits() noexcept override { logits_valid = true; }
  void BeforeVision(const vision::Prompt* prompt,
                    const vision::RopeLayout* layout) override {
    if (native_restoring || validating)
      return;
    BeforeExecution();
    const auto& current = executor->vision_input_.PromptAttachment();
    if (layout) {
      const auto expected =
          current ? current->rope.Prefix(position) : vision::RopeLayout{};
      if (layout->Prefix(position) != expected)
        throw std::logic_error(
            "Qwen image layout replacement changes computed input");
    } else {
      const auto old_identity = current ? current->IdentityForPrefix(position)
                                        : std::span<const std::uint8_t>{};
      const auto new_identity = prompt ? prompt->IdentityForPrefix(position)
                                       : std::span<const std::uint8_t>{};
      if (!std::ranges::equal(old_identity, new_identity))
        throw std::logic_error("Qwen image replacement changes computed input");
    }
  }
  void BeforeRelease() noexcept override {
    Readable readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
  }
  struct Readable {
    explicit Readable(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~Readable() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  void Stable() const {
    if (!IsValid() || verification)
      throw std::logic_error("Qwen capture requires a stable frontier");
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<hip::QwenGpuExecutor> executor;
  std::uint32_t position{};
  bool logits_valid{}, verification{}, restoring{}, guarding{}, invalidating{},
      native_restoring{}, validating{};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

ContinuationAdapter::ContinuationAdapter(
    std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t context,
    cache::Identity identity, hip::QwenExecutionPolicy policy)
    : model_(std::move(model)),
      policy_(policy),
      context_(context),
      layout_(model_ ? model_->GetConfig() : core::ModelConfig{}, policy,
              context),
      identity_(std::move(identity)) {
  if (!model_ || identity_.empty())
    throw std::invalid_argument("invalid Qwen adapter configuration");
  const auto put = [&](std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
  };
  put(1);
  put(context);
  put(policy.Fingerprint());
  const auto& c = model_->GetConfig();
  for (auto value :
       {c.num_layers, c.hidden_size, c.num_attention_heads,
        c.num_key_value_heads, c.head_dim, c.vocab_size,
        c.full_attention_interval, c.ssm_conv_kernel, c.ssm_state_size,
        c.ssm_group_count, c.ssm_time_step_rank, c.ssm_inner_size})
    put(value);
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Qwen adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
hip::QwenGpuExecutor& ContinuationAdapter::GetExecutor(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.executor;
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
  auto executor =
      std::make_unique<hip::QwenGpuExecutor>(model_, context_, policy_);
  return std::make_unique<State>(*this, guard, std::move(executor));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Qwen slot");
  return layout_.Positions(state.position);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Qwen prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    first += std::min<Rows>(end - first, state.executor->GetMaxPromptBatch());
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Qwen component ID");
  return layout_.Components()[id.value - 1];
}
void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Qwen frontiers");
  std::vector<cache::ComponentPosition> ordered;
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("duplicate or missing Qwen frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered != layout_.Positions(ordered.front().valid_rows))
    throw std::invalid_argument("inconsistent Qwen frontiers");
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  Check(hipStreamSynchronize(state.executor->arena_.stream));
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
}
cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(hip::TransferStream&)>& submit) {
  auto& native = dynamic_cast<hip::TransferStream&>(stream);
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
    submit(native);
    job->completion.emplace(native.Complete());
  } catch (...) {
    (void)native.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Qwen private buffer");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  state.Stable();
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Qwen private piece");
  auto& executor = *state.executor;
  auto& arena = executor.arena_;
  Check(hipStreamSynchronize(arena.stream));
  std::array<std::byte, kContinuationMetadataBytes> host{};
  void* source = nullptr;
  bool is_host = false;
  if (component.part == ContinuationPart::kMetadata) {
    Metadata metadata{};
    metadata.position = state.position;
    metadata.logits_valid = state.logits_valid;
    metadata.has_next = executor.next_token_.has_value();
    metadata.next = executor.next_token_.value_or(0);
    const auto& attachment = executor.vision_input_.PromptAttachment();
    const auto identity = attachment
                              ? attachment->IdentityForPrefix(state.position)
                              : std::span<const std::uint8_t>{};
    if (!identity.empty()) {
      if (identity.size() != metadata.image.size())
        throw std::logic_error("invalid Qwen image identity");
      metadata.image_bytes = 32;
      std::ranges::copy(identity, metadata.image.begin());
    }
    std::memcpy(host.data(), &metadata, sizeof(metadata));
    source = host.data();
    is_host = true;
  } else if (component.part == ContinuationPart::kLogits) {
    if (state.logits_valid)
      source = arena.d_logits;
  } else if (component.part == ContinuationPart::kConvolution) {
    source = arena.d_ssm_conv_state +
             std::size_t{model_->GetConfig().SsmLayerIndex(component.layer)} *
                 total / sizeof(float);
  } else if (component.part == ContinuationPart::kRecurrent) {
    source =
        static_cast<std::byte*>(arena.d_ssm_deltanet_state) +
        std::size_t{model_->GetConfig().SsmLayerIndex(component.layer)} * total;
  }
  return Transfer(state, false, stream, [&](hip::TransferStream& transfer) {
    if (bytes.empty())
      return;
    if (!source)
      std::ranges::fill(bytes, std::byte{});
    else if (is_host)
      std::memcpy(bytes.data(), static_cast<std::byte*>(source) + offset,
                  bytes.size());
    else
      Check(hipMemcpyAsync(
          bytes.data(), static_cast<std::byte*>(source) + offset, bytes.size(),
          hipMemcpyDeviceToHost, transfer.Native()));
  });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Qwen load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Qwen private buffer");
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
    throw std::logic_error("Qwen load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Qwen private piece");
    auto& arena = state.executor->arena_;
    void* target = nullptr;
    bool is_host = false;
    if (component.part == ContinuationPart::kMetadata) {
      target = state.metadata.data();
      is_host = true;
    } else if (component.part == ContinuationPart::kLogits)
      target = arena.d_logits;
    else if (component.part == ContinuationPart::kConvolution)
      target = arena.d_ssm_conv_state +
               std::size_t{model_->GetConfig().SsmLayerIndex(component.layer)} *
                   total / sizeof(float);
    else if (component.part == ContinuationPart::kRecurrent)
      target = static_cast<std::byte*>(arena.d_ssm_deltanet_state) +
               std::size_t{model_->GetConfig().SsmLayerIndex(component.layer)} *
                   total;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(state, true, stream, [&](hip::TransferStream& transfer) {
      if (bytes.empty())
        return;
      if (is_host)
        std::memcpy(static_cast<std::byte*>(target) + offset, bytes.data(),
                    bytes.size());
      else
        Check(hipMemcpyAsync(static_cast<std::byte*>(target) + offset,
                             bytes.data(), bytes.size(), hipMemcpyHostToDevice,
                             transfer.Native()));
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
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Qwen slot");
  const auto& component = Component(id);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > state.position ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Qwen row capture");
  Check(hipStreamSynchronize(state.executor->arena_.stream));
  return TransferRows(state, component, false, first, end, bytes.data(),
                      stream);
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Qwen load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Qwen row restore");
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return TransferRows(state, component, true, first, end,
                        const_cast<std::byte*>(bytes.data()), stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::TransferRows(
    State& state, const ContinuationComponent& component, bool load, Rows first,
    Rows end, std::byte* bytes, cache::Stream& stream) {
  const auto& c = model_->GetConfig();
  auto& arena = state.executor->arena_;
  const auto row = component.descriptor.row_bytes;
  const std::size_t layer = component.layer / c.full_attention_interval;
  const std::size_t plane = component.part == ContinuationPart::kValue
                                ? c.FullAttentionLayerCount()
                                : 0;
  auto* native =
      static_cast<std::byte*>(policy_.UsesFp16AttentionKv()
                                  ? arena.d_attention_kv_f16
                                  : static_cast<void*>(arena.d_kv_cache)) +
      (plane + layer) * context_ * row;
  return Transfer(state, load, stream, [&](hip::TransferStream& transfer) {
    if (first == end)
      return;
    const auto direction = load ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
    if (policy_.UsesFp16AttentionKv()) {
      auto* device = native + first * row;
      Check(hipMemcpyAsync(load ? device : bytes, load ? bytes : device,
                           (end - first) * row, direction, transfer.Native()));
    } else {
      const auto head_bytes = std::size_t{c.head_dim} * sizeof(float);
      for (std::uint32_t head = 0; head < c.num_key_value_heads; ++head) {
        auto* device =
            native + (head * std::size_t{context_} + first) * head_bytes;
        auto* host = bytes + head * head_bytes;
        Check(hipMemcpy2DAsync(load ? device : host, load ? head_bytes : row,
                               load ? host : device, load ? row : head_bytes,
                               head_bytes, end - first, direction,
                               transfer.Native()));
      }
    }
  });
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Qwen validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    const auto& descriptor = descriptors_[i];
    if (std::ranges::count(positions, expected) != 1 ||
        !Covered(state.coverage[i],
                 descriptor.row_bytes
                     ? expected.valid_rows * descriptor.row_bytes
                     : descriptor.state_bytes))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  const auto& c = model_->GetConfig();
  auto& executor = *state.executor;
  if (metadata.version != 1 ||
      metadata.position != state.expected.front().valid_rows ||
      metadata.logits_valid > 1 || metadata.has_next > 1 ||
      (metadata.has_next && metadata.next >= c.vocab_size) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32))
    return refuse();
  try {
    struct Scope {
      bool& flag;
      ~Scope() { flag = false; }
    } scope{state.validating};
    state.validating = true;
    const auto& attachment = executor.vision_input_.PromptAttachment();
    const auto identity = attachment
                              ? attachment->IdentityForPrefix(metadata.position)
                              : std::span<const std::uint8_t>{};
    if (metadata.image_bytes) {
      if (identity.size() != 32 ||
          !std::ranges::equal(identity, metadata.image))
        return refuse();
      executor.vision_input_.RestoreLayout(attachment->rope,
                                           executor.arena_.stream);
    } else if (attachment && identity.empty()) {
      // A text checkpoint can precede the first image in this request.
      // Its suffix embeddings and positions must survive validation.
      executor.vision_input_.RestoreLayout(attachment->rope,
                                           executor.arena_.stream);
    } else
      executor.vision_input_.Configure(nullptr, nullptr,
                                       executor.arena_.stream);
    executor.arena_.DisableSsmReplayCapture();
    Check(hipStreamSynchronize(executor.arena_.stream));
  } catch (...) {
    return refuse();
  }
  executor.arena_.has_saved_state_ = false;
  executor.arena_.saved_context_ = 0;
  executor.arena_.replay_captured_positions_ = 0;
  executor.replaying_ssm_state_ = false;
  executor.next_token_ =
      metadata.has_next ? std::optional<tokenization::TokenId>{metadata.next}
                        : std::nullopt;
  executor.h_prompt_hidden_.clear();
  executor.h_verification_hidden_.clear();
  executor.h_verification_logits_.clear();
  executor.h_last_hidden_.clear();
  executor.last_verification_rows_ = 0;
  executor.last_hidden_offset_ = 0;
  executor.graph_executor_.Reset();
  state.position = metadata.position;
  state.logits_valid = metadata.logits_valid;
  state.verification = false;
  state.restoring = false;
  state.expected.clear();
  state.coverage.clear();
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
    struct Scope {
      bool& flag;
      ~Scope() { flag = false; }
    } scope{state.invalidating};
    state.invalidating = true;
    state.executor->Reset();
    if (state.executor->reset_failure_)
      return false;
    state.restoring = false;
    state.failed = false;
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
}  // namespace gufo::models::qwen
