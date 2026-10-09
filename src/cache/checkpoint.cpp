#include "src/cache/checkpoint.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gufo::cache {
namespace {
std::uint64_t NextId() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load();
  while (true) {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("cache identifiers exhausted");
    if (next.compare_exchange_weak(value, value + 1))
      return value;
  }
}
class CaptureScope {
public:
  explicit CaptureScope(bool* active) : active_(active) { *active_ = true; }
  ~CaptureScope() { *active_ = false; }
  CaptureScope(const CaptureScope&) = delete;
  CaptureScope& operator=(const CaptureScope&) = delete;
  CaptureScope(CaptureScope&&) = delete;
  CaptureScope& operator=(CaptureScope&&) = delete;

private:
  bool* active_;
};
std::size_t Add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("cache metadata size overflow");
  return a + b;
}
std::size_t Bytes(Rows rows, std::size_t size) {
  if (size && rows > std::numeric_limits<std::size_t>::max() / size)
    throw std::overflow_error("cache payload size overflow");
  return static_cast<std::size_t>(rows) * size;
}
void ValidateDescriptors(std::span<const ComponentDescriptor> descriptors) {
  if (descriptors.empty())
    throw std::invalid_argument("checkpoint has no components");
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const auto& d = descriptors[i];
    for (std::size_t j = 0; j < i; ++j)
      if (d.id == descriptors[j].id)
        throw std::invalid_argument("duplicate checkpoint component");
    if (d.kind == ComponentKind::kAppendRows) {
      if (!d.row_bytes || !d.rows_per_chunk || d.state_bytes)
        throw std::invalid_argument("invalid append component geometry");
      (void)Bytes(d.rows_per_chunk, d.row_bytes);
    } else if (d.kind != ComponentKind::kPrivateState || !d.state_bytes ||
               d.row_bytes || d.rows_per_chunk) {
      throw std::invalid_argument("invalid private component geometry");
    }
  }
}
void ValidatePayload(const Payload& payload, const PayloadRequest& request) {
  const auto expected =
      request.category == ResourceCategory::kBackingAssigned &&
              !payload.BorrowedFrom()
          ? ResourceCategory::kBackingMaterialized
          : request.category;
  if (!payload.IsValid() || (!payload.BorrowedFrom() && !payload.Owner()) ||
      payload.Bytes() < request.bytes || payload.Category() != expected ||
      (request.category == ResourceCategory::kPrivateState &&
       payload.BorrowedFrom()))
    throw std::invalid_argument(
        "incomplete or incompatible checkpoint payload");
}
}  // namespace
namespace detail {
struct Lineage {
  ResourceCharge metadata;
  LineageId id{NextId()};
};
struct Chunk {
  ResourceCharge metadata;
  std::shared_ptr<const Lineage> lineage;
  ChunkId id{NextId()};
  ComponentId component;
  Rows first, end;
  Payload storage;
  std::array<std::atomic<std::size_t>, 3> references{};
  Chunk(ResourceCharge metadata, std::shared_ptr<const Lineage> lineage,
        ComponentId component, Rows first, Rows end, Payload storage)
      : metadata(std::move(metadata)),
        lineage(std::move(lineage)),
        component(component),
        first(first),
        end(end),
        storage(std::move(storage)) {}
};
}  // namespace detail
Payload Payload::Committed(ResourceCharge charge,
                           std::shared_ptr<const void> owner) {
  Payload result;
  result.charge_ = std::move(charge);
  result.owner_ = std::move(owner);
  const auto info = result.charge_.Info();
  if (!result.owner_ || info.reserved ||
      (info.pool_backing && !info.assigned_backing) ||
      (info.category != ResourceCategory::kBackingMaterialized &&
       info.category != ResourceCategory::kPrivateState &&
       info.category != ResourceCategory::kPrivateTail))
    throw std::invalid_argument("committed payload requires owned backing");
  result.bytes_ = info.bytes;
  result.category_ = info.category;
  return result;
}
Payload Payload::Borrowed(std::shared_ptr<BorrowedRows> rows) {
  if (!rows || !rows->IsValid())
    throw std::invalid_argument("invalid managed borrowed rows");
  Payload result;
  result.bytes_ = rows->Bytes();
  result.category_ = rows->Category();
  result.rows_ = std::move(rows);
  return result;
}
RowPin Payload::PinRows(std::stop_token stop) const {
  if (rows_)
    return rows_->Pin(stop);
  RowPin pin;
  pin.charge_ = charge_;
  pin.owner_ = owner_;
  return pin;
}
bool Payload::IsValid() const {
  return rows_ ? rows_->IsValid() : bool(owner_);
}
bool Checkpoint::IsValid() const {
  for (const auto& c : components_) {
    for (const auto& chunk : c.chunks)
      if (!chunk.Storage().IsValid())
        return false;
    if ((c.tail && !c.tail->IsValid()) ||
        (c.private_state && !c.private_state->IsValid()))
      return false;
  }
  return true;
}
Payload& Payload::operator=(Payload&& other) noexcept {
  if (this != &other) {
    rows_.reset();
    owner_.reset();
    charge_ = {};
    charge_ = std::move(other.charge_);
    bytes_ = other.bytes_;
    category_ = other.category_;
    owner_ = std::move(other.owner_);
    rows_ = std::move(other.rows_);
  }
  return *this;
}
PersistencePin Payload::PinPersistence() const {
  if (rows_)
    return rows_->PinPersistence();
  return charge_.PinPersistence();
}
ChunkReference::ChunkReference(std::shared_ptr<detail::Chunk> chunk, Kind kind,
                               PersistencePin pin, RowPin rows_pin)
    : chunk_(std::move(chunk)),
      kind_(kind),
      pin_(std::move(pin)),
      rows_pin_(std::move(rows_pin)) {
  ++chunk_->references[static_cast<std::size_t>(kind_)];
}
ChunkReference::ChunkReference(const ChunkReference& other)
    : chunk_(other.chunk_),
      kind_(other.kind_),
      pin_(other.pin_),
      rows_pin_(other.rows_pin_.Clone()) {
  if (chunk_)
    ++chunk_->references[static_cast<std::size_t>(kind_)];
}
ChunkReference& ChunkReference::operator=(const ChunkReference& other) {
  if (this != &other) {
    ChunkReference copy(other);
    *this = std::move(copy);
  }
  return *this;
}
ChunkReference::ChunkReference(ChunkReference&& other) noexcept
    : chunk_(std::move(other.chunk_)),
      kind_(other.kind_),
      pin_(std::move(other.pin_)),
      rows_pin_(std::move(other.rows_pin_)) {}
ChunkReference& ChunkReference::operator=(ChunkReference&& other) noexcept {
  if (this != &other) {
    Release();
    chunk_ = std::move(other.chunk_);
    kind_ = other.kind_;
    pin_ = std::move(other.pin_);
    rows_pin_ = std::move(other.rows_pin_);
  }
  return *this;
}
ChunkReference::~ChunkReference() {
  Release();
}
void ChunkReference::Release() noexcept {
  rows_pin_ = {};
  pin_ = {};
  if (chunk_) {
    --chunk_->references[static_cast<std::size_t>(kind_)];
    chunk_.reset();
  }
}
ChunkId ChunkReference::Id() const {
  return chunk_->id;
}
LineageId ChunkReference::Lineage() const {
  return chunk_->lineage->id;
}
ComponentId ChunkReference::Component() const {
  return chunk_->component;
}
Rows ChunkReference::First() const {
  return chunk_->first;
}
Rows ChunkReference::End() const {
  return chunk_->end;
}
const Payload& ChunkReference::Storage() const {
  return chunk_->storage;
}
ChunkReferences ChunkReference::References() const {
  const auto& r = chunk_->references;
  return {r[0].load(), r[1].load(), r[2].load()};
}
ChunkReference ChunkReference::PinReader() const {
  return ChunkReference(chunk_, Kind::kReader, {}, chunk_->storage.PinRows());
}
ChunkReference ChunkReference::PinPersistence() const {
  auto rows_pin = chunk_->storage.PinRows();
  auto pin = chunk_->storage.PinPersistence();
  return ChunkReference(chunk_, Kind::kPersistence, std::move(pin),
                        std::move(rows_pin));
}
LineageId Checkpoint::Lineage() const {
  return lineage_->id;
}
LineageId ExecutionHistory::Lineage() const {
  return lineage_->id;
}
std::size_t ExecutionHistory::ChunkMetadataBytes() {
  return sizeof(detail::Chunk);
}
ExecutionHistory::ExecutionHistory(
    ResourceLedger& ledger, std::span<const ComponentDescriptor> descriptors,
    Identity compatibility, std::shared_ptr<const detail::Lineage> lineage)
    : ledger_(&ledger),
      lineage_(std::move(lineage)),
      compatibility_(std::move(compatibility)) {
  const auto bytes =
      Add(Add(sizeof(ExecutionHistory),
              Bytes(descriptors.size(), sizeof(ComponentDescriptor))),
          compatibility_.capacity());
  auto reservation = ledger.Reserve(ResourceCategory::kMetadata, bytes);
  descriptors_.assign(descriptors.begin(), descriptors.end());
  if (descriptors_.capacity() > descriptors.size())
    throw std::logic_error("allocator exceeded reserved metadata capacity");
  metadata_ = reservation.Convert();
}
ExecutionHistory ExecutionHistory::Cold(
    ResourceLedger& ledger, std::span<const ComponentDescriptor> descriptors,
    Identity compatibility) {
  ValidateDescriptors(descriptors);
  auto reservation =
      ledger.Reserve(ResourceCategory::kMetadata, sizeof(detail::Lineage));
  auto lineage = std::make_shared<detail::Lineage>();
  lineage->metadata = reservation.Convert();
  return ExecutionHistory(ledger, descriptors, std::move(compatibility),
                          std::move(lineage));
}
ExecutionHistory ExecutionHistory::Restored(ResourceLedger& ledger,
                                            const Checkpoint& checkpoint) {
  if (!checkpoint.IsValid())
    throw std::invalid_argument("cannot inherit a retired checkpoint");
  std::vector<ComponentDescriptor> descriptors;
  descriptors.reserve(checkpoint.components_.size());
  for (const auto& c : checkpoint.components_)
    descriptors.push_back(c.descriptor);
  ExecutionHistory history(ledger, descriptors, checkpoint.compatibility_,
                           checkpoint.lineage_);
  std::size_t chunks = 0;
  for (const auto& c : checkpoint.components_)
    chunks = Add(chunks, c.chunks.size());
  auto entries_reservation = chunks == 0
                                 ? ResourceReservation{}
                                 : ledger.Reserve(ResourceCategory::kMetadata,
                                                  Bytes(chunks, sizeof(Entry)));
  const auto token_bytes =
      Add(Add(Bytes(checkpoint.tokens_.size(), sizeof(Token)),
              Bytes(checkpoint.components_.size(), sizeof(ComponentPosition))),
          checkpoint.input_.size());
  auto tokens_reservation =
      ledger.Reserve(ResourceCategory::kMetadata, token_bytes);
  history.tokens_ = checkpoint.tokens_;
  history.input_ = checkpoint.input_;
  history.positions_.reserve(checkpoint.components_.size());
  history.entries_.reserve(chunks);
  for (const auto& c : checkpoint.components_) {
    history.positions_.push_back(c.position);
    for (const auto& ref : c.chunks)
      history.entries_.push_back(
          {ref.chunk_->metadata, ref.Component(), ref.First(), ref.chunk_});
  }
  const auto actual_token_bytes =
      Add(Add(Bytes(history.tokens_.capacity(), sizeof(Token)),
              Bytes(history.positions_.capacity(), sizeof(ComponentPosition))),
          history.input_.capacity());
  if (history.entries_.capacity() > chunks || actual_token_bytes > token_bytes)
    throw std::logic_error("allocator exceeded reserved metadata capacity");
  if (entries_reservation)
    history.entries_metadata_ = entries_reservation.Convert();
  history.tokens_metadata_ = tokens_reservation.Convert();
  return history;
}
std::shared_ptr<const Checkpoint> ExecutionHistory::Capture(
    const CheckpointRequest& request, const CapturePayload& capture) {
  if (capturing_)
    throw std::logic_error("reentrant checkpoint capture");
  const CaptureScope scope(&capturing_);
  if (!capture || request.positions.size() != descriptors_.size() ||
      request.tokens.size() < tokens_.size() ||
      !std::equal(tokens_.begin(), tokens_.end(), request.tokens.begin()))
    throw std::invalid_argument(
        "checkpoint is not a continuation of this history");
  const auto previous_input = request.input.At(tokens_.size());
  if (!positions_.empty() &&
      !std::equal(input_.begin(), input_.end(), previous_input.begin(),
                  previous_input.end()))
    throw std::invalid_argument(
        "checkpoint input changed within inherited prefix");
  const auto position_for = [&](ComponentId id) {
    if (std::count_if(request.positions.begin(), request.positions.end(),
                      [&](const auto& p) { return p.id == id; }) != 1)
      throw std::invalid_argument(
          "checkpoint needs exactly one position per component");
    return *std::find_if(request.positions.begin(), request.positions.end(),
                         [&](const auto& p) { return p.id == id; });
  };
  std::size_t chunk_count = 0;
  for (std::size_t i = 0; i < descriptors_.size(); ++i) {
    const auto& d = descriptors_[i];
    const auto p = position_for(d.id);
    if (!positions_.empty() && p.valid_rows < positions_[i].valid_rows)
      throw std::invalid_argument(
          "checkpoint component rewound without restore");
    if (d.kind == ComponentKind::kAppendRows) {
      (void)Bytes(p.valid_rows, d.row_bytes);
      chunk_count = Add(chunk_count, Bytes(p.valid_rows / d.rows_per_chunk, 1));
    }
  }
  const auto input = request.input.At(request.tokens.size());
  auto bytes =
      Add(sizeof(Checkpoint), Bytes(request.tokens.size(), sizeof(Token)));
  bytes = Add(bytes, Add(compatibility_.size(), input.size()));
  bytes = Add(bytes, Bytes(descriptors_.size(), sizeof(CheckpointComponent)));
  bytes = Add(bytes, Bytes(chunk_count, sizeof(ChunkReference)));
  auto checkpoint_reservation =
      ledger_->Reserve(ResourceCategory::kMetadata, bytes);
  // Charge old and replacement history buffers simultaneously until
  // publication. reserve() is exact with the pinned libstdc++; verify
  // capacities before commit.
  const auto entry_capacity = Add(entries_.size(), chunk_count);
  auto entries_reservation =
      entry_capacity == 0
          ? ResourceReservation{}
          : ledger_->Reserve(ResourceCategory::kMetadata,
                             Bytes(entry_capacity, sizeof(Entry)));
  const auto token_bytes =
      Add(Add(Bytes(request.tokens.size(), sizeof(Token)),
              Bytes(descriptors_.size(), sizeof(ComponentPosition))),
          input.size());
  auto tokens_reservation =
      ledger_->Reserve(ResourceCategory::kMetadata, token_bytes);
  auto checkpoint = std::shared_ptr<Checkpoint>(new Checkpoint);
  checkpoint->id_ = {NextId()};
  checkpoint->lineage_ = lineage_;
  checkpoint->compatibility_ = compatibility_;
  checkpoint->input_.assign(input.begin(), input.end());
  checkpoint->tokens_.assign(request.tokens.begin(), request.tokens.end());
  checkpoint->purpose_ = request.purpose;
  checkpoint->rank_ = request.rank;
  checkpoint->components_.reserve(descriptors_.size());
  std::vector<Entry> entries;
  entries.reserve(entry_capacity);
  for (const auto& entry : entries_)
    if (!entry.chunk.expired())
      entries.push_back(entry);
  std::vector<ComponentPosition> positions;
  positions.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    const auto p = position_for(d.id);
    positions.push_back(p);
    const auto end = p.valid_rows;
    CheckpointComponent component{d, p, {}, {}, {}};
    if (d.kind == ComponentKind::kPrivateState) {
      const PayloadRequest r{d.id, ResourceCategory::kPrivateState, 0, end,
                             d.state_bytes};
      component.private_state.emplace(capture(r));
      ValidatePayload(*component.private_state, r);
    } else {
      const auto full_end = end - end % d.rows_per_chunk;
      component.chunks.reserve(end / d.rows_per_chunk);
      for (Rows first = 0; first < full_end; first += d.rows_per_chunk) {
        std::shared_ptr<detail::Chunk> chunk;
        for (const auto& e : entries)
          if (e.component == d.id && e.first == first) {
            chunk = e.chunk.lock();
            break;
          }
        if (!chunk) {
          const PayloadRequest r{d.id, ResourceCategory::kBackingAssigned,
                                 first, first + d.rows_per_chunk,
                                 Bytes(d.rows_per_chunk, d.row_bytes)};
          auto metadata_reservation = ledger_->Reserve(
              ResourceCategory::kMetadata, ChunkMetadataBytes());
          auto storage = capture(r);
          ValidatePayload(storage, r);
          chunk = std::make_shared<detail::Chunk>(ResourceCharge{}, lineage_,
                                                  d.id, r.first, r.end,
                                                  std::move(storage));
          chunk->metadata = metadata_reservation.Convert();
          entries.push_back({chunk->metadata, d.id, first, chunk});
        }
        if (!chunk->storage.IsValid())
          throw std::invalid_argument("cannot publish retired checkpoint rows");
        component.chunks.push_back(ChunkReference(
            std::move(chunk), ChunkReference::Kind::kCheckpoint));
      }
      if (full_end != end) {
        const PayloadRequest r{d.id, ResourceCategory::kPrivateTail, full_end,
                               end, Bytes(end - full_end, d.row_bytes)};
        component.tail.emplace(capture(r));
        ValidatePayload(*component.tail, r);
      }
    }
    checkpoint->components_.push_back(std::move(component));
  }
  std::vector<Token> tokens(request.tokens.begin(), request.tokens.end());
  Identity new_input(input.begin(), input.end());
  auto actual_bytes = Add(sizeof(Checkpoint),
                          Bytes(checkpoint->tokens_.capacity(), sizeof(Token)));
  actual_bytes = Add(actual_bytes, Add(checkpoint->compatibility_.capacity(),
                                       checkpoint->input_.capacity()));
  actual_bytes = Add(actual_bytes, Bytes(checkpoint->components_.capacity(),
                                         sizeof(CheckpointComponent)));
  for (const auto& c : checkpoint->components_)
    actual_bytes =
        Add(actual_bytes, Bytes(c.chunks.capacity(), sizeof(ChunkReference)));
  const auto actual_token_bytes =
      Add(Add(Bytes(tokens.capacity(), sizeof(Token)),
              Bytes(positions.capacity(), sizeof(ComponentPosition))),
          new_input.capacity());
  if (actual_bytes > bytes || entries.capacity() > entry_capacity ||
      actual_token_bytes > token_bytes)
    throw std::logic_error("allocator exceeded reserved metadata capacity");
  checkpoint->metadata_bytes_ = bytes;
  checkpoint->metadata_ = checkpoint_reservation.Convert();
  auto entries_metadata =
      entries_reservation ? entries_reservation.Convert() : ResourceCharge{};
  auto tokens_metadata = tokens_reservation.Convert();
  // Publication is nonthrowing. No checkpoint or new history escapes earlier.
  entries_.swap(entries);
  tokens_.swap(tokens);
  positions_.swap(positions);
  input_.swap(new_input);
  std::vector<Entry>().swap(entries);
  std::vector<Token>().swap(tokens);
  std::vector<ComponentPosition>().swap(positions);
  Identity().swap(new_input);
  entries_metadata_ = std::move(entries_metadata);
  tokens_metadata_ = std::move(tokens_metadata);
  return checkpoint;
}
void ExecutionHistory::Prune() {
  // Destroy expired control blocks before releasing their accounting. Clearing
  // holes first also makes vector compaction's member-wise moves safe.
  for (auto& entry : entries_) {
    if (entry.chunk.expired()) {
      entry.chunk.reset();
      entry.metadata = {};
    }
  }
  std::erase_if(entries_, [](const auto& entry) { return !entry.metadata; });
}
}  // namespace gufo::cache
