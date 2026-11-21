// Gufo's component cache over Loom rows: named device checkpoints, or
// portable checkpoint images when a disk tier is configured.
#ifndef GUFO_LOOM_CACHE_HPP_
#define GUFO_LOOM_CACHE_HPP_

#include "loom_model.hpp"

namespace gufo::loom {

// ---------------------------------------------------------------- cache ---

// Free Loom rows, shared by the adapter and its slots.
struct RowPool {
  std::mutex mutex;
  std::vector<bool> used;
};

class LoomSlot final : public cache::Slot {
public:
  LoomSlot(std::uint32_t row, std::shared_ptr<RowPool> pool)
      : row_(row), pool_(std::move(pool)) {}
  ~LoomSlot() override {
    std::lock_guard lock(pool_->mutex);
    pool_->used[row_] = false;
  }
  [[nodiscard]] bool IsValid() const noexcept override { return true; }
  [[nodiscard]] std::uint32_t row() const { return row_; }
  // Tokens Gufo counts as consumed, and the prompt token still deferred
  // (consumed by Gufo's count, not yet by the Loom row).
  std::size_t position{0};
  std::int64_t deferred{-1};
  // A restore in progress loaded its checkpoint.
  bool restore_ok{false};
  // The Loom row must be reset before reuse.
  bool stale{true};
  // Bumped whenever the row's history is replaced (reset or restore), so a
  // cached image is keyed by {generation, row position}.
  std::uint64_t generation{0};
  // Image mode: the last exported image of this row and its header, and a
  // restore in progress (private bytes, pages, which pages arrived).
  std::uint64_t image_generation{~std::uint64_t{0}};
  std::uint64_t image_position{0};
  std::vector<std::uint8_t> image;
  std::vector<std::uint8_t> image_header;
  std::vector<std::uint8_t> restore_private;
  std::vector<std::uint8_t> restore_pages;
  std::vector<bool> restore_have;
  std::size_t restore_private_bytes{0};

private:
  std::uint32_t row_;
  std::shared_ptr<RowPool> pool_;
};

// The one component: the row endpoint, held on device by a Loom checkpoint
// and named in Gufo's cache by {checkpoint id, deferred token}.
inline constexpr cache::ComponentId kEndpoint{1};
inline const cache::ComponentDescriptor kEndpointDescriptor{
    .id = kEndpoint,
    .layout_version = 1,
    .kind = cache::ComponentKind::kPrivateState,
    .state_bytes = kMarkerBytes,
};

// Image mode (a disk cache is configured): the checkpoint travels as bytes.
// Full Loom pages are append rows (one row = one page of block_size tokens);
// the fixed state, the partial last page and a header are private state.
inline constexpr cache::ComponentId kImageState{2};
inline constexpr cache::ComponentId kImagePages{3};
inline constexpr cache::Rows kPagesPerChunk = 4;
inline constexpr std::uint32_t kImageMagic = 0x4c4d4931;  // "LMI1"
inline constexpr std::size_t kImageHeaderBytes = 64;
struct ImageHeader {
  std::uint32_t magic;
  std::uint32_t loom_position;
  std::uint64_t position;  // Gufo's count (includes a deferred token)
  std::int64_t deferred;
  std::uint32_t source_position;
  std::int32_t token;
  std::int32_t has_prediction;
};
static_assert(sizeof(ImageHeader) <= kImageHeaderBytes);

// Rows are equal slices of the image's page bytes, small enough for Gufo's
// disk staging buffer (one token's slice of a page when it divides evenly).
inline constexpr std::size_t kMaxImageRowBytes = std::size_t{512} << 10;
struct ImageLayout {
  std::uint32_t block_size;
  std::size_t fixed_bytes;
  std::size_t page_bytes;
  std::size_t rows_per_page;
  [[nodiscard]] std::size_t private_bytes() const {
    return kImageHeaderBytes + fixed_bytes + page_bytes;
  }
  [[nodiscard]] std::size_t row_bytes() const {
    return page_bytes / rows_per_page;
  }
  // The smallest even split of a page into rows that fit.
  static std::optional<std::size_t> RowsPerPage(std::size_t page_bytes,
                                                std::size_t block_size) {
    if (page_bytes % block_size == 0 &&
        page_bytes / block_size <= kMaxImageRowBytes)
      return block_size;
    for (std::size_t rows = 1; rows <= page_bytes; ++rows)
      if (page_bytes % rows == 0 && page_bytes / rows <= kMaxImageRowBytes)
        return rows;
    return std::nullopt;
  }
};

class DoneSignal final : public cache::CompletionSignal {
public:
  explicit DoneSignal(cache::TransferResult result) : result_(result) {}
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return result_;
  }

private:
  cache::TransferResult result_;
};

inline cache::Completion Done(bool ok) {
  return cache::Completion(std::make_unique<DoneSignal>(
      ok ? cache::TransferResult::kSucceeded : cache::TransferResult::kFailed));
}

struct Endpoint {
  std::uint64_t checkpoint;
  std::int64_t deferred;
};

// Owns the bytes of one cache payload; the checkpoint they name lives as
// long as Gufo keeps the payload.
struct EndpointPayload {
  std::shared_ptr<LoomModel> model;
  std::array<std::byte, kMarkerBytes> bytes{};
  ~EndpointPayload() {
    Endpoint endpoint{};
    std::memcpy(&endpoint, bytes.data(), sizeof(endpoint));
    if (endpoint.checkpoint != 0)
      model->Release(endpoint.checkpoint);
  }
};

class LoomAdapter final : public cache::Adapter {
public:
  LoomAdapter(std::shared_ptr<LoomModel> model, std::uint32_t rows,
              std::optional<ImageLayout> images)
      : model_(std::move(model)),
        pool_(std::make_shared<RowPool>()),
        images_(images) {
    pool_->used.assign(rows, false);
    if (images_) {
      image_components_ = {
          cache::ComponentDescriptor{
              .id = kImageState,
              .layout_version = 1,
              .kind = cache::ComponentKind::kPrivateState,
              .state_bytes = images_->private_bytes()},
          cache::ComponentDescriptor{
              .id = kImagePages,
              .layout_version = 1,
              .kind = cache::ComponentKind::kAppendRows,
              .row_bytes = images_->row_bytes(),
              .rows_per_chunk = kPagesPerChunk * images_->rows_per_page}};
    }
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    // Endpoint mode names device checkpoints (never written to disk); image
    // mode carries portable bytes.
    return {.continuation = true, .persistent_encoding = images_.has_value()};
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    if (images_)
      return image_components_;
    return std::span(&kEndpointDescriptor, 1);
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    std::string id = std::string("loom-text-2:") + model_->info().model_id;
    if (images_)
      id += ":image-1:" + std::to_string(images_->block_size) + ":" +
            std::to_string(images_->rows_per_page) + ":" +
            std::to_string(images_->fixed_bytes) + ":" +
            std::to_string(images_->page_bytes) + ":" + model_->Fingerprint();
    return cache::Identity(id.begin(), id.end());
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override {
    std::lock_guard lock(pool_->mutex);
    for (std::uint32_t row = 0; row < pool_->used.size(); ++row) {
      if (!pool_->used[row]) {
        pool_->used[row] = true;
        return std::make_unique<LoomSlot>(row, pool_);
      }
    }
    throw cache::ResourceExhausted();
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override {
    const auto& loom = dynamic_cast<const LoomSlot&>(slot);
    if (images_) {
      const std::size_t consumed = loom.position - (loom.deferred >= 0 ? 1 : 0);
      return {{.id = kImageState, .valid_rows = loom.position},
              {.id = kImagePages,
               .valid_rows =
                   consumed / images_->block_size * images_->rows_per_page}};
    }
    return {{.id = kEndpoint, .valid_rows = loom.position}};
  }
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override {
    std::vector<cache::Rows> boundaries;
    if (first >= end)
      return boundaries;
    const cache::Rows step =
        std::max<cache::Rows>(model_->info().prefill_capacity, 1);
    for (cache::Rows at = first + step; at < end; at += step)
      boundaries.push_back(at);
    boundaries.push_back(end);
    return boundaries;
  }
  void BeginRestore(
      cache::Slot& slot,
      std::span<const cache::ComponentPosition> positions) override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    loom.restore_ok = false;
    loom.position = positions.empty() ? 0 : positions.front().valid_rows;
    if (images_) {
      std::size_t pages = 0;
      for (const auto& position : positions) {
        if (position.id == kImageState)
          loom.position = position.valid_rows;
        if (position.id == kImagePages)
          pages = position.valid_rows;
      }
      loom.restore_private.assign(images_->private_bytes(), 0);
      loom.restore_private_bytes = 0;
      loom.restore_pages.assign(pages * images_->row_bytes(), 0);
      loom.restore_have.assign(pages, false);
    }
  }

  // The row's image, exported through a short-lived pin and kept until the
  // row moves on.
  const std::vector<std::uint8_t>& ImageLocked(const LoomSlot& slot) const {
    auto& loom = const_cast<LoomSlot&>(slot);
    auto* embed = model_->embed();
    const std::uint64_t position =
        loom_text_embed_row_position(embed, loom.row());
    if (loom.image_generation == loom.generation &&
        loom.image_position == position)
      return loom.image;
    loom.image.clear();
    loom.image_generation = ~std::uint64_t{0};
    if (position == 0)
      throw std::runtime_error("loom image of an empty row");
    loom_text_embed_checkpoint_t* checkpoint = nullptr;
    if (model_->Check(loom_text_embed_row_pin(embed, loom.row(), &checkpoint),
                      "pin", 1) != 0)
      throw cache::ResourceExhausted();
    std::size_t size = 0;
    loom_text_embed_checkpoint_info_t info{};
    loom_text_embed_checkpoint_export(embed, checkpoint, nullptr, 0, &size,
                                      &info);
    loom.image.resize(size);
    const int status = loom_text_embed_checkpoint_export(
        embed, checkpoint, loom.image.data(), size, &size, &info);
    loom_text_embed_checkpoint_release(embed, checkpoint);
    model_->Check(status, "export checkpoint");
    const ImageHeader header{.magic = kImageMagic,
                             .loom_position = info.position,
                             .position = loom.position,
                             .deferred = loom.deferred,
                             .source_position = info.source_position,
                             .token = info.token,
                             .has_prediction = info.has_prediction};
    loom.image_header.assign(kImageHeaderBytes, 0);
    std::memcpy(loom.image_header.data(), &header, sizeof(header));
    loom.image_generation = loom.generation;
    loom.image_position = position;
    return loom.image;
  }

  // Header, fixed state and the partial last page (zeros when the position
  // ends a page), as the private component holds them.
  void PrivateBytesLocked(const LoomSlot& slot,
                          std::span<std::uint8_t> out) const {
    const auto& image = ImageLocked(slot);
    std::fill(out.begin(), out.end(), 0);
    std::memcpy(out.data(), slot.image_header.data(), kImageHeaderBytes);
    std::memcpy(out.data() + kImageHeaderBytes, image.data(),
                images_->fixed_bytes);
    ImageHeader header{};
    std::memcpy(&header, slot.image_header.data(), sizeof(header));
    if (header.loom_position % images_->block_size != 0) {
      const std::size_t page = header.loom_position / images_->block_size;
      std::memcpy(
          out.data() + kImageHeaderBytes + images_->fixed_bytes,
          image.data() + images_->fixed_bytes + page * images_->page_bytes,
          images_->page_bytes);
    }
  }
  [[nodiscard]] cache::Completion CapturePrivatePiece(
      const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
      std::span<std::byte> bytes, cache::Stream& stream) override {
    if (!images_)
      return cache::Adapter::CapturePrivatePiece(slot, id, offset, bytes,
                                                 stream);
    const auto& loom = dynamic_cast<const LoomSlot&>(slot);
    std::lock_guard lock(model_->mutex());
    std::vector<std::uint8_t> all(images_->private_bytes());
    try {
      PrivateBytesLocked(loom, all);
    } catch (const std::exception&) {
      return Done(false);
    }
    if (offset > all.size() || bytes.size() > all.size() - offset)
      return Done(false);
    std::memcpy(bytes.data(), all.data() + offset, bytes.size());
    return Done(true);
  }
  [[nodiscard]] cache::Completion CapturePrivate(
      const cache::Slot& slot, cache::ComponentId id,
      std::span<std::byte> bytes, cache::Stream& stream) override {
    if (images_)
      return CapturePrivatePiece(slot, id, 0, bytes, stream);
    const auto& loom = dynamic_cast<const LoomSlot&>(slot);
    std::lock_guard lock(model_->mutex());
    // Loom pins only a nonempty row.
    if (loom_text_embed_row_position(model_->embed(), loom.row()) == 0)
      return Done(false);
    loom_text_embed_checkpoint_t* checkpoint = nullptr;
    if (model_->Check(
            loom_text_embed_row_pin(model_->embed(), loom.row(), &checkpoint),
            "pin", 1) != 0)
      return Done(false);
    const Endpoint endpoint{model_->Keep(checkpoint), loom.deferred};
    std::memcpy(bytes.data(), &endpoint,
                std::min(bytes.size(), sizeof(endpoint)));
    return Done(true);
  }
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot& slot, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte> bytes, cache::Stream&) override {
    if (!images_)
      throw std::logic_error("loom endpoints have no rows");
    const auto& loom = dynamic_cast<const LoomSlot&>(slot);
    std::lock_guard lock(model_->mutex());
    try {
      const auto& image = ImageLocked(loom);
      const std::size_t begin =
          images_->fixed_bytes + first * images_->row_bytes();
      const std::size_t size = (end - first) * images_->row_bytes();
      if (end < first || begin + size > image.size() || bytes.size() < size)
        return Done(false);
      std::memcpy(bytes.data(), image.data() + begin, size);
    } catch (const std::exception&) {
      return Done(false);
    }
    return Done(true);
  }
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot& slot,
                                             cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte> bytes,
                                             cache::Stream&) override {
    if (!images_)
      throw std::logic_error("loom endpoints have no rows");
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    const std::size_t size = (end - first) * images_->row_bytes();
    if (end < first || end > loom.restore_have.size() || bytes.size() < size)
      return Done(false);
    std::memcpy(loom.restore_pages.data() + first * images_->row_bytes(),
                bytes.data(), size);
    for (cache::Rows row = first; row < end; ++row)
      loom.restore_have[row] = true;
    return Done(true);
  }
  [[nodiscard]] cache::Completion LoadPrivatePiece(
      cache::Slot& slot, cache::ComponentId id, std::size_t offset,
      std::span<const std::byte> bytes, cache::Stream& stream) override {
    if (!images_)
      return cache::Adapter::LoadPrivatePiece(slot, id, offset, bytes, stream);
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    if (offset > loom.restore_private.size() ||
        bytes.size() > loom.restore_private.size() - offset)
      return Done(false);
    std::memcpy(loom.restore_private.data() + offset, bytes.data(),
                bytes.size());
    loom.restore_private_bytes += bytes.size();
    return Done(true);
  }
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot& slot,
                                              cache::ComponentId id,
                                              std::span<const std::byte> bytes,
                                              cache::Stream& stream) override {
    if (images_)
      return LoadPrivatePiece(slot, id, 0, bytes, stream);
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    Endpoint endpoint{};
    std::memcpy(&endpoint, bytes.data(),
                std::min(bytes.size(), sizeof(endpoint)));
    std::lock_guard lock(model_->mutex());
    const auto* checkpoint = model_->Find(endpoint.checkpoint);
    int restored = 0;
    if (checkpoint != nullptr)
      model_->Check(loom_text_embed_row_restore(model_->embed(), loom.row(),
                                                checkpoint, &restored),
                    "restore");
    loom.restore_ok = restored != 0;
    loom.deferred = endpoint.deferred;
    loom.stale = !loom.restore_ok;
    return Done(loom.restore_ok);
  }
  [[nodiscard]] bool Validate(
      cache::Slot& slot, std::span<const cache::ComponentPosition>) override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    if (images_ && !loom.restore_ok)
      loom.restore_ok = RestoreImage(loom);
    return loom.restore_ok;
  }

  // Assembles the image from its private bytes and pages, loads it into a
  // fresh checkpoint and restores the row from it.
  bool RestoreImage(LoomSlot& loom) {
    if (loom.restore_private_bytes < loom.restore_private.size() ||
        std::find(loom.restore_have.begin(), loom.restore_have.end(), false) !=
            loom.restore_have.end())
      return false;
    ImageHeader header{};
    std::memcpy(&header, loom.restore_private.data(), sizeof(header));
    const std::size_t pages = loom.restore_have.size() / images_->rows_per_page;
    if (header.magic != kImageMagic || header.position != loom.position ||
        loom.restore_have.size() % images_->rows_per_page != 0 ||
        header.loom_position / images_->block_size != pages ||
        header.loom_position == 0)
      return false;
    const bool partial = header.loom_position % images_->block_size != 0;
    std::vector<std::uint8_t> image;
    image.reserve(images_->fixed_bytes +
                  (pages + (partial ? 1 : 0)) * images_->page_bytes);
    const auto* fixed = loom.restore_private.data() + kImageHeaderBytes;
    image.insert(image.end(), fixed, fixed + images_->fixed_bytes);
    image.insert(image.end(), loom.restore_pages.begin(),
                 loom.restore_pages.end());
    if (partial)
      image.insert(image.end(), fixed + images_->fixed_bytes,
                   fixed + images_->fixed_bytes + images_->page_bytes);
    const loom_text_embed_checkpoint_info_t info{
        .position = header.loom_position,
        .source_position = header.source_position,
        .token = header.token,
        .has_prediction = header.has_prediction};
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    loom_text_embed_checkpoint_t* checkpoint = nullptr;
    if (model_->Check(
            loom_text_embed_checkpoint_import(embed, &info, image.data(),
                                              image.size(), &checkpoint),
            "import checkpoint", 1) != 0)
      return false;
    int restored = 0;
    const int status =
        loom_text_embed_row_restore(embed, loom.row(), checkpoint, &restored);
    loom_text_embed_checkpoint_release(embed, checkpoint);
    model_->Check(status, "restore");
    ++loom.generation;
    loom.deferred = header.deferred;
    loom.stale = restored == 0;
    loom.restore_private.clear();
    loom.restore_pages.clear();
    loom.restore_have.clear();
    return restored != 0;
  }
  [[nodiscard]] bool Invalidate(cache::Slot& slot) noexcept override {
    auto& loom = dynamic_cast<LoomSlot&>(slot);
    ++loom.generation;
    loom.image.clear();
    loom.restore_private.clear();
    loom.restore_pages.clear();
    loom.restore_have.clear();
    loom.position = 0;
    loom.deferred = -1;
    loom.restore_ok = false;
    loom.stale = true;
    return true;
  }

private:
  std::shared_ptr<LoomModel> model_;
  std::shared_ptr<RowPool> pool_;
  std::optional<ImageLayout> images_;
  std::array<cache::ComponentDescriptor, 2> image_components_{};
};

class NullStream final : public cache::Stream {
public:
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};

}  // namespace gufo::loom

#endif  // GUFO_LOOM_CACHE_HPP_
