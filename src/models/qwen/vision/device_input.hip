#include <algorithm>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models::qwen::vision {
namespace {
void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

__global__ void InjectRows(float* hidden, const float* embedding,
                           std::uint32_t rows, std::uint32_t width,
                           std::uint32_t hc) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width * hc)
    return;
  const auto row = i / (width * hc);
  hidden[i] = embedding[row * width + i % width];
}
}  // namespace

DeviceInput::~DeviceInput() {
  if (descriptor_ != nullptr)
    (void)hipFree(descriptor_);
  if (positions_ != nullptr)
    (void)hipFree(positions_);
}

void DeviceInput::Configure(std::shared_ptr<const Prompt> prompt,
                            std::shared_ptr<Encoder> encoder,
                            hipStream_t stream) {
  if (continuation_hooks_) continuation_hooks_->BeforeVision(prompt.get(), nullptr);
  if (prompt && !prompt->images.empty() && !encoder) {
    throw std::invalid_argument("image input has no vision encoder");
  }
  const bool reuse = prompt_ && prompt && !prompt->cache_identity.empty() &&
                     encoder_ == encoder &&
                     prompt_->cache_identity == prompt->cache_identity;
  prompt_ = std::move(prompt);
  encoder_ = std::move(encoder);
  if (!reuse)
    embeddings_.clear();
  if (prompt_)
    embeddings_.resize(prompt_->images.size());
  RestoreLayout(prompt_ ? prompt_->rope : RopeLayout{}, stream);
}

void DeviceInput::RestoreLayout(const RopeLayout& layout, hipStream_t stream) {
  if (continuation_hooks_) continuation_hooks_->BeforeVision(nullptr, &layout);
  layout.Validate(std::numeric_limits<std::int32_t>::max());
  if (layout == layout_ && (descriptor_ != nullptr || layout.images.empty()))
    return;
  const auto length = layout.PrefixLength();
  if (descriptor_ == nullptr)
    Check(hipMalloc(&descriptor_, sizeof(DeviceRope)));
  if (length > capacity_) {
    Check(hipStreamSynchronize(stream));
    if (positions_ != nullptr)
      Check(hipFree(positions_));
    positions_ = nullptr;
    capacity_ = 0;
    Check(
        hipMalloc(&positions_, std::size_t{length} * 3 * sizeof(std::int32_t)));
    capacity_ = length;
  }
  std::vector<std::array<std::int32_t, 3>> positions(length);
  for (std::uint32_t token = 0; token < length; ++token)
    positions[token] = layout.Position(token);
  if (length != 0) {
    Check(hipMemcpyAsync(positions_, positions.data(),
                         positions.size() * sizeof(positions[0]),
                         hipMemcpyHostToDevice, stream));
  }
  const DeviceRope descriptor{positions_, length, layout.Delta()};
  Check(hipMemcpyAsync(descriptor_, &descriptor, sizeof(descriptor),
                       hipMemcpyHostToDevice, stream));
  Check(hipStreamSynchronize(stream));
  layout_ = layout;
}

void DeviceInput::Inject(float* hidden, std::uint32_t position,
                         std::uint32_t count, std::uint32_t width,
                         std::uint32_t hc, hipStream_t stream) {
  if (!prompt_)
    return;
  for (std::size_t index = 0; index < prompt_->images.size(); ++index) {
    const auto& image = prompt_->images[index];
    const auto image_end =
        image.grid.offset + image.grid.height * image.grid.width;
    const auto begin = std::max(position, image.grid.offset);
    const auto end = std::min(position + count, image_end);
    if (begin >= end)
      continue;
    if (!embeddings_[index])
      embeddings_[index] = encoder_->Encode(image.pixels, {}, is_cancelled_);
    const auto& embedding = *embeddings_[index];
    if (embedding.width() != width ||
        embedding.rows() != image.grid.height * image.grid.width) {
      throw std::logic_error("vision embedding does not match language input");
    }
    const auto rows = end - begin;
    InjectRows<<<(rows * width * hc + 255) / 256, 256, 0, stream>>>(
        hidden + std::size_t{begin - position} * width * hc,
        embedding.data() + std::size_t{begin - image.grid.offset} * width, rows,
        width, hc);
    Check(hipGetLastError());
  }
}
}  // namespace gufo::models::qwen::vision
