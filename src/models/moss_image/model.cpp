#include "src/models/moss_image/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/qwen_image_21/weights.hpp"

namespace gufo::models::moss_image {
namespace {

// torch.linspace in FP32: the first half counts up from start, the second
// half down from end, both with the same FP32 step.
std::vector<float> Linspace(float start, float end, int count) {
  std::vector<float> values(count);
  if (count == 1) {
    values[0] = start;
    return values;
  }
  const float step = (end - start) / static_cast<float>(count - 1);
  const int half = count / 2;
  for (int i = 0; i < count; ++i)
    values[i] = i < half ? start + step * static_cast<float>(i)
                         : end - step * static_cast<float>(count - 1 - i);
  return values;
}

// MOSSImageScheduler.set_timesteps' grid(count, shift).
std::vector<float> Grid(int count, float shift) {
  constexpr int kTrainSteps = 1000;
  const float lower = static_cast<float>(static_cast<double>(shift) /
                                         (kTrainSteps + shift - 1));
  auto values = Linspace(1.0F, lower, count);
  for (float& v : values)
    v = shift * v / (1.0F + (shift - 1.0F) * v);
  return values;
}

}  // namespace

bool IsMossImageDirectory(const std::filesystem::path& root) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(root / "model_index.json", error))
    return false;
  try {
    return qwen_image_21::ReadJson(root / "model_index.json")
               .member_str("_class_name") == "MOSSImagePipeline";
  } catch (const std::exception&) {
    return false;
  }
}

void ValidateRequest(const Request& request) {
  if (!request.images.empty())
    throw std::invalid_argument("MOSS-Image does not support reference images");
  if (request.width < 256 || request.height < 256 || request.width > 2048 ||
      request.height > 2048 || request.width % 16 || request.height % 16)
    throw std::invalid_argument(
        "MOSS-Image size must be 256..2048 pixels in multiples of 16");
  if (request.steps < 1 || request.steps > 100)
    throw std::invalid_argument("MOSS-Image steps must be between 1 and 100");
}

std::vector<float> FlowSigmas(int steps) {
  constexpr int kTrainSteps = 1000;
  const auto sigmas = Grid(steps, 5.0F);
  auto training = Grid(kTrainSteps, 1.0F);
  std::reverse(training.begin(), training.end());
  std::vector<float> snapped;
  for (const float sigma : sigmas) {
    // torch.searchsorted (left) over the ascending training grid.
    const int upper =
        std::min(static_cast<int>(
                     std::lower_bound(training.begin(), training.end(), sigma) -
                     training.begin()),
                 kTrainSteps - 1);
    const int lower = std::max(upper - 1, 0);
    const int nearest =
        std::fabs(sigma - training[lower]) <= std::fabs(training[upper] - sigma)
            ? lower
            : upper;
    // torch.unique_consecutive.
    if (snapped.empty() || snapped.back() != training[nearest])
      snapped.push_back(training[nearest]);
  }
  snapped.push_back(0.0F);
  return snapped;
}

}  // namespace gufo::models::moss_image
