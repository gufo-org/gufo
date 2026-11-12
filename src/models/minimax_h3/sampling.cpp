// Host geometry, packing, schedule, and RNG behavior adapted from antirez/h3.c
// h3_host.c and h3_dit.c at 8974cc055ea9c02fcd14cc27dfda3e1027c05153
// (MIT). The C++ ownership model and ROCm integration are independent code.

#include "src/models/minimax_h3/sampling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include "src/models/minimax_h3/sha256.hpp"

namespace gufo::minimax_h3 {
namespace {

constexpr std::array<int, 5> kFramesPerToken = {1, 4, 4, 4, 4};
constexpr double kFrameRescale = 5.0 / 3.0;
constexpr float kVideoSigmaShift = 12.0F;
constexpr float kAudioSigmaShift = 3.0F;
constexpr std::uint32_t kDitModalities = 3;

float ShiftSigma(float base, float shift) noexcept {
  // Diffusers evaluates these as distinct eager float32 tensor operations.
  // Volatile temporaries prevent the host compiler from contracting the
  // denominator multiply/add into an FMA.
  const volatile float numerator = shift * base;
  const volatile float shifted_offset = (shift - 1.0F) * base;
  const volatile float denominator = 1.0F + shifted_offset;
  const volatile float result = numerator / denominator;
  return result;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool CheckedMultiply(std::size_t left, std::size_t right, std::size_t* output) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

bool VideoShape(int channels, int time, int height, int width,
                std::size_t* elements) {
  if (channels < 1 || time < 1 || height < 2 || width < 2 || height % 2 != 0 ||
      width % 2 != 0) {
    return false;
  }
  std::size_t count = 0;
  std::size_t partial = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(channels),
                       static_cast<std::size_t>(time), &partial) ||
      !CheckedMultiply(partial, static_cast<std::size_t>(height), &partial) ||
      !CheckedMultiply(partial, static_cast<std::size_t>(width), &count)) {
    return false;
  }
  *elements = count;
  return true;
}

std::vector<Position3D> FrameGrid(int latent_height, int latent_width) {
  const std::size_t rows = static_cast<std::size_t>(latent_height / 2);
  const std::size_t columns = static_cast<std::size_t>(latent_width / 2);
  std::vector<Position3D> grid(rows * columns);
  const double area =
      std::sqrt(static_cast<double>(latent_height) * latent_width);
  const double ratio_height = latent_height / area;
  const double ratio_width = latent_width / area;
  const double step_height = ratio_height / static_cast<double>(rows);
  const double step_width = ratio_width / static_cast<double>(columns);
  const double base_height = (1.0 - ratio_height) / 2.0;
  const double base_width = (1.0 - ratio_width) / 2.0;
  std::size_t offset = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const double height =
        (static_cast<double>(row) * step_height + base_height) * 32.0;
    for (std::size_t column = 0; column < columns; ++column) {
      const double width =
          (static_cast<double>(column) * step_width + base_width) * 32.0;
      grid[offset++] = {0.0, height, width};
    }
  }
  return grid;
}

void AppendLittleEndian(std::vector<unsigned char>* output,
                        std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    output->push_back(static_cast<unsigned char>((value >> shift) & 0xFFU));
  }
}

void AppendString(std::vector<unsigned char>* output, std::string_view value) {
  output->insert(output->end(), value.begin(), value.end());
}

}  // namespace

int AlignFrameCount(int requested_frames) noexcept {
  int value = std::max(requested_frames, 5);
  int remainder = (value - 5) % 17;
  if (remainder < 0) {
    remainder += 17;
  }
  if (remainder != 0) {
    value += 17 - remainder;
  }
  return value;
}

int VideoLatentFrameCount(int frame_count) noexcept {
  if (frame_count <= 5) {
    return 2;
  }
  return ((frame_count - 5) / 17) * 5 + 2;
}

int VideoEncoderLatentFrameCount(int frame_count) noexcept {
  return frame_count > 0 ? (frame_count + 3) / 4 : 0;
}

namespace {

// Python round() on a double: nearest, ties to even.
double RoundHalfEven(double value) noexcept {
  const double floored = std::floor(value);
  const double fraction = value - floored;
  if (fraction > 0.5) {
    return floored + 1.0;
  }
  if (fraction < 0.5) {
    return floored;
  }
  return std::fmod(floored, 2.0) == 0.0 ? floored : floored + 1.0;
}

}  // namespace

std::optional<CanvasSize> ResolveCanvasSize(double aspect_width,
                                            double aspect_height,
                                            int short_edge,
                                            std::int64_t max_pixels,
                                            std::string* error) {
  // modular_pipeline.py resolve_canvas_size, step for step: `** 0.5` is C
  // pow() in CPython, kept here rather than sqrt() for bit parity.
  if (!(aspect_width > 0.0) || !(aspect_height > 0.0) ||
      !std::isfinite(aspect_width) || !std::isfinite(aspect_height) ||
      short_edge < kH3CanvasMultiple || max_pixels <= 0) {
    SetError(error, "MiniMax H3 canvas aspect ratio must be positive");
    return std::nullopt;
  }
  const double ratio = aspect_width / aspect_height;
  if (ratio < kH3MinimumAspectRatio || ratio > kH3MaximumAspectRatio) {
    SetError(error,
             "MiniMax H3 supports aspect ratios from 1:4 to 4:1; the "
             "keyframe is outside that range");
    return std::nullopt;
  }
  double width = 0.0;
  double height = 0.0;
  if (ratio >= 1.0) {
    width = static_cast<double>(short_edge) * ratio;
    height = static_cast<double>(short_edge);
  } else {
    width = static_cast<double>(short_edge);
    height = static_cast<double>(short_edge) / ratio;
  }
  const double area = width * height;
  if (area > static_cast<double>(max_pixels)) {
    const double scale = std::pow(static_cast<double>(max_pixels) / area, 0.5);
    width *= scale;
    height *= scale;
  }
  const double multiple = kH3CanvasMultiple;
  return CanvasSize{
      .width = std::max(kH3CanvasMultiple,
                        static_cast<int>(RoundHalfEven(width / multiple)) *
                            kH3CanvasMultiple),
      .height = std::max(kH3CanvasMultiple,
                         static_cast<int>(RoundHalfEven(height / multiple)) *
                             kH3CanvasMultiple),
  };
}

std::optional<GenerationGeometry> ResolveGenerationGeometry(
    int width, int height, int requested_frames, std::string* error) {
  if (width < kH3CanvasMultiple || height < kH3CanvasMultiple ||
      width % kH3CanvasMultiple != 0 || height % kH3CanvasMultiple != 0) {
    SetError(error,
             "MiniMax H3 width and height must be multiples of 32 and at "
             "least 32");
    return std::nullopt;
  }
  if (static_cast<std::int64_t>(width) * height > kH3MaximumPixels) {
    SetError(error,
             "MiniMax H3 canvas exceeds the largest canvas of the released "
             "768*1344 rule (1856x576 pixels)");
    return std::nullopt;
  }
  if (requested_frames < 5 || requested_frames > kH3MaximumFrames) {
    SetError(error,
             "MiniMax H3 requested frame count must be in the released "
             "5..345 range");
    return std::nullopt;
  }
  const int frames = AlignFrameCount(requested_frames);
  if (frames > kH3MaximumFrames) {
    SetError(error,
             "MiniMax H3 aligned frame count exceeds the released 15-second "
             "limit (345 aligned frames)");
    return std::nullopt;
  }
  GenerationGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.latent_width = width / kH3VaeSpatialRatio;
  geometry.latent_height = height / kH3VaeSpatialRatio;
  geometry.temporal.frames = frames;
  geometry.temporal.video_latent_frames = VideoLatentFrameCount(frames);
  geometry.temporal.audio_latent_frames = static_cast<int>(
      std::llround(static_cast<double>(frames) * kH3AudioLatentFramesPerSecond /
                   kH3FramesPerSecond));
  return geometry;
}

std::optional<SigmaSchedule> BuildServingSchedule(int evaluations,
                                                  std::string* error) {
  if (evaluations < 2 || evaluations > kH3MaximumEvaluations) {
    SetError(error,
             "MiniMax H3 evaluation count must be in the range [2, 1000]");
    return std::nullopt;
  }
  SigmaSchedule schedule;
  try {
    schedule.video.resize(static_cast<std::size_t>(evaluations) + 1);
    schedule.audio.resize(static_cast<std::size_t>(evaluations) + 1);
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory allocating MiniMax H3 sigma schedule");
    return std::nullopt;
  }
  // Match torch.linspace(1, 0, evaluations + 1, dtype=float32), including its
  // halfway split and fused multiply-add rounding. The split computes the
  // second half from the exact endpoint, avoiding accumulated endpoint error.
  const int points = evaluations + 1;
  const int halfway = points / 2;
  const float step = -1.0F / static_cast<float>(evaluations);
  for (int index = 0; index <= evaluations; ++index) {
    const float base =
        index < halfway
            ? std::fma(step, static_cast<float>(index), 1.0F)
            : std::fma(-step, static_cast<float>(evaluations - index), 0.0F);
    schedule.video[static_cast<std::size_t>(index)] =
        ShiftSigma(base, kVideoSigmaShift);
    schedule.audio[static_cast<std::size_t>(index)] =
        ShiftSigma(base, kAudioSigmaShift);
  }
  schedule.video.back() = 0.0F;
  schedule.audio.back() = 0.0F;
  return schedule;
}

std::optional<TimeRowSchedule> BuildTimeRowSchedule(
    const SigmaSchedule& schedule, std::string* error) {
  const int evaluations = schedule.evaluations();
  if (evaluations < 2 || schedule.audio.size() != schedule.video.size() ||
      evaluations > kH3MaximumEvaluations) {
    SetError(error, "invalid MiniMax H3 sigma schedule");
    return std::nullopt;
  }
  if (schedule.video.back() != 0.0F || schedule.audio.back() != 0.0F) {
    SetError(error, "MiniMax H3 sigma schedule must terminate at zero");
    return std::nullopt;
  }
  for (int step = 0; step < evaluations; ++step) {
    const std::size_t index = static_cast<std::size_t>(step);
    if (!std::isfinite(schedule.video[index]) ||
        !std::isfinite(schedule.audio[index]) ||
        schedule.video[index] <= schedule.video[index + 1] ||
        schedule.audio[index] <= schedule.audio[index + 1] ||
        schedule.video[index] > 1.0F || schedule.audio[index] > 1.0F ||
        schedule.video[index + 1] < 0.0F || schedule.audio[index + 1] < 0.0F) {
      SetError(error, "MiniMax H3 sigma schedule is not finite and decreasing");
      return std::nullopt;
    }
  }
  TimeRowSchedule rows;
  try {
    rows.video_rows.resize(static_cast<std::size_t>(evaluations));
    rows.audio_rows.resize(static_cast<std::size_t>(evaluations));
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory allocating MiniMax H3 timestep rows");
    return std::nullopt;
  }
  std::uint32_t count = 0;
  for (int step = 0; step < evaluations; ++step) {
    const float video = 1.0F - schedule.video[static_cast<std::size_t>(step)];
    const float audio = 1.0F - schedule.audio[static_cast<std::size_t>(step)];
    if (video == audio) {
      rows.video_rows[static_cast<std::size_t>(step)] = count;
      rows.audio_rows[static_cast<std::size_t>(step)] = count++;
    } else if (video < audio) {
      rows.video_rows[static_cast<std::size_t>(step)] = count++;
      rows.audio_rows[static_cast<std::size_t>(step)] = count++;
    } else {
      rows.audio_rows[static_cast<std::size_t>(step)] = count++;
      rows.video_rows[static_cast<std::size_t>(step)] = count++;
    }
  }
  rows.time_rows = count;
  return rows;
}

std::optional<TimeRowSchedule> BuildConditionedTimeRowSchedule(
    const SigmaSchedule& schedule, std::string* error) {
  std::optional<TimeRowSchedule> rows = BuildTimeRowSchedule(schedule, error);
  if (!rows) {
    return std::nullopt;
  }
  const int evaluations = schedule.evaluations();
  rows->condition_rows.resize(static_cast<std::size_t>(evaluations));
  std::uint32_t shared = 0;
  float shared_value = 0.0F;
  bool have_shared = false;
  for (int step = 0; step < evaluations; ++step) {
    const std::size_t index = static_cast<std::size_t>(step);
    // Reference: conditioned rows carry max(video timestep, 0.999) every
    // step, deduplicated per forward by exact float equality (torch.unique);
    // mirrors the video == audio branch above.
    const float video = 1.0F - schedule.video[index];
    const float audio = 1.0F - schedule.audio[index];
    const float condition = video > kH3KeyframeConditionTimestep
                                ? video
                                : kH3KeyframeConditionTimestep;
    if (condition == video) {
      rows->condition_rows[index] = rows->video_rows[index];
    } else if (condition == audio) {
      rows->condition_rows[index] = rows->audio_rows[index];
    } else if (have_shared && condition == shared_value) {
      rows->condition_rows[index] = shared;
    } else {
      shared = rows->time_rows++;
      shared_value = condition;
      have_shared = true;
      rows->condition_rows[index] = shared;
    }
  }
  return rows;
}

std::string_view ToString(SegmentKind kind) noexcept {
  switch (kind) {
    case SegmentKind::kText:
      return "text";
    case SegmentKind::kAudioTarget:
      return "audio";
    case SegmentKind::kVideoTarget:
      return "video";
    case SegmentKind::kVideoCondition:
      return "video-condition";
  }
  return "unknown";
}

std::optional<PackedLayout> BuildLayout(std::size_t text_rows,
                                        const GenerationGeometry& geometry,
                                        std::span<const double> anchor_temporal,
                                        std::string* error) {
  if (text_rows == 0 || text_rows > std::numeric_limits<std::uint32_t>::max() ||
      geometry.width < kH3CanvasMultiple ||
      geometry.height < kH3CanvasMultiple ||
      geometry.width % kH3CanvasMultiple != 0 ||
      geometry.height % kH3CanvasMultiple != 0 ||
      static_cast<std::int64_t>(geometry.width) * geometry.height >
          kH3MaximumPixels ||
      geometry.latent_width != geometry.width / kH3VaeSpatialRatio ||
      geometry.latent_height != geometry.height / kH3VaeSpatialRatio ||
      geometry.temporal.video_latent_frames < 1 ||
      geometry.temporal.audio_latent_frames < 0 ||
      geometry.temporal.frames < 5 ||
      geometry.temporal.frames > kH3MaximumFrames ||
      (geometry.temporal.frames - 5) % 17 != 0 ||
      geometry.temporal.video_latent_frames !=
          VideoLatentFrameCount(geometry.temporal.frames) ||
      geometry.temporal.audio_latent_frames !=
          static_cast<int>(std::llround(
              static_cast<double>(geometry.temporal.frames) *
              kH3AudioLatentFramesPerSecond / kH3FramesPerSecond)) ||
      geometry.latent_height < 2 || geometry.latent_width < 2 ||
      geometry.latent_height % 2 != 0 || geometry.latent_width % 2 != 0) {
    SetError(error, "invalid MiniMax H3 text-only layout geometry");
    return std::nullopt;
  }
  std::size_t frame_rows = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(geometry.latent_height / 2),
                       static_cast<std::size_t>(geometry.latent_width / 2),
                       &frame_rows)) {
    SetError(error, "MiniMax H3 frame row count overflows");
    return std::nullopt;
  }
  std::size_t condition_rows = 0;
  if (!CheckedMultiply(anchor_temporal.size(), frame_rows, &condition_rows)) {
    SetError(error, "MiniMax H3 condition row count overflows");
    return std::nullopt;
  }
  std::size_t video_rows = 0;
  std::size_t audio_rows = 0;
  if (!CheckedMultiply(
          frame_rows,
          static_cast<std::size_t>(geometry.temporal.video_latent_frames),
          &video_rows) ||
      !CheckedMultiply(
          2, static_cast<std::size_t>(geometry.temporal.audio_latent_frames),
          &audio_rows) ||
      text_rows > std::numeric_limits<std::size_t>::max() - condition_rows ||
      text_rows + condition_rows >
          std::numeric_limits<std::size_t>::max() - audio_rows ||
      text_rows + condition_rows + audio_rows >
          std::numeric_limits<std::size_t>::max() - video_rows ||
      text_rows + condition_rows + audio_rows + video_rows >
          std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "MiniMax H3 packed row count overflows");
    return std::nullopt;
  }

  PackedLayout layout;
  layout.text_rows = text_rows;
  layout.audio_target_rows = audio_rows;
  layout.video_condition_rows = condition_rows;
  layout.video_target_rows = video_rows;
  layout.video_latent_frames = geometry.temporal.video_latent_frames;
  layout.latent_height = geometry.latent_height;
  layout.latent_width = geometry.latent_width;
  layout.audio_latent_frames = geometry.temporal.audio_latent_frames;
  try {
    layout.positions.reserve(text_rows + condition_rows + audio_rows +
                             video_rows);
    layout.segments.reserve(4);
    for (std::size_t row = 0; row < text_rows; ++row) {
      layout.positions.push_back({static_cast<double>(row), 0.0, 0.0});
    }
    layout.segments.push_back({0, text_rows, SegmentKind::kText});

    const std::vector<Position3D> frame =
        FrameGrid(geometry.latent_height, geometry.latent_width);
    if (!anchor_temporal.empty()) {
      for (const double anchor_time : anchor_temporal) {
        for (const Position3D& position : frame) {
          layout.positions.push_back(
              {anchor_time, position.height, position.width});
        }
      }
      layout.segments.push_back({text_rows, text_rows + condition_rows,
                                 SegmentKind::kVideoCondition});
    }
    const double low_width = frame.front().width;
    const double high_width =
        frame[static_cast<std::size_t>(geometry.latent_width / 2) - 1].width;
    const double cursor = static_cast<double>(text_rows);
    for (int index = 0; index < geometry.temporal.audio_latent_frames;
         ++index) {
      layout.positions.push_back({cursor + index, 0.0, low_width});
    }
    for (int index = 0; index < geometry.temporal.audio_latent_frames;
         ++index) {
      layout.positions.push_back({cursor + index, 0.0, high_width});
    }
    layout.segments.push_back({text_rows + condition_rows,
                               text_rows + condition_rows + audio_rows,
                               SegmentKind::kAudioTarget});

    double time = cursor;
    for (int index = 0; index < geometry.temporal.video_latent_frames;
         ++index) {
      for (const Position3D& position : frame) {
        layout.positions.push_back({time, position.height, position.width});
      }
      time += kFrameRescale * kFramesPerToken[static_cast<std::size_t>(index) %
                                              kFramesPerToken.size()];
    }
    layout.segments.push_back({text_rows + condition_rows + audio_rows,
                               layout.positions.size(),
                               SegmentKind::kVideoTarget});
  } catch (const std::exception&) {
    SetError(error,
             "MiniMax H3 packed layout cannot be represented or allocated");
    return std::nullopt;
  }
  return layout;
}

std::optional<PackedLayout> BuildTextOnlyLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::string* error) {
  return BuildLayout(text_rows, geometry, {}, error);
}

std::optional<PackedLayout> BuildKeyframeLayout(
    std::size_t text_rows, const GenerationGeometry& geometry,
    std::span<const double> anchor_temporal, std::string* error) {
  if (anchor_temporal.empty()) {
    SetError(error, "MiniMax H3 keyframe layout requires an anchor");
    return std::nullopt;
  }
  return BuildLayout(text_rows, geometry, anchor_temporal, error);
}

namespace {

double NumpyPairwiseSum(std::span<const double> values) {
  // Reproduce numpy 2.5.x np.float64 summation exactly (loops_utils.h.src,
  // DOUBLE_pairwise_sum): fewer than 8 elements accumulate sequentially from
  // -0.0; up to the 128-element block use eight accumulators with the exact
  // fold tree and a sequential tail; longer inputs recurse on a split
  // rounded down to a multiple of 8. The FL2VA last-anchor rotary time is
  // defined by this summation order; a plain loop differs in the last ulp.
  const std::size_t n = values.size();
  if (n < 8U) {
    double res = -0.0;
    for (std::size_t i = 0; i < n; ++i) {
      res += values[i];
    }
    return res;
  }
  if (n <= 128U) {
    double r[8];
    for (std::size_t i = 0; i < 8U; ++i) {
      r[i] = values[i];
    }
    std::size_t i = 8U;
    for (; i + 8U <= n; i += 8U) {
      for (std::size_t lane = 0; lane < 8U; ++lane) {
        r[lane] += values[i + lane];
      }
    }
    double res =
        ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      res += values[i];
    }
    return res;
  }
  std::size_t n2 = n / 2U;
  n2 -= n2 % 8U;
  return NumpyPairwiseSum(values.subspan(0, n2)) +
         NumpyPairwiseSum(values.subspan(n2));
}

}  // namespace

std::optional<std::vector<double>> KeyframeAnchorTimes(std::size_t text_rows,
                                                       int video_latent_frames,
                                                       bool have_first,
                                                       bool have_last,
                                                       std::string* error) {
  if (!have_first && !have_last) {
    return std::vector<double>{};
  }
  if (video_latent_frames < 2 || text_rows == 0U) {
    SetError(error, "invalid MiniMax H3 keyframe anchor geometry");
    return std::nullopt;
  }
  const double text_time = static_cast<double>(text_rows);
  std::vector<double> anchors;
  if (have_first) {
    // The first anchor shares the generated frame-zero rotary time.
    anchors.push_back(text_time);
  }
  if (have_last) {
    // spans = ones(L) * (5/3); spans[o::5] *= (1,4,4,4,4)[o]; the anchor
    // sits one rescale unit before the pairwise sum of all spans.
    std::vector<double> spans(static_cast<std::size_t>(video_latent_frames),
                              kFrameRescale);
    for (std::size_t offset = 0; offset < kFramesPerToken.size(); ++offset) {
      const double factor = static_cast<double>(kFramesPerToken[offset]);
      for (std::size_t index = offset; index < spans.size(); index += 5U) {
        spans[index] *= factor;
      }
    }
    anchors.push_back((text_time + NumpyPairwiseSum(spans)) - kFrameRescale);
  }
  return anchors;
}

std::optional<std::vector<std::uint32_t>> BuildModulationRowMap(
    const TimeRowSchedule& time_rows, int step, const PackedLayout& layout,
    std::span<const std::uint8_t> text_tags, std::string* error) {
  if (step < 0 ||
      static_cast<std::size_t>(step) >= time_rows.video_rows.size() ||
      time_rows.video_rows.size() != time_rows.audio_rows.size() ||
      layout.positions.empty() ||
      (layout.segments.size() != 3 && layout.segments.size() != 4) ||
      (!text_tags.empty() && text_tags.size() != layout.text_rows)) {
    SetError(error, "invalid MiniMax H3 modulation row-map arguments");
    return std::nullopt;
  }
  std::vector<std::uint32_t> rows;
  try {
    rows.resize(layout.rows());
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory allocating MiniMax H3 modulation row map");
    return std::nullopt;
  }
  std::size_t text_index = 0;
  for (const LayoutSegment& segment : layout.segments) {
    if (segment.begin > segment.end || segment.end > rows.size()) {
      SetError(error, "MiniMax H3 layout has invalid segment bounds");
      return std::nullopt;
    }
    std::uint32_t time_row = 0;
    std::uint32_t modality = 0;
    switch (segment.kind) {
      case SegmentKind::kText:
        time_row = time_rows.video_rows[static_cast<std::size_t>(step)];
        for (std::size_t row = segment.begin; row < segment.end; ++row) {
          const std::uint32_t tag =
              text_tags.empty() ? 1U : text_tags[text_index];
          if (tag >= kDitModalities) {
            SetError(error, "MiniMax H3 text tag is outside [0, 2]");
            return std::nullopt;
          }
          rows[row] = time_row * kDitModalities + tag;
          ++text_index;
        }
        continue;
      case SegmentKind::kAudioTarget:
        time_row = time_rows.audio_rows[static_cast<std::size_t>(step)];
        modality = 2;
        break;
      case SegmentKind::kVideoTarget:
        time_row = time_rows.video_rows[static_cast<std::size_t>(step)];
        modality = 0;
        break;
      case SegmentKind::kVideoCondition:
        if (time_rows.condition_rows.size() != time_rows.video_rows.size()) {
          SetError(error, "MiniMax H3 conditioned layout lacks a row schedule");
          return std::nullopt;
        }
        time_row = time_rows.condition_rows[static_cast<std::size_t>(step)];
        modality = 0;
        break;
    }
    const std::uint32_t modulation = time_row * kDitModalities + modality;
    std::fill(rows.begin() + static_cast<std::ptrdiff_t>(segment.begin),
              rows.begin() + static_cast<std::ptrdiff_t>(segment.end),
              modulation);
  }
  if (text_index != layout.text_rows) {
    SetError(error, "MiniMax H3 text segment length changed unexpectedly");
    return std::nullopt;
  }
  return rows;
}

std::string PackedLayoutSha256(const PackedLayout& layout) {
  std::vector<unsigned char> bytes;
  bytes.reserve(128 + layout.segments.size() * 24 +
                layout.positions.size() * 24);
  AppendString(&bytes, "gufo.minimax-h3-layout.v1");
  AppendLittleEndian(&bytes, layout.text_rows);
  AppendLittleEndian(&bytes, layout.audio_target_rows);
  AppendLittleEndian(&bytes, layout.video_target_rows);
  AppendLittleEndian(&bytes,
                     static_cast<std::uint64_t>(layout.video_latent_frames));
  AppendLittleEndian(&bytes, static_cast<std::uint64_t>(layout.latent_height));
  AppendLittleEndian(&bytes, static_cast<std::uint64_t>(layout.latent_width));
  AppendLittleEndian(&bytes,
                     static_cast<std::uint64_t>(layout.audio_latent_frames));
  AppendLittleEndian(&bytes, layout.segments.size());
  for (const LayoutSegment& segment : layout.segments) {
    AppendLittleEndian(&bytes, segment.begin);
    AppendLittleEndian(&bytes, segment.end);
    AppendLittleEndian(&bytes, static_cast<std::uint8_t>(segment.kind));
  }
  AppendLittleEndian(&bytes, layout.positions.size());
  for (const Position3D& position : layout.positions) {
    AppendLittleEndian(&bytes, std::bit_cast<std::uint64_t>(position.temporal));
    AppendLittleEndian(&bytes, std::bit_cast<std::uint64_t>(position.height));
    AppendLittleEndian(&bytes, std::bit_cast<std::uint64_t>(position.width));
  }
  return Sha256(bytes);
}

std::string ModulationRowMapSha256(std::span<const std::uint32_t> rows) {
  std::vector<unsigned char> bytes;
  bytes.reserve(32 + rows.size() * 4);
  AppendString(&bytes, "gufo.minimax-h3-row-map.v1");
  AppendLittleEndian(&bytes, rows.size());
  for (const std::uint32_t row : rows) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<unsigned char>((row >> shift) & 0xFFU));
    }
  }
  return Sha256(bytes);
}

bool PatchifyVideo(std::span<const float> latent, int channels, int time,
                   int height, int width, std::span<float> rows,
                   std::string* error) {
  std::size_t elements = 0;
  if (!VideoShape(channels, time, height, width, &elements) ||
      latent.size() != elements || rows.size() != elements) {
    SetError(error, "invalid MiniMax H3 video patchify geometry");
    return false;
  }
  std::size_t output = 0;
  for (int t = 0; t < time; ++t) {
    for (int h = 0; h < height; h += 2) {
      for (int w = 0; w < width; w += 2) {
        for (int channel = 0; channel < channels; ++channel) {
          for (int delta_h = 0; delta_h < 2; ++delta_h) {
            for (int delta_w = 0; delta_w < 2; ++delta_w) {
              const std::size_t input =
                  (((static_cast<std::size_t>(channel) * time + t) * height +
                    static_cast<std::size_t>(h + delta_h)) *
                       width +
                   static_cast<std::size_t>(w + delta_w));
              rows[output++] = latent[input];
            }
          }
        }
      }
    }
  }
  return output == rows.size();
}

bool UnpatchifyVideo(std::span<const float> rows, int channels, int time,
                     int height, int width, std::span<float> latent,
                     std::string* error) {
  std::size_t elements = 0;
  if (!VideoShape(channels, time, height, width, &elements) ||
      latent.size() != elements || rows.size() != elements) {
    SetError(error, "invalid MiniMax H3 video unpatchify geometry");
    return false;
  }
  std::size_t input = 0;
  for (int t = 0; t < time; ++t) {
    for (int h = 0; h < height; h += 2) {
      for (int w = 0; w < width; w += 2) {
        for (int channel = 0; channel < channels; ++channel) {
          for (int delta_h = 0; delta_h < 2; ++delta_h) {
            for (int delta_w = 0; delta_w < 2; ++delta_w) {
              const std::size_t output =
                  (((static_cast<std::size_t>(channel) * time + t) * height +
                    static_cast<std::size_t>(h + delta_h)) *
                       width +
                   static_cast<std::size_t>(w + delta_w));
              latent[output] = rows[input++];
            }
          }
        }
      }
    }
  }
  return input == rows.size();
}

bool PackAudio(std::span<const float> latent, int channels, int time,
               std::span<float> rows, std::string* error) {
  std::size_t elements = 0;
  std::size_t stereo_time = 0;
  if (channels < 1 || time < 1 ||
      !CheckedMultiply(2, static_cast<std::size_t>(time), &stereo_time) ||
      !CheckedMultiply(static_cast<std::size_t>(channels), stereo_time,
                       &elements) ||
      latent.size() != elements || rows.size() != elements) {
    SetError(error, "invalid MiniMax H3 audio pack geometry");
    return false;
  }
  std::size_t output = 0;
  for (int stream = 0; stream < 2; ++stream) {
    for (int t = 0; t < time; ++t) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t input =
            (static_cast<std::size_t>(channel) * 2 + stream) * time + t;
        rows[output++] = latent[input];
      }
    }
  }
  return output == rows.size();
}

bool UnpackAudio(std::span<const float> rows, int channels, int time,
                 std::span<float> latent, std::string* error) {
  std::size_t elements = 0;
  std::size_t stereo_time = 0;
  if (channels < 1 || time < 1 ||
      !CheckedMultiply(2, static_cast<std::size_t>(time), &stereo_time) ||
      !CheckedMultiply(static_cast<std::size_t>(channels), stereo_time,
                       &elements) ||
      latent.size() != elements || rows.size() != elements) {
    SetError(error, "invalid MiniMax H3 audio unpack geometry");
    return false;
  }
  std::size_t input = 0;
  for (int stream = 0; stream < 2; ++stream) {
    for (int t = 0; t < time; ++t) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::size_t output =
            (static_cast<std::size_t>(channel) * 2 + stream) * time + t;
        latent[output] = rows[input++];
      }
    }
  }
  return input == rows.size();
}

NormalRng::NormalRng(std::uint64_t seed) noexcept
    : increment_((seed << 1U) | 1U) {
  (void)NextU32();
  state_ += seed ^ UINT64_C(0x9e3779b97f4a7c15);
  (void)NextU32();
}

std::uint32_t NormalRng::NextU32() noexcept {
  const std::uint64_t old_state = state_;
  state_ = old_state * UINT64_C(6364136223846793005) + increment_;
  const std::uint32_t shifted =
      static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
  const std::uint32_t rotation = static_cast<std::uint32_t>(old_state >> 59U);
  return (shifted >> rotation) |
         (shifted << ((-static_cast<std::int32_t>(rotation)) & 31));
}

float NormalRng::NextNormal() noexcept {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  const double uniform1 = (static_cast<double>(NextU32()) + 1.0) / 4294967297.0;
  const double uniform2 = (static_cast<double>(NextU32()) + 0.5) / 4294967296.0;
  const double radius = std::sqrt(-2.0 * std::log(uniform1));
  const double angle = 2.0 * 3.14159265358979323846 * uniform2;
  spare_ = static_cast<float>(radius * std::sin(angle));
  has_spare_ = true;
  return static_cast<float>(radius * std::cos(angle));
}

void NormalRng::Fill(std::span<float> output) noexcept {
  for (float& value : output) {
    value = NextNormal();
  }
}

std::optional<InitialNoise> BuildInitialNoise(std::uint64_t seed,
                                              std::size_t video_elements,
                                              int audio_channels,
                                              int audio_time,
                                              std::string* error) {
  return BuildInitialNoise(seed, {}, video_elements, audio_channels, audio_time,
                           error);
}

std::optional<InitialNoise> BuildInitialNoise(
    std::uint64_t seed, const std::vector<std::size_t>& condition_elements,
    std::size_t video_elements, int audio_channels, int audio_time,
    std::string* error) {
  std::size_t stereo_time = 0;
  std::size_t audio_elements = 0;
  const bool condition_ok =
      std::none_of(condition_elements.begin(), condition_elements.end(),
                   [](std::size_t elements) { return elements == 0; });
  if (video_elements == 0 || audio_channels < 1 || audio_time < 1 ||
      !condition_ok ||
      !CheckedMultiply(2, static_cast<std::size_t>(audio_time), &stereo_time) ||
      !CheckedMultiply(static_cast<std::size_t>(audio_channels), stereo_time,
                       &audio_elements)) {
    SetError(error, "MiniMax H3 noise tensors must be non-empty");
    return std::nullopt;
  }
  InitialNoise noise;
  std::vector<float> audio_rows;
  try {
    noise.video.resize(video_elements);
    noise.audio.resize(audio_elements);
    audio_rows.resize(audio_elements);
    for (const std::size_t elements : condition_elements) {
      noise.conditions.emplace_back(elements);
    }
  } catch (const std::exception&) {
    SetError(error, "out of memory allocating MiniMax H3 initial noise");
    return std::nullopt;
  }
  // Request-stream consumption order (reference before_denoise.py): one
  // augmentation tensor per keyframe in packed order, then video, then
  // audio; drawing in any other order shifts every later tensor.
  NormalRng rng(seed);
  for (std::vector<float>& condition : noise.conditions) {
    rng.Fill(condition);
  }
  rng.Fill(noise.video);
  rng.Fill(audio_rows);
  if (!UnpackAudio(audio_rows, audio_channels, audio_time, noise.audio,
                   error)) {
    return std::nullopt;
  }
  return noise;
}

float VelocityExtrapolationRatio(float current_sigma, float last_sigma,
                                 float previous_sigma,
                                 bool have_previous) noexcept {
  if (!have_previous) {
    return 0.0F;
  }
  const float denominator = last_sigma - previous_sigma;
  const float ratio =
      denominator != 0.0F ? (current_sigma - last_sigma) / denominator : 0.0F;
  return std::clamp(ratio, -2.0F, 2.0F);
}

std::optional<std::vector<std::uint8_t>> BuildReuseSelection(
    int steps, int reuse_interval, std::string* error) {
  if (steps < 1 || steps > kH3MaximumEvaluations || reuse_interval < 1 ||
      reuse_interval > 32) {
    SetError(error, "invalid MiniMax H3 denoiser reuse selection");
    return std::nullopt;
  }
  std::vector<std::uint8_t> selected;
  try {
    selected.resize(static_cast<std::size_t>(steps), 0);
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory allocating MiniMax H3 reuse selection");
    return std::nullopt;
  }
  for (int step = 0; step < steps; ++step) {
    if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
        step % reuse_interval == 0) {
      selected[static_cast<std::size_t>(step)] = 1;
    }
  }
  return selected;
}

std::optional<std::vector<EulerStepPlan>> BuildEulerPlan(
    const SigmaSchedule& schedule, int reuse_interval, std::string* error) {
  const int steps = schedule.evaluations();
  if (steps < 2 || schedule.audio.size() != schedule.video.size()) {
    SetError(error, "invalid MiniMax H3 schedule for Euler planning");
    return std::nullopt;
  }
  std::optional<std::vector<std::uint8_t>> selected =
      BuildReuseSelection(steps, reuse_interval, error);
  if (!selected.has_value()) {
    return std::nullopt;
  }
  std::vector<EulerStepPlan> plan;
  try {
    plan.resize(static_cast<std::size_t>(steps));
  } catch (const std::bad_alloc&) {
    SetError(error, "out of memory allocating MiniMax H3 Euler plan");
    return std::nullopt;
  }
  int last_evaluated = -1;
  int previous_evaluated = -1;
  for (int step = 0; step < steps; ++step) {
    const std::size_t index = static_cast<std::size_t>(step);
    EulerStepPlan& item = plan[index];
    item.evaluate = (*selected)[index] != 0;
    if (item.evaluate) {
      previous_evaluated = last_evaluated;
      last_evaluated = step;
    }
    if (last_evaluated < 0) {
      SetError(error, "MiniMax H3 Euler plan does not evaluate step zero");
      return std::nullopt;
    }
    item.last_evaluated = last_evaluated;
    item.previous_evaluated = previous_evaluated;
    const float video_timestep = 1.0F - schedule.video[index];
    const float audio_timestep = 1.0F - schedule.audio[index];
    item.video_sigma_from_timestep = 1.0F - video_timestep;
    item.audio_sigma_from_timestep = 1.0F - audio_timestep;
    item.video_ratio = schedule.video[index + 1] / schedule.video[index];
    item.audio_ratio = schedule.audio[index + 1] / schedule.audio[index];
    if (!std::isfinite(item.video_sigma_from_timestep) ||
        item.video_sigma_from_timestep <= 0.0F ||
        !std::isfinite(item.audio_sigma_from_timestep) ||
        item.audio_sigma_from_timestep <= 0.0F ||
        !std::isfinite(item.video_ratio) || item.video_ratio < 0.0F ||
        item.video_ratio >= 1.0F || !std::isfinite(item.audio_ratio) ||
        item.audio_ratio < 0.0F || item.audio_ratio >= 1.0F) {
      SetError(error, "invalid MiniMax H3 Diffusers Euler coefficients");
      return std::nullopt;
    }
    if (!item.evaluate) {
      item.video_extrapolation = VelocityExtrapolationRatio(
          schedule.video[index],
          schedule.video[static_cast<std::size_t>(last_evaluated)],
          previous_evaluated >= 0
              ? schedule.video[static_cast<std::size_t>(previous_evaluated)]
              : 0.0F,
          previous_evaluated >= 0);
      item.audio_extrapolation = VelocityExtrapolationRatio(
          schedule.audio[index],
          schedule.audio[static_cast<std::size_t>(last_evaluated)],
          previous_evaluated >= 0
              ? schedule.audio[static_cast<std::size_t>(previous_evaluated)]
              : 0.0F,
          previous_evaluated >= 0);
    }
  }
  return plan;
}

#if !defined(ENGINE_ENABLE_HIP)
bool HipEulerUpdate(void* sample_f32, std::size_t sample_elements,
                    std::size_t sample_offset, const void* last_f32,
                    const void* previous_f32, std::size_t velocity_elements,
                    float sigma_from_timestep, float ratio, float extrapolation,
                    void* stream, std::string* error) {
  (void)sample_f32;
  (void)sample_elements;
  (void)sample_offset;
  (void)last_f32;
  (void)previous_f32;
  (void)velocity_elements;
  (void)sigma_from_timestep;
  (void)ratio;
  (void)extrapolation;
  (void)stream;
  SetError(error,
           "MiniMax H3 GPU-state Euler updates require ENGINE_ENABLE_HIP");
  return false;
}
#endif

}  // namespace gufo::minimax_h3
