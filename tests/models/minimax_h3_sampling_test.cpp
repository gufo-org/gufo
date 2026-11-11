#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace {

using gufo::minimax_h3::BuildConditionedTimeRowSchedule;
using gufo::minimax_h3::BuildEulerPlan;
using gufo::minimax_h3::BuildInitialNoise;
using gufo::minimax_h3::BuildKeyframeLayout;
using gufo::minimax_h3::BuildModulationRowMap;
using gufo::minimax_h3::BuildReuseSelection;
using gufo::minimax_h3::BuildServingSchedule;
using gufo::minimax_h3::BuildTextOnlyLayout;
using gufo::minimax_h3::BuildTimeRowSchedule;
using gufo::minimax_h3::GenerationGeometry;
using gufo::minimax_h3::KeyframeAnchorTimes;
using gufo::minimax_h3::ModulationRowMapSha256;
using gufo::minimax_h3::NormalRng;
using gufo::minimax_h3::PackedLayoutSha256;
using gufo::minimax_h3::ResolveCanvasSize;
using gufo::minimax_h3::ResolveGenerationGeometry;
using gufo::minimax_h3::SegmentKind;
using gufo::minimax_h3::Sha256;
using gufo::minimax_h3::SigmaSchedule;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::string ScheduleSha256(const SigmaSchedule& schedule) {
  std::vector<unsigned char> bytes;
  bytes.reserve((schedule.video.size() + schedule.audio.size()) * 4);
  const auto append = [&bytes](std::span<const float> values) {
    for (const float value : values) {
      const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
      for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<unsigned char>((bits >> shift) & 0xFFU));
      }
    }
  };
  append(schedule.video);
  append(schedule.audio);
  return Sha256(bytes);
}

std::string NoiseSha256(const std::vector<float>& video,
                        const std::vector<float>& audio) {
  std::vector<unsigned char> bytes;
  bytes.reserve((video.size() + audio.size()) * 4);
  const auto append = [&bytes](const std::vector<float>& values) {
    for (const float value : values) {
      const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
      for (unsigned int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<unsigned char>((bits >> shift) & 0xFFU));
      }
    }
  };
  append(video);
  append(audio);
  return Sha256(bytes);
}

GenerationGeometry Geometry(int width, int height, int frames) {
  std::string error;
  auto geometry = ResolveGenerationGeometry(width, height, frames, &error);
  if (!geometry.has_value()) {
    std::cerr << "unexpected geometry failure: " << error << '\n';
    ++failures;
    return {};
  }
  return *geometry;
}

void TestGeometry() {
  struct Case {
    int requested;
    int frames;
    int video;
    int audio;
  };
  constexpr Case cases[] = {
      {22, 22, 7, 37},
      {39, 39, 12, 65},
      {56, 56, 17, 93},
      {345, 345, 102, 575},
  };
  for (const Case& expected : cases) {
    const GenerationGeometry geometry = Geometry(256, 256, expected.requested);
    CHECK(geometry.temporal.frames == expected.frames);
    CHECK(geometry.temporal.video_latent_frames == expected.video);
    CHECK(geometry.temporal.audio_latent_frames == expected.audio);
    CHECK(geometry.latent_width == 16);
    CHECK(geometry.latent_height == 16);
  }
  CHECK(Geometry(256, 256, 6).temporal.frames == 22);

  std::string error;
  CHECK(!ResolveGenerationGeometry(255, 256, 22, &error).has_value());
  CHECK(!ResolveGenerationGeometry(1344, 800, 22, &error).has_value());
  CHECK(!ResolveGenerationGeometry(256, 256, 4, &error).has_value());
  CHECK(!ResolveGenerationGeometry(256, 256, 346, &error).has_value());
  CHECK(!ResolveGenerationGeometry(256, 256, std::numeric_limits<int>::max(),
                                   &error)
             .has_value());
}

void TestSchedules() {
  struct Case {
    int evaluations;
    const char* sha256;
  };
  constexpr Case cases[] = {
      {4, "dee27052a1aa4aa508a7b81bbe39e6bb4b880dd2d61dd3953380c5d0da3eb686"},
      {7, "51488c2e9779388e9b62aa3e0981bd5d654c38ac1ce5358be5e317fb63213ffc"},
      {19, "0f71f36f5f3066f0c1aece22ffb9dd2150cc85c3b52d3f15e6a0cec2bd593e9b"},
      {49, "ddb451d3edad496ef895ce468774632a570434e23be3d7b05a3b326c9090099c"},
  };
  for (const Case& expected : cases) {
    std::string error;
    auto schedule = BuildServingSchedule(expected.evaluations, &error);
    CHECK(schedule.has_value());
    if (!schedule.has_value()) {
      continue;
    }
    const std::string got = ScheduleSha256(*schedule);
    if (got != expected.sha256) {
      std::cerr << "schedule " << expected.evaluations << " SHA-256: " << got
                << '\n';
    }
    CHECK(got == expected.sha256);
    CHECK(schedule->video.front() == 1.0F);
    CHECK(schedule->audio.front() == 1.0F);
    CHECK(schedule->video.back() == 0.0F);
    CHECK(schedule->audio.back() == 0.0F);
    for (int step = 0; step < expected.evaluations; ++step) {
      CHECK(schedule->video[static_cast<std::size_t>(step)] >
            schedule->video[static_cast<std::size_t>(step + 1)]);
      CHECK(schedule->audio[static_cast<std::size_t>(step)] >
            schedule->audio[static_cast<std::size_t>(step + 1)]);
    }
    auto rows = BuildTimeRowSchedule(*schedule, &error);
    CHECK(rows.has_value());
    if (rows.has_value()) {
      CHECK(rows->time_rows ==
            static_cast<std::uint32_t>(2 * expected.evaluations - 1));
      CHECK(rows->video_rows.front() == 0);
      CHECK(rows->audio_rows.front() == 0);
    }
  }
  std::string error;
  CHECK(!BuildServingSchedule(1, &error).has_value());
  CHECK(!BuildServingSchedule(1001, &error).has_value());
}

void TestLayouts() {
  struct Case {
    int pixels;
    std::size_t rows;
    const char* layout_sha256;
    const char* row_map_sha256;
  };
  constexpr Case cases[] = {
      {256, 528,
       "080663c29e42fd3859e0a81f2b9e26bf504c6bac9cce5c74d380dd382c0cced5",
       "9e017a1625d24cc7fdc5f43ccd92f820788ec20d6a0b17ac99b133a04926c6ff"},
      {320, 780, nullptr, nullptr},
      {384, 1088, nullptr, nullptr},
      {512, 1872,
       "c78f49c365e1bbab95388f2f9f9e7a7e3281a46f820534e4c821492d027365cb",
       "155afa2335594655653cbff6722cdc6853cac7185576cfd43d5bb8ceee3288f4"},
  };
  for (const Case& expected : cases) {
    std::string error;
    const GenerationGeometry geometry =
        Geometry(expected.pixels, expected.pixels, 22);
    auto layout = BuildTextOnlyLayout(6, geometry, &error);
    CHECK(layout.has_value());
    if (!layout.has_value()) {
      continue;
    }
    CHECK(layout->rows() == expected.rows);
    CHECK(layout->segments.size() == 3);
    CHECK(layout->text_rows == 6);
    CHECK(layout->audio_target_rows == 74);
    CHECK(layout->segments[0].begin == 0);
    CHECK(layout->segments[0].end == 6);
    CHECK(layout->segments[1].begin == 6);
    CHECK(layout->segments[1].end == 80);
    CHECK(layout->segments[2].begin == 80);
    CHECK(layout->segments[2].end == expected.rows);

    const std::size_t video_begin = layout->segments[2].begin;
    const std::size_t frame_columns =
        static_cast<std::size_t>(geometry.latent_width / 2);
    const double expected_step =
        64.0 / std::sqrt(static_cast<double>(geometry.latent_height) *
                         geometry.latent_width);
    const double height_ratio =
        geometry.latent_height /
        std::sqrt(static_cast<double>(geometry.latent_height) *
                  geometry.latent_width);
    const double width_ratio =
        geometry.latent_width /
        std::sqrt(static_cast<double>(geometry.latent_height) *
                  geometry.latent_width);
    CHECK(std::abs(layout->positions[video_begin].height -
                   (1.0 - height_ratio) * 16.0) < 1.0e-12);
    CHECK(std::abs(layout->positions[video_begin].width -
                   (1.0 - width_ratio) * 16.0) < 1.0e-12);
    CHECK(std::abs(layout->positions[video_begin + 1].width -
                   layout->positions[video_begin].width - expected_step) <
          1.0e-12);
    CHECK(std::abs(layout->positions[video_begin + frame_columns].height -
                   layout->positions[video_begin].height - expected_step) <
          1.0e-12);
    if (expected.pixels == 256) {
      CHECK(layout->positions[video_begin + 1].width == 4.0);
      CHECK(layout->positions[video_begin + frame_columns].height == 4.0);
    } else if (expected.pixels == 512) {
      CHECK(layout->positions[video_begin + 1].width == 2.0);
      CHECK(layout->positions[video_begin + frame_columns].height == 2.0);
    }

    if (expected.layout_sha256 != nullptr) {
      const std::string layout_hash = PackedLayoutSha256(*layout);
      if (layout_hash != expected.layout_sha256) {
        std::cerr << "layout " << expected.pixels << " SHA-256: " << layout_hash
                  << '\n';
      }
      CHECK(layout_hash == expected.layout_sha256);
    }

    auto schedule = BuildServingSchedule(20, &error);
    CHECK(schedule.has_value());
    if (!schedule.has_value()) {
      continue;
    }
    auto time_rows = BuildTimeRowSchedule(*schedule, &error);
    CHECK(time_rows.has_value());
    if (!time_rows.has_value()) {
      continue;
    }
    auto row_map = BuildModulationRowMap(*time_rows, 7, *layout, {}, &error);
    CHECK(row_map.has_value());
    if (!row_map.has_value()) {
      continue;
    }
    if (expected.row_map_sha256 != nullptr) {
      const std::string row_map_hash = ModulationRowMapSha256(*row_map);
      if (row_map_hash != expected.row_map_sha256) {
        std::cerr << "row map " << expected.pixels
                  << " SHA-256: " << row_map_hash << '\n';
      }
      CHECK(row_map_hash == expected.row_map_sha256);
    }
    CHECK((*row_map)[0] == time_rows->video_rows[7] * 3 + 1);
    CHECK((*row_map)[6] == time_rows->audio_rows[7] * 3 + 2);
    CHECK((*row_map)[80] == time_rows->video_rows[7] * 3);
    CHECK(!BuildModulationRowMap(*time_rows, -1, *layout, {}, &error)
               .has_value());
    CHECK(!BuildModulationRowMap(*time_rows, 20, *layout, {}, &error)
               .has_value());
  }

  std::string error;
  auto tiny_layout = BuildTextOnlyLayout(12, Geometry(32, 32, 5), &error);
  CHECK(tiny_layout.has_value());
  if (tiny_layout.has_value()) {
    double temporal_sum = 0.0;
    double temporal_weighted = 0.0;
    for (std::size_t index = 0; index < tiny_layout->positions.size();
         ++index) {
      temporal_sum += tiny_layout->positions[index].temporal;
      temporal_weighted += static_cast<double>(index + 1) *
                           tiny_layout->positions[index].temporal;
    }
    CHECK(std::abs(temporal_sum - 339.6666666666667) < 1.0e-10);
    CHECK(std::abs(temporal_weighted - 6498.0) < 1.0e-10);
  }

  auto rectangular = BuildTextOnlyLayout(6, Geometry(1344, 768, 22), &error);
  CHECK(rectangular.has_value());
  if (rectangular.has_value()) {
    const std::size_t begin = rectangular->segments[2].begin;
    const std::size_t columns =
        static_cast<std::size_t>(rectangular->latent_width / 2);
    const double sqrt_area =
        std::sqrt(static_cast<double>(rectangular->latent_height) *
                  rectangular->latent_width);
    const double step = 64.0 / sqrt_area;
    CHECK(std::abs(rectangular->positions[begin + 1].width -
                   rectangular->positions[begin].width - step) < 1.0e-12);
    CHECK(std::abs(rectangular->positions[begin + columns].height -
                   rectangular->positions[begin].height - step) < 1.0e-12);
    CHECK(rectangular->positions[begin].height > 0.0);
    CHECK(rectangular->positions[begin].width < 0.0);
  }

  GenerationGeometry invalid = Geometry(256, 256, 22);
  invalid.latent_width = 15;
  CHECK(!BuildTextOnlyLayout(6, invalid, &error).has_value());
  CHECK(!BuildTextOnlyLayout(0, Geometry(256, 256, 22), &error).has_value());
  invalid = Geometry(256, 256, 22);
  invalid.latent_width = std::numeric_limits<int>::max() - 1;
  CHECK(!BuildTextOnlyLayout(6, invalid, &error).has_value());
}

void TestPacking() {
  constexpr int channels = 3;
  constexpr int time = 2;
  constexpr int height = 4;
  constexpr int width = 6;
  constexpr std::size_t elements = channels * time * height * width;
  std::vector<float> latent(elements);
  for (std::size_t index = 0; index < latent.size(); ++index) {
    latent[index] = static_cast<float>(index);
  }
  std::vector<float> rows(elements);
  std::vector<float> round_trip(elements);
  std::string error;
  CHECK(gufo::minimax_h3::PatchifyVideo(latent, channels, time, height, width,
                                        rows, &error));
  constexpr float first[] = {0, 1, 6, 7, 48, 49, 54, 55, 96, 97, 102, 103};
  CHECK(std::memcmp(rows.data(), first, sizeof(first)) == 0);
  CHECK(gufo::minimax_h3::UnpatchifyVideo(rows, channels, time, height, width,
                                          round_trip, &error));
  CHECK(std::memcmp(latent.data(), round_trip.data(),
                    latent.size() * sizeof(float)) == 0);
  CHECK(!gufo::minimax_h3::PatchifyVideo(
      latent, channels, time, height, width,
      std::span<float>(rows.data(), rows.size() - 1), &error));
  CHECK(!gufo::minimax_h3::PatchifyVideo({}, std::numeric_limits<int>::max(),
                                         std::numeric_limits<int>::max(), 4, 4,
                                         {}, &error));

  constexpr int audio_channels = 4;
  constexpr int audio_time = 3;
  constexpr std::size_t audio_elements = audio_channels * 2 * audio_time;
  std::vector<float> audio(audio_elements);
  for (std::size_t index = 0; index < audio.size(); ++index) {
    audio[index] = static_cast<float>(index);
  }
  std::vector<float> packed(audio_elements);
  std::vector<float> unpacked(audio_elements);
  CHECK(gufo::minimax_h3::PackAudio(audio, audio_channels, audio_time, packed,
                                    &error));
  constexpr float expected[] = {0, 6, 12, 18, 1, 7,  13, 19, 2, 8,  14, 20,
                                3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23};
  CHECK(std::memcmp(packed.data(), expected, sizeof(expected)) == 0);
  CHECK(gufo::minimax_h3::UnpackAudio(packed, audio_channels, audio_time,
                                      unpacked, &error));
  CHECK(std::memcmp(audio.data(), unpacked.data(),
                    audio.size() * sizeof(float)) == 0);
}

void TestNoise() {
  constexpr std::uint32_t expected_u32[] = {
      0x8daf78a1U, 0x78625b1aU, 0x843daf20U, 0x3314b8ddU,
      0xf33e7a58U, 0x7a10c688U, 0x77a0a5e8U, 0xdba4433bU,
  };
  gufo::minimax_h3::NormalRng rng(42);
  for (const std::uint32_t expected : expected_u32) {
    CHECK(rng.NextU32() == expected);
  }

  const GenerationGeometry geometry = Geometry(256, 256, 22);
  const std::size_t video_elements =
      24ULL * geometry.temporal.video_latent_frames * geometry.latent_height *
      geometry.latent_width;
  constexpr int audio_channels = 32;
  const int audio_time = geometry.temporal.audio_latent_frames;
  const std::size_t audio_elements =
      static_cast<std::size_t>(audio_channels * 2 * audio_time);
  std::string error;
  auto first =
      BuildInitialNoise(42, video_elements, audio_channels, audio_time, &error);
  auto second =
      BuildInitialNoise(42, video_elements, audio_channels, audio_time, &error);
  CHECK(first.has_value());
  CHECK(second.has_value());
  if (!first.has_value() || !second.has_value()) {
    return;
  }
  CHECK(first->video == second->video);
  CHECK(first->audio == second->audio);
  std::vector<float> packed_audio(audio_elements);
  CHECK(gufo::minimax_h3::PackAudio(first->audio, audio_channels, audio_time,
                                    packed_audio, &error));
  CHECK(std::memcmp(first->video.data(), packed_audio.data(),
                    audio_elements * sizeof(float)) != 0);
  gufo::minimax_h3::NormalRng sequential(42);
  std::vector<float> expected_video(video_elements);
  std::vector<float> expected_audio_rows(audio_elements);
  sequential.Fill(expected_video);
  sequential.Fill(expected_audio_rows);
  CHECK(first->video == expected_video);
  CHECK(packed_audio == expected_audio_rows);
  const std::string hash = NoiseSha256(first->video, first->audio);
  constexpr std::string_view expected_hash =
      "6318dbfea74c61415d470c12c019cda9df6a8491f86b075e403d1c2fc2403b4d";
  if (hash != expected_hash) {
    std::cerr << "noise 256x256x22 seed42 SHA-256: " << hash << '\n';
  }
  CHECK(hash == expected_hash);
  CHECK(!BuildInitialNoise(42, 0, audio_channels, audio_time, &error)
             .has_value());
  CHECK(!BuildInitialNoise(42, video_elements, 0, audio_time, &error)
             .has_value());
}

void TestReuseAndEulerPlan() {
  std::string error;
  auto aggressive = BuildReuseSelection(20, 3, &error);
  CHECK(aggressive.has_value());
  if (aggressive.has_value()) {
    constexpr int expected[] = {0, 3, 6, 9, 12, 15, 18, 19};
    std::size_t cursor = 0;
    for (int step = 0; step < 20; ++step) {
      const bool selected =
          cursor < std::size(expected) && step == expected[cursor];
      CHECK((*aggressive)[static_cast<std::size_t>(step)] ==
            static_cast<std::uint8_t>(selected));
      if (selected) {
        ++cursor;
      }
    }
  }
  CHECK(!BuildReuseSelection(20, 0, &error).has_value());
  CHECK(!BuildReuseSelection(20, 33, &error).has_value());

  auto schedule = BuildServingSchedule(20, &error);
  CHECK(schedule.has_value());
  if (!schedule.has_value()) {
    return;
  }
  SigmaSchedule malformed = *schedule;
  malformed.video[2] = malformed.video[1];
  CHECK(!BuildTimeRowSchedule(malformed, &error).has_value());
  auto plan = BuildEulerPlan(*schedule, 3, &error);
  CHECK(plan.has_value());
  if (!plan.has_value()) {
    return;
  }
  CHECK(plan->size() == 20);
  CHECK((*plan)[0].evaluate);
  CHECK(!(*plan)[1].evaluate);
  CHECK((*plan)[3].evaluate);
  CHECK((*plan)[19].evaluate);
  CHECK((*plan)[1].last_evaluated == 0);
  CHECK((*plan)[1].previous_evaluated == -1);
  CHECK((*plan)[4].last_evaluated == 3);
  CHECK((*plan)[4].previous_evaluated == 0);
  for (std::size_t index = 0; index < plan->size(); ++index) {
    const auto& step = (*plan)[index];
    const float video_timestep = 1.0F - schedule->video[index];
    const float video_sigma_from_timestep = 1.0F - video_timestep;
    const float video_ratio =
        schedule->video[index + 1] / schedule->video[index];
    const float audio_timestep = 1.0F - schedule->audio[index];
    const float audio_sigma_from_timestep = 1.0F - audio_timestep;
    const float audio_ratio =
        schedule->audio[index + 1] / schedule->audio[index];
    CHECK(std::bit_cast<std::uint32_t>(step.video_sigma_from_timestep) ==
          std::bit_cast<std::uint32_t>(video_sigma_from_timestep));
    CHECK(std::bit_cast<std::uint32_t>(step.audio_sigma_from_timestep) ==
          std::bit_cast<std::uint32_t>(audio_sigma_from_timestep));
    CHECK(std::bit_cast<std::uint32_t>(step.video_ratio) ==
          std::bit_cast<std::uint32_t>(video_ratio));
    CHECK(std::bit_cast<std::uint32_t>(step.audio_ratio) ==
          std::bit_cast<std::uint32_t>(audio_ratio));
    CHECK(step.video_sigma_from_timestep > 0.0F);
    CHECK(step.audio_sigma_from_timestep > 0.0F);
    CHECK(step.video_ratio >= 0.0F && step.video_ratio < 1.0F);
    CHECK(step.audio_ratio >= 0.0F && step.audio_ratio < 1.0F);
    CHECK(step.video_extrapolation >= -2.0F);
    CHECK(step.video_extrapolation <= 2.0F);
    CHECK(step.audio_extrapolation >= -2.0F);
    CHECK(step.audio_extrapolation <= 2.0F);
  }
  for (const int evaluations : {4, 19, 49}) {
    auto preset_schedule = BuildServingSchedule(evaluations, &error);
    CHECK(preset_schedule.has_value());
    if (!preset_schedule.has_value()) {
      continue;
    }
    auto preset_plan = BuildEulerPlan(*preset_schedule, 1, &error);
    CHECK(preset_plan.has_value());
    if (!preset_plan.has_value()) {
      continue;
    }
    for (std::size_t index = 0; index < preset_plan->size(); ++index) {
      const auto& item = (*preset_plan)[index];
      const float video_timestep = 1.0F - preset_schedule->video[index];
      const float audio_timestep = 1.0F - preset_schedule->audio[index];
      CHECK(std::bit_cast<std::uint32_t>(item.video_sigma_from_timestep) ==
            std::bit_cast<std::uint32_t>(1.0F - video_timestep));
      CHECK(std::bit_cast<std::uint32_t>(item.audio_sigma_from_timestep) ==
            std::bit_cast<std::uint32_t>(1.0F - audio_timestep));
      CHECK(std::bit_cast<std::uint32_t>(item.video_ratio) ==
            std::bit_cast<std::uint32_t>(preset_schedule->video[index + 1] /
                                         preset_schedule->video[index]));
      CHECK(std::bit_cast<std::uint32_t>(item.audio_ratio) ==
            std::bit_cast<std::uint32_t>(preset_schedule->audio[index + 1] /
                                         preset_schedule->audio[index]));
    }
  }

#if !defined(ENGINE_ENABLE_HIP)
  CHECK(!gufo::minimax_h3::HipEulerUpdate(nullptr, 0, 0, nullptr, nullptr, 0,
                                          0.25F, 0.5F, 0.0F, nullptr, &error));
  CHECK(error.find("ENGINE_ENABLE_HIP") != std::string::npos);
#endif
}

void TestKeyframeConditioning() {
  std::string error;

  // Anchor rotary times: goldens captured from numpy float64 evaluation of
  // the reference expression (float(T) + float(spans.sum()) - 5/3) with the
  // released spans construction and numpy pairwise summation order.
  {
    auto only_first = KeyframeAnchorTimes(13, 7, true, false, &error);
    CHECK(only_first.has_value());
    CHECK(only_first->size() == 1);
    CHECK(only_first->at(0) == 13.0);
    auto only_last = KeyframeAnchorTimes(13, 7, false, true, &error);
    CHECK(only_last.has_value());
    CHECK(only_last->size() == 1);
    CHECK(std::bit_cast<std::uint64_t>(only_last->at(0)) ==
          std::bit_cast<std::uint64_t>(0x1.8000000000001p+5));
    auto pair = KeyframeAnchorTimes(1300, 47, true, true, &error);
    CHECK(pair.has_value());
    CHECK(pair->size() == 2);
    CHECK((*pair)[0] == 1300.0);
    CHECK(std::bit_cast<std::uint64_t>((*pair)[1]) ==
          std::bit_cast<std::uint64_t>(0x1.866aaaaaaaaaap+10));
    auto long_window = KeyframeAnchorTimes(13, 102, false, true, &error);
    CHECK(long_window.has_value());
    CHECK(std::bit_cast<std::uint64_t>(long_window->at(0)) ==
          std::bit_cast<std::uint64_t>(0x1.252aaaaaaaaa9p+9));
    CHECK(KeyframeAnchorTimes(13, 1, true, true, &error) == std::nullopt);
    CHECK(KeyframeAnchorTimes(0, 47, false, true, &error) == std::nullopt);
    CHECK(KeyframeAnchorTimes(13, 47, false, false, &error)->empty());
  }

  // Conditioned packed sequence: [text | condition | audio | video-target]
  // with anchors occupying full target-canvas frame grids at their times.
  {
    const GenerationGeometry geometry = Geometry(512, 512, 22);
    const auto anchors = *KeyframeAnchorTimes(
        6, geometry.temporal.video_latent_frames, true, true, &error);
    auto layout = BuildKeyframeLayout(6, geometry, anchors, &error);
    CHECK(layout.has_value());
    if (!layout.has_value()) {
      return;
    }
    const std::size_t frame_rows = 256;  // (32/2) * (32/2)
    CHECK(layout->video_condition_rows == 2 * frame_rows);
    CHECK(layout->audio_target_rows == 74);
    CHECK(layout->rows() == 6 + 512 + 74 + 7 * frame_rows);
    CHECK(layout->segments.size() == 4);
    CHECK(layout->segments[0].kind == SegmentKind::kText);
    CHECK(layout->segments[1].kind == SegmentKind::kVideoCondition);
    CHECK(layout->segments[1].begin == 6);
    CHECK(layout->segments[1].end == 6 + 2 * frame_rows);
    CHECK(layout->segments[2].kind == SegmentKind::kAudioTarget);
    CHECK(layout->segments[3].kind == SegmentKind::kVideoTarget);
    CHECK(layout->positions[6].temporal == 6.0);
    CHECK(layout->positions[6 + frame_rows].temporal == anchors[1]);
    const std::size_t target_begin = 6 + 2 * frame_rows + 74;
    CHECK(layout->positions[6].height ==
          layout->positions[target_begin].height);
    CHECK(layout->positions[6].width == layout->positions[target_begin].width);
    // Audio and video rotary clocks are unchanged by conditioning: they
    // stay pinned to text_rows, not to the sequence offsets.
    CHECK(layout->positions[target_begin].temporal == 6.0);
    CHECK(std::abs(layout->positions[6 + frame_rows].temporal -
                   (layout->positions[target_begin + 6 * frame_rows].temporal +
                    5.0)) < 1.0e-12);
    CHECK(!BuildKeyframeLayout(6, geometry, {}, &error).has_value());
  }

  // Time-row table: one extra shared 0.999 row; text-only schedule plus one.
  {
    auto schedule = BuildServingSchedule(49, &error);
    CHECK(schedule.has_value());
    auto plain = BuildTimeRowSchedule(*schedule, &error);
    auto conditioned = BuildConditionedTimeRowSchedule(*schedule, &error);
    CHECK(plain.has_value() && conditioned.has_value());
    CHECK(conditioned->time_rows == plain->time_rows + 1);
    CHECK(conditioned->video_rows == plain->video_rows);
    CHECK(conditioned->audio_rows == plain->audio_rows);
    CHECK(conditioned->condition_rows.size() == 49);
    for (const std::uint32_t id : conditioned->condition_rows) {
      CHECK(id == conditioned->condition_rows.front());
      CHECK(id != conditioned->video_rows.front());
    }
  }

  // Row map: conditioned rows carry modality 0 (video) at the 0.999 row.
  {
    const GenerationGeometry geometry = Geometry(256, 256, 22);
    const auto anchors = *KeyframeAnchorTimes(4, 7, true, false, &error);
    auto layout = BuildKeyframeLayout(4, geometry, anchors, &error);
    auto schedule = BuildServingSchedule(4, &error);
    auto rows = BuildConditionedTimeRowSchedule(*schedule, &error);
    CHECK(layout.has_value() && rows.has_value());
    auto map = BuildModulationRowMap(*rows, 1, *layout, {}, &error);
    CHECK(map.has_value());
    if (map.has_value()) {
      const std::uint32_t condition_id = rows->condition_rows[1];
      CHECK((*map)[4] == condition_id * 3 + 0);  // first anchor row
      CHECK((*map)[4 + 1] == condition_id * 3 + 0);
      CHECK((*map)[layout->segments[2].begin] == rows->audio_rows[1] * 3 + 2);
      CHECK((*map)[layout->segments[3].begin] == rows->video_rows[1] * 3 + 0);
    }
    // A conditioned layout with the unconditioned schedule must be refused.
    auto plain_rows = BuildTimeRowSchedule(*schedule, &error);
    CHECK(!BuildModulationRowMap(*plain_rows, 1, *layout, {}, &error)
               .has_value());
  }

  // Request-stream lineage: augmentation noise for each keyframe is drawn
  // first (packed order), then video, then audio.
  {
    constexpr std::uint64_t seed = 4242;
    constexpr std::size_t condition_elements = 97;
    constexpr std::size_t video_elements = 24;
    NormalRng reference(seed);
    std::vector<float> expected_condition(condition_elements);
    for (float& value : expected_condition) {
      value = reference.NextNormal();
    }
    std::vector<float> expected_video(video_elements);
    for (float& value : expected_video) {
      value = reference.NextNormal();
    }
    auto noise = BuildInitialNoise(seed, {condition_elements}, video_elements,
                                   32, 3, &error);
    CHECK(noise.has_value());
    if (noise.has_value()) {
      CHECK(noise->conditions.size() == 1);
      CHECK(noise->conditions[0] == expected_condition);
      CHECK(noise->video == expected_video);
      CHECK(noise->audio.size() == 32 * 2 * 3);
    }
    CHECK(!BuildInitialNoise(seed, {0}, video_elements, 32, 3, &error)
               .has_value());
  }
}

}  // namespace

void TestCanvasRule() {
  // Expected canvases were produced by running the released rule itself
  // (diffusers modular_pipelines/minimax_h3/modular_pipeline.py
  // resolve_canvas_size with canvas_multiple 32), not by this port.
  struct Case {
    double aspect_width;
    double aspect_height;
    int short_edge;
    std::int64_t max_pixels;
    int width;
    int height;
  };
  constexpr std::int64_t kBudget = 768LL * 1344LL;
  constexpr Case kCases[] = {
      {1664, 928, 768, kBudget, 1376, 768},
      {1664, 928, 512, 512 * 512, 672, 384},
      {1664, 928, 384, 384 * 384, 512, 288},
      {1664, 928, 320, 320 * 320, 416, 224},
      {1664, 928, 256, 256 * 256, 352, 192},
      {16, 9, 768, kBudget, 1344, 768},
      {9, 16, 768, kBudget, 768, 1344},
      {1, 1, 768, kBudget, 768, 768},
      {4, 3, 768, kBudget, 1024, 768},
      {1, 4, 768, kBudget, 512, 2016},
      {4, 1, 768, kBudget, 2016, 512},
      {1856, 576, 768, kBudget, 1824, 576},
      {928, 1664, 512, 512 * 512, 384, 672},
      {1, 1, 512, 512 * 512, 512, 512},
      {21, 9, 768, kBudget, 1536, 672},
      {3, 2, 512, 512 * 512, 640, 416},
  };
  for (const Case& test : kCases) {
    std::string error;
    const auto canvas =
        ResolveCanvasSize(test.aspect_width, test.aspect_height,
                          test.short_edge, test.max_pixels, &error);
    CHECK(canvas.has_value());
    if (canvas.has_value()) {
      CHECK(canvas->width == test.width);
      CHECK(canvas->height == test.height);
    }
  }
  std::string error;
  CHECK(!ResolveCanvasSize(1, 5, 768, kBudget, &error).has_value());
  CHECK(!ResolveCanvasSize(5, 1, 768, kBudget, &error).has_value());
  CHECK(!ResolveCanvasSize(0, 1, 768, kBudget, &error).has_value());

  // Rounding after the budget lets a resolved canvas exceed 768*1344; the
  // largest the rule produces (1856x576) must stay a legal geometry.
  CHECK(ResolveGenerationGeometry(1376, 768, 22, &error).has_value());
  CHECK(ResolveGenerationGeometry(1856, 576, 22, &error).has_value());
  CHECK(ResolveGenerationGeometry(576, 1856, 22, &error).has_value());
  CHECK(!ResolveGenerationGeometry(1888, 576, 22, &error).has_value());
}

int main() {
  TestGeometry();
  TestCanvasRule();
  TestSchedules();
  TestLayouts();
  TestKeyframeConditioning();
  TestPacking();
  TestNoise();
  TestReuseAndEulerPlan();
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 sampling checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 geometry/layout/schedule/RNG checks passed\n";
  return 0;
}
