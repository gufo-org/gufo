// Independent pinned official processor fixtures, including encoded media.
#include "src/models/embedding_gemma2/video_processor.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace model = gufo::models::embedding_gemma2;
namespace {
void Check(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Refused(F run, const char* name) {
  bool refused = false;
  try {
    run();
  } catch (const std::exception&) {
    refused = true;
  }
  Check(refused, name);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const std::filesystem::path golden = argc > 1 ? argv[1] : GUFO_VIDEO_GOLDEN;
    std::ifstream file(golden);
    const auto root = gufo::json::parse(
        std::string(std::istreambuf_iterator<char>(file), {}));
    const std::filesystem::path fixture = golden.parent_path();
    for (const auto& probe : root.find("cases")->items()) {
      model::VideoOptions options;
      const auto& settings = *probe.find("options");
      options.fps = settings.member_double("fps", 1);
      options.timestamps = (settings.find("add_timestamps") &&
                            settings.find("add_timestamps")->as_bool());
      options.max_frames = settings.member_size("max_frames", 32);
      options.truncate_overflow =
          settings.member_str("overflow_strategy") == "truncate";
      for (const auto& video : probe.find("videos")->items()) {
        std::ifstream media(fixture / video.member_str("path"),
                            std::ios::binary);
        std::vector<std::uint8_t> bytes(std::istreambuf_iterator<char>(media),
                                        {});
        Check(gufo::crypto::Sha256Hex(bytes) == video.member_str("sha256"),
              "media hash");
        const auto frames = model::ProcessVideo(bytes, options);
        const auto& indices = video.find("indices")->items();
        const auto& expected = video.find("patches")->items();
        Check(frames.size() == indices.size(), "sampled frame count");
        for (std::size_t i = 0; i < frames.size(); ++i) {
          const auto& frame = frames[i];
          Check(frame.index == indices[i].as_size(), "selected frame index");
          Check(frame.patches.rows == expected[i].member_size("rows") &&
                    frame.patches.columns == expected[i].member_size("columns"),
                "patch geometry");
          std::vector<std::uint8_t> pixels;
          for (float value : frame.patches.pixels)
            pixels.push_back(
                static_cast<std::uint8_t>(std::lround(value * 255)));
          Check(gufo::crypto::Sha256Hex(pixels) ==
                    expected[i].member_str("sha256"),
                probe.member_str("name") +
                    " decoded RGB and resized patch bytes");
        }
        Refused([&] { (void)model::ProcessVideo(bytes, options, {}, 0); },
                "request frame budget");
        const auto framed = model::FrameVideo(frames, options);
        Check(!framed.empty() && framed.find("<|video|>") != std::string::npos,
              "video framing");
        Refused(
            [&] { (void)model::ProcessVideo(bytes, {}, [] { return true; }); },
            "cancelled decoder");
      }
      std::cout << "PASS " << probe.member_str("name") << '\n';
    }
    for (const auto* name :
         {"oversized.mp4", "overduration.mp4", "overframes.mp4"}) {
      std::ifstream media(fixture / "video" / name, std::ios::binary);
      Check(media.good(), std::string("missing resource fixture ") + name);
      std::vector<std::uint8_t> bytes(std::istreambuf_iterator<char>(media),
                                      {});
      Refused([&] { (void)model::ProcessVideo(bytes); }, name);
    }
    Check(model::SampleVideoFrames(60, 30000.0 / 1001) ==
              std::vector<std::size_t>{0, 29},
          "fractional floor sampling");
    model::VideoOptions tiny_fps;
    tiny_fps.fps = 1e-320;
    Check(model::SampleVideoFrames(60, 30, tiny_fps) ==
              std::vector<std::size_t>{0},
          "tiny positive FPS avoids nonfinite arithmetic");
    model::VideoOptions rounding;
    rounding.max_frames = 23;
    const auto rounded = model::SampleVideoFrames(31, 1, rounding);
    Check(rounded.size() == 23 && rounded[11] == 14 && rounded.back() == 30,
          "uniform overflow matches NumPy floating linspace floor");
    const auto capped = model::SampleVideoFrames(66, 2);
    Check(capped.size() == 32 && capped.front() == 0 && capped.back() == 64 &&
              capped[30] == 60,
          "uniform overflow boundary");
    Refused([] { (void)model::SampleVideoFrames(1, 0); }, "missing fps");
    Refused([] { (void)model::SampleVideoFrames(4097, 25); },
            "frame work limit");
    Refused([] { (void)model::SampleVideoFrames(301, 1); }, "duration limit");
    Refused([] { (void)model::ProcessVideo({}); }, "empty video");
    const std::vector<std::uint8_t> garbage{1, 2, 3, 4};
    Refused([&] { (void)model::ProcessVideo(garbage); }, "malformed video");
    const std::vector<std::uint8_t> oversized(20 * 1024 * 1024 + 1);
    Refused([&] { (void)model::ProcessVideo(oversized); },
            "encoded video bound");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL video_processor_test: " << e.what() << '\n';
    return 1;
  }
}
