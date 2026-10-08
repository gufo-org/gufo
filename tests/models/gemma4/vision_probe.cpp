// Gemma 4 vision encoder probe.
//
//   gemma4_vision_probe --prepare IMAGE OUT.rgb [--tokens N]
//     decodes and resizes like the prompt builder; prints WIDTH HEIGHT.
//   gemma4_vision_probe --mmproj F --image RGB W H [--no-reference] [--gpu]
//                       [--compare G4VE] [--output G4VE] [--dump-dir DIR]
//     runs the scalar reference and/or the HIP encoder on a raw RGB8 image
//     (sides multiples of 48). With both, every GPU stage is compared with
//     the reference; --compare checks the final embeddings (reference if it
//     ran, else GPU) against a G4VE file (tools/gemma4/llama_vision.cpp).
//     DIR receives reference stages as u32 dims[4] ([width][rows][1][1]) +
//     f32 data, the layout of llama_vision --dump-dir.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "src/models/gemma4/vision/reference.hpp"
#ifdef GUFO_GEMMA4_VISION_GPU
#include <hip/hip_runtime.h>

#include "src/models/gemma4/vision/encoder.hpp"
#endif

namespace vision = gufo::models::gemma4::vision;

namespace {

/// The sidecar's projection width (5376 for the 31B, 2816 for the
/// 26B-A4B), read before any embedding is handled.
std::uint32_t projection_width = 0;

struct Difference {
  double relative_rms{0};
  double max_abs{0};
  double worst_row_cosine{1};
};

Difference Compare(std::span<const float> a, std::span<const float> b,
                   std::size_t width) {
  Difference out;
  double err = 0, ref = 0;
  for (std::size_t r = 0; r < a.size() / width; ++r) {
    double dot = 0, a2 = 0, b2 = 0;
    for (std::size_t d = 0; d < width; ++d) {
      const double x = a[r * width + d], y = b[r * width + d];
      dot += x * y;
      a2 += x * x;
      b2 += y * y;
      err += (x - y) * (x - y);
      out.max_abs = std::max(out.max_abs, std::abs(x - y));
    }
    ref += a2;
    out.worst_row_cosine =
        std::min(out.worst_row_cosine, dot / std::sqrt(a2 * b2));
  }
  out.relative_rms = std::sqrt(err / ref);
  return out;
}

std::vector<float> ReadEmbedding(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::uint32_t header[4] = {};
  in.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!in || header[0] != 0x45563447U || header[1] != 1U ||
      header[3] != projection_width) {
    throw std::runtime_error("not a G4VE file of the sidecar's width: " + path);
  }
  std::vector<float> data(std::size_t{header[2]} * projection_width);
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * 4));
  if (!in) {
    throw std::runtime_error("truncated " + path);
  }
  return data;
}

void WriteStage(const std::string& path, std::span<const float> data,
                std::uint32_t width) {
  std::ofstream out(path, std::ios::binary);
  const std::uint32_t dims[4] = {
      width, static_cast<std::uint32_t>(data.size() / width), 1, 1};
  out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
}

int Run(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "--prepare") {
    const std::uint32_t tokens =
        argc >= 6 && std::string(argv[4]) == "--tokens"
            ? static_cast<std::uint32_t>(std::stoul(argv[5]))
            : vision::kDefaultSoftTokens;
    const auto image = vision::ResizeImage(
        gufo::core::DecodeImage(gufo::core::ReadImageFile(argv[2])), tokens);
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size()));
    std::printf("%u %u\n", image.width, image.height);
    return 0;
  }

  std::string mmproj, image_path, compare, output, dump_dir;
  std::uint32_t width = 0, height = 0;
  bool reference_enabled = true, gpu = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--mmproj" && i + 1 < argc)
      mmproj = argv[++i];
    else if (arg == "--image" && i + 3 < argc) {
      image_path = argv[++i];
      width = std::stoul(argv[++i]);
      height = std::stoul(argv[++i]);
    } else if (arg == "--compare" && i + 1 < argc)
      compare = argv[++i];
    else if (arg == "--output" && i + 1 < argc)
      output = argv[++i];
    else if (arg == "--dump-dir" && i + 1 < argc)
      dump_dir = argv[++i];
    else if (arg == "--gpu")
      gpu = true;
    else if (arg == "--no-reference")
      reference_enabled = false;
    else {
      std::cerr << "see the usage at the top of vision_probe.cpp\n";
      return 2;
    }
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(mmproj, &error);
  if (!reader)
    throw std::runtime_error(error);
  projection_width = static_cast<std::uint32_t>(
      reader->GetMetadataUint64("clip.vision.projection_dim").value_or(0));
  const auto weights = vision::Weights::Resolve(*reader, projection_width);
  gufo::core::Image image{
      width, height,
      std::vector<std::uint8_t>(std::size_t{width} * height * 3)};
  std::ifstream in(image_path, std::ios::binary);
  in.read(reinterpret_cast<char*>(image.pixels.data()),
          static_cast<std::streamsize>(image.pixels.size()));
  if (!in)
    throw std::runtime_error("cannot read " + image_path);

  std::map<std::string, std::vector<float>, std::less<>> stages;
  std::vector<float> embedding;
  if (reference_enabled) {
    const vision::Reference reference(weights);
    embedding = reference.Encode(image, [&](std::string_view stage,
                                            std::span<const float> data) {
      const std::string name(stage);
      if (!dump_dir.empty()) {
        WriteStage(dump_dir + "/" + name + ".f32", data,
                   name == "embedding" ? projection_width : vision::kHidden);
      }
      stages.emplace(name, std::vector<float>(data.begin(), data.end()));
    });
  }
  if (gpu) {
#ifdef GUFO_GEMMA4_VISION_GPU
    vision::Encoder encoder(mmproj, projection_width);
    (void)encoder.Encode(image);  // uploads weights and warms up
    const auto start = std::chrono::steady_clock::now();
    const auto device = encoder.Encode(image);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    std::printf("gpu encode %ux%u (%u rows): %.1f ms\n", width, height,
                device->rows(), ms);
    if (reference_enabled) {
      (void)encoder.Encode(
          image, [&](std::string_view stage, std::span<const float> data) {
            const auto d = Compare(
                stages.at(std::string(stage)), data,
                stage == "embedding" ? projection_width : vision::kHidden);
            std::printf("gpu %-10s rel rms %.3e  max abs %.3e\n",
                        std::string(stage).c_str(), d.relative_rms, d.max_abs);
          });
    } else {
      embedding.resize(std::size_t{device->rows()} * projection_width);
      if (hipMemcpy(embedding.data(), device->data(), embedding.size() * 4,
                    hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("cannot copy embeddings");
      }
    }
#else
    throw std::runtime_error("built without HIP");
#endif
  }
  if (!output.empty() && !embedding.empty()) {
    std::ofstream out(output, std::ios::binary);
    const std::uint32_t header[4] = {
        0x45563447U, 1U,
        static_cast<std::uint32_t>(embedding.size() / projection_width),
        projection_width};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(embedding.data()),
              static_cast<std::streamsize>(embedding.size() * 4));
  }
  if (!compare.empty()) {
    const auto other = ReadEmbedding(compare);
    if (other.size() != embedding.size()) {
      throw std::runtime_error("embedding shapes differ");
    }
    const auto d = Compare(embedding, other, projection_width);
    std::printf("vs %s: worst row cosine %.6f  rel rms %.3e  max abs %.3e\n",
                compare.c_str(), d.worst_row_cosine, d.relative_rms, d.max_abs);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
