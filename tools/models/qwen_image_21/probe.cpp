#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/core/image.hpp"
#include "src/models/qwen_image_21/model.hpp"

int main(int argc, char** argv) {
  using namespace gufo::models::qwen_image_21;
  if (argc < 3) {
    std::cerr << "usage: qwen_image_21_probe MODEL_DIR OUTPUT_DIR [PROMPT] "
                 "[SIZE] [STEPS] [IMAGE|-] [--trajectory]\n";
    return 2;
  }
  try {
    const auto output = std::filesystem::path(argv[2]);
    std::filesystem::create_directories(output);
    Model model(argv[1]);
    Request request;
    request.prompt = argc > 3 ? argv[3] : "A red cube on a white table.";
    request.width = request.height = argc > 4 ? std::stoi(argv[4]) : 256;
    request.steps = argc > 5 ? std::stoi(argv[5]) : 2;
    request.seed = 42;
    if (argc > 6 && std::string_view(argv[6]) != "-")
      request.images.push_back(DecodeImage(gufo::core::ReadImageFile(argv[6])));
    const bool trajectory =
        argc > 7 && std::string_view(argv[7]) == "--trajectory";
    if (argc > 8 || (argc > 7 && !trajectory))
      throw std::invalid_argument("unknown probe option");
    std::ofstream ids(output / "tokens.txt");
    for (auto id : model.Tokenize(request.prompt))
      ids << id << '\n';
    Observer observer;
    observer.include = [trajectory](std::string_view name) {
      return !trajectory || name == "noise" || name == "sigmas" ||
             name == "vae.decoded" || name.starts_with("latent.");
    };
    observer.write = [&](std::string_view name, std::span<const float> values,
                         int rows, int cols) {
      const auto path = output / (std::string(name) + ".f32");
      std::ofstream stream(path, std::ios::binary);
      stream.write(reinterpret_cast<const char*>(values.data()),
                   values.size_bytes());
      std::cout << name << ' ' << rows << 'x' << cols << '\n' << std::flush;
    };
    const auto result = model.Generate(request, {}, observer);
    const auto png = EncodePng(result.image);
    std::ofstream image(output / "image.png", std::ios::binary);
    image.write(reinterpret_cast<const char*>(png.data()), png.size());
    std::cout << "total_ms=" << result.metrics.total_ms
              << " prompt_ms=" << result.metrics.prompt_ms
              << " denoise_ms=" << result.metrics.denoise_ms
              << " vae_ms=" << result.metrics.vae_ms << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
