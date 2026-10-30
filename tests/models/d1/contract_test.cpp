#include <cmath>
#include <iostream>
#include <stdexcept>

#include "src/models/d1/model.hpp"
#include "src/models/lfm3_vl/vision.hpp"
namespace d = gufo::models::d1;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    d::Question q;
    q.type = "noul";
    q.temperature = 2;
    q.readout = {{0, 2}, {1}};
    const float logits[] = {0, 2, 4};
    auto yes = d::ReadAnswer(q, logits);
    Require(
        std::abs(yes.member_double("noul") - 1 / (1 + std::exp(-1.0))) < 1e-12,
        "yes forms must max-pool before temperature scaling");
    q.type = "choice";
    q.labels = {"left", "right"};
    auto choice = d::ReadAnswer(q, logits);
    Require(choice.member_str("choice") == "left", "choice argmax");
    q.type = "score";
    q.labels = {"low", "middle", "high"};
    q.temperature = 1;
    q.readout = {{0}, {1}, {2}};
    const float ordinal[] = {std::log(0.40F), std::log(0.35F), std::log(0.25F)};
    auto score = d::ReadAnswer(q, ordinal);
    Require(score.member_size("score") == 1,
            "score must select median, not argmax");
    q.temperature = 0;
    bool rejected = false;
    try {
      (void)d::ReadAnswer(q, ordinal);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "zero temperature accepted");
    gufo::core::Image image{256, 256, std::vector<std::uint8_t>(256 * 256 * 3)};
    // Exact-size input isolates merge-block/channel/patch layout from resize.
    for (unsigned y = 0; y < 256; ++y)
      for (unsigned x = 0; x < 256; ++x)
        for (unsigned c = 0; c < 3; ++c)
          image.pixels[(y * 256 + x) * 3 + c] = (y / 16) * 8 + x / 16 + c;
    auto patches = gufo::models::lfm3_vl::ProcessImage(image, 1638400);
    Require(patches.height == 16 && patches.width == 16 &&
                patches.patches.size() == 256 * 768,
            "wrong d1 image grid");
    for (unsigned i = 0; i < 256; ++i) {
      const unsigned y = i / 4 / 8 * 2 + i % 4 / 2, x = i / 4 % 8 * 2 + i % 2;
      for (unsigned c = 0; c < 3; ++c)
        Require(std::abs(patches.patches[i * 768 + c * 256] -
                         ((y * 8 + x + c) / 255.0F * 2 - 1)) < 2e-7,
                "wrong merged patch order or normalization");
    }
    // Independent torchvision uint8 antialiased bicubic golden (73x91 ->
    // 256x288). The model processor uses torchvision rather than PIL's
    // fixed-22-bit filter coefficients.
    gufo::core::Image small{73, 91, std::vector<std::uint8_t>(73 * 91 * 3)};
    for (unsigned y = 0; y < 91; ++y)
      for (unsigned x = 0; x < 73; ++x) {
        small.pixels[(y * 73 + x) * 3] = (x * 17 + y * 5) % 256;
        small.pixels[(y * 73 + x) * 3 + 1] = (x * 3 + y * 23) % 256;
        small.pixels[(y * 73 + x) * 3 + 2] = (x * 13 + y * 7) % 256;
      }
    auto resized = gufo::models::lfm3_vl::ProcessImage(small, 1638400);
    Require(resized.width == 16 && resized.height == 18,
            "wrong smart resize dimensions");
    std::uint64_t sum = 0, squares = 0;
    for (float v : resized.patches) {
      const auto pixel = std::uint64_t(std::lround((v + 1) * 127.5F));
      sum += pixel;
      squares += pixel * pixel;
    }
    Require(sum == 28224533 && squares == 4704744449ULL,
            "bicubic pixels differ from independent torchvision golden");
    image.pixels.pop_back();
    rejected = false;
    try {
      (void)gufo::models::lfm3_vl::ProcessImage(image, 1638400);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Require(rejected, "malformed RGB image accepted");
    d::Question box;
    box.type = "box";
    std::vector<std::vector<double>> distributions(5);
    distributions[0] = {1.0, 0.0};
    const int bins[] = {100, 900, 200, 800};
    for (int edge = 0; edge < 4; ++edge) {
      distributions[edge + 1].resize(1000);
      distributions[edge + 1][bins[edge]] = 1;
    }
    auto read = d::ReadBoxDistributions(box, distributions);
    Require(read.find("box") &&
                std::abs(read.find("box")->member_double("top") - .1) < 1e-12,
            "box edge median differs");
    Require(
        std::abs(read.member_double("confidence") - 0.936923076923077) < 1e-12,
        "box Halton confidence/calibration differs");
    distributions[0] = {0.0, 1.0};
    read = d::ReadBoxDistributions(box, distributions);
    Require(
        read.find("box")->is_null() && read.member_double("confidence") == 0,
        "absent box must be null");
    distributions[0] = {1.0, 0.0};
    std::swap(distributions[1], distributions[2]);
    read = d::ReadBoxDistributions(box, distributions);
    Require(
        std::abs(read.find("box")->member_double("top") - .4995) < 1e-12 &&
            std::abs(read.find("box")->member_double("bottom") - .5005) < 1e-12,
        "crossed edges must form a single thousandth bin");
    box.type = "boxes";
    box.blocks = 3;
    auto many = distributions;
    many.insert(many.end(), distributions.begin(), distributions.end());
    many.insert(many.end(), distributions.begin(), distributions.end());
    many[5] = {.01, .99};
    read = d::ReadBoxDistributions(box, many);
    Require(read.member_size("count") == 1 && !read.find("at_least")->as_bool(),
            "boxes presence must stay decreasing");
    Require(read.find("presence")->items()[2].as_double() ==
                read.find("presence")->items()[1].as_double(),
            "later presence cannot increase");
    box.blocks = 1;
    read = d::ReadBoxDistributions(box, distributions);
    Require(read.member_size("count") == 1 && read.find("at_least")->as_bool(),
            "partial boxes must report at_least");
    std::cout << "PASS d1 readout and image contracts\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
