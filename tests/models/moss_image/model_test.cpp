#include "src/models/moss_image/model.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/models/qwen_image_21/weights.hpp"
#include "tests/support/temporary_directory.hpp"

namespace {
namespace moss = gufo::models::moss_image;
namespace image = gufo::models::qwen_image_21;

void Write(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out << text;
  assert(out.good());
}

bool Rejected(const moss::Request& request) {
  try {
    moss::ValidateRequest(request);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void Schedule() {
  // MOSSImageScheduler.set_timesteps(4).sigmas from the pinned pipeline
  // (UPSTREAM.md), as FP32 bit patterns.
  const auto four = moss::FlowSigmas(4);
  const std::vector<std::uint32_t> expected{0x3f800000, 0x3f68f5c2, 0x3f378d50,
                                            0x3cc49ba7, 0x00000000};
  assert(four.size() == expected.size());
  for (std::size_t i = 0; i < four.size(); ++i)
    assert(std::bit_cast<std::uint32_t>(four[i]) == expected[i]);
  for (const int steps : {1, 2, 30, 50, 100}) {
    const auto sigmas = moss::FlowSigmas(steps);
    // Snapping to the 1000-step training grid may merge adjacent steps.
    assert(sigmas.size() >= 2 &&
           sigmas.size() <= static_cast<std::size_t>(steps) + 1);
    assert(sigmas.front() == 1.0F && sigmas.back() == 0.0F);
    for (std::size_t i = 1; i < sigmas.size(); ++i)
      assert(std::isfinite(sigmas[i]) && sigmas[i] < sigmas[i - 1]);
  }
}

void Requests() {
  moss::ValidateRequest(moss::Request{.prompt = "a red cube"});
  moss::ValidateRequest(
      moss::Request{.width = 256, .height = 2048, .steps = 1});
  moss::ValidateRequest(moss::Request{.width = 1040, .height = 528});
  for (const auto& invalid :
       {moss::Request{.width = 240}, moss::Request{.height = 2064},
        moss::Request{.width = 1032}, moss::Request{.steps = 0},
        moss::Request{.steps = 101}})
    assert(Rejected(invalid));
  // Text-to-image only.
  moss::Request edit;
  edit.images.push_back(image::Image{1, 1, {0, 0, 0, 255}});
  assert(Rejected(edit));
}

void Detection() {
  TemporaryDirectory directory("gufo-moss-image");
  const auto root = directory.path();
  assert(!moss::IsMossImageDirectory(root / "missing"));
  Write(root / "moss/model_index.json",
        R"({"_class_name":"MOSSImagePipeline","_diffusers_version":"0.38.0"})");
  assert(moss::IsMossImageDirectory(root / "moss"));
  Write(root / "qwen/model_index.json",
        R"({"_class_name":"QwenImage21Pipeline"})");
  assert(!moss::IsMossImageDirectory(root / "qwen"));
  Write(root / "broken/model_index.json", R"({"_class_name":)");
  assert(!moss::IsMossImageDirectory(root / "broken"));
}

// One safetensors file holding an F32 tensor and an I64 counter, as in the
// FLUX.2 VAE's BatchNorm (num_batches_tracked).
void Safetensors(const std::filesystem::path& path, std::size_t i64_end) {
  std::string header =
      R"({"bn.running_mean":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
      R"("bn.num_batches_tracked":{"dtype":"I64","shape":[],"data_offsets":[8,)" +
      std::to_string(i64_end) + "]}}";
  std::ofstream out(path, std::ios::binary);
  const std::uint64_t size = header.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));
  out << header;
  const std::array<float, 2> mean{0.25F, -1.5F};
  out.write(reinterpret_cast<const char*>(mean.data()), sizeof(mean));
  const std::int64_t tracked = 7;
  out.write(reinterpret_cast<const char*>(&tracked), sizeof(tracked));
  assert(out.good());
}

void WeightFiles() {
  TemporaryDirectory directory("gufo-moss-weights");
  const auto good = directory.path() / "good.safetensors";
  Safetensors(good, 16);
  const std::array sources{image::SafetensorsSource{good, "vae."}};
  const auto weights = image::Weights::FromFiles(
      sources, gufo::json::parse(R"({"_class_name":"AutoencoderKLFlux2"})"));
  assert(weights.Contains("vae.bn.running_mean"));
  // Integer bookkeeping is skipped, not exposed as a weight.
  assert(!weights.Contains("vae.bn.num_batches_tracked"));
  const auto& mean = weights.Get("vae.bn.running_mean");
  assert(!mean.bf16 && mean.bytes == 8);
  assert(static_cast<const float*>(mean.data)[1] == -1.5F);
  assert(weights.VaeConfig().member_str("_class_name") == "AutoencoderKLFlux2");
  // The skipped extent still has to lie inside the payload.
  const auto bad = directory.path() / "bad.safetensors";
  Safetensors(bad, 24);
  bool rejected = false;
  try {
    const std::array bad_sources{image::SafetensorsSource{bad, "vae."}};
    (void)image::Weights::FromFiles(bad_sources);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

void Service() {
  const image::Image pixel{1, 1, {12, 34, 56, 255}};
  int calls = 0;
  gufo::server::ImageService service(
      {}, "MOSS-Image",
      [&](const image::Request& request, const image::CancellationCheck&) {
        ++calls;
        assert(request.prompt == "a red cube" && request.images.empty());
        return image::Result{.image = pixel};
      },
      false);
  assert(!service.editing());
  gufo::server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  http.body =
      R"({"model":"MOSS-Image","prompt":"a red cube","size":"256x256"})";
  assert(gufo::server::HandleImageApiRequest(http, service).status == 200);
  assert(calls == 1);
  http.body = R"({"prompt":"a red cube","background":"transparent"})";
  auto response = gufo::server::HandleImageApiRequest(http, service);
  assert(response.status == 400 &&
         response.body.find("transparent") != std::string::npos);
  http.path = "/v1/images/edits";
  http.headers = {{"Content-Type", "multipart/form-data; boundary=b"}};
  const auto png = image::EncodePng(pixel);
  http.body = "--b\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  http.body +=
      "\r\n--b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
      "a red cube\r\n--b--\r\n";
  response = gufo::server::HandleImageApiRequest(http, service);
  assert(response.status == 400 &&
         response.body.find("does not support image edits") !=
             std::string::npos);
  assert(calls == 1);
}

}  // namespace

int main() {
  Schedule();
  Requests();
  Detection();
  WeightFiles();
  Service();
  std::cout << "MOSS-Image schedule, request and serving contracts passed\n";
}
