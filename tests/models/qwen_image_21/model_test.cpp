#include "src/models/qwen_image_21/model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/tokenizer.hpp"

int main(int argc, char** argv) {
  using namespace gufo::models::qwen_image_21;
  if (argc > 2)
    return 2;
  if (argc == 2) {
    // Official tokenizers output from the checkpoint pinned in UPSTREAM.md.
    // Optional local-artifact qualification; hosted tests never download
    // weights.
    Tokenizer tokenizer(argv[1]);
    const std::vector<std::pair<std::string, std::vector<std::uint32_t>>> cases{
        {"A red cube on a white table.",
         {32, 2518, 23739, 389, 264, 4158, 1965, 13}},
        {"  cafe\xcc\x81  déjà vu", {220, 51950, 220, 45839, 32514}},
        {"नमस्ते 世界 안녕하세요",
         {60096, 87244, 78368, 30484, 97, 34370, 220, 99489, 95170, 144370,
          91145}},
        {"if x == 123:\n    print(\"hello\")\n\n",
         {333, 856, 621, 220, 16, 17, 18, 510, 262, 1173, 445, 14990, 5130}},
        {"I'm don't they've won't",
         {40, 2776, 1513, 944, 807, 3003, 2765, 944}},
        {"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|"
         ">",
         {151644, 872, 198, 27, 1805, 16, 29, 151652, 151655, 151653}}};
    for (const auto& [text, expected] : cases)
      assert(tokenizer.Encode(text) == expected);
  }
  const Image image{
      2, 2, {255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 127, 63, 21, 255}};
  const auto roundtrip = DecodeImage(EncodePng(image));
  assert(roundtrip.width == image.width && roundtrip.height == image.height);
  assert(roundtrip.rgba == image.rgba);
  assert(ResizeImage(image, 2, 2).rgba == image.rgba);
  assert(ResizeImage(image, 4, 4).rgba.size() == 64);
  // Pillow RGBA Lanczos: premultiplication, two uint8 passes, unpremultiply.
  const std::vector<std::uint8_t> resized{
      255, 0, 0, 255, 225, 33,  0,  254, 87,  180, 0,  151, 0,   255, 0,  80,
      255, 0, 0, 196, 215, 37,  1,  186, 103, 131, 6,  167, 63,  238, 10, 141,
      255, 0, 0, 59,  179, 46,  10, 98,  120, 81,  16, 187, 132, 106, 22, 220,
      0,   0, 0, 0,   8,   102, 51, 30,  130, 48,  23, 203, 175, 61,  29, 255};
  assert(ResizeImage(image, 4, 4).rgba == resized);
  const auto schedule = FlowSigmas(40, 4096);
  assert(schedule.size() == 41);
  assert(schedule.front() == 1 && schedule.back() == 0);
  assert(std::abs(schedule[39] - 0.02F) < 1e-6F);
  for (std::size_t i = 1; i < schedule.size(); ++i)
    assert(std::isfinite(schedule[i]) && schedule[i] < schedule[i - 1]);
  ValidateRequest(Request{.prompt = "a red cube"});
  for (const auto invalid :
       {Request{.width = 31}, Request{.steps = 1}, Request{.height = 8192}}) {
    bool rejected = false;
    try {
      ValidateRequest(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  int calls = 0;
  gufo::server::ImageService service(
      {}, "image-test", [&](const Request& request, const CancellationCheck&) {
        ++calls;
        assert(request.prompt == "a red cube");
        assert(request.width == 256 && request.height == 256);
        assert(request.seed == static_cast<std::uint64_t>(calls + 40));
        return Result{.image = image};
      });
  gufo::server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  http.body =
      R"({"model":"image-test","prompt":"a red cube","size":"256x256","n":2,"seed":41})";
  const auto response = gufo::server::HandleImageApiRequest(http, service);
  assert(response.status == 200 && calls == 2);
  const auto json = gufo::json::parse(response.body);
  assert(json.find("data")->size() == 2);
  for (const auto body :
       {R"({"prompt":)", R"({"prompt":"a red cube","seed":-1})",
        R"({"prompt":"a red cube","n":1.5})",
        R"({"prompt":"a red cube","n":0})",
        R"({"prompt":"a red cube","output_format":"webp"})",
        R"({"prompt":"a red cube","stream":true})",
        R"({"prompt":"a red cube","size":"256x256junk"})",
        R"({"prompt":"a red cube","unknown":true})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, service).status == 400);
  }
  http.body = R"({"prompt":"a red cube","model":"unknown"})";
  assert(gufo::server::HandleImageApiRequest(http, service).status == 404);
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  const auto png = EncodePng(image);
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image[]\"; "
      "filename=\"x.png\"\r\n"
      "Content-Type: image/png\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] : {std::pair{"model", "image-test"},
                                    {"prompt", "a red cube"},
                                    {"size", "256x256"},
                                    {"seed", "43"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, service).status == 200 &&
         calls == 3);
  http.body.resize(http.body.size() - 8);
  assert(gufo::server::HandleImageApiRequest(http, service).status == 400);
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
