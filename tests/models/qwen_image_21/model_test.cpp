#include "src/models/qwen_image_21/model.hpp"

#include <algorithm>
#include <array>
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
  // Initial noise must not repeat the noise that generated a reference at the
  // same seed and size. Re-editing the result must also use a new stream.
  assert(InitialNoiseSeed(42, {}) == 42);
  assert(InitialNoiseSeed(0, {}) == 0);
  const std::array references{image};
  const auto edit_seed = InitialNoiseSeed(42, references);
  // Independent hashlib SHA-256 golden, little-endian fields and digest.
  assert(edit_seed == 9781428742393689377ULL);
  assert(InitialNoiseSeed(42, references) == edit_seed);
  assert(InitialNoiseSeed(43, references) != edit_seed);
  auto changed = image;
  changed.rgba[0] ^= 1;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  changed = image;
  changed.rgba[3] ^= 1;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  const auto pair_seed = InitialNoiseSeed(42, std::array{image, changed});
  assert(pair_seed != edit_seed);
  assert(pair_seed != InitialNoiseSeed(42, std::array{changed, image}));
  changed = image;
  changed.width = 1;
  changed.height = 4;
  assert(InitialNoiseSeed(42, std::array{changed}) != edit_seed);
  const auto roundtrip = DecodeImage(EncodePng(image));
  assert(roundtrip.width == image.width && roundtrip.height == image.height);
  assert(roundtrip.rgba == image.rgba);
  assert(InitialNoiseSeed(42, std::array{roundtrip}) == edit_seed);
  Image textured{65, 17, std::vector<std::uint8_t>(65 * 17 * 4)};
  for (std::size_t i = 0; i < textured.rgba.size(); ++i)
    textured.rgba[i] = static_cast<std::uint8_t>((i * 73) ^ (i >> 3));
  const auto textured_roundtrip = DecodeImage(EncodePng(textured));
  assert(textured_roundtrip.width == textured.width &&
         textured_roundtrip.height == textured.height);
  assert(textured_roundtrip.rgba == textured.rgba);
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

  // Image to image: the start image binds its own noise stream.
  assert(InitialNoiseSeed(42, {}, &image) != 42);
  assert(InitialNoiseSeed(42, {}, &image) != edit_seed);
  assert(InitialNoiseSeed(42, references, &image) !=
         InitialNoiseSeed(42, {}, &image));
  // ComfyUI denoise: the last steps + 1 sigmas of int(steps / strength).
  assert(DenoiseSigmas(40, 1.0, 4096) == schedule);
  const auto full80 = DenoiseSigmas(80, 1.0, 4096);
  const auto half = DenoiseSigmas(40, 0.5, 4096);
  assert(half.size() == 41 && half.back() == 0);
  assert(std::equal(half.begin(), half.end(), full80.end() - 41));
  assert(DenoiseSigmas(25, 0.6, 1024).size() == 26);
  assert(DenoiseSigmas(25, 0.6, 1024).front() ==
         DenoiseSigmas(41, 1.0, 1024)[16]);
  for (const auto invalid : {Request{.guidance = 21}, Request{.strength = 0.5},
                             Request{.strength = 0}, Request{.guidance = -1}}) {
    bool rejected = false;
    try {
      ValidateRequest(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  ValidateRequest(Request{.init_image = image, .strength = 0.5});

  Request seen;
  gufo::server::ImageService extended(
      {}, "image-test", [&](const Request& request, const CancellationCheck&) {
        seen = request;
        return Result{.image = image};
      });
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  // Defaults stay the unguided full-strength generation.
  http.body = R"({"prompt":"a red cube","size":"256x256","seed":1})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.guidance == 1 && seen.negative_prompt.empty() &&
         seen.strength == 1 && !seen.init_image);
  http.body =
      R"({"prompt":"a red cube","size":"256x256","gufo":{"negative_prompt":"blurry","guidance":2.5}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.guidance == 2.5F && seen.negative_prompt == "blurry");
  std::string encoded;
  constexpr char kDigits[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (std::size_t i = 0; i < png.size(); i += 3) {
    const std::uint32_t v = png[i] << 16 |
                            (i + 1 < png.size() ? png[i + 1] << 8 : 0) |
                            (i + 2 < png.size() ? png[i + 2] : 0);
    const std::size_t digits = std::min<std::size_t>(png.size() - i, 3) + 1;
    for (std::size_t k = 0; k < 4; ++k)
      encoded += k < digits ? kDigits[(v >> (18 - 6 * k)) & 63] : '=';
  }
  http.body =
      R"({"prompt":"a red cube","gufo":{"strength":0.5,"init_image":")" +
      encoded + R"("}})";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.strength == 0.5 && seen.init_image &&
         seen.init_image->rgba == image.rgba && seen.images.empty());
  // A 2x2 start image rounds to the smallest 32-pixel canvas.
  assert(seen.width == 32 && seen.height == 32);
  for (const auto body :
       {R"({"prompt":"a","gufo":{"guidance":"2"}})",
        R"({"prompt":"a","gufo":{"strength":0.5}})",
        R"({"prompt":"a","gufo":{"unknown":1}})",
        R"({"prompt":"a","gufo":{"init_image":"https://example.com/x.png"}})",
        R"({"prompt":"a","gufo":[]})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, extended).status == 400);
  }
  // Multipart: one "image" plus gufo.strength is the start image.
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image\"; "
      "filename=\"x.png\"\r\n"
      "Content-Type: image/png\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] : {std::pair{"prompt", "a red cube"},
                                    {"size", "256x256"},
                                    {"gufo.strength", "0.5"},
                                    {"gufo.guidance", "3"},
                                    {"gufo.negative_prompt", "blurry"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, extended).status == 200);
  assert(seen.strength == 0.5 && seen.init_image && seen.images.empty() &&
         seen.guidance == 3 && seen.negative_prompt == "blurry" &&
         seen.width == 256);
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
