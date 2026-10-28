#include "src/models/qwen_image_21/model.hpp"

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
  ValidateRequest(Request{.prompt = "a red cube",
                          .images = {},
                          .loras = {{"style", 0.8F}, {"ink", -10}}});
  for (auto loras : {std::vector<Lora>(17), std::vector<Lora>{{"", 1}},
                     std::vector<Lora>{{"style", 1}, {"style", 0.5F}},
                     std::vector<Lora>{{"style", NAN}},
                     std::vector<Lora>{{"style", INFINITY}},
                     std::vector<Lora>{{"style", 10.5F}}}) {
    for (std::size_t i = 0; loras.size() == 17 && i < loras.size(); ++i)
      loras[i].name = "a" + std::to_string(i);
    bool rejected = false;
    try {
      ValidateRequest(
          Request{.prompt = "a red cube", .images = {}, .loras = loras});
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
  // Adapters: registered by name only (an injected runner opens no files).
  std::vector<Lora> seen;
  gufo::server::ImageService adapters(
      {}, "image-test",
      [&](const Request& request, const CancellationCheck&) {
        seen = request.loras;
        return Result{.image = image, .metrics = {}};
      },
      {.files = {{"style", "/nonexistent/style.safetensors"},
                 {"ink", "/x/ink.safetensors"}},
       .defaults = {{"style", 0.8F}}});
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  const auto generate = [&](const std::string& loras) {
    http.body = R"({"prompt":"a red cube","size":"256x256")" +
                (loras.empty() ? "" : ",\"loras\":" + loras) + "}";
    return gufo::server::HandleImageApiRequest(http, adapters).status;
  };
  const std::vector<Lora> defaults{{"style", 0.8F}};
  const std::vector<Lora> both{{"ink", 0.8F}, {"style", 1}};
  const std::vector<Lora> negative{{"ink", -0.25F}};
  assert(generate("") == 200 && seen == defaults);
  assert(generate("[]") == 200 && seen.empty());
  assert(generate(R"([{"name":"ink","scale":0.8},{"name":"style"}])") == 200 &&
         seen == both);
  seen = {};
  for (const auto& bad : std::vector<std::string>{
           R"([{"name":"unknown"}])", R"([{"name":"ink"},{"name":"ink"}])",
           R"([{"name":"ink","scale":"x"}])", R"([{"name":"ink","path":"/x"}])",
           R"({"name":"ink"})", R"("ink")", R"([{"scale":1}])", R"(["ink"])",
           R"([{"name":"ink","scale":11}])", R"([{"name":"ink","scale":1e40}])",
           R"([{"name":"ink","scale":1e300}])"})
    assert(generate(bad) == 400 && seen.empty());
  // More than 16 distinct registered adapters.
  gufo::server::ImageLoraOptions seventeen;
  std::string many = "[";
  for (int i = 0; i < 17; ++i) {
    const auto name = "a" + std::to_string(i);
    seventeen.files.emplace_back(name, "/x/" + name + ".safetensors");
    many += std::string(i ? "," : "") + R"({"name":")" + name + R"("})";
  }
  gufo::server::ImageService capped(
      {}, "image-test",
      [&](const Request&, const CancellationCheck&) {
        return Result{.image = image, .metrics = {}};
      },
      std::move(seventeen));
  http.body = R"({"prompt":"a red cube","loras":)" + many + "]}";
  const auto over = gufo::server::HandleImageApiRequest(http, capped);
  assert(over.status == 400 &&
         over.body.find("at most 16") != std::string::npos);
  http.body = R"({"prompt":"a red cube","loras":[{"name":"unknown"}]})";
  assert(gufo::server::HandleImageApiRequest(http, adapters)
             .body.find("unknown LoRA adapter: unknown") != std::string::npos);
  // Multipart edits carry `loras` as a JSON field.
  http.path = "/v1/images/edits";
  http.headers = {
      {"Content-Type", "multipart/form-data; boundary=\"boundary\""}};
  http.body =
      "--boundary\r\nContent-Disposition: form-data; name=\"image\"\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] :
       {std::pair{"prompt", "a red cube"},
        {"size", "256x256"},
        {"loras", R"([{"name":"ink","scale":-0.25}])"}})
    http.body += "\r\n--boundary\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + value;
  http.body += "\r\n--boundary--\r\n";
  assert(gufo::server::HandleImageApiRequest(http, adapters).status == 200 &&
         seen == negative);
  http.method = "GET";
  http.path = "/sdapi/v1/loras";
  http.headers = {};
  http.body.clear();
  const auto listed = gufo::server::HandleImageLorasRequest(http, adapters);
  assert(listed.status == 200);
  const auto list = gufo::json::parse(listed.body);
  assert(list.is_array() && list.size() == 2);
  const auto& style = list.items()[0];
  const auto& ink = list.items()[1];
  assert(style.member_str("name") == "style" &&
         style.member_str("alias") == "style" &&
         style.member_str("path") == "style.safetensors");
  assert(ink.member_str("path") == "ink.safetensors");
  assert(style.find("metadata")->is_object());
  // Listed as the shortest float text; `active_scale` uses the same formatting.
  assert(style.find("default_scale")->as_double() == 0.8);
  assert(ink.find("default_scale")->is_null());
  // Only a loaded model merges adapters; an injected runner lists none.
  assert(style.find("active_scale")->is_null() &&
         ink.find("active_scale")->is_null());
  http.method = "POST";
  assert(gufo::server::HandleImageLorasRequest(http, adapters).status == 400);
  for (const auto& options : std::vector<gufo::server::ImageLoraOptions>{
           {.files = {{"a b", "/x"}}, .defaults = {}},
           {.files = {{"", "/x"}}, .defaults = {}},
           {.files = {{std::string(65, 'a'), "/x"}}, .defaults = {}},
           {.files = {{"a", "/x"}, {"a", "/y"}}, .defaults = {}},
           {.files = {{"a", "/x"}}, .defaults = {{"b", 1}}},
           {.files = {{"a", "/x"}}, .defaults = {{"a", 1}, {"a", 2}}},
           {.files = {{"a", "/x"}}, .defaults = {{"a", 12}}}}) {
    bool rejected = false;
    try {
      gufo::server::ImageService invalid(
          {}, "image-test",
          [&](const Request&, const CancellationCheck&) { return Result{}; },
          options);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
