#include "src/models/qwen_image_21/model.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen_image_21/tokenizer.hpp"
#include "src/models/qwen_image_21/weights.hpp"

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
  // Independent checkpoint metadata, not a call to the native schedule builder.
  const auto turbo_index = gufo::json::parse(R"({"sample_sigmas":
    [1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568]})");
  const auto turbo_scheduler = gufo::json::parse(R"({
    "_class_name":"FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len":256,"max_image_seq_len":8192,
    "base_shift":0.5,"max_shift":0.9,"num_train_timesteps":1000,
    "time_shift_type":"exponential","shift":1.0,"shift_terminal":null,
    "use_dynamic_shifting":false,"invert_sigmas":false,
    "stochastic_sampling":false,"use_beta_sigmas":false,
    "use_exponential_sigmas":false,"use_karras_sigmas":false})");
  const auto turbo = ReadSamplingConfig(turbo_index, turbo_scheduler);
  const std::vector<float> expected_turbo{1,         0.978453F, 0.95418F,
                                          0.926626F, 0.89508F,  0.845148F,
                                          0.704534F, 0.414568F, 0};
  assert(turbo.IsTurbo() && turbo.DefaultSteps() == 8);
  assert(turbo.ResolveSteps(0) == 8 && turbo.ResolveSteps(8) == 8);
  assert(turbo.Sigmas(0, 256) == expected_turbo);
  assert(turbo.Sigmas(8, 8160) == expected_turbo);
  const SamplingConfig base;
  assert(!base.IsTurbo() && base.ResolveSteps(0) == 40);
  assert(base.Sigmas(0, 4096) == schedule);
  auto base_scheduler = turbo_scheduler;
  base_scheduler["use_dynamic_shifting"] = true;
  base_scheduler["shift_terminal"] = 0.02;
  assert(!ReadSamplingConfig(gufo::json::Value::object(), base_scheduler)
              .IsTurbo());
  const auto reject_config = [](const auto& index, const auto& scheduler) {
    bool rejected = false;
    try {
      (void)ReadSamplingConfig(index, scheduler);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  };
  reject_config(turbo_index, base_scheduler);
  reject_config(gufo::json::Value::object(), turbo_scheduler);
  for (const auto* sigmas :
       {"[]", "[1,0.5]", "[1,1,0.9,0.8,0.7,0.6,0.5,0.4]",
        "[1,0.9,0.8,0.7,0.6,0.5,0.4,0]", "[1,0.9,0.8,0.7,0.6,0.5,0.4,1e-100]",
        "[1,0.9,0.8,0.7,0.6,0.5,0.4,\"x\"]"})
    reject_config(
        gufo::json::parse(std::string("{\"sample_sigmas\":") + sigmas + "}"),
        turbo_scheduler);
  for (const int steps : {2, 7, 9, 40, 100}) {
    bool rejected = false;
    try {
      (void)turbo.ResolveSteps(steps);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
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
  int turbo_calls = 0;
  std::vector<Request> turbo_requests;
  gufo::server::ImageService turbo_service(
      {}, "turbo-test",
      [&](const Request& request, const CancellationCheck&) {
        assert(request.steps == 8);
        turbo_requests.push_back(request);
        ++turbo_calls;
        return Result{image, {}};
      },
      turbo);
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  for (const auto* body : {R"({"prompt":"cube","size":"256x256"})",
                           R"({"prompt":"cube","steps":8,"size":"256x256"})"}) {
    http.body = body;
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           200);
  }
  assert(turbo_calls == 2);
  for (const int steps : {0, 2, 7, 9, 40, 100}) {
    http.body = "{\"prompt\":\"cube\",\"steps\":" + std::to_string(steps) + "}";
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           400);
  }
  http.path = "/v1/images/edits";
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  assert(turbo_calls == 2);
  // Editing shares the ordered-reference path, including RGBA, masks,
  // background controls and independent outputs; only the schedule differs.
  const auto edit_body = [&](const std::vector<Image>& inputs, int steps,
                             const std::string& background, int count) {
    std::string body;
    for (const auto& input : inputs) {
      const auto bytes = EncodePng(input);
      body +=
          "--boundary\r\nContent-Disposition: form-data; "
          "name=\"image[]\"; filename=\"reference.png\"\r\n"
          "Content-Type: image/png\r\n\r\n";
      body.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      body += "\r\n";
    }
    std::vector<std::pair<std::string, std::string>> fields{
        {"model", "turbo-test"},
        {"prompt", "edit image one using image two"},
        {"size", "256x256"},
        {"seed", "77"},
        {"background", background},
        {"n", std::to_string(count)}};
    if (steps >= 0)
      fields.emplace_back("steps", std::to_string(steps));
    for (const auto& [name, value] : fields)
      body += "--boundary\r\nContent-Disposition: form-data; name=\"" + name +
              "\"\r\n\r\n" + value + "\r\n";
    return body + "--boundary--\r\n";
  };
  http.headers = {{"Content-Type", "multipart/form-data; boundary=boundary"}};
  Image mask{
      2,
      2,
      {255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255}};
  for (const int steps : {-1, 8}) {
    http.body = edit_body({image, mask}, steps, "transparent", 2);
    const auto edited =
        gufo::server::HandleImageApiRequest(http, turbo_service);
    assert(edited.status == 200);
    assert(gufo::json::parse(edited.body).find("data")->size() == 2);
    for (std::size_t i = turbo_requests.size() - 2; i < turbo_requests.size();
         ++i) {
      const auto& captured = turbo_requests[i];
      assert(captured.steps == 8 && captured.images.size() == 2);
      assert(captured.images[0].rgba == image.rgba);
      assert(captured.images[1].rgba == mask.rgba);
      assert(captured.seed == 77 + i % 2);
      assert(captured.prompt.find("RGBA image with transparency") !=
             std::string::npos);
    }
  }
  http.body = edit_body(std::vector<Image>(10, image), 8, "opaque", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         200);
  assert(turbo_requests.back().images.size() == 10);
  const auto accepted = turbo_calls;
  for (const int steps : {0, 7, 9, 40}) {
    http.body = edit_body({image}, steps, "auto", 1);
    assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
           400);
  }
  http.body = edit_body(std::vector<Image>(11, image), 8, "auto", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  http.body = edit_body({}, 8, "auto", 1);
  assert(gufo::server::HandleImageApiRequest(http, turbo_service).status ==
         400);
  assert(turbo_calls == accepted);
  std::cout << "Qwen-Image image and schedule contracts passed\n";
}
