// The A1111 (sdapi) image API in stable-diffusion.cpp's dialect, and its
// separation from the OpenAI image API. Host-only: a callback runner records
// the requests, so no weights load and no sockets open.
#include "src/cli/serve/sdapi.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/core/base64.hpp"
#include "src/core/json.hpp"

namespace {
using namespace gufo;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error("check failed: " + std::string(message));
}

struct Recorder {
  std::vector<ext::ImageRequest> seen;
  bool exhausted{false};
  // Report the applied guidance scale, as the Qwen-Image runner does.
  bool reports_guidance{true};
};

server::ImageService Service(
    Recorder& recorder,
    ext::ImageCapabilities capabilities = {.edits = true,
                                           .transparent_background = true,
                                           .guidance = true,
                                           .image_to_image = true},
    server::ImageLoraOptions loras = {}) {
  return server::ImageService(
      std::make_shared<server::CallbackImageRunner>(
          [&recorder](const ext::ImageRequest& request,
                      const ext::CancellationCheck&) {
            if (recorder.exhausted)
              throw ext::ImageResourceError("does not fit now");
            recorder.seen.push_back(request);
            return ext::ImageResult{
                .image = {.width = 2,
                          .height = 1,
                          .rgba = {1, 2, 3, 255, 4, 5, 6, 255}},
                .metrics = {
                    .steps = request.steps ? request.steps : 40,
                    .guidance_scale =
                        recorder.reports_guidance
                            ? std::optional(request.guidance_scale.value_or(1))
                            : std::nullopt}};
          },
          capabilities),
      "Qwen-Image-2.1", {"qwen-image"}, std::move(loras));
}

server::HttpRequest Request(std::string method, std::string path,
                            std::string body = {}) {
  server::HttpRequest request;
  request.method = std::move(method);
  request.path = std::move(path);
  request.body = std::move(body);
  request.headers = {{"Content-Type", "application/json"}};
  request.request_id = "r1";
  return request;
}

server::HttpResponse Post(server::ImageService& service, std::string_view route,
                          std::string body) {
  return server::HandleSdapiRequest(
      Request("POST", "/sdapi/v1/" + std::string(route), std::move(body)),
      service);
}

json::Value Body(const server::HttpResponse& response) {
  return json::parse(response.body);
}

void RequireError(const server::HttpResponse& response, int status,
                  std::string_view message) {
  Require(response.status == status,
          std::to_string(response.status) + " " + response.body);
  const auto body = Body(response);
  Require(body.is_object() && body.size() == 1 &&
              body.member_str("error").find(message) != std::string::npos,
          response.body);
}

std::string PngBase64(int width, int height) {
  ext::ImageBuffer image{width, height, {}};
  image.rgba.assign(static_cast<std::size_t>(width) * height * 4, 128);
  return core::EncodeBase64(server::EncodePng(image));
}

void TestTxt2Img() {
  Recorder recorder;
  auto service = Service(recorder);
  // Model defaults, not A1111's processing defaults.
  auto response = Post(service, "txt2img", R"({"prompt":"a red cube"})");
  Require(response.status == 200, response.body);
  auto& request = recorder.seen.back();
  Require(request.prompt == "a red cube" && request.negative_prompt.empty() &&
              !request.guidance_scale && request.steps == 0 &&
              request.width == 1024 && request.height == 1024 &&
              request.strength == 1 && !request.init_image &&
              request.loras.empty(),
          "txt2img defaults");
  auto body = Body(response);
  Require(body.find("images")->size() == 1 &&
              body.find("parameters")->member_str("prompt") == "a red cube",
          "response shape");
  std::vector<std::uint8_t> png;
  Require(core::DecodeBase64(body.find("images")->items()[0].str(), {}, png) ==
                  core::Base64Status::kOk &&
              server::DecodeImage(png).width == 2,
          "plain base64 PNG");
  const auto* info_text = body.find("info");
  Require(info_text && info_text->is_string(), "info is a JSON string");
  auto info = json::parse(info_text->str());
  // An omitted cfg_scale reports the scale the runner applied.
  Require(info.find("cfg_scale")->as_double() == 1 &&
              info.find("steps")->as_double() == 40 &&
              info.member_str("sampler_name") == "Euler" &&
              !info.contains("denoising_strength") &&
              info.find("all_seeds")->size() == 1 &&
              info.find("infotexts")->size() == 1,
          info_text->str());

  // Client values, A1111 extras ignored, seeds advancing per batch image.
  response = Post(service, "txt2img", R"({
    "prompt":"a <lora:style:1> cube","negative_prompt":"blur","cfg_scale":7,
    "steps":20,"seed":42,"width":512,"height":768,"batch_size":3,"n_iter":4,
    "sampler_name":"DPM++ 2M","scheduler":"karras","clip_skip":2,
    "denoising_strength":0.7,"init_images":["x"],"enable_hr":false,
    "restore_faces":false,"override_settings":{},"guidance_scale":2,
    "loras":[{"name":"x"}],"size":"256x256","n":2,"seed_resize_from_h":-1,
    "alwayson_scripts":{}})");
  Require(response.status == 200, response.body);
  Require(recorder.seen.size() == 4, "batch_size images, n_iter ignored");
  for (int i = 0; i < 3; ++i) {
    const auto& image = recorder.seen[1 + i];
    Require(image.prompt == "a <lora:style:1> cube" &&
                image.negative_prompt == "blur" &&
                image.guidance_scale == 7.0F && image.steps == 20 &&
                image.seed == 42U + i && image.width == 512 &&
                image.height == 768 && image.strength == 1 &&
                !image.init_image && image.loras.empty(),
            "txt2img values");
  }
  body = Body(response);
  info = json::parse(body.member_str("info"));
  Require(
      body.find("images")->size() == 3 &&
          info.find("seed")->as_double() == 42 &&
          info.find("cfg_scale")->as_double() == 7 &&
          info.find("all_seeds")->items()[2].as_double() == 44 &&
          info.find("infotexts")
                  ->items()[0]
                  .str()
                  .find("Negative prompt: blur\nSteps: 20, Sampler: Euler, CFG "
                        "scale: 7, Seed: 42, Size: 512x768") !=
              std::string::npos,
      body.member_str("info"));
  Require(body.find("parameters")->find("sampler_name")->str() == "DPM++ 2M",
          "parameters echo the request");

  // A runner that does not report its scale leaves an omitted one null.
  recorder.reports_guidance = false;
  response = Post(service, "txt2img", R"({"prompt":"p"})");
  Require(
      response.status == 200 && json::parse(Body(response).member_str("info"))
                                    .find("cfg_scale")
                                    ->is_null(),
      response.body);
  recorder.reports_guidance = true;

  // Null means default; any negative seed is random.
  response = Post(service, "txt2img",
                  R"({"prompt":"p","seed":-7,"steps":null,"cfg_scale":null})");
  Require(response.status == 200 && recorder.seen.back().steps == 0 &&
              !recorder.seen.back().guidance_scale,
          response.body);

  const auto count = recorder.seen.size();
  RequireError(Post(service, "txt2img", ""), 400, "empty body");
  RequireError(Post(service, "txt2img", "{"), 400, "invalid JSON");
  RequireError(Post(service, "txt2img", "[]"), 400, "JSON object");
  RequireError(Post(service, "txt2img", R"({"prompt":""})"), 400,
               "prompt required");
  RequireError(Post(service, "txt2img", R"({"negative_prompt":"x"})"), 400,
               "prompt required");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","width":0})"), 400,
               "width and height must be positive");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","width":500})"), 400,
               "multiples of 32");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","steps":1})"), 400,
               "steps");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","steps":"20"})"), 400,
               "steps must be a number");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","batch_size":11})"),
               400, "batch_size");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","cfg_scale":101})"),
               400, "cfg_scale");
  RequireError(Post(service, "txt2img", R"({"prompt":"p","seed":1.5})"), 400,
               "seed");
  Require(recorder.seen.size() == count, "no inference for rejections");

  recorder.exhausted = true;
  RequireError(Post(service, "txt2img", R"({"prompt":"p"})"), 503,
               "does not fit now");

  // A model without guidance refuses a negative prompt in sdapi's shape.
  Recorder narrow_recorder;
  auto narrow = Service(narrow_recorder, {});
  RequireError(Post(narrow, "txt2img", R"({"prompt":"p","cfg_scale":7})"), 400,
               "does not support negative_prompt or guidance_scale");
  Require(
      Post(narrow, "txt2img", R"({"prompt":"p","cfg_scale":1})").status == 200,
      "unguided scale accepted");
}

void TestImg2Img() {
  Recorder recorder;
  auto service = Service(recorder);
  const auto init = PngBase64(100, 60);
  auto response = Post(service, "img2img",
                       R"({"prompt":"p","init_images":[")" + init +
                           R"(","ignored"],"denoising_strength":0.5})");
  Require(response.status == 200, response.body);
  auto request = recorder.seen.back();
  Require(request.init_image && request.init_image->width == 100 &&
              request.width == 96 && request.height == 64 &&
              request.strength == 0.5 && request.images.empty(),
          "first init image on the 32-pixel grid");
  auto info = json::parse(Body(response).member_str("info"));
  Require(info.find("denoising_strength")->as_double() == 0.5 &&
              info.find("steps")->is_null(),
          "img2img info");

  // Omitted or negative strength is sd.cpp's and A1111's 0.75; values
  // clamp into range.
  for (const auto& [field, strength] :
       {std::pair<std::string, double>{"", 0.75},
        {R"(,"denoising_strength":-1)", 0.75},
        {R"(,"denoising_strength":1.5)", 1},
        {R"(,"denoising_strength":0)", 0.01}}) {
    response = Post(service, "img2img",
                    R"({"prompt":"p","init_images":["data:image/png;base64,)" +
                        init + R"("])" + field + "}");
    Require(response.status == 200 && recorder.seen.back().strength == strength,
            field + response.body);
  }
  // One given side keeps the aspect ratio; both given are used as sent.
  response =
      Post(service, "img2img",
           R"({"prompt":"p","width":320,"init_images":[")" + init + R"("]})");
  Require(response.status == 200 && recorder.seen.back().width == 320 &&
              recorder.seen.back().height == 192,
          "aspect from init image");
  response = Post(service, "img2img",
                  R"({"prompt":"p","width":256,"height":512,"init_images":[")" +
                      init + R"("]})");
  Require(response.status == 200 && recorder.seen.back().width == 256 &&
              recorder.seen.back().height == 512,
          "explicit size");
  // A canvas over the size limits scales down to fit.
  response = Post(
      service, "img2img",
      R"({"prompt":"p","init_images":[")" + PngBase64(4200, 40) + R"("]})");
  Require(response.status == 200 && recorder.seen.back().width == 4096 &&
              recorder.seen.back().height == 32,
          "oversized canvas fitted");

  RequireError(Post(service, "img2img", R"({"prompt":"p"})"), 400,
               "init_images required");
  RequireError(Post(service, "img2img", R"({"prompt":"p","init_images":[]})"),
               400, "init_images required");
  RequireError(
      Post(service, "img2img",
           R"({"prompt":"p","init_images":["https://example.com/a.png"]})"),
      400, "base64");
  RequireError(
      Post(service, "img2img",
           R"({"prompt":"p","mask":"x","init_images":[")" + init + R"("]})"),
      400, "mask");

  Recorder plain_recorder;
  auto plain = Service(plain_recorder, {.guidance = true});
  RequireError(
      Post(plain, "img2img",
           R"({"prompt":"p","denoising_strength":0.5,"init_images":[")" + init +
               R"("]})"),
      400, "image to image");
}

void TestLoras() {
  Recorder recorder;
  auto service = Service(recorder, {.guidance = true},
                         {.files = {{"style", "/x/one/lora.safetensors"},
                                    {"ink", "/x/two/lora.safetensors"},
                                    {"fix", "/x/fix.safetensors"}},
                          .defaults = {{"fix", 0.5F}}});
  const auto listed =
      server::HandleSdapiRequest(Request("GET", "/sdapi/v1/loras"), service);
  Require(listed.status == 200, listed.body);
  const auto list = Body(listed);
  Require(list.size() == 3 && list.items()[0].member_str("name") == "style" &&
              list.items()[0].member_str("path") == "style/lora.safetensors" &&
              list.items()[0].member_str("display_path") ==
                  "/x/one/lora.safetensors" &&
              list.items()[2].member_str("path") == "fix.safetensors" &&
              list.items()[2].find("default_scale")->as_double() == 0.5,
          listed.body);

  const auto loras = [&](const std::string& field) {
    const auto response =
        Post(service, "txt2img", R"({"prompt":"p")" + field + "}");
    Require(response.status == 200, response.body);
    return recorder.seen.back().loras;
  };
  using Loras = std::vector<ext::ImageLora>;
  Require(loras("") == Loras{{"fix", 0.5F}}, "omitted uses the defaults");
  Require(loras(R"(,"lora":null)") == Loras{{"fix", 0.5F}}, "null as omitted");
  Require(
      loras(R"(,"lora":{"path":"fix.safetensors"})") == Loras{{"fix", 0.5F}},
      "non-array ignored");
  Require(loras(R"(,"lora":[])").empty(), "empty list clears the defaults");
  Require(loras(R"(,"lora":[{"path":"style/lora.safetensors","multiplier":0.8},
                {"path":"fix.safetensors"}])") ==
              Loras{{"style", 0.8F}, {"fix", 1}},
          "two adapters");
  Require(loras(R"(,"lora":[{"path":"fix.safetensors","multiplier":0.25},
                "skip",{"path":"ink/lora.safetensors","multiplier":-1},
                {"path":"fix.safetensors","multiplier":0.5}])") ==
              Loras{{"fix", 0.75F}, {"ink", -1}},
          "repeated paths sum, non-objects skipped");
  Require(loras(R"(,"lora":[{"path":"fix.safetensors","is_high_noise":true}])")
              .empty(),
          "high-noise entries skipped on a single-stage model");

  const auto count = recorder.seen.size();
  RequireError(Post(service, "txt2img",
                    R"({"prompt":"p","lora":[{"path":"none.safetensors"}]})"),
               400, "invalid lora path: none.safetensors");
  RequireError(Post(service, "txt2img",
                    R"({"prompt":"p","lora":[{"path":"lora.safetensors"}]})"),
               400, "invalid lora path");
  RequireError(
      Post(service, "txt2img", R"({"prompt":"p","lora":[{"multiplier":1}]})"),
      400, "lora.path required");
  RequireError(Post(service, "txt2img",
                    R"({"prompt":"p","lora":[{"path":"fix.safetensors",
                    "multiplier":6},{"path":"fix.safetensors","multiplier":6}]})"),
               400, "at most 10");
  RequireError(Post(service, "txt2img",
                    R"({"prompt":"p","lora":[{"path":"fix.safetensors",
                    "is_high_noise":1}]})"),
               400, "is_high_noise");
  Require(recorder.seen.size() == count, "no inference for rejections");

  // The OpenAI route keeps its own schema: `lora` there is unknown.
  auto openai = Request("POST", "/v1/images/generations",
                        R"({"prompt":"p","size":"256x256","cfg_scale":7,
                        "lora":[],"batch_size":3,"denoising_strength":0.5})");
  const auto response = server::HandleImageApiRequest(openai, service);
  Require(response.status == 200 && recorder.seen.size() == count + 1 &&
              recorder.seen.back().loras == Loras{{"fix", 0.5F}} &&
              !recorder.seen.back().guidance_scale &&
              recorder.seen.back().strength == 1,
          "sdapi fields are not OpenAI fields");
}

void TestDiscovery() {
  Recorder recorder;
  auto service = Service(recorder);
  const auto get = [&](std::string_view route) {
    const auto response = server::HandleSdapiRequest(
        Request("GET", "/sdapi/v1/" + std::string(route)), service);
    Require(response.status == 200, std::string(route) + response.body);
    return Body(response);
  };
  auto options = get("options");
  Require(options.member_str("samples_format") == "png" &&
              options.member_str("sd_model_checkpoint") == "Qwen-Image-2.1",
          "options");
  const auto models = get("sd-models");
  Require(models.size() == 1 &&
              models.items()[0].member_str("title") == "Qwen-Image-2.1" &&
              models.items()[0].member_str("model_name") == "Qwen-Image-2.1" &&
              models.items()[0].contains("sha256") &&
              models.items()[0].find("config")->is_null(),
          "sd-models");
  Require(get("samplers").items()[0].member_str("name") == "Euler" &&
              get("schedulers").size() == 1 &&
              get("upscalers").items()[0].member_str("name") == "None" &&
              get("latent-upscale-modes").is_array(),
          "selection lists");
  const auto progress = get("progress");
  Require(progress.find("progress")->as_double() == 0 &&
              progress.find("state")->find("job_count")->as_double() == 0 &&
              progress.find("current_image")->is_null(),
          "idle progress");

  // Clients select the loaded checkpoint (by ID or alias) before generating.
  for (const auto* model : {"Qwen-Image-2.1", "qwen-image"}) {
    options["sd_model_checkpoint"] = model;
    Require(Post(service, "options", options.dump()).status == 200, model);
  }
  RequireError(Post(service, "options", R"({"sd_model_checkpoint":"sdxl"})"),
               400, "Qwen-Image-2.1");
  Require(
      Post(service, "options", R"({"CLIP_stop_at_last_layers":2})").status ==
          200,
      "other options accepted");

  RequireError(
      server::HandleSdapiRequest(Request("GET", "/sdapi/v1/txt2img"), service),
      405, "POST");
  RequireError(server::HandleSdapiRequest(
                   Request("POST", "/sdapi/v1/sd-models"), service),
               405, "GET");
  RequireError(
      server::HandleSdapiRequest(Request("POST", "/sdapi/v1/loras"), service),
      405, "GET");
  RequireError(server::HandleSdapiRequest(
                   Request("POST", "/sdapi/v1/interrupt"), service),
               404, "no sdapi route");
  Require(server::IsSdapiPath("/sdapi/v1/txt2img") &&
              !server::IsSdapiPath("/v1/images/generations"),
          "path prefix");
}

}  // namespace

int main() {
  try {
    TestTxt2Img();
    TestImg2Img();
    TestLoras();
    TestDiscovery();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "sdapi contracts passed\n";
  return 0;
}
