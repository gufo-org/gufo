#include "src/cli/serve/sdapi.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/http_server.hpp"
#include "src/core/base64.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/ext/image.hpp"

namespace gufo::server {
namespace {

// The OpenAI route's `n` cap; stable-diffusion.cpp has none.
constexpr std::int64_t kMaxBatchSize = 10;
// Qwen-Image samples the flow-matching schedule with Euler steps; requested
// samplers and schedulers have no equivalent and are ignored.
constexpr const char* kSampler = "Euler";

HttpResponse Reply(int status, const json::Value& body) {
  return {.status = status,
          .reason = status == 200   ? "OK"
                    : status == 400 ? "Bad Request"
                    : status == 404 ? "Not Found"
                    : status == 405 ? "Method Not Allowed"
                    : status == 503 ? "Service Unavailable"
                                    : "Internal Server Error",
          .body = body.dump(),
          .headers = {{"Content-Type", "application/json"}},
          .streaming_body = {},
          .log_details = {}};
}

/// stable-diffusion.cpp's error shape, not A1111's `detail` or the OpenAI
/// envelope.
HttpResponse Error(int status, const std::string& message) {
  auto body = json::Value::object();
  body["error"] = message;
  return Reply(status, body);
}

template<typename T>
std::string Shortest(T value) {
  char text[32];
  return {text, std::to_chars(text, text + sizeof(text), value).ptr};
}

/// A float as the double its shortest text spells, so 0.8F reads 0.8.
double Decimal(float value) {
  const auto text = Shortest(value);
  double decimal = value;
  std::from_chars(text.data(), text.data() + text.size(), decimal);
  return decimal;
}

/// The field, or null when absent; A1111 clients send null for "default".
const json::Value* Field(const json::Value& body, const char* name) {
  const auto* value = body.find(name);
  return value && !value->is_null() ? value : nullptr;
}

std::string Text(const json::Value& body, const char* name) {
  const auto* value = Field(body, name);
  if (!value)
    return {};
  if (!value->is_string())
    throw std::invalid_argument(std::string(name) + " must be a string");
  return value->str();
}

std::optional<double> Real(const json::Value& body, const char* name) {
  const auto* value = Field(body, name);
  if (!value)
    return std::nullopt;
  if (!value->is_number() || !std::isfinite(value->as_double()))
    throw std::invalid_argument(std::string(name) + " must be a number");
  return value->as_double();
}

std::optional<std::int64_t> Integer(const json::Value& body, const char* name,
                                    std::int64_t low, std::int64_t high) {
  const auto value = Real(body, name);
  if (!value)
    return std::nullopt;
  if (std::floor(*value) != *value || *value < static_cast<double>(low) ||
      *value > static_cast<double>(high))
    throw std::invalid_argument(
        std::string(name) + " must be an integer from " + std::to_string(low) +
        " to " + std::to_string(high));
  return static_cast<std::int64_t>(*value);
}

/// Plain base64 or a base64 data URL; URLs are never fetched.
ext::ImageBuffer InitImage(const json::Value& value) {
  if (value.is_string()) {
    const auto& text = value.str();
    if (text.starts_with("data:"))
      return DecodeImage(core::ReadImageUrl(text));
    std::vector<std::uint8_t> bytes;
    if (core::DecodeBase64(text, {.max_decoded = core::kMaxEncodedImageBytes},
                           bytes) == core::Base64Status::kOk)
      return DecodeImage(bytes);
  }
  throw std::invalid_argument(
      "init_images entries must be base64 images or data URLs");
}

/// The init image's canvas on the 32-pixel grid, scaled down to fit the
/// 4096-pixel side and 4-megapixel limits.
std::pair<int, int> InitCanvas(int width, int height) {
  constexpr double kMaxPixels = 4194304, kMaxSide = 4096;
  const double scale = std::min({1.0, std::sqrt(kMaxPixels / width / height),
                                 kMaxSide / width, kMaxSide / height});
  const auto grid = [&](int side, bool down) {
    const double units = side * scale / 32;
    return 32 * std::max(1, static_cast<int>(down ? std::floor(units)
                                                  : std::nearbyint(units)));
  };
  int w = grid(width, false), h = grid(height, false);
  if (w > kMaxSide || h > kMaxSide || static_cast<double>(w) * h > kMaxPixels)
    w = grid(width, true), h = grid(height, true);
  return {w, h};
}

/// `lora`, as in stable-diffusion.cpp: an array (even empty) replaces the
/// server defaults, non-object entries are skipped and repeated paths sum
/// their multipliers. High-noise entries select the second stage of
/// two-stage models; single-stage models validate and skip them.
std::vector<ext::ImageLora> Loras(const json::Value& body,
                                  const ImageService& service) {
  const auto* value = body.find("lora");
  if (!value || !value->is_array())
    return service.loras().defaults;
  const auto listings = service.LoraListings();
  std::vector<std::pair<std::string, double>> sums;
  for (const auto& entry : value->items()) {
    if (!entry.is_object())
      continue;
    const auto path = Text(entry, "path");
    if (path.empty())
      throw std::invalid_argument("lora.path required");
    const auto listing =
        std::ranges::find(listings, path, &ImageLoraListing::path);
    if (listing == listings.end())
      throw std::invalid_argument("invalid lora path: " + path);
    const double multiplier = Real(entry, "multiplier").value_or(1);
    const auto* high_noise = Field(entry, "is_high_noise");
    if (high_noise && !high_noise->is_bool())
      throw std::invalid_argument("lora.is_high_noise must be a boolean");
    if (high_noise && high_noise->as_bool())
      continue;
    const auto& name = service.loras().files[listing - listings.begin()].first;
    const auto sum =
        std::ranges::find(sums, name, &std::pair<std::string, double>::first);
    if (sum == sums.end())
      sums.emplace_back(name, multiplier);
    else
      sum->second += multiplier;
  }
  std::vector<ext::ImageLora> loras;
  for (const auto& [name, multiplier] : sums) {
    // Converting an out-of-range double to float is undefined.
    if (!std::isfinite(multiplier) || std::abs(multiplier) > 10)
      throw std::invalid_argument(
          "lora multiplier must be finite and at most 10 in magnitude");
    loras.push_back({name, static_cast<float>(multiplier)});
  }
  return loras;
}

struct Generation {
  ext::ImageRequest request;
  std::size_t count{1};
};

/// stable-diffusion.cpp's `build_sdapi_img_gen_request`, with the image
/// service's defaults instead of A1111's processing defaults.
Generation Parse(const json::Value& body, const ImageService& service,
                 bool img2img) {
  Generation parsed;
  auto& generation = parsed.request;
  generation.prompt = Text(body, "prompt");
  if (generation.prompt.empty())
    throw std::invalid_argument("prompt required");
  generation.negative_prompt = Text(body, "negative_prompt");
  if (const auto scale = Real(body, "cfg_scale")) {
    if (*scale < 0 || *scale > 100)
      throw std::invalid_argument("cfg_scale must be from 0 to 100");
    generation.guidance_scale = static_cast<float>(*scale);
  }
  // Omitted steps keep zero so the runner applies its checkpoint default.
  generation.steps =
      static_cast<int>(Integer(body, "steps", 2, 100).value_or(0));
  // Any negative seed is random, as -1 is in A1111.
  constexpr auto kSeedBound = static_cast<std::int64_t>(kMaxImageSeed);
  const auto seed = Integer(body, "seed", -kSeedBound, kSeedBound).value_or(-1);
  generation.seed =
      seed < 0 ? RandomImageSeed() : static_cast<std::uint64_t>(seed);
  parsed.count = static_cast<std::size_t>(
      Integer(body, "batch_size", 1, kMaxBatchSize).value_or(1));
  std::optional<int> width, height;
  for (const auto& [name, side] :
       {std::pair{"width", &width}, {"height", &height}}) {
    const auto value = Real(body, name);
    if (value && *value <= 0)
      throw std::invalid_argument("width and height must be positive");
    if (value)
      *side = static_cast<int>(Integer(body, name, 1, 65536).value());
  }
  generation.loras = Loras(body, service);
  if (img2img) {
    const auto* images = Field(body, "init_images");
    if (!images || !images->is_array() || images->empty())
      throw std::invalid_argument("init_images required");
    if (Field(body, "mask"))
      throw std::invalid_argument("mask (inpainting) is not supported");
    // Only the first entry is used, as in stable-diffusion.cpp.
    generation.init_image = InitImage(images->items().front());
    // Omitted or negative values use sd.cpp's and A1111's 0.75; the
    // pipeline needs at least 0.01.
    const auto strength = Real(body, "denoising_strength");
    generation.strength =
        strength && *strength >= 0 ? std::clamp(*strength, 0.01, 1.0) : 0.75;
    const auto [w, h] =
        InitCanvas(generation.init_image->width, generation.init_image->height);
    // A single given side keeps the init image's aspect ratio.
    const auto other = [](int side, int from, int to) {
      return std::max(32, static_cast<int>(std::nearbyint(
                              side * static_cast<double>(to) / from / 32)) *
                              32);
    };
    const int iw = generation.init_image->width;
    const int ih = generation.init_image->height;
    generation.width = width.value_or(height ? other(*height, ih, iw) : w);
    generation.height = height.value_or(width ? other(*width, iw, ih) : h);
  } else {
    if (width)
      generation.width = *width;
    if (height)
      generation.height = *height;
  }
  return parsed;
}

/// stable-diffusion.cpp's `prepare_info_field`: an A1111 `info` object.
/// Values a request left to the model come from the runner's metrics, or are
/// null when it does not report them (`cfg_scale`; `steps` below full
/// strength).
std::string Info(const ext::ImageRequest& request, bool img2img,
                 const std::vector<GeneratedImage>& images,
                 const std::string& model) {
  const auto applied = request.guidance_scale
                           ? request.guidance_scale
                           : images.front().metrics.guidance_scale;
  json::Value cfg_scale = nullptr;
  if (applied)
    cfg_scale = Decimal(*applied);
  json::Value steps = nullptr;
  if (request.steps != 0)
    steps = request.steps;
  else if (request.strength == 1 && !images.empty())
    steps = images.front().metrics.steps;
  auto info = json::Value::object();
  info["prompt"] = request.prompt;
  info["negative_prompt"] = request.negative_prompt;
  info["seed"] = images.front().seed;
  info["cfg_scale"] = cfg_scale;
  info["width"] = request.width;
  info["height"] = request.height;
  info["steps"] = steps;
  info["sampler_name"] = kSampler;
  info["version"] = "gufo";
  if (img2img)
    info["denoising_strength"] = request.strength;
  auto infotexts = json::Value::array(), prompts = json::Value::array(),
       negatives = json::Value::array(), seeds = json::Value::array(),
       details = json::Value::array();
  for (const auto& image : images) {
    std::ostringstream text;
    text << request.prompt << '\n';
    if (!request.negative_prompt.empty())
      text << "Negative prompt: " << request.negative_prompt << '\n';
    if (!steps.is_null())
      text << "Steps: " << steps.as_double() << ", ";
    text << "Sampler: " << kSampler << ", ";
    if (!cfg_scale.is_null())
      text << "CFG scale: " << Shortest(*applied) << ", ";
    text << "Seed: " << image.seed << ", Size: " << request.width << 'x'
         << request.height << ", Model: " << model;
    if (img2img)
      text << ", Denoising strength: " << Shortest(request.strength);
    text << ", Version: gufo";
    infotexts.push_back(text.str());
    prompts.push_back(request.prompt);
    negatives.push_back(request.negative_prompt);
    seeds.push_back(image.seed);
    auto detail = json::Value::object();
    detail["seed"] = image.seed;
    detail["steps_run"] = image.metrics.steps;
    auto loras = json::Value::array();
    for (const auto& lora : request.loras) {
      auto item = json::Value::object();
      item["name"] = lora.name;
      item["multiplier"] = Decimal(lora.scale);
      loras.push_back(std::move(item));
    }
    detail["loras"] = std::move(loras);
    detail["total_ms"] = image.metrics.total_ms;
    details.push_back(std::move(detail));
  }
  info["infotexts"] = std::move(infotexts);
  info["all_prompts"] = std::move(prompts);
  info["all_negative_prompts"] = std::move(negatives);
  info["all_seeds"] = std::move(seeds);
  info["generation_info"] = std::move(details);
  return info.dump();
}

HttpResponse Generate(const HttpRequest& request, ImageService& service,
                      bool img2img) {
  if (request.body.empty())
    return Error(400, "empty body");
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& error) {
    return Error(400, std::string("invalid JSON: ") + error.what());
  }
  if (!body.is_object())
    return Error(400, "request body must be a JSON object");
  try {
    auto parsed = Parse(body, service, img2img);
    ValidateImageRequest(parsed.request);
    service.Validate(parsed.request);
    const auto images = GenerateImages(
        service, parsed.request, parsed.count, request.request_id,
        img2img ? "img2img" : "txt2img", request.is_cancelled);
    auto out = json::Value::object();
    auto encoded = json::Value::array();
    for (const auto& image : images)
      encoded.push_back(core::EncodeBase64(image.png));
    // Not a mistake: A1111 sends `info` as a JSON string.
    auto info = Info(parsed.request, img2img, images, service.model_id());
    out["images"] = std::move(encoded);
    out["parameters"] = std::move(body);
    out["info"] = std::move(info);
    return Reply(200, out);
  } catch (const std::invalid_argument& error) {
    return Error(400, error.what());
  } catch (const ext::ImageResourceError& error) {
    return Error(503, error.what());
  } catch (const std::exception& error) {
    return Error(500, error.what());
  }
}

json::Value Named(const char* name) {
  auto item = json::Value::object();
  item["name"] = name;
  return item;
}

HttpResponse Options(const HttpRequest& request, const ImageService& service) {
  if (request.method == "POST") {
    // Clients select the loaded checkpoint here; other settings are fixed.
    json::Value body;
    try {
      body = json::parse(request.body);
    } catch (const std::exception& error) {
      return Error(400, std::string("invalid JSON: ") + error.what());
    }
    if (!body.is_object())
      return Error(400, "request body must be a JSON object");
    if (const auto* model = Field(body, "sd_model_checkpoint");
        model && (!model->is_string() || !service.serves_model(model->str())))
      return Error(400, "sd_model_checkpoint must name the loaded model " +
                            service.model_id());
    return Reply(200, nullptr);
  }
  auto body = json::Value::object();
  body["samples_format"] = "png";
  body["sd_model_checkpoint"] = service.model_id();
  return Reply(200, body);
}

HttpResponse Progress(const ImageService& service) {
  // No previews or step progress: the job count shows whether generation is
  // running, so clients that wait for an idle server do not wait forever.
  auto state = json::Value::object();
  state["skipped"] = false;
  state["interrupted"] = false;
  state["job"] = "";
  state["job_count"] = service.pending();
  state["job_no"] = 0;
  state["job_timestamp"] = "0";
  state["sampling_step"] = 0;
  state["sampling_steps"] = 0;
  auto body = json::Value::object();
  body["progress"] = 0.0;
  body["eta_relative"] = 0.0;
  body["state"] = std::move(state);
  body["current_image"] = nullptr;
  body["textinfo"] = nullptr;
  return Reply(200, body);
}

}  // namespace

bool IsSdapiPath(std::string_view path) {
  return path.starts_with("/sdapi/v1/");
}

HttpResponse HandleSdapiRequest(const HttpRequest& request,
                                ImageService& service) {
  const std::string_view route =
      std::string_view(request.path)
          .substr(std::string_view("/sdapi/v1/").size());
  const bool get = request.method == "GET";
  const bool post = request.method == "POST";
  if (route == "txt2img" || route == "img2img")
    return post ? Generate(request, service, route == "img2img")
                : Error(405,
                        "/sdapi/v1/" + std::string(route) + " requires POST");
  if (route == "options")
    return get || post ? Options(request, service)
                       : Error(405, "/sdapi/v1/options requires GET or POST");
  if (route == "loras")
    return HandleImageLorasRequest(request, service);
  constexpr std::string_view kReadRoutes[] = {"sd-models",
                                              "samplers",
                                              "schedulers",
                                              "upscalers",
                                              "latent-upscale-modes",
                                              "progress"};
  if (std::ranges::find(kReadRoutes, route) == std::end(kReadRoutes))
    return Error(404, "no sdapi route for " + request.path);
  if (!get)
    return Error(405, request.path + " requires GET");
  if (route == "sd-models") {
    auto model = json::Value::object();
    model["title"] = service.model_id();
    model["model_name"] = service.model_id();
    model["filename"] = service.model_id();
    model["hash"] = nullptr;
    model["sha256"] = nullptr;
    model["config"] = nullptr;
    auto body = json::Value::array();
    body.push_back(std::move(model));
    return Reply(200, body);
  }
  if (route == "samplers") {
    auto sampler = Named(kSampler);
    auto aliases = json::Value::array();
    aliases.push_back("euler");
    aliases.push_back("k_euler");
    sampler["aliases"] = std::move(aliases);
    sampler["options"] = json::Value::object();
    auto body = json::Value::array();
    body.push_back(std::move(sampler));
    return Reply(200, body);
  }
  if (route == "schedulers") {
    auto scheduler = Named("default");
    scheduler["label"] = "Default";
    auto body = json::Value::array();
    body.push_back(std::move(scheduler));
    return Reply(200, body);
  }
  if (route == "upscalers") {
    // Highres fix is unsupported; "None" is the only choice.
    auto none = Named("None");
    none["model_name"] = nullptr;
    none["model_path"] = nullptr;
    none["model_url"] = nullptr;
    none["scale"] = 4;
    auto body = json::Value::array();
    body.push_back(std::move(none));
    return Reply(200, body);
  }
  if (route == "latent-upscale-modes")
    return Reply(200, json::Value::array());
  return Progress(service);
}

HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service) {
  if (request.method != "GET")
    return Error(405, "/sdapi/v1/loras requires GET");
  const auto active = service.ActiveLoras();
  const auto scale = [](const std::vector<ext::ImageLora>& loras,
                        const std::string& name) -> json::Value {
    for (const auto& lora : loras)
      if (lora.name == name)
        return Decimal(lora.scale);
    return nullptr;
  };
  auto body = json::Value::array();
  const auto listings = service.LoraListings();
  for (std::size_t i = 0; i < listings.size(); ++i) {
    const auto& name = service.loras().files[i].first;
    const auto& info = service.lora_info(name);
    auto metadata = json::Value::object();
    metadata["format"] = info.format;
    metadata["targets"] = info.targets;
    metadata["rank"] =
        std::to_string(info.min_rank) + '-' + std::to_string(info.max_rank);
    auto item = json::Value::object();
    item["name"] = name;
    item["alias"] = name;
    item["path"] = listings[i].path;
    item["display_path"] = listings[i].display_path;
    item["metadata"] = std::move(metadata);
    item["default_scale"] = scale(service.loras().defaults, name);
    item["active_scale"] = scale(active, name);
    body.push_back(std::move(item));
  }
  return Reply(200, body);
}

}  // namespace gufo::server
