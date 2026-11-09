// Image provider startup options and request capability validation (#974).
// Host-only: providers are fakes registered in this process, no weights load.
#include "src/cli/serve/image_api.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/provider_options.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"

namespace {
using namespace gufo;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error("check failed: " + std::string(message));
}

template<class Function>
void Reject(Function function, std::string_view text) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    Require(std::string_view(error.what()).find(text) != std::string_view::npos,
            error.what());
    return;
  }
  throw std::runtime_error("expected rejection: " + std::string(text));
}

template<class Function>
void Fail(Function function, std::string_view text) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error("expected a runtime failure, got usage error: " +
                             std::string(error.what()));
  } catch (const std::runtime_error& error) {
    Require(std::string_view(error.what()).find(text) != std::string_view::npos,
            error.what());
    return;
  }
  throw std::runtime_error("expected failure: " + std::string(text));
}

ext::ParsedOptions Parse(const ext::Registry& registry,
                         std::initializer_list<std::string_view> arguments) {
  return ext::ParseOptions(registry, ext::kImageServeCommand,
                           {arguments.begin(), arguments.size()});
}

ext::ImageBuffer Pixel() {
  return {.width = 1, .height = 1, .rgba = {200, 100, 0, 0}};
}

// The factory records what it was handed so the test can prove forwarding.
ext::ImageRunnerOptions g_received;
int g_created = 0;

class RestrictedRunner final : public ext::ImageRunner {
public:
  ext::ImageResult Generate(const ext::ImageRequest&,
                            const ext::CancellationCheck&) override {
    return {.image = Pixel(), .metrics = {}};
  }
  void ValidateRequest(const ext::ImageRequest& request) const override {
    if (request.width > 512)
      throw std::invalid_argument("restricted image is at most 512 wide");
  }
};

std::unique_ptr<ext::ImageRunner> CreateRestricted(
    const ext::ImageRunnerOptions& options, std::string*) {
  g_received = options;
  ++g_created;
  return std::make_unique<RestrictedRunner>();
}

std::unique_ptr<ext::ImageRunner> CreateLegacy(const std::string&,
                                               std::string*) {
  ++g_created;
  return std::make_unique<RestrictedRunner>();
}

void RegisterProviders(ext::Registry& registry) {
  registry.BeginExtension({"image_fixture", "1.0.0", "test"});
  registry.AddProvider({
      .id = "restricted-image",
      .capability = ext::Capability::kImageGeneration,
      .keys = {},
      .model_ids = {},
      .speculative = {},
      .options = {{"--text-encoder",
                   ext::OptionType::kPath,
                   {"serve image"},
                   "Text encoder GGUF"},
                  {"--encoder-threads",
                   ext::OptionType::kInteger,
                   {"serve image"},
                   ""}},
      .available = true,
      .unavailable_reason = {},
      .extension = {},
  });
  // A bare `serve` declaration reaches every serve modality, as in Nix.
  registry.AddProvider({
      .id = "legacy-image",
      .capability = ext::Capability::kImageGeneration,
      .keys = {},
      .model_ids = {},
      .speculative = {},
      .options = {{"--legacy-flag", ext::OptionType::kBoolean, {"serve"}, ""}},
      .available = true,
      .unavailable_reason = {},
      .extension = {},
  });
  registry.AddProvider({
      .id = "probe-only-image",
      .capability = ext::Capability::kImageGeneration,
      .keys = {},
      .model_ids = {},
      .speculative = {},
      .options = {},
      .available = true,
      .unavailable_reason = {},
      .extension = {},
  });
  registry.AddProvider({
      .id = "text-only",
      .capability = ext::Capability::kTextGeneration,
      .keys = {},
      .model_ids = {},
      .speculative = {},
      .options = {{"--llm-draft", ext::OptionType::kPath, {"serve llm"}, ""}},
      .available = true,
      .unavailable_reason = {},
      .extension = {},
  });
  registry.AddImageProvider({
      .id = "restricted-image",
      .probe = [](std::string_view root) { return root == "restricted"; },
      .resolve = nullptr,
      .create = nullptr,
      .create_with_options = &CreateRestricted,
  });
  registry.AddImageProvider({
      .id = "legacy-image",
      .probe = [](std::string_view root) { return root == "legacy"; },
      .resolve = nullptr,
      .create = &CreateLegacy,
      .create_with_options = nullptr,
  });
  // CPU-only builds register probes without an execution factory.
  registry.AddImageProvider({
      .id = "probe-only-image",
      .probe = [](std::string_view root) { return root == "probe-only"; },
      .resolve = nullptr,
      .create = nullptr,
      .create_with_options = nullptr,
  });
}

void TestCommandParsing(const ext::Registry& registry,
                        const std::string& encoder_arg) {
  const auto parse = [&](std::vector<const char*> args) {
    cli::ArgParser parser("gufo serve image");
    std::string model;
    std::string served_name;
    parser.AddOption("-m", "--model", "DIR", "", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "", "Model",
                     &served_name);
    cli::ProviderOptions options(registry, std::string(ext::kImageServeCommand),
                                 ext::Capability::kImageGeneration);
    options.Register(parser);
    Require(parser.FindOption("--text-encoder") != nullptr,
            "image options registered");
    Require(parser.FindOption("--legacy-flag") != nullptr,
            "bare serve registered");
    Require(parser.FindOption("--llm-draft") == nullptr,
            "text options excluded");
    std::string error;
    if (!parser.Parse(args, &error))
      throw std::invalid_argument(error);
    auto values = options.Parse();
    if (!served_name.empty())
      Require(served_name == "--text-encoder", "host value preserved");
    return values;
  };
  auto values = parse(
      {"--model", "restricted", encoder_arg.c_str(), "--encoder-threads", "4"});
  Require(values.size() == 2 &&
              std::get<std::int64_t>(values.at("--encoder-threads")) == 4,
          "shared parser forwards typed options");
  values = parse({"--served-model-name", "--text-encoder"});
  Require(values.empty(), "host value is not a provider option");
  values = parse({"--legacy-flag=false"});
  Require(!std::get<bool>(values.at("--legacy-flag")),
          "inline false preserved");
  Reject([&] { parse({"--text-encoder"}); }, "Missing argument");
  Reject([&] { parse({"--encoder-threads=four"}); }, "invalid value");
  Reject([&] { parse({"--encoder-threads=1", "--encoder-threads=2"}); },
         "duplicate");
}

void TestStartupOptions(const std::filesystem::path& encoder) {
  ext::Registry registry;
  RegisterProviders(registry);
  const auto encoder_arg = "--text-encoder=" + encoder.string();
  TestCommandParsing(registry, encoder_arg);

  // Forwarding: typed, explicit-only options reach the selected provider.
  auto options = Parse(registry, {encoder_arg, "--encoder-threads", "4"});
  const auto& restricted =
      server::SelectImageProvider(registry, "restricted", options);
  Require(restricted.id == "restricted-image", "probe selects provider");
  const auto runner =
      server::CreateImageRunner(restricted, "restricted", options);
  Require(runner != nullptr && g_created == 1, "factory called once");
  Require(g_received.model_root == "restricted", "model root forwarded");
  Require(g_received.options.size() == 2, "only supplied options forwarded");
  Require(std::get<std::string>(g_received.options.at("--text-encoder")) ==
              encoder.string(),
          "path option forwarded");
  Require(
      std::get<std::int64_t>(g_received.options.at("--encoder-threads")) == 4,
      "integer option forwarded");
  // No options: the provider keeps its defaults (for example a sidecar beside
  // the model) and receives an empty map.
  (void)server::CreateImageRunner(
      server::SelectImageProvider(registry, "restricted", {}), "restricted",
      {});
  Require(g_received.options.empty() && g_created == 2, "defaults untouched");

  // Type errors and unknown spellings fail in the shared parser.
  Reject([&] { Parse(registry, {"--encoder-threads=four"}); }, "invalid value");
  Reject([&] { Parse(registry, {"--text-encoder"}); }, "missing value");
  Reject([&] { Parse(registry, {"--llm-draft=x"}); }, "unknown option");
  Reject([&] { Parse(registry, {"--made-up=1"}); }, "unknown option");

  // Options of another provider are rejected once the model selects one.
  Reject([&] { server::SelectImageProvider(registry, "legacy", options); },
         "provider legacy-image does not support --");
  const auto flag = Parse(registry, {"--legacy-flag"});
  Require(std::get<bool>(flag.at("--legacy-flag")), "bare serve applies");
  Reject([&] { server::SelectImageProvider(registry, "restricted", flag); },
         "provider restricted-image does not support --legacy-flag");

  // Missing sidecars fail before the factory runs.
  const auto created = g_created;
  const auto missing = Parse(
      registry,
      {"--text-encoder=" + (encoder.parent_path() / "absent.gguf").string()});
  Reject([&] { server::SelectImageProvider(registry, "restricted", missing); },
         "--text-encoder does not exist");
  Require(g_created == created, "no runner for a missing sidecar");

  // A provider that declares options must accept them in its factory.
  Fail(
      [&] {
        server::CreateImageRunner(
            server::SelectImageProvider(registry, "legacy", flag), "legacy",
            flag);
      },
      "no options-aware factory");
  // CPU-only probe behaviour: selection still works, creation explains why
  // it cannot run instead of calling a null factory.
  const auto& probe_only =
      server::SelectImageProvider(registry, "probe-only", {});
  Require(probe_only.id == "probe-only-image", "probe-only selection");
  Fail([&] { server::CreateImageRunner(probe_only, "probe-only", {}); },
       "cannot run in this build");

  // `serve` and modality-specific declarations overlap for conflict checks.
  ext::Registry conflicting;
  conflicting.BeginExtension({"conflict", "1.0.0", "test"});
  conflicting.AddProvider(
      {.id = "a",
       .capability = ext::Capability::kImageGeneration,
       .keys = {},
       .model_ids = {},
       .speculative = {},
       .options = {{"--shared", ext::OptionType::kPath, {"serve image"}, ""}},
       .available = true,
       .unavailable_reason = {},
       .extension = {}});
  Reject(
      [&] {
        conflicting.AddProvider(
            {.id = "b",
             .capability = ext::Capability::kTextGeneration,
             .keys = {},
             .model_ids = {},
             .speculative = {},
             .options =
                 {{"--shared", ext::OptionType::kInteger, {"serve"}, ""}},
             .available = true,
             .unavailable_reason = {},
             .extension = {}});
      },
      "conflicting option types");
  Require(ext::OptionApplies({"--x", ext::OptionType::kPath, {"serve"}, ""},
                             "serve image"),
          "bare serve applies to serve image");
  Require(!ext::OptionApplies(
              {"--x", ext::OptionType::kPath, {"serve image"}, ""}, "serve"),
          "modality option does not apply to bare serve");
  Require(!ext::OptionApplies({"--x", ext::OptionType::kPath, {"serve"}, ""},
                              "prompt"),
          "serve does not apply to prompt");
}

server::HttpRequest Generation(std::string_view body) {
  server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/generations";
  http.headers = {{"Content-Type", "application/json"}};
  http.body = std::string(body);
  return http;
}

server::HttpRequest Edit(std::string_view background) {
  server::HttpRequest http;
  http.method = "POST";
  http.path = "/v1/images/edits";
  http.headers = {{"Content-Type", "multipart/form-data; boundary=b"}};
  // 1x1 RGBA PNG, chunks and CRCs generated with zlib.
  static const std::array<unsigned char, 70> png{
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
      0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x38, 0x91, 0xc2, 0xd0,
      0x00, 0x00, 0x04, 0xd1, 0x01, 0xad, 0x85, 0x3c, 0x2f, 0x4d, 0x00, 0x00,
      0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
  http.body =
      "--b\r\nContent-Disposition: form-data; name=\"image\"\r\n"
      "Content-Type: image/png\r\n\r\n";
  http.body.append(reinterpret_cast<const char*>(png.data()), png.size());
  for (const auto& [name, value] :
       {std::pair<std::string_view, std::string_view>{"prompt", "cube"},
        {"size", "256x256"},
        {"background", background}})
    http.body += "\r\n--b\r\nContent-Disposition: form-data; name=\"" +
                 std::string(name) + "\"\r\n\r\n" + std::string(value);
  http.body += "\r\n--b--\r\n";
  return http;
}

std::string Message(const server::HttpResponse& response) {
  return json::parse(response.body).find("error")->find("message")->str();
}

class LegacyRunner final : public ext::ImageRunner {
public:
  ext::ImageRequest seen;
  ext::ImageResult Generate(const ext::ImageRequest& request,
                            const ext::CancellationCheck&) override {
    seen = request;
    return {.image = Pixel(), .metrics = {}};
  }
};

void TestLegacyRequests() {
  const auto runner = std::make_shared<LegacyRunner>();
  server::ImageService service(runner, "legacy");
  const auto check = [&](const server::HttpRequest& request, std::size_t images,
                         std::string_view prompt) {
    const auto response = server::HandleImageApiRequest(request, service);
    Require(response.status == 200, response.body);
    Require(
        runner->seen.prompt == prompt && runner->seen.images.size() == images,
        "legacy prompt and edits preserved");
  };
  check(Generation(R"({"prompt":"cube","size":"256x256"})"), 0, "cube");
  check(
      Generation(R"({"prompt":"cube","size":"256x256","background":"opaque"})"),
      0, "cube");
  check(Edit("auto"), 1, "cube");
  constexpr std::string_view framed =
      "This is an RGBA image with transparency. cube"
      " The image has alpha channel and the background is transparent.";
  check(Generation(
            R"({"prompt":"cube","size":"256x256","background":"transparent"})"),
        0, framed);
  check(Edit("transparent"), 1, framed);
}

void TestRequestValidation() {
  std::vector<ext::ImageRequest> seen;
  const auto record = [&](const ext::ImageRequest& request,
                          const ext::CancellationCheck&) {
    seen.push_back(request);
    return ext::ImageResult{.image = Pixel(), .metrics = {}};
  };

  // A runner that advertises nothing: text-to-image with auto/opaque only.
  server::ImageService narrow(std::make_shared<server::CallbackImageRunner>(
                                  record, ext::ImageCapabilities{}),
                              "narrow");
  for (const auto& [background, expected] :
       {std::pair<std::string_view, ext::ImageBackground>{
            "", ext::ImageBackground::kAuto},
        {R"(,"background":"auto")", ext::ImageBackground::kAuto},
        {R"(,"background":"opaque")", ext::ImageBackground::kOpaque}}) {
    const auto response = server::HandleImageApiRequest(
        Generation(R"({"prompt":"a cube","size":"256x256")" +
                   std::string(background) + "}"),
        narrow);
    Require(response.status == 200, response.body);
    Require(seen.back().background == expected, "background forwarded");
    Require(seen.back().prompt == "a cube", "prompt reaches provider as sent");
  }
  const auto accepted = seen.size();
  auto response = server::HandleImageApiRequest(
      Generation(
          R"({"prompt":"a cube","size":"256x256","background":"transparent"})"),
      narrow);
  Require(response.status == 400, "transparent rejected");
  Require(Message(response).find("does not support transparent") !=
              std::string::npos,
          Message(response));
  response = server::HandleImageApiRequest(
      Generation(R"({"prompt":"a cube","background":"blue"})"), narrow);
  Require(response.status == 400 &&
              Message(response).find("auto, opaque or transparent") !=
                  std::string::npos,
          "invalid background");
  response = server::HandleImageApiRequest(Edit("auto"), narrow);
  Require(response.status == 400, "edit rejected");
  Require(Message(response).find("does not support image edits") !=
              std::string::npos,
          Message(response));
  Require(seen.size() == accepted, "no inference for rejected requests");

  // Provider-specific validation runs before inference as well.
  server::ImageService restricted(std::make_shared<RestrictedRunner>(),
                                  "restricted");
  response = server::HandleImageApiRequest(
      Generation(R"({"prompt":"a cube","size":"1024x1024"})"), restricted);
  Require(response.status == 400 &&
              Message(response).find("at most 512 wide") != std::string::npos,
          "provider validation");
  response = server::HandleImageApiRequest(
      Generation(R"({"prompt":"a cube","size":"512x512"})"), restricted);
  Require(response.status == 200, "provider accepts valid request");

  // A runner that advertises both keeps transparent and edit requests,
  // with the background intent intact for its own conditioning.
  server::ImageService full(
      std::make_shared<server::CallbackImageRunner>(
          record, ext::ImageCapabilities{.edits = true,
                                         .transparent_background = true}),
      "full");
  response = server::HandleImageApiRequest(
      Generation(
          R"({"prompt":"a cube","size":"256x256","background":"transparent"})"),
      full);
  Require(response.status == 200, response.body);
  Require(seen.back().background == ext::ImageBackground::kTransparent &&
              seen.back().prompt == "a cube",
          "transparent intent forwarded unchanged");
  response = server::HandleImageApiRequest(Edit("transparent"), full);
  Require(response.status == 200, response.body);
  Require(seen.back().images.size() == 1 &&
              seen.back().background == ext::ImageBackground::kTransparent,
          "edit forwarded");
}

}  // namespace

int main() {
  try {
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("gufo-image-api-test-" + std::to_string(platform::ProcessId()));
    std::filesystem::create_directories(directory);
    const auto encoder = directory / "encoder.gguf";
    std::ofstream(encoder) << "GGUF";
    TestStartupOptions(encoder);
    TestRequestValidation();
    TestLegacyRequests();
    std::filesystem::remove_all(directory);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "image API provider contracts passed\n";
  return 0;
}
