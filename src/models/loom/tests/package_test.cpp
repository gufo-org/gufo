// Tests for loom_package.hpp: preset choice per GPU and parameter files found
// by origin (the layout `gufo model install` writes). No GPU, no Loom.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/platform/os_info.hpp>
#include <iostream>
#include <string>

#include "../loom_package.hpp"

using namespace gufo;
using namespace gufo::loom;
namespace fs = std::filesystem;

namespace {

int failures = 0;
#define CHECK(cond, what)                                       \
  do {                                                          \
    if (!(cond)) {                                              \
      std::cerr << "FAIL " << __LINE__ << ": " << what << "\n"; \
      ++failures;                                               \
    }                                                           \
  } while (0)

void Write(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

const char* kCard = R"({
  "schema_version": 1,
  "model": {"id": "Test/Model"},
  "programs": {"source": {"inputs": ["program/service.loom"]}},
  "services": {"chat": {"kind": "text", "version": 1}},
  "variants": {"gguf": {"services": {"chat": {"programs": ["source"],
    "parameters": {
      "model": {"files": [{"path": "parameter/model-Q4_K_M.gguf", "format": "gguf",
                           "origin": "@checkpoint/sub/model-Q4_K_M.gguf"}]},
      "draft": {"files": [{"path": "parameter/draft.gguf", "format": "gguf",
                           "origin": "@drafter/draft.gguf"}]}}}}}},
  "presets": {
    "chat": {"service": "chat", "variant": "gguf"},
    "chat-r9700": {"service": "chat", "variant": "gguf"},
    "chat-16gb": {"service": "chat", "variant": "gguf"}},
  "origins": {"checkpoint": {"kind": "hf", "repository": "a/b", "revision": "x"},
              "drafter": {"kind": "hf", "repository": "c/d", "revision": "y"}}
})";

std::string Preset(const json::Value& card, unsigned target, std::uint64_t gib,
                   const std::string& requested = "") {
  return package::ChoosePreset(card, requested, {target, gib << 30}, nullptr);
}

void TestPresets() {
  const auto card = json::parse(kCard);
  CHECK(Preset(card, 110501, 128) == "chat", "Strix Halo serves chat");
  CHECK(Preset(card, 120001, 32) == "chat-r9700", "R9700 serves chat-r9700");
  CHECK(Preset(card, 120001, 16) == "chat-16gb",
        "16 GB gfx1201 serves chat-16gb");
  CHECK(Preset(card, 120001, 16, "chat") == "chat", "--loom-preset wins");
  CHECK(Preset(card, 0, 0) == "chat", "no KFD: chat");
  const auto strix_only = json::parse(
      R"({"presets": {"chat": {"variant": "gguf"}, "chat-gfx1150": {"variant": "gguf"}}})");
  CHECK(Preset(strix_only, 110500, 64) == "chat-gfx1150", "gfx1150 preset");
  bool threw = false;
  try {
    Preset(strix_only, 120001, 32);
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("no preset for gfx1201") !=
            std::string::npos;
  }
  CHECK(threw, "a package without a gfx1201 preset says so");
  threw = false;
  try {
    Preset(card, 120000, 16);
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("no preset for gfx1200") !=
            std::string::npos;
  }
  CHECK(threw, "gfx1200 gets no gfx1201 preset");
  threw = false;
  try {
    package::ChoosePreset(card, "", {.named_device = true}, nullptr);
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("--loom-device names a device") !=
            std::string::npos;
  }
  CHECK(threw, "a named --loom-device needs --loom-preset");
  CHECK(package::ChoosePreset(card, "chat-r9700", {.named_device = true},
                              nullptr) == "chat-r9700",
        "a named device with --loom-preset");
  CHECK(!package::NamesDevice("amdgpu://?completion_spin_ns=1") &&
            !package::NamesDevice(""),
        "Loom's default device");
  CHECK(package::NamesDevice("amdgpu://1") &&
            package::NamesDevice("amdgpu://1?x=2"),
        "a named device");
  const auto single =
      json::parse(R"({"presets": {"chat": {"variant": "gguf"}}})");
  CHECK(Preset(single, 120001, 32).empty(), "one preset: Loom picks it");
  CHECK(package::GpuName(110501) == "gfx1151", "gfx1151 name");
  CHECK(package::GpuName(120001) == "gfx1201", "gfx1201 name");
  CHECK(package::GpuName(90010) == "gfx90a", "gfx90a name");
}

fs::path TempDir() {
  static int count = 0;
  const auto path = fs::temp_directory_path() /
                    ("loom_package_test_" + std::to_string(++count));
  fs::create_directories(path);
  return path;
}

void TestInstalledLayout() {
  // The repository files, with each origin's files at their repository paths.
  const auto root = TempDir();
  Write(root / "model.json", kCard);
  Write(root / "program" / "service.loom", "program");
  Write(root / "tokenizer.json", "{}");
  Write(root / "sub" / "model-Q4_K_M.gguf", "weights");
  Write(root / "draft.gguf", "draft");
  fs::path overlay;
  {
    const auto resolved =
        package::Resolve(root.string(), "", {}, {110501, 128ull << 30});
    CHECK(resolved.preset == "chat", "preset chosen");
    overlay = resolved.overlay;
    CHECK(!overlay.empty() && resolved.load_path == overlay.string(),
          "links directory used");
    std::ifstream weights(overlay / "parameter" / "model-Q4_K_M.gguf");
    std::string text;
    weights >> text;
    CHECK(text == "weights", "checkpoint linked at its card path");
    CHECK(fs::exists(overlay / "parameter" / "draft.gguf"), "drafter linked");
    CHECK(fs::exists(overlay / "program" / "service.loom"), "programs linked");
    CHECK(fs::exists(overlay / "model.json"), "card linked");
  }
  CHECK(!fs::exists(overlay), "links directory removed with the value");
  fs::remove_all(root);
}

void TestFetchedLayout() {
  // loom-fetch output: the files are at their card paths; nothing to link.
  const auto root = TempDir();
  Write(root / "model.json", kCard);
  Write(root / "parameter" / "model-Q4_K_M.gguf", "weights");
  Write(root / "parameter" / "draft.gguf", "draft");
  const auto resolved = package::Resolve(root.string(), "chat", {}, {});
  CHECK(resolved.overlay.empty() && resolved.load_path == root.string(),
        "loom-fetch layout opens in place");
  fs::remove_all(root);
}

void TestOriginOptions() {
  const auto root = TempDir();
  const auto elsewhere = TempDir();
  Write(root / "model.json", kCard);
  Write(elsewhere / "sub" / "model-Q4_K_M.gguf", "weights");
  Write(elsewhere / "draft.gguf", "draft");
  const auto resolved = package::Resolve(
      root.string(), "",
      {{"checkpoint", elsewhere}, {"drafter", elsewhere / "draft.gguf"}},
      {110501, 128ull << 30});
  CHECK(fs::equivalent(resolved.overlay / "parameter" / "model-Q4_K_M.gguf",
                       elsewhere / "sub" / "model-Q4_K_M.gguf"),
        "--loom-checkpoint directory");
  CHECK(fs::equivalent(resolved.overlay / "parameter" / "draft.gguf",
                       elsewhere / "draft.gguf"),
        "--loom-drafter file");
  fs::remove_all(root);
  fs::remove_all(elsewhere);
}

void TestMissing() {
  const auto root = TempDir();
  Write(root / "model.json", kCard);
  Write(root / "sub" / "model-Q4_K_M.gguf", "weights");
  std::string message;
  try {
    package::Resolve(root.string(), "", {}, {110501, 128ull << 30});
  } catch (const std::exception& error) {
    message = error.what();
  }
  CHECK(message.find("parameter/draft.gguf (@drafter/draft.gguf)") !=
            std::string::npos,
        "missing files are named with their origin");
  CHECK(message.find("gufo model install") != std::string::npos,
        "the error says how to fix it");
  std::size_t leftovers = 0;
  for (const auto& entry : fs::directory_iterator(fs::temp_directory_path()))
    leftovers +=
        entry.path().filename().string().starts_with("gufo_loom_") ? 1 : 0;
  CHECK(leftovers == 0, "no links directory is left behind");
  fs::remove_all(root);
}

void TestStaleAndLayouts() {
  // A private directory whose owner lock nobody holds (its process ended) is
  // removed; one that is still locked is kept.
  const auto stale = fs::temp_directory_path() / "gufo_loom_999999999_abcdef";
  Write(stale / "model.json", "{}");
  Write(stale / ".owner", "");
  const auto live = fs::temp_directory_path() / "gufo_loom_999999998_abcdef";
  Write(live / ".owner", "");
  const auto live_owner = package::LockOwner(live);
  CHECK(platform::FileValid(live_owner), "a free owner lock can be taken");
  const auto root = TempDir();
  Write(root / "model.json", kCard);
  Write(root / "sub" / "model-Q4_K_M.gguf", "weights");
  Write(root / "draft.gguf", "draft");
  // The package's own parameter/ directory, reached through a symbolic link.
  const auto shared = TempDir();
  Write(shared / "extra.bin", "x");
  std::error_code symlink_error;
  fs::create_directory_symlink(shared, root / "parameter", symlink_error);
  if (symlink_error) {
    // Windows without Developer Mode: the package's own directory is real.
    fs::create_directories(root / "parameter");
    Write(root / "parameter" / "extra.bin", "x");
  }
  {
    const auto resolved =
        package::Resolve(root.string(), "", {}, {110501, 128ull << 30});
    CHECK(!fs::exists(stale), "a dead process's directory is removed");
    CHECK(fs::exists(live), "a live process's directory is kept");
    CHECK(fs::exists(resolved.overlay / "parameter" / "extra.bin"),
          "a linked directory's files are kept");
    CHECK(fs::exists(resolved.overlay / "parameter" / "model-Q4_K_M.gguf"),
          "and the found files added");
    CHECK(!fs::is_symlink(resolved.overlay / "parameter"),
          "in a real directory");
  }
  platform::CloseFile(live_owner);
  fs::remove_all(live);
  fs::remove_all(root / "parameter");
  Write(root / "parameter", "a file");
  bool threw = false;
  try {
    package::Resolve(root.string(), "", {}, {110501, 128ull << 30});
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("expects a directory") !=
            std::string::npos;
  }
  CHECK(threw, "a file where the card expects a directory is an error");
  fs::remove_all(root);
  fs::remove_all(shared);
}

void TestSdk() {
  const auto root = TempDir();
  Write(root / "model.json", kCard);
  Write(root / "sub" / "model-Q4_K_M.gguf", "weights");
  Write(root / "draft.gguf", "draft");
  auto refused = [&](std::string_view runtime) {
    try {
      package::Resolve(root.string(), "chat", {}, {}, runtime);
    } catch (const std::exception& error) {
      return std::string(error.what()).find("built for Loom SDK") !=
             std::string::npos;
    }
    return false;
  };
  CHECK(!refused("0.2.0"), "no llm_config.json: nothing to check");
  Write(root / "llm_config.json", R"({"chat": {}})");
  CHECK(package::PackageSdk(root) == "0.1.0", "no sdk field: a 0.1.0 package");
  CHECK(refused("0.2.0"), "a 0.1.0 package on the 0.2.0 runtime is refused");
  Write(root / "llm_config.json",
        R"({"sdk": {"hrx_loom": "0.2.0", "source": "pypi"}})");
  CHECK(!refused("0.2.0") && !refused("0.2.1"), "same series serves");
  CHECK(refused("0.3.0"), "a newer runtime refuses an older package");
  CHECK(!refused(""), "no runtime version: nothing to check");
  CHECK(package::SdkSeries("0.2.0") == "0.2" &&
            package::SdkSeries("1.10.3") == "1.10",
        "series");
  fs::remove_all(root);
}

void TestUnsafePaths() {
  CHECK(package::SafeRelative("parameter/a.gguf"), "plain path");
  CHECK(!package::SafeRelative("../a.gguf"), "parent");
  CHECK(!package::SafeRelative("parameter/../../a"), "parent inside");
  CHECK(!package::SafeRelative("/etc/passwd"), "absolute");
  CHECK(!package::SafeRelative("a//b"), "empty part");
  const auto root = TempDir();
  std::string card = kCard;
  card.replace(card.find("parameter/draft.gguf"), 20, "../escaped.gguf");
  Write(root / "model.json", card);
  Write(root / "sub" / "model-Q4_K_M.gguf", "weights");
  Write(root / "draft.gguf", "draft");
  bool threw = false;
  try {
    package::Resolve(root.string(), "", {}, {110501, 128ull << 30});
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("invalid parameter path") !=
            std::string::npos;
  }
  CHECK(threw, "a card path outside the package is refused");
  fs::remove_all(root);
}

}  // namespace

int main() {
  // CTest runs this in a private temporary folder (TMPDIR, TMP): the
  // leftovers check counts this run's directories only.
  TestPresets();
  TestInstalledLayout();
  TestFetchedLayout();
  TestOriginOptions();
  TestMissing();
  TestUnsafePaths();
  TestSdk();
  TestStaleAndLayouts();
  if (failures) {
    std::cerr << failures << " failed\n";
    return 1;
  }
  std::cout << "package: all passed\n";
  return 0;
}
