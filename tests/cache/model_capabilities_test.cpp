#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <string_view>

#include "src/models/deepseek_v4_flash/cache_capabilities.hpp"
#include "src/models/minimax_h3/cache_capabilities.hpp"
#include "src/models/qwen/cache_capabilities.hpp"
#include "src/models/qwen38_flash_next/cache_capabilities.hpp"
#include "src/models/qwen3_asr/cache_capabilities.hpp"
#include "src/models/qwen3_tts/cache_capabilities.hpp"
#include "src/models/qwen_image_21/cache_capabilities.hpp"

namespace {

struct Record {
  std::string_view family;
  gufo::cache::Capabilities capabilities;
};

// One record per src/models/ directory, keyed by the directory name.
constexpr std::array kRecords{
    Record{"deepseek_v4_flash",
           gufo::models::deepseek_v4_flash::kCacheCapabilities},
    Record{"minimax_h3", gufo::minimax_h3::kCacheCapabilities},
    Record{"qwen", gufo::models::qwen::kCacheCapabilities},
    Record{"qwen38_flash_next",
           gufo::models::qwen38_flash_next::kCacheCapabilities},
    Record{"qwen3_asr", gufo::models::qwen3_asr::kCacheCapabilities},
    Record{"qwen3_tts", gufo::models::qwen3_tts::kCacheCapabilities},
    Record{"qwen_image_21", gufo::models::qwen_image_21::kCacheCapabilities},
};

const Record* Find(std::string_view family) {
  for (const auto& record : kRecords)
    if (record.family == family)
      return &record;
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path models = argv[1];
  assert(std::filesystem::is_directory(models));

  std::set<std::string> families;
  for (const auto& entry : std::filesystem::directory_iterator(models))
    if (entry.is_directory())
      families.insert(entry.path().filename().string());

  bool complete = true;
  for (const auto& family : families) {
    if (Find(family) == nullptr) {
      std::cerr << "src/models/" << family
                << " has no cache capability record\n";
      complete = false;
    }
  }
  std::set<std::string_view> recorded;
  for (const auto& record : kRecords) {
    if (!recorded.insert(record.family).second ||
        !families.contains(std::string(record.family))) {
      std::cerr << record.family
                << " is duplicated or has no model directory\n";
      complete = false;
    }
    // Persistent encoding describes continuation state; it cannot exist alone.
    assert(record.capabilities.continuation ||
           !record.capabilities.persistent_encoding);
  }
  assert(complete);

  for (const auto family :
       {"qwen3_asr", "qwen3_tts", "qwen_image_21", "minimax_h3"})
    assert(!Find(family)->capabilities.continuation);
  for (const auto family : {"qwen", "qwen38_flash_next", "deepseek_v4_flash"})
    assert(Find(family)->capabilities.continuation);
}
