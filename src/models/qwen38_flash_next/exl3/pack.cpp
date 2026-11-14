#include "src/models/qwen38_flash_next/exl3/pack.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <gufo/ext/sha256.hpp>
#include <sstream>
#include <system_error>

namespace gufo::models::qwen38_flash_next::exl3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return {};
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

bool ParseDType(const std::string& name, DType* out, std::size_t* bytes) {
  static constexpr struct {
    std::string_view name;
    DType dtype;
    std::size_t bytes;
  } kTypes[] = {{"F16", DType::kF16, 2}, {"BF16", DType::kBF16, 2},
                {"F32", DType::kF32, 4}, {"I16", DType::kI16, 2},
                {"I32", DType::kI32, 4}, {"I64", DType::kI64, 8}};
  for (const auto& t : kTypes) {
    if (t.name == name) {
      *out = t.dtype;
      *bytes = t.bytes;
      return true;
    }
  }
  return false;
}

}  // namespace

std::unique_ptr<Pack> Pack::Open(const std::filesystem::path& directory,
                                 std::string* error) {
  std::unique_ptr<Pack> pack(new Pack());
  pack->directory_ = directory;
  const std::string config_text = ReadFile(directory / "config.json");
  if (config_text.empty()) {
    SetError(error,
             "EXL3 checkpoint has no config.json: " + directory.string());
    return nullptr;
  }
  try {
    pack->config_ = json::parse(config_text);
  } catch (const std::exception& e) {
    SetError(error, std::string("invalid config.json: ") + e.what());
    return nullptr;
  }
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    if (entry.is_regular_file(ec) && entry.path().extension() == ".safetensors")
      files.push_back(entry.path());
  }
  if (ec || files.empty()) {
    SetError(error,
             "EXL3 checkpoint has no safetensors files: " + directory.string());
    return nullptr;
  }
  std::sort(files.begin(), files.end());
  crypto::Sha256Hasher identity;
  identity.Update({reinterpret_cast<const std::uint8_t*>(config_text.data()),
                   config_text.size()});
  pack->files_.reserve(files.size());
  for (const auto& path : files) {
    if (!pack->AddFile(path, error))
      return nullptr;
    const auto& mapped = pack->files_.back();
    std::uint64_t header = 0;
    std::memcpy(&header, mapped.data(), sizeof(header));
    identity.Update({mapped.data(), static_cast<std::size_t>(8 + header)});
  }
  pack->identity_ = identity.FinishHex();
  pack->index_.reserve(pack->tensors_.size());
  for (std::size_t i = 0; i < pack->tensors_.size(); ++i) {
    if (!pack->index_.emplace(pack->tensors_[i].name, i).second) {
      SetError(error, "duplicate EXL3 tensor " + pack->tensors_[i].name);
      return nullptr;
    }
  }
  return pack;
}

bool Pack::AddFile(const std::filesystem::path& path, std::string* error) {
  std::string map_error;
  auto mapped = platform::ReadOnlyMappedFile::Open(path, &map_error);
  if (!mapped.valid()) {
    SetError(error, "cannot map " + path.string() + ": " + map_error);
    return false;
  }
  if (mapped.size() < 8) {
    SetError(error, "truncated safetensors file " + path.string());
    return false;
  }
  std::uint64_t header = 0;
  std::memcpy(&header, mapped.data(), sizeof(header));
  if (header > mapped.size() - 8 || header > (std::uint64_t{1} << 30)) {
    SetError(error, "invalid safetensors header in " + path.string());
    return false;
  }
  json::Value meta;
  try {
    meta = json::parse(std::string_view(
        reinterpret_cast<const char*>(mapped.data()) + 8, header));
  } catch (const std::exception& e) {
    SetError(error, "invalid safetensors header in " + path.string() + ": " +
                        e.what());
    return false;
  }
  const auto file = static_cast<std::uint32_t>(files_.size());
  const std::uint64_t data_start = 8 + header;
  tensors_.reserve(tensors_.size() + meta.size());
  for (const auto& [name, value] : meta.members()) {
    if (name == "__metadata__")
      continue;
    PackTensor t;
    t.name = name;
    std::size_t element_bytes = 0;
    if (!ParseDType(value.member_str("dtype"), &t.dtype, &element_bytes)) {
      SetError(error, "unsupported dtype of " + name);
      return false;
    }
    const json::Value* shape = value.find("shape");
    const json::Value* offsets = value.find("data_offsets");
    if (shape == nullptr || offsets == nullptr || offsets->size() != 2) {
      SetError(error, "incomplete safetensors entry " + name);
      return false;
    }
    for (const auto& d : shape->items())
      t.shape.push_back(d.as_size());
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    if (end < begin || data_start + end > mapped.size() ||
        end - begin != t.Elements() * element_bytes) {
      SetError(error, "safetensors entry out of range: " + name);
      return false;
    }
    t.size_bytes = end - begin;
    t.file = file;
    t.file_offset = data_start + begin;
    t.data = mapped.data() + t.file_offset;
    tensors_.push_back(std::move(t));
  }
  core::GgufMappedRegion region;
  region.data = mapped.data();
  region.size = mapped.size();
  region.file_descriptor = mapped.PlatformHandle();
  region.source_path = path;
  files_.push_back(std::move(mapped));
  regions_.push_back(std::move(region));
  return true;
}

const PackTensor* Pack::Find(std::string_view name) const noexcept {
  const auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const json::Value& Pack::TextConfig() const noexcept {
  const json::Value* text = config_.find("text_config");
  return text != nullptr && text->is_object() ? *text : config_;
}

std::string Pack::ReadText(std::string_view file) const {
  return ReadFile(directory_ / std::filesystem::path(file));
}

bool IsFlashNextPack(const std::filesystem::path& directory) {
  std::error_code ec;
  if (!std::filesystem::is_directory(directory, ec))
    return false;
  const std::string text = ReadFile(directory / "config.json");
  if (text.empty())
    return false;
  try {
    const auto config = json::parse(text);
    const json::Value* quant = config.find("quantization_config");
    return config.member_str("model_type") == "qwen4_exp" && quant != nullptr &&
           quant->member_str("quant_method") == "exl3";
  } catch (const std::exception&) {
    return false;
  }
}

double PackBits(const std::filesystem::path& directory) {
  const std::string text = ReadFile(directory / "config.json");
  try {
    const auto config = json::parse(text);
    const json::Value* quant = config.find("quantization_config");
    return quant != nullptr ? quant->member_double("bits") : 0.0;
  } catch (const std::exception&) {
    return 0.0;
  }
}

std::uint32_t PackContextLength(const std::filesystem::path& directory) {
  const std::string text = ReadFile(directory / "config.json");
  try {
    const auto config = json::parse(text);
    const json::Value* t = config.find("text_config");
    return static_cast<std::uint32_t>(
        (t != nullptr ? *t : config).member_size("max_position_embeddings"));
  } catch (const std::exception&) {
    return 0;
  }
}

}  // namespace gufo::models::qwen38_flash_next::exl3
