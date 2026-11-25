#include "mmproj_gguf.hpp"

#include <gufo/ext/gguf_reader.hpp>
#include <variant>

namespace gufo::models::inkling2::vision {

MmprojGguf::MmprojGguf() = default;
MmprojGguf::~MmprojGguf() = default;

bool MmprojGguf::Open(const std::string& path, std::string* err) {
  std::string why;
  reader_ = core::GgufReader::OpenFile(path, &why);
  if (reader_ == nullptr) {
    if (err)
      *err = "cannot read " + path + ": " + why;
    return false;
  }
  tensors_.clear();
  for (const auto& info : reader_->GetTensors()) {
    MmprojTensor t;
    t.name = std::string(info.name);
    t.type = static_cast<int>(info.type);
    t.ne = info.dimensions;
    t.data = static_cast<const uint8_t*>(info.data);
    t.nbytes = info.size_bytes;
    tensors_.emplace(t.name, std::move(t));
  }
  return true;
}

const MmprojTensor* MmprojGguf::Find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}
bool MmprojGguf::GetU32(const std::string& k, uint32_t* v) const {
  const auto value = reader_->GetMetadataUint32(k);
  if (!value)
    return false;
  *v = *value;
  return true;
}
bool MmprojGguf::GetF32(const std::string& k, float* v) const {
  const auto value = reader_->GetMetadataFloat32(k);
  if (!value)
    return false;
  *v = *value;
  return true;
}
bool MmprojGguf::GetStr(const std::string& k, std::string* v) const {
  const auto value = reader_->GetMetadataString(k);
  if (!value)
    return false;
  v->assign(*value);
  return true;
}
bool MmprojGguf::GetF32Array(const std::string& k,
                             std::vector<float>* v) const {
  const auto* meta = reader_->FindMetadata(k);
  const auto* items =
      meta ? std::get_if<std::vector<double>>(&meta->value) : nullptr;
  if (items == nullptr)
    return false;
  v->assign(items->begin(), items->end());
  return true;
}

}  // namespace gufo::models::inkling2::vision
