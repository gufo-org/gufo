// Minimal, self-contained GGUF (v2/v3) reader for the Inkling 2 mmproj file.
// MIT. Maps the file read-only; tensor data pointers stay valid while the
// object lives.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gufo::models::inkling2::vision {

struct MmprojTensor {
  std::string name;
  int type = 0;  // ggml type id (0 F32, 1 F16, 8 Q8_0, 30 BF16)
  std::vector<uint64_t>
      ne;  // ggml order: ne[0] is the contiguous (innermost) dim
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;
  uint64_t nelem() const {
    uint64_t n = 1;
    for (auto d : ne)
      n *= d;
    return n;
  }
};

class MmprojGguf {
public:
  MmprojGguf() = default;
  ~MmprojGguf();
  MmprojGguf(const MmprojGguf&) = delete;
  MmprojGguf& operator=(const MmprojGguf&) = delete;

  /// Maps and parses `path`. On failure returns false and sets `*err`.
  bool Open(const std::string& path, std::string* err);

  const MmprojTensor* Find(const std::string& name) const;
  bool GetU32(const std::string& key, uint32_t* v) const;
  bool GetF32(const std::string& key, float* v) const;
  bool GetStr(const std::string& key, std::string* v) const;
  bool GetF32Array(const std::string& key, std::vector<float>* v) const;

private:
  void* map_ = nullptr;
  uint64_t size_ = 0;
  std::map<std::string, MmprojTensor> tensors_;
  // scalar kv stored as double/string/array of double (enough for the clip.*
  // keys)
  std::map<std::string, double> num_;
  std::map<std::string, std::string> str_;
  std::map<std::string, std::vector<double>> arr_;
};

}  // namespace gufo::models::inkling2::vision
