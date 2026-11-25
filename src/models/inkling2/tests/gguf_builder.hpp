#pragma once

// Builds tiny in-memory GGUF files for refusal tests.

#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

namespace inkling2_test {

struct U32Array {
  std::vector<std::uint32_t> values;
};
struct BoolArray {
  std::vector<bool> values;
};
using Value =
    std::variant<std::string, std::uint32_t, float, bool, U32Array, BoolArray>;
struct Kv {
  std::string key;
  Value value;
};
struct TensorSpec {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type = 0;   // ggml type id
  std::uint64_t bytes = 0;  // payload size reserved in the data section
};

class Writer {
public:
  template<class T>
  void Pod(const T& v) {
    const auto* p = reinterpret_cast<const char*>(&v);
    bytes_.insert(bytes_.end(), p, p + sizeof(T));
  }
  void Str(const std::string& s) {
    Pod<std::uint64_t>(s.size());
    bytes_.insert(bytes_.end(), s.begin(), s.end());
  }
  void Val(const Value& v) {
    if (const auto* s = std::get_if<std::string>(&v)) {
      Pod<std::uint32_t>(8);
      Str(*s);
    } else if (const auto* u = std::get_if<std::uint32_t>(&v)) {
      Pod<std::uint32_t>(4);
      Pod(*u);
    } else if (const auto* f = std::get_if<float>(&v)) {
      Pod<std::uint32_t>(6);
      Pod(*f);
    } else if (const auto* b = std::get_if<bool>(&v)) {
      Pod<std::uint32_t>(7);
      Pod<std::uint8_t>(*b);
    } else if (const auto* a = std::get_if<U32Array>(&v)) {
      Pod<std::uint32_t>(9);
      Pod<std::uint32_t>(4);
      Pod<std::uint64_t>(a->values.size());
      for (const auto x : a->values) {
        Pod(x);
      }
    } else {
      const auto& a2 = std::get<BoolArray>(v);
      Pod<std::uint32_t>(9);
      Pod<std::uint32_t>(7);
      Pod<std::uint64_t>(a2.values.size());
      for (const bool x : a2.values) {
        Pod<std::uint8_t>(x);
      }
    }
  }
  void Pad(std::size_t alignment) {
    while (bytes_.size() % alignment != 0) {
      bytes_.push_back(0);
    }
  }
  std::vector<char>& bytes() { return bytes_; }

private:
  std::vector<char> bytes_;
};

/// A GGUF v3 image: `kvs`, then `tensors` laid out back to back (32-byte
/// aligned) with zeroed payloads.
inline std::vector<char> BuildGguf(const std::vector<Kv>& kvs,
                                   const std::vector<TensorSpec>& tensors) {
  Writer w;
  w.Pod<char>('G');
  w.Pod<char>('G');
  w.Pod<char>('U');
  w.Pod<char>('F');
  w.Pod<std::uint32_t>(3);
  w.Pod<std::uint64_t>(tensors.size());
  w.Pod<std::uint64_t>(kvs.size());
  for (const auto& kv : kvs) {
    w.Str(kv.key);
    w.Val(kv.value);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    w.Str(t.name);
    w.Pod<std::uint32_t>(static_cast<std::uint32_t>(t.dims.size()));
    for (const auto d : t.dims) {
      w.Pod(d);
    }
    w.Pod<std::uint32_t>(t.type);
    w.Pod(offset);
    offset += (t.bytes + 31) / 32 * 32;
  }
  w.Pad(32);
  w.bytes().resize(w.bytes().size() + offset, 0);
  return std::move(w.bytes());
}

}  // namespace inkling2_test
