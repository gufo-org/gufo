#include "mmproj_gguf.hpp"

#include <cstring>

namespace gufo::models::inkling2::vision {
namespace {

struct Cur {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  template<class T>
  T Get() {
    T v{};
    if (p + sizeof(T) > end) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string Str() {
    uint64_t n = Get<uint64_t>();
    if (!ok || p + n > end) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
  }
};

// reads one value of gguf type t into a double (numbers/bools); strings via *s
double ReadScalar(Cur& c, uint32_t t, std::string* s) {
  switch (t) {
    case 0:
      return c.Get<uint8_t>();
    case 1:
      return c.Get<int8_t>();
    case 2:
      return c.Get<uint16_t>();
    case 3:
      return c.Get<int16_t>();
    case 4:
      return c.Get<uint32_t>();
    case 5:
      return c.Get<int32_t>();
    case 6:
      return c.Get<float>();
    case 7:
      return c.Get<uint8_t>();
    case 8: {
      std::string v = c.Str();
      if (s)
        *s = v;
      return 0;
    }
    case 10:
      return static_cast<double>(c.Get<uint64_t>());
    case 11:
      return static_cast<double>(c.Get<int64_t>());
    case 12:
      return c.Get<double>();
    default:
      c.ok = false;
      return 0;
  }
}

uint64_t TypeBytes(int type, uint64_t n) {
  switch (type) {
    case 0:
      return n * 4;
    case 1:
    case 30:
      return n * 2;
    case 8:
      return n / 32 * 34;
    default:
      return 0;
  }
}

}  // namespace

bool MmprojGguf::Open(const std::string& path, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err)
      *err = m;
    return false;
  };
  std::string map_error;
  file_ = gufo::platform::ReadOnlyMappedFile::Open(path, &map_error);
  if (!file_.valid())
    return fail("cannot map " + path + ": " + map_error);
  const uint8_t* base = file_.data();
  const uint64_t size = file_.size();
  Cur c{base, base + size};
  if (c.Get<uint32_t>() != 0x46554747u)
    return fail("not a GGUF file");
  uint32_t ver = c.Get<uint32_t>();
  if (ver < 2 || ver > 3)
    return fail("unsupported GGUF version");
  uint64_t n_t = c.Get<uint64_t>(), n_kv = c.Get<uint64_t>();
  uint64_t align = 32;
  for (uint64_t i = 0; i < n_kv && c.ok; i++) {
    std::string key = c.Str();
    uint32_t t = c.Get<uint32_t>();
    if (t == 9) {
      uint32_t it = c.Get<uint32_t>();
      uint64_t n = c.Get<uint64_t>();
      std::vector<double> v;
      for (uint64_t j = 0; j < n && c.ok; j++)
        v.push_back(ReadScalar(c, it, nullptr));
      arr_[key] = std::move(v);
    } else {
      std::string s;
      double d = ReadScalar(c, t, &s);
      if (t == 8)
        str_[key] = s;
      else
        num_[key] = d;
      if (key == "general.alignment")
        align = static_cast<uint64_t>(d);
    }
  }
  struct Info {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> ne;
    uint64_t off;
  };
  std::vector<Info> infos;
  for (uint64_t i = 0; i < n_t && c.ok; i++) {
    Info in;
    in.name = c.Str();
    uint32_t nd = c.Get<uint32_t>();
    if (nd > 4)
      return fail("bad tensor rank");
    for (uint32_t d = 0; d < nd; d++)
      in.ne.push_back(c.Get<uint64_t>());
    in.type = c.Get<uint32_t>();
    in.off = c.Get<uint64_t>();
    infos.push_back(std::move(in));
  }
  if (!c.ok)
    return fail("truncated GGUF header");
  uint64_t hdr = static_cast<uint64_t>(c.p - base);
  uint64_t data0 = (hdr + align - 1) / align * align;
  for (auto& in : infos) {
    MmprojTensor t;
    t.name = in.name;
    t.type = static_cast<int>(in.type);
    t.ne = in.ne;
    t.nbytes = TypeBytes(t.type, t.nelem());
    if (t.nbytes == 0)
      return fail("unsupported tensor type " + std::to_string(in.type) +
                  " for " + in.name);
    if (data0 + in.off + t.nbytes > size)
      return fail("tensor out of range: " + in.name);
    t.data = base + data0 + in.off;
    tensors_[t.name] = std::move(t);
  }
  return true;
}

const MmprojTensor* MmprojGguf::Find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}
bool MmprojGguf::GetU32(const std::string& k, uint32_t* v) const {
  auto it = num_.find(k);
  if (it == num_.end())
    return false;
  *v = static_cast<uint32_t>(it->second);
  return true;
}
bool MmprojGguf::GetF32(const std::string& k, float* v) const {
  auto it = num_.find(k);
  if (it == num_.end())
    return false;
  *v = static_cast<float>(it->second);
  return true;
}
bool MmprojGguf::GetStr(const std::string& k, std::string* v) const {
  auto it = str_.find(k);
  if (it == str_.end())
    return false;
  *v = it->second;
  return true;
}
bool MmprojGguf::GetF32Array(const std::string& k,
                             std::vector<float>* v) const {
  auto it = arr_.find(k);
  if (it == arr_.end())
    return false;
  v->assign(it->second.begin(), it->second.end());
  return true;
}

}  // namespace gufo::models::inkling2::vision
