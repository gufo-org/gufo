#include "src/models/inkling2/text/gguf_meta.hpp"

#include <cstdio>
#include <cstring>
#include <memory>

namespace inkling2::text {

namespace {

enum Type : std::uint32_t {
  kU8 = 0,
  kI8,
  kU16,
  kI16,
  kU32,
  kI32,
  kF32,
  kBool,
  kString,
  kArray,
  kU64,
  kI64,
  kF64
};

struct File {
  std::FILE* f{nullptr};
  explicit File(const std::string& p) : f(std::fopen(p.c_str(), "rb")) {}
  ~File() {
    if (f != nullptr)
      std::fclose(f);
  }
  bool Read(void* dst, std::size_t n) { return std::fread(dst, 1, n, f) == n; }
  bool Skip(std::uint64_t n) {
    return fseeko(f, static_cast<off_t>(n), SEEK_CUR) == 0;
  }
  template<typename T>
  bool Get(T* v) {
    return Read(v, sizeof(T));
  }  // little-endian host
  bool GetString(std::string* s) {
    std::uint64_t n;
    if (!Get(&n) || n > (1ULL << 32))
      return false;
    s->resize(n);
    return n == 0 || Read(s->data(), n);
  }
};

std::size_t ScalarSize(std::uint32_t t) {
  switch (t) {
    case kU8:
    case kI8:
    case kBool:
      return 1;
    case kU16:
    case kI16:
      return 2;
    case kU32:
    case kI32:
    case kF32:
      return 4;
    case kU64:
    case kI64:
    case kF64:
      return 8;
    default:
      return 0;
  }
}

bool SkipValue(File& f, std::uint32_t type) {
  if (type == kString) {
    std::uint64_t n;
    return f.Get(&n) && f.Skip(n);
  }
  if (type == kArray) {
    std::uint32_t et;
    std::uint64_t n;
    if (!f.Get(&et) || !f.Get(&n))
      return false;
    if (et == kString || et == kArray) {
      for (std::uint64_t i = 0; i < n; ++i)
        if (!SkipValue(f, et))
          return false;
      return true;
    }
    return f.Skip(n * ScalarSize(et));
  }
  return f.Skip(ScalarSize(type));
}

bool ReadUint(File& f, std::uint32_t type, std::uint64_t* v) {
  switch (type) {
    case kU8: {
      std::uint8_t x;
      if (!f.Get(&x))
        return false;
      *v = x;
      return true;
    }
    case kU16: {
      std::uint16_t x;
      if (!f.Get(&x))
        return false;
      *v = x;
      return true;
    }
    case kU32: {
      std::uint32_t x;
      if (!f.Get(&x))
        return false;
      *v = x;
      return true;
    }
    case kI32: {
      std::int32_t x;
      if (!f.Get(&x))
        return false;
      *v = static_cast<std::uint64_t>(x);
      return true;
    }
    case kU64: {
      std::uint64_t x;
      if (!f.Get(&x))
        return false;
      *v = x;
      return true;
    }
    default:
      return false;
  }
}

bool ReadStringArray(File& f, std::vector<std::string>* out) {
  std::uint32_t et;
  std::uint64_t n;
  if (!f.Get(&et) || !f.Get(&n) || et != kString)
    return false;
  out->resize(n);
  for (auto& s : *out)
    if (!f.GetString(&s))
      return false;
  return true;
}

}  // namespace

bool ReadTokenizerDataFromGguf(const std::string& path, TokenizerData* out,
                               std::string* error) {
  const auto fail = [&](const char* m) {
    if (error != nullptr)
      *error = std::string(m) + ": " + path;
    return false;
  };
  File f(path);
  if (f.f == nullptr)
    return fail("cannot open");
  char magic[4];
  std::uint32_t version;
  std::uint64_t n_tensors, n_kv;
  if (!f.Read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0)
    return fail("not a GGUF file");
  if (!f.Get(&version) || version < 2 || !f.Get(&n_tensors) || !f.Get(&n_kv))
    return fail("bad GGUF header");
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    std::uint32_t type;
    if (!f.GetString(&key) || !f.Get(&type))
      return fail("truncated metadata");
    bool ok = true;
    std::uint64_t u = 0;
    if (key == "tokenizer.ggml.model")
      ok = type == kString && f.GetString(&out->model);
    else if (key == "tokenizer.ggml.pre")
      ok = type == kString && f.GetString(&out->pre);
    else if (key == "tokenizer.chat_template")
      ok = type == kString && f.GetString(&out->chat_template);
    else if (key == "tokenizer.ggml.tokens")
      ok = type == kArray && ReadStringArray(f, &out->tokens);
    else if (key == "tokenizer.ggml.merges")
      ok = type == kArray && ReadStringArray(f, &out->merges);
    else if (key == "tokenizer.ggml.token_type") {
      std::uint32_t et;
      std::uint64_t n;
      ok = type == kArray && f.Get(&et) && f.Get(&n) && et == kI32;
      if (ok) {
        out->token_types.resize(n);
        ok = n == 0 || f.Read(out->token_types.data(), n * 4);
      }
    } else if (key == "tokenizer.ggml.eos_token_id") {
      ok = ReadUint(f, type, &u);
      out->eos_id = static_cast<std::uint32_t>(u);
    } else if (key == "tokenizer.ggml.eot_token_id") {
      ok = ReadUint(f, type, &u);
      out->eot_id = static_cast<std::uint32_t>(u);
    } else if (key == "tokenizer.ggml.bos_token_id") {
      ok = ReadUint(f, type, &u);
      out->bos_id = static_cast<std::uint32_t>(u);
    } else if (key == "tokenizer.ggml.padding_token_id") {
      ok = ReadUint(f, type, &u);
      out->pad_id = static_cast<std::uint32_t>(u);
    } else if (key == "tokenizer.ggml.add_bos_token" && type == kBool) {
      std::uint8_t b;
      ok = f.Get(&b);
      out->add_bos = b != 0;
    } else if (key == "tokenizer.ggml.add_eos_token" && type == kBool) {
      std::uint8_t b;
      ok = f.Get(&b);
      out->add_eos = b != 0;
    } else
      ok = SkipValue(f, type);
    if (!ok)
      return fail(("bad metadata value for " + key).c_str());
  }
  if (out->tokens.empty())
    return fail("no tokenizer.ggml.tokens");
  return true;
}

}  // namespace inkling2::text
