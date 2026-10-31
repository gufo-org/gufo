#include "src/models/fish_drama3/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace gufo::models::fish_drama3 {
namespace {

namespace json = gufo::json;

std::uint64_t ReadLeU64(const std::uint8_t* ptr) {
  std::uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(ptr[i]) << (8 * i);
  }
  return value;
}

bool ParseU64(const json::Value& value, std::uint64_t* out) {
  if (!value.is_number()) {
    return false;
  }
  const double num = value.as_double();
  if (!std::isfinite(num) || num < 0.0 || std::floor(num) != num) {
    return false;
  }
  *out = static_cast<std::uint64_t>(num);
  return true;
}

bool LoadSingleSafetensorsShard(
    const std::filesystem::path& file_path,
    std::vector<gufo::platform::ReadOnlyMappedFile>* shards,
    std::unordered_map<std::string, TensorView>* tensors) {
  auto mapped = gufo::platform::ReadOnlyMappedFile::Open(file_path);
  if (!mapped.valid() || mapped.size() < 8) {
    return false;
  }
  const auto* base = mapped.data();
  const std::uint64_t header_len = ReadLeU64(base);
  if (header_len == 0 || header_len > mapped.size() - 8) {
    return false;
  }
  const std::string header_str(reinterpret_cast<const char*>(base + 8),
                               static_cast<std::size_t>(header_len));
  json::Value root;
  try {
    root = json::parse(header_str);
  } catch (const std::exception&) {
    return false;
  }
  if (!root.is_object()) {
    return false;
  }

  const std::size_t data_base_offset = 8 + static_cast<std::size_t>(header_len);
  const std::size_t data_region_bytes = mapped.size() - data_base_offset;
  const std::uint8_t* data_base = base + data_base_offset;

  for (const auto& [name, entry] : root.members()) {
    if (name == "__metadata__") {
      continue;
    }
    if (!entry.is_object()) {
      return false;
    }
    const auto* dtype_val = entry.find("dtype");
    const auto* shape_val = entry.find("shape");
    const auto* offsets_val = entry.find("data_offsets");
    if (dtype_val == nullptr || !dtype_val->is_string() ||
        shape_val == nullptr || !shape_val->is_array() ||
        offsets_val == nullptr || !offsets_val->is_array() ||
        offsets_val->size() != 2) {
      return false;
    }

    TensorView view;
    view.name = name;
    view.dtype = ParseDType(dtype_val->str());
    if (view.dtype == DType::kUnknown) {
      return false;
    }

    view.shape.reserve(shape_val->size());
    for (const auto& dim_val : shape_val->items()) {
      std::uint64_t dim = 0;
      if (!ParseU64(dim_val, &dim)) {
        return false;
      }
      view.shape.push_back(dim);
    }

    std::uint64_t begin_off = 0;
    std::uint64_t end_off = 0;
    if (!ParseU64(offsets_val->items()[0], &begin_off) ||
        !ParseU64(offsets_val->items()[1], &end_off) || end_off < begin_off ||
        end_off > data_region_bytes) {
      return false;
    }

    view.data = data_base + static_cast<std::size_t>(begin_off);
    view.byte_size = static_cast<std::size_t>(end_off - begin_off);
    if (!view.IsContiguousValid()) {
      return false;
    }
    (*tensors)[name] = std::move(view);
  }

  shards->push_back(std::move(mapped));
  return true;
}

std::uint16_t ReadLeU16(const std::uint8_t* ptr) {
  return static_cast<std::uint16_t>(ptr[0]) |
         (static_cast<std::uint16_t>(ptr[1]) << 8U);
}

std::uint32_t ReadLeU32(const std::uint8_t* ptr) {
  return static_cast<std::uint32_t>(ptr[0]) |
         (static_cast<std::uint32_t>(ptr[1]) << 8U) |
         (static_cast<std::uint32_t>(ptr[2]) << 16U) |
         (static_cast<std::uint32_t>(ptr[3]) << 24U);
}

struct ZipEntrySpan {
  const std::uint8_t* data{nullptr};
  std::size_t size{0};
};

bool ParseStoredZipArchive(const std::uint8_t* base, std::size_t total_size,
                           std::unordered_map<std::string, ZipEntrySpan>* out) {
  if (total_size < 22) {
    return false;
  }
  // Locate End of Central Directory (EOCD) signature 0x06054b50 near EOF
  const std::size_t max_search = std::min<std::size_t>(total_size, 65557U);
  std::size_t eocd_pos = total_size;
  for (std::size_t back = 22; back <= max_search; ++back) {
    const std::size_t pos = total_size - back;
    if (ReadLeU32(base + pos) == 0x06054b50U) {
      eocd_pos = pos;
      break;
    }
  }
  if (eocd_pos == total_size) {
    return false;
  }

  std::uint64_t cd_offset = ReadLeU32(base + eocd_pos + 16);
  std::uint64_t cd_size = ReadLeU32(base + eocd_pos + 12);
  // Check for ZIP64 EOCD Locator (0x07064b50) 20 bytes before EOCD
  if (eocd_pos >= 20 && ReadLeU32(base + eocd_pos - 20) == 0x07064b50U) {
    const std::uint64_t zip64_eocd_off = ReadLeU64(base + eocd_pos - 12);
    if (zip64_eocd_off + 56 <= total_size &&
        ReadLeU32(base + zip64_eocd_off) == 0x06064b50U) {
      cd_size = ReadLeU64(base + zip64_eocd_off + 40);
      cd_offset = ReadLeU64(base + zip64_eocd_off + 48);
    }
  }
  if (cd_offset + cd_size > total_size) {
    return false;
  }

  std::size_t pos = static_cast<std::size_t>(cd_offset);
  const std::size_t cd_end = pos + static_cast<std::size_t>(cd_size);
  while (pos + 46 <= cd_end) {
    if (ReadLeU32(base + pos) != 0x02014b50U) {
      break;
    }
    const std::uint16_t method = ReadLeU16(base + pos + 10);
    std::uint64_t comp_size = ReadLeU32(base + pos + 20);
    std::uint64_t uncomp_size = ReadLeU32(base + pos + 24);
    const std::uint16_t name_len = ReadLeU16(base + pos + 28);
    const std::uint16_t extra_len = ReadLeU16(base + pos + 30);
    const std::uint16_t comment_len = ReadLeU16(base + pos + 32);
    std::uint64_t lfh_offset = ReadLeU32(base + pos + 42);

    if (pos + 46 + name_len + extra_len + comment_len > cd_end) {
      return false;
    }
    const std::string name(reinterpret_cast<const char*>(base + pos + 46),
                           name_len);
    // Parse ZIP64 extra field if present
    const std::uint8_t* extra = base + pos + 46 + name_len;
    std::size_t epos = 0;
    while (epos + 4 <= extra_len) {
      const std::uint16_t header_id = ReadLeU16(extra + epos);
      const std::uint16_t data_size = ReadLeU16(extra + epos + 2);
      if (epos + 4 + data_size > extra_len) {
        break;
      }
      if (header_id == 0x0001U) {
        std::size_t zpos = epos + 4;
        if (uncomp_size == 0xFFFFFFFFU && zpos + 8 <= epos + 4 + data_size) {
          uncomp_size = ReadLeU64(extra + zpos);
          zpos += 8;
        }
        if (comp_size == 0xFFFFFFFFU && zpos + 8 <= epos + 4 + data_size) {
          comp_size = ReadLeU64(extra + zpos);
          zpos += 8;
        }
        if (lfh_offset == 0xFFFFFFFFU && zpos + 8 <= epos + 4 + data_size) {
          lfh_offset = ReadLeU64(extra + zpos);
        }
      }
      epos += 4 + data_size;
    }

    if (method == 0 && lfh_offset + 30 <= total_size &&
        ReadLeU32(base + lfh_offset) == 0x04034b50U) {
      const std::uint16_t lfh_name_len = ReadLeU16(base + lfh_offset + 26);
      const std::uint16_t lfh_extra_len = ReadLeU16(base + lfh_offset + 28);
      const std::uint64_t data_off =
          lfh_offset + 30 + lfh_name_len + lfh_extra_len;
      if (data_off + uncomp_size <= total_size) {
        (*out)[name] = ZipEntrySpan{
            .data = base + static_cast<std::size_t>(data_off),
            .size = static_cast<std::size_t>(uncomp_size),
        };
      }
    }
    pos += 46 + name_len + extra_len + comment_len;
  }
  return !out->empty();
}

struct PthRawTensor {
  std::string dtype;
  std::string storage_key;
  std::vector<std::uint64_t> shape;
};

bool ParseTorchDataPkl(const std::uint8_t* pkl, std::size_t len,
                       std::unordered_map<std::string, PthRawTensor>* tensors) {
  enum class Kind { kMark, kInt, kStr, kTuple, kStorage, kTensor, kOther };
  struct Val {
    Kind kind = Kind::kOther;
    std::int64_t i = 0;
    std::string s{};
    std::vector<Val> tuple{};
    PthRawTensor tensor{};
    static Val Make(Kind k) {
      Val v;
      v.kind = k;
      return v;
    }
    static Val MakeInt(std::int64_t x) {
      Val v;
      v.kind = Kind::kInt;
      v.i = x;
      return v;
    }
    static Val MakeStr(std::string str) {
      Val v;
      v.kind = Kind::kStr;
      v.s = std::move(str);
      return v;
    }
    static Val MakeTuple(std::vector<Val> items) {
      Val v;
      v.kind = Kind::kTuple;
      v.tuple = std::move(items);
      return v;
    }
  };

  std::vector<Val> stack;
  std::unordered_map<std::uint32_t, Val> memo;
  std::size_t pos = 0;

  while (pos < len) {
    const std::uint8_t op = pkl[pos++];
    switch (op) {
      case 0x80:  // PROTO
        if (pos >= len)
          return false;
        ++pos;
        break;
      case '(':  // MARK
        stack.push_back(Val::Make(Kind::kMark));
        break;
      case '.':  // STOP
        return !tensors->empty();
      case ')':  // EMPTY_TUPLE
        stack.push_back(Val::Make(Kind::kTuple));
        break;
      case '}':   // EMPTY_DICT
      case 0x89:  // NEWFALSE
      case 0x88:  // NEWTRUE
      case 'N':   // NONE
        stack.push_back(Val::Make(Kind::kOther));
        break;
      case 'K':  // BININT1
        if (pos >= len)
          return false;
        stack.push_back(Val::MakeInt(pkl[pos++]));
        break;
      case 'M':  // BININT2
        if (pos + 2 > len)
          return false;
        stack.push_back(Val::MakeInt(ReadLeU16(pkl + pos)));
        pos += 2;
        break;
      case 'J': {  // BININT
        if (pos + 4 > len)
          return false;
        const auto raw = static_cast<std::int32_t>(ReadLeU32(pkl + pos));
        stack.push_back(Val::MakeInt(raw));
        pos += 4;
        break;
      }
      case 'X': {  // BINUNICODE
        if (pos + 4 > len)
          return false;
        const std::uint32_t slen = ReadLeU32(pkl + pos);
        pos += 4;
        if (pos + slen > len)
          return false;
        stack.push_back(Val::MakeStr(
            std::string(reinterpret_cast<const char*>(pkl + pos), slen)));
        pos += slen;
        break;
      }
      case 'c': {  // GLOBAL (two newline-terminated strings)
        std::size_t nl1 = pos;
        while (nl1 < len && pkl[nl1] != '\n')
          ++nl1;
        if (nl1 >= len)
          return false;
        std::size_t nl2 = nl1 + 1;
        while (nl2 < len && pkl[nl2] != '\n')
          ++nl2;
        if (nl2 >= len)
          return false;
        const std::string mod(reinterpret_cast<const char*>(pkl + pos),
                              nl1 - pos);
        const std::string name(reinterpret_cast<const char*>(pkl + nl1 + 1),
                               nl2 - (nl1 + 1));
        stack.push_back(Val::MakeStr(mod + " " + name));
        pos = nl2 + 1;
        break;
      }
      case 'q':  // BINPUT
        if (pos >= len || stack.empty())
          return false;
        memo[pkl[pos++]] = stack.back();
        break;
      case 'r': {  // LONG_BINPUT
        if (pos + 4 > len || stack.empty())
          return false;
        memo[ReadLeU32(pkl + pos)] = stack.back();
        pos += 4;
        break;
      }
      case 'h': {  // BINGET
        if (pos >= len)
          return false;
        stack.push_back(memo[pkl[pos++]]);
        break;
      }
      case 'j': {  // LONG_BINGET
        if (pos + 4 > len)
          return false;
        stack.push_back(memo[ReadLeU32(pkl + pos)]);
        pos += 4;
        break;
      }
      case 't': {  // TUPLE
        std::vector<Val> items;
        while (!stack.empty() && stack.back().kind != Kind::kMark) {
          items.push_back(std::move(stack.back()));
          stack.pop_back();
        }
        if (stack.empty())
          return false;
        stack.pop_back();  // pop MARK
        std::reverse(items.begin(), items.end());
        stack.push_back(Val::MakeTuple(std::move(items)));
        break;
      }
      case 0x85:    // TUPLE1
      case 0x86:    // TUPLE2
      case 0x87: {  // TUPLE3
        const std::size_t n = (op == 0x85) ? 1U : (op == 0x86 ? 2U : 3U);
        if (stack.size() < n)
          return false;
        std::vector<Val> items(n);
        for (std::size_t i = 0; i < n; ++i) {
          items[n - 1U - i] = std::move(stack.back());
          stack.pop_back();
        }
        stack.push_back(Val::MakeTuple(std::move(items)));
        break;
      }
      case 'Q': {  // BINPERSID
        if (stack.empty() || stack.back().kind != Kind::kTuple ||
            stack.back().tuple.size() < 5) {
          return false;
        }
        const auto tup = std::move(stack.back().tuple);
        stack.pop_back();
        Val st = Val::Make(Kind::kStorage);
        st.tensor.dtype = tup[1].s;
        st.tensor.storage_key = tup[2].s;
        stack.push_back(std::move(st));
        break;
      }
      case 'R': {  // REDUCE
        if (stack.size() < 2)
          return false;
        Val args = std::move(stack.back());
        stack.pop_back();
        Val fn = std::move(stack.back());
        stack.pop_back();
        if (fn.kind == Kind::kStr &&
            fn.s == "torch._utils _rebuild_tensor_v2" &&
            args.kind == Kind::kTuple && args.tuple.size() >= 3 &&
            args.tuple[0].kind == Kind::kStorage &&
            args.tuple[2].kind == Kind::kTuple) {
          Val tv = Val::Make(Kind::kTensor);
          tv.tensor = args.tuple[0].tensor;
          for (const auto& dim_val : args.tuple[2].tuple) {
            tv.tensor.shape.push_back(static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, dim_val.i)));
          }
          stack.push_back(std::move(tv));
        } else {
          stack.push_back(Val::Make(Kind::kOther));
        }
        break;
      }
      case 's': {  // SETITEM
        if (stack.size() < 3)
          return false;
        Val v = std::move(stack.back());
        stack.pop_back();
        Val k = std::move(stack.back());
        stack.pop_back();
        if (k.kind == Kind::kStr && v.kind == Kind::kTensor) {
          (*tensors)[k.s] = std::move(v.tensor);
        }
        break;
      }
      case 'u': {  // SETITEMS
        std::vector<Val> kv;
        while (!stack.empty() && stack.back().kind != Kind::kMark) {
          kv.push_back(std::move(stack.back()));
          stack.pop_back();
        }
        if (stack.empty())
          return false;
        stack.pop_back();  // pop MARK
        for (std::size_t i = 0; i + 1 < kv.size(); i += 2) {
          const Val& v = kv[i];
          const Val& k = kv[i + 1];
          if (k.kind == Kind::kStr && v.kind == Kind::kTensor) {
            (*tensors)[k.s] = v.tensor;
          }
        }
        break;
      }
      case 'b':  // BUILD
        if (stack.size() < 2)
          return false;
        stack.pop_back();
        break;
      default:
        return false;
    }
  }
  return !tensors->empty();
}

bool LoadCodecFromPthFile(
    const std::filesystem::path& pth_path,
    std::vector<gufo::platform::ReadOnlyMappedFile>* shards,
    std::vector<std::vector<float>>* owned_f32_buffers,
    std::unordered_map<std::string, TensorView>* tensors) {
  auto mapped = gufo::platform::ReadOnlyMappedFile::Open(pth_path);
  if (!mapped.valid()) {
    return false;
  }
  std::unordered_map<std::string, ZipEntrySpan> entries;
  if (!ParseStoredZipArchive(mapped.data(), mapped.size(), &entries)) {
    return false;
  }

  const ZipEntrySpan* pkl_span = nullptr;
  std::string archive_prefix;
  for (const auto& [name, span] : entries) {
    if (name.ends_with("/data.pkl")) {
      pkl_span = &span;
      archive_prefix = name.substr(0, name.size() - 8);  // keep trailing '/'
      break;
    }
  }
  if (pkl_span == nullptr) {
    return false;
  }

  std::unordered_map<std::string, PthRawTensor> raw_tensors;
  if (!ParseTorchDataPkl(pkl_span->data, pkl_span->size, &raw_tensors)) {
    return false;
  }

  auto get_f32_span = [&](const PthRawTensor& rt,
                          std::size_t* num_elems) -> const float* {
    if (rt.dtype != "torch FloatStorage") {
      return nullptr;
    }
    const auto it = entries.find(archive_prefix + "data/" + rt.storage_key);
    if (it == entries.end()) {
      return nullptr;
    }
    std::size_t prod = 1;
    for (std::uint64_t d : rt.shape) {
      prod *= static_cast<std::size_t>(d);
    }
    if (it->second.size < prod * sizeof(float)) {
      return nullptr;
    }
    *num_elems = prod;
    return reinterpret_cast<const float*>(it->second.data);
  };

  // First pass: fold weight_norm (.parametrizations.weight.original0/1 AND
  // .weight_g/v)
  static constexpr std::string_view kOrig0 =
      ".parametrizations.weight.original0";
  static constexpr std::string_view kOrig1 =
      ".parametrizations.weight.original1";
  static constexpr std::string_view kWg = ".weight_g";
  static constexpr std::string_view kWv = ".weight_v";
  std::set<std::string> consumed;

  for (const auto& [name, rt0] : raw_tensors) {
    std::string prefix;
    std::string name1;
    if (name.ends_with(kOrig0)) {
      prefix = name.substr(0, name.size() - kOrig0.size());
      name1 = prefix + std::string(kOrig1);
    } else if (name.ends_with(kWg)) {
      prefix = name.substr(0, name.size() - kWg.size());
      name1 = prefix + std::string(kWv);
    } else {
      continue;
    }
    const auto it1 = raw_tensors.find(name1);
    if (it1 == raw_tensors.end()) {
      return false;
    }
    std::size_t n0 = 0;
    std::size_t n1 = 0;
    const float* g = get_f32_span(rt0, &n0);
    const float* v = get_f32_span(it1->second, &n1);
    if (g == nullptr || v == nullptr || n0 == 0 || n1 % n0 != 0) {
      return false;
    }
    const std::size_t slice_len = n1 / n0;
    std::vector<float> folded(n1);
    for (std::size_t i = 0; i < n0; ++i) {
      double sq_sum = 0.0;
      const float* v_row = v + i * slice_len;
      for (std::size_t j = 0; j < slice_len; ++j) {
        sq_sum += static_cast<double>(v_row[j]) * static_cast<double>(v_row[j]);
      }
      const float scale =
          g[i] / static_cast<float>(std::max(std::sqrt(sq_sum), 1e-12));
      float* out_row = folded.data() + i * slice_len;
      for (std::size_t j = 0; j < slice_len; ++j) {
        out_row[j] = v_row[j] * scale;
      }
    }
    owned_f32_buffers->push_back(std::move(folded));
    const auto& buf = owned_f32_buffers->back();

    const std::string target_name = prefix + ".weight";
    TensorView view;
    view.name = target_name;
    view.dtype = DType::kF32;
    view.shape = it1->second.shape;
    view.data = reinterpret_cast<const std::uint8_t*>(buf.data());
    view.byte_size = buf.size() * sizeof(float);
    (*tensors)[target_name] = std::move(view);

    consumed.insert(name);
    consumed.insert(name1);
  }

  // Second pass: add remaining FloatStorage tensors and split
  // .attention.wqkv.weight
  static constexpr std::string_view kWqkv = ".attention.wqkv.weight";
  for (const auto& [name, rt] : raw_tensors) {
    if (consumed.contains(name) || rt.dtype != "torch FloatStorage") {
      continue;
    }
    std::size_t n = 0;
    const float* ptr = get_f32_span(rt, &n);
    if (ptr == nullptr) {
      return false;
    }
    if (name.ends_with(kWqkv) && rt.shape.size() == 2 &&
        rt.shape[0] == 3U * rt.shape[1]) {
      const std::uint64_t dim = rt.shape[1];
      const std::size_t chunk_elems = static_cast<std::size_t>(dim * dim);
      const std::string attn_prefix =
          name.substr(0, name.size() - std::string_view("wqkv.weight").size());
      const char* proj_names[3] = {"q_proj.weight", "k_proj.weight",
                                   "v_proj.weight"};
      for (int p = 0; p < 3; ++p) {
        TensorView v;
        v.name = attn_prefix + proj_names[p];
        v.dtype = DType::kF32;
        v.shape = {dim, dim};
        v.data = reinterpret_cast<const std::uint8_t*>(ptr + p * chunk_elems);
        v.byte_size = chunk_elems * sizeof(float);
        (*tensors)[v.name] = std::move(v);
      }
      continue;
    }

    TensorView view;
    view.name = name;
    view.dtype = DType::kF32;
    if (name.ends_with(".alpha") && rt.shape.size() == 3 && rt.shape[0] == 1U &&
        rt.shape[2] == 1U) {
      view.shape = {rt.shape[1]};
    } else {
      view.shape = rt.shape;
    }
    view.data = reinterpret_cast<const std::uint8_t*>(ptr);
    view.byte_size = n * sizeof(float);
    (*tensors)[name] = std::move(view);
  }

  shards->push_back(std::move(mapped));
  return !tensors->empty();
}

}  // namespace

const TensorView* ModelSafetensors::Find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& ModelSafetensors::Require(const std::string& name) const {
  const auto* view = Find(name);
  if (view == nullptr) {
    throw std::runtime_error("Missing required fish_drama3 tensor: " + name);
  }
  return *view;
}

std::unique_ptr<ModelSafetensors> LoadSingleSafetensorsFile(
    const std::string& file_path) {
  const std::filesystem::path path(file_path);
  if (!std::filesystem::exists(path)) {
    return nullptr;
  }
  auto result = std::make_unique<ModelSafetensors>();
  if (!LoadSingleSafetensorsShard(path, &result->shards_, &result->tensors_)) {
    return nullptr;
  }
  return result;
}

std::unique_ptr<ModelSafetensors> LoadSafetensors(
    const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  auto result = std::make_unique<ModelSafetensors>();

  const auto single_path = dir / "model.safetensors";
  if (std::filesystem::exists(single_path)) {
    if (!LoadSingleSafetensorsShard(single_path, &result->shards_,
                                    &result->tensors_)) {
      return nullptr;
    }
    return result;
  }

  const auto index_path = dir / "model.safetensors.index.json";
  if (!std::filesystem::exists(index_path)) {
    return nullptr;
  }

  std::ifstream index_file(index_path);
  if (!index_file.is_open()) {
    return nullptr;
  }
  std::stringstream buffer;
  buffer << index_file.rdbuf();

  json::Value index_root;
  try {
    index_root = json::parse(buffer.str());
  } catch (const std::exception&) {
    return nullptr;
  }
  if (!index_root.is_object()) {
    return nullptr;
  }
  const auto* weight_map = index_root.find("weight_map");
  if (weight_map == nullptr || !weight_map->is_object()) {
    return nullptr;
  }

  std::set<std::string> shard_names;
  for (const auto& [tensor_name, shard_val] : weight_map->members()) {
    (void)tensor_name;
    if (!shard_val.is_string()) {
      return nullptr;
    }
    shard_names.insert(shard_val.str());
  }
  if (shard_names.empty()) {
    return nullptr;
  }

  for (const auto& shard_name : shard_names) {
    const auto shard_path = dir / shard_name;
    if (!std::filesystem::exists(shard_path) ||
        !LoadSingleSafetensorsShard(shard_path, &result->shards_,
                                    &result->tensors_)) {
      return nullptr;
    }
  }
  return result;
}

std::unique_ptr<ModelSafetensors> LoadCodecSafetensorsOrPth(
    const std::string& model_dir) {
  const std::filesystem::path dir(model_dir);
  const auto pth_path = dir / "codec.pth";
  if (std::filesystem::exists(pth_path)) {
    auto result = std::make_unique<ModelSafetensors>();
    if (LoadCodecFromPthFile(pth_path, &result->shards_,
                             &result->owned_f32_buffers_, &result->tensors_)) {
      return result;
    }
  }
  const auto st_path = dir / "codec.safetensors";
  if (std::filesystem::exists(st_path)) {
    return LoadSingleSafetensorsFile(st_path.string());
  }
  return nullptr;
}

}  // namespace gufo::models::fish_drama3
