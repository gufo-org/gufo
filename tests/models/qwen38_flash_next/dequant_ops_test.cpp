#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace {

namespace core = gufo::core;
namespace qfn = gufo::models::qwen38_flash_next;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Check(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}

struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

qfn::TensorRef Ref(const core::GgufTensorInfo& info) {
  qfn::TensorRef t;
  t.data = info.data;
  t.type = info.type;
  t.cols = info.dimensions[0];
  t.rows = info.dimensions.size() > 1 ? info.dimensions[1] : 1;
  t.name = info.name;
  return t;
}

std::string FirstLayerWith(const core::GgufReader& reader,
                           std::string_view tensor) {
  for (int layer = 0; layer < 256; ++layer) {
    const std::string name =
        "blk." + std::to_string(layer) + "." + std::string(tensor);
    if (reader.FindTensor(name) != nullptr)
      return "blk." + std::to_string(layer) + ".";
  }
  throw std::runtime_error("no layer has " + std::string(tensor));
}

bool Expandable(core::GgmlType type) {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ5_1 || type == core::GgmlType::kIQ4_NL ||
         type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
}

// The CPU reference row: the model's scalar oracle, or the quant library for
// the IQ codebook formats the oracle does not decode.
void ReferenceRow(const qfn::TensorRef& t, std::uint64_t e, std::uint64_t r,
                  float* out) {
  const std::uint8_t* src = t.Expert(e) + r * t.RowBytes();
  switch (t.type) {
    case core::GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(src, out, t.cols);
      return;
    case core::GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, out, t.cols);
      return;
    case core::GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(src, out, t.cols);
      return;
    default:
      qfn::cpu::DequantizeRow(t, e, r, out);
  }
}

// Uploads the parts back to back, as Stack does for K-quant weights (or the
// first `experts` matrices of an expert tensor), expands them in one launch
// and compares every F16 bit pattern with the CPU reference rounded once to
// F16.
void CheckGroup(const core::GgufReader& reader, std::string_view label,
                const std::vector<std::string>& names,
                std::uint64_t experts = 1) {
  std::vector<qfn::TensorRef> parts;
  std::size_t bytes = 0;
  std::size_t count = 0;
  for (const auto& name : names) {
    const auto* info = reader.FindTensor(name);
    Require(info != nullptr, "missing tensor " + name);
    parts.push_back(Ref(*info));
    parts.back().experts = experts;
    Require(parts.back().type == parts.front().type &&
                parts.back().cols == parts.front().cols,
            std::string(label) + ": parts differ in type or width");
    bytes += parts.back().SizeBytes();
    count += parts.back().cols * parts.back().rows * experts;
  }
  const core::GgmlType type = parts.front().type;
  if (!Expandable(type)) {
    std::cout << label << ": " << core::ToString(type) << " skipped\n";
    return;
  }

  Buffer packed(bytes);
  std::size_t offset = 0;
  for (const auto& part : parts) {
    Check(hipMemcpy(static_cast<std::uint8_t*>(packed.data) + offset, part.data,
                    part.SizeBytes(), hipMemcpyHostToDevice));
    offset += part.SizeBytes();
  }
  Buffer expanded(count * sizeof(__half));
  Require(qfn::rocm::DequantizeHalf(packed.data, type,
                                    static_cast<__half*>(expanded.data), count,
                                    nullptr),
          std::string(label) + ": launch rejected");
  Check(hipDeviceSynchronize());
  std::vector<std::uint16_t> gpu(count);
  Check(hipMemcpy(gpu.data(), expanded.data, count * sizeof(std::uint16_t),
                  hipMemcpyDeviceToHost));

  std::size_t mismatches = 0;
  std::size_t index = 0;
  std::vector<float> row;
  for (const auto& part : parts) {
    row.resize(part.cols);
    for (std::uint64_t r = 0; r < part.rows * experts; ++r) {
      ReferenceRow(part, r / part.rows, r % part.rows, row.data());
      for (const float value : row) {
        const __half h = __float2half_rn(value);
        std::uint16_t want = 0;
        std::memcpy(&want, &h, sizeof(want));
        if (gpu[index] != want && mismatches++ < 4) {
          std::cerr << label << ": element " << index << " gpu=0x" << std::hex
                    << gpu[index] << " cpu=0x" << want << std::dec << '\n';
        }
        ++index;
      }
    }
  }
  std::cout << label << ": " << core::ToString(type) << " " << count
            << " values, " << mismatches << " mismatches\n";
  Require(mismatches == 0, std::string(label) + " is not bit-exact");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--model") {
    std::cerr << "Usage: dequant_ops_test --model FIRST.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto reader = core::GgufReader::OpenFile(argv[2], &error);
    Require(reader != nullptr, error);
    const std::string linear = FirstLayerWith(*reader, "attn_qkv.weight");
    const std::string full = FirstLayerWith(*reader, "attn_q.weight");
    CheckGroup(*reader, "ssm_in stack",
               {linear + "attn_qkv.weight", linear + "attn_gate.weight"});
    CheckGroup(*reader, "attn_qkv stack",
               {full + "attn_q.weight", full + "attn_k.weight",
                full + "attn_v.weight"});
    CheckGroup(*reader, "ssm_out", {linear + "ssm_out.weight"});
    CheckGroup(*reader, "head", {"output.weight"});
    CheckGroup(*reader, "shexp_down", {"blk.0.ffn_down_shexp.weight"});
    // Routed experts: the first eight matrices of each distinct format.
    constexpr std::uint64_t kExperts = 8;
    std::vector<core::GgmlType> seen;
    for (int layer = 0; layer < 256; ++layer) {
      for (const char* proj : {"ffn_gate_exps", "ffn_down_exps"}) {
        const std::string name =
            "blk." + std::to_string(layer) + "." + proj + ".weight";
        const auto* info = reader->FindTensor(name);
        if (info == nullptr ||
            std::find(seen.begin(), seen.end(), info->type) != seen.end())
          continue;
        seen.push_back(info->type);
        CheckGroup(*reader, name, {name}, kExperts);
      }
    }
    std::cout << "dequant_ops: bit-exact\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dequant_ops_test: " << e.what() << '\n';
    return 1;
  }
}
