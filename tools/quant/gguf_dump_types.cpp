#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "src/core/gguf_reader.hpp"
int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &err);
  if (!reader) {
    std::fprintf(stderr, "open failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("tensors=%llu\n", (unsigned long long)reader->GetTensorCount());
  std::unordered_map<unsigned, std::pair<int, unsigned long long>> by_type;
  unsigned long long q8_0 = 0, q6_k = 0, q8_k = 0, q4_k = 0, bf16 = 0;
  for (const auto& t : reader->GetTensors()) {
    auto& e = by_type[(unsigned)t.type];
    e.first++;
    e.second += t.ElementCount();
    if (t.type == gufo::core::GgmlType::kQ8_0)
      q8_0 += t.ElementCount();
    if (t.type == gufo::core::GgmlType::kQ6_K)
      q6_k += t.ElementCount();
    if (t.type == gufo::core::GgmlType::kQ8_K)
      q8_k += t.ElementCount();
    if (t.type == gufo::core::GgmlType::kQ4_K)
      q4_k += t.ElementCount();
    if (t.type == gufo::core::GgmlType::kBF16)
      bf16 += t.ElementCount();
  }
  std::printf("\n--- type histogram (raw enum) ---\n");
  for (auto& [v, e] : by_type)
    std::printf(
        "enum=%-4u name=%-10s tensors=%-4d elements=%llu\n", v,
        std::string(gufo::core::ToString((gufo::core::GgmlType)v)).c_str(),
        e.first, e.second);
  std::printf(
      "\nproj elements: Q8_0=%llu Q6_K=%llu Q8_K=%llu Q4_K=%llu BF16=%llu\n",
      q8_0, q6_k, q8_k, q4_k, bf16);
  std::printf("\n--- first 40 tensors ---\n");
  unsigned i = 0;
  for (const auto& t : reader->GetTensors()) {
    if (i++ >= 40)
      break;
    std::printf("%-48s enum=%-4u %-8s dims=[", std::string(t.name).c_str(),
                (unsigned)t.type,
                std::string(gufo::core::ToString(t.type)).c_str());
    for (std::size_t d = 0; d < t.dimensions.size(); ++d)
      std::printf("%llu%s", (unsigned long long)t.dimensions[d],
                  d + 1 < t.dimensions.size() ? "," : "");
    std::printf("] bytes=%llu\n", (unsigned long long)t.size_bytes);
  }
  return 0;
}
