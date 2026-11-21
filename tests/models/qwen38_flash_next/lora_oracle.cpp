// Real-weight LoRA oracle: merges an adapter into the stock trunk on the CPU
// with the production routine and compares every patched tensor byte for
// byte with an offline merge of the same checkpoint. Reads only; skips
// (exit 77) when a file is absent.
//
//   qwen38_flash_next_lora_oracle STOCK.gguf ADAPTER.gguf MERGED.gguf [SCALE]
//
// STOCK and MERGED name the first shard of each split checkpoint.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/lora.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "usage: " << argv[0]
              << " STOCK.gguf ADAPTER.gguf MERGED.gguf [SCALE]\n";
    return 2;
  }
  for (int i = 1; i < 4; ++i) {
    if (!std::filesystem::exists(argv[i])) {
      std::cout << "SKIP: " << argv[i] << " is absent\n";
      return 77;
    }
  }
  const float scale = argc == 5 ? std::strtof(argv[4], nullptr) : 1.0F;
  std::string error;
  const auto stock = gufo::core::GgufReader::OpenFile(argv[1], &error);
  const auto merged =
      stock ? gufo::core::GgufReader::OpenFile(argv[3], &error) : nullptr;
  if (!merged) {
    std::cerr << "open failed: " << error << '\n';
    return 1;
  }
  const auto weights = qfn::ModelWeights::Bind(*stock, &error);
  if (!weights) {
    std::cerr << "bind failed: " << error << '\n';
    return 1;
  }
  const auto targets = qfn::LoraTargets(*weights);
  const auto adapter = qfn::LoraAdapter::Open(
      argv[2], scale,
      stock->GetMetadataString("general.architecture").value_or(""), targets,
      &error);
  if (!adapter) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "adapter sha256 " << adapter->sha256() << ", " << adapter->size()
            << " patches, identity " << adapter->CacheIdentity() << '\n';

  std::uint64_t tensors = 0, compared = 0, mismatched = 0, changed = 0;
  std::chrono::duration<double> merge_time{};
  std::vector<std::uint8_t> buffer;
  for (const auto& target : targets) {
    const auto* patch = adapter->Find(target.name);
    if (patch == nullptr) {
      continue;
    }
    const auto* reference = merged->FindTensor(target.name);
    if (reference == nullptr || reference->type != target.type ||
        reference->size_bytes != target.SizeBytes()) {
      std::cerr << target.name << ": missing or reshaped in the merged file\n";
      return 1;
    }
    const std::size_t row_bytes = target.RowBytes();
    const std::uint64_t chunk =
        std::max<std::uint64_t>(1, (4 << 20) / row_bytes);
    std::uint64_t tensor_mismatch = 0;
    for (std::uint64_t row = 0; row < target.rows; row += chunk) {
      const std::uint64_t count = std::min(chunk, target.rows - row);
      buffer.resize(count * row_bytes);
      const auto start = std::chrono::steady_clock::now();
      if (!adapter->MergeRows(*patch, row, count, buffer.data(), &error)) {
        std::cerr << error << '\n';
        return 1;
      }
      merge_time += std::chrono::steady_clock::now() - start;
      const auto* expected =
          static_cast<const std::uint8_t*>(reference->data) + row * row_bytes;
      const auto* original =
          static_cast<const std::uint8_t*>(target.data) + row * row_bytes;
      for (std::size_t i = 0; i < buffer.size(); ++i) {
        tensor_mismatch += buffer[i] != expected[i];
        changed += buffer[i] != original[i];
      }
    }
    if (tensor_mismatch != 0) {
      std::cout << "MISMATCH " << target.name << ": " << tensor_mismatch
                << " of " << target.SizeBytes() << " bytes\n";
    }
    ++tensors;
    compared += target.SizeBytes();
    mismatched += tensor_mismatch;
  }
  std::cout << tensors << " tensors, " << compared << " bytes compared, "
            << mismatched << " mismatching, " << changed
            << " bytes differ from stock; merge " << merge_time.count()
            << " s\n";
  return mismatched == 0 && tensors == adapter->size() ? 0 : 1;
}
