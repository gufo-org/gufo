// Times the narrow (decode / MTP verify) attention on synthetic caches and
// selections: rows 1..8 at the given depths, 512 four-token blocks per row,
// neighbouring rows mostly sharing them. Also times the block selection
// (indexer scores and top-k marks) of the same rows and of a 512-row prefill
// chunk. Prints FNV-1a hashes of the attention output, the selection masks
// and the selection scores.
//
// decode_attn_bench [reps] [depths...]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kSelected = 512;
constexpr std::uint32_t kSplits = 8;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

template<typename T>
std::uint64_t Hash(const T* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(T));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  std::vector<std::uint32_t> depths;
  for (int i = 2; i < argc; ++i)
    depths.push_back(static_cast<std::uint32_t>(std::atoi(argv[i])));
  if (depths.empty())
    depths = {32768};
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  constexpr std::uint32_t kMaxRows = 8;
  for (const std::uint32_t depth : depths) {
    const std::uint32_t n_kv = depth + kMaxRows;
    const std::size_t kv_count = std::size_t{n_kv} * kKvHeads * kDim;
    std::vector<float> qh(std::size_t{kMaxRows} * kHeads * kDim);
    std::uint32_t s = 0x1234ABCDU;
    for (auto& v : qh)
      v = 4.0F * (static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
    std::vector<float> kh(kv_count);
    std::vector<float> vh(kv_count);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F;
      vh[i] = static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F;
    }
    const std::uint32_t blocks = (n_kv + kRatio - 1) / kRatio;
    const std::uint32_t mask_words = (blocks + 31) / 32;
    std::vector<std::uint32_t> mask(std::size_t{kMaxRows} * mask_words);
    std::vector<std::uint32_t> base;
    for (std::uint32_t t = 0; t < kMaxRows; ++t) {
      const std::uint32_t complete = (depth + t + 1) / kRatio;
      if (t == 0)
        for (std::uint32_t j = 0; j < kSelected; ++j)
          base.push_back(Next(&s) % complete);
      for (std::uint32_t b : base) {
        if (Next(&s) % 5 == 0)
          b = Next(&s) % complete;
        mask[std::size_t{t} * mask_words + b / 32] |= 1U << (b % 32);
      }
    }
    float* d_q = Upload(qh);
    // The production writer fills the 8-bit caches.
    std::uint8_t* d_k = Upload(std::vector<std::uint8_t>(
        std::size_t{n_kv} *
        gufo::models::qwen38_flash_next::KvRowBytes(kKvHeads * kDim)));
    std::uint8_t* d_v = Upload(std::vector<std::uint8_t>(
        std::size_t{n_kv} *
        gufo::models::qwen38_flash_next::KvRowBytes(kKvHeads * kDim)));
    {
      float* rows = Upload(kh);
      std::uint32_t* zero = Upload(std::vector<std::uint32_t>{0});
      q::StoreKv(rows, d_k, n_kv, kKvHeads * kDim, zero, nullptr);
      CheckHip(hipDeviceSynchronize(), "store keys");
      CheckHip(hipMemcpy(rows, vh.data(), vh.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "value rows");
      q::StoreKv(rows, d_v, n_kv, kKvHeads * kDim, zero, nullptr);
      CheckHip(hipDeviceSynchronize(), "store values");
      CheckHip(hipFree(rows), "hipFree");
      CheckHip(hipFree(zero), "hipFree");
    }
    std::uint32_t* d_mask = Upload(mask);
    std::uint32_t* d_pos = Upload(std::vector<std::uint32_t>{depth});
    float* d_out = Upload(std::vector<float>(qh.size()));
    float* d_part = Upload(std::vector<float>(std::size_t{kMaxRows} * kHeads *
                                              kSplits * (kDim + 2)));
    std::uint32_t* d_lists =
        Upload(std::vector<std::uint32_t>(std::size_t{kMaxRows} * kSelected));
    std::uint32_t* d_counts = Upload(std::vector<std::uint32_t>(kMaxRows));
    for (const std::uint32_t rows : {1u, 2u, 4u, 8u}) {
      const auto run = [&] {
        q::Attention(d_q, d_k, d_v, d_mask, mask_words, d_out, d_part, kSplits,
                     rows, d_pos, kHeads, kKvHeads, kDim, kRatio, nullptr,
                     d_lists, d_counts, kSelected);
      };
      run();
      CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
      const std::uint64_t hash = Hash(d_out, std::size_t{rows} * kHeads * kDim);
      std::vector<float> us(reps);
      for (int i = 0; i < reps; ++i) {
        CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
        run();
        CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
        CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
        CheckHip(hipEventElapsedTime(&us[i], start, stop),
                 "hipEventElapsedTime");
        us[i] *= 1000.0F;
      }
      std::sort(us.begin(), us.end());
      std::printf("depth %6u rows %u  %8.1f us  hash %016llx\n", depth, rows,
                  us[reps / 2], static_cast<unsigned long long>(hash));
    }
    // Block selection: four F32 indexer heads of 128 dims per row against
    // one F16 key per four-token block.
    constexpr std::uint32_t kIndexHeads = 4;
    constexpr std::uint32_t kIndexDim = 128;
    constexpr std::uint32_t kSelectRows = 512;  // the prefill score chunk
    std::vector<float> iq(std::size_t{kSelectRows} * kIndexHeads * kIndexDim);
    for (auto& v : iq)
      v = static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F;
    // The prefill chunk's last row completes (depth + kSelectRows) / kRatio
    // blocks. Size its rows as the executor sizes a wide batch: by the
    // populated context, in whole cache lines.
    const std::uint32_t wide_live = (depth + kSelectRows) / kRatio;
    const std::uint32_t wide_blocks =
        ((depth + kSelectRows + kRatio - 1) / kRatio + 31) / 32 * 32;
    const std::uint32_t wide_words = wide_blocks / 32;
    std::vector<__half> bk(std::size_t{wide_blocks} * kIndexDim);
    for (auto& v : bk)
      v = __float2half(static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
    float* d_iq = Upload(iq);
    __half* d_bk = Upload(bk);
    float* d_scores =
        Upload(std::vector<float>(std::size_t{kSelectRows} * wide_blocks));
    std::uint32_t* d_sel = Upload(
        std::vector<std::uint32_t>(std::size_t{kSelectRows} * wide_words));
    for (const std::uint32_t rows : {1u, 2u, 4u, 8u, kSelectRows}) {
      const bool wide = rows == kSelectRows;
      const std::uint32_t row_blocks = wide ? wide_blocks : blocks;
      const std::uint32_t row_words = wide ? wide_words : mask_words;
      const auto run = [&] {
        q::SelectBlocks(d_iq, d_bk, d_sel, d_scores, rows, d_pos, 0,
                        kIndexHeads, kIndexDim, kRatio, kSelected, row_words,
                        row_blocks, nullptr, wide ? wide_live : blocks);
      };
      run();
      CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
      const std::uint64_t hash = Hash(d_sel, std::size_t{rows} * row_words);
      const std::uint64_t score_hash =
          Hash(d_scores, std::size_t{rows} * row_blocks);
      std::vector<float> us(reps);
      for (int i = 0; i < reps; ++i) {
        CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
        run();
        CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
        CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
        CheckHip(hipEventElapsedTime(&us[i], start, stop),
                 "hipEventElapsedTime");
        us[i] *= 1000.0F;
      }
      std::sort(us.begin(), us.end());
      std::printf(
          "depth %6u rows %3u  select %8.1f us  hash %016llx  scores "
          "%016llx\n",
          depth, rows, us[reps / 2], static_cast<unsigned long long>(hash),
          static_cast<unsigned long long>(score_hash));
    }
    for (void* p : {static_cast<void*>(d_iq), static_cast<void*>(d_bk),
                    static_cast<void*>(d_scores), static_cast<void*>(d_sel)})
      CheckHip(hipFree(p), "hipFree");
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_k),
                    static_cast<void*>(d_v), static_cast<void*>(d_mask),
                    static_cast<void*>(d_pos), static_cast<void*>(d_out),
                    static_cast<void*>(d_part), static_cast<void*>(d_lists),
                    static_cast<void*>(d_counts)})
      CheckHip(hipFree(p), "hipFree");
  }
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
