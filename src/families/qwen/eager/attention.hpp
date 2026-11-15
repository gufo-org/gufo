#ifndef GUFO_FAMILIES_QWEN_EAGER_ATTENTION_HPP_
#define GUFO_FAMILIES_QWEN_EAGER_ATTENTION_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>

namespace gufo::qwen::eager {

/// Keys and values split across up to three head-major segments (for example
/// cached condition rows and the rows of the current pass). Logical key rows
/// of segment i are [end[i-1], end[i]), stored at a per-head stride of
/// stride[i] rows. Unused trailing segments repeat the previous end.
struct KeySegments {
  const std::uint16_t* keys[3]{};
  const std::uint16_t* values[3]{};
  int end[3]{};
  int stride[3]{};
};

/// Fused BF16 attention for head dimension 128: online softmax with BF16 QK
/// matrix instructions and an FP32 value product, no probability narrowing.
/// Queries are head-major `[heads][rows][128]`; the output is row-major
/// `[query][heads][128]`. `limits[q]`, when given, is the number of visible
/// keys of query q (causal masks). Query head h reads key head
/// `h / kv_group`; `kv_group` is 1 or 2. With `bfloat16_probabilities` the
/// segmented form rounds the probabilities to BF16 and runs the value product
/// on matrix instructions as well, as PyTorch's fused BF16 attention does.
void FusedAttention128(const std::uint16_t* queries, const std::uint16_t* keys,
                       const std::uint16_t* values, std::uint16_t* output,
                       int heads, int rows, int count, int query_start,
                       hipStream_t stream, int query_count = 128,
                       const int* limits = nullptr, int kv_group = 1);
void FusedAttention128Segmented(const std::uint16_t* queries,
                                const KeySegments& segments,
                                std::uint16_t* output, int heads, int rows,
                                int query_start, int query_count,
                                hipStream_t stream, const int* limits,
                                int kv_group = 1,
                                bool bfloat16_probabilities = false);

}  // namespace gufo::qwen::eager

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_FAMILIES_QWEN_EAGER_ATTENTION_HPP_
