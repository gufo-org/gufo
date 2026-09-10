#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_decode_attention.hip.hpp"
#include "src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_indexer_score.hip.hpp"
#include "src/models/deepseek_v4_flash/runtime/dspark_internal.h"

namespace {
void check(hipError_t error) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(error));
    std::exit(1);
  }
}

// Historical production kernel, independent of the production implementation.
// A sampled CPU double calculation below also checks the scoring formula.
__global__ static void reference_indexer_scores(float* scores, const float* q,
                                                const float* weights,
                                                const float* index_comp,
                                                uint32_t n_comp, uint32_t pos0,
                                                uint32_t ratio, float scale,
                                                int causal) {
  const uint32_t token = blockIdx.y;
  scores += (uint64_t)token * n_comp;
  q += (uint64_t)token * 64u * 128u;
  weights += (uint64_t)token * 64u;
  pos0 += token;
  const uint32_t c = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;
  if (c >= n_comp || tid >= 128u)
    return;
  if (causal) {
    const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
    if (c >= visible) {
      if (tid == 0)
        scores[c] = -INFINITY;
      return;
    }
  }

  __shared__ float krow[128];
  __shared__ float partial[4];
  if (tid < 128u)
    krow[tid] = index_comp[(uint64_t)c * 128u + tid];
  __syncthreads();

  float total = 0.0f;
  for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
    const uint32_t h = h0 + warp;
    const float4 qv = ((const float4*)(q + (uint64_t)h * 128u))[lane];
    const float4 kv = ((const float4*)krow)[lane];
    float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_down(dot, offset, 32);
    if (lane == 0)
      partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
    __syncthreads();
    if (tid == 0)
      total += partial[0] + partial[1] + partial[2] + partial[3];
    __syncthreads();
  }
  if (tid == 0)
    scores[c] = total;
}

void check_indexer_scores(bool benchmark) {
  constexpr uint32_t heads = 64, dim = 128, ratio = 4;
  constexpr float scale = 0.08838834764831845f;
  std::mt19937 gen(410);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (uint32_t tokens : {1u, 6u}) {
    for (uint32_t rows : {17u, 513u, 1025u, 4096u}) {
      for (bool causal : {false, true}) {
        // The verifier crosses compression boundaries within its block.
        const uint32_t pos0 = (rows - 2) * ratio - 1;
        const uint32_t last_visible = (pos0 + tokens) / ratio;
        std::vector<float> q(tokens * heads * dim), k(rows * dim),
            weights(tokens * heads), ref(tokens * rows), out(ref.size());
        for (auto& v : q)
          v = dist(gen);
        for (auto& v : k)
          v = dist(gen);
        for (auto& v : weights)
          v = dist(gen);
        std::fill_n(q.begin(), dim, 0.f);
        std::copy_n(k.begin(), dim, k.begin() + dim);  // Exact score ties.
        if (causal) {
          std::fill(k.begin() + last_visible * dim, k.end(),
                    std::numeric_limits<float>::quiet_NaN());
        }
        float *dq, *dk, *dw, *dr, *dout;
        check(hipMalloc(&dq, q.size() * sizeof(float)));
        check(hipMalloc(&dk, k.size() * sizeof(float)));
        check(hipMalloc(&dw, weights.size() * sizeof(float)));
        check(hipMalloc(&dr, ref.size() * sizeof(float)));
        check(hipMalloc(&dout, out.size() * sizeof(float)));
        check(hipMemcpy(dq, q.data(), q.size() * sizeof(float),
                        hipMemcpyHostToDevice));
        check(hipMemcpy(dk, k.data(), k.size() * sizeof(float),
                        hipMemcpyHostToDevice));
        check(hipMemcpy(dw, weights.data(), weights.size() * sizeof(float),
                        hipMemcpyHostToDevice));
        reference_indexer_scores<<<dim3(rows, tokens), 128>>>(
            dr, dq, dw, dk, rows, pos0, ratio, scale, causal);
        check(hipGetLastError());
        check(hipMemcpy(ref.data(), dr, ref.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (int repeat = 0; repeat < 2; ++repeat) {
          check(hipMemset(dout, 0xff, out.size() * sizeof(float)));
          indexer_score_one_direct_kernel<<<dim3(rows, tokens), 128>>>(
              dout, dq, dw, dk, rows, pos0, ratio, scale, causal);
          check(hipGetLastError());
          check(hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          if (std::memcmp(ref.data(), out.data(), ref.size() * sizeof(float)) !=
              0) {
            std::fprintf(stderr,
                         "Indexer bitwise mismatch: tokens=%u rows=%u "
                         "causal=%d repeat=%d\n",
                         tokens, rows, causal, repeat);
            std::exit(7);
          }
        }
        for (uint32_t t = 0; t < tokens; ++t) {
          const uint32_t visible = causal ? (pos0 + t + 1) / ratio : rows;
          for (uint32_t c = 0; c < rows; ++c) {
            const float score = out[t * rows + c];
            if ((c < visible && !std::isfinite(score)) ||
                (c >= visible && score != -INFINITY)) {
              std::fprintf(stderr,
                           "Indexer visibility mismatch: token=%u row=%u\n", t,
                           c);
              std::exit(7);
            }
          }
          // Bound CPU work independently of context size; cover the mask edge.
          for (uint32_t c :
               {0u, 1u, 2u, rows / 2, rows - 3, rows - 2, rows - 1}) {
            if (c >= visible)
              continue;
            double expected = 0.0;
            for (uint32_t h = 0; h < heads; ++h) {
              double dot = 0.0;
              for (uint32_t d = 0; d < dim; ++d)
                dot += double(q[(t * heads + h) * dim + d]) * k[c * dim + d];
              expected += std::max(dot, 0.0) * weights[t * heads + h] * scale;
            }
            const double error = std::abs(expected - out[t * rows + c]);
            if (error > 1e-4 + 2e-6 * std::abs(expected)) {
              std::fprintf(
                  stderr,
                  "Indexer CPU formula mismatch: token=%u row=%u error=%g\n", t,
                  c, error);
              std::exit(7);
            }
          }
        }
        if (benchmark) {
          hipEvent_t begin, end;
          check(hipEventCreate(&begin));
          check(hipEventCreate(&end));
          for (int pass = 0; pass < 2; ++pass) {
            float us[2]{};
            for (int order = 0; order < 2; ++order) {
              const bool candidate = (order + pass) % 2;
              auto launch = [&] {
                if (candidate)
                  indexer_score_one_direct_kernel<<<dim3(rows, tokens), 128>>>(
                      dout, dq, dw, dk, rows, pos0, ratio, scale, causal);
                else
                  reference_indexer_scores<<<dim3(rows, tokens), 128>>>(
                      dr, dq, dw, dk, rows, pos0, ratio, scale, causal);
              };
              for (int i = 0; i < 5; ++i)
                launch();
              check(hipEventRecord(begin));
              for (int i = 0; i < 50; ++i)
                launch();
              check(hipEventRecord(end));
              check(hipEventSynchronize(end));
              check(hipEventElapsedTime(&us[candidate], begin, end));
              us[candidate] *= 20.f;
            }
            std::printf(
                "indexer tokens=%u rows=%u causal=%d pass=%d old_us=%.3f "
                "new_us=%.3f\n",
                tokens, rows, causal, pass, us[0], us[1]);
          }
          check(hipEventDestroy(begin));
          check(hipEventDestroy(end));
        }
        for (auto ptr : {dq, dk, dw, dr, dout})
          check(hipFree(ptr));
      }
    }
  }
  std::puts(
      "Indexer: 16 cases passed bitwise twice and sampled CPU formula, "
      "including verifier masks and poisoned rows");
}

// Previous production arithmetic, retained only as the bitwise oracle. The
// independent CPU calculation below additionally checks the attention formula.
__global__ static void reference_attention(float* heads, const float* sinks,
                                           const float* q, const float* raw_kv,
                                           uint32_t n_tokens, uint32_t n_raw,
                                           uint32_t raw_cap, uint32_t raw_start,
                                           uint32_t n_head, uint32_t head_dim) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= n_tokens || h >= n_head)
    return;

  extern __shared__ float sh_scores[];
  const float* qh = q + ((uint64_t)tok * n_head + h) * head_dim;
  const float scale = rsqrtf((float)head_dim);
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    const uint32_t row = (raw_start + r) % raw_cap;
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++)
      dot += qh[d] * kv[d];
    sh_scores[r] = dot * scale;
  }
  __syncthreads();

  __shared__ float partial[256];
  __shared__ float max_s;
  __shared__ float denom;
  float local_max = sinks[h];
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    local_max = fmaxf(local_max, sh_scores[r]);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    max_s = partial[0];
  __syncthreads();

  float den_local = 0.0f;
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    sh_scores[r] = expf(sh_scores[r] - max_s);
    den_local += sh_scores[r];
  }
  partial[threadIdx.x] = den_local;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    denom = partial[0] + expf(sinks[h] - max_s);
  __syncthreads();

  float* oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row = (raw_start + r) % raw_cap;
      acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
    }
    oh[d] = acc / denom;
  }
}

__global__ static void reference_target_attention(
    float* heads, const float* q, const float* raw_kv, const float* comp_kv,
    const float* comp_mask, const float* sinks, uint32_t n_raw,
    uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t use_mask,
    uint32_t n_head, uint32_t head_dim, uint32_t use_vec4) {
  const uint32_t h = (uint32_t)blockIdx.x;
  if (h >= n_head)
    return;
  extern __shared__ float scores[];
  const uint32_t tid = threadIdx.x;
  const uint32_t n_rows = n_raw + n_comp;
  const float* qh = q + (uint64_t)h * head_dim;
  const float scale = rsqrtf((float)head_dim);

  float local_max = sinks[h];
  for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
    const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        s += qh[i] * kv[i];
    }
    s *= scale;
    scores[r] = s;
    local_max = fmaxf(local_max, s);
  }
  for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
    float s = -3.4e38f;
    if (!(use_mask && comp_mask && comp_mask[c] <= -5.0e29f)) {
      const float* kv = comp_kv + (uint64_t)c * head_dim;
      float dot =
          use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
      if (!use_vec4) {
        for (uint32_t i = 0; i < head_dim; i++)
          dot += qh[i] * kv[i];
      }
      s = dot * scale;
      if (use_mask && comp_mask)
        s += comp_mask[c];
    }
    scores[n_raw + c] = s;
    local_max = fmaxf(local_max, s);
  }
  const float max_score = attention_block_max_oldhip_w32(local_max);

  float local_sum = 0.0f;
  for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
    const float w = expf(scores[r] - max_score);
    scores[r] = w;
    local_sum += w;
  }
  if (tid == 0u)
    local_sum += expf(sinks[h] - max_score);
  const float denom = attention_block_sum_oldhip_w32(local_sum);
  const float inv_denom = 1.0f / denom;

  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
      acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
    }
    for (uint32_t c = 0; c < n_comp; c++) {
      acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
    }
    heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
  }
}

__global__ static void reference_indexed_attention(
    float* heads, const float* q, const float* raw_kv, const float* comp_kv,
    const int32_t* topk, const float* sinks, uint32_t n_raw, uint32_t raw_cap,
    uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t pos0,
    uint32_t ratio, uint32_t n_head, uint32_t head_dim, uint32_t use_vec4) {
  const uint32_t h = (uint32_t)blockIdx.x;
  if (h >= n_head)
    return;
  extern __shared__ float scores[];
  __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
  __shared__ uint32_t comp_count_s;
  const uint32_t tid = threadIdx.x;
  const float* qh = q + (uint64_t)h * head_dim;
  const float scale = rsqrtf((float)head_dim);

  uint32_t visible_comp = n_comp;
  if (ratio != 0u) {
    visible_comp = (pos0 + 1u) / ratio;
    if (visible_comp > n_comp)
      visible_comp = n_comp;
  }
  if (tid == 0u) {
    comp_count_s = 0;
    for (uint32_t i = 0;
         i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP; i++) {
      const int32_t ci = topk[i];
      if (ci < 0)
        continue;
      const uint32_t c = (uint32_t)ci;
      if (c < n_comp && c < visible_comp)
        comp_rows[comp_count_s++] = c;
    }
  }
  __syncthreads();
  const uint32_t comp_count = comp_count_s;
  const uint32_t n_rows = n_raw + comp_count;

  float local_max = sinks[h];
  for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
    const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        s += qh[i] * kv[i];
    }
    s *= scale;
    scores[r] = s;
    local_max = fmaxf(local_max, s);
  }
  for (uint32_t c = tid; c < comp_count; c += blockDim.x) {
    const uint32_t row = comp_rows[c];
    const float* kv = comp_kv + (uint64_t)row * head_dim;
    float dot =
        use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        dot += qh[i] * kv[i];
    }
    const float s = dot * scale;
    scores[n_raw + c] = s;
    local_max = fmaxf(local_max, s);
  }
  const float max_score = attention_block_max_oldhip_w32(local_max);

  float local_sum = 0.0f;
  for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
    const float w = expf(scores[r] - max_score);
    scores[r] = w;
    local_sum += w;
  }
  if (tid == 0u)
    local_sum += expf(sinks[h] - max_score);
  const float denom = attention_block_sum_oldhip_w32(local_sum);
  const float inv_denom = 1.0f / denom;

  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row = raw_cap ? ((raw_start + r) % raw_cap) : r;
      acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
    }
    for (uint32_t c = 0; c < comp_count; c++) {
      const uint32_t row = comp_rows[c];
      acc += scores[n_raw + c] * comp_kv[(uint64_t)row * head_dim + d];
    }
    heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
  }
}

void check_target_attention() {
  constexpr unsigned heads = 64, dim = 512, cap = 4352, max_comp = 1024;
  std::mt19937 gen(61);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (unsigned raw : {16u, 128u, 256u}) {
    for (unsigned comp : {0u, 64u, 512u, 1024u}) {
      const bool indexed = comp > 512;
      constexpr unsigned top_k = 512, position = 2049, ratio = 4;
      const unsigned start = cap - raw / 2;
      const unsigned visible = std::min(comp, (position + 1) / ratio);
      std::vector<float> q(heads * dim), kv(cap * dim),
          compressed(max_comp * dim), sinks(heads), mask(comp),
          expected(q.size()), actual(q.size());
      for (auto* values : {&q, &kv, &compressed, &sinks})
        for (auto& value : *values)
          value = dist(gen);
      for (unsigned i = 0; i < comp; ++i)
        mask[i] = i % 7 == 0 ? -1e30f : 0.f;
      std::vector<int32_t> indices(max_comp);
      for (unsigned i = 0; i < max_comp; ++i)
        indices[i] = i;
      std::shuffle(indices.begin(), indices.end(), gen);
      indices.resize(top_k);
      indices[0] = -1;
      indices[1] = max_comp;
      float *dq, *dk, *dc, *ds, *dm, *dref, *dout;
      int32_t* di;
      check(hipMalloc(&dq, q.size() * 4));
      check(hipMalloc(&dk, kv.size() * 4));
      check(hipMalloc(&dc, compressed.size() * 4));
      check(hipMalloc(&ds, sinks.size() * 4));
      check(hipMalloc(&dm, std::max(comp, 1u) * 4));
      check(hipMalloc(&dref, q.size() * 4));
      check(hipMalloc(&dout, q.size() * 4));
      check(hipMalloc(&di, indices.size() * 4));
      check(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
      check(hipMemcpy(dk, kv.data(), kv.size() * 4, hipMemcpyHostToDevice));
      check(hipMemcpy(dc, compressed.data(), compressed.size() * 4,
                      hipMemcpyHostToDevice));
      check(
          hipMemcpy(ds, sinks.data(), sinks.size() * 4, hipMemcpyHostToDevice));
      if (comp)
        check(hipMemcpy(dm, mask.data(), comp * 4, hipMemcpyHostToDevice));
      check(hipMemcpy(di, indices.data(), indices.size() * 4,
                      hipMemcpyHostToDevice));
      const size_t shared = (raw + std::min(comp, top_k)) * sizeof(float);
      if (indexed) {
        reference_indexed_attention<<<heads, 256, shared>>>(
            dref, dq, dk, dc, di, ds, raw, cap, start, comp, top_k, position,
            ratio, heads, dim, 1);
        attention_decode_indexed_mixed_one_fast_oldhip_kernel<<<heads, 256,
                                                                shared>>>(
            dout, dq, dk, dc, di, ds, raw, cap, start, comp, top_k, position,
            ratio, heads, dim, 1);
      } else {
        reference_target_attention<<<heads, 256, shared>>>(
            dref, dq, dk, dc, dm, ds, raw, cap, start, comp, comp != 0, heads,
            dim, 1);
        attention_decode_mixed_one_fast_oldhip_kernel<<<heads, 256, shared>>>(
            dout, dq, dk, dc, dm, ds, raw, cap, start, comp, comp != 0, heads,
            dim, 1);
      }
      check(hipGetLastError());
      check(hipMemcpy(expected.data(), dref, q.size() * 4,
                      hipMemcpyDeviceToHost));
      check(
          hipMemcpy(actual.data(), dout, q.size() * 4, hipMemcpyDeviceToHost));
      if (std::memcmp(expected.data(), actual.data(), q.size() * 4) != 0 ||
          !std::all_of(actual.begin(), actual.end(),
                       [](float v) { return std::isfinite(v); })) {
        std::fprintf(stderr,
                     "target attention scalar mismatch: raw=%u comp=%u\n", raw,
                     comp);
        std::exit(4);
      }
      // Independent double-precision formula, including masks, sparse index
      // ordering, invalid indices, causal visibility, and the attention sink.
      std::vector<const float*> rows;
      for (unsigned i = 0; i < raw; ++i)
        rows.push_back(kv.data() + ((start + i) % cap) * dim);
      if (indexed) {
        for (int32_t i : indices)
          if (i >= 0 && static_cast<unsigned>(i) < visible)
            rows.push_back(compressed.data() + i * dim);
      } else {
        for (unsigned i = 0; i < comp; ++i)
          if (mask[i] > -5e29f)
            rows.push_back(compressed.data() + i * dim);
      }
      for (unsigned head : {0u, heads - 1}) {
        std::vector<double> scores;
        double max_score = sinks[head];
        for (const auto* row : rows) {
          double dot = 0;
          for (unsigned d = 0; d < dim; ++d)
            dot += double(q[head * dim + d]) * row[d];
          scores.push_back(dot / std::sqrt(double(dim)));
          max_score = std::max(max_score, scores.back());
        }
        double denom = std::exp(double(sinks[head]) - max_score);
        for (auto& score : scores) {
          score = std::exp(score - max_score);
          denom += score;
        }
        for (unsigned d = 0; d < dim; ++d) {
          double sum = 0;
          for (size_t i = 0; i < rows.size(); ++i)
            sum += scores[i] * rows[i][d];
          if (std::abs(sum / denom - actual[head * dim + d]) > 1e-5) {
            std::fprintf(stderr,
                         "target attention independent formula mismatch\n");
            std::exit(5);
          }
        }
      }
      for (auto ptr : {dq, dk, dc, ds, dm, dref, dout})
        check(hipFree(ptr));
      check(hipFree(di));
    }
  }
  std::puts(
      "Target attention: 12 exact GPU comparisons and independent "
      "double-precision checks passed");
}

void check_dspark_window() {
  constexpr unsigned heads = 64, dim = 512, cap = 4352;
  std::mt19937 gen(73);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (unsigned prefix : {1u, 127u, 128u, 129u, 4096u, 4351u, 4352u, 8707u}) {
    for (unsigned block : {1u, 5u}) {
      std::vector<float> q(block * heads * dim), sinks(heads), out(q.size()),
          kv(cap * dim, NAN);
      for (auto* values : {&q, &sinks})
        for (auto& value : *values)
          value = dist(gen);

      // Build the official logical sequence independently of the production
      // span helper: last 128 target rows followed immediately by every draft
      // row. Poison all other physical slots, including the old encoder slot.
      std::vector<std::vector<float>> rows;
      const unsigned first = prefix > 128 ? prefix - 128 : 0;
      for (unsigned pos = first; pos < prefix + block; ++pos) {
        auto& row = rows.emplace_back(dim);
        for (unsigned d = 0; d < dim; ++d)
          row[d] = dist(gen);
        std::copy(row.begin(), row.end(), kv.begin() + (pos % cap) * dim);
      }
      const auto span = ds4_dspark_visible_span(prefix, block, cap);
      float *dq, *dk, *ds, *dout;
      check(hipMalloc(&dq, q.size() * sizeof(float)));
      check(hipMalloc(&dk, kv.size() * sizeof(float)));
      check(hipMalloc(&ds, sinks.size() * sizeof(float)));
      check(hipMalloc(&dout, out.size() * sizeof(float)));
      check(hipMemcpy(dq, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      check(hipMemcpy(dk, kv.data(), kv.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      check(hipMemcpy(ds, sinks.data(), sinks.size() * sizeof(float),
                      hipMemcpyHostToDevice));
      attention_noncausal_raw_batch_heads_kernel<<<
          dim3(block, heads), 256, span.count * sizeof(float)>>>(
          dout, ds, dq, dk, block, span.count, cap, span.start, heads, dim);
      check(hipGetLastError());
      check(hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      double max_error = 0;
      for (unsigned token : {0u, block - 1})
        for (unsigned head : {0u, heads - 1}) {
          const size_t offset = (token * heads + head) * dim;
          std::vector<double> scores;
          double maximum = sinks[head];
          for (const auto& row : rows) {
            double dot = 0;
            for (unsigned d = 0; d < dim; ++d)
              dot += double(q[offset + d]) * row[d];
            scores.push_back(dot / std::sqrt(double(dim)));
            maximum = std::max(maximum, scores.back());
          }
          double denominator = std::exp(double(sinks[head]) - maximum);
          for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (unsigned d = 0; d < dim; ++d) {
            double sum = 0;
            for (size_t r = 0; r < rows.size(); ++r)
              sum += scores[r] * rows[r][d];
            max_error = std::max(max_error,
                                 std::abs(sum / denominator - out[offset + d]));
          }
        }
      if (max_error > 1e-5 || !std::all_of(out.begin(), out.end(), [](float v) {
            return std::isfinite(v);
          })) {
        std::fprintf(
            stderr,
            "DSpark official window mismatch: prefix=%u block=%u error=%g\n",
            prefix, block, max_error);
        std::exit(6);
      }
      for (auto ptr : {dq, dk, ds, dout})
        check(hipFree(ptr));
    }
  }
  std::puts(
      "DSpark window: 16 official-formula cases passed, including ring wrap "
      "and poisoned stale rows");
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::puts("SKIP: no HIP device for DSpark support attention");
    return 77;
  }
  const bool benchmark =
      argc == 2 && std::string_view(argv[1]) == "--benchmark";
  if (argc > 1 && !benchmark) {
    std::fprintf(stderr, "usage: ds4_attention_test [--benchmark]\n");
    return 2;
  }

  check_indexer_scores(benchmark);
  check_target_attention();
  check_dspark_window();
  constexpr unsigned heads = 64, dim = 512, cap = 4352;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (unsigned raw : {1u, 128u, 4096u, 4352u})
    for (unsigned tokens : {1u, 2u, 3u, 5u}) {
      const unsigned start = cap - std::max(raw / 2, 1u);
      std::vector<float> q(tokens * heads * dim), kv(cap * dim), sinks(heads),
          ref(q.size()), out(q.size());
      for (auto& x : q)
        x = dist(gen);
      for (auto& x : kv)
        x = std::round(dist(gen) * 16.f) / 16.f;
      for (auto& x : sinks)
        x = dist(gen);
      float *dq, *dk, *ds, *dr, *dout;
      check(hipMalloc(&dq, q.size() * 4));
      check(hipMalloc(&dk, kv.size() * 4));
      check(hipMalloc(&ds, sinks.size() * 4));
      check(hipMalloc(&dr, ref.size() * 4));
      check(hipMalloc(&dout, out.size() * 4));
      check(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
      check(hipMemcpy(dk, kv.data(), kv.size() * 4, hipMemcpyHostToDevice));
      check(
          hipMemcpy(ds, sinks.data(), sinks.size() * 4, hipMemcpyHostToDevice));
      auto launch = [&](bool candidate) {
        if (candidate)
          attention_noncausal_raw_batch_heads_kernel<<<dim3(tokens, heads), 256,
                                                       raw * 4>>>(
              dout, ds, dq, dk, tokens, raw, cap, start, heads, dim);
        else
          reference_attention<<<dim3(tokens, heads), 256, raw * 4>>>(
              dr, ds, dq, dk, tokens, raw, cap, start, heads, dim);
        check(hipGetLastError());
      };
      launch(false);
      launch(true);
      check(hipDeviceSynchronize());
      check(hipMemcpy(ref.data(), dr, ref.size() * 4, hipMemcpyDeviceToHost));
      check(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
      double squared = 0;
      float maxerr = 0;
      for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i]) || !std::isfinite(ref[i]))
          return 2;
        const float err = std::abs(out[i] - ref[i]);
        maxerr = std::max(maxerr, err);
        squared += double(err) * err;
      }
      const double rmse = std::sqrt(squared / out.size());
      double oracle_err = 0;
      for (unsigned token : {0u, tokens - 1})
        for (unsigned head : {0u, heads - 1}) {
          std::vector<double> scores(raw);
          double maxscore = sinks[head];
          for (unsigned r = 0; r < raw; ++r) {
            const unsigned row = (start + r) % cap;
            double dot = 0;
            for (unsigned d = 0; d < dim; ++d)
              dot += double(q[(token * heads + head) * dim + d]) *
                     kv[row * dim + d];
            scores[r] = dot / std::sqrt(double(dim));
            maxscore = std::max(maxscore, scores[r]);
          }
          double denom = std::exp(double(sinks[head]) - maxscore);
          for (auto& score : scores) {
            score = std::exp(score - maxscore);
            denom += score;
          }
          for (unsigned d = 0; d < dim; ++d) {
            double sum = 0;
            for (unsigned r = 0; r < raw; ++r)
              sum += scores[r] * kv[((start + r) % cap) * dim + d];
            oracle_err = std::max(
                oracle_err,
                std::abs(sum / denom - out[(token * heads + head) * dim + d]));
          }
        }
      if (std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) !=
              0 ||
          oracle_err > 1e-5) {
        std::fprintf(stderr,
                     "support attention mismatch: raw=%u rows=%u max_error=%g "
                     "oracle_error=%g\n",
                     raw, tokens, maxerr, oracle_err);
        return 3;
      }
      if (benchmark) {
        float times[2]{};
        hipEvent_t begin, end;
        check(hipEventCreate(&begin));
        check(hipEventCreate(&end));
        constexpr int reps = 10;
        for (int candidate = 0; candidate < 2; ++candidate) {
          check(hipEventRecord(begin));
          for (int i = 0; i < reps; ++i)
            launch(candidate);
          check(hipEventRecord(end));
          check(hipEventSynchronize(end));
          check(hipEventElapsedTime(&times[candidate], begin, end));
          times[candidate] /= reps;
        }
        std::printf(
            "raw=%u rows=%u old_ms=%.4f wrapped_ms=%.4f speedup=%.2f "
            "max_error=%g rmse=%g cpu_double_error=%g\n",
            raw, tokens, times[0], times[1], times[0] / times[1], maxerr, rmse,
            oracle_err);
        std::fflush(stdout);
        check(hipEventDestroy(begin));
        check(hipEventDestroy(end));
      }
      for (auto p : {dq, dk, ds, dr, dout})
        check(hipFree(p));
    }
  std::puts(
      "DSpark support attention: 16 exact GPU comparisons and independent "
      "double-precision checks passed");
}
