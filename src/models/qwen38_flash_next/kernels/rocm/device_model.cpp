#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <utility>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// Whether every trunk layer's GDN or attention heads split evenly across
/// `world` ranks with whole query groups, and every piece is whole rows or
/// whole quantization blocks of Q8_0 projections.
bool SplitMixersFit(const ModelWeights& w, std::uint32_t world,
                    std::string* why) {
  const Config& c = w.config;
  if (c.ssm_num_k_heads == 0 || c.ssm_num_k_heads % world != 0 ||
      c.ssm_num_v_heads % c.ssm_num_k_heads != 0 || c.num_kv_heads == 0 ||
      c.num_kv_heads % world != 0 || c.num_heads % c.num_kv_heads != 0 ||
      c.ssm_head_dim % 32 != 0 || c.head_dim % 32 != 0) {
    *why = "head counts";
    return false;
  }
  const auto q8 = [](const TensorRef& t) {
    return !t.empty() && t.type == core::GgmlType::kQ8_0;
  };
  for (const auto& l : w.layers) {
    const bool fits =
        l.linear ? q8(l.ssm_qkv) && q8(l.ssm_gate) && q8(l.ssm_out) &&
                       l.ssm_qkv.cols == l.ssm_gate.cols &&
                       (l.ssm_alpha.type == core::GgmlType::kF32 ||
                        l.ssm_alpha.type == core::GgmlType::kQ8_0) &&
                       l.ssm_beta.type == l.ssm_alpha.type &&
                       l.ssm_beta.cols == l.ssm_alpha.cols &&
                       l.ssm_conv1d.type == core::GgmlType::kF32 &&
                       l.ssm_dt.type == core::GgmlType::kF32 &&
                       l.ssm_a.type == core::GgmlType::kF32
                 : q8(l.attn_q) && q8(l.attn_k) && q8(l.attn_v) &&
                       q8(l.attn_out) && l.attn_q.cols == l.attn_k.cols &&
                       l.attn_q.cols == l.attn_v.cols;
    if (!fits) {
      *why = "projection types of a trunk layer";
      return false;
    }
  }
  return true;
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  const distributed::TpPartition* partition{nullptr};
  const Config* config{nullptr};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor CopyRange(const TensorRef& t, std::uint64_t relative_offset,
                         std::size_t size, std::uint32_t experts) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t tensor_bytes = t.SizeBytes();
    if (size > tensor_bytes || relative_offset > tensor_bytes - size ||
        size > std::numeric_limits<std::size_t>::max() - kTailMargin ||
        t.cols > std::numeric_limits<std::uint32_t>::max() ||
        t.rows > std::numeric_limits<std::uint32_t>::max()) {
      Fail("weight range geometry is invalid for " + std::string(t.name));
      return d;
    }
    if (relative_offset >
        std::numeric_limits<std::uint64_t>::max() - t.file_offset) {
      Fail("weight range offset overflow for " + std::string(t.name));
      return d;
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset + relative_offset,
                     size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = experts;
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  DeviceTensor Copy(const TensorRef& t) {
    if (t.empty() || !ok) {
      return {};
    }
    return CopyRange(t, 0, t.SizeBytes(),
                     static_cast<std::uint32_t>(t.experts));
  }

  /// A gate or up projection: this rank's rows of every expert. Each share is
  /// a contiguous range of the file, so only it is read.
  DeviceTensor CopyExpertRows(const TensorRef& t) {
    if (t.empty() || !ok || partition == nullptr) {
      return Copy(t);
    }
    const std::size_t row_bytes = t.RowBytes();
    if (t.rows != partition->expert_ff || row_bytes == 0 ||
        t.experts > std::numeric_limits<std::uint32_t>::max() ||
        t.cols > std::numeric_limits<std::uint32_t>::max()) {
      Fail("routed tensor does not match the TP split: " + std::string(t.name));
      return {};
    }
    const std::size_t share = partition->ff_count * row_bytes;
    const std::size_t size = share * t.experts;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return {};
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    for (std::uint64_t e = 0; e < t.experts; ++e) {
      const std::uint64_t first_row = e * t.rows + partition->ff_begin;
      if (!stager.Copy(shard_base + t.shard,
                       t.file_offset + first_row * row_bytes, share,
                       static_cast<std::uint8_t*>(ptr) + e * share, error)) {
        Fail("upload failed for " + std::string(t.name) +
             (error != nullptr ? ": " + *error : std::string()));
        return {};
      }
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    DeviceTensor d;
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = partition->ff_count;
    d.experts = static_cast<std::uint32_t>(t.experts);
    return d;
  }

  /// A down projection: this rank's columns of every expert's rows.
  DeviceTensor CopyExpertColumns(const TensorRef& t) {
    if (t.empty() || !ok || partition == nullptr) {
      return Copy(t);
    }
    if (t.cols != partition->expert_ff) {
      Fail("routed tensor does not match the TP split: " + std::string(t.name));
      return {};
    }
    if (SpanBytes(t.type, partition->ff_count) == 0) {
      Fail("routed tensor share is not whole quantization blocks: " +
           std::string(t.name) + ": " + std::to_string(partition->ff_count) +
           " columns must be a multiple "
           "of the quantization block size (32 for Q8_0 and Q5_1, 256 for "
           "Q4_K and Q5_K)");
      return {};
    }
    return CopyColumns(t, partition->ff_begin, partition->ff_count);
  }

  /// Bytes of `elements` consecutive values of one row, or zero when the
  /// count is not a whole number of quantization blocks.
  static std::size_t SpanBytes(core::GgmlType type, std::size_t elements) {
    switch (type) {
      case core::GgmlType::kF32:
        return elements * 4;
      case core::GgmlType::kF16:
      case core::GgmlType::kBF16:
        return elements * 2;
      default:
        return gufo::quant::QuantizedRowBytes(type, elements);
    }
  }

  /// Rows [begin, begin + count) of a matrix.
  DeviceTensor CopyRows(const TensorRef& t, std::uint32_t begin,
                        std::uint32_t count) {
    const std::size_t row_bytes = t.RowBytes();
    DeviceTensor d = CopyRange(t, static_cast<std::uint64_t>(begin) * row_bytes,
                               static_cast<std::size_t>(count) * row_bytes, 1);
    d.rows = count;
    return d;
  }

  /// Columns [begin, begin + count) of every row (of every stacked expert),
  /// split on a quantization-block boundary without dequantizing or changing
  /// any weight.
  DeviceTensor CopyColumns(const TensorRef& t, std::uint32_t begin,
                           std::uint32_t count) {
    return GatherColumns(t, {{begin, count}});
  }

  /// The listed column ranges of every row, side by side in order, each on a
  /// quantization-block boundary.
  DeviceTensor GatherColumns(
      const TensorRef& t,
      const std::vector<std::pair<std::uint32_t, std::uint32_t>>& ranges) {
    const std::size_t row_bytes = t.RowBytes();
    std::size_t part_row = 0;
    std::uint32_t cols = 0;
    for (const auto& [begin, count] : ranges) {
      const std::size_t span = SpanBytes(t.type, count);
      if ((begin != 0 && SpanBytes(t.type, begin) == 0) || span == 0 ||
          begin + count > t.cols) {
        Fail("column ranges are not whole quantization blocks: " +
             std::string(t.name));
        return {};
      }
      part_row += span;
      cols += count;
    }
    DeviceTensor full = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("column split upload failed for " + std::string(t.name));
      return {};
    }
    DeviceTensor d = full;
    d.cols = cols;
    const std::size_t rows = t.rows * t.experts;
    const std::size_t part_bytes = part_row * rows;
    if (hipMalloc(&d.data, part_bytes + kTailMargin) != hipSuccess) {
      Fail("column split allocation failed for " + std::string(t.name));
      return {};
    }
    allocations.push_back(d.data);
    bytes += part_bytes + kTailMargin;
    std::size_t at = 0;
    for (const auto& [begin, count] : ranges) {
      const std::size_t span = SpanBytes(t.type, count);
      if (hipMemcpy2D(static_cast<std::uint8_t*>(d.data) + at, part_row,
                      static_cast<const std::uint8_t*>(full.data) +
                          (begin == 0 ? 0 : SpanBytes(t.type, begin)),
                      row_bytes, span, rows,
                      hipMemcpyDeviceToDevice) != hipSuccess) {
        Fail("column split copy failed for " + std::string(t.name));
        return {};
      }
      at += span;
    }
    if (hipMemset(static_cast<std::uint8_t*>(d.data) + part_bytes, 0,
                  kTailMargin) != hipSuccess) {
      Fail("column split copy failed for " + std::string(t.name));
      return {};
    }
    std::erase(allocations, full.data);
    (void)hipFree(full.data);
    bytes -= t.SizeBytes() + kTailMargin;
    return d;
  }

  /// A contiguous range of a matrix's rows, or of a vector's elements.
  struct Span {
    const TensorRef* t;
    std::uint64_t begin;
    std::uint64_t count;
  };

  /// Rows of matrices of one type and width, in the listed order, read
  /// straight from the files. With `narrow`, F32 rows become F16 as in Stack.
  DeviceTensor GatherRows(const std::vector<Span>& spans, bool narrow) {
    DeviceTensor d;
    if (!ok || spans.empty()) {
      return d;
    }
    const TensorRef& first = *spans.front().t;
    const std::size_t row_bytes = first.RowBytes();
    std::uint64_t rows = 0;
    for (const Span& span : spans) {
      if (span.t->empty() || span.t->type != first.type ||
          span.t->cols != first.cols || span.t->experts != 1 ||
          span.begin + span.count > span.t->rows ||
          (narrow && first.type != core::GgmlType::kF32)) {
        Fail("row gather needs rows of one type and width: " +
             std::string(span.t->name));
        return d;
      }
      rows += span.count;
    }
    const std::size_t size = rows * row_bytes;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for gathered tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const Span& span : spans) {
      if (!stager.Copy(shard_base + span.t->shard,
                       span.t->file_offset + span.begin * row_bytes,
                       span.count * row_bytes,
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(span.t->name));
        return d;
      }
      offset += span.count * row_bytes;
    }
    d.type = first.type;
    d.cols = static_cast<std::uint32_t>(first.cols);
    d.rows = static_cast<std::uint32_t>(rows);
    if (!narrow) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      if (d.type == core::GgmlType::kQ8_0) {
        max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      }
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for gathered tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  /// Elements of an F32 vector in the listed order.
  DeviceTensor GatherElements(const std::vector<Span>& spans) {
    DeviceTensor d;
    if (!ok || spans.empty()) {
      return d;
    }
    std::uint64_t count = 0;
    for (const Span& span : spans) {
      if (span.t->empty() || span.t->type != core::GgmlType::kF32 ||
          span.t->rows != 1 || span.begin + span.count > span.t->cols) {
        Fail("element gather needs an F32 vector: " +
             std::string(span.t->name));
        return d;
      }
      count += span.count;
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, count * sizeof(float) + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for gathered vector");
      return d;
    }
    allocations.push_back(ptr);
    bytes += count * sizeof(float) + kTailMargin;
    std::size_t offset = 0;
    for (const Span& span : spans) {
      if (!stager.Copy(shard_base + span.t->shard,
                       span.t->file_offset + span.begin * sizeof(float),
                       span.count * sizeof(float),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(span.t->name));
        return d;
      }
      offset += span.count * sizeof(float);
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + offset, 0,
                         kTailMargin, nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(count);
    d.rows = 1;
    return d;
  }

  /// This rank's heads of a GDN layer. Value head h reads key head h % k, so
  /// a rank's k/world key heads serve the value heads in v/k groups of k/world.
  void SplitLinear(const LayerWeights& l, DeviceLayer& d) {
    const Config& c = *config;
    const std::uint64_t k = c.ssm_num_k_heads;
    const std::uint64_t dim = c.ssm_head_dim;
    const std::uint64_t k_local = k / partition->world_size;
    const std::uint64_t first = k_local * partition->rank;
    const std::uint64_t groups = c.ssm_num_v_heads / k;
    std::vector<Span> qkv{{&l.ssm_qkv, first * dim, k_local * dim},
                          {&l.ssm_qkv, (k + first) * dim, k_local * dim}};
    std::vector<Span> z, alpha, beta;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out_columns;
    for (std::uint64_t g = 0; g < groups; ++g) {
      const std::uint64_t head = g * k + first;
      qkv.push_back({&l.ssm_qkv, (2 * k + head) * dim, k_local * dim});
      z.push_back({&l.ssm_gate, head * dim, k_local * dim});
      alpha.push_back({&l.ssm_alpha, head, k_local});
      beta.push_back({&l.ssm_beta, head, k_local});
      out_columns.emplace_back(static_cast<std::uint32_t>(head * dim),
                               static_cast<std::uint32_t>(k_local * dim));
    }
    // The convolution runs per channel over the q|k|v projection rows.
    std::vector<Span> conv;
    for (const Span& span : qkv) {
      conv.push_back({&l.ssm_conv1d, span.begin, span.count});
    }
    std::vector<Span> in = qkv;
    in.insert(in.end(), z.begin(), z.end());
    d.ssm_in = GatherRows(in, false);
    std::vector<Span> alpha_beta = alpha;
    alpha_beta.insert(alpha_beta.end(), beta.begin(), beta.end());
    // As in Stack: F32 rows become F16, Q8_0 rows stay as they are.
    d.ssm_alpha_beta =
        GatherRows(alpha_beta, l.ssm_alpha.type == core::GgmlType::kF32);
    d.ssm_conv1d = GatherRows(conv, false);
    std::vector<Span> dt, a;
    for (const Span& span : alpha) {
      dt.push_back({&l.ssm_dt, span.begin, span.count});
      a.push_back({&l.ssm_a, span.begin, span.count});
    }
    d.ssm_dt = GatherElements(dt);
    d.ssm_a = GatherElements(a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = GatherColumns(l.ssm_out, out_columns);
  }

  /// This rank's heads of an attention layer: query head h reads KV head
  /// h / (heads / kv_heads), so each rank keeps whole query groups.
  void SplitAttention(const LayerWeights& l, DeviceLayer& d) {
    const Config& c = *config;
    const std::uint64_t dim = c.head_dim;
    const std::uint64_t kv_local = c.num_kv_heads / partition->world_size;
    const std::uint64_t q_local = kv_local * (c.num_heads / c.num_kv_heads);
    const std::uint64_t kv_first = kv_local * partition->rank;
    const std::uint64_t q_first = q_local * partition->rank;
    // Query rows interleave each head's query and output gate.
    d.attn_qkv = GatherRows({{&l.attn_q, q_first * 2 * dim, q_local * 2 * dim},
                             {&l.attn_k, kv_first * dim, kv_local * dim},
                             {&l.attn_v, kv_first * dim, kv_local * dim}},
                            false);
    d.attn_out =
        CopyColumns(l.attn_out, static_cast<std::uint32_t>(q_first * dim),
                    static_cast<std::uint32_t>(q_local * dim));
  }

  /// Splits the shared expert's intermediate dimension across ranks when
  /// every piece falls on a block boundary; returns false to keep it whole.
  bool SplitSharedExpert(const LayerWeights& l, DeviceLayer& d) {
    if (partition == nullptr || !partition->distributed() ||
        l.shexp_gate.empty() || l.shexp_up.empty() || l.shexp_down.empty()) {
      return false;
    }
    const std::uint64_t ff = l.shexp_gate.rows;
    const std::uint32_t world = partition->world_size;
    if (ff == 0 || ff % world != 0 || l.shexp_up.rows != ff ||
        l.shexp_down.cols != ff) {
      return false;
    }
    // A share of whole blocks puts every rank's first column on a block
    // boundary too.
    const auto share = static_cast<std::uint32_t>(ff / world);
    const auto begin = share * partition->rank;
    if (SpanBytes(l.shexp_down.type, share) == 0) {
      return false;
    }
    d.shexp_gate = CopyRows(l.shexp_gate, begin, share);
    d.shexp_up = CopyRows(l.shexp_up, begin, share);
    d.shexp_down = CopyColumns(l.shexp_down, begin, share);
    d.shexp_split = true;
    return true;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  /// `heads` are the layer's heads on this rank; fewer than all split its
  /// GDN or attention block.
  DeviceLayer Layer(const LayerWeights& l, const MixerHeads& heads,
                    bool split) {
    DeviceLayer d;
    d.linear = l.linear;
    d.heads = heads;
    d.mixer_split = split;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (split && l.linear) {
      SplitLinear(l, d);
    } else if (split) {
      SplitAttention(l, d);
    } else {
      if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
        d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
      } else {
        d.ssm_qkv = Copy(l.ssm_qkv);
        d.ssm_gate = Copy(l.ssm_gate);
      }
      d.ssm_conv1d = Copy(l.ssm_conv1d);
      if (l.linear) {
        d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
      }
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_norm = Copy(l.ssm_norm);
      d.ssm_out = Copy(l.ssm_out);
      if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
        d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
      } else {
        d.attn_q = Copy(l.attn_q);
        d.attn_k = Copy(l.attn_k);
        d.attn_v = Copy(l.attn_v);
      }
      d.attn_out = Copy(l.attn_out);
    }
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = CopyExpertRows(l.ffn_gate_exps);
    d.ffn_up_exps = CopyExpertRows(l.ffn_up_exps);
    d.ffn_down_exps = CopyExpertColumns(l.ffn_down_exps);
    if (!SplitSharedExpert(l, d)) {
      d.shexp_gate = Copy(l.shexp_gate);
      d.shexp_up = Copy(l.shexp_up);
      d.shexp_down = Copy(l.shexp_down);
    }
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg, const distributed::TpPartition* partition) {
  // The CPU reference also reads Q6_K, but the production embedding, dense
  // and routed kernels do not. Reject it before allocating device weights.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  if (partition != nullptr) {
    if (!partition->Valid() || partition->world_size > 2 ||
        partition->expert_ff != w.config.expert_ff) {
      if (error_msg != nullptr) {
        *error_msg = "TP split does not match the model's expert size";
      }
      return nullptr;
    }
    m->tp_rank_ = partition->rank;
    m->tp_world_size_ = partition->world_size;
  }
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,
              conversions,
              m->allocations_,
              m->bytes_,
              m->max_half_cols_,
              m->max_q8_cols_,
              error_msg,
              true,
              0,
              partition,
              &m->config_};
  // Under TP every trunk layer's GDN or attention heads are split across the
  // ranks; the draft block, which runs one short step at a time, is not.
  const MixerHeads all = MixerHeads::All(w.config);
  m->trunk_split_ = partition != nullptr && partition->distributed();
  m->trunk_heads_ = all;
  if (m->trunk_split_) {
    std::string why;
    if (!SplitMixersFit(w, partition->world_size, &why)) {
      if (error_msg != nullptr) {
        *error_msg = "TP head split does not fit the model: " + why;
      }
      return nullptr;
    }
    const std::uint32_t world = partition->world_size;
    m->trunk_heads_.ssm_k /= world;
    m->trunk_heads_.ssm_v /= world;
    m->trunk_heads_.attn /= world;
    m->trunk_heads_.attn_kv /= world;
  }
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l, m->trunk_heads_, m->trunk_split_));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ = up.Layer(mtp->block, all, false);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
