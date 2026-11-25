#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <initializer_list>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/weight_source.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

namespace flash_next = ext::qwen::flash_next;

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
  /// Expands Q8_0 weights to F32 instead of narrowing F32 to F16.
  bool expand_q8{false};
  std::size_t source_bytes{0};
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};
  /// Uploads the tensors with format_state; null for a GGUF upload.
  flash_next::WeightSource* source{nullptr};
  /// Bytes the measuring walk requested: the whole upload.
  std::uint64_t measured{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  /// What a weight source's Upload sees of this walk.
  struct Context final : flash_next::UploadContext {
    Uploader& up;
    explicit Context(Uploader& uploader) : up(uploader) {}
    [[nodiscard]] bool Measuring() const noexcept override {
      return up.Measuring();
    }
    [[nodiscard]] bool Allocate(void** pointer, std::size_t size) override {
      *pointer = nullptr;
      if (up.Allocate(pointer, size) != hipSuccess)
        return false;
      if (!up.Measuring()) {
        up.allocations.push_back(*pointer);
        up.bytes += size;
      }
      return true;
    }
    // The measuring walk copies nothing.
    [[nodiscard]] bool CopyFromRegion(std::uint32_t region,
                                      std::uint64_t offset, std::size_t size,
                                      void* device) override {
      return up.Measuring() || up.stager.Copy(up.shard_base + region, offset,
                                              size, device, up.error);
    }
    [[nodiscard]] bool Finish() override {
      return up.Measuring() || up.stager.Finish(up.error);
    }
    void Fail(std::string_view message) override {
      up.Fail(std::string(message));
    }
  };

  /// Queues one payload copy from a mapped region, or from host memory the
  /// weight source prepared (kHostShard).
  [[nodiscard]] bool Source(const TensorRef& t, std::size_t size,
                            void* device) {
    if (t.shard == kHostShard) {
      if (hipMemcpy(device, t.data, size, hipMemcpyHostToDevice) == hipSuccess)
        return true;
      if (error != nullptr)
        *error = "host tensor copy failed";
      return false;
    }
    return stager.Copy(shard_base + t.shard, t.file_offset, size, device,
                       error);
  }

  DeviceTensor CopyFormat(const TensorRef& t,
                          const flash_next::UploadHint& hint) {
    if (source == nullptr) {
      Fail("no weight source uploads " + std::string(t.name));
      return {};
    }
    Context context(*this);
    const flash_next::Tensor d = source->Upload(t, hint, context);
    if (!ok || Measuring())
      return {};
    if (d.empty() || d.format_state() == nullptr) {
      Fail("weight source upload failed for " + std::string(t.name));
      return {};
    }
    // No GGUF route reads an encoded tensor: every type-gated kernel choice
    // (Q8_0 GEMVs, WMMA experts, tiled projections) declines kI32.
    DeviceTensor device = FromTensor(d);
    device.type = core::GgmlType::kI32;
    return device;
  }

  /// A tensor with format_state outside the projections the stages run:
  /// its bytes would reach a raw GGUF kernel.
  bool RejectFormat(const TensorRef& t) {
    if (t.format_state == nullptr)
      return false;
    Fail("encoded tensor " + std::string(t.name) +
         " is only supported on a stage-run projection");
    return true;
  }

  /// A failed allocation of `size` bytes for `name`. The placement and the
  /// share already allocated tell a full device heap (hipMalloc) from
  /// system-wide GPU memory that other processes hold (managed).
  void FailAllocation(std::string_view name, std::size_t size) {
    std::ostringstream message;
    message << "device allocation failed for " << name << " (" << size
            << " bytes";
    if (weights != nullptr) {
      constexpr double kGib = 1024.0 * 1024.0 * 1024.0;
      message << std::fixed << std::setprecision(1) << "; "
              << static_cast<double>(requested - size) / kGib << " of "
              << static_cast<double>(measured) / kGib
              << " GiB of weights allocated, as "
              << (weights->placement() == hip::Placement::kMalloc
                      ? "device memory"
                      : "managed memory");
    }
    message << ')';
    Fail(message.str());
  }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  /// A projection the executor runs through Dense / Experts, which hand
  /// encoded tensors to the stages (Linear, LinearGroup).
  DeviceTensor Projection(const TensorRef& t,
                          const flash_next::UploadHint& hint = {}) {
    if (t.empty() || !ok)
      return {};
    if (t.format_state != nullptr)
      return CopyFormat(t, hint);
    return Copy(t);
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok || RejectFormat(t)) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      FailAllocation(t.name, size + kTailMargin);
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!Source(t, size, ptr)) {
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
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Uploads a narrow projection the quantized tier does not cover (the
  /// hyper-connection inject) as F32. A Q8_0 tensor is uploaded as is and
  /// expanded once the disk pipeline drains, so it runs the F32 kernels with
  /// its dequantized values instead of quantizing their activations. A BF16
  /// tensor is widened exactly on the host (see HcInjectUploadFor).
  DeviceTensor CopyF32(const TensorRef& t) {
    if (t.empty() || !ok || RejectFormat(t)) {
      return {};
    }
    switch (HcInjectUploadFor(t.type)) {
      case HcInjectUpload::kCopy:
        return Copy(t);
      case HcInjectUpload::kWidenBF16:
        return UploadWidened(t);
      case HcInjectUpload::kExpandQ8_0:
        break;
    }
    DeviceTensor d = Copy(t);
    const std::size_t count = t.cols * t.rows * t.experts;
    void* f32 = nullptr;
    if (!ok) {
      return {};
    }
    if (Allocate(&f32, count * sizeof(float) + kTailMargin) != hipSuccess) {
      FailAllocation(t.name, count * sizeof(float) + kTailMargin);
      return {};
    }
    if (Measuring()) {
      return {};
    }
    allocations.push_back(f32);
    bytes += count * sizeof(float) + kTailMargin;
    conversions.push_back({d.data, f32, count, true, t.SizeBytes()});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(f32) + count * 4, 0,
                         kTailMargin, nullptr);
    d.data = f32;
    d.type = core::GgmlType::kF32;
    return d;
  }

  DeviceTensor UploadWidened(const TensorRef& t) {
    DeviceTensor d;
    const std::size_t count = t.cols * t.rows * t.experts;
    void* f32 = nullptr;
    if (Allocate(&f32, count * sizeof(float) + kTailMargin) != hipSuccess) {
      FailAllocation(t.name, count * sizeof(float) + kTailMargin);
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(f32);
    bytes += count * sizeof(float) + kTailMargin;
    std::vector<float> host(count);
    WidenBF16(static_cast<const std::uint16_t*>(t.data), host.data(), count);
    if (hipMemcpy(f32, host.data(), count * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(f32) + count * sizeof(float), 0,
                  kTailMargin) != hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = f32;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok || RejectFormat(t))
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
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
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV, unless `expand_q8` asks for the F32 stack's F16
  /// route over its dequantized values.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts,
                     bool expand_q8 = false) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (RejectFormat(*t))
        return d;
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!Source(*t, t->SizeBytes(),
                    static_cast<std::uint8_t*>(ptr) + offset)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0 && !expand_q8) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    void* f32 = ptr;
    std::size_t f32_bytes = size;
    if (type == core::GgmlType::kQ8_0) {
      const std::size_t elements = rows * (*parts.begin())->cols;
      f32_bytes = elements * sizeof(float);
      if (Allocate(&f32, f32_bytes + kTailMargin) != hipSuccess) {
        Fail("stacked tensor allocation failed");
        return d;
      }
      if (!Measuring()) {
        allocations.push_back(f32);
        bytes += f32_bytes + kTailMargin;
        conversions.push_back({ptr, f32, elements, true, size});
      }
    }
    const std::size_t count = f32_bytes / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({f32, half, count, false, f32_bytes});
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
    return {Copy(m.norm), Copy(m.down), Copy(m.up), CopyF32(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->format_state != nullptr ||
            t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Projection(l.ssm_qkv);
      d.ssm_gate = Projection(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta}, true);
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Projection(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Projection(l.attn_q);
      d.attn_k = Projection(l.attn_k);
      d.attn_v = Projection(l.attn_v);
    }
    d.attn_out = Projection(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Projection(l.indexer_q);
    d.indexer_k = Projection(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Projection(l.ffn_gate_exps);
    d.ffn_up_exps = Projection(l.ffn_up_exps);
    d.ffn_down_exps = Projection(l.ffn_down_exps);
    constexpr flash_next::UploadHint kShared{.beside_routed_experts = true};
    d.shexp_gate = Projection(l.shexp_gate, kShared);
    d.shexp_up = Projection(l.shexp_up, kShared);
    d.shexp_down = Projection(l.shexp_down, kShared);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    if (!l.nextn_fc_embedding.empty()) {
      d.nextn_fc_embedding = Projection(l.nextn_fc_embedding);
      d.nextn_fc_hidden = Projection(l.nextn_fc_hidden);
    }
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output =
        w.output.format_state == nullptr && w.output.data == w.token_embd.data
            ? u.token_embd
            : Projection(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  const DeviceTensor& head = draft_source();
  if (head.empty() || head.rows == 0 || head.format_state != nullptr ||
      (draft_source_.empty() && output_bytes_ % head.rows != 0))
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= head.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= head.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  draft_output_ = head;
  draft_output_.rows = prefix_rows;
  if (head.type == core::GgmlType::kQ8_0 && head.cols % 256 == 0) {
    // Drafts read a Q4_K copy of the Q8_0 prefix: the head is the larger
    // part of each draft step's weight traffic, and its proposals are
    // verified, so this changes acceptance only. Each row of the copy is
    // the same at any batch width.
    const std::size_t bytes =
        std::size_t{prefix_rows} * (head.cols / 256) * 144;
    void* q4k = nullptr;
    if (hipMalloc(&q4k, bytes + kTailMargin) != hipSuccess)
      return fail("device allocation failed for the Q4_K draft head");
    allocations_.push_back(q4k);
    bytes_ += bytes + kTailMargin;
    QuantizeQ8_0ToQ4K(head.data, q4k, prefix_rows, head.cols, nullptr);
    if (hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + bytes, 0, kTailMargin,
                       nullptr) != hipSuccess ||
        hipStreamSynchronize(nullptr) != hipSuccess)
      return fail("Q4_K draft head quantization failed");
    draft_output_.data = q4k;
    draft_output_.type = core::GgmlType::kQ4_K;
  }
  draft_specials_ = std::move(specials);
  return true;
}

bool DeviceModel::UseDraftSource(const DeviceTensor& head,
                                 std::string* error_msg) {
  TensorRef ref;
  ref.type = head.type;
  ref.cols = head.cols;
  const std::size_t row_bytes = ref.RowBytes();
  if (head.empty() || head.format_state != nullptr || row_bytes == 0 ||
      head.rows != output_.rows || head.cols != output_.cols ||
      head.experts != 1) {
    if (error_msg != nullptr)
      *error_msg = "draft source must be a GGUF matrix of the head's shape";
    return false;
  }
  draft_source_ = head;
  draft_source_bytes_ = row_bytes * head.rows;
  return true;
}

void WidenBF16(const std::uint16_t* src, float* dst,
               std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
    std::memcpy(dst + i, &bits, sizeof(float));
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const MtpWeights* mtp,
    flash_next::WeightSource& source, std::string* error_msg) {
  const auto trunk = source.Regions(false);
  std::vector<core::GgufMappedRegion> shards(trunk.begin(), trunk.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    const auto extra = source.Regions(true);
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  return UploadShards(w, mtp, std::move(shards), shard_count, &source,
                      error_msg);
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
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
  return UploadShards(w, mtp, std::move(shards), shard_count, nullptr,
                      error_msg);
}

std::unique_ptr<DeviceModel> DeviceModel::UploadShards(
    const ModelWeights& w, const MtpWeights* mtp,
    std::vector<core::GgufMappedRegion> shards, std::uint32_t shard_count,
    flash_next::WeightSource* source, std::string* error_msg) {
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
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, error_msg};
    measure.source = source;
    (void)measure.Model(w, mtp, shard_count);
    if (source != nullptr)
      source->UploadDone(true);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  up.measured = measured;
  up.source = source;
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  // The source hears of the walk's end only once its queued region reads
  // have landed, so it may free what they read into.
  bool finished = false;
  if (up.ok)
    finished = stager->Finish(error_msg);
  else if (source != nullptr)
    (void)stager->Finish(nullptr);
  if (source != nullptr)
    source->UploadDone(false);
  if (!up.ok || !finished) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    if (c.expand_q8) {
      ExpandQ8_0(c.source, static_cast<float*>(c.destination), c.count,
                 nullptr);
    } else {
      NarrowActivations(static_cast<const float*>(c.source), c.destination,
                        false, c.count, nullptr);
    }
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
    m->bytes_ -= c.source_bytes + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
