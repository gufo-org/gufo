#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
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

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

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

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
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

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
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
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
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
    }
    if (type == core::GgmlType::kQ8_0) {
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
    const std::size_t count = size / sizeof(float);
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

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
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
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : Copy(w.output);
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

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
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
#ifdef GUFO_NPU
  // Wake the NPU before any heavy GPU work starts: resuming it re-initializes
  // the SMU and PSP, which the GPU shares. A heartbeat then keeps it awake.
  std::unique_ptr<NpuGemm> early_npu;
  if (const char* dir = std::getenv("GUFO_NPU_DIR");
      dir != nullptr && dir[0] != '\0') {
    NpuGemm::Config cfg;
    cfg.xclbin = std::string(dir) + "/final_4096x2560x1024_128x64x128.xclbin";
    cfg.insts = std::string(dir) + "/insts_4096x2560x1024_128x64x128.txt";
    std::string npu_error;
    early_npu = NpuGemm::Create(cfg, &npu_error);
    if (early_npu == nullptr) {
      std::fprintf(stderr, "NPU off: %s\n", npu_error.c_str());
    }
  }
#endif
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
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
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
#ifdef GUFO_NPU
  // GUFO_NPU_DIR names a directory holding the NPU kernel pair
  // (final_4096x2560x1024_128x64x128.xclbin and the matching insts file).
  if (early_npu != nullptr) {
    std::string npu_error;
    auto npu = std::move(early_npu);
    {
      constexpr std::uint32_t kRows = 16384;
      // GUFO_NPU_SLICES sets how many 1024-row slices the NPU computes (the
      // last rows of ssm_in); the GPU does the rest.
      std::uint32_t kSlices = 6;
      if (const char* sl = std::getenv("GUFO_NPU_SLICES"); sl != nullptr) {
        kSlices = std::clamp(static_cast<std::uint32_t>(std::atoi(sl)), 1U, 6U);
      }
      const std::uint32_t kZRow0 = kRows - kSlices * npu->n();
      std::size_t staged = 0;
      std::uint32_t layers_on = 0;
      // GUFO_NPU_LAYERS limits how many linear-attention layers use the NPU.
      std::uint32_t layer_limit = 0xFFFFFFFFu;
      if (const char* limit = std::getenv("GUFO_NPU_LAYERS");
          limit != nullptr) {
        layer_limit = static_cast<std::uint32_t>(std::atoi(limit));
      }
      for (auto& l : m->layers_) {
        if (layers_on >= layer_limit) {
          break;
        }
        if (l.ssm_in.data == nullptr ||
            l.ssm_in.type != core::GgmlType::kQ8_0 ||
            l.ssm_in.rows != kZRow0 + kSlices * npu->n() ||
            l.ssm_in.cols != npu->k()) {
          continue;
        }
        std::vector<int> ids;
        for (std::uint32_t i = 0; i < kSlices; ++i) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(l.ssm_in.data, l.ssm_in.cols, kZRow0 + i * npu->n(),
                           npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                           nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == kSlices) {
          l.npu_z = std::move(ids);
          ++layers_on;
        }
      }
      // GUFO_NPU_GATE_SLICES sets how many 1024-row slices of attn_qkv (two
      // heads' query and gate rows each) the NPU computes in the attention
      // layers.
      std::uint32_t gate_slices = 8;
      if (const char* gs = std::getenv("GUFO_NPU_GATE_SLICES"); gs != nullptr) {
        gate_slices = std::min(static_cast<std::uint32_t>(std::atoi(gs)), 12U);
      }
      std::uint32_t gate_layers = 0;
      for (auto& l : m->layers_) {
        if (gate_slices == 0 || l.attn_qkv.empty() ||
            l.attn_qkv.data == nullptr ||
            l.attn_qkv.type != core::GgmlType::kQ8_0 ||
            l.attn_qkv.rows != 13312 || l.attn_qkv.cols != npu->k()) {
          continue;
        }
        std::vector<int> ids;
        for (std::uint32_t i = 0; i < gate_slices; ++i) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(l.attn_qkv.data, l.attn_qkv.cols, i * npu->n(),
                           npu->n(), npu->k(), 0, npu->k(), npu->Device(id),
                           nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == gate_slices) {
          l.npu_gate = std::move(ids);
          ++gate_layers;
        }
      }
      if (gate_layers != 0) {
        std::fprintf(stderr, "NPU: attention query/gate of %u layers\n",
                     gate_layers);
      }
      // GUFO_NPU_SHEXP=0 keeps the shared expert gate and up projections on
      // the GPU.
      const char* shexp_env = std::getenv("GUFO_NPU_SHEXP");
      std::uint32_t shexp_layers = 0;
      for (auto& l : m->layers_) {
        if ((shexp_env != nullptr && shexp_env[0] == '0') ||
            l.shexp_gate.data == nullptr || l.shexp_up.data == nullptr ||
            l.shexp_gate.type != core::GgmlType::kQ8_0 ||
            l.shexp_up.type != core::GgmlType::kQ8_0 ||
            l.shexp_gate.cols != npu->k() || l.shexp_up.cols != npu->k() ||
            l.shexp_gate.rows != l.shexp_up.rows ||
            l.shexp_gate.rows > npu->n()) {
          continue;
        }
        std::vector<int> ids;
        for (const DeviceTensor* w : {&l.shexp_gate, &l.shexp_up}) {
          const int id = npu->NewBuffer(npu->BBytes(), &npu_error);
          if (id < 0) {
            break;
          }
          NpuEncodeWeights(w->data, w->cols, 0, w->rows, npu->k(), 0, npu->k(),
                           npu->Device(id), nullptr);
          ids.push_back(id);
          staged += npu->BBytes();
        }
        if (hipDeviceSynchronize() != hipSuccess) {
          ids.clear();
        }
        for (int id : ids) {
          npu->FinalizeWeights(id);
        }
        if (ids.size() == 2) {
          l.npu_shexp = std::move(ids);
          ++shexp_layers;
        }
      }
      if (shexp_layers != 0) {
        std::fprintf(stderr, "NPU: shared expert of %u layers\n", shexp_layers);
      }
      if (hipDeviceSynchronize() != hipSuccess) {
        if (error_msg != nullptr) {
          *error_msg = "NPU weight staging failed";
        }
        return nullptr;
      }
      std::fprintf(stderr, "NPU: z projection of %u layers, %.2f GiB staged\n",
                   layers_on,
                   static_cast<double>(staged) / (1024.0 * 1024.0 * 1024.0));
      if (layers_on != 0) {
        m->npu_ = std::move(npu);
      }
    }
  }
#endif
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
