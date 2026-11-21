#include "src/models/yue2/hip/backbone.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/families/qwen/eager/attention.hpp"
#include "src/families/qwen/eager/blas.hpp"
#include "src/families/qwen/eager/ops.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "src/models/yue2/hip/kernels.hpp"
#include "src/models/yue2/safetensors.hpp"

namespace gufo::models::yue2::hip {
namespace {

namespace eager = ::gufo::qwen::eager;

constexpr std::size_t kPrefillRows = 2048;
// The smallest batch the planned hipBLASLt GEMM takes (eager::Gemm::Run).
constexpr std::size_t kPackedRows = 64;
constexpr std::size_t kTimeFrequencies = 256;

void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("YuE2 ") + std::string(operation) +
                             ": " + hipGetErrorString(status));
}

float RoundBfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xFFFF0000U);
}

std::uint16_t Bfloat16Bits(float value) {
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(RoundBfloat16(value)) >> 16U);
}

template<typename T>
class DeviceArray {
public:
  DeviceArray() = default;
  ~DeviceArray() { Free(); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  /// Grows to at least `count` elements; contents are not kept.
  void Reserve(std::size_t count) {
    if (count <= count_)
      return;
    Free();
    RequireHip(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
               "hipMalloc");
    count_ = count;
  }
  [[nodiscard]] T* get() const { return data_; }

private:
  void Free() {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    count_ = 0;
  }
  T* data_{nullptr};
  std::size_t count_{0};
};

struct Path {
  const void* q{nullptr};
  const void* k{nullptr};
  const void* v{nullptr};
  const void* o{nullptr};
  const void* gate{nullptr};
  const void* up{nullptr};
  const void* down{nullptr};
  float* input_norm{nullptr};
  float* post_norm{nullptr};
  float* q_norm{nullptr};
  float* k_norm{nullptr};
};

struct Layer {
  Path ar;
  Path nar;
  // 12-bit copies of the AR projections for one-token decode; q, k and v
  // are one matrix, as are gate and up.
  PackedMatrix qkv, o, gate_up, down;
};

// Where a pass writes its keys and values and which rows it attends to.
struct Pass {
  bool nar{false};
  std::size_t position{0};
  // AR: the cache this pass appends to. NAR: the frozen prefix.
  const KvCache* cache{nullptr};
  std::uint16_t* cache_keys{nullptr};
  std::uint16_t* cache_values{nullptr};
  std::size_t capacity{0};
  std::size_t prefix{0};
};

}  // namespace

KvCache::~KvCache() {
  if (keys_ != nullptr)
    (void)hipFree(keys_);
  if (values_ != nullptr)
    (void)hipFree(values_);
}

struct Backbone::Impl {
  BackboneConfig config;
  std::unique_ptr<SafetensorsFile> file;
  void* weights_allocation{nullptr};
  const std::uint8_t* weights{nullptr};
  std::vector<Layer> layers;
  std::vector<float*> norm_allocations;
  float* final_norm{nullptr};
  const void* embed{nullptr};
  const void* lm_head{nullptr};
  const void* vae2llm_weight{nullptr};
  const void* vae2llm_bias{nullptr};
  const void* llm2vae_weight{nullptr};
  const void* llm2vae_bias{nullptr};
  const void* time_weight0{nullptr};
  const void* time_bias0{nullptr};
  const void* time_weight2{nullptr};
  const void* time_bias2{nullptr};
  const void* positions{nullptr};
  void* packed_allocation{nullptr};
  void* packed_escapes{nullptr};
  PackedMatrix packed_head;

  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  eager::Gemm planned;
  std::size_t rows_reserved{0};
  DeviceArray<float> hidden, q, k, v, o, gate, up, ffn;
  DeviceArray<std::uint16_t> query_heads, attention, scratch;
  DeviceArray<std::uint16_t> nar_keys, nar_values;
  // Transposed values of the frozen prefix (every layer) and of the latent
  // rows (the current layer), as the acoustic attention reads them.
  DeviceArray<std::uint16_t> prefix_values_t, nar_values_t;
  DeviceArray<std::uint32_t> ids;
  DeviceArray<int> limits;
  DeviceArray<float> logits, logits_rounded, decode_scratch;
  DeviceArray<float> state, midpoint, first, second, projected;
  DeviceArray<std::uint16_t> frequencies, time_row;
  DeviceArray<float> time_hidden;
  // BF16 RoPE cosines and sines of every position, computed once.
  DeviceArray<float> rotary;
  std::size_t decode_scratch_capacity{0};

  ~Impl() {
    for (float* norm : norm_allocations)
      (void)hipFree(norm);
    if (weights_allocation != nullptr)
      (void)hipFree(weights_allocation);
    if (packed_allocation != nullptr)
      (void)hipFree(packed_allocation);
    if (packed_escapes != nullptr)
      (void)hipFree(packed_escapes);
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  const void* Matrix(std::string_view name, std::uint64_t rows,
                     std::uint64_t columns) const {
    const std::array<std::uint64_t, 2> shape{rows, columns};
    return weights + file->Require(name, "BF16", shape).offset;
  }
  const void* Vector(std::string_view name, std::uint64_t size) const {
    const std::array<std::uint64_t, 1> shape{size};
    return weights + file->Require(name, "BF16", shape).offset;
  }
  float* Norm(std::string_view name, std::uint64_t size) {
    const std::array<std::uint64_t, 1> shape{size};
    const std::vector<float> host =
        file->ReadFloat(file->Require(name, "BF16", shape));
    float* device = nullptr;
    RequireHip(hipMalloc(reinterpret_cast<void**>(&device),
                         host.size() * sizeof(float)),
               "hipMalloc norm");
    norm_allocations.push_back(device);
    RequireHip(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "hipMemcpy norm");
    return device;
  }

  void Upload() {
    // Only the payload is copied, and tensor offsets inside it are arbitrary
    // multiples of 8. Shift the copy so the large matrices start on 256-byte
    // boundaries: the decode GEMV then takes its 16-byte loads (about 10%
    // faster than 8-byte ones) and hipBLASLt reads aligned tiles.
    std::array<std::size_t, 256> residues{};
    for (const auto& [name, tensor] : file->tensors()) {
      if (tensor.bytes >= (std::uint64_t{1} << 20))
        ++residues[tensor.offset % 256];
    }
    const std::size_t common = static_cast<std::size_t>(
        std::ranges::max_element(residues) - residues.begin());
    const std::size_t shift = (256 - common) % 256;
    RequireHip(hipMalloc(&weights_allocation, file->payload_size() + shift),
               "hipMalloc weights");
    auto* device = static_cast<std::uint8_t*>(weights_allocation) + shift;
    std::string error;
    const core::GgufMappedRegion region{file->file().data(),
                                        file->file().size(),
                                        file->file().PlatformHandle()};
    auto upload = gufo::hip::WeightUpload::Create({&region, 1}, &error);
    if (!upload ||
        !upload->CopyMapped(file->payload(), file->payload_size(), device,
                            &error) ||
        !upload->Finish(&error))
      throw std::runtime_error("YuE2 weight upload failed: " + error);
    weights = device;
  }

  void Resolve() {
    const auto& c = config;
    const std::size_t qd = c.heads * c.head_dim;
    const std::size_t kvd = c.kv_heads * c.head_dim;
    layers.resize(c.layers);
    for (std::size_t l = 0; l < c.layers; ++l) {
      const std::string base = "model.layers." + std::to_string(l) + ".";
      const auto path = [&](std::string_view attention, std::string_view mlp,
                            std::string_view input, std::string_view post) {
        const std::string a = base + std::string(attention) + ".";
        const std::string m = base + std::string(mlp) + ".";
        Path p;
        p.q = Matrix(a + "q_proj.weight", qd, c.hidden);
        p.k = Matrix(a + "k_proj.weight", kvd, c.hidden);
        p.v = Matrix(a + "v_proj.weight", kvd, c.hidden);
        p.o = Matrix(a + "o_proj.weight", c.hidden, qd);
        p.q_norm = Norm(a + "q_norm.weight", c.head_dim);
        p.k_norm = Norm(a + "k_norm.weight", c.head_dim);
        p.gate = Matrix(m + "gate_proj.weight", c.intermediate, c.hidden);
        p.up = Matrix(m + "up_proj.weight", c.intermediate, c.hidden);
        p.down = Matrix(m + "down_proj.weight", c.hidden, c.intermediate);
        p.input_norm = Norm(base + std::string(input) + ".weight", c.hidden);
        p.post_norm = Norm(base + std::string(post) + ".weight", c.hidden);
        return p;
      };
      layers[l].ar = path("self_attn", "mlp", "input_layernorm",
                          "post_attention_layernorm");
      layers[l].nar = path("nar_self_attn", "nar_mlp", "nar_input_layernorm",
                           "nar_pre_mlp_layernorm");
    }
    final_norm = Norm("model.norm.weight", c.hidden);
    embed = Matrix("model.embed_tokens.weight", c.vocabulary, c.hidden);
    lm_head = Matrix("lm_head.weight", c.vocabulary, c.hidden);
    vae2llm_weight = Matrix("vae2llm.weight", c.hidden, c.latent_dim);
    vae2llm_bias = Vector("vae2llm.bias", c.hidden);
    llm2vae_weight = Matrix("llm2vae.weight", c.latent_dim, c.hidden);
    llm2vae_bias = Vector("llm2vae.bias", c.latent_dim);
    time_weight0 =
        Matrix("time_embedder.mlp.0.weight", c.hidden, kTimeFrequencies);
    time_bias0 = Vector("time_embedder.mlp.0.bias", c.hidden);
    time_weight2 = Matrix("time_embedder.mlp.2.weight", c.hidden, c.hidden);
    time_bias2 = Vector("time_embedder.mlp.2.bias", c.hidden);
    positions = Matrix("latent_pos_embed.pe", c.max_latent_frames, c.hidden);
  }

  /// Builds the 12-bit decode copies (`kernels.hpp`) of the AR projections
  /// and of the head from the uploaded tensors, 2.5 GiB on the device:
  /// one-token decode then reads three quarters of the weight bytes. Prefill
  /// and the NAR path keep the BF16 tensors.
  void PackDecode() {
    const auto& c = config;
    const std::size_t qd = c.heads * c.head_dim;
    const std::size_t kvd = c.kv_heads * c.head_dim;
    struct Source {
      const void* weights{nullptr};
      std::size_t rows{0};
    };
    struct Job {
      PackedMatrix* matrix{nullptr};
      std::size_t columns{0};
      std::vector<Source> sources;
      std::size_t units_offset{0};
      std::size_t rows_offset{0};
    };
    std::vector<Job> jobs;
    for (Layer& layer : layers) {
      const Path& w = layer.ar;
      jobs.push_back(
          {&layer.qkv, c.hidden, {{w.q, qd}, {w.k, kvd}, {w.v, kvd}}});
      jobs.push_back({&layer.o, qd, {{w.o, c.hidden}}});
      jobs.push_back({&layer.gate_up,
                      c.hidden,
                      {{w.gate, c.intermediate}, {w.up, c.intermediate}}});
      jobs.push_back({&layer.down, c.intermediate, {{w.down, c.hidden}}});
    }
    jobs.push_back({&packed_head, c.hidden, {{lm_head, c.vocabulary}}});

    const auto aligned = [](std::size_t bytes) {
      return (bytes + 255) & ~std::size_t{255};
    };
    // Three counters lead the allocation: the escape total, the cursor the
    // write pass hands escape ranges out from, and the verify mismatches.
    std::size_t total = aligned(3 * sizeof(std::uint32_t));
    for (Job& job : jobs) {
      std::size_t rows = 0;
      for (const Source& source : job.sources)
        rows += source.rows;
      job.units_offset = total;
      total +=
          aligned(rows * PackedRowWords(job.columns) * sizeof(std::uint32_t));
      job.rows_offset = total;
      total += aligned(rows * sizeof(PackedRow));
    }
    RequireHip(hipMalloc(&packed_allocation, total), "hipMalloc packed");
    auto* device = static_cast<std::uint8_t*>(packed_allocation);
    auto* counters = static_cast<std::uint32_t*>(packed_allocation);
    RequireHip(hipMemsetAsync(counters, 0, 3 * sizeof(std::uint32_t), stream),
               "hipMemsetAsync packed counters");
    for (const Job& job : jobs) {
      job.matrix->units =
          reinterpret_cast<std::uint32_t*>(device + job.units_offset);
      job.matrix->rows = reinterpret_cast<PackedRow*>(device + job.rows_offset);
    }
    const auto each = [&](auto&& launch) {
      for (const Job& job : jobs) {
        std::size_t first = 0;
        for (const Source& source : job.sources) {
          launch(source.weights, source.rows, job.columns, *job.matrix, first);
          first += source.rows;
        }
      }
    };
    each([&](const void* weights, std::size_t rows, std::size_t columns,
             const PackedMatrix& matrix, std::size_t first) {
      LaunchPackedRowsScan(weights, rows, columns, matrix, first, &counters[0],
                           stream);
    });
    std::uint32_t escape_count = 0;
    RequireHip(hipMemcpyAsync(&escape_count, &counters[0], sizeof(escape_count),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync packed escape count");
    RequireHip(hipStreamSynchronize(stream), "packed scan");
    // Never empty, so the kernels always hold a valid pointer.
    RequireHip(hipMalloc(&packed_escapes, (std::size_t{escape_count} + 1) *
                                              sizeof(PackedEscape)),
               "hipMalloc packed escapes");
    for (const Job& job : jobs)
      job.matrix->escapes = static_cast<PackedEscape*>(packed_escapes);
    each([&](const void* weights, std::size_t rows, std::size_t columns,
             const PackedMatrix& matrix, std::size_t first) {
      LaunchPackedRowsWrite(weights, rows, columns, matrix, first, &counters[1],
                            stream);
    });
    // The copies must hold every weight exactly: decode them all once.
    each([&](const void* weights, std::size_t rows, std::size_t columns,
             const PackedMatrix& matrix, std::size_t first) {
      LaunchPackedRowsVerify(weights, rows, columns, matrix, first,
                             &counters[2], stream);
    });
    std::uint32_t mismatches = 0;
    RequireHip(hipMemcpyAsync(&mismatches, &counters[2], sizeof(mismatches),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync packed mismatches");
    RequireHip(hipStreamSynchronize(stream), "packed write");
    RequireHip(hipGetLastError(), "packed launch");
    if (mismatches != 0)
      throw std::runtime_error("YuE2 packed decode weights differ from BF16");
  }

  void ReserveRows(std::size_t rows) {
    if (rows <= rows_reserved)
      return;
    const auto& c = config;
    hidden.Reserve(rows * c.hidden);
    q.Reserve(rows * c.heads * c.head_dim);
    k.Reserve(rows * c.kv_heads * c.head_dim);
    v.Reserve(rows * c.kv_heads * c.head_dim);
    o.Reserve(rows * c.hidden);
    gate.Reserve(rows * c.intermediate);
    up.Reserve(rows * c.intermediate);
    ffn.Reserve(rows * c.hidden);
    query_heads.Reserve(rows * c.heads * c.head_dim);
    attention.Reserve(rows * c.heads * c.head_dim);
    scratch.Reserve(rows * std::max(c.hidden, c.intermediate));
    ids.Reserve(rows);
    limits.Reserve(rows);
    rows_reserved = rows;
  }

  void Gemm(const void* weight, const void* input, float* output,
            std::size_t rows, std::size_t outputs, std::size_t inputs) {
    // Decode streams each weight row; prefill-sized batches use a fixed
    // hipBLASLt plan, about 6x hipBLAS's default on gfx1151.
    if (eager::LaunchBfloat16Gemv(weight, input, output, rows, outputs, inputs,
                                  stream))
      return;
    if (planned.Run(weight, input, output, static_cast<int>(rows),
                    static_cast<int>(outputs), static_cast<int>(inputs),
                    /*float_output=*/true, stream))
      return;
    gufo::hip::LaunchHipblasGEMMBF16(blas, weight, input, output, rows, outputs,
                                     inputs, stream);
  }

  /// A prefill-sized projection that writes its BF16 boundary directly, about
  /// 1.5x faster than the FP32 output every consumer rounds anyway.
  void GemmBfloat16(const void* weight, const void* input, void* output,
                    std::size_t rows, std::size_t outputs, std::size_t inputs) {
    // The 6144-wide gate and up projections run on hipBLASLt's first
    // candidate (a 128-row macro tile): 1528 instead of 2161 us at 1800 rows
    // in the acoustic stage on gfx1151, where the shared policy's second
    // candidate re-reads each weight tile for every 64 rows.
    const int rank = rows > 512 && outputs >= 4096 ? 0 : -1;
    if (planned.Run(weight, input, output, static_cast<int>(rows),
                    static_cast<int>(outputs), static_cast<int>(inputs),
                    /*float_output=*/false, stream, rank))
      return;
    const float alpha = 1.0F, beta = 0.0F;
    if (hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                      static_cast<int>(outputs), static_cast<int>(rows),
                      static_cast<int>(inputs), &alpha, weight, HIP_R_16BF,
                      static_cast<int>(inputs), input, HIP_R_16BF,
                      static_cast<int>(inputs), &beta, output, HIP_R_16BF,
                      static_cast<int>(outputs), HIPBLAS_COMPUTE_32F,
                      HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("YuE2 hipblasGemmEx failed");
  }

  void GemmGrouped(std::span<const eager::Bfloat16GemvGroup> groups,
                   const void* input, std::size_t rows, std::size_t inputs) {
    if (eager::LaunchBfloat16GemvGrouped(groups.data(), groups.size(), input,
                                         rows, inputs, stream))
      return;
    for (const auto& group : groups)
      Gemm(group.weights_bfloat16, input, group.output, rows, group.rows,
           inputs);
  }

  void ResidualNorm(const float* update, const float* weight,
                    std::size_t rows) {
    eager::LaunchBfloat16ResidualAddRMSNorm(
        hidden.get(), update, weight, scratch.get(), rows, config.hidden,
        config.rms_epsilon, stream, eager::RmsScale::kBfloat16);
  }
  void ResidualNormFromBfloat16(const std::uint16_t* update,
                                const float* weight, std::size_t rows) {
    eager::LaunchBfloat16ResidualAddRMSNormFromBfloat16(
        hidden.get(), update, weight, scratch.get(), rows, config.hidden,
        config.rms_epsilon, stream, eager::RmsScale::kBfloat16);
  }

  void EnsureDecodeScratch(std::size_t keys) {
    if (keys <= decode_scratch_capacity)
      return;
    decode_scratch.Reserve(eager::DecodeAttentionScratch(keys, config.heads));
    decode_scratch_capacity = keys;
  }

  /// Runs every layer of one path over `rows` rows already in `hidden`; the
  /// final-normed BF16 rows are left in `scratch`.
  void Layers(std::size_t rows, const Pass& pass) {
    const auto& c = config;
    const std::size_t qd = c.heads * c.head_dim;
    const std::size_t kvd = c.kv_heads * c.head_dim;
    const auto path = [&](std::size_t l) -> const Path& {
      return pass.nar ? layers[l].nar : layers[l].ar;
    };
    const auto kv_layer = [&](std::uint16_t* base, std::size_t l) {
      return base + l * c.kv_heads * pass.capacity * c.head_dim;
    };
    if (!pass.nar && rows > 1) {
      std::vector<int> host(rows);
      for (std::size_t r = 0; r < rows; ++r)
        host[r] = static_cast<int>(pass.position + r + 1);
      RequireHip(hipMemcpyAsync(limits.get(), host.data(), rows * sizeof(int),
                                hipMemcpyHostToDevice, stream),
                 "hipMemcpyAsync limits");
      // `host` must outlive the copy.
      RequireHip(hipStreamSynchronize(stream), "limits upload");
    }
    if (!pass.nar && rows == 1)
      EnsureDecodeScratch(pass.position + 1);
    // From planned-GEMM size up, projections write BF16 into the FP32
    // buffers' storage; the BF16-input kernels then round nothing twice.
    const bool packed = rows >= kPackedRows;
    // One AR token: the weight-bound projections read the 12-bit copies.
    const bool single = !pass.nar && rows == 1;
    const auto bfloat16 = [](DeviceArray<float>& buffer) {
      return reinterpret_cast<std::uint16_t*>(buffer.get());
    };
    ResidualNorm(nullptr, path(0).input_norm, rows);
    for (std::size_t l = 0; l < c.layers; ++l) {
      const Path& w = path(l);
      if (packed) {
        GemmBfloat16(w.q, scratch.get(), bfloat16(q), rows, qd, c.hidden);
        GemmBfloat16(w.k, scratch.get(), bfloat16(k), rows, kvd, c.hidden);
        GemmBfloat16(w.v, scratch.get(), bfloat16(v), rows, kvd, c.hidden);
      } else if (single) {
        const std::array<PackedOutput, 3> qkv{{
            {q.get(), static_cast<std::uint32_t>(qd)},
            {k.get(), static_cast<std::uint32_t>(kvd)},
            {v.get(), static_cast<std::uint32_t>(kvd)},
        }};
        LaunchPackedGemv(layers[l].qkv, 0, qkv.data(), qkv.size(),
                         scratch.get(), c.hidden, stream);
      } else {
        const std::array<eager::Bfloat16GemvGroup, 3> qkv{{
            {w.q, q.get(), qd},
            {w.k, k.get(), kvd},
            {w.v, v.get(), kvd},
        }};
        GemmGrouped(qkv, scratch.get(), rows, c.hidden);
      }
      std::uint16_t* keys_out = nullptr;
      std::uint16_t* values_out = nullptr;
      std::size_t stride = 0, row = 0;
      if (pass.nar) {
        keys_out = nar_keys.get();
        values_out = nar_values.get();
        stride = rows;
      } else {
        keys_out = kv_layer(pass.cache_keys, l);
        values_out = kv_layer(pass.cache_values, l);
        stride = pass.capacity;
        row = pass.position;
      }
      const auto heads = static_cast<std::uint32_t>(c.heads);
      const auto kv_heads = static_cast<std::uint32_t>(c.kv_heads);
      const auto head_dim = static_cast<std::uint32_t>(c.head_dim);
      const auto position = static_cast<std::uint32_t>(pass.position);
      if (packed)
        eager::LaunchBfloat16QkNormRoPEHeadMajorFromBfloat16(
            bfloat16(q), bfloat16(k), bfloat16(v), w.q_norm, w.k_norm, rows,
            heads, kv_heads, head_dim, position, rotary.get(), c.rms_epsilon,
            eager::RmsScale::kBfloat16, query_heads.get(), rows, 0, keys_out,
            values_out, stride, row, stream);
      else
        eager::LaunchBfloat16QkNormRoPEHeadMajor(
            q.get(), k.get(), v.get(), w.q_norm, w.k_norm, rows, heads,
            kv_heads, head_dim, position, rotary.get(), c.rms_epsilon,
            eager::RmsScale::kBfloat16, query_heads.get(), rows, 0, keys_out,
            values_out, stride, row, stream);
      if (pass.nar) {
        const auto kv = static_cast<std::uint32_t>(c.kv_heads);
        const auto prefix_count = static_cast<std::uint32_t>(pass.prefix);
        const auto latent_count = static_cast<std::uint32_t>(rows);
        LaunchTransposeValues(nar_values.get(), latent_count, latent_count, kv,
                              nar_values_t.get(), stream);
        const std::uint32_t prefix_pitch = NarValuePitch(prefix_count);
        const NarSegment prefix{
            kv_layer(pass.cache_keys, l),
            prefix_values_t.get() + l * c.kv_heads * c.head_dim * prefix_pitch,
            prefix_count, static_cast<std::uint32_t>(pass.capacity),
            prefix_pitch};
        const NarSegment latent{nar_keys.get(), nar_values_t.get(),
                                latent_count, latent_count,
                                NarValuePitch(latent_count)};
        LaunchNarAttention(query_heads.get(), prefix, latent, attention.get(),
                           heads, latent_count, heads / kv_heads, stream);
      } else if (rows == 1) {
        eager::LaunchBfloat16DecodeAttention(
            query_heads.get(), kv_layer(pass.cache_keys, l),
            kv_layer(pass.cache_values, l), pass.capacity, pass.position + 1,
            c.heads, c.kv_heads, decode_scratch.get(), attention.get(), stream);
      } else {
        eager::KeySegments segments;
        const int total = static_cast<int>(pass.position + rows);
        for (int s = 0; s < 3; ++s) {
          segments.keys[s] = kv_layer(pass.cache_keys, l);
          segments.values[s] = kv_layer(pass.cache_values, l);
          segments.end[s] = total;
          segments.stride[s] = static_cast<int>(pass.capacity);
        }
        eager::FusedAttention128Segmented(
            query_heads.get(), segments, attention.get(),
            static_cast<int>(c.heads), static_cast<int>(rows), 0,
            static_cast<int>(rows), stream, limits.get(),
            static_cast<int>(c.heads / c.kv_heads));
      }
      const float* next_norm =
          l + 1 < c.layers ? path(l + 1).input_norm : final_norm;
      if (packed) {
        GemmBfloat16(w.o, attention.get(), bfloat16(o), rows, c.hidden, qd);
        ResidualNormFromBfloat16(bfloat16(o), w.post_norm, rows);
        GemmBfloat16(w.gate, scratch.get(), bfloat16(gate), rows,
                     c.intermediate, c.hidden);
        GemmBfloat16(w.up, scratch.get(), bfloat16(up), rows, c.intermediate,
                     c.hidden);
        eager::LaunchBfloat16SwiGluFromBfloat16(bfloat16(gate), bfloat16(up),
                                                scratch.get(),
                                                rows * c.intermediate, stream);
        GemmBfloat16(w.down, scratch.get(), bfloat16(ffn), rows, c.hidden,
                     c.intermediate);
        ResidualNormFromBfloat16(bfloat16(ffn), next_norm, rows);
        continue;
      }
      if (single) {
        const auto hidden_rows = static_cast<std::uint32_t>(c.hidden);
        const auto wide_rows = static_cast<std::uint32_t>(c.intermediate);
        const PackedOutput attended{o.get(), hidden_rows};
        LaunchPackedGemv(layers[l].o, 0, &attended, 1, attention.get(), qd,
                         stream);
        ResidualNorm(o.get(), w.post_norm, rows);
        const std::array<PackedOutput, 2> gate_up{{
            {gate.get(), wide_rows},
            {up.get(), wide_rows},
        }};
        LaunchPackedGemv(layers[l].gate_up, 0, gate_up.data(), gate_up.size(),
                         scratch.get(), c.hidden, stream);
        eager::LaunchBfloat16SwiGlu(gate.get(), up.get(), nullptr,
                                    scratch.get(), c.intermediate, stream);
        const PackedOutput projected_down{ffn.get(), hidden_rows};
        LaunchPackedGemv(layers[l].down, 0, &projected_down, 1, scratch.get(),
                         c.intermediate, stream);
        ResidualNorm(ffn.get(), next_norm, rows);
        continue;
      }
      Gemm(w.o, attention.get(), o.get(), rows, c.hidden, qd);
      ResidualNorm(o.get(), w.post_norm, rows);
      const std::array<eager::Bfloat16GemvGroup, 2> gate_up{{
          {w.gate, gate.get(), c.intermediate},
          {w.up, up.get(), c.intermediate},
      }};
      GemmGrouped(gate_up, scratch.get(), rows, c.hidden);
      eager::LaunchBfloat16SwiGlu(gate.get(), up.get(), nullptr, scratch.get(),
                                  rows * c.intermediate, stream);
      Gemm(w.down, scratch.get(), ffn.get(), rows, c.hidden, c.intermediate);
      ResidualNorm(ffn.get(), next_norm, rows);
    }
  }

  void Embed(std::span<const std::uint32_t> tokens) {
    for (const std::uint32_t token : tokens) {
      if (token >= config.vocabulary)
        throw std::out_of_range("YuE2 token id is out of range");
    }
    RequireHip(hipMemcpyAsync(ids.get(), tokens.data(),
                              tokens.size() * sizeof(std::uint32_t),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync ids");
    gufo::hip::LaunchBatchedEmbeddingLookup(
        embed, core::GgmlType::kBF16, ids.get(), hidden.get(), tokens.size(),
        config.hidden, stream);
  }

  /// Window logits of final-normed row `row` of `scratch`, rounded to BF16.
  void Logits(std::size_t row, VocabularyWindow window,
              std::vector<float>* output) {
    const std::size_t count = window.size();
    logits.Reserve(count);
    logits_rounded.Reserve(count);
    const PackedOutput rows{logits.get(), static_cast<std::uint32_t>(count)};
    LaunchPackedGemv(packed_head, window.begin, &rows, 1,
                     scratch.get() + row * config.hidden, config.hidden,
                     stream);
    LaunchBfloat16Logits(logits.get(), count, logits_rounded.get(), stream);
    output->resize(count);
    RequireHip(
        hipMemcpyAsync(output->data(), logits_rounded.get(),
                       count * sizeof(float), hipMemcpyDeviceToHost, stream),
        "hipMemcpyAsync logits");
  }

  void Append(KvCache& cache, std::span<const std::uint32_t> tokens,
              VocabularyWindow window, std::vector<float>* output) {
    if (tokens.empty())
      throw std::invalid_argument("YuE2 append needs tokens");
    if (cache.length_ + tokens.size() > cache.capacity_)
      throw std::length_error("YuE2 KV cache capacity exceeded");
    for (std::size_t begin = 0; begin < tokens.size(); begin += kPrefillRows) {
      const auto chunk =
          tokens.subspan(begin, std::min(kPrefillRows, tokens.size() - begin));
      ReserveRows(chunk.size());
      Embed(chunk);
      Pass pass;
      pass.position = cache.length_;
      pass.cache = &cache;
      pass.cache_keys = cache.keys_;
      pass.cache_values = cache.values_;
      pass.capacity = cache.capacity_;
      Layers(chunk.size(), pass);
      cache.length_ += chunk.size();
      if (begin + chunk.size() == tokens.size() && window.size() != 0)
        Logits(chunk.size() - 1, window, output);
    }
    RequireHip(hipStreamSynchronize(stream), "append");
    RequireHip(hipGetLastError(), "append launch");
  }

  /// Time embedding row of the shifted time `t`, as BF16 in `time_row`.
  void TimeEmbedding(float t) {
    std::array<std::uint16_t, kTimeFrequencies> host{};
    constexpr std::size_t half = kTimeFrequencies / 2;
    for (std::size_t i = 0; i < half; ++i) {
      const float frequency =
          std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                   static_cast<float>(half));
      const float argument = t * frequency;
      host[i] = Bfloat16Bits(std::cos(argument));
      host[half + i] = Bfloat16Bits(std::sin(argument));
    }
    frequencies.Reserve(kTimeFrequencies);
    time_row.Reserve(config.hidden);
    time_hidden.Reserve(2 * config.hidden);
    RequireHip(hipMemcpyAsync(frequencies.get(), host.data(), sizeof(host),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync time frequencies");
    float* first_out = time_hidden.get();
    float* second_out = time_hidden.get() + config.hidden;
    Gemm(time_weight0, frequencies.get(), first_out, 1, config.hidden,
         kTimeFrequencies);
    eager::LaunchBfloat16BiasSilu(first_out, time_bias0, nullptr,
                                  time_row.get(), 1, config.hidden, stream);
    Gemm(time_weight2, time_row.get(), second_out, 1, config.hidden,
         config.hidden);
    eager::LaunchBfloat16Bias(second_out, time_bias2, second_out, 1,
                              config.hidden, stream);
    gufo::hip::LaunchFloatToBfloat16(second_out, time_row.get(), config.hidden,
                                     stream);
    // `host` must outlive the copy.
    RequireHip(hipStreamSynchronize(stream), "time embedding");
  }

  /// Upstream `_shift_t_value`: sigmoid of the BF16 raw time, in BF16.
  float ShiftedTime(double raw) const {
    const float t = RoundBfloat16(
        1.0F / (1.0F + std::exp(-RoundBfloat16(static_cast<float>(raw)))));
    const float shift = config.timestep_shift;
    if (shift == 1.0F)
      return t;
    const float numerator = RoundBfloat16(shift * t);
    const float denominator =
        RoundBfloat16(1.0F + RoundBfloat16((shift - 1.0F) * t));
    return RoundBfloat16(numerator / denominator);
  }

  /// Velocity of the device state `x` (`[frames][64]`) into `out`.
  void VelocityDevice(const KvCache& prefix, const float* x, std::size_t frames,
                      double raw, float* out) {
    const auto& c = config;
    const std::size_t rows = frames + 2;
    TimeEmbedding(ShiftedTime(raw));
    LaunchLatentInput(x, frames, vae2llm_weight, vae2llm_bias, time_row.get(),
                      positions, hidden.get(), stream);
    Pass pass;
    pass.nar = true;
    pass.position = prefix.length_;
    pass.cache = &prefix;
    pass.cache_keys = prefix.keys_;
    pass.cache_values = prefix.values_;
    pass.capacity = prefix.capacity_;
    pass.prefix = prefix.length_;
    Layers(rows, pass);
    gufo::hip::LaunchHipblasGEMMBF16(blas, llm2vae_weight, scratch.get(),
                                     projected.get(), rows, c.latent_dim,
                                     c.hidden, stream);
    LaunchVelocityOutput(projected.get(), llm2vae_bias, frames, out, stream);
  }

  void PrepareAcoustic(const KvCache& prefix, std::size_t frames) {
    const auto& c = config;
    const std::size_t rows = frames + 2;
    if (frames == 0 || rows > c.max_latent_frames ||
        prefix.length_ + rows > c.context)
      throw std::length_error("YuE2 acoustic chunk exceeds the model context");
    ReserveRows(rows);
    nar_keys.Reserve(c.kv_heads * rows * c.head_dim);
    nar_values.Reserve(c.kv_heads * rows * c.head_dim);
    const std::size_t elements = frames * c.latent_dim;
    state.Reserve(elements);
    midpoint.Reserve(elements);
    first.Reserve(elements);
    second.Reserve(elements);
    projected.Reserve(rows * c.latent_dim);
    nar_values_t.Reserve(c.kv_heads * c.head_dim * NarValuePitch(rows));
    // The prefix is frozen for every velocity evaluation: transpose its
    // values once.
    const auto count = static_cast<std::uint32_t>(prefix.length_);
    const std::size_t pitch = NarValuePitch(count);
    const std::size_t layer = c.kv_heads * c.head_dim * pitch;
    prefix_values_t.Reserve(c.layers * layer);
    for (std::size_t l = 0; l < c.layers; ++l)
      LaunchTransposeValues(
          prefix.values_ + l * c.kv_heads * prefix.capacity_ * c.head_dim,
          static_cast<std::uint32_t>(prefix.capacity_), count,
          static_cast<std::uint32_t>(c.kv_heads),
          prefix_values_t.get() + l * layer, stream);
  }
};

Backbone::Backbone(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Backbone::~Backbone() = default;

std::unique_ptr<Backbone> Backbone::Create(
    const std::filesystem::path& model_root, const BackboneConfig& config,
    std::string* error) {
  try {
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->file = SafetensorsFile::Open(model_root / "model.safetensors", error);
    if (!impl->file)
      return nullptr;
    RequireHip(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags");
    if (hipblasCreate(&impl->blas) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetStream(impl->blas, impl->stream) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetAtomicsMode(impl->blas, HIPBLAS_ATOMICS_NOT_ALLOWED) !=
            HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("YuE2 cannot create a hipBLAS handle");
    impl->Upload();
    impl->Resolve();
    impl->PackDecode();
    impl->ReserveRows(kPrefillRows);
    impl->rotary.Reserve(config.context * config.head_dim);
    eager::LaunchBfloat16RotaryTable(
        impl->rotary.get(), config.context,
        static_cast<std::uint32_t>(config.head_dim), config.rope_theta,
        impl->stream);
    RequireHip(hipStreamSynchronize(impl->stream), "load");
    return std::unique_ptr<Backbone>(new Backbone(std::move(impl)));
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = exception.what();
    return nullptr;
  }
}

const BackboneConfig& Backbone::config() const {
  return impl_->config;
}

std::unique_ptr<KvCache> Backbone::NewCache(std::size_t capacity) {
  const auto& c = impl_->config;
  if (capacity == 0 || capacity > c.context)
    throw std::length_error("YuE2 KV cache capacity exceeds the model context");
  auto cache = std::unique_ptr<KvCache>(new KvCache());
  const std::size_t bytes =
      c.layers * c.kv_heads * capacity * c.head_dim * sizeof(std::uint16_t);
  RequireHip(hipMalloc(reinterpret_cast<void**>(&cache->keys_), bytes),
             "hipMalloc key cache");
  RequireHip(hipMalloc(reinterpret_cast<void**>(&cache->values_), bytes),
             "hipMalloc value cache");
  cache->capacity_ = capacity;
  return cache;
}

void Backbone::Append(KvCache& cache, std::span<const std::uint32_t> tokens,
                      VocabularyWindow window, std::vector<float>* logits) {
  impl_->Append(cache, tokens, window, logits);
}

void Backbone::TeacherForce(std::span<const std::uint32_t> tokens,
                            std::span<const std::uint32_t> rows,
                            VocabularyWindow window,
                            std::vector<float>* logits) {
  auto cache = NewCache(tokens.size());
  logits->clear();
  std::vector<float> row_logits;
  std::size_t done = 0;
  for (const std::uint32_t row : rows) {
    if (row < done || row >= tokens.size())
      throw std::invalid_argument("teacher-forced rows must be increasing");
    impl_->Append(*cache, tokens.subspan(done, row + 1 - done), window,
                  &row_logits);
    logits->insert(logits->end(), row_logits.begin(), row_logits.end());
    done = row + 1;
  }
}

void Backbone::Velocity(const KvCache& prefix, std::span<const float> state,
                        double raw_t, std::vector<float>* velocity) {
  auto& impl = *impl_;
  const std::size_t frames = state.size() / impl.config.latent_dim;
  impl.PrepareAcoustic(prefix, frames);
  RequireHip(hipMemcpyAsync(impl.state.get(), state.data(),
                            state.size() * sizeof(float), hipMemcpyHostToDevice,
                            impl.stream),
             "hipMemcpyAsync state");
  impl.VelocityDevice(prefix, impl.state.get(), frames, raw_t,
                      impl.first.get());
  velocity->resize(state.size());
  RequireHip(hipMemcpyAsync(velocity->data(), impl.first.get(),
                            state.size() * sizeof(float), hipMemcpyDeviceToHost,
                            impl.stream),
             "hipMemcpyAsync velocity");
  RequireHip(hipStreamSynchronize(impl.stream), "velocity");
}

bool Backbone::Synthesize(const KvCache& prefix, std::span<const float> noise,
                          int steps, const StepProgress& progress,
                          std::vector<float>* latents, std::string* error) {
  auto& impl = *impl_;
  const std::size_t elements = noise.size();
  const std::size_t frames = elements / impl.config.latent_dim;
  if (steps < 1 || frames == 0 || elements % impl.config.latent_dim != 0) {
    if (error != nullptr)
      *error = "invalid YuE2 acoustic request";
    return false;
  }
  impl.PrepareAcoustic(prefix, frames);
  RequireHip(
      hipMemcpyAsync(impl.state.get(), noise.data(), elements * sizeof(float),
                     hipMemcpyHostToDevice, impl.stream),
      "hipMemcpyAsync noise");
  const auto logit = [](double t) {
    if (t >= 1.0)
      return 20.0;
    return std::clamp(std::log(t / (1.0 - t)), -20.0, 20.0);
  };
  const double dt = 1.0 / steps;
  for (int step = 0; step < steps; ++step) {
    const double t = 1.0 - step * dt;
    impl.VelocityDevice(prefix, impl.state.get(), frames, logit(t),
                        impl.first.get());
    LaunchOdeStep(impl.state.get(), impl.first.get(),
                  static_cast<float>(dt / 2), elements, impl.midpoint.get(),
                  impl.stream);
    impl.VelocityDevice(prefix, impl.midpoint.get(), frames, logit(t - dt / 2),
                        impl.second.get());
    LaunchOdeStep(impl.state.get(), impl.second.get(), static_cast<float>(dt),
                  elements, impl.state.get(), impl.stream);
    RequireHip(hipStreamSynchronize(impl.stream), "acoustic step");
    if (progress && !progress(step + 1, steps)) {
      if (error != nullptr)
        *error = "cancelled";
      return false;
    }
  }
  latents->resize(elements);
  RequireHip(hipMemcpy(latents->data(), impl.state.get(),
                       elements * sizeof(float), hipMemcpyDeviceToHost),
             "hipMemcpy latents");
  RequireHip(hipGetLastError(), "acoustic launch");
  if (!std::ranges::all_of(*latents,
                           [](float value) { return std::isfinite(value); })) {
    if (error != nullptr)
      *error = "acoustic flow matching produced non-finite latents";
    return false;
  }
  return true;
}

}  // namespace gufo::models::yue2::hip
