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

  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  eager::Gemm planned;
  std::size_t rows_reserved{0};
  DeviceArray<float> hidden, q, k, v, o, gate, up, ffn;
  DeviceArray<std::uint16_t> query_heads, attention, scratch;
  DeviceArray<std::uint16_t> nar_keys, nar_values;
  DeviceArray<std::uint32_t> ids;
  DeviceArray<int> limits;
  DeviceArray<float> logits, logits_rounded, decode_scratch;
  DeviceArray<float> state, midpoint, first, second, projected;
  DeviceArray<std::uint16_t> frequencies, time_row;
  DeviceArray<float> time_hidden;
  std::size_t decode_scratch_capacity{0};

  ~Impl() {
    for (float* norm : norm_allocations)
      (void)hipFree(norm);
    if (weights_allocation != nullptr)
      (void)hipFree(weights_allocation);
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
    // Safetensors headers have arbitrary length; shift the copy so the payload
    // starts where 16-byte packed loads are legal.
    const std::size_t shift = (16 - (file->payload_offset() % 16)) % 16;
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
    ResidualNorm(nullptr, path(0).input_norm, rows);
    for (std::size_t l = 0; l < c.layers; ++l) {
      const Path& w = path(l);
      const std::array<eager::Bfloat16GemvGroup, 3> qkv{{
          {w.q, q.get(), qd},
          {w.k, k.get(), kvd},
          {w.v, v.get(), kvd},
      }};
      GemmGrouped(qkv, scratch.get(), rows, c.hidden);
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
      eager::LaunchBfloat16QkNormRoPEHeadMajor(
          q.get(), k.get(), v.get(), w.q_norm, w.k_norm, rows,
          static_cast<std::uint32_t>(c.heads),
          static_cast<std::uint32_t>(c.kv_heads),
          static_cast<std::uint32_t>(c.head_dim),
          static_cast<std::uint32_t>(pass.position), c.rope_theta,
          c.rms_epsilon, eager::RmsScale::kBfloat16, query_heads.get(), rows, 0,
          keys_out, values_out, stride, row, stream);
      if (!pass.nar && rows == 1) {
        eager::LaunchBfloat16DecodeAttention(
            query_heads.get(), kv_layer(pass.cache_keys, l),
            kv_layer(pass.cache_values, l), pass.capacity, pass.position + 1,
            c.heads, c.kv_heads, decode_scratch.get(), attention.get(), stream);
      } else {
        eager::KeySegments segments;
        const auto* cache_keys = kv_layer(pass.cache_keys, l);
        const auto* cache_values = kv_layer(pass.cache_values, l);
        if (pass.nar) {
          const int prefix = static_cast<int>(pass.prefix);
          const int total = prefix + static_cast<int>(rows);
          segments.keys[0] = cache_keys;
          segments.values[0] = cache_values;
          segments.end[0] = prefix;
          segments.stride[0] = static_cast<int>(pass.capacity);
          for (int s = 1; s < 3; ++s) {
            segments.keys[s] = nar_keys.get();
            segments.values[s] = nar_values.get();
            segments.end[s] = total;
            segments.stride[s] = static_cast<int>(rows);
          }
        } else {
          const int total = static_cast<int>(pass.position + rows);
          for (int s = 0; s < 3; ++s) {
            segments.keys[s] = cache_keys;
            segments.values[s] = cache_values;
            segments.end[s] = total;
            segments.stride[s] = static_cast<int>(pass.capacity);
          }
        }
        eager::FusedAttention128Segmented(
            query_heads.get(), segments, attention.get(),
            static_cast<int>(c.heads), static_cast<int>(rows), 0,
            static_cast<int>(rows), stream, pass.nar ? nullptr : limits.get(),
            static_cast<int>(c.heads / c.kv_heads));
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
      ResidualNorm(ffn.get(),
                   l + 1 < c.layers ? path(l + 1).input_norm : final_norm,
                   rows);
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
    const auto* head = static_cast<const std::uint8_t*>(lm_head) +
                       static_cast<std::size_t>(window.begin) * config.hidden *
                           sizeof(std::uint16_t);
    Gemm(head, scratch.get() + row * config.hidden, logits.get(), 1, count,
         config.hidden);
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
    impl->ReserveRows(kPrefillRows);
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
