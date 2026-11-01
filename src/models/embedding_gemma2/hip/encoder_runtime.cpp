#include "encoder_runtime.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <map>
#include <string>
#include <utility>

#include "audio_runtime.hpp"
#include "batch_scheduler.hpp"
#include "blas.hpp"
#include "image_processor.hpp"
#include "loader.hpp"
#include "ops.hpp"
#include "vision_ops.hpp"

namespace gufo::models::embedding_gemma2::hip {
namespace {

/// Owning device allocation.
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    HIP_CHECK(hipMalloc(&data_, bytes));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      gufo::hip::LogCleanupError(hipFree(data_));
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        gufo::hip::LogCleanupError(hipFree(data_));
      }
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  template<typename T = void>
  [[nodiscard]] T* get() const {
    return static_cast<T*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

  void Upload(std::size_t offset, const void* host, std::size_t bytes) const {
    HIP_CHECK(hipMemcpy(static_cast<std::byte*>(data_) + offset, host, bytes,
                        hipMemcpyHostToDevice));
  }

  /// Queues the copy on `stream`, ordered with the kernels that read it. The
  /// host memory must stay valid until the stream is synchronized.
  void UploadAsync(std::size_t offset, const void* host, std::size_t bytes,
                   hipStream_t stream) const {
    HIP_CHECK(hipMemcpyAsync(static_cast<std::byte*>(data_) + offset, host,
                             bytes, hipMemcpyHostToDevice, stream));
  }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

/// Concatenates BF16 tensors row-wise into one device buffer. The tensors
/// must share a column count.
DeviceBuffer UploadConcatenated(const TensorStore& store,
                                const std::vector<std::string>& names) {
  std::size_t total = 0;
  for (const std::string& name : names) {
    total += store.Find(name)->byte_count;
  }
  DeviceBuffer buffer(total);
  std::size_t offset = 0;
  for (const std::string& name : names) {
    const Tensor* tensor = store.Find(name);
    buffer.Upload(offset, tensor->data, tensor->byte_count);
    offset += tensor->byte_count;
  }
  return buffer;
}

/// Rows per chunk when a GEMM and its consumer run back to back. Chunking
/// bounds the float32 scratch. Unchunked, the gate/up buffer for 8,192 tokens
/// would be 8,192 x 4,096 x 4 bytes = 134 MB and the QKV buffer about 100 MB.
constexpr std::size_t kChunkTokens = 1024;
// Gemm pads a chunk up to a row bucket, so the scratch behind every chunk must
// hold a full chunk.
static_assert(kMaxSequenceTokens % kChunkTokens == 0);

/// Rotary cos/sin for every position up to the context limit, one pair of
/// [position][head_dim / 2] tables per head width and base. The code computes
/// them in double precision so the float tables are correctly rounded.
struct RopeTable {
  DeviceBuffer cos;
  DeviceBuffer sin;
};

RopeTable BuildRopeTable(std::uint32_t head_dim, double theta,
                         std::size_t positions) {
  const std::size_t half = head_dim / 2U;
  std::vector<float> cosines(positions * half);
  std::vector<float> sines(positions * half);
  for (std::size_t i = 0; i < half; ++i) {
    const double frequency =
        std::pow(theta, -static_cast<double>(2U * i) / head_dim);
    for (std::size_t p = 0; p < positions; ++p) {
      const double angle = static_cast<double>(p) * frequency;
      cosines[p * half + i] = static_cast<float>(std::cos(angle));
      sines[p * half + i] = static_cast<float>(std::sin(angle));
    }
  }
  RopeTable table{DeviceBuffer(cosines.size() * sizeof(float)),
                  DeviceBuffer(sines.size() * sizeof(float))};
  table.cos.Upload(0, cosines.data(), cosines.size() * sizeof(float));
  table.sin.Upload(0, sines.data(), sines.size() * sizeof(float));
  return table;
}

struct LayerWeights {
  DeviceBuffer input_norm, post_attention_norm, pre_feedforward_norm,
      post_feedforward_norm, layer_scalar;
  DeviceBuffer qkv, output, query_norm, key_norm;
  DeviceBuffer gate_up, down;
  DeviceBuffer ple_gate, ple_projection, ple_post_norm;
};

}  // namespace

struct EncoderRuntime::Impl {
  LoadResult loaded;
  std::string model_dir;
  struct Vision {
    DeviceBuffer input, position_table, projection;
    std::vector<LayerWeights> layers;
    DeviceBuffer pixels, pixels_bf16, state, normed, qkv, query, key, value,
        context, temp, gate_up, act, pooled, features;
    /// Soft tokens per image the scratch above holds.
    std::size_t capacity{0};
  };
  std::unique_ptr<Vision> vision;
  std::unique_ptr<AudioRuntime> audio;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  GemmLt gemm_lt;

  // Host staging for the per-batch index uploads. Asynchronous copies read
  // these until the stream is synchronized, so they outlive RunBatch.
  std::vector<std::uint32_t> host_ids, host_positions, host_begin, host_end,
      host_offsets;

  DeviceBuffer embed_table, final_norm, embedding_projection, ple_projection,
      ple_norm;
  std::vector<LayerWeights> layers;
  std::map<std::pair<std::uint32_t, float>, RopeTable> rope;

  // Scratch, sized for kMaxSequenceTokens.
  DeviceBuffer token_ids, positions, sequence_begin, sequence_end, offsets;
  DeviceBuffer state, embeddings_bf16, normed_bf16, qkv_f32, query_bf16,
      key_bf16, value_bf16, value_t_bf16, context_bf16, temp_f32, gate_up_f32,
      act_bf16, ple_f32, gate_f32, projected_f32, pooled_f32;
  // Target of the device probe that runs after a failed batch.
  DeviceBuffer probe;
  std::atomic<bool> lost{false};

  // Packs concurrent text calls into shared batches, gives each call with an
  // image a batch of its own, and runs one batch at a time.
  BatchScheduler scheduler{
      kMaxSequenceTokens,
      [this](const std::vector<const EncoderSequence*>& sequences,
             std::vector<std::vector<float>>* embeddings, std::string* error) {
        return OnDevice([&] { RunBatch(sequences, embeddings); }, error);
      }};

  ~Impl() {
    audio.reset();
    if (blas != nullptr) {
      gufo::hip::LogCleanupError(hipblasDestroy(blas));
    }
    if (stream != nullptr) {
      gufo::hip::LogCleanupError(hipStreamDestroy(stream));
    }
  }

  void Upload();
  void LoadVision();
  void ReserveVision(std::size_t soft_tokens);
  void RunVision(const ImagePatches& image, std::size_t offset);
  void Allocate();
  // Mirrors the text server's probe: after a reset even a 4-byte memset
  // fails, while an ordinary kernel or library failure leaves it working.
  [[nodiscard]] bool DeviceUsable();

  /// Runs one sequence for every GEMM row bucket, so hipBLASLt has tuned all
  /// of them and every kernel is loaded before the first request.
  void Warmup();

  /// Y[tokens, m] = X[tokens, k] * W[m, k]^T through the tuned hipBLASLt
  /// plan, or rocBLAS when hipBLASLt has none for the shape.
  void Gemm(const void* weights, const void* inputs, float* output,
            std::size_t tokens, std::size_t m, std::size_t k) {
    if (!gemm_lt.Run(weights, inputs, output, tokens, m, k, kChunkTokens,
                     stream)) {
      LaunchGemmBf16(blas, weights, inputs, output, tokens, m, k, stream);
    }
  }

  void RunBatch(const std::vector<const EncoderSequence*>& sequences,
                std::vector<std::vector<float>>* embeddings);
  // Runs scheduled device work and turns a failure into an error, confirming
  // a lost device with the probe.
  bool OnDevice(const std::function<void()>& work, std::string* error);
};

void EncoderRuntime::Impl::Upload() {
  const TensorStore& store = *loaded.store;
  const std::string prefix = kTextTensorPrefix;
  const auto upload = [&](const std::string& name) {
    const Tensor* tensor = store.Find(name);
    DeviceBuffer buffer(tensor->byte_count);
    buffer.Upload(0, tensor->data, tensor->byte_count);
    return buffer;
  };
  embed_table = upload(prefix + "embed_tokens.weight");
  final_norm = upload(prefix + "norm.weight");
  embedding_projection = upload(prefix + "embedding_projection.weight");
  ple_projection = upload(prefix + "ple.per_layer_model_projection.weight");
  ple_norm = upload(prefix + "ple.per_layer_projection_norm.weight");

  const TextConfig& text = loaded.config.text;
  for (const LayerGeometry& geometry : text.layers) {
    const auto key = std::make_pair(geometry.head_dim, geometry.rope_theta);
    if (!rope.contains(key)) {
      rope.emplace(key, BuildRopeTable(geometry.head_dim, geometry.rope_theta,
                                       kMaxSequenceTokens));
    }
  }
  layers.resize(text.num_hidden_layers);
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const std::string p = prefix + "layers." + std::to_string(index) + ".";
    LayerWeights& layer = layers[index];
    layer.input_norm = upload(p + "input_layernorm.weight");
    layer.post_attention_norm = upload(p + "post_attention_layernorm.weight");
    layer.pre_feedforward_norm = upload(p + "pre_feedforward_layernorm.weight");
    layer.post_feedforward_norm =
        upload(p + "post_feedforward_layernorm.weight");
    layer.layer_scalar = upload(p + "layer_scalar");
    layer.qkv = UploadConcatenated(
        store, {p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight"});
    layer.output = upload(p + "self_attn.o_proj.weight");
    layer.query_norm = upload(p + "self_attn.q_norm.weight");
    layer.key_norm = upload(p + "self_attn.k_norm.weight");
    layer.gate_up = UploadConcatenated(
        store, {p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"});
    layer.down = upload(p + "mlp.down_proj.weight");
    layer.ple_gate = upload(p + "ple_block.per_layer_input_gate.weight");
    layer.ple_projection = upload(p + "ple_block.per_layer_projection.weight");
    layer.ple_post_norm =
        upload(p + "ple_block.post_per_layer_input_norm.weight");
  }
}

void EncoderRuntime::Impl::LoadVision() {
  if (vision)
    return;
  std::string processor_error;
  if (!ValidateImageProcessor(model_dir, &processor_error))
    throw std::runtime_error(processor_error);
  auto weights = LoadModelDirectory(model_dir, true);
  if (!weights.ok)
    throw std::runtime_error(weights.error);
  auto candidate = std::make_unique<Vision>();
  const auto upload = [&](const std::string& name) {
    const auto* tensor = weights.store->Find(name);
    DeviceBuffer buffer(tensor->byte_count);
    buffer.Upload(0, tensor->data, tensor->byte_count);
    return buffer;
  };
  candidate->input = upload("vision_tower.patch_embedder.input_proj.weight");
  candidate->position_table =
      upload("vision_tower.patch_embedder.position_embedding_table");
  candidate->projection = upload("embed_vision.embedding_projection.weight");
  candidate->layers.resize(16);
  for (unsigned i = 0; i < 16; ++i) {
    auto& layer = candidate->layers[i];
    const std::string p =
        "vision_tower.encoder.layers." + std::to_string(i) + ".";
    layer.input_norm = upload(p + "input_layernorm.weight");
    layer.post_attention_norm = upload(p + "post_attention_layernorm.weight");
    layer.pre_feedforward_norm = upload(p + "pre_feedforward_layernorm.weight");
    layer.post_feedforward_norm =
        upload(p + "post_feedforward_layernorm.weight");
    layer.qkv = UploadConcatenated(*weights.store,
                                   {p + "self_attn.q_proj.linear.weight",
                                    p + "self_attn.k_proj.linear.weight",
                                    p + "self_attn.v_proj.linear.weight"});
    layer.output = upload(p + "self_attn.o_proj.linear.weight");
    layer.query_norm = upload(p + "self_attn.q_norm.weight");
    layer.key_norm = upload(p + "self_attn.k_norm.weight");
    layer.gate_up = UploadConcatenated(
        *weights.store,
        {p + "mlp.gate_proj.linear.weight", p + "mlp.up_proj.linear.weight"});
    layer.down = upload(p + "mlp.down_proj.linear.weight");
  }
  vision = std::move(candidate);
}

// The scratch starts at the default budget and grows to the smallest budget
// that holds a larger image, so requests at the default keep their footprint.
void EncoderRuntime::Impl::ReserveVision(std::size_t soft_tokens) {
  auto& v = *vision;
  if (soft_tokens <= v.capacity)
    return;
  // Validation caps an image at kMaxImageSoftTokens, the largest budget.
  std::size_t tokens = kMaxImageSoftTokens;
  for (const std::size_t budget : kImageSoftTokenBudgets) {
    if (budget >= std::max<std::size_t>(soft_tokens, kDefaultImageSoftTokens)) {
      tokens = budget;
      break;
    }
  }
  // Earlier images may still be using the old buffers.
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::size_t n = tokens * 9, h = 768;
  v.pixels = DeviceBuffer(n * h * sizeof(float));
  v.pixels_bf16 = DeviceBuffer(n * h * 2);
  v.state = DeviceBuffer(n * h * sizeof(float));
  v.normed = DeviceBuffer(n * h * 2);
  v.qkv = DeviceBuffer(n * h * 3 * sizeof(float));
  v.query = DeviceBuffer(n * h * 2);
  v.key = DeviceBuffer(n * h * 2);
  v.value = DeviceBuffer(n * h * 2);
  v.context = DeviceBuffer(n * h * 2);
  v.temp = DeviceBuffer(n * h * sizeof(float));
  v.gate_up = DeviceBuffer(std::min(n, kChunkTokens) * 6144 * sizeof(float));
  v.act = DeviceBuffer(n * 3072 * 2);
  v.pooled = DeviceBuffer(tokens * h * 2);
  v.features = DeviceBuffer(tokens * 512 * sizeof(float));
  v.capacity = tokens;
}

void EncoderRuntime::Impl::RunVision(const ImagePatches& image,
                                     std::size_t offset) {
  LoadVision();
  ReserveVision(image.soft_tokens());
  auto& v = *vision;
  const std::size_t n = std::size_t{image.rows} * image.columns;
  v.pixels.UploadAsync(0, image.pixels.data(),
                       image.pixels.size() * sizeof(float), stream);
  LaunchVisionPixels(v.pixels.get<float>(), v.pixels_bf16.get(), n * 768,
                     stream);
  LaunchGemmBf16(blas, v.input.get(), v.pixels_bf16.get(), v.temp.get<float>(),
                 n, 768, 768, stream);
  LaunchVisionPositions(v.temp.get<float>(), v.position_table.get(),
                        v.state.get<float>(), n, image.columns, stream);
  for (const auto& layer : v.layers) {
    LaunchRmsNormToBf16(v.state.get<float>(), layer.input_norm.get(),
                        v.normed.get(), n, 768, 1e-6F, stream);
    LaunchGemmBf16(blas, layer.qkv.get(), v.normed.get(), v.qkv.get<float>(), n,
                   2304, 768, stream);
    LaunchVisionQkv(v.qkv.get<float>(), layer.query_norm.get(),
                    layer.key_norm.get(), v.query.get(), v.key.get(),
                    v.value.get(), n, image.columns, stream);
    LaunchVisionAttention(v.query.get(), v.key.get(), v.value.get(),
                          v.context.get(), n, stream);
    LaunchGemmBf16(blas, layer.output.get(), v.context.get(),
                   v.temp.get<float>(), n, 768, 768, stream);
    LaunchRmsNormResidual(v.temp.get<float>(), layer.post_attention_norm.get(),
                          v.state.get<float>(), n, 768, 1e-6F, stream);
    LaunchRmsNormToBf16(v.state.get<float>(), layer.pre_feedforward_norm.get(),
                        v.normed.get(), n, 768, 1e-6F, stream);
    for (std::size_t row = 0; row < n; row += kChunkTokens) {
      const std::size_t count = std::min(kChunkTokens, n - row);
      LaunchGemmBf16(blas, layer.gate_up.get(),
                     v.normed.get<std::uint16_t>() + row * 768,
                     v.gate_up.get<float>(), count, 6144, 768, stream);
      LaunchGeglu(v.gate_up.get<float>(),
                  v.act.get<std::uint16_t>() + row * 3072, count, 3072, stream);
    }
    LaunchGemmBf16(blas, layer.down.get(), v.act.get(), v.temp.get<float>(), n,
                   768, 3072, stream);
    LaunchRmsNormResidual(v.temp.get<float>(),
                          layer.post_feedforward_norm.get(),
                          v.state.get<float>(), n, 768, 1e-6F, stream);
  }
  LaunchVisionPool(v.state.get<float>(), v.pooled.get(), image.rows,
                   image.columns, stream);
  LaunchGemmBf16(blas, v.projection.get(), v.pooled.get(),
                 v.features.get<float>(), image.soft_tokens(), 512, 768,
                 stream);
  LaunchVisionInsert(v.features.get<float>(), state.get<float>() + offset * 512,
                     embeddings_bf16.get<std::uint16_t>() + offset * 512,
                     image.soft_tokens(), stream);
  HIP_CHECK(hipGetLastError());
}

void EncoderRuntime::Impl::Allocate() {
  const TextConfig& text = loaded.config.text;
  std::size_t qkv_width = 0;
  std::size_t query_width = 0;
  std::size_t key_width = 0;
  for (const LayerGeometry& layer : text.layers) {
    query_width = std::max<std::size_t>(
        query_width, std::size_t{text.num_attention_heads} * layer.head_dim);
    key_width = std::max<std::size_t>(
        key_width, std::size_t{layer.num_key_value_heads} * layer.head_dim);
    qkv_width = std::max(qkv_width, query_width + 2U * key_width);
  }
  const std::size_t n = kMaxSequenceTokens;
  const std::size_t hidden = text.hidden_size;
  const std::size_t per_layer = text.hidden_size_per_layer_input;
  token_ids = DeviceBuffer(n * sizeof(std::uint32_t));
  positions = DeviceBuffer(n * sizeof(std::uint32_t));
  sequence_begin = DeviceBuffer(n * sizeof(std::uint32_t));
  sequence_end = DeviceBuffer(n * sizeof(std::uint32_t));
  offsets = DeviceBuffer((n + 1U) * sizeof(std::uint32_t));
  state = DeviceBuffer(n * hidden * sizeof(float));
  embeddings_bf16 = DeviceBuffer(n * hidden * 2U);
  normed_bf16 = DeviceBuffer(n * hidden * 2U);
  // Only one chunk of rows is live at a time.
  qkv_f32 = DeviceBuffer(std::min(n, kChunkTokens) * qkv_width * sizeof(float));
  query_bf16 = DeviceBuffer(n * query_width * 2U);
  key_bf16 = DeviceBuffer(n * key_width * 2U);
  // The transposed copy the attention kernel reads has a stride that is a
  // multiple of 32 tokens. The transpose zero-fills the tail of each column.
  value_bf16 = DeviceBuffer(n * key_width * 2U);
  value_t_bf16 = DeviceBuffer(ValueStride(n) * key_width * 2U);
  context_bf16 = DeviceBuffer(n * query_width * 2U);
  temp_f32 = DeviceBuffer(n * hidden * sizeof(float));
  gate_up_f32 = DeviceBuffer(std::min(n, kChunkTokens) * 2U *
                             text.intermediate_size * sizeof(float));
  act_bf16 = DeviceBuffer(n * text.intermediate_size * 2U);
  ple_f32 = DeviceBuffer(n * per_layer * sizeof(float));
  gate_f32 = DeviceBuffer(n * per_layer * sizeof(float));
  projected_f32 = DeviceBuffer(n * text.embedding_dim * sizeof(float));
  pooled_f32 = DeviceBuffer(n * text.embedding_dim * sizeof(float));
}

void EncoderRuntime::Impl::Warmup() {
  std::vector<std::vector<float>> unused;
  std::size_t rows = 0;
  do {
    rows = GemmLt::Bucket(rows + 1, kChunkTokens);
    const EncoderSequence filler{std::vector<std::uint32_t>(rows, 1000), {}};
    unused.clear();
    RunBatch({&filler}, &unused);
  } while (rows < kChunkTokens);
}

void EncoderRuntime::Impl::RunBatch(
    const std::vector<const EncoderSequence*>& sequences,
    std::vector<std::vector<float>>* embeddings) {
  const TextConfig& text = loaded.config.text;
  const std::size_t hidden = text.hidden_size;
  const std::size_t per_layer = text.hidden_size_per_layer_input;
  const float epsilon = text.rms_norm_eps;

  // Pack the sequences and describe each token's sequence bounds.
  host_ids.clear();
  host_positions.clear();
  host_begin.clear();
  host_end.clear();
  host_offsets.assign(1, 0);
  for (const EncoderSequence* sequence : sequences) {
    const auto begin = static_cast<std::uint32_t>(host_ids.size());
    const auto end =
        begin + static_cast<std::uint32_t>(sequence->tokens.size());
    for (std::size_t t = 0; t < sequence->tokens.size(); ++t) {
      host_ids.push_back(sequence->tokens[t]);
      host_positions.push_back(static_cast<std::uint32_t>(t));
      host_begin.push_back(begin);
      host_end.push_back(end);
    }
    host_offsets.push_back(end);
  }
  const std::size_t tokens = host_ids.size();
  token_ids.UploadAsync(0, host_ids.data(), tokens * sizeof(std::uint32_t),
                        stream);
  positions.UploadAsync(0, host_positions.data(),
                        tokens * sizeof(std::uint32_t), stream);
  sequence_begin.UploadAsync(0, host_begin.data(),
                             tokens * sizeof(std::uint32_t), stream);
  sequence_end.UploadAsync(0, host_end.data(), tokens * sizeof(std::uint32_t),
                           stream);
  offsets.UploadAsync(0, host_offsets.data(),
                      host_offsets.size() * sizeof(std::uint32_t), stream);

  LaunchEmbeddingLookup(embed_table.get(), token_ids.get<std::uint32_t>(),
                        state.get<float>(), embeddings_bf16.get(), tokens,
                        hidden, kEmbedScale, stream);

  for (std::size_t s = 0; s < sequences.size(); ++s) {
    for (const auto& clip : sequences[s]->audio) {
      if (!audio)
        audio = std::make_unique<AudioRuntime>(model_dir, stream);
      const auto* features =
          audio->Run(clip.features, sequences[s]->is_cancelled);
      const std::size_t offset = host_offsets[s] + clip.offset;
      LaunchVisionInsert(features, state.get<float>() + offset * 512,
                         embeddings_bf16.get<std::uint16_t>() + offset * 512,
                         clip.features.soft_tokens(), stream);
    }
    for (const auto& image : sequences[s]->images) {
      if (sequences[s]->is_cancelled && sequences[s]->is_cancelled())
        throw std::runtime_error(kCancelledMessage);
      RunVision(image.patches, host_offsets[s] + image.offset);
      // Bound outstanding video work to one frame, so a disconnected long
      // clip releases the device before encoding the remaining frames.
      if (image.token_id == kVideoTokenId)
        HIP_CHECK(hipStreamSynchronize(stream));
    }
  }

  const float ple_scale = 1.0F / std::sqrt(static_cast<float>(hidden));
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const LayerWeights& layer = layers[index];
    const LayerGeometry& geometry = text.layers[index];
    const QkvLayout layout{text.num_attention_heads,
                           geometry.num_key_value_heads, geometry.head_dim};
    const std::size_t query_width =
        std::size_t{layout.num_heads} * layout.head_dim;
    const std::size_t key_width =
        std::size_t{layout.num_key_value_heads} * layout.head_dim;
    const bool full = geometry.type == LayerType::kFullAttention;
    const RopeTable& rope_table =
        rope.at(std::make_pair(geometry.head_dim, geometry.rope_theta));

    LaunchRmsNormToBf16(state.get<float>(), layer.input_norm.get(),
                        normed_bf16.get(), tokens, hidden, epsilon, stream);
    for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
      const std::size_t rows = std::min(kChunkTokens, tokens - row);
      Gemm(layer.qkv.get(), normed_bf16.get<std::uint16_t>() + row * hidden,
           qkv_f32.get<float>(), rows, query_width + 2U * key_width, hidden);
      LaunchQkvNormRope(qkv_f32.get<float>(), layer.query_norm.get(),
                        layer.key_norm.get(),
                        query_bf16.get<std::uint16_t>() + row * query_width,
                        key_bf16.get<std::uint16_t>() + row * key_width,
                        value_bf16.get<std::uint16_t>() + row * key_width,
                        positions.get<std::uint32_t>() + row, rows, layout,
                        rope_table.cos.get<float>(),
                        rope_table.sin.get<float>(), epsilon, stream);
    }
    LaunchTransposeValues(value_bf16.get(), value_t_bf16.get(), tokens,
                          static_cast<std::uint32_t>(key_width), stream);
    if (!LaunchAttention(query_bf16.get(), key_bf16.get(), value_t_bf16.get(),
                         context_bf16.get(),
                         sequence_begin.get<std::uint32_t>(),
                         sequence_end.get<std::uint32_t>(), tokens, layout,
                         full ? 0U : text.sliding_window, stream)) {
      throw std::runtime_error("EmbeddingGemma 2 attention geometry");
    }
    for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
      const std::size_t rows = std::min(kChunkTokens, tokens - row);
      Gemm(layer.output.get(),
           context_bf16.get<std::uint16_t>() + row * query_width,
           temp_f32.get<float>(), rows, hidden, query_width);
      LaunchRmsNormResidual(
          temp_f32.get<float>(), layer.post_attention_norm.get(),
          state.get<float>() + row * hidden, rows, hidden, epsilon, stream);
    }

    LaunchRmsNormToBf16(state.get<float>(), layer.pre_feedforward_norm.get(),
                        normed_bf16.get(), tokens, hidden, epsilon, stream);
    for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
      const std::size_t rows = std::min(kChunkTokens, tokens - row);
      Gemm(layer.gate_up.get(), normed_bf16.get<std::uint16_t>() + row * hidden,
           gate_up_f32.get<float>(), rows, 2U * text.intermediate_size, hidden);
      LaunchGeglu(gate_up_f32.get<float>(),
                  act_bf16.get<std::uint16_t>() + row * text.intermediate_size,
                  rows, text.intermediate_size, stream);
    }
    for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
      const std::size_t rows = std::min(kChunkTokens, tokens - row);
      Gemm(layer.down.get(),
           act_bf16.get<std::uint16_t>() + row * text.intermediate_size,
           temp_f32.get<float>(), rows, hidden, text.intermediate_size);
      LaunchRmsNormResidual(
          temp_f32.get<float>(), layer.post_feedforward_norm.get(),
          state.get<float>() + row * hidden, rows, hidden, epsilon, stream);
    }

    // The per-layer input block takes this layer's slice of the per-layer
    // inputs, then runs a gated bottleneck that mixes it into the residual
    // stream.
    for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
      const std::size_t rows = std::min(kChunkTokens, tokens - row);
      float* const state_rows = state.get<float>() + row * hidden;
      std::uint16_t* const normed_rows =
          normed_bf16.get<std::uint16_t>() + row * hidden;
      Gemm(ple_projection.get<std::byte>() + index * per_layer * hidden * 2U,
           embeddings_bf16.get<std::uint16_t>() + row * hidden,
           ple_f32.get<float>(), rows, per_layer, hidden);
      LaunchScaledRmsNorm(ple_f32.get<float>(), ple_norm.get(), rows, per_layer,
                          ple_scale, epsilon, stream);
      LaunchFloatToBf16(state_rows, normed_rows, rows * hidden, stream);
      Gemm(layer.ple_gate.get(), normed_rows, gate_f32.get<float>(), rows,
           per_layer, hidden);
      LaunchPleGate(gate_f32.get<float>(), ple_f32.get<float>(), act_bf16.get(),
                    rows * per_layer, stream);
      Gemm(layer.ple_projection.get(), act_bf16.get(), temp_f32.get<float>(),
           rows, hidden, per_layer);
      LaunchPleResidual(temp_f32.get<float>(), layer.ple_post_norm.get(),
                        layer.layer_scalar.get(), state_rows, rows, hidden,
                        epsilon, stream);
    }
    HIP_CHECK(hipGetLastError());
  }

  LaunchRmsNormToBf16(state.get<float>(), final_norm.get(), normed_bf16.get(),
                      tokens, hidden, epsilon, stream);
  for (std::size_t row = 0; row < tokens; row += kChunkTokens) {
    Gemm(embedding_projection.get(),
         normed_bf16.get<std::uint16_t>() + row * hidden,
         projected_f32.get<float>() + row * text.embedding_dim,
         std::min(kChunkTokens, tokens - row), text.embedding_dim, hidden);
  }
  const std::size_t count = sequences.size();
  if (!LaunchMeanPoolNormalize(
          projected_f32.get<float>(), offsets.get<std::uint32_t>(),
          pooled_f32.get<float>(), count, text.embedding_dim, stream)) {
    throw std::runtime_error("EmbeddingGemma 2 pooling width");
  }
  HIP_CHECK(hipGetLastError());

  std::vector<float> host(count * text.embedding_dim);
  HIP_CHECK(hipMemcpyAsync(host.data(), pooled_f32.get(),
                           host.size() * sizeof(float), hipMemcpyDeviceToHost,
                           stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  for (std::size_t s = 0; s < count; ++s) {
    embeddings->emplace_back(
        host.begin() + static_cast<std::ptrdiff_t>(s * text.embedding_dim),
        host.begin() +
            static_cast<std::ptrdiff_t>((s + 1U) * text.embedding_dim));
  }
}

bool EncoderRuntime::Impl::OnDevice(const std::function<void()>& work,
                                    std::string* error) {
  if (lost.load(std::memory_order_acquire)) {
    *error = kDeviceLostMessage;
    return false;
  }
  try {
    work();
    return true;
  } catch (const std::exception& failure) {
    // Kernels from the failed work may still be queued. They read the index
    // buffers the next batch overwrites, so drain the stream before the next
    // batch starts.
    gufo::hip::LogCleanupError(hipStreamSynchronize(stream));
    if (!DeviceUsable()) {
      lost.store(true, std::memory_order_release);
      *error = std::string(kDeviceLostMessage) + ": " + failure.what();
      return false;
    }
    *error = std::string("embedding_gemma2: ") + failure.what();
    return false;
  }
}

bool EncoderRuntime::Impl::DeviceUsable() {
  (void)hipGetLastError();
  return hipMemsetAsync(probe.get(), 0, probe.bytes(), stream) == hipSuccess &&
         hipStreamSynchronize(stream) == hipSuccess;
}

EncoderRuntime::EncoderRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

EncoderRuntime::~EncoderRuntime() = default;

std::unique_ptr<EncoderRuntime> EncoderRuntime::Create(
    const std::string& model_dir, std::string* error) {
  const auto fail =
      [&](const std::string& message) -> std::unique_ptr<EncoderRuntime> {
    if (error != nullptr) {
      *error = message;
    }
    return nullptr;
  };
  auto impl = std::make_unique<Impl>();
  try {
    impl->model_dir = model_dir;
    impl->loaded = LoadModelDirectory(model_dir);
    if (!impl->loaded.ok) {
      return fail(impl->loaded.error);
    }
    HIP_CHECK(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking));
    HIPBLAS_CHECK(hipblasCreate(&impl->blas));
    impl->Upload();
    impl->Allocate();
    impl->probe = DeviceBuffer(sizeof(std::uint32_t));
    // The device now holds every weight. Unmapping the checkpoint releases
    // its page-cache residency, about 650 MB of host memory.
    impl->loaded.store.reset();
    impl->loaded.mappings.clear();
    impl->Warmup();
    return std::unique_ptr<EncoderRuntime>(new EncoderRuntime(std::move(impl)));
  } catch (const std::exception& failure) {
    // Warm-up copies may still be queued. They read the host staging vectors
    // that `impl` is about to destroy.
    if (impl != nullptr && impl->stream != nullptr) {
      gufo::hip::LogCleanupError(hipStreamSynchronize(impl->stream));
    }
    return fail(std::string("embedding_gemma2: ") + failure.what());
  }
}

const ModelConfig& EncoderRuntime::config() const {
  return impl_->loaded.config;
}

bool EncoderRuntime::device_lost() const noexcept {
  return impl_->lost.load(std::memory_order_acquire);
}

bool EncoderRuntime::Embed(
    const std::vector<std::vector<std::uint32_t>>& sequences,
    std::vector<std::vector<float>>* embeddings, std::string* error,
    const std::function<bool()>& is_cancelled) {
  std::vector<EncoderSequence> inputs;
  inputs.reserve(sequences.size());
  for (const auto& tokens : sequences)
    inputs.push_back({tokens, {}});
  return EmbedMultimodal(inputs, embeddings, error, is_cancelled);
}

bool EncoderRuntime::EmbedMultimodal(
    const std::vector<EncoderSequence>& sequences,
    std::vector<std::vector<float>>* embeddings, std::string* error,
    const std::function<bool()>& is_cancelled) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  const TextConfig& text = impl_->loaded.config.text;
  for (const auto& sequence : sequences) {
    if (sequence.tokens.empty()) {
      return fail("embedding_gemma2: empty token sequence");
    }
    if (sequence.tokens.size() > kMaxSequenceTokens) {
      return fail("embedding_gemma2: sequence exceeds " +
                  std::to_string(kMaxSequenceTokens) + " tokens");
    }
    std::size_t previous_end = 0;
    for (const auto& image : sequence.images) {
      const auto& patches = image.patches;
      const std::size_t n = patches.soft_tokens();
      if (patches.rows == 0 || patches.columns == 0 || patches.rows % 3 ||
          patches.columns % 3 || n > kMaxImageSoftTokens ||
          patches.pixels.size() != n * 9 * 768 || image.offset < previous_end ||
          image.offset > sequence.tokens.size() ||
          n > sequence.tokens.size() - image.offset) {
        return fail("embedding_gemma2: invalid image patches or token offset");
      }
      for (std::size_t j = 0; j < n; ++j) {
        if ((image.token_id != kImageTokenId &&
             image.token_id != kVideoTokenId) ||
            sequence.tokens[image.offset + j] != image.token_id) {
          return fail("embedding_gemma2: image placeholder mismatch");
        }
      }
      previous_end = image.offset + n;
    }
    previous_end = 0;
    for (const auto& clip : sequence.audio) {
      const auto& features = clip.features;
      const std::size_t n = features.soft_tokens();
      bool padding = false;
      for (const auto valid : features.mask) {
        if (valid > 1 || (padding && valid))
          return fail("embedding_gemma2: invalid audio feature mask");
        padding = padding || !valid;
      }
      if (features.mask.empty() || features.mask.size() > 2999 || n == 0 ||
          features.values.size() != features.mask.size() * 128 ||
          !std::ranges::all_of(features.values,
                               [](float x) { return std::isfinite(x); }) ||
          clip.offset < previous_end || clip.offset > sequence.tokens.size() ||
          n > sequence.tokens.size() - clip.offset)
        return fail("embedding_gemma2: invalid audio features or token offset");
      for (std::size_t j = 0; j < n; ++j)
        if (sequence.tokens[clip.offset + j] != kAudioTokenId)
          return fail("embedding_gemma2: audio placeholder mismatch");
      previous_end = clip.offset + n;
    }
    for (const std::uint32_t id : sequence.tokens) {
      if (id >= text.vocab_size) {
        return fail("embedding_gemma2: token id out of range");
      }
    }
  }
  std::string failure;
  const auto outcome =
      impl_->scheduler.Run(sequences, is_cancelled, embeddings, &failure);
  if (outcome == BatchScheduler::Outcome::kOk) {
    return true;
  }
  return fail(outcome == BatchScheduler::Outcome::kCancelled ? kCancelledMessage
                                                             : failure);
}

}  // namespace gufo::models::embedding_gemma2::hip
