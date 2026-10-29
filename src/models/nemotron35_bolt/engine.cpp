#include "src/models/nemotron35_bolt/engine.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron35_bolt/drafter.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/device_model.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/gemm.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/gemv.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/mmq/nb_mmq.h"
#include "src/models/nemotron35_bolt/kernels/rocm/sample.hpp"
#include "src/models/nemotron35_bolt/weights.hpp"

namespace gufo::models::nemotron35_bolt {
namespace {

/// Tokens per forward chunk; prefill splits longer appends.
constexpr int kChunk = 2048;
/// Most tokens one Verify pass takes: a drafted block.
constexpr int kVerifyMax = 8;

/// Device buffer owned by a session.
template<class T>
struct Buffer {
  T* data{nullptr};
  std::size_t count{0};

  Buffer() = default;
  Buffer(Buffer&& other) noexcept
      : data(std::exchange(other.data, nullptr)), count(other.count) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (data != nullptr)
      static_cast<void>(hipFree(data));
  }
  /// (Re)allocates room for `n` elements; the contents are undefined.
  bool Allocate(std::size_t n) {
    if (data != nullptr)
      static_cast<void>(hipFree(data));
    data = nullptr;
    count = n;
    return hipMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)) ==
           hipSuccess;
  }
};

}  // namespace

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->ResidentBytes();
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  auto fail = [&](std::string message) -> std::shared_ptr<Model> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  std::shared_ptr<Model> m(new Model());
  m->options_ = options;
  m->name_ = std::filesystem::path(model_path).stem().string();
  std::string error;
  m->reader_ = core::GgufReader::OpenFile(model_path, &error);
  if (!m->reader_)
    return fail("open " + model_path + ": " + error);
  auto weights = ModelWeights::Bind(*m->reader_, &error);
  if (!weights)
    return fail(error);
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.max_context == 0 ||
      options.max_context > m->weights_->config.context_length)
    return fail("max_context must be between 1 and the native context");
  m->tokenizer_ = Tokenizer::FromGguf(*m->reader_, &error);
  if (!m->tokenizer_)
    return fail(error);
  if (nb_mmq_init(0) != 0)
    return fail("quantized matrix kernels failed to initialize");
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, &error);
  if (!m->device_)
    return fail(error);
  return m;
}

bool Model::LoadDrafter(const std::string& path, std::string* error_msg) {
  drafter_ = Drafter::Load(path, weights_->config, error_msg);
  // A batch state built without the drafter captures no features.
  const std::lock_guard lock(batch_mutex_);
  batch_.reset();
  return drafter_ != nullptr;
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  const Config& c = weights_->config;
  std::size_t bytes = 0;
  for (const BlockType b : c.blocks) {
    if (b == BlockType::kAttention)
      bytes += 2ull * context * c.AttentionKvDim() * sizeof(__half);
    if (b == BlockType::kMamba)
      bytes +=
          sizeof(float) * (static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                               c.SsmConvChannels() +
                           static_cast<std::size_t>(c.ssm_heads) *
                               c.SsmHeadDim() * c.ssm_state_size);
  }
  return bytes;
}

struct Session::State {
  hipStream_t stream{nullptr};
  /// Runs the shared expert of short chunks beside the routed experts,
  /// forked from and joined back into `stream`.
  hipStream_t side{nullptr};
  hipEvent_t fork{nullptr};
  hipEvent_t join{nullptr};
  hipblasHandle_t blas{nullptr};
  // Per block; empty for blocks of the other kinds.
  std::vector<Buffer<__half>> k_cache;
  std::vector<Buffer<__half>> v_cache;
  std::vector<Buffer<float>> conv_state;
  std::vector<Buffer<float>> ssm_state;
  // Chunk scratch.
  Buffer<std::int32_t> tokens;
  Buffer<float> x, h, proj, xbc, y, out, q, k, v, ctx;
  /// Prefill inputs of the NVFP4 Mamba projections, as FP16.
  Buffer<__half> h16, y16;
  Buffer<float> router, weights, latent, up, down, acc, shared_up, shared;
  Buffer<std::int32_t> ids;
  /// Prefill expert routing for the IQ2_XXS matrix products (gemm.hpp).
  Buffer<std::int32_t> route_cursor, route_order, route_count;
  Buffer<int4> route_tiles;
  rocm::Iq2xxsRoute route;
  /// Q8_1 activations of short chunks for the vector products.
  Buffer<std::uint8_t> xq;
  Buffer<std::int32_t>
      position;  ///< Device copy of the chunk's first position.
  Buffer<float> attention_partial;  ///< Split decode attention partials.
  Buffer<float> last_logits;        ///< One row, written by the decode graph.
  Buffer<float> logits;             ///< One row per token when requested.
  /// Device summaries of returned rows for sampling, and their host copies.
  Buffer<std::uint8_t> head_scratch;
  Buffer<rocm::LogitHeadRecord> head_records;
  Buffer<rocm::LogitHeadCandidate> head_candidates;
  std::vector<rocm::LogitHeadRecord> host_head_records;
  std::vector<rocm::LogitHeadCandidate> host_head_candidates;
  /// Single-token forward with its head, captured once and replayed: the
  /// buffers and shapes are fixed, the position is read from device memory.
  hipGraphExec_t decode_graph{nullptr};
  bool graph_unavailable{false};
  // Drafter: trunk features of the chunk (FeatureSize floats per token) at
  // the blocks in feature_slot, and the first position they inject at.
  std::unique_ptr<DraftContext> draft;
  Buffer<float> features;
  Buffer<std::int32_t> inject_position;
  std::vector<int> feature_slot;  ///< Per block: -1 or its feature column.
  int feature_width{0};
  int inject_skip{0};  ///< Leading chunk rows the drafter will never read.
  // Verify keeps, per Mamba block, the chunk's ssm_in rows and convolved
  // x|B|C so Commit can replay the accepted prefix into the states.
  std::vector<Buffer<float>> verify_proj, verify_xbc;
  // Batched decoding (BatchState only): `rows` sessions, one token each.
  // Per block, the rows' conv|SSM states or K|V caches ([block][2][rows]),
  // and their positions.
  int rows{0};
  Buffer<void*> row_tables;
  Buffer<std::int32_t> row_positions;

  ~State() {
    if (decode_graph != nullptr)
      static_cast<void>(hipGraphExecDestroy(decode_graph));
    if (blas != nullptr)
      static_cast<void>(hipblasDestroy(blas));
    if (side != nullptr) {
      static_cast<void>(hipStreamSynchronize(side));
      static_cast<void>(hipStreamDestroy(side));
    }
    for (hipEvent_t e : {fork, join}) {
      if (e != nullptr)
        static_cast<void>(hipEventDestroy(e));
    }
    if (stream != nullptr) {
      static_cast<void>(hipStreamSynchronize(stream));
      nb_mmq_release_stream(stream);
      static_cast<void>(hipStreamDestroy(stream));
    }
  }
};

namespace {

/// The stream, side stream, events and hipBLAS handle of a state.
bool CreateStreams(Session::State& st) {
  return hipStreamCreateWithFlags(&st.stream, hipStreamNonBlocking) ==
             hipSuccess &&
         hipStreamCreateWithFlags(&st.side, hipStreamNonBlocking) ==
             hipSuccess &&
         hipEventCreateWithFlags(&st.fork, hipEventDisableTiming) ==
             hipSuccess &&
         hipEventCreateWithFlags(&st.join, hipEventDisableTiming) ==
             hipSuccess &&
         hipblasCreate(&st.blas) == HIPBLAS_STATUS_SUCCESS &&
         hipblasSetStream(st.blas, st.stream) == HIPBLAS_STATUS_SUCCESS &&
         nb_mmq_bind_stream(st.stream) == 0;
}

/// Scratch for chunks of up to `T` tokens and the heads of kVerifyMax rows.
bool AllocateChunk(Session::State& st, const Config& c,
                   const rocm::DeviceModel& device, std::size_t T) {
  std::uint32_t max_ff = 0;
  std::uint32_t max_experts = 0;
  for (std::size_t il = 0; il < c.num_layers; ++il) {
    if (c.blocks[il] != BlockType::kMoe)
      continue;
    max_ff = std::max(max_ff, c.expert_ff[il]);
    max_experts =
        std::max(max_experts,
                 static_cast<std::uint32_t>(device.layers()[il].num_experts));
  }
  const std::size_t H = c.hidden_size;
  std::uint32_t max_used = 0;
  for (auto u : c.experts_used)
    max_used = std::max(max_used, u);
  const bool ok =
      st.tokens.Allocate(T) && st.x.Allocate(T * H) && st.h.Allocate(T * H) &&
      st.proj.Allocate(T * c.SsmInProjection()) &&
      st.xbc.Allocate(T * c.SsmConvChannels()) &&
      st.y.Allocate(T * c.ssm_inner_size) && st.out.Allocate(T * H) &&
      st.h16.Allocate(T * H) && st.y16.Allocate(T * c.ssm_inner_size) &&
      st.q.Allocate(T * c.AttentionQDim()) &&
      st.k.Allocate(T * c.AttentionKvDim()) &&
      st.v.Allocate(T * c.AttentionKvDim()) &&
      st.ctx.Allocate(T * c.AttentionQDim()) &&
      st.router.Allocate(T * max_experts) &&
      st.weights.Allocate(T * max_used) && st.ids.Allocate(T * max_used) &&
      st.route_cursor.Allocate(max_experts) &&
      st.route_order.Allocate(T * max_used) && st.route_count.Allocate(1) &&
      st.route_tiles.Allocate(static_cast<std::size_t>(
          rocm::Iq2xxsRouteMaxTiles(static_cast<int>(T * max_used),
                                    static_cast<int>(max_experts)))) &&
      st.latent.Allocate(T * c.latent_size) &&
      st.up.Allocate(T * max_used * max_ff) &&
      st.down.Allocate(T * max_used * c.latent_size) &&
      st.acc.Allocate(T * c.latent_size) &&
      st.shared_up.Allocate(T * c.shared_expert_ff) &&
      st.shared.Allocate(T * H) &&
      st.xq.Allocate(static_cast<std::size_t>(rocm::kGemvMaxTokens) *
                     std::max<std::size_t>(H, c.ssm_inner_size) / 32 *
                     rocm::kQ8_1BlockBytes) &&
      st.position.Allocate(1) &&
      st.attention_partial.Allocate(
          static_cast<std::size_t>(rocm::kAttentionShortTokens) * c.num_heads *
          rocm::kAttentionSplits * (c.head_dim + 2)) &&
      st.last_logits.Allocate(c.vocab_size) &&
      st.head_scratch.Allocate(rocm::LogitHeadScratchBytes(
          kVerifyMax, static_cast<int>(c.vocab_size))) &&
      st.head_records.Allocate(kVerifyMax) &&
      st.head_candidates.Allocate(kVerifyMax * rocm::kLogitHeadCandidates);
  st.host_head_records.resize(kVerifyMax);
  st.host_head_candidates.resize(kVerifyMax * rocm::kLogitHeadCandidates);
  st.route = {.cursor = st.route_cursor.data,
              .order = st.route_order.data,
              .tiles = st.route_tiles.data,
              .num_tiles = st.route_count.data,
              .max_tiles = static_cast<int>(st.route_tiles.count)};
  return ok;
}

}  // namespace

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  auto fail = [&](std::string message) -> std::unique_ptr<Session> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  if (max_context == 0 || max_context > options_.max_context)
    return fail("session context exceeds the model's configured maximum");
  const Config& c = weights_->config;
  std::unique_ptr<Session> s(new Session());
  s->model_ = shared_from_this();
  s->max_context_ = max_context;
  s->state_ = std::make_unique<Session::State>();
  Session::State& st = *s->state_;
  if (!CreateStreams(st))
    return fail("could not create the session stream");

  bool ok = true;
  const std::size_t n = c.num_layers;
  st.k_cache.resize(n);
  st.v_cache.resize(n);
  st.conv_state.resize(n);
  st.ssm_state.resize(n);
  for (std::size_t il = 0; il < n; ++il) {
    switch (c.blocks[il]) {
      case BlockType::kAttention: {
        const std::size_t kv =
            static_cast<std::size_t>(max_context) * c.AttentionKvDim();
        ok = ok && st.k_cache[il].Allocate(kv) && st.v_cache[il].Allocate(kv);
        break;
      }
      case BlockType::kMamba:
        ok = ok &&
             st.conv_state[il].Allocate(
                 static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                 c.SsmConvChannels()) &&
             st.ssm_state[il].Allocate(static_cast<std::size_t>(c.ssm_heads) *
                                       c.SsmHeadDim() * c.ssm_state_size);
        break;
      case BlockType::kMoe:
        break;
    }
  }
  const std::size_t T = kChunk;
  ok = ok && AllocateChunk(st, c, *device_, T);
  if (!ok)
    return fail("could not allocate session memory");
  std::string error;
  if (drafter_ != nullptr) {
    const DrafterConfig& dc = drafter_->config();
    st.draft = DraftContext::Create(*drafter_, *device_, kChunk, st.stream,
                                    st.blas, &error);
    if (!st.draft)
      return fail(error);
    st.feature_slot.assign(n, -1);
    for (std::size_t i = 0; i < dc.target_layers.size(); ++i)
      st.feature_slot[dc.target_layers[i]] = static_cast<int>(i);
    st.feature_width = static_cast<int>(dc.FeatureSize());
    if (!st.features.Allocate(T * dc.FeatureSize()) ||
        !st.inject_position.Allocate(1))
      return fail("could not allocate drafter feature memory");
  }
  st.verify_proj.resize(n);
  st.verify_xbc.resize(n);
  for (std::size_t il = 0; il < n; ++il) {
    if (c.blocks[il] == BlockType::kMamba &&
        (!st.verify_proj[il].Allocate(kVerifyMax * c.SsmInProjection()) ||
         !st.verify_xbc[il].Allocate(kVerifyMax * c.SsmConvChannels())))
      return fail("could not allocate verification memory");
  }
  if (!s->Reset(&error))
    return fail(error);
  return s;
}

Session::~Session() = default;

namespace {

static_assert(sizeof(rocm::LogitHeadCandidate) ==
                  sizeof(sampling::LogitCandidate) &&
              offsetof(sampling::LogitCandidate, value) ==
                  offsetof(rocm::LogitHeadCandidate, value));

/// Enqueues the heads of `n` device rows and their download; false when the
/// filter keeps more than a head holds, so the rows get no heads.
bool EnqueueHeads(Session::State& st, const float* rows, int n, int vocab,
                  const sampling::LogitHeadFilter& filter, bool* ok) {
  if (filter.keep > static_cast<std::size_t>(rocm::kLogitHeadCandidates) ||
      filter.minimum > static_cast<std::size_t>(rocm::kLogitHeadCandidates))
    return false;
  const rocm::LogitHeadParams params{
      .temperature = filter.temperature,
      .keep = static_cast<int>(filter.keep),
      .top_p = filter.top_p,
      .minimum = static_cast<int>(filter.minimum)};
  *ok =
      rocm::LogitHeads(rows, n, vocab, params, st.head_scratch.data,
                       st.head_records.data, st.head_candidates.data,
                       st.stream) == hipSuccess &&
      hipMemcpyAsync(st.host_head_records.data(), st.head_records.data,
                     n * sizeof(rocm::LogitHeadRecord), hipMemcpyDeviceToHost,
                     st.stream) == hipSuccess &&
      hipMemcpyAsync(st.host_head_candidates.data(), st.head_candidates.data,
                     static_cast<std::size_t>(n) * rocm::kLogitHeadCandidates *
                         sizeof(rocm::LogitHeadCandidate),
                     hipMemcpyDeviceToHost, st.stream) == hipSuccess;
  return true;
}

/// After the stream synchronized: the heads EnqueueHeads downloaded, or
/// empty heads when it summarized nothing.
void TakeHeads(const Session::State& st, bool summarized, float temperature,
               std::span<sampling::LogitHead> heads) {
  for (std::size_t r = 0; r < heads.size(); ++r) {
    sampling::LogitHead& head = heads[r];
    head.candidates.clear();
    if (!summarized || st.host_head_records[r].count == 0)
      continue;
    const rocm::LogitHeadRecord& record = st.host_head_records[r];
    head.maximum = record.maximum;
    head.softmax_sum = record.softmax_sum;
    head.temperature = temperature;
    const auto* first = reinterpret_cast<const sampling::LogitCandidate*>(
        st.host_head_candidates.data() + r * rocm::kLogitHeadCandidates);
    head.candidates.assign(first, first + record.count);
  }
}

}  // namespace

DraftContext* Session::draft() noexcept {
  return state_->draft.get();
}

namespace {

/// Runs one chunk of at most kChunk tokens through the trunk.
class Forward {
public:
  /// `verify` keeps the Mamba states and records their inputs instead.
  Forward(const Config& c, const rocm::DeviceModel& m, Session::State& st,
          bool verify = false)
      : c_(c), m_(m), st_(st), verify_(verify) {}

  /// Enqueues the chunk; `head_rows` final rows (0, 1 or n) go through the
  /// output head into `logits` on the device. With a drafter, the chunk's
  /// features from row st.inject_skip on go into its context. A batch state
  /// (st.rows > 0) takes one token of each of its rows' sessions and only
  /// captures their features.
  bool Enqueue(int n, int head_rows, float* logits) {
    const int H = static_cast<int>(c_.hidden_size);
    const bool features = !st_.feature_slot.empty() && st_.inject_skip < n;
    Ok(rocm::EmbedQ8_0(m_.token_embd().data, st_.tokens.data, st_.x.data, n, H,
                       st_.stream));
    for (std::size_t il = 0; ok_ && il < m_.layers().size(); ++il) {
      const rocm::DeviceLayer& l = m_.layers()[il];
      if (features && st_.feature_slot[il] >= 0)
        Capture(st_.feature_slot[il], n);
      if (n <= rocm::kGemvMaxTokens) {
        // Short chunks: the vector products read the normalized rows as
        // Q8_1; the router and the shared expert read them as F32.
        Ok(rocm::RmsNormQ8(st_.x.data, l.norm.f32(),
                           l.type == BlockType::kMoe ? st_.h.data : nullptr,
                           st_.xq.data, n, H, c_.rms_eps, st_.stream));
      } else if (l.type == BlockType::kMamba) {
        // Only the FP16 projection reads a Mamba block's normalized rows.
        Ok(rocm::RmsNorm(st_.x.data, l.norm.f32(), st_.h16.data, n, H,
                         c_.rms_eps, st_.stream));
      } else {
        Ok(rocm::RmsNorm(st_.x.data, l.norm.f32(), st_.h.data, n, H, c_.rms_eps,
                         st_.stream));
      }
      switch (l.type) {
        case BlockType::kMamba:
          Mamba(l, il, n);
          break;
        case BlockType::kAttention:
          Attention(l, il, n);
          break;
        case BlockType::kMoe:
          Moe(l, n);
          break;
      }
    }
    if (ok_ && features && st_.draft != nullptr) {
      const int skip = st_.inject_skip;
      std::string error;
      if (!st_.draft->Inject(
              st_.features.data +
                  static_cast<std::size_t>(skip) * st_.feature_width,
              st_.inject_position.data, n - skip, &error)) {
        ok_ = false;
        error_ = error;
      }
    }
    if (ok_ && head_rows > 0)
      Head(n, head_rows, logits);
    return ok_;
  }

  [[nodiscard]] const std::string& error() const { return error_; }

private:
  void Ok(hipError_t e) {
    if (ok_ && e != hipSuccess) {
      ok_ = false;
      error_ = hipGetErrorString(e);
    }
  }
  void Mmq(int rc, const char* what) {
    if (ok_ && rc != 0) {
      ok_ = false;
      error_ = std::string("quantized matrix product failed: ") + what;
    }
  }
  void Dense(const rocm::DeviceTensor& w, const float* x, float* out, int n,
             const char* what) {
    Mmq(nb_mmq_dense(static_cast<int>(w.type), w.data, x, out, w.rows, n,
                     w.cols, st_.stream),
        what);
  }

  /// Short-chunk product on the Q8_1 rows in st_.xq.
  void Q8Dense(const rocm::DeviceTensor& w, float* out, int n,
               const char* what) {
    Mmq(nb_mmq_dense_q8_1(static_cast<int>(w.type), w.data, st_.xq.data, 1.0F,
                          nullptr, out, w.rows, n, w.cols, st_.stream),
        what);
  }

  /// x += scale * W st_.xq: fused into the product for one token, a
  /// separate add for more. The scale stays out of the activations, whose
  /// Q8_1 block scales are FP16.
  void Q8Residual(const rocm::DeviceTensor& w, float scale, int n,
                  const char* what) {
    if (n == 1) {
      Mmq(nb_mmq_dense_q8_1(static_cast<int>(w.type), w.data, st_.xq.data,
                            scale, st_.x.data, st_.x.data, w.rows, n, w.cols,
                            st_.stream),
          what);
      return;
    }
    Q8Dense(w, st_.out.data, n, what);
    Ok(rocm::AddScaled(st_.x.data, st_.out.data, scale, nullptr,
                       static_cast<std::size_t>(n) * c_.hidden_size,
                       st_.stream));
  }

  /// Copies the chunk's residual stream, the input of a target block, into
  /// its feature column.
  void Capture(int slot, int n) {
    const std::size_t H = c_.hidden_size;
    Ok(hipMemcpy2DAsync(st_.features.data + slot * H,
                        st_.feature_width * sizeof(float), st_.x.data,
                        H * sizeof(float), H * sizeof(float), n,
                        hipMemcpyDeviceToDevice, st_.stream));
  }

  /// The rows' entries j (0: conv state or K cache, 1: SSM state or V cache)
  /// of block il in a batch state's table.
  template<class T>
  T* const* Rows(std::size_t il, int j) const {
    return reinterpret_cast<T* const*>(st_.row_tables.data +
                                       (il * 2 + j) * kVerifyMax);
  }

  void Mamba(const rocm::DeviceLayer& l, std::size_t il, int n) {
    const int stride = static_cast<int>(c_.SsmInProjection());
    const int inner = static_cast<int>(c_.ssm_inner_size);
    const int channels = static_cast<int>(c_.SsmConvChannels());
    const int kernel = static_cast<int>(c_.ssm_conv_kernel);
    const float scale = l.ssm_in.scale;
    const bool rows = st_.rows > 0;
    float* proj = verify_ ? st_.verify_proj[il].data : st_.proj.data;
    float* xbc = verify_ ? st_.verify_xbc[il].data : st_.xbc.data;
    const bool short_chunk = n <= rocm::kGemvMaxTokens;
    if (short_chunk)
      Q8Dense(l.ssm_in, proj, n, "ssm_in");
    else
      Ok(rocm::Nvfp4Gemm(l.ssm_in.data, st_.h16.data, proj, l.ssm_in.rows,
                         l.ssm_in.cols, n, st_.stream));
    if (rows) {
      Ok(rocm::MambaConvRows(proj, stride, inner, scale, l.ssm_conv1d.f32(),
                             l.ssm_conv1d_b.f32(), Rows<float>(il, 0), xbc, n,
                             channels, kernel, st_.stream));
    } else {
      Ok(rocm::MambaConv(proj, stride, inner, scale, l.ssm_conv1d.f32(),
                         l.ssm_conv1d_b.f32(), st_.conv_state[il].data, xbc, n,
                         channels, kernel, st_.stream));
      if (!verify_)
        Ok(rocm::MambaConvState(proj, stride, inner, scale,
                                st_.conv_state[il].data, n, channels, kernel,
                                st_.stream));
    }
    rocm::MambaScanArgs scan{};
    scan.xbc = xbc;
    scan.channels = channels;
    scan.proj = proj;
    scan.stride = stride;
    scan.dt_offset = inner + channels;
    scan.in_scale = scale;
    scan.dt_bias = l.ssm_dt_b.f32();
    scan.a = l.ssm_a.f32();
    scan.d = l.ssm_d.f32();
    scan.state = rows ? nullptr : st_.ssm_state[il].data;
    scan.y = st_.y.data;
    scan.n_tokens = n;
    scan.heads = static_cast<int>(c_.ssm_heads);
    scan.head_dim = static_cast<int>(c_.SsmHeadDim());
    scan.groups = static_cast<int>(c_.ssm_groups);
    scan.write_state = !verify_;
    if (rows)
      scan.row_states = Rows<float>(il, 1);
    Ok(rocm::MambaScan(scan, st_.stream));
    if (short_chunk) {
      Ok(rocm::MambaGateNormQ8(
          st_.y.data, proj, stride, scale, l.ssm_norm.f32(), st_.xq.data, n,
          inner, static_cast<int>(c_.ssm_groups), c_.rms_eps, st_.stream));
      Q8Residual(l.ssm_out, l.ssm_out.scale, n, "ssm_out");
      return;
    }
    Ok(rocm::MambaGateNorm(
        st_.y.data, proj, stride, scale, l.ssm_norm.f32(), st_.y16.data, n,
        inner, static_cast<int>(c_.ssm_groups), c_.rms_eps, st_.stream));
    Ok(rocm::Nvfp4Gemm(l.ssm_out.data, st_.y16.data, st_.out.data,
                       l.ssm_out.rows, l.ssm_out.cols, n, st_.stream));
    Ok(rocm::AddScaled(st_.x.data, st_.out.data, l.ssm_out.scale, nullptr,
                       static_cast<std::size_t>(n) * c_.hidden_size,
                       st_.stream));
  }

  void Attention(const rocm::DeviceLayer& l, std::size_t il, int n) {
    const bool short_chunk = n <= rocm::kGemvMaxTokens;
    if (short_chunk) {
      Q8Dense(l.attn_q, st_.q.data, n, "attn_q");
      Q8Dense(l.attn_k, st_.k.data, n, "attn_k");
      Q8Dense(l.attn_v, st_.v.data, n, "attn_v");
    } else {
      Dense(l.attn_q, st_.h.data, st_.q.data, n, "attn_q");
      Dense(l.attn_k, st_.h.data, st_.k.data, n, "attn_k");
      Dense(l.attn_v, st_.h.data, st_.v.data, n, "attn_v");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(c_.head_dim));
    if (st_.rows > 0) {
      Ok(rocm::KvAppendRows(st_.k.data, st_.v.data, Rows<__half>(il, 0),
                            Rows<__half>(il, 1), st_.row_positions.data, n,
                            static_cast<int>(c_.AttentionKvDim()), st_.stream));
      Ok(rocm::AttentionShortRows(
          st_.q.data, Rows<const __half>(il, 0), Rows<const __half>(il, 1),
          st_.ctx.data, st_.row_positions.data, n,
          static_cast<int>(c_.num_heads), static_cast<int>(c_.num_kv_heads),
          scale, st_.attention_partial.data, st_.stream));
    } else {
      Ok(rocm::KvAppend(st_.k.data, st_.v.data, st_.k_cache[il].data,
                        st_.v_cache[il].data, st_.position.data, n,
                        static_cast<int>(c_.AttentionKvDim()), st_.stream));
      if (n <= rocm::kAttentionShortTokens) {
        Ok(rocm::AttentionShort(
            st_.q.data, st_.k_cache[il].data, st_.v_cache[il].data,
            st_.ctx.data, st_.position.data, n, static_cast<int>(c_.num_heads),
            static_cast<int>(c_.num_kv_heads), scale,
            st_.attention_partial.data, st_.stream));
      } else {
        Ok(rocm::Attention(
            st_.q.data, st_.k_cache[il].data, st_.v_cache[il].data,
            st_.ctx.data, st_.position.data, n, static_cast<int>(c_.num_heads),
            static_cast<int>(c_.num_kv_heads), scale, st_.stream));
      }
    }
    if (short_chunk) {
      Ok(rocm::QuantizeQ8_1(st_.ctx.data, st_.xq.data, n,
                            static_cast<int>(c_.AttentionQDim()), st_.stream));
      Q8Residual(l.attn_out, 1.0F, n, "attn_output");
      return;
    }
    Dense(l.attn_out, st_.ctx.data, st_.out.data, n, "attn_output");
    Ok(rocm::AddScaled(st_.x.data, st_.out.data, 1.0F, nullptr,
                       static_cast<std::size_t>(n) * c_.hidden_size,
                       st_.stream));
  }

  void Moe(const rocm::DeviceLayer& l, int n) {
    const int H = static_cast<int>(c_.hidden_size);
    const int latent = static_cast<int>(c_.latent_size);
    const int used = l.experts_used;
    const int ff = l.expert_ff;
    const int shared_ff = static_cast<int>(c_.shared_expert_ff);
    const bool short_chunk = n <= rocm::kGemvMaxTokens;
    if (short_chunk) {
      // The shared expert depends only on h; it overlaps the router, the
      // routed experts and their small kernels on the side stream.
      Ok(hipEventRecord(st_.fork, st_.stream));
      Ok(hipStreamWaitEvent(st_.side, st_.fork, 0));
      rocm::Iq2xxsGemvArgs s{};
      s.grid = m_.iq2xxs_grid();
      s.n_tokens = n;
      s.weights = l.up_shexp.data;
      s.x = st_.h.data;
      s.out = st_.shared_up.data;
      s.rows = shared_ff;
      s.cols = H;
      s.relu2 = true;
      Ok(rocm::Iq2xxsGemv(s, st_.side));
      // Nothing else touches the residual stream until the join.
      s.weights = l.down_shexp.data;
      s.x = st_.shared_up.data;
      s.out = st_.x.data;
      s.rows = H;
      s.cols = shared_ff;
      s.relu2 = false;
      s.accumulate = true;
      Ok(rocm::Iq2xxsGemv(s, st_.side));
      Ok(hipEventRecord(st_.join, st_.side));
    }
    // Router logits in F32: [n][experts] = h[n][H] * router[experts][H]^T.
    // Decode-sized batches stream the weights once; hipBLAS takes prefill.
    if (n <= rocm::kRouterMaxTokens) {
      Ok(rocm::RouterLogits(l.router.f32(), st_.h.data, st_.router.data,
                            l.num_experts, H, n, st_.stream));
    } else {
      const float one = 1.0F;
      const float zero = 0.0F;
      if (ok_ && hipblasSgemm(st_.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                              l.num_experts, n, H, &one, l.router.f32(), H,
                              st_.h.data, H, &zero, st_.router.data,
                              l.num_experts) != HIPBLAS_STATUS_SUCCESS) {
        ok_ = false;
        error_ = "router GEMM failed";
      }
    }
    Ok(rocm::RouterTopK(st_.router.data, l.router_bias.f32(), l.num_experts, n,
                        used, c_.expert_weights_norm, c_.expert_weights_scale,
                        st_.ids.data, st_.weights.data, st_.stream));
    if (short_chunk)
      Q8Dense(l.latent_down, st_.latent.data, n, "ffn_latent_down");
    else
      Dense(l.latent_down, st_.h.data, st_.latent.data, n, "ffn_latent_down");
    if (short_chunk) {
      // Short chunks: the vector products apply ReLU² as they store.
      rocm::Iq2xxsGemvArgs e{};
      e.grid = m_.iq2xxs_grid();
      e.n_tokens = n;
      e.weights = l.up_exps.data;
      e.x = st_.latent.data;
      e.ids = st_.ids.data;
      e.used = used;
      e.out = st_.up.data;
      e.rows = ff;
      e.cols = latent;
      e.relu2 = true;
      Ok(rocm::Iq2xxsGemv(e, st_.stream));
      e.weights = l.down_exps.data;
      e.x = st_.up.data;
      e.x_per_slot = true;
      e.out = st_.down.data;
      e.rows = latent;
      e.cols = ff;
      e.relu2 = false;
      Ok(rocm::Iq2xxsGemv(e, st_.stream));
      Ok(rocm::WeightedSlotSumQ8(st_.down.data, st_.weights.data, st_.xq.data,
                                 n, used, latent, st_.stream));
      Ok(hipStreamWaitEvent(st_.stream, st_.join, 0));
      Q8Residual(l.latent_up, 1.0F, n, "ffn_latent_up");
      return;
    } else {
      // Routed experts over the slots grouped by expert (the routed product
      // applies ReLU² as it stores), then the dense shared expert.
      Ok(rocm::Iq2xxsRouteSlots(st_.ids.data, n * used, l.num_experts,
                                st_.route, st_.stream));
      rocm::Iq2xxsGemmArgs e{};
      e.grid = m_.iq2xxs_grid();
      e.weights = l.up_exps.data;
      e.x = st_.latent.data;
      e.route = &st_.route;
      e.used = used;
      e.out = st_.up.data;
      e.rows = ff;
      e.cols = latent;
      e.relu2 = true;
      Ok(rocm::Iq2xxsRoutedGemm(e, st_.stream));
      e.weights = l.down_exps.data;
      e.x = st_.up.data;
      e.x_per_slot = true;
      e.out = st_.down.data;
      e.rows = latent;
      e.cols = ff;
      e.relu2 = false;
      Ok(rocm::Iq2xxsRoutedGemm(e, st_.stream));
      Ok(rocm::WeightedSlotSum(st_.down.data, st_.weights.data, st_.acc.data, n,
                               used, latent, st_.stream));
      Dense(l.latent_up, st_.acc.data, st_.out.data, n, "ffn_latent_up");
      Dense(l.up_shexp, st_.h.data, st_.shared_up.data, n, "ffn_up_shexp");
      Ok(rocm::Relu2(st_.shared_up.data,
                     static_cast<std::size_t>(n) * shared_ff, st_.stream));
      Dense(l.down_shexp, st_.shared_up.data, st_.shared.data, n,
            "ffn_down_shexp");
    }
    Ok(rocm::AddScaled(st_.x.data, st_.out.data, 1.0F, st_.shared.data,
                       static_cast<std::size_t>(n) * H, st_.stream));
  }

  void Head(int n, int rows, float* logits) {
    const int H = static_cast<int>(c_.hidden_size);
    const float* last = st_.x.data + static_cast<std::size_t>(n - rows) * H;
    if (rows <= rocm::kGemvMaxTokens) {
      Ok(rocm::RmsNormQ8(last, m_.output_norm().f32(), nullptr, st_.xq.data,
                         rows, H, c_.rms_eps, st_.stream));
      Q8Dense(m_.output(), logits, rows, "output");
      return;
    }
    Ok(rocm::RmsNorm(last, m_.output_norm().f32(), st_.h.data, rows, H,
                     c_.rms_eps, st_.stream));
    Dense(m_.output(), st_.h.data, logits, rows, "output");
  }

  const Config& c_;
  const rocm::DeviceModel& m_;
  Session::State& st_;
  bool verify_;
  bool ok_{true};
  std::string error_;
};

}  // namespace

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::span<float> logits, std::string* error_msg) {
  return AppendRows(tokens, logits, nullptr, {}, error_msg);
}

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::span<float> logits,
                     const sampling::LogitHeadFilter& filter,
                     sampling::LogitHead* head, std::string* error_msg) {
  if (logits.size() != model_->config().vocab_size) {
    if (error_msg != nullptr)
      *error_msg = "a summarized append returns one logits row";
    return false;
  }
  return AppendRows(tokens, logits, &filter, std::span(head, 1), error_msg);
}

bool Session::AppendRows(std::span<const std::int32_t> tokens,
                         std::span<float> logits,
                         const sampling::LogitHeadFilter* filter,
                         std::span<sampling::LogitHead> heads,
                         std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  if (needs_reset_ && !Reset(error_msg))
    return false;
  verified_ = 0;  // An uncommitted Verify left the states untouched.
  const Config& c = model_->config();
  const std::size_t vocab = c.vocab_size;
  // One token's single row is the last-row case, served by the decode graph.
  const bool all_positions =
      tokens.size() > 1 && logits.size() == tokens.size() * vocab;
  if (!logits.empty() && logits.size() != vocab && !all_positions)
    return fail("logits must hold one row or one row per token");
  if (tokens.size() > max_context_ - position_)
    return fail("the session context is full");
  for (const std::int32_t t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size)
      return fail("token outside the vocabulary");
  }
  State& st = *state_;
  const rocm::DeviceModel& device = *model_->device_;
  // A draft anchored at the end of this append or later reads at most a
  // window (plus its own block) back; older rows skip the drafter.
  std::uint32_t keep_from = 0;
  if (const Drafter* d = model_->drafter_.get()) {
    const std::uint32_t end =
        position_ + static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t reach =
        d->config().sliding_window + d->config().block_size;
    keep_from = end > reach ? end - reach : 0;
  }
  std::size_t done = 0;
  bool summarized = false;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(kChunk, tokens.size() - done);
    const bool last = done + n == tokens.size();
    const auto first = static_cast<std::int32_t>(position_);
    st.inject_skip = static_cast<int>(std::min<std::size_t>(
        n, keep_from > position_ ? keep_from - position_ : 0));
    const std::int32_t inject_first = first + st.inject_skip;
    if (hipMemcpyAsync(st.tokens.data, tokens.data() + done,
                       n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                       st.stream) != hipSuccess ||
        hipMemcpyAsync(st.position.data, &first, sizeof(first),
                       hipMemcpyHostToDevice, st.stream) != hipSuccess ||
        (st.draft != nullptr &&
         hipMemcpyAsync(st.inject_position.data, &inject_first,
                        sizeof(inject_first), hipMemcpyHostToDevice,
                        st.stream) != hipSuccess))
      return fail("could not upload tokens");
    const int head_rows = all_positions             ? static_cast<int>(n)
                          : last && !logits.empty() ? 1
                                                    : 0;
    if (all_positions && st.logits.count < n * vocab &&
        !st.logits.Allocate(n * vocab))
      return fail("could not allocate the logits buffer");
    float* device_logits = all_positions ? st.logits.data : st.last_logits.data;
    std::string error;
    if (n == 1 && head_rows == 1) {
      if (!RunDecodeGraph(&error))
        return fail(error);
    } else {
      Forward forward(c, device, st);
      if (!forward.Enqueue(static_cast<int>(n), head_rows, device_logits))
        return fail(forward.error());
    }
    if (head_rows > 0) {
      float* dst = all_positions ? logits.data() + done * vocab : logits.data();
      if (hipMemcpyAsync(
              dst, device_logits,
              static_cast<std::size_t>(head_rows) * vocab * sizeof(float),
              hipMemcpyDeviceToHost, st.stream) != hipSuccess)
        return fail("could not copy logits to the host");
      bool enqueued = true;
      if (filter != nullptr && last)
        summarized = EnqueueHeads(st, device_logits, 1, static_cast<int>(vocab),
                                  *filter, &enqueued);
      if (!enqueued)
        return fail("could not summarize the logits");
    }
    if (hipStreamSynchronize(st.stream) != hipSuccess)
      return fail(std::string("forward failed: ") +
                  hipGetErrorString(hipGetLastError()));
    position_ += static_cast<std::uint32_t>(n);
    done += n;
  }
  if (filter != nullptr)
    TakeHeads(st, summarized, filter->temperature, heads);
  return true;
}

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::span<float> logits, std::string* error_msg) {
  return VerifyRows(tokens, logits, nullptr, {}, error_msg);
}

bool Session::Verify(std::span<const std::int32_t> tokens,
                     std::span<float> logits,
                     const sampling::LogitHeadFilter& filter,
                     std::span<sampling::LogitHead> heads,
                     std::string* error_msg) {
  if (heads.size() != tokens.size()) {
    if (error_msg != nullptr)
      *error_msg = "verify summarizes one head per token";
    return false;
  }
  return VerifyRows(tokens, logits, &filter, heads, error_msg);
}

bool Session::VerifyRows(std::span<const std::int32_t> tokens,
                         std::span<float> logits,
                         const sampling::LogitHeadFilter* filter,
                         std::span<sampling::LogitHead> heads,
                         std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  if (needs_reset_ && !Reset(error_msg))
    return false;
  verified_ = 0;
  const Config& c = model_->config();
  const std::size_t vocab = c.vocab_size;
  const std::size_t n = tokens.size();
  if (n == 0 || n > static_cast<std::size_t>(kVerifyMax))
    return fail("verify takes 1 to 8 tokens");
  if (logits.size() != n * vocab)
    return fail("verify returns one logits row per token");
  if (n > max_context_ - position_)
    return fail("the session context is full");
  for (const std::int32_t t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size)
      return fail("token outside the vocabulary");
  }
  State& st = *state_;
  if (st.logits.count < n * vocab && !st.logits.Allocate(n * vocab))
    return fail("could not allocate the logits buffer");
  const auto first = static_cast<std::int32_t>(position_);
  st.inject_skip = 0;
  if (hipMemcpyAsync(st.tokens.data, tokens.data(), n * sizeof(std::int32_t),
                     hipMemcpyHostToDevice, st.stream) != hipSuccess ||
      hipMemcpyAsync(st.position.data, &first, sizeof(first),
                     hipMemcpyHostToDevice, st.stream) != hipSuccess ||
      (st.draft != nullptr &&
       hipMemcpyAsync(st.inject_position.data, &first, sizeof(first),
                      hipMemcpyHostToDevice, st.stream) != hipSuccess))
    return fail("could not upload tokens");
  Forward forward(c, *model_->device_, st, /*verify=*/true);
  if (!forward.Enqueue(static_cast<int>(n), static_cast<int>(n),
                       st.logits.data))
    return fail(forward.error());
  bool enqueued = true;
  const bool summarized =
      filter != nullptr &&
      EnqueueHeads(st, st.logits.data, static_cast<int>(n),
                   static_cast<int>(vocab), *filter, &enqueued);
  if (!enqueued ||
      hipMemcpyAsync(logits.data(), st.logits.data, n * vocab * sizeof(float),
                     hipMemcpyDeviceToHost, st.stream) != hipSuccess ||
      hipStreamSynchronize(st.stream) != hipSuccess)
    return fail(std::string("verify failed: ") +
                hipGetErrorString(hipGetLastError()));
  if (filter != nullptr)
    TakeHeads(st, summarized, filter->temperature, heads);
  verified_ = static_cast<std::uint32_t>(n);
  return true;
}

bool Session::Commit(std::uint32_t n, std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    needs_reset_ = true;
    return false;
  };
  if (n > verified_)
    return fail("commit exceeds the verified tokens");
  verified_ = 0;
  if (n == 0)
    return true;
  const Config& c = model_->config();
  const rocm::DeviceModel& device = *model_->device_;
  State& st = *state_;
  const int stride = static_cast<int>(c.SsmInProjection());
  const int inner = static_cast<int>(c.ssm_inner_size);
  const int channels = static_cast<int>(c.SsmConvChannels());
  const int kernel = static_cast<int>(c.ssm_conv_kernel);
  bool ok = true;
  // Replays the kept prefix through the same kernels the forward pass uses,
  // so the states equal those of appending the prefix.
  for (std::size_t il = 0; ok && il < c.num_layers; ++il) {
    if (c.blocks[il] != BlockType::kMamba)
      continue;
    const rocm::DeviceLayer& l = device.layers()[il];
    const float scale = l.ssm_in.scale;
    ok = rocm::MambaConvState(st.verify_proj[il].data, stride, inner, scale,
                              st.conv_state[il].data, static_cast<int>(n),
                              channels, kernel, st.stream) == hipSuccess;
    rocm::MambaScanArgs scan{};
    scan.xbc = st.verify_xbc[il].data;
    scan.channels = channels;
    scan.proj = st.verify_proj[il].data;
    scan.stride = stride;
    scan.dt_offset = inner + channels;
    scan.in_scale = scale;
    scan.dt_bias = l.ssm_dt_b.f32();
    scan.a = l.ssm_a.f32();
    scan.d = l.ssm_d.f32();
    scan.state = st.ssm_state[il].data;
    scan.y = st.y.data;
    scan.n_tokens = static_cast<int>(n);
    scan.heads = static_cast<int>(c.ssm_heads);
    scan.head_dim = static_cast<int>(c.SsmHeadDim());
    scan.groups = static_cast<int>(c.ssm_groups);
    ok = ok && rocm::MambaScan(scan, st.stream) == hipSuccess;
  }
  if (!ok || hipStreamSynchronize(st.stream) != hipSuccess)
    return fail("could not commit the verified tokens");
  position_ += n;
  return true;
}

/// The model's state for Session::DecodeBatch: chunk scratch of kVerifyMax
/// rows with its own streams, and one captured forward per batch width.
struct BatchState {
  Session::State st;
  std::array<hipGraphExec_t, kVerifyMax + 1> graphs{};
  std::array<bool, kVerifyMax + 1> graph_unavailable{};
  hipEvent_t forward_done{nullptr};

  ~BatchState() {
    for (hipGraphExec_t g : graphs) {
      if (g != nullptr)
        static_cast<void>(hipGraphExecDestroy(g));
    }
    if (forward_done != nullptr)
      static_cast<void>(hipEventDestroy(forward_done));
  }
};

namespace {

std::unique_ptr<BatchState> CreateBatchState(const Config& c,
                                             const rocm::DeviceModel& device,
                                             const Drafter* drafter,
                                             std::string* error) {
  auto b = std::make_unique<BatchState>();
  Session::State& st = b->st;
  if (!CreateStreams(st) ||
      hipEventCreateWithFlags(&b->forward_done, hipEventDisableTiming) !=
          hipSuccess) {
    *error = "could not create the batch stream";
    return nullptr;
  }
  bool ok = AllocateChunk(st, c, device, kVerifyMax) &&
            st.logits.Allocate(kVerifyMax * c.vocab_size) &&
            st.row_tables.Allocate(c.num_layers * 2 * kVerifyMax) &&
            st.row_positions.Allocate(kVerifyMax);
  if (ok && drafter != nullptr) {
    // Batched rows only capture their features; each session injects its row.
    const DrafterConfig& dc = drafter->config();
    st.feature_slot.assign(c.num_layers, -1);
    for (std::size_t i = 0; i < dc.target_layers.size(); ++i)
      st.feature_slot[dc.target_layers[i]] = static_cast<int>(i);
    st.feature_width = static_cast<int>(dc.FeatureSize());
    ok = st.features.Allocate(kVerifyMax * dc.FeatureSize());
  }
  if (!ok) {
    *error = "could not allocate batch memory";
    return nullptr;
  }
  return b;
}

/// The head parameters of `filter`; false when it keeps more than a head.
bool HeadParams(const sampling::LogitHeadFilter& filter,
                rocm::LogitHeadParams* params) {
  if (filter.keep > static_cast<std::size_t>(rocm::kLogitHeadCandidates) ||
      filter.minimum > static_cast<std::size_t>(rocm::kLogitHeadCandidates))
    return false;
  *params = {.temperature = filter.temperature,
             .keep = static_cast<int>(filter.keep),
             .top_p = filter.top_p,
             .minimum = static_cast<int>(filter.minimum)};
  return true;
}

}  // namespace

bool Session::DecodeBatch(
    std::span<Session* const> sessions, std::span<const std::int32_t> tokens,
    std::span<const std::span<float>> logits,
    std::span<const sampling::LogitHeadFilter* const> filters,
    std::span<sampling::LogitHead> heads, std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return false;
  };
  const std::size_t n = sessions.size();
  if (n < 2 || n > static_cast<std::size_t>(kVerifyMax))
    return fail("a batch decodes 2 to 8 sessions");
  if (tokens.size() != n || logits.size() != n || filters.size() != n ||
      heads.size() != n)
    return fail("a batch takes one token, row, filter and head per session");
  const Model& model = *sessions[0]->model_;
  const Config& c = model.config();
  const std::size_t vocab = c.vocab_size;
  for (std::size_t r = 0; r < n; ++r) {
    Session& s = *sessions[r];
    if (s.model_.get() != &model)
      return fail("batched sessions must share their model");
    for (std::size_t q = 0; q < r; ++q) {
      if (sessions[q] == &s)
        return fail("a session appears twice in the batch");
    }
    if (tokens[r] < 0 || static_cast<std::uint32_t>(tokens[r]) >= vocab)
      return fail("token outside the vocabulary");
    if (logits[r].size() != vocab)
      return fail("each batched session returns one logits row");
    if (s.needs_reset_ && !s.Reset(error_msg))
      return false;
    s.verified_ = 0;  // An uncommitted Verify left the states untouched.
    if (s.position_ >= s.max_context_)
      return fail("the session context is full");
  }
  const std::lock_guard lock(model.batch_mutex_);
  const rocm::DeviceModel& device = *model.device_;
  if (!model.batch_) {
    std::string error;
    model.batch_ = CreateBatchState(c, device, model.drafter_.get(), &error);
    if (!model.batch_)
      return fail(error);
  }
  BatchState& b = *model.batch_;
  Session::State& st = b.st;

  // The rows' states and positions, read from device memory by the forward.
  std::vector<void*> table(c.num_layers * 2 * kVerifyMax, nullptr);
  std::array<std::int32_t, kVerifyMax> positions{};
  for (std::size_t r = 0; r < n; ++r) {
    State& row = *sessions[r]->state_;
    for (std::size_t il = 0; il < c.num_layers; ++il) {
      void** entry = table.data() + il * 2 * kVerifyMax + r;
      if (c.blocks[il] == BlockType::kMamba) {
        entry[0] = row.conv_state[il].data;
        entry[kVerifyMax] = row.ssm_state[il].data;
      } else if (c.blocks[il] == BlockType::kAttention) {
        entry[0] = row.k_cache[il].data;
        entry[kVerifyMax] = row.v_cache[il].data;
      }
    }
    positions[r] = static_cast<std::int32_t>(sessions[r]->position_);
  }
  if (hipMemcpyAsync(st.tokens.data, tokens.data(), n * sizeof(std::int32_t),
                     hipMemcpyHostToDevice, st.stream) != hipSuccess ||
      hipMemcpyAsync(st.row_positions.data, positions.data(),
                     n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                     st.stream) != hipSuccess ||
      hipMemcpyAsync(st.row_tables.data, table.data(),
                     table.size() * sizeof(void*), hipMemcpyHostToDevice,
                     st.stream) != hipSuccess)
    return fail("could not upload the batch");
  st.rows = static_cast<int>(n);
  st.inject_skip = 0;

  // One captured forward per width: tables, tokens and positions are device
  // memory, so a replay serves any sessions.
  hipGraphExec_t& graph = b.graphs[n];
  if (graph == nullptr && !b.graph_unavailable[n]) {
    hipGraph_t captured = nullptr;
    bool ok = hipStreamBeginCapture(
                  st.stream, hipStreamCaptureModeThreadLocal) == hipSuccess;
    if (ok) {
      Forward forward(c, device, st);
      ok = forward.Enqueue(static_cast<int>(n), static_cast<int>(n),
                           st.logits.data);
      ok = hipStreamEndCapture(st.stream, &captured) == hipSuccess && ok &&
           captured != nullptr;
    }
    ok = ok && hipGraphInstantiate(&graph, captured, nullptr, nullptr, 0) ==
                   hipSuccess;
    if (captured != nullptr)
      static_cast<void>(hipGraphDestroy(captured));
    if (!ok) {
      // Capture is an optimization; launch the same kernels directly.
      static_cast<void>(hipGetLastError());
      graph = nullptr;
      b.graph_unavailable[n] = true;
    }
  }
  if (graph != nullptr) {
    if (hipGraphLaunch(graph, st.stream) != hipSuccess)
      return fail("batch decode graph launch failed");
  } else {
    Forward forward(c, device, st);
    if (!forward.Enqueue(static_cast<int>(n), static_cast<int>(n),
                         st.logits.data))
      return fail(forward.error());
  }

  // Each session injects its row's features into its own drafter context.
  std::array<std::int32_t, kVerifyMax> inject_first{};
  std::vector<Session::State*> injected;
  if (!st.feature_slot.empty()) {
    if (hipEventRecord(b.forward_done, st.stream) != hipSuccess)
      return fail("could not order the drafter injection");
    for (std::size_t r = 0; r < n; ++r) {
      State& row = *sessions[r]->state_;
      if (row.draft == nullptr)
        continue;
      inject_first[r] = positions[r];
      std::string error;
      if (hipMemcpyAsync(row.inject_position.data, &inject_first[r],
                         sizeof(std::int32_t), hipMemcpyHostToDevice,
                         row.stream) != hipSuccess ||
          hipStreamWaitEvent(row.stream, b.forward_done, 0) != hipSuccess ||
          !row.draft->Inject(st.features.data + r * st.feature_width,
                             row.inject_position.data, 1, &error))
        return fail("drafter injection failed: " + error);
      injected.push_back(&row);
    }
  }

  // Heads with each row's own filter, then the rows to the host.
  std::array<bool, kVerifyMax> summarized{};
  bool any_head = false;
  for (std::size_t r = 0; r < n; ++r) {
    rocm::LogitHeadParams params{};
    if (filters[r] == nullptr || !HeadParams(*filters[r], &params))
      continue;
    if (rocm::LogitHeads(
            st.logits.data + r * vocab, 1, static_cast<int>(vocab), params,
            st.head_scratch.data, st.head_records.data + r,
            st.head_candidates.data + r * rocm::kLogitHeadCandidates,
            st.stream) != hipSuccess)
      return fail("could not summarize the logits");
    summarized[r] = true;
    any_head = true;
  }
  bool ok = true;
  if (any_head) {
    ok = hipMemcpyAsync(st.host_head_records.data(), st.head_records.data,
                        n * sizeof(rocm::LogitHeadRecord),
                        hipMemcpyDeviceToHost, st.stream) == hipSuccess &&
         hipMemcpyAsync(
             st.host_head_candidates.data(), st.head_candidates.data,
             n * rocm::kLogitHeadCandidates * sizeof(rocm::LogitHeadCandidate),
             hipMemcpyDeviceToHost, st.stream) == hipSuccess;
  }
  for (std::size_t r = 0; ok && r < n; ++r)
    ok = hipMemcpyAsync(logits[r].data(), st.logits.data + r * vocab,
                        vocab * sizeof(float), hipMemcpyDeviceToHost,
                        st.stream) == hipSuccess;
  ok = ok && hipStreamSynchronize(st.stream) == hipSuccess;
  for (State* row : injected)
    ok = ok && hipStreamSynchronize(row->stream) == hipSuccess;
  if (!ok) {
    // The rows' states may be partly advanced; none of them can continue.
    for (Session* s : sessions)
      s->needs_reset_ = true;
    return fail(std::string("batch decode failed: ") +
                hipGetErrorString(hipGetLastError()));
  }
  for (std::size_t r = 0; r < n; ++r) {
    sampling::LogitHead& head = heads[r];
    head.candidates.clear();
    const rocm::LogitHeadRecord& record = st.host_head_records[r];
    if (summarized[r] && record.count != 0) {
      head.maximum = record.maximum;
      head.softmax_sum = record.softmax_sum;
      head.temperature = filters[r]->temperature;
      const auto* first = reinterpret_cast<const sampling::LogitCandidate*>(
          st.host_head_candidates.data() + r * rocm::kLogitHeadCandidates);
      head.candidates.assign(first, first + record.count);
    }
    ++sessions[r]->position_;
  }
  return true;
}

bool Session::RunDecodeGraph(std::string* error_msg) {
  State& st = *state_;
  const Config& c = model_->config();
  const rocm::DeviceModel& device = *model_->device_;
  if (st.decode_graph == nullptr && !st.graph_unavailable) {
    hipGraph_t graph = nullptr;
    bool ok = hipStreamBeginCapture(
                  st.stream, hipStreamCaptureModeThreadLocal) == hipSuccess;
    if (ok) {
      Forward forward(c, device, st);
      ok = forward.Enqueue(1, 1, st.last_logits.data);
      ok = hipStreamEndCapture(st.stream, &graph) == hipSuccess && ok &&
           graph != nullptr;
    }
    ok = ok && hipGraphInstantiate(&st.decode_graph, graph, nullptr, nullptr,
                                   0) == hipSuccess;
    if (graph != nullptr)
      static_cast<void>(hipGraphDestroy(graph));
    if (!ok) {
      // Capture is an optimization; replay the same kernels directly.
      static_cast<void>(hipGetLastError());
      st.decode_graph = nullptr;
      st.graph_unavailable = true;
    }
  }
  if (st.decode_graph != nullptr) {
    if (hipGraphLaunch(st.decode_graph, st.stream) == hipSuccess)
      return true;
    if (error_msg != nullptr)
      *error_msg = "decode graph launch failed";
    return false;
  }
  Forward forward(c, device, st);
  if (forward.Enqueue(1, 1, st.last_logits.data))
    return true;
  if (error_msg != nullptr)
    *error_msg = forward.error();
  return false;
}

bool Session::Reset(std::string* error_msg) {
  State& st = *state_;
  bool ok = true;
  for (auto& b : st.conv_state)
    ok = ok && (b.data == nullptr ||
                hipMemsetAsync(b.data, 0, b.count * sizeof(float), st.stream) ==
                    hipSuccess);
  for (auto& b : st.ssm_state)
    ok = ok && (b.data == nullptr ||
                hipMemsetAsync(b.data, 0, b.count * sizeof(float), st.stream) ==
                    hipSuccess);
  ok = ok && hipStreamSynchronize(st.stream) == hipSuccess;
  position_ = 0;
  verified_ = 0;
  needs_reset_ = !ok;
  if (!ok && error_msg != nullptr)
    *error_msg = "could not clear the session state";
  return ok;
}

void Session::Invalidate() noexcept {
  position_ = 0;
  verified_ = 0;
  needs_reset_ = true;
}

std::size_t SessionSnapshot::Bytes() const noexcept {
  std::size_t bytes = draft.size();
  for (const auto& rows : kv)
    bytes += rows.size() * sizeof(std::uint16_t);
  for (const auto& state : recurrent)
    bytes += state.size() * sizeof(float);
  return bytes;
}

std::size_t Session::SnapshotBytes() const noexcept {
  const Config& c = model_->config();
  std::size_t bytes = 0;
  for (std::size_t il = 0; il < c.num_layers; ++il) {
    if (c.blocks[il] == BlockType::kAttention)
      bytes += 2ull * position_ * c.AttentionKvDim() * sizeof(__half);
    if (c.blocks[il] == BlockType::kMamba)
      bytes += (state_->conv_state[il].count + state_->ssm_state[il].count) *
               sizeof(float);
  }
  if (state_->draft != nullptr)
    bytes += state_->draft->StateBytes();
  return bytes;
}

bool Session::Save(SessionSnapshot* out, std::string* error_msg) const {
  const Config& c = model_->config();
  const State& st = *state_;
  if (needs_reset_) {
    if (error_msg != nullptr)
      *error_msg = "the session was invalidated";
    return false;
  }
  out->position = position_;
  out->kv.clear();
  out->recurrent.clear();
  const std::size_t kv_count =
      static_cast<std::size_t>(position_) * c.AttentionKvDim();
  bool ok = true;
  for (std::size_t il = 0; ok && il < c.num_layers; ++il) {
    if (c.blocks[il] == BlockType::kAttention) {
      auto& rows = out->kv.emplace_back(2 * kv_count);
      ok = hipMemcpyAsync(rows.data(), st.k_cache[il].data,
                          kv_count * sizeof(__half), hipMemcpyDeviceToHost,
                          st.stream) == hipSuccess &&
           hipMemcpyAsync(rows.data() + kv_count, st.v_cache[il].data,
                          kv_count * sizeof(__half), hipMemcpyDeviceToHost,
                          st.stream) == hipSuccess;
    } else if (c.blocks[il] == BlockType::kMamba) {
      const std::size_t conv = st.conv_state[il].count;
      auto& state = out->recurrent.emplace_back(conv + st.ssm_state[il].count);
      ok = hipMemcpyAsync(state.data(), st.conv_state[il].data,
                          conv * sizeof(float), hipMemcpyDeviceToHost,
                          st.stream) == hipSuccess &&
           hipMemcpyAsync(state.data() + conv, st.ssm_state[il].data,
                          st.ssm_state[il].count * sizeof(float),
                          hipMemcpyDeviceToHost, st.stream) == hipSuccess;
    }
  }
  ok = ok && hipStreamSynchronize(st.stream) == hipSuccess;
  out->draft.clear();
  if (ok && st.draft != nullptr) {
    out->draft.resize(st.draft->StateBytes());
    ok = st.draft->SaveState(out->draft);
  }
  if (!ok && error_msg != nullptr)
    *error_msg = "could not copy the session state to the host";
  return ok;
}

bool Session::Restore(const SessionSnapshot& snapshot, std::string* error_msg) {
  auto fail = [&](std::string message) {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    needs_reset_ = true;
    return false;
  };
  const Config& c = model_->config();
  State& st = *state_;
  if (snapshot.position > max_context_)
    return fail("the snapshot exceeds this session's context");
  const std::size_t kv_count =
      static_cast<std::size_t>(snapshot.position) * c.AttentionKvDim();
  std::size_t attention = 0;
  std::size_t mamba = 0;
  bool ok = true;
  for (std::size_t il = 0; ok && il < c.num_layers; ++il) {
    if (c.blocks[il] == BlockType::kAttention) {
      if (attention >= snapshot.kv.size() ||
          snapshot.kv[attention].size() != 2 * kv_count)
        return fail("the snapshot does not match this model");
      const auto& rows = snapshot.kv[attention++];
      ok = hipMemcpyAsync(st.k_cache[il].data, rows.data(),
                          kv_count * sizeof(__half), hipMemcpyHostToDevice,
                          st.stream) == hipSuccess &&
           hipMemcpyAsync(st.v_cache[il].data, rows.data() + kv_count,
                          kv_count * sizeof(__half), hipMemcpyHostToDevice,
                          st.stream) == hipSuccess;
    } else if (c.blocks[il] == BlockType::kMamba) {
      const std::size_t conv = st.conv_state[il].count;
      if (mamba >= snapshot.recurrent.size() ||
          snapshot.recurrent[mamba].size() != conv + st.ssm_state[il].count)
        return fail("the snapshot does not match this model");
      const auto& state = snapshot.recurrent[mamba++];
      ok = hipMemcpyAsync(st.conv_state[il].data, state.data(),
                          conv * sizeof(float), hipMemcpyHostToDevice,
                          st.stream) == hipSuccess &&
           hipMemcpyAsync(st.ssm_state[il].data, state.data() + conv,
                          st.ssm_state[il].count * sizeof(float),
                          hipMemcpyHostToDevice, st.stream) == hipSuccess;
    }
  }
  if (!ok || hipStreamSynchronize(st.stream) != hipSuccess)
    return fail("could not copy the snapshot to the device");
  // A snapshot without drafter state leaves the drafter's context stale:
  // drafts get worse until it refills, outputs stay the same.
  if (st.draft != nullptr && !snapshot.draft.empty() &&
      !st.draft->RestoreState(snapshot.draft))
    return fail("could not restore the drafter state");
  position_ = snapshot.position;
  verified_ = 0;
  needs_reset_ = false;
  return true;
}

}  // namespace gufo::models::nemotron35_bolt
