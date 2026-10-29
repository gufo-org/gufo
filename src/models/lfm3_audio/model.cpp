#include "src/models/lfm3_audio/model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/lfm3_audio/hip/ops.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
namespace gufo::models::lfm3_audio {
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
  void* data{};
  std::size_t bytes{};
  Buffer() = default;
  explicit Buffer(std::size_t size) : bytes(size) {
    if (size)
      Check(hipMalloc(&data, size));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(Buffer&& b) noexcept
      : data(std::exchange(b.data, nullptr)),
        bytes(std::exchange(b.bytes, 0)) {}
  Buffer& operator=(Buffer&& b) noexcept {
    std::swap(data, b.data);
    std::swap(bytes, b.bytes);
    return *this;
  }
  Buffer(const Buffer&) = delete;
  float* floats() const { return static_cast<float*>(data); }
  std::int32_t* ints() const { return static_cast<std::int32_t*>(data); }
  // Grow-only reallocation; contents are not preserved.
  void Reserve(std::size_t size) {
    if (size > bytes)
      *this = Buffer(size);
  }
};
struct Stream {
  hipStream_t value{};
  Stream() { Check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() {
    if (value)
      (void)hipStreamDestroy(value);
  }
  Stream(const Stream&) = delete;
};
using gpu::Activation;
constexpr int kRows = 512;  // rows per stack launch

// Recurrent state of one causal stack: convolution histories and sliding
// key/value rings, by block.
struct StackState {
  std::vector<Buffer> conv, k, v;
  std::uint32_t capacity{};
  int position{};
  StackState() = default;
  StackState(const Stack& s, std::uint32_t window) : capacity(window) {
    for (const auto& b : s.blocks) {
      if (b.attention()) {
        conv.emplace_back();
        k.emplace_back(std::size_t(window) * s.kv_heads * s.head_dim * 4);
        v.emplace_back(std::size_t(window) * s.kv_heads * s.head_dim * 4);
      } else {
        conv.emplace_back(std::size_t(s.hidden) * (s.conv_width - 1) * 4);
        k.emplace_back();
        v.emplace_back();
      }
    }
  }
  void Clear(const Stack& s, hipStream_t stream) {
    for (auto& c : conv)
      if (c.data)
        Check(hipMemsetAsync(c.data, 0, c.bytes, stream));
    position = 0;
  }
};
// Activations of one stack for up to kRows rows.
struct Scratch {
  Buffer x, z, o, p, q, k, v, gate, a, ff_gate, ff_up, quantized, row_max,
      frequencies;
  // min_width bounds the input width of extra projections run with it.
  Scratch(const Stack& s, hipStream_t stream, std::size_t min_width = 0) {
    const std::size_t R = kRows, H = s.hidden;
    x = Buffer(R * H * 4);
    z = Buffer(R * H * 4);
    o = Buffer(R * H * 4);
    p = Buffer(R * 3 * H * 4);
    q = Buffer(R * s.heads * s.head_dim * 4);
    k = Buffer(R * s.kv_heads * s.head_dim * 4);
    v = Buffer(R * s.kv_heads * s.head_dim * 4);
    gate = Buffer(R * s.heads * 4);
    a = Buffer(R * std::max<std::size_t>(H, s.heads * s.head_dim) * 4);
    ff_gate = Buffer(R * s.intermediate * 4);
    ff_up = Buffer(R * s.intermediate * 4);
    const auto width =
        std::max<std::size_t>({3 * H, s.intermediate, H, min_width});
    quantized = Buffer(lfm3::gpu::DenseWorkspaceBytes(R, width));
    row_max = Buffer(R * sizeof(unsigned));
    frequencies = Buffer(std::size_t(s.head_dim) * 2);
    lfm3::gpu::RopeFrequencies(frequencies.floats(), s.head_dim, s.theta,
                               stream);
  }
};
void Dense(const Tensor& w, const float* x, float* y, int n, hipStream_t s,
           Scratch& scratch) {
  lfm3::gpu::Dense(w, x, y, n, s, {scratch.quantized.data});
}
void DenseBias(const Tensor& w, const Tensor& bias, const float* x, float* y,
               int n, Activation act, hipStream_t s, Scratch& scratch) {
  Dense(w, x, y, n, s, scratch);
  gpu::BiasAct(y, static_cast<const float*>(bias.data), n, w.rows, act, s);
}
const float* F(const Tensor& t) {
  return static_cast<const float*>(t.data);
}
// Runs n <= kRows rows of scratch.x through the stack and writes the
// output-normed rows to scratch.z. Without state, attention is
// bidirectional over the rows and convolutions are absent.
void RunStack(const Stack& st, StackState* state, int n, int window,
              Scratch& sc, hipStream_t s) {
  const int H = st.hidden;
  const int position = state ? state->position : 0;
  auto norm = [&](const float* in, const Tensor& w, float* out) {
    gufo::hip::LaunchBatchedRMSNorm(in, F(w), out, nullptr, n, H, st.eps, s);
  };
  for (std::size_t i = 0; i < st.blocks.size(); ++i) {
    const auto& b = st.blocks[i];
    norm(sc.x.floats(), b.norm, sc.z.floats());
    if (b.attention()) {
      Dense(b.q, sc.z.floats(), sc.q.floats(), n, s, sc);
      Dense(b.k, sc.z.floats(), sc.k.floats(), n, s, sc);
      Dense(b.v, sc.z.floats(), sc.v.floats(), n, s, sc);
      if (b.attn_gate.data)
        Dense(b.attn_gate, sc.z.floats(), sc.gate.floats(), n, s, sc);
      gufo::hip::LaunchBatchedPerHeadRMSNorm(sc.q.floats(), F(b.q_norm),
                                             sc.q.floats(), n, st.heads,
                                             st.head_dim, st.eps, s);
      gufo::hip::LaunchBatchedPerHeadRMSNorm(sc.k.floats(), F(b.k_norm),
                                             sc.k.floats(), n, st.kv_heads,
                                             st.head_dim, st.eps, s);
      lfm3::gpu::Rope(sc.q.floats(), sc.frequencies.floats(), nullptr, n,
                      st.heads, st.head_dim, position, s);
      lfm3::gpu::Rope(sc.k.floats(), sc.frequencies.floats(), nullptr, n,
                      st.kv_heads, st.head_dim, position, s);
      const float* ck = state ? state->k[i].floats() : nullptr;
      const float* cv = state ? state->v[i].floats() : nullptr;
      gpu::Attention(sc.q.floats(), sc.k.floats(), sc.v.floats(),
                     b.attn_gate.data ? sc.gate.floats() : nullptr, ck, cv,
                     sc.a.floats(), n, st.heads, st.kv_heads, st.head_dim,
                     position, state ? int(state->capacity) : 1, window,
                     state != nullptr, s);
      if (state)
        lfm3::gpu::StoreKV(sc.k.floats(), sc.v.floats(), state->k[i].floats(),
                           state->v[i].floats(), n, st.kv_heads * st.head_dim,
                           position, state->capacity, s);
      Dense(b.attn_out, sc.a.floats(), sc.o.floats(), n, s, sc);
    } else {
      Dense(b.in_proj, sc.z.floats(), sc.p.floats(), n, s, sc);
      lfm3::gpu::ShortConv(sc.p.floats(), F(b.conv), state->conv[i].floats(),
                           sc.a.floats(), n, H, st.conv_width, s);
      Dense(b.out_proj, sc.a.floats(), sc.o.floats(), n, s, sc);
    }
    if (b.post_norm.data)
      norm(sc.o.floats(), b.post_norm, sc.o.floats());
    gufo::hip::LaunchBatchedResidualAdd(sc.x.floats(), sc.o.floats(),
                                        sc.x.floats(), n, H, s);
    norm(sc.x.floats(), b.ffn_norm, sc.z.floats());
    lfm3::gpu::ProjectFfn(
        b.gate, b.up, b.down, sc.z.floats(), sc.ff_gate.floats(),
        sc.ff_up.floats(), sc.o.floats(), n, s,
        {sc.quantized.data, static_cast<unsigned*>(sc.row_max.data)});
    if (b.post_ffn_norm.data)
      norm(sc.o.floats(), b.post_ffn_norm, sc.o.floats());
    gufo::hip::LaunchBatchedResidualAdd(sc.x.floats(), sc.o.floats(),
                                        sc.x.floats(), n, H, s);
  }
  norm(sc.x.floats(), st.output_norm, sc.z.floats());
  if (state)
    state->position += n;
}
// Pinned host staging for small per-frame transfers.
struct Pinned {
  void* data{};
  explicit Pinned(std::size_t bytes) {
    Check(hipHostMalloc(&data, bytes, hipHostMallocDefault));
  }
  ~Pinned() {
    if (data)
      (void)hipHostFree(data);
  }
  Pinned(const Pinned&) = delete;
  std::int32_t* ints() const { return static_cast<std::int32_t*>(data); }
  float* floats() const { return static_cast<float*>(data); }
};
}  // namespace

struct Model::Impl {
  std::shared_ptr<lfm3::Model> backbone;
  std::unique_ptr<core::GgufReader> reader;
  AudioWeights weights;
  Tokens tokens;
  std::vector<Buffer> allocations;
  Buffer istft_basis;  // [n_fft][n_fft + 2], synthesis window included
  std::vector<float> window;
  // Encoder state, serialized by encoder_mutex.
  std::mutex encoder_mutex;
  std::unique_ptr<Stream> encoder_stream;
  std::unique_ptr<Scratch> encoder_scratch, projector_scratch;
  Buffer samples, stft_frames, spectrum, mel, c1, c2, c3, flat, embeddings;
  Buffer stft_window, dft_basis, mel_filters;
  // Computes impl.mel ([frames][128]) on the encoder stream.
  std::uint32_t ComputeMel(std::span<const float> pcm, Scratch& sc);
};
namespace {
constexpr int kFft = 400, kHop = 160, kBins = kFft / 2 + 1;
double HzToMel(double f) {
  return f < 1000 ? 3 * f / 200
                  : 15 + std::log(f / 1000) / (std::log(6.4) / 27);
}
double MelToHz(double m) {
  return m < 15 ? 200 * m / 3 : 1000 * std::exp(std::log(6.4) / 27 * (m - 15));
}
}  // namespace
std::uint32_t MelFrames(std::size_t samples) {
  const std::size_t padded =
      samples < kFft ? kFft : samples + (kHop - (samples - kFft) % kHop) % kHop;
  return std::uint32_t(1 + (padded - kFft) / kHop);
}
std::uint32_t Model::Impl::ComputeMel(std::span<const float> pcm, Scratch& sc) {
  constexpr int B = AudioWeights::kMelBins;
  const auto s = encoder_stream->value;
  const auto T = MelFrames(pcm.size());
  const std::size_t padded = std::size_t(T - 1) * kHop + kFft;
  samples.Reserve(padded * 4);
  stft_frames.Reserve(std::size_t(T) * kFft * 4);
  spectrum.Reserve(std::size_t(T) * 2 * kBins * 4);
  mel.Reserve(std::size_t(T) * B * 4);
  Check(hipMemsetAsync(samples.data, 0, padded * 4, s));
  Check(hipMemcpyAsync(samples.data, pcm.data(), pcm.size_bytes(),
                       hipMemcpyHostToDevice, s));
  gpu::Frames(samples.floats(), stft_window.floats(), stft_frames.floats(),
              int(T), kFft, kHop, s);
  gpu::Sgemm(stft_frames.floats(), dft_basis.floats(), nullptr,
             spectrum.floats(), int(T), 2 * kBins, kFft, Activation::kNone, s);
  gpu::Power(spectrum.floats(), stft_frames.floats(), int(T), kBins, s);
  gpu::Sgemm(stft_frames.floats(), mel_filters.floats(), nullptr, mel.floats(),
             int(T), B, kBins, Activation::kNone, s);
  gpu::LogMelNormalize(mel.floats(), std::size_t(T) * B, -12.0F, 4.0F, 4.0F, s);
  (void)sc;
  return T;
}
Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
lfm3::Model& Model::backbone() const {
  return *impl_->backbone;
}
const lfm3::Tokenizer& Model::tokenizer() const {
  return impl_->backbone->tokenizer();
}
const AudioWeights& Model::weights() const {
  return impl_->weights;
}
const Tokens& Model::tokens() const {
  return impl_->tokens;
}
std::shared_ptr<Model> Model::Load(const std::string& backbone,
                                   const std::string& mmproj,
                                   std::string* error) {
  try {
    auto m = std::shared_ptr<Model>(new Model);
    auto& impl = *m->impl_;
    impl.backbone = lfm3::Model::Load(backbone, error);
    if (!impl.backbone)
      return nullptr;
    impl.reader = core::GgufReader::OpenFile(mmproj, error);
    if (!impl.reader)
      return nullptr;
    auto w =
        AudioWeights::Bind(*impl.reader, impl.backbone->config().hidden, error);
    if (!w)
      return nullptr;
    impl.weights = std::move(*w);
    const auto& tok = impl.backbone->tokenizer();
    auto id = [&](std::string_view text) {
      const auto ids = tok.Encode(text, true);
      if (ids.size() != 1)
        throw std::runtime_error("tokenizer lacks LFM3-Audio token " +
                                 std::string(text));
      return ids[0];
    };
    impl.tokens = {id("<|audio|>"),       id("<|audio_out|>"),
                   id("<|audio_start|>"), id("<|text_end|>"),
                   id("<|mixed_start|>"), id("<|im_end|>"),
                   id("<|im_start|>"),    id("<|startoftext|>")};
    std::unordered_map<const void*, void*> uploaded;
    impl.weights.Visit([&](Tensor& t) {
      if (!t.data)
        return;
      if (!uploaded.contains(t.data)) {
        impl.allocations.emplace_back(t.bytes);
        auto* p = impl.allocations.back().data;
        Check(hipMemcpy(p, t.data, t.bytes, hipMemcpyHostToDevice));
        uploaded[t.data] = p;
      }
      t.data = uploaded.at(t.data);
    });
    // Inverse real FFT with the periodic Hann synthesis window folded in.
    const int N = int(impl.weights.n_fft), bins = N / 2 + 1;
    impl.window.resize(N);
    for (int n = 0; n < N; ++n)
      impl.window[n] =
          float(0.5 - 0.5 * std::cos(2 * std::numbers::pi * n / N));
    std::vector<float> basis(std::size_t(N) * 2 * bins);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < bins; ++k) {
        const double c = (k == 0 || k == N / 2) ? 1.0 : 2.0;
        // Exact phase reduction keeps the angle small for large k * n.
        const double angle =
            2 * std::numbers::pi * double((std::int64_t(k) * n) % N) / N;
        const double scale = c / N * impl.window[n];
        basis[std::size_t(n) * 2 * bins + k] = float(scale * std::cos(angle));
        basis[std::size_t(n) * 2 * bins + bins + k] =
            float(-scale * std::sin(angle));
      }
    impl.istft_basis = Buffer(basis.size() * 4);
    Check(hipMemcpy(impl.istft_basis.data, basis.data(), basis.size() * 4,
                    hipMemcpyHostToDevice));
    {
      // Analysis tables: periodic Hann, DFT rows [cos | -sin], Slaney mel.
      std::vector<float> hann(kFft), dft(std::size_t(2) * kBins * kFft);
      for (int n = 0; n < kFft; ++n)
        hann[n] = float(0.5 - 0.5 * std::cos(2 * std::numbers::pi * n / kFft));
      for (int k = 0; k < kBins; ++k)
        for (int n = 0; n < kFft; ++n) {
          const double angle =
              2 * std::numbers::pi * double((k * n) % kFft) / kFft;
          dft[std::size_t(k) * kFft + n] = float(std::cos(angle));
          dft[std::size_t(kBins + k) * kFft + n] = float(-std::sin(angle));
        }
      constexpr int B = AudioWeights::kMelBins;
      std::vector<double> points(B + 2);
      const double low = HzToMel(0), high = HzToMel(8000);
      for (int i = 0; i < B + 2; ++i)
        points[i] = MelToHz(low + (high - low) * i / (B + 1));
      std::vector<float> filters(std::size_t(B) * kBins);
      for (int m = 0; m < B; ++m)
        for (int k = 0; k < kBins; ++k) {
          const double f = 8000.0 * k / (kBins - 1);
          const double down = (f - points[m]) / (points[m + 1] - points[m]);
          const double up =
              (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
          const double value = std::max(0.0, std::min(down, up));
          filters[std::size_t(m) * kBins + k] =
              float(value * 2 / (points[m + 2] - points[m]));
        }
      auto upload = [](Buffer& b, const std::vector<float>& v) {
        b = Buffer(v.size() * 4);
        Check(hipMemcpy(b.data, v.data(), v.size() * 4, hipMemcpyHostToDevice));
      };
      upload(impl.stft_window, hann);
      upload(impl.dft_basis, dft);
      upload(impl.mel_filters, filters);
    }
    impl.encoder_stream = std::make_unique<Stream>();
    impl.encoder_scratch = std::make_unique<Scratch>(impl.weights.encoder,
                                                     impl.encoder_stream->value,
                                                     impl.weights.pre_out.cols);
    // The projector reuses the Dense workspace sized for backbone width.
    Stack wide = impl.weights.encoder;
    wide.hidden = impl.backbone->config().hidden;
    wide.intermediate = wide.hidden;
    impl.projector_scratch =
        std::make_unique<Scratch>(wide, impl.encoder_stream->value);
    Check(hipStreamSynchronize(impl.encoder_stream->value));
    return m;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
std::uint32_t EncoderTokens(std::uint32_t frames) {
  return (frames + 7) / 8;
}
bool Model::Features(std::span<const float> pcm, std::vector<float>* mel,
                     std::uint32_t* frames, std::string* error) {
  try {
    auto& impl = *impl_;
    const std::lock_guard lock(impl.encoder_mutex);
    const auto T = impl.ComputeMel(pcm, *impl.encoder_scratch);
    mel->resize(std::size_t(T) * AudioWeights::kMelBins);
    Check(hipMemcpyAsync(mel->data(), impl.mel.data, mel->size() * 4,
                         hipMemcpyDeviceToHost, impl.encoder_stream->value));
    Check(hipStreamSynchronize(impl.encoder_stream->value));
    *frames = T;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Model::Encode(std::span<const float> pcm, std::vector<float>* embeddings,
                   std::string* error) {
  try {
    constexpr int B = AudioWeights::kMelBins;
    if (pcm.empty())
      throw std::invalid_argument("empty LFM3-Audio clip");
    auto& impl = *impl_;
    const auto& w = impl.weights;
    const auto& e = w.encoder;
    const std::lock_guard lock(impl.encoder_mutex);
    const auto s = impl.encoder_stream->value;
    auto& sc = *impl.encoder_scratch;
    auto& ps = *impl.projector_scratch;
    const auto frames = impl.ComputeMel(pcm, sc);
    const int C = int(w.downsample_channels);
    const int hidden = int(impl.backbone->config().hidden);
    const int tokens = int(EncoderTokens(frames));
    StackState state(e, w.encoder_window);
    state.Clear(e, s);
    embeddings->resize(std::size_t(tokens) * hidden);
    // Token t reads frames 8t - 7 .. 8t + 7 through three stride-2 levels;
    // two tokens of margin make every kept token exact at window edges.
    constexpr int kMargin = 16;
    const int max_frames = kRows * 8 + 2 * kMargin;
    impl.c1.Reserve(std::size_t(max_frames / 2 + 1) * (B / 2) * C * 4);
    impl.c2.Reserve(std::size_t(max_frames / 4 + 1) * (B / 4) * C * 4);
    impl.c3.Reserve(std::size_t(max_frames / 4 + 1) * (B / 4) * C * 4);
    impl.flat.Reserve(std::size_t(max_frames / 8 + 1) * (B / 8) * C * 4);
    impl.embeddings.Reserve(std::size_t(kRows) * hidden * 4);
    for (int t0 = 0; t0 < tokens; t0 += kRows) {
      const int t1 = std::min(tokens, t0 + kRows), n = t1 - t0;
      const int f0 = std::max(0, 8 * t0 - kMargin);
      const int f1 = std::min(int(frames), 8 * t1 + kMargin);
      const int T = f1 - f0;
      const int W1 = (T + 1) / 2, H1 = (B + 1) / 2;
      const int W2 = (W1 + 1) / 2, H2 = (H1 + 1) / 2;
      const int W3 = (W2 + 1) / 2, H3 = (H2 + 1) / 2;
      gpu::ConvFirst(impl.mel.floats() + std::size_t(f0) * B,
                     F(w.conv_weight[0]), F(w.conv_bias[0]), impl.c1.floats(),
                     B, T, C, s);
      gpu::ConvDepthwise(impl.c1.floats(), F(w.conv_weight[1]),
                         F(w.conv_bias[1]), impl.c2.floats(), W1, H1, C, s);
      gpu::Sgemm(impl.c2.floats(), F(w.conv_weight[2]), F(w.conv_bias[2]),
                 impl.c3.floats(), W2 * H2, C, C, Activation::kSilu, s);
      gpu::ConvDepthwise(impl.c3.floats(), F(w.conv_weight[3]),
                         F(w.conv_bias[3]), impl.c2.floats(), W2, H2, C, s);
      gpu::Sgemm(impl.c2.floats(), F(w.conv_weight[4]), F(w.conv_bias[4]),
                 impl.c3.floats(), W3 * H3, C, C, Activation::kSilu, s);
      gpu::FlattenChannels(impl.c3.floats(), impl.flat.floats(), W3, H3, C, s);
      const int skip = t0 - f0 / 8;
      DenseBias(w.pre_out, w.pre_out_bias,
                impl.flat.floats() + std::size_t(skip) * H3 * C, sc.x.floats(),
                n, Activation::kNone, s, sc);
      RunStack(e, &state, n, int(w.encoder_window), sc, s);
      // Projector: LayerNorm, Linear + GELU, Linear.
      gpu::LayerNorm(sc.z.floats(), F(w.proj_norm), F(w.proj_norm_bias),
                     sc.o.floats(), n, e.hidden, 1e-5F, s);
      DenseBias(w.proj1, w.proj1_bias, sc.o.floats(), ps.x.floats(), n,
                Activation::kGelu, s, ps);
      DenseBias(w.proj2, w.proj2_bias, ps.x.floats(), impl.embeddings.floats(),
                n, Activation::kNone, s, ps);
      Check(hipMemcpyAsync(embeddings->data() + std::size_t(t0) * hidden,
                           impl.embeddings.data, std::size_t(n) * hidden * 4,
                           hipMemcpyDeviceToHost, s));
      Check(hipStreamSynchronize(s));
    }
    Check(hipGetLastError());
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}

struct Voice::Impl {
  static constexpr int K = AudioWeights::kCodebooks;
  static constexpr int V = AudioWeights::kCodebookSize;
  std::shared_ptr<Model> model;
  // Frames sample on stream; the codec decodes on codec_stream, so a
  // caller may overlap decoding with the next backbone step.
  Stream stream, codec_stream;
  Scratch depth, codec;
  StackState codec_state;
  Buffer depth_base, head_inputs, logits, codes, masked, which, frame, counts,
      ring, position, randoms, codec_codes, latents, quantized_latents,
      spectrum, frames;
  Pinned host_codes, host_masked, host_which, host_logits, host_randoms,
      host_frame, host_codec_codes;
  // The captured frame graph and what it was captured with.
  hipGraphExec_t graph{};
  const float* graph_hidden{};
  float* graph_embedding{};
  gpu::FrameSampling graph_sampling{};
  bool graph_failed{};
  std::size_t window{};
  // Overlap-add carry of the codec's synthesis.
  std::vector<float> carry_audio, carry_envelope, host_frames;
  bool started{};
  explicit Impl(std::shared_ptr<Model> m)
      : model(std::move(m)),
        depth(model->weights().depth, stream.value),
        codec(model->weights().codec, codec_stream.value),
        codec_state(model->weights().codec, model->weights().codec_window),
        host_codes(K * 4),
        host_masked(K * 4),
        host_which(K * 4),
        host_logits(std::size_t(K) * V * 4),
        host_randoms(2 * K * 4),
        host_frame(K * 4),
        host_codec_codes(std::size_t(K) * kRows * 4) {
    const auto& w = model->weights();
    depth_base = Buffer(std::size_t(K) * w.depth.hidden * 4);
    head_inputs = Buffer(std::size_t(K) * w.depth.hidden * 4);
    logits = Buffer(std::size_t(K) * V * 4);
    codes = Buffer(K * 4);
    masked = Buffer(K * 4);
    which = Buffer(K * 4);
    frame = Buffer(std::size_t(model->backbone().config().hidden) * 4);
    counts = Buffer(std::size_t(K) * V * 4);
    position = Buffer(2 * 4);
    randoms = Buffer(2 * K * 4);  // reveal order, then uniforms
    codec_codes = Buffer(std::size_t(K) * kRows * 4);
    const int max_frames = kRows / int(w.frame_stack);
    latents = Buffer(std::size_t(max_frames) * K * 4 * 4);
    quantized_latents = Buffer(std::size_t(max_frames) * w.fsq_out.rows * 4);
    spectrum = Buffer(std::size_t(kRows) * w.codec_out.rows * 4);
    frames = Buffer(std::size_t(kRows) * w.n_fft * 4);
    Reset();
  }
  ~Impl() {
    if (graph)
      (void)hipGraphExecDestroy(graph);
  }
  void Reset() {
    codec_state.Clear(model->weights().codec, codec_stream.value);
    Check(hipStreamSynchronize(codec_stream.value));
    Check(hipMemsetAsync(counts.data, 0, counts.bytes, stream.value));
    Check(hipMemsetAsync(position.data, 0, position.bytes, stream.value));
    Check(hipStreamSynchronize(stream.value));
    carry_audio.clear();
    carry_envelope.clear();
    started = false;
  }
  gpu::FrameState State() {
    return {codes.ints(), masked.ints(), counts.ints(), ring.ints(),
            position.ints()};
  }
  void Base(const float* device_hidden, hipStream_t s) {
    const auto& w = model->weights();
    if (!gpu::WideGemv(w.depth_in, device_hidden, depth_base.floats(), s))
      lfm3::gpu::Dense(w.depth_in, device_hidden, depth_base.floats(), 1, s);
    gpu::BiasAct(depth_base.floats(), F(w.depth_in_bias), 1, w.depth_in.rows,
                 Activation::kNone, s);
  }
  // The depth stack over the 12 slots for the current codes and mask.
  void Trunk(hipStream_t s) {
    const auto& w = model->weights();
    gpu::DepthInputs(depth_base.floats(), F(w.mask_embedding),
                     w.code_embeddings, codes.ints(), masked.ints(),
                     depth.x.floats(), K, V, s);
    RunStack(w.depth, nullptr, K, 0, depth, s);
  }
  // Every kernel of one frame: three reveal passes, sampling, repetition
  // window and the next backbone input. No host synchronization inside.
  void Record(const float* hidden, float* embedding,
              const gpu::FrameSampling& sampling, hipStream_t s) {
    const auto& w = model->weights();
    Check(hipMemcpyAsync(randoms.data, host_randoms.data, randoms.bytes,
                         hipMemcpyHostToDevice, s));
    Base(hidden, s);
    gpu::StartFrame(State(), K, s);
    const auto* order = randoms.ints();
    const auto* uniforms = randoms.floats() + K;
    constexpr int kPasses = 3, kPerPass = K / kPasses;
    for (int pass = 0; pass < kPasses; ++pass) {
      const auto* picked = order + pass * kPerPass;
      Trunk(s);
      gpu::CodebookNorm(depth.z.floats(), F(w.head_norms), picked,
                        head_inputs.floats(), kPerPass, int(w.depth.hidden),
                        w.depth.eps, s);
      gpu::CodebookLogits(w.code_embeddings, head_inputs.floats(), picked,
                          logits.floats(), kPerPass, V, s);
      gpu::SampleCodes(logits.floats(), picked, uniforms, State(), sampling,
                       kPerPass, V, s);
    }
    gpu::FinishFrame(w.audio_out, State(), sampling, embedding, K, V,
                     AudioWeights::kCodes, s);
    Check(hipMemcpyAsync(host_frame.data, codes.data, K * 4,
                         hipMemcpyDeviceToHost, s));
  }
  void Frame(const float* hidden, float* embedding, const Sampling& sampling,
             std::mt19937_64& rng, bool ordered, std::span<std::int32_t> out) {
    if (sampling.repetition_window > window) {
      window = sampling.repetition_window;
      ring = Buffer(std::max<std::size_t>(window, 1) * K * 4);
      Check(hipMemsetAsync(position.data, 0, position.bytes, stream.value));
      Check(hipMemsetAsync(counts.data, 0, counts.bytes, stream.value));
      if (graph) {
        (void)hipGraphExecDestroy(graph);
        graph = nullptr;
      }
    }
    std::array<int, K> order;
    for (int i = 0; i < K; ++i)
      order[i] = i;
    if (!ordered)
      std::shuffle(order.begin(), order.end(), rng);
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    for (int i = 0; i < K; ++i) {
      host_randoms.ints()[i] = order[i];
      host_randoms.floats()[K + i] = uniform(rng);
    }
    const gpu::FrameSampling fs{sampling.temperature,
                                sampling.repetition_penalty, sampling.top_k,
                                int(sampling.repetition_window)};
    if (!embedding)
      embedding = frame.floats();
    const auto s = stream.value;
    const bool reusable = graph && graph_hidden == hidden &&
                          graph_embedding == embedding &&
                          std::memcmp(&graph_sampling, &fs, sizeof(fs)) == 0;
    if (!reusable && !graph_failed) {
      if (graph) {
        (void)hipGraphExecDestroy(graph);
        graph = nullptr;
      }
      hipGraph_t captured{};
      bool ok = hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal) ==
                hipSuccess;
      if (ok) {
        Record(hidden, embedding, fs, s);
        ok = hipStreamEndCapture(s, &captured) == hipSuccess && captured;
      }
      ok = ok && hipGraphInstantiate(&graph, captured, nullptr, nullptr, 0) ==
                     hipSuccess;
      if (captured)
        (void)hipGraphDestroy(captured);
      (void)hipGetLastError();
      if (ok) {
        graph_hidden = hidden;
        graph_embedding = embedding;
        graph_sampling = fs;
      } else {
        graph = nullptr;
        graph_failed = true;
      }
    }
    if (graph)
      Check(hipGraphLaunch(graph, s));
    else
      Record(hidden, embedding, fs, s);
    Check(hipStreamSynchronize(s));
    std::memcpy(out.data(), host_frame.data, K * 4);
  }
  // Logits of the codebooks in which for given codes and mask (tests).
  void Pass(std::span<const std::int32_t> code_values,
            std::span<const std::int32_t> mask_values,
            std::span<const std::int32_t> which_values) {
    const auto& w = model->weights();
    const auto s = stream.value;
    std::memcpy(host_codes.data, code_values.data(), K * 4);
    std::memcpy(host_masked.data, mask_values.data(), K * 4);
    std::memcpy(host_which.data, which_values.data(), which_values.size() * 4);
    Check(hipMemcpyAsync(codes.data, host_codes.data, K * 4,
                         hipMemcpyHostToDevice, s));
    Check(hipMemcpyAsync(masked.data, host_masked.data, K * 4,
                         hipMemcpyHostToDevice, s));
    Check(hipMemcpyAsync(which.data, host_which.data, which_values.size() * 4,
                         hipMemcpyHostToDevice, s));
    Trunk(s);
    gpu::CodebookNorm(depth.z.floats(), F(w.head_norms), which.ints(),
                      head_inputs.floats(), int(which_values.size()),
                      int(w.depth.hidden), w.depth.eps, s);
    gpu::CodebookLogits(w.code_embeddings, head_inputs.floats(), which.ints(),
                        logits.floats(), int(which_values.size()), V, s);
    Check(hipMemcpyAsync(host_logits.data, logits.data,
                         which_values.size() * V * 4, hipMemcpyDeviceToHost,
                         s));
    Check(hipStreamSynchronize(s));
  }
};
Voice::Voice(std::shared_ptr<Model> model)
    : impl_(std::make_unique<Impl>(std::move(model))) {}
Voice::~Voice() {
  if (impl_) {
    (void)hipStreamSynchronize(impl_->stream.value);
    (void)hipStreamSynchronize(impl_->codec_stream.value);
  }
}
std::unique_ptr<Voice> Model::CreateVoice(std::string* error) {
  try {
    return std::unique_ptr<Voice>(new Voice(shared_from_this()));
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
void Voice::Reset() {
  impl_->Reset();
}
bool Voice::SampleFrame(const float* hidden, const Sampling& sampling,
                        std::mt19937_64& rng, std::span<std::int32_t> codes,
                        float* embedding, bool ordered, std::string* error) {
  try {
    if (codes.size() != AudioWeights::kCodebooks)
      throw std::invalid_argument("frame needs 12 codes");
    impl_->Frame(hidden, embedding, sampling, rng, ordered, codes);
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Voice::DepthLogits(std::span<const float> hidden,
                        std::span<const std::int32_t> codes,
                        std::span<const std::int32_t> masked,
                        std::span<float> logits, std::string* error) {
  try {
    constexpr auto K = AudioWeights::kCodebooks;
    constexpr auto V = AudioWeights::kCodebookSize;
    auto& impl = *impl_;
    if (hidden.size() != impl.model->backbone().config().hidden ||
        codes.size() != K || masked.size() != K ||
        logits.size() != std::size_t(K) * V)
      throw std::invalid_argument("invalid depth logit buffers");
    Check(hipMemcpyAsync(impl.frame.data, hidden.data(), hidden.size_bytes(),
                         hipMemcpyHostToDevice, impl.stream.value));
    impl.Base(impl.frame.floats(), impl.stream.value);
    std::array<std::int32_t, K> all;
    for (int i = 0; i < int(K); ++i)
      all[i] = i;
    impl.Pass(codes, masked, all);
    std::memcpy(logits.data(), impl.host_logits.data, logits.size_bytes());
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Voice::FrameEmbedding(std::span<const std::int32_t> codes, float* out,
                           std::string* error) {
  try {
    constexpr auto K = AudioWeights::kCodebooks;
    auto& impl = *impl_;
    const auto& w = impl.model->weights();
    if (codes.size() != K)
      throw std::invalid_argument("frame needs 12 codes");
    for (auto c : codes)
      if (c < 0 || c >= std::int32_t(AudioWeights::kCodebookSize))
        throw std::invalid_argument("frame code out of range");
    std::memcpy(impl.host_codes.data, codes.data(), K * 4);
    Check(hipMemcpyAsync(impl.codes.data, impl.host_codes.data, K * 4,
                         hipMemcpyHostToDevice, impl.stream.value));
    gpu::FrameEmbeddings(w.audio_out, impl.codes.ints(), out, 1, K,
                         AudioWeights::kCodebookSize, impl.stream.value);
    Check(hipStreamSynchronize(impl.stream.value));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Voice::Decode(std::span<const std::int32_t> all_frames, bool finalize,
                   std::vector<float>* samples, std::string* error) {
  try {
    constexpr int K = AudioWeights::kCodebooks;
    auto& impl = *impl_;
    const auto& w = impl.model->weights();
    const auto s = impl.codec_stream.value;
    if (all_frames.size() % K)
      throw std::invalid_argument("codec frames need 12 codes each");
    for (auto c : all_frames)
      if (c < 0 || c >= std::int32_t(AudioWeights::kCodes))
        throw std::invalid_argument("codec code out of range");
    const int N = int(w.n_fft), hop = int(w.hop), stack = int(w.frame_stack);
    const int trim = (N - hop) / 2, tail = 2 * trim;
    const int bins = N / 2 + 1;
    const auto& window = impl.model->impl_->window;
    samples->clear();
    const int total = int(all_frames.size() / K);
    const int chunk = kRows / stack;
    auto emit = [&](std::vector<float>& audio, std::vector<float>& envelope,
                    bool last) {
      if (!impl.carry_audio.empty()) {
        for (int i = 0; i < tail; ++i) {
          audio[i] += impl.carry_audio[i];
          envelope[i] += impl.carry_envelope[i];
        }
      }
      std::size_t begin = 0, end = audio.size();
      if (!impl.started) {
        begin = trim;
        impl.started = true;
      }
      if (last) {
        end -= trim;
        impl.carry_audio.clear();
        impl.carry_envelope.clear();
      } else {
        impl.carry_audio.assign(audio.end() - tail, audio.end());
        impl.carry_envelope.assign(envelope.end() - tail, envelope.end());
        end -= tail;
      }
      for (std::size_t i = begin; i < end; ++i)
        samples->push_back(audio[i] / envelope[i]);
    };
    if (!total) {
      if (finalize && impl.started && !impl.carry_audio.empty()) {
        std::vector<float> audio(tail), envelope(tail);
        emit(audio, envelope, true);
      }
      return true;
    }
    for (int f0 = 0; f0 < total; f0 += chunk) {
      const int frames = std::min(chunk, total - f0), rows = frames * stack;
      const bool last = finalize && f0 + frames == total;
      std::memcpy(impl.host_codec_codes.data,
                  all_frames.data() + std::size_t(f0) * K,
                  std::size_t(frames) * K * 4);
      Check(hipMemcpyAsync(impl.codec_codes.data, impl.host_codec_codes.data,
                           std::size_t(frames) * K * 4, hipMemcpyHostToDevice,
                           s));
      gpu::FsqLatents(F(w.fsq_table), impl.codec_codes.ints(),
                      impl.latents.floats(), frames, K, s);
      gpu::Sgemm(impl.latents.floats(), F(w.fsq_out), F(w.fsq_out_bias),
                 impl.quantized_latents.floats(), frames, int(w.fsq_out.rows),
                 int(w.fsq_out.cols), Activation::kNone, s);
      DenseBias(w.codec_in, w.codec_in_bias, impl.quantized_latents.floats(),
                impl.codec.o.floats(), frames, Activation::kNone, s,
                impl.codec);
      gpu::Repeat(impl.codec.o.floats(), impl.codec.x.floats(), frames,
                  int(w.codec.hidden), stack, s);
      RunStack(w.codec, &impl.codec_state, rows, int(w.codec_window),
               impl.codec, s);
      DenseBias(w.codec_out, w.codec_out_bias, impl.codec.z.floats(),
                impl.spectrum.floats(), rows, Activation::kNone, s, impl.codec);
      gpu::Polar(impl.spectrum.floats(), rows, bins, s);
      gpu::Sgemm(impl.spectrum.floats(),
                 impl.model->impl_->istft_basis.floats(), nullptr,
                 impl.frames.floats(), rows, N, 2 * bins, Activation::kNone, s);
      impl.host_frames.resize(std::size_t(rows) * N);
      Check(hipMemcpyAsync(impl.host_frames.data(), impl.frames.data,
                           impl.host_frames.size() * 4, hipMemcpyDeviceToHost,
                           s));
      Check(hipStreamSynchronize(s));
      const std::size_t length = std::size_t(rows - 1) * hop + N;
      std::vector<float> audio(length), envelope(length);
      for (int r = 0; r < rows; ++r)
        for (int n = 0; n < N; ++n) {
          audio[std::size_t(r) * hop + n] +=
              impl.host_frames[std::size_t(r) * N + n];
          envelope[std::size_t(r) * hop + n] += window[n] * window[n];
        }
      emit(audio, envelope, last);
    }
    Check(hipGetLastError());
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
}  // namespace gufo::models::lfm3_audio
