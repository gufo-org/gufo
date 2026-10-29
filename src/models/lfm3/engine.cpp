#include "src/models/lfm3/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
namespace gufo::models::lfm3 {
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
  void* data{};
  Buffer() = default;
  explicit Buffer(std::size_t bytes) {
    if (bytes)
      Check(hipMalloc(&data, bytes));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(Buffer&& b) noexcept : data(std::exchange(b.data, nullptr)) {}
  Buffer& operator=(Buffer&& b) noexcept {
    std::swap(data, b.data);
    return *this;
  }
  Buffer(const Buffer&) = delete;
  float* floats() const { return static_cast<float*>(data); }
};
struct Stream {
  hipStream_t value{};
  Stream() { Check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() {
    if (value)
      (void)hipStreamDestroy(value);
  }
};
template<class F>
void Visit(Weights& w, F f) {
  f(w.embedding);
  f(w.output);
  f(w.output_norm);
  f(w.rope_freqs);
  for (auto& l : w.layers)
    for (auto* t :
         {&l.norm, &l.ffn_norm, &l.post_norm, &l.post_ffn_norm, &l.gate, &l.up,
          &l.down, &l.conv, &l.in_proj, &l.out_proj, &l.q, &l.k, &l.v,
          &l.q_norm, &l.k_norm, &l.attn_gate, &l.attn_out})
      if (t->data)
        f(*t);
}
}  // namespace
struct Model::Impl {
  std::unique_ptr<core::GgufReader> reader;
  std::unique_ptr<Tokenizer> tokenizer;
  Weights weights;
  std::vector<Buffer> allocations;
};
Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
std::shared_ptr<Model> Model::Load(const std::string& path,
                                   std::string* error) {
  try {
    auto m = std::shared_ptr<Model>(new Model);
    m->impl_->reader = core::GgufReader::OpenFile(path, error);
    if (!m->impl_->reader)
      return nullptr;
    auto w = Weights::Bind(*m->impl_->reader, error);
    if (!w)
      return nullptr;
    m->impl_->tokenizer = Tokenizer::Load(*m->impl_->reader, error);
    if (!m->impl_->tokenizer)
      return nullptr;
    m->impl_->weights = *w;
    // Tied embeddings share one allocation. Copy packed weights verbatim so
    // quantization and mapped-file lifetime never depend on a session.
    std::unordered_map<const void*, void*> uploaded;
    Visit(m->impl_->weights, [&](Tensor& t) {
      if (!uploaded.contains(t.data)) {
        m->impl_->allocations.emplace_back(t.bytes);
        auto* p = m->impl_->allocations.back().data;
        Check(hipMemcpy(p, t.data, t.bytes, hipMemcpyHostToDevice));
        uploaded[t.data] = p;
      }
      t.data = uploaded.at(t.data);
    });
    return m;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
const Config& Model::config() const {
  return impl_->weights.config;
}
const Tokenizer& Model::tokenizer() const {
  return *impl_->tokenizer;
}
struct Session::Impl {
  static constexpr std::size_t kChunk = Session::kPrefillChunk;
  std::shared_ptr<Model> model;
  Stream stream;
  std::uint32_t context, position{};
  bool poisoned{}, logits_ready{};
  Buffer ids, x, z, out, p, q, k, v, gate, a, ff_gate, ff_up, logits, frontier;
  struct State {
    Buffer conv, k, v;
    std::uint32_t capacity{};
  };
  std::vector<State> states;
  Buffer quantized, row_max, attention_scratch, frequencies;
  Impl(std::shared_ptr<Model> m, std::uint32_t ctx)
      : model(std::move(m)),
        context(ctx),
        ids(kChunk * sizeof(std::uint32_t)),
        x(kChunk * model->config().hidden * 4),
        z(kChunk * model->config().hidden * 4),
        out(kChunk * model->config().hidden * 4),
        p(kChunk * 3 * model->config().hidden * 4),
        q(kChunk * model->config().heads * model->config().head_dim * 4),
        k(kChunk * model->config().heads * model->config().head_dim * 4),
        v(kChunk * model->config().heads * model->config().head_dim * 4),
        gate(kChunk * model->config().heads * 4),
        a(kChunk * model->config().heads * model->config().head_dim * 4),
        ff_gate(kChunk * model->config().intermediate * 4),
        ff_up(kChunk * model->config().intermediate * 4),
        logits(kMaxBatch * model->config().vocab * 4),
        frontier(model->config().hidden * 4) {
    const auto& c = model->config();
    const auto width =
        std::max({c.hidden * 3, c.intermediate, c.heads * c.head_dim});
    quantized = Buffer(gpu::DenseWorkspaceBytes(kChunk, width));
    row_max = Buffer(kChunk * sizeof(unsigned));
    attention_scratch =
        Buffer(gpu::AttentionScratchBytes(c.heads, c.head_dim, ctx, kChunk));
    // Global then sliding-window RoPE frequencies.
    frequencies = Buffer(std::size_t(c.head_dim) * 4);
    gpu::RopeFrequencies(frequencies.floats(), c.head_dim, c.theta,
                         stream.value);
    gpu::RopeFrequencies(frequencies.floats() + c.head_dim / 2, c.head_dim,
                         c.theta_swa, stream.value);
    Check(hipStreamSynchronize(stream.value));

    states.reserve(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      State s;
      if (c.layers[i] == LayerType::kConvolution)
        s.conv = Buffer(std::size_t(c.hidden) * (c.conv_width - 1) * 4);
      else {
        s.capacity = c.layers[i] == LayerType::kSlidingAttention
                         ? std::min(ctx, c.window)
                         : ctx;
        s.k = Buffer(std::size_t(s.capacity) * c.kv_heads[i] * c.head_dim * 4);
        s.v = Buffer(std::size_t(s.capacity) * c.kv_heads[i] * c.head_dim * 4);
      }
      states.push_back(std::move(s));
    }
  }
  void Run(std::span<const std::uint32_t> tokens, bool output_logits,
           std::span<Impl* const> batch = {}) {
    const auto& w = model->impl_->weights;
    const auto& c = w.config;
    const auto s = stream.value;
    const auto n = tokens.size();
    Check(hipMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(),
                         hipMemcpyHostToDevice, s));
    gpu::Embed(w.embedding, static_cast<const std::uint32_t*>(ids.data),
               x.floats(), n, s);
    auto dense = [&](const Tensor& weight, const float* input, float* output,
                     int count, hipStream_t stream) {
      gpu::Dense(weight, input, output, count, stream, {quantized.data});
    };
    auto norm = [&](const float* input, const Tensor& weight, float* output,
                    std::size_t count) {
      gufo::hip::LaunchBatchedRMSNorm(
          input, static_cast<const float*>(weight.data), output, nullptr, count,
          weight.cols, c.eps, s);
    };
    const gpu::DenseWorkspace workspace{quantized.data,
                                        static_cast<unsigned*>(row_max.data)};
    // x += post(branch), then z = norm(x), split for the next projections
    // when they all read split prompt inputs. Returns whether z was split.
    auto residual_norm = [&](const float* branch, const Tensor* post,
                             const Tensor* next,
                             std::span<const gpu::Output> consumers) {
      const bool split = next && gpu::SharesSplit(consumers, n, workspace);
      if (gpu::ResidualNorm(branch, post && post->data ? post : nullptr,
                            x.floats(), next, z.floats(), n, c.hidden, c.eps,
                            split, s, workspace))
        return split;
      if (branch) {
        if (post && post->data)
          norm(branch, *post, out.floats(), n);
        gufo::hip::LaunchBatchedResidualAdd(
            x.floats(), post && post->data ? out.floats() : branch, x.floats(),
            n, c.hidden, s);
      }
      if (next)
        norm(x.floats(), *next, z.floats(), n);
      return false;
    };
    // Projections reading the normalized input of layer i's mixer.
    std::array<gpu::Output, 4> mixer;
    std::span<const gpu::Output> inputs;
    const auto mixer_inputs = [&](std::size_t i) {
      const auto& l = w.layers[i];
      if (c.layers[i] == LayerType::kConvolution) {
        mixer[0] = {&l.in_proj, p.floats()};
        return std::span<const gpu::Output>(mixer.data(), 1);
      }
      mixer = {{{&l.q, q.floats()},
                {&l.k, k.floats()},
                {&l.v, v.floats()},
                {&l.attn_gate, gate.floats()}}};
      return std::span<const gpu::Output>(mixer);
    };
    inputs = mixer_inputs(0);
    bool split = residual_norm(nullptr, nullptr, &w.layers[0].norm, inputs);
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const auto& l = w.layers[i];
      if (c.layers[i] == LayerType::kConvolution) {
        gpu::Project(inputs, z.floats(), n, s, workspace, split);
        if (batch.empty()) {
          gpu::ShortConv(p.floats(), static_cast<const float*>(l.conv.data),
                         states[i].conv.floats(), a.floats(), n, c.hidden,
                         c.conv_width, s);
        } else
          for (std::size_t row = 0; row < batch.size(); ++row) {
            gpu::ShortConv(p.floats() + row * 3 * c.hidden,
                           static_cast<const float*>(l.conv.data),
                           batch[row]->states[i].conv.floats(),
                           a.floats() + row * c.hidden, 1, c.hidden,
                           c.conv_width, s);
          }
        dense(l.out_proj, a.floats(), out.floats(), n, s);
      } else {
        gpu::Project(inputs, z.floats(), n, s, workspace, split);
        gufo::hip::LaunchBatchedPerHeadRMSNorm(
            q.floats(), static_cast<const float*>(l.q_norm.data), q.floats(), n,
            c.heads, c.head_dim, c.eps, s);
        gufo::hip::LaunchBatchedPerHeadRMSNorm(
            k.floats(), static_cast<const float*>(l.k_norm.data), k.floats(), n,
            c.kv_heads[i], c.head_dim, c.eps, s);
        const bool swa = c.layers[i] == LayerType::kSlidingAttention;
        const auto* factors =
            swa ? nullptr : static_cast<const float*>(w.rope_freqs.data);
        const auto attend = [&](Impl& owner, std::size_t row,
                                std::size_t count) {
          auto& state = owner.states[i];
          auto* qr = q.floats() + row * c.heads * c.head_dim;
          auto* kr = k.floats() + row * c.kv_heads[i] * c.head_dim;
          auto* vr = v.floats() + row * c.kv_heads[i] * c.head_dim;
          const float* table =
              frequencies.floats() + (swa ? c.head_dim / 2 : 0);
          gpu::Rope(qr, table, factors, count, c.heads, c.head_dim,
                    owner.position, s);
          gpu::Rope(kr, table, factors, count, c.kv_heads[i], c.head_dim,
                    owner.position, s);
          gpu::Attention(qr, kr, vr, gate.floats() + row * c.heads,
                         state.k.floats(), state.v.floats(),
                         a.floats() + row * c.heads * c.head_dim, count,
                         c.heads, c.kv_heads[i], c.head_dim, owner.position,
                         state.capacity, swa ? c.window : 0, s,
                         owner.attention_scratch.floats());
          gpu::StoreKV(kr, vr, state.k.floats(), state.v.floats(), count,
                       c.kv_heads[i] * c.head_dim, owner.position,
                       state.capacity, s);
        };
        if (batch.empty())
          attend(*this, 0, n);
        else
          for (std::size_t row = 0; row < batch.size(); ++row)
            attend(*batch[row], row, 1);
        dense(l.attn_out, a.floats(), out.floats(), n, s);
      }
      const gpu::Output ffn[] = {{&l.gate, ff_gate.floats()},
                                 {&l.up, ff_up.floats()}};
      split = residual_norm(out.floats(), &l.post_norm, &l.ffn_norm, ffn);
      gpu::ProjectFfn(l.gate, l.up, l.down, z.floats(), ff_gate.floats(),
                      ff_up.floats(), out.floats(), n, s, workspace, split);
      const bool last = i + 1 == c.layers.size();
      if (!last)
        inputs = mixer_inputs(i + 1);
      split = residual_norm(out.floats(), &l.post_ffn_norm,
                            last ? nullptr : &w.layers[i + 1].norm, inputs);
    }
    if (batch.empty()) {
      norm(x.floats() + (n - 1) * c.hidden, w.output_norm, frontier.floats(),
           1);
      logits_ready = output_logits;
      if (output_logits)
        dense(w.output, frontier.floats(), logits.floats(), 1, s);
    } else {
      norm(x.floats(), w.output_norm, z.floats(), n);
      dense(w.output, z.floats(), logits.floats(), n, s);
      for (std::size_t row = 0; row < batch.size(); ++row) {
        Check(hipMemcpyAsync(
            batch[row]->frontier.data, z.floats() + row * c.hidden,
            std::size_t(c.hidden) * 4, hipMemcpyDeviceToDevice, s));
        if (row)
          Check(hipMemcpyAsync(
              batch[row]->logits.data, logits.floats() + row * c.vocab,
              std::size_t(c.vocab) * 4, hipMemcpyDeviceToDevice, s));
      }
    }
    Check(hipGetLastError());
    Check(hipStreamSynchronize(s));
    if (batch.empty())
      position += n;
    else
      for (auto* row : batch) {
        ++row->position;
        row->logits_ready = true;
      }
  }
};
Session::Session(std::shared_ptr<Model> m, std::uint32_t context)
    : impl_(std::make_unique<Impl>(std::move(m), context)) {}
Session::~Session() {
  if (impl_)
    (void)hipStreamSynchronize(impl_->stream.value);
}
std::unique_ptr<Session> Model::CreateSession(std::uint32_t context,
                                              std::string* error) {
  try {
    if (!context || context > config().context)
      throw std::invalid_argument("invalid LFM3 context");
    auto session =
        std::unique_ptr<Session>(new Session(shared_from_this(), context));
    if (!session->Reset(error))
      return nullptr;
    return session;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
bool Session::Forward(std::span<const std::uint32_t> tokens,
                      std::span<float> logits, std::string* error) {
  try {
    const auto& c = impl_->model->config();
    if (impl_->poisoned)
      throw std::runtime_error(
          "LFM3 session requires Reset after a device error");
    if (tokens.empty() || tokens.size() > impl_->context - impl_->position ||
        (!logits.empty() && logits.size() != c.vocab))
      throw std::invalid_argument(
          "invalid LFM3 token/logit count or context exhausted");
    for (auto t : tokens)
      if (t >= c.vocab)
        throw std::invalid_argument("LFM3 token out of range");
    for (std::size_t offset = 0; offset < tokens.size();
         offset += Impl::kChunk) {
      auto count = std::min(Impl::kChunk, tokens.size() - offset);
      try {
        impl_->Run(tokens.subspan(offset, count),
                   !logits.empty() && offset + count == tokens.size());
      } catch (...) {
        impl_->poisoned = true;
        throw;
      }
    }
    if (!logits.empty())
      Check(hipMemcpy(logits.data(), impl_->logits.data, logits.size_bytes(),
                      hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::ForwardBatch(std::span<Session* const> sessions,
                           std::span<const std::uint32_t> tokens,
                           std::span<std::span<float>> outputs,
                           std::string* error) {
  try {
    if (sessions.empty() || sessions.size() > kMaxBatch ||
        tokens.size() != sessions.size() || outputs.size() != sessions.size())
      throw std::invalid_argument("invalid LFM3 decode batch shape");
    std::vector<Impl*> batch;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      auto* session = sessions[i];
      if (!session || std::find(sessions.begin(), sessions.begin() + i,
                                session) != sessions.begin() + i)
        throw std::invalid_argument("duplicate or null LFM3 batch session");
      auto* state = session->impl_.get();
      if (state->poisoned || state->position >= state->context ||
          tokens[i] >= state->model->config().vocab ||
          outputs[i].size() != state->model->config().vocab ||
          (!batch.empty() && state->model != batch.front()->model))
        throw std::invalid_argument("incompatible LFM3 batch session");
      batch.push_back(state);
    }
    try {
      batch.front()->Run(tokens, true, batch);
      for (std::size_t i = 0; i < batch.size(); ++i)
        Check(hipMemcpy(outputs[i].data(), batch[i]->logits.data,
                        outputs[i].size_bytes(), hipMemcpyDeviceToHost));
    } catch (...) {
      for (auto* state : batch)
        state->poisoned = true;
      throw;
    }
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::Reset(std::string* error) {
  try {
    Check(hipStreamSynchronize(impl_->stream.value));
    const auto& c = impl_->model->config();
    for (std::size_t i = 0; i < impl_->states.size(); ++i)
      if (impl_->states[i].conv.data)
        Check(hipMemsetAsync(impl_->states[i].conv.data, 0,
                             std::size_t(c.hidden) * (c.conv_width - 1) * 4,
                             impl_->stream.value));
    Check(hipStreamSynchronize(impl_->stream.value));
    impl_->position = 0;
    impl_->logits_ready = false;
    impl_->poisoned = false;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::uint32_t Session::Position() const {
  return impl_->position;
}
namespace {
constexpr std::uint32_t kSnapshotMagic = 0x334d464c, kSnapshotVersion = 1;
constexpr std::size_t kSnapshotHeaderBytes = 32;
void WriteWord(std::span<std::uint8_t> data, std::size_t word,
               std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    data[word * 4 + i] = value >> (i * 8);
}
std::uint32_t ReadWord(std::span<const std::uint8_t> data, std::size_t word) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i)
    value |= std::uint32_t(data[word * 4 + i]) << (i * 8);
  return value;
}
// Counts only executed rows. Sliding rings are copied in physical order, so
// restoring after a wrap preserves the absolute-position modulo mapping.
std::uint32_t ConfigSignature(const Config& c) {
  std::uint32_t hash = 2166136261U;
  const auto mix = [&](std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
      hash = (hash ^ ((value >> (8 * i)) & 255)) * 16777619U;
  };
  for (auto value :
       {c.hidden, c.intermediate, c.vocab, c.heads, c.head_dim, c.window,
        c.conv_width, std::bit_cast<std::uint32_t>(c.eps),
        std::bit_cast<std::uint32_t>(c.theta),
        std::bit_cast<std::uint32_t>(c.theta_swa)})
    mix(value);
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    mix(static_cast<std::uint32_t>(c.layers[i]));
    mix(c.kv_heads[i]);
  }
  return hash;
}
std::size_t StateBytes(const Config& c, std::uint32_t position) {
  std::size_t bytes =
      kSnapshotHeaderBytes + (std::size_t(c.hidden) + c.vocab) * 4;
  for (std::size_t i = 0; i < c.layers.size(); ++i)
    bytes += c.layers[i] == LayerType::kConvolution
                 ? std::size_t(c.hidden) * (c.conv_width - 1) * 4
                 : std::size_t(c.layers[i] == LayerType::kSlidingAttention
                                   ? std::min(position, c.window)
                                   : position) *
                       c.kv_heads[i] * c.head_dim * 8;
  return bytes;
}
}  // namespace
std::size_t Session::SnapshotBytes() const {
  return StateBytes(impl_->model->config(), impl_->position);
}
bool Session::ReadLogits(std::span<float> output, std::string* error) const {
  try {
    const auto& c = impl_->model->config();
    if (impl_->poisoned || !impl_->position || output.size() != c.vocab)
      throw std::invalid_argument("invalid LFM3 logit frontier");
    if (!impl_->logits_ready) {
      gpu::Dense(impl_->model->impl_->weights.output, impl_->frontier.floats(),
                 impl_->logits.floats(), 1, impl_->stream.value);
      Check(hipGetLastError());
      Check(hipStreamSynchronize(impl_->stream.value));
      impl_->logits_ready = true;
    }
    Check(hipMemcpy(output.data(), impl_->logits.data, output.size_bytes(),
                    hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::vector<std::uint8_t> Session::SaveSnapshot() const {
  if (impl_->poisoned)
    throw std::runtime_error("cannot snapshot a poisoned LFM3 session");
  const auto& c = impl_->model->config();
  std::vector<std::uint8_t> bytes(SnapshotBytes());
  const std::uint32_t header[]{kSnapshotMagic,
                               kSnapshotVersion,
                               impl_->context,
                               impl_->position,
                               c.hidden,
                               c.vocab,
                               std::uint32_t(c.layers.size()),
                               ConfigSignature(c)};
  for (std::size_t i = 0; i < 8; ++i)
    WriteWord(bytes, i, header[i]);
  auto offset = kSnapshotHeaderBytes;
  auto copy = [&](const void* source, std::size_t size) {
    if (size)
      Check(hipMemcpy(bytes.data() + offset, source, size,
                      hipMemcpyDeviceToHost));
    offset += size;
  };
  if (impl_->position) {
    std::vector<float> output(c.vocab);
    std::string error;
    if (!ReadLogits(output, &error))
      throw std::runtime_error(error);
    copy(impl_->frontier.data, std::size_t(c.hidden) * 4);
    std::memcpy(bytes.data() + offset, output.data(), output.size() * 4);
    offset += output.size() * 4;
  } else
    offset += (std::size_t(c.hidden) + c.vocab) * 4;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    const auto& state = impl_->states[i];
    if (state.conv.data)
      copy(state.conv.data, std::size_t(c.hidden) * (c.conv_width - 1) * 4);
    else {
      auto size = std::size_t(std::min(impl_->position, state.capacity)) *
                  c.kv_heads[i] * c.head_dim * 4;
      copy(state.k.data, size);
      copy(state.v.data, size);
    }
  }
  return bytes;
}
bool Session::RestoreSnapshot(std::span<const std::uint8_t> bytes,
                              std::string* error) {
  try {
    const auto& c = impl_->model->config();
    // Validate the complete header and size before touching the existing state.
    if (bytes.size() < kSnapshotHeaderBytes ||
        ReadWord(bytes, 0) != kSnapshotMagic ||
        ReadWord(bytes, 1) != kSnapshotVersion ||
        ReadWord(bytes, 2) != impl_->context ||
        ReadWord(bytes, 3) > impl_->context || ReadWord(bytes, 4) != c.hidden ||
        ReadWord(bytes, 5) != c.vocab ||
        ReadWord(bytes, 6) != c.layers.size() ||
        ReadWord(bytes, 7) != ConfigSignature(c) ||
        bytes.size() != StateBytes(c, ReadWord(bytes, 3)))
      throw std::invalid_argument("invalid LFM3 snapshot payload");
    const auto position = ReadWord(bytes, 3);
    auto offset = kSnapshotHeaderBytes;
    auto copy = [&](void* destination, std::size_t size) {
      if (size)
        Check(hipMemcpyAsync(destination, bytes.data() + offset, size,
                             hipMemcpyHostToDevice, impl_->stream.value));
      offset += size;
    };
    try {
      Check(hipStreamSynchronize(impl_->stream.value));
      copy(impl_->frontier.data, std::size_t(c.hidden) * 4);
      copy(impl_->logits.data, std::size_t(c.vocab) * 4);
      for (std::size_t i = 0; i < c.layers.size(); ++i) {
        auto& state = impl_->states[i];
        if (state.conv.data)
          copy(state.conv.data, std::size_t(c.hidden) * (c.conv_width - 1) * 4);
        else {
          auto size = std::size_t(std::min(position, state.capacity)) *
                      c.kv_heads[i] * c.head_dim * 4;
          copy(state.k.data, size);
          copy(state.v.data, size);
        }
      }
      Check(hipStreamSynchronize(impl_->stream.value));
    } catch (...) {
      impl_->poisoned = true;
      throw;
    }
    impl_->position = position;
    impl_->logits_ready = position != 0;
    impl_->poisoned = false;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
}  // namespace gufo::models::lfm3
