#include "src/models/embedding_gemma2/npu/encoder_npu.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <thread>
#include <utility>

#include "gufo/ext/experimental/npu.hpp"
#include "src/models/embedding_gemma2/config.hpp"
#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/tensor.hpp"

namespace gufo::models::embedding_gemma2::npu {
namespace host = gufo::npu::host;

namespace {

// Sequences at least this long run attention products on the NPU.
constexpr std::size_t kNpuAttentionRows = 128;
constexpr std::size_t kMaxHeads = 16;
// Rows per NPU band (BfpGemm), the unit of attention spans and row parts:
// 512, or 128 for sequences short enough for BfpGemm's 128-row program.
constexpr std::size_t kBandRows = 512, kSmallBandRows = 128;
// A single sequence up to this long splits into two parts of at most 128
// rows, each within one band of the 128-row program.
constexpr std::size_t kSmallSplitRows = 256;
// Concurrent parts of a batch (BfpGemm has as many staging lanes).
constexpr std::size_t kParts = 4;
constexpr std::size_t kGateBlock = 64;  // gate rows per interleaved tile

struct Layer {
  int qkv, o, gate_up, down, ple_gate, ple_proj;
  std::vector<float> input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm,
      q_norm, k_norm, ple_norm;
  float scalar;
};

std::vector<float> ToFloat(const Tensor& t) {
  const std::size_t n = std::size_t(*t.NumElements());
  std::vector<float> out(n);
  host::Bf16ToFloat(reinterpret_cast<const std::uint16_t*>(t.data), out.data(),
                    n);
  return out;
}

}  // namespace

struct Encoder::Impl {
  gufo::npu::Device device;
  gufo::npu::BfpGemm gemm;
  // One per concurrent half batch (attention keys and values).
  gufo::npu::BfpGemm::Scratch scratches[kParts];
  LoadResult loaded;
  const TextConfig* config{};
  const std::uint16_t* embed{};
  int ple_model, projection;
  std::vector<float> ple_norm, final_norm;
  std::vector<Layer> layers;
  std::mutex mutex;
  bool profile{std::getenv("GUFO_NPU_PROFILE") != nullptr};
  ModelConfig text_only;

  const Tensor& Find(const std::string& name) const {
    const Tensor* t = loaded.store->Find(std::string(kTextTensorPrefix) + name);
    if (!t || t->dtype != DType::kBF16)
      throw std::runtime_error("missing BF16 tensor " + name);
    return *t;
  }
  std::vector<float> Vector(const std::string& name) const {
    return ToFloat(Find(name));
  }
  // Uploads the row-wise concatenation of BF16 matrices [n_i][k].
  int Upload(std::initializer_list<std::string> names) {
    std::vector<std::uint16_t> rows;
    int n = 0, k = 0;
    for (const std::string& name : names) {
      const Tensor& t = Find(name);
      n += int(t.shape[0]);
      k = int(t.shape[1]);
      const auto* p = reinterpret_cast<const std::uint16_t*>(t.data);
      rows.insert(rows.end(), p, p + t.shape[0] * t.shape[1]);
    }
    return gemm.UploadBf16(n, k, rows);
  }

  // [gate; up] with each 128-output tile holding 64 gate rows and the 64 up
  // rows of the same channels, so one output tile yields 64 activations.
  int UploadInterleaved(const std::string& gate, const std::string& up) {
    const Tensor& g = Find(gate);
    const Tensor& u = Find(up);
    const std::size_t n = g.shape[0], k = g.shape[1];
    const auto* gp = reinterpret_cast<const std::uint16_t*>(g.data);
    const auto* up_p = reinterpret_cast<const std::uint16_t*>(u.data);
    std::vector<std::uint16_t> rows(2 * n * k);
    for (std::size_t b = 0; b < n / kGateBlock; ++b) {
      std::memcpy(&rows[(2 * b) * kGateBlock * k], gp + b * kGateBlock * k,
                  kGateBlock * k * 2);
      std::memcpy(&rows[(2 * b + 1) * kGateBlock * k],
                  up_p + b * kGateBlock * k, kGateBlock * k * 2);
    }
    return gemm.UploadBf16(int(2 * n), int(k), rows);
  }

  void Load(const std::string& model_dir) {
    loaded = LoadModelDirectory(model_dir, false, false);
    if (!loaded.ok)
      throw std::runtime_error(loaded.error);
    config = &loaded.config.text;
    text_only = loaded.config;
    text_only.supported_vision = text_only.supported_video =
        text_only.supported_audio = false;
    device = gufo::npu::Device::Open();
    gemm = gufo::npu::BfpGemm::Open(device, gufo::npu::FindDesign("bfpgemm"),
                                    std::size_t(1400) << 20);
    for (auto& scratch : scratches)
      scratch = gemm.NewScratch(std::size_t(288) << 20);
    embed = reinterpret_cast<const std::uint16_t*>(
        Find("embed_tokens.weight").data);
    ple_model = Upload({"ple.per_layer_model_projection.weight"});
    projection = Upload({"embedding_projection.weight"});
    ple_norm = Vector("ple.per_layer_projection_norm.weight");
    final_norm = Vector("norm.weight");
    for (std::uint32_t l = 0; l < config->num_hidden_layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      Layer layer;
      layer.qkv =
          Upload({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                  p + "self_attn.v_proj.weight"});
      layer.o = Upload({p + "self_attn.o_proj.weight"});
      layer.gate_up = UploadInterleaved(p + "mlp.gate_proj.weight",
                                        p + "mlp.up_proj.weight");
      layer.down = Upload({p + "mlp.down_proj.weight"});
      layer.ple_gate = Upload({p + "ple_block.per_layer_input_gate.weight"});
      layer.ple_proj = Upload({p + "ple_block.per_layer_projection.weight"});
      layer.input_norm = Vector(p + "input_layernorm.weight");
      layer.post_attn_norm = Vector(p + "post_attention_layernorm.weight");
      layer.pre_ffn_norm = Vector(p + "pre_feedforward_layernorm.weight");
      layer.post_ffn_norm = Vector(p + "post_feedforward_layernorm.weight");
      layer.q_norm = Vector(p + "self_attn.q_norm.weight");
      layer.k_norm = Vector(p + "self_attn.k_norm.weight");
      layer.ple_norm = Vector(p + "ple_block.post_per_layer_input_norm.weight");
      layer.scalar = Vector(p + "layer_scalar")[0];
      layers.push_back(std::move(layer));
    }
  }

  // Attention of the long sequences on the NPU: per sequence, query band of
  // up to 512 rows and head, scores = Q K^T over the band's key span and
  // out = P V, with the span's keys and values uploaded per request; the
  // softmax runs on the CPU. `visible(r, lo, hi, &b, &e)` narrows row r's
  // keys within [lo, hi).
  template<typename Visible>
  void NpuAttention(gufo::npu::BfpGemm::Scratch& scratch, const float* qkv,
                    std::size_t row_w, std::size_t heads, std::size_t kv,
                    std::size_t hd, const std::vector<std::size_t>& starts,
                    const std::vector<char>& long_seq, std::size_t row0,
                    std::size_t row1, std::size_t span_rows,
                    const Visible& visible, std::uint16_t* context16,
                    std::size_t qw) {
    struct Span {
      std::size_t seq, q0, q1, k0, k1;         // absolute rows
      int keys[kMaxHeads], values[kMaxHeads];  // per query head
    };
    struct Item {
      const Span* span;
      std::size_t head;
      std::vector<std::uint16_t> q, scores;  // bf16
    };
    std::vector<Span> spans;
    for (std::size_t s = 0; s + 1 < starts.size(); ++s) {
      if (!long_seq[s])
        continue;
      for (std::size_t q0 = starts[s]; q0 < starts[s + 1]; q0 += span_rows) {
        if (q0 < row0 || q0 >= row1)  // another part's queries
          continue;
        const std::size_t q1 = std::min(starts[s + 1], q0 + span_rows);
        std::size_t k0 = starts[s + 1], k1 = starts[s];
        for (std::size_t r : {q0, q1 - 1}) {
          std::size_t b, e;
          visible(r, starts[s], starts[s + 1], &b, &e);
          k0 = std::min(k0, b);
          k1 = std::max(k1, e);
        }
        spans.push_back({s, q0, q1, k0, k1, {}, {}});
      }
    }
    if (spans.empty())
      return;
    if (heads > kMaxHeads)
      throw std::runtime_error("NPU attention: too many heads");
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a) {
      return std::chrono::duration<double, std::milli>(Clock::now() - a)
          .count();
    };
    auto t = Clock::now();
    double t_upload, t_qk, t_soft;
    scratch.Reset();
    // Keys, then values, per span and query head in item order, so that the
    // products of one layer batch into few NPU jobs (BfpGemm batches calls
    // whose matrices lie at a constant stride). Spans longer than one output
    // group of keys (full attention over long sequences) share one upload
    // per KV head instead.
    const std::size_t group_heads = heads / kv;
    // Specs in upload order; slot[i] is where spec i's id goes. Long spans
    // with the same keys reuse the first one's ids (`alias`).
    using Spec = gufo::npu::BfpGemm::Scratch::Spec;
    std::vector<Spec> specs;
    std::vector<int*> slot;
    std::vector<std::pair<int*, int*>> alias;  // (target, source) after upload
    std::map<std::pair<std::size_t, std::size_t>, Span*> shared[2];
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> first_spec[2];
    for (int pass = 0; pass < 2; ++pass)
      for (Span& sp : spans) {
        const std::size_t n = sp.k1 - sp.k0;
        const float* base = qkv + sp.k0 * row_w + qw;
        int* ids = pass == 0 ? sp.keys : sp.values;
        const bool long_span = n > 1024;
        if (long_span) {
          if (auto it = shared[pass].find({sp.k0, sp.k1});
              it != shared[pass].end()) {
            int* src = pass == 0 ? it->second->keys : it->second->values;
            for (std::size_t h = 0; h < heads; ++h)
              alias.push_back({&ids[h], &src[h]});
            continue;
          }
          shared[pass][{sp.k0, sp.k1}] = &sp;
        }
        // A short span with the keys of an earlier one copies its encoded
        // matrices (still one matrix per call, evenly spaced).
        if (!long_span) {
          auto [it, fresh] =
              first_spec[pass].try_emplace({sp.k0, sp.k1}, specs.size());
          if (!fresh) {
            for (std::size_t h = 0; h < heads; ++h) {
              Spec copy = specs[it->second + h];
              copy.same_as = int(it->second + h);
              specs.push_back(copy);
              slot.push_back(&ids[h]);
            }
            continue;
          }
        }
        for (std::size_t h = 0; h < heads; ++h) {
          const std::size_t kh = h / group_heads;
          if (h % group_heads != 0) {
            // A query head sharing its KV head: the same matrix, copied so
            // the calls stay evenly spaced (long spans reuse it).
            if (long_span) {
              alias.push_back({&ids[h], &ids[h - 1]});
            } else {
              Spec copy = specs.back();
              copy.same_as = int(specs.size()) - 1;
              specs.push_back(copy);
              slot.push_back(&ids[h]);
            }
            continue;
          }
          if (pass == 0)
            specs.push_back({int(n), int(hd), base + kh * hd, row_w, 1});
          else
            specs.push_back(
                {int(hd), int(n), base + kv * hd + kh * hd, 1, row_w});
          slot.push_back(&ids[h]);
        }
      }
    const std::vector<int> uploaded = scratch.UploadMany(specs);
    for (std::size_t i = 0; i < uploaded.size(); ++i)
      *slot[i] = uploaded[i];
    for (const auto& [target, source] : alias)
      *target = *source;
    t_upload = ms(t);
    t = Clock::now();
    std::vector<Item> items;
    for (const Span& sp : spans)
      for (std::size_t h = 0; h < heads; ++h)
        items.push_back({&sp, h, {}, {}});
    // Queries and scores stay bf16 (the NPU rounds them to bf16 anyway).
    host::ParallelFor(items.size(), 1, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        Item& it = items[i];
        const std::size_t rows = it.span->q1 - it.span->q0;
        it.q.resize(rows * hd);
        for (std::size_t r = 0; r < rows; ++r)
          host::FloatToBf16(qkv + (it.span->q0 + r) * row_w + it.head * hd,
                            it.q.data() + r * hd, hd);
        it.scores.resize(rows * (it.span->k1 - it.span->k0));
      }
    });
    std::vector<gufo::npu::BfpGemm::Epilogue> epilogues;
    epilogues.reserve(items.size());
    for (Item& it : items) {
      std::uint16_t* scores = it.scores.data();
      const std::size_t n = it.span->k1 - it.span->k0;
      epilogues.push_back([scores, n](int r0, int n_rows, int n0, int w,
                                      const std::uint16_t* tile) {
        for (int i = 0; i < n_rows; ++i)
          std::memcpy(scores + std::size_t(r0 + i) * n + std::size_t(n0),
                      tile + std::size_t(i) * 128, std::size_t(w) * 2);
      });
    }
    std::vector<gufo::npu::BfpGemm::Call> calls;
    for (std::size_t i = 0; i < items.size(); ++i)
      calls.push_back({items[i].span->keys[items[i].head], nullptr,
                       int(items[i].span->q1 - items[i].span->q0), nullptr,
                       items[i].q.data(), &epilogues[i]});
    gemm.Run(calls);
    t_qk = ms(t);
    t = Clock::now();
    host::ParallelFor(items.size() * 16, 1, [&](std::size_t b, std::size_t e) {
      std::vector<float> row;
      for (std::size_t w = b; w < e; ++w) {
        Item& it = items[w / 16];
        const Span& sp = *it.span;
        const std::size_t rows = sp.q1 - sp.q0, n = sp.k1 - sp.k0;
        const std::size_t part = w % 16;
        row.resize(n);
        for (std::size_t r = rows * part / 16; r < rows * (part + 1) / 16;
             ++r) {
          std::size_t lo, hi;
          visible(sp.q0 + r, sp.k0, sp.k1, &lo, &hi);
          lo -= sp.k0;
          hi -= sp.k0;
          std::uint16_t* s16 = it.scores.data() + r * n;
          host::Bf16ToFloat(s16 + lo, row.data() + lo, hi - lo);
          float top = -INFINITY;
          for (std::size_t t = lo; t < hi; ++t)
            top = std::max(top, row[t]);
          std::fill(row.begin(), row.begin() + std::ptrdiff_t(lo), 0.0f);
          std::fill(row.begin() + std::ptrdiff_t(hi), row.end(), 0.0f);
          float sum = 0;
          host::ExpInPlace(row.data() + lo, hi - lo, top, &sum);
          const float inv = 1.0f / sum;
          for (std::size_t t = lo; t < hi; ++t)
            row[t] *= inv;
          host::FloatToBf16(row.data(), s16, n);
        }
      }
    });
    t_soft = ms(t);
    t = Clock::now();
    calls.clear();
    // PV tiles go straight to the bf16 context rows.
    std::vector<gufo::npu::BfpGemm::Epilogue> pv_epilogues;
    pv_epilogues.reserve(items.size());
    for (Item& it : items) {
      const std::size_t q0 = it.span->q0, col = it.head * hd;
      pv_epilogues.push_back(
          [context16, q0, col, qw](int r0, int n_rows, int n0, int w,
                                   const std::uint16_t* tile) {
            for (int i = 0; i < n_rows; ++i)
              std::memcpy(context16 + (q0 + std::size_t(r0 + i)) * qw + col +
                              std::size_t(n0),
                          tile + std::size_t(i) * 128, std::size_t(w) * 2);
          });
    }
    for (std::size_t i = 0; i < items.size(); ++i)
      calls.push_back({items[i].span->values[items[i].head], nullptr,
                       int(items[i].span->q1 - items[i].span->q0), nullptr,
                       items[i].scores.data(), &pv_epilogues[i]});
    gemm.Run(calls);
    if (profile)
      std::fprintf(stderr,
                   "attention: upload %.2f qk %.2f softmax %.2f pv %.2f ms\n",
                   t_upload, t_qk, t_soft, ms(t));
  }

  // Splits the sequences into two halves of similar token counts that run on
  // two threads, so one half's CPU work overlaps the other's NPU work.
  template<typename T>
  static void Grow(std::vector<T>& v, std::size_t n) {
    if (v.size() < n)
      v.resize(n);
  }

  // One request's state, shared by its parts: each part owns a range of
  // rows and only reads other rows' keys and values (qkv, double-buffered by
  // layer parity so a part can start the next layer while another still
  // attends to this one).
  struct Work {
    const std::uint32_t* tokens{};
    std::size_t span{};  // query rows per NPU attention span
    std::vector<std::size_t> starts;
    std::vector<std::int32_t> positions;
    std::vector<std::size_t> seq_of;
    std::vector<char> long_seq;
    std::vector<float> state, ple, qkv[2], context, tmp, projected;
    std::vector<std::uint16_t> normed16, context16, act16;
  };
  Work work;  // reused by every request (they run one at a time)

  // Splits the request into up to kParts parts of rows that run on their own
  // threads, each with a share of the CPU threads and its own NPU staging
  // lane, so the parts' CPU stages overlap each other's NPU work. Parts cut
  // at sequence boundaries when there are several sequences; a single long
  // sequence cuts at 512-row bands, and its parts meet once per layer, after
  // every part has its keys and values.
  std::vector<std::vector<float>> RunSplit(
      const std::vector<std::uint32_t>& tokens,
      const std::vector<std::size_t>& starts,
      const std::function<bool()>& cancelled) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    const TextConfig& c = *config;
    const std::size_t seqs = starts.size() - 1, rows = tokens.size();
    const std::size_t hidden = c.hidden_size,
                      per_layer = c.hidden_size_per_layer_input,
                      layers_n = c.num_hidden_layers,
                      heads = c.num_attention_heads,
                      inner = c.intermediate_size;
    Work& wk = work;  // kept across requests: no fresh pages to fault in
    wk.tokens = tokens.data();
    wk.span = kBandRows;
    wk.starts = starts;
    wk.positions.resize(rows);
    wk.seq_of.resize(rows);
    for (std::size_t q = 0; q < seqs; ++q)
      for (std::size_t r = starts[q]; r < starts[q + 1]; ++r) {
        wk.positions[r] = std::int32_t(r - starts[q]);
        wk.seq_of[r] = q;
      }
    wk.long_seq.assign(seqs, 0);
    for (std::size_t q = 0; q < seqs; ++q)
      wk.long_seq[q] = starts[q + 1] - starts[q] >= kNpuAttentionRows;
    std::size_t max_q = 0, max_kv = 0;
    for (const auto& g : c.layers) {
      max_q = std::max<std::size_t>(max_q, heads * g.head_dim);
      max_kv =
          std::max<std::size_t>(max_kv, g.num_key_value_heads * g.head_dim);
    }
    Grow(wk.state, rows * hidden);
    Grow(wk.ple, rows * layers_n * per_layer);
    for (auto& q : wk.qkv)
      Grow(q, rows * (max_q + 2 * max_kv));
    Grow(wk.context, rows * max_q);
    Grow(wk.tmp, rows * hidden);
    Grow(wk.projected, rows * c.embedding_dim);
    Grow(wk.normed16, rows * hidden);
    Grow(wk.context16, rows * max_q);
    Grow(wk.act16, rows * std::max(inner, per_layer));

    // Row cuts.
    std::vector<std::size_t> cuts{0};
    bool coupled = false;
    if (seqs >= 2 && rows >= 256) {
      const std::size_t parts = std::min<std::size_t>(
          {kParts, seqs, std::max<std::size_t>(1, rows / 256)});
      for (std::size_t p = 1; p < parts; ++p) {
        std::size_t q = 1;
        while (q < seqs && starts[q] * parts < rows * p)
          ++q;
        if (starts[q] > cuts.back() && starts[q] < rows)
          cuts.push_back(starts[q]);
      }
    } else if (seqs == 1 && (rows <= kSmallSplitRows || rows > 1024)) {
      // 257-1024 rows stay one part: more parts cost more dispatches than
      // they overlap.
      const std::size_t band =
          rows <= kSmallSplitRows ? kSmallBandRows : kBandRows;
      wk.span = band;
      const std::size_t bands = (rows + band - 1) / band;
      // Short sequences: at most two parts (each part's products are their
      // own dispatches, which cost more than they save on few rows).
      const std::size_t parts =
          std::min<std::size_t>(band == kSmallBandRows ? 2 : kParts, bands);
      for (std::size_t p = 1; p < parts; ++p)
        cuts.push_back(bands * p / parts * band);
      coupled = parts > 1;
    }
    cuts.push_back(rows);
    const std::size_t parts = cuts.size() - 1;
    std::barrier sync(std::ptrdiff_t(coupled ? parts : 1));
    std::vector<std::exception_ptr> failure(parts);
    std::atomic<bool> stop{false};
    auto part = [&](std::size_t p) {
      try {
        const gufo::npu::host::SharePool share(unsigned(p), unsigned(parts));
        const auto tp = Clock::now();
        RunPart(wk, cuts[p], cuts[p + 1], int(p), coupled ? &sync : nullptr,
                cancelled, stop);
        if (profile)
          std::fprintf(
              stderr, "embedding_gemma2 npu: part %zu rows %zu-%zu %.1f ms\n",
              p, cuts[p], cuts[p + 1],
              std::chrono::duration<double, std::milli>(Clock::now() - tp)
                  .count());
      } catch (...) {
        failure[p] = std::current_exception();
        stop = true;
        if (coupled)
          sync.arrive_and_drop();
      }
    };
    std::vector<std::thread> threads;
    for (std::size_t p = 1; p < parts; ++p)
      threads.emplace_back(part, p);
    part(0);
    for (auto& th : threads)
      th.join();
    for (auto& f : failure)
      if (f)
        std::rethrow_exception(f);
    if (stop)
      return {};
    const std::size_t dim = c.embedding_dim;
    std::vector<std::vector<float>> out(seqs);
    host::ParallelFor(seqs, 1, [&](std::size_t b, std::size_t e) {
      std::vector<double> sum(dim);
      for (std::size_t q = b; q < e; ++q) {
        std::fill(sum.begin(), sum.end(), 0.0);
        for (std::size_t r = starts[q]; r < starts[q + 1]; ++r)
          for (std::size_t i = 0; i < dim; ++i)
            sum[i] += wk.projected[r * dim + i];
        double norm = 0;
        for (double v : sum)
          norm += v * v;
        const double inv = norm > 0 ? 1.0 / std::sqrt(norm) : 0.0;
        out[q].resize(dim);
        for (std::size_t i = 0; i < dim; ++i)
          out[q][i] = float(sum[i] * inv);
      }
    });
    if (profile)
      std::fprintf(
          stderr,
          "embedding_gemma2 npu: %zu tokens in %zu parts, %.2f ms, NPU busy "
          "%.1f ms\n",
          rows, parts,
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
          gemm.TakeBusyMs());
    return out;
  }

  // The encoder over rows [r0, r1) of `wk`, through the final projection.
  void RunPart(Work& wk, std::size_t r0, std::size_t r1, int lane,
               std::barrier<>* sync, const std::function<bool()>& cancelled,
               std::atomic<bool>& stop) {
    const TextConfig& c = *config;
    const std::size_t rows = r1 - r0, hidden = c.hidden_size,
                      per_layer = c.hidden_size_per_layer_input,
                      layers_n = c.num_hidden_layers,
                      heads = c.num_attention_heads,
                      inner = c.intermediate_size;
    const float eps = c.rms_norm_eps;
    const auto& starts = wk.starts;
    float* state = wk.state.data() + r0 * hidden;
    float* ple = wk.ple.data() + r0 * layers_n * per_layer;
    float* tmp = wk.tmp.data() + r0 * hidden;
    std::uint16_t* normed16 = wk.normed16.data() + r0 * hidden;
    std::uint16_t* act16 = wk.act16.data() + r0 * std::max(inner, per_layer);

    host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
      std::vector<float> row(hidden);
      for (std::size_t r = b; r < e; ++r) {
        host::Bf16ToFloat(embed + std::size_t(wk.tokens[r0 + r]) * hidden,
                          row.data(), hidden);
        for (std::size_t i = 0; i < hidden; ++i)
          row[i] *= kEmbedScale;
        host::RoundBf16(row.data(), hidden);
        std::copy(row.begin(), row.end(), state + r * hidden);
      }
    });
    gemm.Run(ple_model, state, int(rows), ple);
    const float ple_scale = 1.0f / std::sqrt(float(hidden));
    host::ParallelFor(rows * layers_n, 64, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b * per_layer; i < e * per_layer; ++i)
        ple[i] *= ple_scale;
    });
    host::RmsNormRows(ple, ple, rows * layers_n, per_layer, ple_norm.data(),
                      eps);

    using Epilogue = gufo::npu::BfpGemm::Epilogue;
    // act(gate) * up of an interleaved [gate 64 | up 64] tile, as bf16 (rows
    // relative to this part).
    auto gated = [&](std::uint16_t* out, std::size_t out_w, const float* add,
                     std::size_t add_stride) {
      return [=](int t0, int n_rows, int n0, int w, const std::uint16_t* tile) {
        float buf[128], res[kGateBlock];
        for (int i = 0; i < n_rows; ++i) {
          const std::size_t r = std::size_t(t0 + i);
          host::Bf16ToFloat(tile + std::size_t(i) * 128, buf, std::size_t(w));
          if (add) {  // ple: act(gate tile) * ple row slice
            host::GatedRows(buf, 0, add + r * add_stride + std::size_t(n0), 0,
                            res, 0, 1, std::size_t(w),
                            host::Activation::kGeluTanh);
            host::FloatToBf16(res, out + r * out_w + std::size_t(n0),
                              std::size_t(w));
          } else {
            host::GatedRows(buf, 0, buf + kGateBlock, 0, res, 0, 1, kGateBlock,
                            host::Activation::kGeluTanh);
            host::FloatToBf16(res, out + r * out_w + std::size_t(n0) / 2,
                              kGateBlock);
          }
        }
      };
    };

    for (std::size_t l = 0; l < layers_n; ++l) {
      if ((cancelled && cancelled()) || stop) {
        stop = true;
        if (sync)
          sync->arrive_and_drop();
        return;
      }
      const Layer& w = layers[l];
      const LayerGeometry& g = c.layers[l];
      const std::size_t hd = g.head_dim, kv = g.num_key_value_heads;
      const std::size_t qw = heads * hd, kw = kv * hd, row_w = qw + 2 * kw;
      float* qkv_all = wk.qkv[l % 2].data();
      float* qkv = qkv_all + r0 * row_w;
      float* context = wk.context.data() + r0 * qw;
      std::uint16_t* context16 = wk.context16.data() + r0 * qw;
      host::RmsNormRowsBf16(state, normed16, rows, hidden, w.input_norm.data(),
                            eps);
      gemm.Run(gufo::npu::BfpGemm::Call{w.qkv, nullptr, int(rows), qkv,
                                        normed16, nullptr});
      host::ParallelFor(rows, 16, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
          float* q = qkv + r * row_w;
          host::RmsNormRows(q, q, heads, hd, w.q_norm.data(), eps);
          host::RmsNormRows(q + qw, q + qw, kv, hd, w.k_norm.data(), eps);
          host::RmsNormRows(q + qw + kw, q + qw + kw, kv, hd, nullptr, eps);
        }
      });
      // Rotary embeddings on q and k: k heads follow q heads in each row.
      host::RopeNeox(qkv, rows, row_w, heads + kv, hd, hd,
                     wk.positions.data() + r0, g.rope_theta);
      if (sync)
        sync->arrive_and_wait();  // every part's keys and values are ready
      if (stop)
        return;
      const bool windowed = g.type == LayerType::kSlidingAttention;
      const std::size_t window = c.sliding_window;
      auto visible = [&](std::size_t r, std::size_t lo, std::size_t hi,
                         std::size_t* b, std::size_t* e) {
        if (windowed) {
          lo = std::max(lo, r > window ? r - window : 0);
          hi = std::min(hi, r + window + 1);
        }
        *b = lo;
        *e = hi;
      };
      // Long sequences: QK^T and PV on the NPU, the softmax on the CPU.
      NpuAttention(scratches[lane], qkv_all, row_w, heads, kv, hd, starts,
                   wk.long_seq, r0, r1, wk.span, visible, wk.context16.data(),
                   qw);
      host::Attention(
          {.q = qkv,
           .q_stride = row_w,
           .k = qkv_all + qw,
           .k_stride = row_w,
           .v = qkv_all + qw + kw,
           .v_stride = row_w,
           .out = context,
           .out_stride = qw,
           .rows = rows,
           .heads = heads,
           .kv_heads = kv,
           .dim = hd,
           .dim_v = hd,
           .scale = 1.0f},
          [&](std::size_t r, std::size_t* b, std::size_t* e) {
            const std::size_t q = wk.seq_of[r0 + r];
            if (wk.long_seq[q]) {
              *b = *e = 0;  // done on the NPU; leave its rows
              return;
            }
            visible(r0 + r, starts[q], starts[q + 1], b, e);
          },
          /*skip_empty=*/true);
      // CPU-attention rows (short sequences) join the bf16 context.
      host::ParallelFor(rows, 64, [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r)
          if (!wk.long_seq[wk.seq_of[r0 + r]])
            host::FloatToBf16(context + r * qw, context16 + r * qw, qw);
      });
      gemm.Run(gufo::npu::BfpGemm::Call{w.o, nullptr, int(rows), tmp, context16,
                                        nullptr});
      host::AddNormRows(state, tmp, rows, hidden, w.post_attn_norm.data(), eps,
                        1.0f);

      host::RmsNormRowsBf16(state, normed16, rows, hidden,
                            w.pre_ffn_norm.data(), eps);
      const Epilogue ffn = gated(act16, inner, nullptr, 0);
      gemm.Run(gufo::npu::BfpGemm::Call{w.gate_up, nullptr, int(rows), nullptr,
                                        normed16, &ffn});
      gemm.Run(gufo::npu::BfpGemm::Call{w.down, nullptr, int(rows), tmp, act16,
                                        nullptr});
      host::AddNormRows(state, tmp, rows, hidden, w.post_ffn_norm.data(), eps,
                        1.0f);

      const Epilogue ple_gate =
          gated(act16, per_layer, ple + l * per_layer, layers_n * per_layer);
      gemm.Run(gufo::npu::BfpGemm::Call{w.ple_gate, state, int(rows), nullptr,
                                        nullptr, &ple_gate});
      gemm.Run(gufo::npu::BfpGemm::Call{w.ple_proj, nullptr, int(rows), tmp,
                                        act16, nullptr});
      host::AddNormRows(state, tmp, rows, hidden, w.ple_norm.data(), eps,
                        w.scalar);
    }
    host::RmsNormRows(state, state, rows, hidden, final_norm.data(), eps);
    gemm.Run(projection, state, int(rows),
             wk.projected.data() + r0 * c.embedding_dim);
  }
};

Encoder::Encoder() : impl_(std::make_unique<Impl>()) {}

const ModelConfig& Encoder::config() const {
  return impl_->text_only;
}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::Create(const std::string& model_dir,
                                         std::string* error) {
  std::unique_ptr<Encoder> encoder(new Encoder());
  try {
    encoder->impl_->Load(model_dir);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("NPU encoder: ") + e.what();
    return nullptr;
  }
  return encoder;
}

bool Encoder::EmbedMultimodal(const std::vector<EncoderSequence>& sequences,
                              std::vector<std::vector<float>>* embeddings,
                              std::string* error,
                              const std::function<bool()>& is_cancelled) {
  std::vector<std::uint32_t> tokens;
  std::vector<std::size_t> starts{0};
  for (const EncoderSequence& s : sequences) {
    if (!s.images.empty() || !s.audio.empty()) {
      if (error)
        *error = "embedding_gemma2: --npu embeds text only";
      return false;
    }
    if (s.tokens.empty()) {
      if (error)
        *error = "embedding_gemma2: empty sequence";
      return false;
    }
    for (std::uint32_t t : s.tokens)
      if (t >= impl_->config->vocab_size) {
        if (error)
          *error = "embedding_gemma2: token id out of range";
        return false;
      }
    tokens.insert(tokens.end(), s.tokens.begin(), s.tokens.end());
    starts.push_back(tokens.size());
  }
  std::lock_guard lock(impl_->mutex);
  try {
    auto out = impl_->RunSplit(tokens, starts, is_cancelled);
    if (out.size() != sequences.size()) {
      if (error)
        *error = "embedding_gemma2: request cancelled";
      return false;
    }
    *embeddings = std::move(out);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("embedding_gemma2 NPU: ") + e.what();
    std::fprintf(stderr, "embedding_gemma2 NPU: %s\n", e.what());
    return false;
  }
  return true;
}

}  // namespace gufo::models::embedding_gemma2::npu
