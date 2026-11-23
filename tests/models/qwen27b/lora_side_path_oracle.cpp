// Real-weight GPU tool for the Qwen3.8 27B LoRA side path. It loads the trunk
// with an adapter whose targets are not all Q8_0 (so the side path applies)
// and has three modes.
//
//   qwen27b_lora_side_path_oracle MODEL.gguf ADAPTER.gguf SCALE OUT.bin
//
// Kernel results for an independent FP64 reference to check, and decode
// equality checked here. For every patch, through the layer's uploaded
// descriptor, and for two synthetic patches of rank 7 and 64 (a second rank
// chunk), the kernel adds the update for 5 seeded rows into a zero and into a
// seeded nonzero destination, and from the same rows rounded to IEEE half into
// a zero destination; each row is also computed alone and must equal its
// batched row bit for bit. OUT.bin holds one record per patch, in layer order
// with the output projection before ffn_down, then the synthetic ones: a
// 64-byte NUL-padded name ("blk.N.<tensor>.weight" or "synthetic.rankR"), u32
// rows, in, out, rank, f32 multiplier, then f32 [rows][out] zero-destination,
// nonzero-destination and half-input results. The reference
// (gufo-build qwen27/side_path_oracle.py) reads the factors of a real patch
// from the adapter GGUF and computes multiplier * B * (A * x) in float64; it
// regenerates every input with this generator: state s0, then
// s = s * 1664525 + 1013904223 (mod 2^32) and value (s >> 8) / 2^23 - 1 per
// step. Rows x and then the nonzero destination come from s0 = 0x9E3779B9 ^
// (2 * layer + ffn_down), with layer 1000 + R for synthetic rank R. Synthetic
// factors come from s0 = 0xC0FFEE ^ R: A [R][in] then B [out][R], each value
// times 1/64, with multiplier 0.75. Afterwards the logits and DFlash2 target
// features of eight tokens decoded one at a time must equal, bit for bit, a
// verification chunk of the same tokens and two sessions decoding them
// together. ADAPTER "none" runs only this equality check, on the stock model.
//
//   qwen27b_lora_side_path_oracle MODEL.gguf ADAPTER.gguf SCALE --prefill
//
// Exact prefill check. Single-pair adapters cut from ADAPTER patch the first
// non-Q8_0 ssm_out, attn_output and ffn_down of the trunk. Stock and adapted
// prefill run the same prompt and are read after the patched layer; the
// adapted projection output (Q8_1 route, 64 rows) or hidden state (FP16 route,
// 1,024 and 1,025 rows, ffn_down only: an output projection's sum is not
// visible at a layer boundary there) must exceed stock's by
// multiplier * B * (A * x) for the input x the adapted pass fed the
// projection, computed here in float64, in every row, within float32 rounding.
// A control that drops one row's update must fail. With the whole adapter, the
// last logits of a prefill whose final layer computes only its last 256-row
// tile must equal, bit for bit, those of a prefill that captures every row of
// the final layer, at 1,024 and 1,025 rows.
//
//   qwen27b_lora_side_path_oracle MODEL.gguf ADAPTER.gguf SCALE
//   --prefill-decode
//
// Diagnostics only: prefill against token-by-token decoding, stock and
// adapted, for 64 to 1,025 rows; prints distances and always succeeds.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/lora_side_path.hpp"
#include "tests/families/qwen/lora_fixture.hpp"

namespace gufo::hip {

// Reads the arena between prefill layers (a friend of the executor).
class QwenGpuExecutorTestAccess {
public:
  static const QwenGpuArena& Arena(const QwenGpuExecutor& executor) {
    return executor.arena_;
  }
};

}  // namespace gufo::hip

namespace {

constexpr std::uint32_t kRows = 5;

class Generator {
public:
  explicit Generator(std::uint32_t seed) : state_(seed) {}
  float Next() {
    state_ = state_ * 1664525U + 1013904223U;
    return static_cast<float>(state_ >> 8) / 8388608.0F - 1.0F;
  }

private:
  std::uint32_t state_;
};

using Executor = gufo::hip::QwenGpuExecutor;
using Model = std::shared_ptr<const gufo::hip::QwenGpuModel>;
using Token = gufo::tokenization::TokenId;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << '\n';
    std::exit(1);
  }
}

bool BitEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && !a.empty() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::vector<float> FromDevice(const void* data, std::size_t count) {
  std::vector<float> host(count);
  Check(hipMemcpy(host.data(), data, count * sizeof(float),
                  hipMemcpyDeviceToHost),
        "read");
  return host;
}

std::vector<float> HalfFromDevice(const void* data, std::size_t count) {
  std::vector<std::uint16_t> bits(count);
  Check(hipMemcpy(bits.data(), data, count * sizeof(std::uint16_t),
                  hipMemcpyDeviceToHost),
        "read");
  std::vector<float> host(count);
  for (std::size_t i = 0; i < count; ++i)
    host[i] = gufo::quant::Fp16ToFloat(bits[i]);
  return host;
}

Model Load(const char* path, gufo::models::qwen::lora::LoraSelection lora,
           std::string* error) {
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  return reader ? gufo::hip::QwenGpuModel::CreateFromGguf(
                      std::shared_ptr<const gufo::core::GgufReader>(
                          std::move(reader)),
                      error, {}, lora)
                : nullptr;
}

// Adds `update` for kRows seeded rows and writes one OUT.bin record; counts
// rows whose lone computation differs from their batched one.
void WriteRecord(std::ofstream& out, const std::string& name,
                 const gufo::models::QwenLowRankUpdate& update,
                 std::uint32_t seed, std::size_t* row_mismatches) {
  Generator generator(seed);
  std::vector<float> x(std::size_t{kRows} * update.in);
  for (auto& v : x)
    v = generator.Next();
  std::vector<float> y0(std::size_t{kRows} * update.out);
  for (auto& v : y0)
    v = generator.Next();
  std::vector<std::uint16_t> x_half(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    x_half[i] = gufo::models::qwen::lora::FloatToHalf(x[i]);
  const std::size_t y_bytes = y0.size() * sizeof(float);
  float *dx = nullptr, *dzero = nullptr, *dnonzero = nullptr,
        *dsingle = nullptr, *dhalf_out = nullptr;
  void* dx_half = nullptr;
  Check(hipMalloc(&dx, x.size() * sizeof(float)), "hipMalloc");
  Check(hipMalloc(&dx_half, x_half.size() * sizeof(std::uint16_t)),
        "hipMalloc");
  for (auto** p : {&dzero, &dnonzero, &dsingle, &dhalf_out})
    Check(hipMalloc(p, y_bytes), "hipMalloc");
  Check(
      hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "copy x");
  Check(hipMemcpy(dx_half, x_half.data(), x_half.size() * sizeof(std::uint16_t),
                  hipMemcpyHostToDevice),
        "copy x half");
  for (auto* p : {dzero, dsingle, dhalf_out})
    Check(hipMemset(p, 0, y_bytes), "memset");
  Check(hipMemcpy(dnonzero, y0.data(), y_bytes, hipMemcpyHostToDevice),
        "copy y0");
  gufo::hip::AddLoraSidePath(update, dx, dzero, kRows, nullptr);
  gufo::hip::AddLoraSidePath(update, dx, dnonzero, kRows, nullptr);
  for (std::uint32_t r = 0; r < kRows; ++r)
    gufo::hip::AddLoraSidePath(update, dx + std::size_t{r} * update.in,
                               dsingle + std::size_t{r} * update.out, 1,
                               nullptr);
  gufo::hip::AddLoraSidePathFp16(update, dx_half, dhalf_out, kRows, nullptr);
  const auto zero = FromDevice(dzero, y0.size());
  const auto nonzero = FromDevice(dnonzero, y0.size());
  const auto single = FromDevice(dsingle, y0.size());
  const auto half = FromDevice(dhalf_out, y0.size());
  (void)hipFree(dx_half);
  for (auto* p : {dx, dzero, dnonzero, dsingle, dhalf_out})
    (void)hipFree(p);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    *row_mismatches += std::memcmp(zero.data() + std::size_t{r} * update.out,
                                   single.data() + std::size_t{r} * update.out,
                                   update.out * sizeof(float)) != 0;
  }
  char padded[64] = {};
  std::strncpy(padded, name.c_str(), sizeof(padded) - 1);
  const std::uint32_t header[] = {kRows, update.in, update.out, update.rank};
  out.write(padded, sizeof(padded));
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(&update.multiplier), sizeof(float));
  for (const auto* values : {&zero, &nonzero, &half})
    out.write(reinterpret_cast<const char*>(values->data()),
              static_cast<std::streamsize>(y_bytes));
}

// A synthetic patch of rank `rank`; owns its device factors.
struct SyntheticPatch {
  gufo::models::QwenLowRankUpdate update;
  SyntheticPatch(std::uint32_t rank, std::uint32_t in, std::uint32_t out) {
    Generator generator(0xC0FFEEU ^ rank);
    std::vector<float> a(std::size_t{rank} * in), b(std::size_t{out} * rank);
    for (auto* factor : {&a, &b})
      for (auto& v : *factor)
        v = generator.Next() / 64.0F;
    float *da = nullptr, *db = nullptr;
    Check(hipMalloc(&da, a.size() * sizeof(float)), "hipMalloc");
    Check(hipMalloc(&db, b.size() * sizeof(float)), "hipMalloc");
    Check(hipMemcpy(da, a.data(), a.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "copy A");
    Check(hipMemcpy(db, b.data(), b.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "copy B");
    update = {.a = da,
              .b = db,
              .in = in,
              .out = out,
              .rank = rank,
              .multiplier = 0.75F};
  }
  ~SyntheticPatch() {
    (void)hipFree(const_cast<float*>(update.a));
    (void)hipFree(const_cast<float*>(update.b));
  }
  SyntheticPatch(const SyntheticPatch&) = delete;
  SyntheticPatch& operator=(const SyntheticPatch&) = delete;
};

// Scalar decoding against batched verification and concurrent decoding of
// the same rows. Fresh sessions replace snapshots.
bool CheckDecodeEquality(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model) {
  constexpr std::uint32_t kPrompt = 24;
  constexpr std::size_t kSteps = 8;
  constexpr std::uint32_t kTaps[] = {1, 16, 31, 46, 61};
  auto session = [&] {
    auto executor = std::make_unique<Executor>(model, 4096);
    executor->SetPromptHiddenCapture(true, kTaps);
    return executor;
  };
  const auto tokens = model->GetTokenizer().Encode(
      "The side path adds a low-rank update beside three projections of "
      "every layer of the trunk. Each row of a batch must come out exactly "
      "as it does alone, whether it is decoded one token at a time, verified "
      "as part of a drafted chunk, or decoded next to another session.");
  if (tokens.size() < kPrompt + kSteps) {
    std::cerr << "equality fixture has too few tokens\n";
    return false;
  }
  const auto prompt = std::span(tokens).first(kPrompt);
  const auto suffix = std::span(tokens).subspan(kPrompt, kSteps);
  std::vector<std::vector<float>> logits, features;
  auto scalar = session();
  (void)scalar->ForwardPromptBatch(prompt);
  for (std::size_t i = 0; i < kSteps; ++i) {
    (void)scalar->ForwardToken(suffix[i], kPrompt + i);
    const auto l = scalar->CopyLastLogits();
    const auto f = scalar->CopyLastHidden();
    logits.emplace_back(l.begin(), l.end());
    features.emplace_back(f.begin(), f.end());
  }
  bool ok = true;
  auto verify = session();
  (void)verify->ForwardPromptBatch(prompt);
  verify->SaveState(kPrompt);
  (void)verify->ForwardVerificationChunk(suffix, kPrompt, true);
  const auto hidden = verify->GetVerificationHiddenStates();
  const std::size_t width = features.front().size();
  std::size_t verified = 0;
  for (std::size_t i = 0; i < kSteps; ++i) {
    verified += BitEqual(verify->CopyVerificationLogits(i), logits[i]) &&
                BitEqual(hidden.subspan(i * width, width), features[i]);
  }
  std::cout << "verification chunk of " << kSteps
            << " rows equal to scalar decoding: " << verified << '\n';
  ok = ok && verified == kSteps;
  auto first = session();
  auto second = session();
  (void)first->ForwardPromptBatch(prompt);
  (void)second->ForwardPromptBatch(prompt);
  std::size_t batched = 0;
  for (std::size_t i = 0; i < kSteps; ++i) {
    const auto position = static_cast<std::uint32_t>(kPrompt + i);
    const gufo::hip::QwenGpuBatchItem items[] = {
        {first.get(), suffix[i], position},
        {second.get(), suffix[i], position}};
    (void)Executor::ForwardTokenBatch(items);
    for (auto* executor : {first.get(), second.get()})
      batched += BitEqual(executor->CopyLastLogits(), logits[i]) &&
                 BitEqual(executor->CopyLastHidden(), features[i]);
  }
  std::cout << "two-session decode rows equal to scalar decoding: " << batched
            << " of " << 2 * kSteps << '\n';
  return ok && batched == 2 * kSteps;
}

// Varied rows: one repeated sentence drives the model into a regime where
// stock prefill itself strays far from decoding. Empty when too short.
std::vector<Token> PrefillTokens(const gufo::hip::QwenGpuModel& model) {
  std::string text;
  for (int i = 0; text.size() < 8000; ++i) {
    static constexpr const char* kColours[] = {"red", "green", "blue", "amber",
                                               "grey"};
    text += "Item " + std::to_string(i) + ": the " + kColours[i % 5] +
            " box on shelf " + std::to_string(i * 7 % 13) + " holds " +
            std::to_string(i * 11 % 17) + " stones and " +
            std::to_string(i * 3 % 7) + " keys. ";
  }
  const auto encoded = model.GetTokenizer().Encode(text);
  if (encoded.size() < 1025) {
    std::cerr << "prefill fixture has too few tokens\n";
    return {};
  }
  return {encoded.begin(), encoded.begin() + 1025};
}

struct Rows {
  std::vector<float> features;  ///< [rows][taps][hidden]
  std::vector<float> logits;    ///< Last row.
};

constexpr std::uint32_t kPrefillTaps[] = {1, 16, 31, 46, 61};

Rows Prefill(const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
             std::span<const Token> tokens) {
  Executor executor(model, 4096);
  executor.SetPromptHiddenCapture(true, kPrefillTaps);
  (void)executor.ForwardPromptBatch(tokens);
  const auto features = executor.GetPromptHiddenStates();
  const auto logits = executor.CopyLastLogits();
  return {{features.begin(), features.end()}, {logits.begin(), logits.end()}};
}

Rows Decode(const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
            std::span<const Token> tokens) {
  Executor executor(model, 4096);
  executor.SetPromptHiddenCapture(true, kPrefillTaps);
  Rows rows;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    (void)executor.ForwardToken(tokens[i], static_cast<std::uint32_t>(i),
                                i + 1 == tokens.size());
    const auto features = executor.CopyLastHidden();
    rows.features.insert(rows.features.end(), features.begin(), features.end());
  }
  const auto logits = executor.CopyLastLogits();
  rows.logits.assign(logits.begin(), logits.end());
  return rows;
}

/// sqrt(sum (a - b)^2 / sum b^2) over every element, and per row.
struct Distance {
  double all{0.0};
  std::vector<double> rows;

  [[nodiscard]] double Median() const {
    auto sorted = rows;
    std::ranges::sort(sorted);
    return sorted[sorted.size() / 2];
  }
};

Distance Compare(std::span<const float> a, std::span<const float> b,
                 std::size_t row) {
  Distance d;
  double error = 0.0, norm = 0.0;
  for (std::size_t r = 0; r * row < a.size(); ++r) {
    double row_error = 0.0, row_norm = 0.0;
    for (std::size_t i = r * row; i < (r + 1) * row; ++i) {
      const double diff = static_cast<double>(a[i]) - b[i];
      row_error += diff * diff;
      row_norm += static_cast<double>(b[i]) * b[i];
    }
    error += row_error;
    norm += row_norm;
    d.rows.push_back(std::sqrt(row_error / row_norm));
  }
  d.all = std::sqrt(error / norm);
  return d;
}

void PrefillDecodeDiagnostics(const char* model_path, const char* adapter,
                              float scale) {
  std::string error;
  const auto load = [&](gufo::models::qwen::lora::LoraSelection lora) {
    auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
    return reader ? gufo::hip::QwenGpuModel::CreateFromGguf(
                        std::shared_ptr<const gufo::core::GgufReader>(
                            std::move(reader)),
                        &error, {}, lora)
                  : nullptr;
  };
  const auto stock = load({});
  const auto patched =
      stock ? load({.path = adapter, .scale = scale}) : nullptr;
  if (!patched) {
    std::cerr << "load failed: " << error << '\n';
    return;
  }
  std::cout << "identity " << patched->AdapterIdentity() << '\n';
  const auto tokens = PrefillTokens(*patched);
  if (tokens.empty())
    return;
  const std::size_t width =
      std::size(kPrefillTaps) * patched->GetConfig().hidden_size;
  for (const std::size_t n : {64U, 1023U, 1024U, 1025U}) {
    const auto prompt = std::span(tokens).first(n);
    const auto stock_prefill = Prefill(stock, prompt);
    const auto stock_decode = Decode(stock, prompt);
    const auto patched_prefill = Prefill(patched, prompt);
    const auto patched_decode = Decode(patched, prompt);
    const auto d0 =
        Compare(stock_prefill.features, stock_decode.features, width);
    const auto d1 =
        Compare(patched_prefill.features, patched_decode.features, width);
    const auto d2 =
        Compare(stock_decode.features, patched_decode.features, width);
    // Rows where the prefill with the adapter is nearer stock decoding than
    // decoding with the adapter, among rows the update visibly moves.
    const auto to_stock =
        Compare(patched_prefill.features, stock_decode.features, width);
    std::size_t moved = 0, unpatched = 0, stock_far = 0, patched_far = 0;
    for (std::size_t r = 0; r < n; ++r) {
      stock_far += d0.rows[r] > 0.3;
      patched_far += d1.rows[r] > 0.3;
      if (d1.rows[r] > 0.3 && n < 1000) {
        const std::size_t hidden = width / std::size(kPrefillTaps);
        const auto row_of = [&](const Rows& rows) {
          return std::span(rows.features).subspan(r * width, width);
        };
        const auto t0 =
            Compare(row_of(stock_prefill), row_of(stock_decode), hidden);
        const auto t1 =
            Compare(row_of(patched_prefill), row_of(patched_decode), hidden);
        std::cout << "  row " << r << " far from decoding, per tap (stock / "
                  << "adapter):";
        for (std::size_t t = 0; t < std::size(kPrefillTaps); ++t)
          std::cout << ' ' << kPrefillTaps[t] << ": " << t0.rows[t] << " / "
                    << t1.rows[t];
        std::cout << '\n';
      }
      if (d2.rows[r] > 2.0 * d0.rows[r]) {
        ++moved;
        if (to_stock.rows[r] <= d1.rows[r]) {
          ++unpatched;
          std::cout << "  row " << r << " nearer stock decoding: stock "
                    << "prefill-decode " << d0.rows[r] << ", adapter "
                    << "prefill-decode " << d1.rows[r] << ", adapter prefill-"
                    << "stock decode " << to_stock.rows[r]
                    << ", stock-adapter decode " << d2.rows[r] << '\n';
        }
      }
    }
    const std::size_t vocab = stock_decode.logits.size();
    const auto l0 = Compare(stock_prefill.logits, stock_decode.logits, vocab);
    const auto l1 =
        Compare(patched_prefill.logits, patched_decode.logits, vocab);
    const auto l2 = Compare(stock_decode.logits, patched_decode.logits, vocab);
    const auto top = [](const std::vector<float>& v) {
      return std::ranges::max_element(v) - v.begin();
    };
    std::cout << "prefill rows=" << n << " features: stock prefill-decode "
              << d0.all << ", adapter prefill-decode " << d1.all
              << ", stock-adapter decode " << d2.all
              << "; rows the update moves " << moved
              << ", of which nearer stock decoding " << unpatched
              << "; rows over 0.3 from decoding: stock " << stock_far
              << ", adapter " << patched_far << "; last logits: " << l0.all
              << ", " << l1.all << ", " << l2.all
              << "; top1 adapter prefill=" << top(patched_prefill.logits)
              << " decode=" << top(patched_decode.logits) << '\n';
    std::cout << "  median row: stock prefill-decode " << d0.Median()
              << ", adapter prefill-decode " << d1.Median()
              << ", stock-adapter decode " << d2.Median() << '\n';
  }
}

// The projection a single-pair adapter patches, its layer, and the buffers
// that hold the projection's input and the sum it adds into after that layer.
struct Case {
  std::string tensor;  ///< blk.N.<name>.weight
  std::uint32_t layer{0};
  bool ffn_down{false};
};

struct LayerState {
  std::vector<float> x;  ///< [rows][in], as the projection read it
  std::vector<float> y;  ///< [rows][hidden]
};

// Runs a prefill of `tokens` and reads the patched projection's input and
// output sum once its layer has finished.
LayerState PrefillThroughLayer(const Model& model, const Case& c,
                               std::span<const Token> tokens) {
  const auto& config = model->GetConfig();
  const auto& layer = model->GetWeights().layers[c.layer];
  const std::size_t rows = tokens.size();
  const bool half = rows >= 1024;  // The FP16 route, as prefill selects it.
  const std::size_t in = c.ffn_down                ? config.intermediate_size
                         : layer.is_full_attention ? config.AttentionSize()
                                                   : config.ssm_inner_size;
  Executor executor(model, 4096);
  LayerState state;
  executor.SetPrefillYieldLayers({c.layer + 1}, [&] {
    const auto& arena = gufo::hip::QwenGpuExecutorTestAccess::Arena(executor);
    const std::size_t hidden = config.hidden_size;
    if (half) {
      // ffn_down: SwiGLU output in half, update added into the hidden state.
      state.x = HalfFromDevice(arena.d_ffn_up, rows * in);
      state.y = FromDevice(arena.d_hidden, rows * hidden);
    } else {
      state.x =
          FromDevice(c.ffn_down ? arena.d_ffn_act : arena.d_ssm_out, rows * in);
      state.y = FromDevice(c.ffn_down ? arena.d_ffn_out : arena.d_attn_out,
                           rows * hidden);
    }
  });
  (void)executor.ForwardPromptBatch(tokens);
  return state;
}

// Worst ratio, over every element of every row, of
// |(adapted - stock) - multiplier * B * (A * x)| to its tolerance: one float32
// ulp of the sum plus 1e-6 of the magnitude-weighted update. Above 1 fails.
// `drop_row` expects no update in that row (the control).
double Worst(const LayerState& stock, const LayerState& adapted,
             const gufo::models::QwenLowRankUpdate& update,
             std::optional<std::size_t> drop_row = std::nullopt) {
  const auto a = FromDevice(update.a, std::size_t{update.rank} * update.in);
  const auto b = FromDevice(update.b, std::size_t{update.out} * update.rank);
  const std::size_t rows = adapted.x.size() / update.in;
  const double m = update.multiplier;
  double worst = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    std::vector<double> t(update.rank), magnitude(update.rank);
    for (std::uint32_t k = 0; k < update.rank; ++k) {
      for (std::uint32_t i = 0; i < update.in; ++i) {
        const double term = static_cast<double>(a[k * update.in + i]) *
                            adapted.x[r * update.in + i];
        t[k] += term;
        magnitude[k] += std::abs(term);
      }
    }
    for (std::uint32_t o = 0; o < update.out; ++o) {
      double expected = 0.0, bound = 0.0;
      for (std::uint32_t k = 0; k < update.rank; ++k) {
        expected += static_cast<double>(b[o * update.rank + k]) * t[k];
        bound += std::abs(static_cast<double>(b[o * update.rank + k])) *
                 magnitude[k];
      }
      expected = drop_row == r ? 0.0 : m * expected;
      const float sum = adapted.y[r * update.out + o];
      const float base = stock.y[r * update.out + o];
      const float larger = std::max(std::abs(sum), std::abs(base));
      const double ulp =
          std::nextafter(larger, std::numeric_limits<float>::infinity()) -
          larger;
      const double error =
          std::abs((static_cast<double>(sum) - base) - expected);
      worst = std::max(worst, error / (ulp + 1e-6 * std::abs(m) * bound));
    }
  }
  return worst;
}

// Writes a one-pair adapter with ADAPTER's pair for `tensor` and returns its
// path.
std::filesystem::path SinglePairAdapter(const char* adapter_path,
                                        const std::string& tensor) {
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(adapter_path, &error);
  if (!reader)
    throw std::runtime_error("adapter: " + error);
  std::vector<lora_fixture::Tensor> pair;
  for (const char* half : {".lora_a", ".lora_b"}) {
    const auto* info = reader->FindTensor(tensor + half);
    if (info == nullptr)
      throw std::runtime_error("adapter has no " + tensor + half);
    const auto* bytes = static_cast<const std::uint8_t*>(info->data);
    pair.push_back({std::string(info->name),
                    info->dimensions,
                    info->type,
                    {bytes, bytes + info->size_bytes}});
  }
  const auto alpha = reader->GetMetadataFloat32("adapter.lora.alpha");
  const auto bytes =
      lora_fixture::Adapter("qwen35", alpha.value_or(1.0F), pair);
  const auto path = std::filesystem::temp_directory_path() /
                    ("qwen27b-side-path-" + tensor + ".gguf");
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return path;
}

bool CheckPrefillExact(const char* model_path, const char* adapter,
                       float scale) {
  std::string error;
  const auto stock = Load(model_path, {}, &error);
  if (!stock) {
    std::cerr << "load failed: " << error << '\n';
    return false;
  }
  const auto tokens = PrefillTokens(*stock);
  if (tokens.empty())
    return false;
  // The first non-Q8_0 target of each kind, so the pair takes the side path.
  std::vector<Case> cases;
  const auto& layers = stock->GetWeights().layers;
  for (const auto kind : {"ssm_out", "attn_output", "ffn_down"}) {
    for (std::uint32_t l = 0; l < layers.size(); ++l) {
      const auto& layer = layers[l];
      const bool ffn_down = std::string_view(kind) == "ffn_down";
      const auto& tensor = ffn_down                  ? layer.ffn_down
                           : layer.is_full_attention ? layer.attn_output
                                                     : layer.ssm_out;
      const bool matches =
          ffn_down ||
          (std::string_view(kind) == "attn_output") == layer.is_full_attention;
      if (matches && !tensor.empty() &&
          tensor.type != gufo::core::GgmlType::kQ8_0) {
        cases.push_back(
            {"blk." + std::to_string(l) + "." + kind + ".weight", l, ffn_down});
        break;
      }
    }
  }
  bool ok = cases.size() == 3;
  for (const auto& c : cases) {
    const auto path = SinglePairAdapter(adapter, c.tensor);
    const auto adapted =
        Load(model_path, {.path = path.string(), .scale = scale}, &error);
    std::filesystem::remove(path);
    if (!adapted || adapted->AdapterIdentity().find("low-rank-fp32-v1") ==
                        std::string::npos) {
      std::cerr << c.tensor << ": not loaded on the side path: " << error
                << '\n';
      return false;
    }
    const auto& layer = adapted->GetWeights().layers[c.layer];
    const auto& update = c.ffn_down ? layer.ffn_down_lora : layer.output_lora;
    for (const std::size_t rows : {64U, 1024U, 1025U}) {
      const char* route = rows < 1024 ? "Q8_1" : "FP16";
      if (rows >= 1024 && !c.ffn_down) {
        std::cout << "prefill " << route << " rows=" << rows << ' ' << c.tensor
                  << ": not visible at a layer boundary (the FFN follows in "
                     "the same layer); see the kernel oracle's half inputs\n";
        continue;
      }
      const auto prompt = std::span(tokens).first(rows);
      const auto base = PrefillThroughLayer(stock, c, prompt);
      const auto sum = PrefillThroughLayer(adapted, c, prompt);
      const double worst = Worst(base, sum, update);
      const double control = Worst(base, sum, update, rows / 2);
      const bool pass = worst <= 1.0 && control > 1.0;
      ok = ok && pass;
      std::cout << "prefill " << route << " rows=" << rows << ' ' << c.tensor
                << ": worst error/tolerance " << worst << " over " << rows
                << " rows; control without row " << rows / 2 << "'s update "
                << control << (pass ? " (pass)" : " (FAIL)") << '\n';
    }
  }

  // The final layer's last-tile crop against a prefill that captures every
  // row of the final layer, in fresh sessions.
  const auto patched =
      Load(model_path, {.path = adapter, .scale = scale}, &error);
  if (!patched) {
    std::cerr << "load failed: " << error << '\n';
    return false;
  }
  const std::array final_layer{patched->GetConfig().num_layers - 1U};
  for (const auto* model : {&stock, &patched}) {
    for (const std::size_t rows : {1024U, 1025U}) {
      const auto prompt = std::span(tokens).first(rows);
      Executor cropped(*model, 4096);
      (void)cropped.ForwardPromptBatch(prompt);
      const auto crop = cropped.CopyLastLogits();
      const std::vector<float> crop_logits(crop.begin(), crop.end());
      Executor full(*model, 4096);
      full.SetPromptHiddenCapture(true, final_layer);
      (void)full.ForwardPromptBatch(prompt);
      const bool equal = BitEqual(full.CopyLastLogits(), crop_logits);
      ok = ok && equal;
      std::cout << "final tile " << (model == &stock ? "stock" : "adapter")
                << " rows=" << rows << ": cropped and full final layer logits "
                << (equal ? "equal" : "DIFFER") << '\n';
    }
  }
  std::cout << "exact prefill check: " << (ok ? "pass" : "FAIL") << '\n';
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 5 && std::string_view(argv[4]) == "--prefill")
    return CheckPrefillExact(argv[1], argv[2], std::strtof(argv[3], nullptr))
               ? 0
               : 1;
  if (argc == 5 && std::string_view(argv[4]) == "--prefill-decode") {
    PrefillDecodeDiagnostics(argv[1], argv[2], std::strtof(argv[3], nullptr));
    return 0;
  }
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.gguf ADAPTER.gguf SCALE OUT.bin|--prefill|"
                 "--prefill-decode\n";
    return 2;
  }
  const bool stock = std::string_view(argv[2]) == "none";
  for (int i = 1; i < (stock ? 2 : 3); ++i) {
    if (!std::filesystem::exists(argv[i])) {
      std::cout << "SKIP: " << argv[i] << " is absent\n";
      return 77;
    }
  }
  std::string error;
  const auto model = Load(
      argv[1],
      stock ? gufo::models::qwen::lora::LoraSelection{}
            : gufo::models::qwen::lora::LoraSelection{.path = argv[2],
                                                      .scale = std::strtof(
                                                          argv[3], nullptr)},
      &error);
  if (!model) {
    std::cerr << "load failed: " << error << '\n';
    return 1;
  }
  std::cout << "identity " << model->AdapterIdentity() << '\n';
  if (stock)
    return CheckDecodeEquality(model) ? 0 : 1;
  if (model->AdapterIdentity().find(";merge=low-rank-fp32-v1;") ==
      std::string::npos) {
    std::cerr << "the adapter was not loaded on the side path\n";
    return 1;
  }

  std::ofstream out(argv[4], std::ios::binary);
  const auto& layers = model->GetWeights().layers;
  std::size_t records = 0, row_mismatches = 0;
  for (std::uint32_t l = 0; l < layers.size(); ++l) {
    const auto& layer = layers[l];
    for (const bool ffn_down : {false, true}) {
      const auto& update = ffn_down ? layer.ffn_down_lora : layer.output_lora;
      if (update.empty())
        continue;
      const std::string name = "blk." + std::to_string(l) + "." +
                               (ffn_down                  ? "ffn_down"
                                : layer.is_full_attention ? "attn_output"
                                                          : "ssm_out") +
                               ".weight";
      WriteRecord(out, name, update, 0x9E3779B9U ^ (2 * l + (ffn_down ? 1 : 0)),
                  &row_mismatches);
      ++records;
    }
  }
  // Ranks above the kernel's four-rank chunk.
  for (const std::uint32_t rank : {7U, 64U}) {
    const SyntheticPatch patch(rank, 6144, 5120);
    WriteRecord(out, "synthetic.rank" + std::to_string(rank), patch.update,
                0x9E3779B9U ^ (2 * (1000 + rank)), &row_mismatches);
    ++records;
  }
  std::cout << "records " << records << ", rows computed alone that differ "
            << "from their batched row: " << row_mismatches << '\n';
  const bool equal = CheckDecodeEquality(model);
  return out && row_mismatches == 0 && equal ? 0 : 1;
}
