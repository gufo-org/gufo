// The stage and weight-source seam (stages.hpp, weight_source.hpp) on real
// weights. A weight source wrapping the GGUF one tags projections with a
// format_state, and stages that forward to the GGUF defaults run them:
//
// - neutral: only the output head is tagged (it reaches Linear on every
//   route, and the draft tables through DraftHead/UseDraftSource), one mixer
//   norm is served as a host-shard payload, and PromptLayer re-sets a layer
//   to its own tensors. Logits must be bit-identical to the default model.
// - routing: one attention layer's q/k/v, one GDN layer's qkv/z and one
//   layer's routed and shared experts are tagged as well. Tagging leaves
//   their stacked and fused GGUF routes (F16/WMMA prompt tiles, fused
//   gate/up), so logits round differently and greedy text may drift from the
//   default model. Its first greedy tokens must match the default model's,
//   and its MTP decoding must reproduce its own AR greedy text.
//
// Each run asserts that the hooks it relies on were called.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/stages.hpp"
#include "src/models/qwen38_flash_next/weight_source.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace fn = gufo::ext::qwen::flash_next;
namespace qfn = gufo::models::qwen38_flash_next;

namespace {

constexpr std::uint32_t kContext = 1024;
constexpr std::size_t kDecodeSteps = 6;
constexpr std::size_t kMtpTokens = 24;

void Require(bool ok, const std::string& what) {
  if (!ok)
    throw std::runtime_error(what);
}

/// The format_state of a tagged tensor: its GGUF encoding.
struct Format {
  gufo::core::GgmlType type;
};

struct Counters {
  int uploads{0};
  int draft_heads{0};
  int linear{0};
  int groups{0};
  int prompt_layers{0};
  int prompt_layers_used{0};
  int prompt_rows_missing{0};
  int moe_on_prompt_copy{0};
  int linear_with_layer{0};
};

enum class Mode { kNeutral, kRouting };

class TaggingSource final : public fn::WeightSource {
public:
  TaggingSource(Mode mode, Counters& counters)
      : inner_(fn::CreateGgufWeightSource()), mode_(mode), c_(counters) {}

  [[nodiscard]] bool Probe(const std::string& path) const override {
    return inner_->Probe(path);
  }
  bool Open(const std::string& path, const std::string& mtp,
            std::string* error) override {
    return inner_->Open(path, mtp, error);
  }
  bool BindTrunk(qfn::ModelWeights* w, std::string* error) override {
    if (!inner_->BindTrunk(w, error))
      return false;
    Tag(w->output);
    // A host payload: the first mixer norm, copied out of its mapping.
    auto& norm = w->layers[0].hc_attn.norm;
    host_.assign(
        static_cast<const std::uint8_t*>(norm.data),
        static_cast<const std::uint8_t*>(norm.data) + norm.SizeBytes());
    norm.data = host_.data();
    norm.shard = qfn::kHostShard;
    if (mode_ == Mode::kRouting) {
      for (std::size_t il = 0; il < w->layers.size(); ++il) {
        auto& l = w->layers[il];
        if (!l.linear && attention_layer_ < 0) {
          attention_layer_ = static_cast<int>(il);
          Tag(l.attn_q);
          Tag(l.attn_k);
          Tag(l.attn_v);
        }
        if (l.linear && linear_layer_ < 0) {
          linear_layer_ = static_cast<int>(il);
          Tag(l.ssm_qkv);
          Tag(l.ssm_gate);
        }
      }
      auto& l = w->layers[moe_layer_];
      for (auto* t : {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps,
                      &l.shexp_gate, &l.shexp_up, &l.shexp_down})
        Tag(*t);
    }
    return true;
  }
  bool BindMtp(const qfn::Config& trunk, qfn::MtpWeights* mtp,
               std::string* error) override {
    return inner_->BindMtp(trunk, mtp, error);
  }
  std::unique_ptr<gufo::tokenization::QwenTokenizer> Tokenizer(
      const qfn::Config& config, std::string* error) override {
    return inner_->Tokenizer(config, error);
  }
  std::unique_ptr<qfn::NgramTable> Ngram(const qfn::ModelWeights& w,
                                         std::string* error) override {
    return inner_->Ngram(w, error);
  }
  [[nodiscard]] std::span<const gufo::core::GgufMappedRegion> Regions(
      bool mtp) const override {
    return inner_->Regions(mtp);
  }
  fn::Tensor Upload(const qfn::TensorRef& t, const fn::UploadHint& hint,
                    fn::UploadContext& up) override {
    (void)hint;
    const auto& format = *static_cast<const Format*>(t.format_state);
    qfn::TensorRef gguf = t;
    gguf.type = format.type;
    gguf.format_state = nullptr;
    const std::size_t size = gguf.SizeBytes();
    constexpr std::size_t kTail = 4096;
    void* p = nullptr;
    if (!up.Allocate(&p, size + kTail)) {
      up.Fail("tagged tensor allocation failed");
      return {};
    }
    if (up.Measuring())
      return {};
    if (!up.CopyFromRegion(t.shard, t.file_offset, size, p) ||
        hipMemsetAsync(static_cast<std::uint8_t*>(p) + size, 0, kTail,
                       nullptr) != hipSuccess) {
      up.Fail("tagged tensor copy failed");
      return {};
    }
    ++c_.uploads;
    // The type is the source's to choose; the upload stores kI32 anyway.
    return {p,
            format.type,
            static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows),
            static_cast<std::uint32_t>(t.experts),
            &format};
  }
  bool DraftHead(const fn::Tensor& output, fn::Tensor* head,
                 std::string* error) override {
    (void)error;
    const auto& format = *static_cast<const Format*>(output.format_state());
    *head = {output.data(), format.type, output.cols(), output.rows()};
    ++c_.draft_heads;
    return true;
  }
  [[nodiscard]] std::string ModelName() const override {
    return inner_->ModelName();
  }

  int attention_layer() const { return attention_layer_; }
  int linear_layer() const { return linear_layer_; }
  static constexpr int moe_layer_ = 0;

private:
  void Tag(qfn::TensorRef& t) {
    formats_.push_back({t.type});
    t.format_state = &formats_.back();
  }

  std::unique_ptr<fn::WeightSource> inner_;
  Mode mode_;
  Counters& c_;
  std::deque<Format> formats_;
  std::vector<std::uint8_t> host_;
  int attention_layer_{-1};
  int linear_layer_{-1};
};

class ForwardingStages final : public fn::Stages {
public:
  ForwardingStages(std::vector<std::uint32_t> layers,
                   std::vector<fn::TensorId> ids, Counters& counters)
      : layers_(std::move(layers)), ids_(std::move(ids)), c_(counters) {}

  [[nodiscard]] std::uint32_t Hooks() const override {
    return fn::kPromptLayers;
  }
  bool Linear(const fn::StageContext& ctx, const fn::Tensor& w,
              const fn::LinearInput& in, float* out) override {
    ++c_.linear;
    if (ctx.layer.valid())
      ++c_.linear_with_layer;
    const auto* format = static_cast<const Format*>(w.format_state());
    if (format == nullptr)
      return ctx.defaults.Linear(ctx, w, in, out);
    const fn::Tensor gguf{w.data(), format->type, w.cols(), w.rows(),
                          w.experts()};
    return ctx.defaults.Linear(ctx, gguf, in, out);
  }
  bool LinearGroup(const fn::StageContext& ctx,
                   std::span<const fn::LinearJob> jobs,
                   const fn::LinearInput& in) override {
    ++c_.groups;
    return fn::Stages::LinearGroup(ctx, jobs, in);
  }
  bool MoeFfn(const fn::StageContext& ctx, const fn::MoeRouting& routing,
              const float* x, float* out) override {
    if (ctx.phase == fn::Phase::kPrompt && Edited(ctx.layer.index()) &&
        ctx.layer.tensor(fn::TensorId::kRouter).data() != nullptr)
      ++c_.moe_on_prompt_copy;
    return ctx.defaults.MoeFfn(ctx, routing, x, out);
  }
  bool PromptLayer(const fn::StageContext& ctx, fn::LayerEdit& edit,
                   bool* use) override {
    ++c_.prompt_layers;
    if (ctx.n_tokens == 0)
      ++c_.prompt_rows_missing;
    *use = Edited(ctx.layer.index());
    if (*use) {
      ++c_.prompt_layers_used;
      // The same tensors again: the view must run exactly as the layer.
      const fn::Layer layer = edit.layer();
      for (const auto id : ids_)
        edit.Set(id, layer.tensor(id));
    }
    return true;
  }

private:
  bool Edited(std::uint32_t il) const {
    return std::find(layers_.begin(), layers_.end(), il) != layers_.end();
  }
  std::vector<std::uint32_t> layers_;
  std::vector<fn::TensorId> ids_;
  Counters& c_;
};

struct Result {
  std::vector<float> prefill;
  std::vector<std::vector<float>> decode;
  std::vector<std::int32_t> ar_tokens;
  std::vector<std::int32_t> mtp_tokens;
};

std::string Prompt() {
  std::string text = "Shipping manifest.\n";
  for (int i = 0; i < 24; ++i)
    text += "Crate " + std::to_string(i) + " weighs " +
            std::to_string(i * 37 % 91 + 5) + " kg and holds part " +
            std::to_string(i * 13 % 29) + ".\n";
  return text + "Which crate is the heaviest? Explain briefly.";
}

Result Run(const std::string& path, const std::string& mtp,
           std::unique_ptr<fn::Stages> stages,
           std::unique_ptr<fn::WeightSource> source) {
  std::string error;
  qfn::ModelOptions options;
  options.max_context = kContext;
  options.mtp_model_path = mtp;
  auto model = qfn::Model::Load(path, options, std::move(stages),
                                std::move(source), &error);
  Require(model != nullptr, "load: " + error);
  const auto prompt = model->Tokenize(Prompt());
  Result r;
  {
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, kContext, &error);
    Require(session != nullptr, error);
    Require(session->Sync(prompt, &error), error);
    r.prefill.assign(session->Logits().begin(), session->Logits().end());
    for (std::size_t i = 0; i < kMtpTokens; ++i) {
      const auto logits = session->Logits();
      const auto token = static_cast<std::int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      r.ar_tokens.push_back(token);
      Require(session->Evaluate(token, &error), error);
      if (i < kDecodeSteps)
        r.decode.emplace_back(session->Logits().begin(),
                              session->Logits().end());
    }
  }
  {
    auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                        kContext, &error);
    Require(session != nullptr, error);
    Require(session->Sync(prompt, &error), error);
    gufo::sampling::SamplerState sampler(
        {}, std::vector<gufo::sampling::TokenId>(prompt.begin(), prompt.end()));
    while (r.mtp_tokens.size() < kMtpTokens) {
      qfn::Session::DecodeResult step;
      Require(session->DecodeStep(kMtpTokens - r.mtp_tokens.size(), sampler,
                                  &step, &error, false),
              error);
      Require(!step.tokens.empty(), "MTP step produced no tokens");
      r.mtp_tokens.insert(r.mtp_tokens.end(), step.tokens.begin(),
                          step.tokens.end());
    }
    r.mtp_tokens.resize(kMtpTokens);
  }
  return r;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

float MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float d = 0.0F;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

}  // namespace

int main() {
  const char* model = std::getenv("QWEN38_FLASH_NEXT_MODEL");
  const char* head = std::getenv("QWEN38_FLASH_NEXT_MTP_MODEL");
  if (!model || !*model || !head || !*head) {
    std::cerr << "QWEN38_FLASH_NEXT_MODEL and QWEN38_FLASH_NEXT_MTP_MODEL "
                 "are not set; skipping\n";
    return 77;
  }
  try {
    const Result reference = Run(model, head, nullptr, nullptr);
    bool passed = true;
    const auto check = [&](bool ok, const std::string& what) {
      std::cout << what << ',' << (ok ? "PASS" : "FAIL") << '\n';
      passed = ok && passed;
    };

    {
      Counters c;
      // Layer 0's PLE-free tensors re-set on its prompt view.
      auto stages = std::make_unique<ForwardingStages>(
          std::vector<std::uint32_t>{0},
          std::vector<fn::TensorId>{fn::TensorId::kRouter,
                                    fn::TensorId::kFfnGateExps,
                                    fn::TensorId::kShexpDown},
          c);
      const Result r = Run(model, head, std::move(stages),
                           std::make_unique<TaggingSource>(Mode::kNeutral, c));
      check(Same(r.prefill, reference.prefill), "neutral,prefill_bitwise");
      bool decode_same = r.decode.size() == reference.decode.size();
      for (std::size_t i = 0; decode_same && i < r.decode.size(); ++i)
        decode_same = Same(r.decode[i], reference.decode[i]);
      check(decode_same, "neutral,decode_bitwise");
      check(r.ar_tokens == reference.ar_tokens, "neutral,ar_tokens");
      check(r.mtp_tokens == reference.mtp_tokens, "neutral,mtp_tokens");
      check(c.uploads == 1 && c.draft_heads == 1, "neutral,upload_hooks");
      check(c.linear > 0 && c.linear_with_layer == 0, "neutral,linear_hook");
      check(c.prompt_layers > 0 && c.prompt_layers_used > 0 &&
                c.prompt_rows_missing == 0,
            "neutral,prompt_layer_hook");
      check(c.moe_on_prompt_copy > 0, "neutral,prompt_copy_layer_handle");
    }
    {
      Counters c;
      auto source = std::make_unique<TaggingSource>(Mode::kRouting, c);
      // The prompt view of the tagged MoE layer re-sets its experts.
      std::vector<std::uint32_t> edited{TaggingSource::moe_layer_};
      auto stages = std::make_unique<ForwardingStages>(
          edited,
          std::vector<fn::TensorId>{
              fn::TensorId::kFfnGateExps, fn::TensorId::kFfnUpExps,
              fn::TensorId::kFfnDownExps, fn::TensorId::kShexpGate,
              fn::TensorId::kShexpUp, fn::TensorId::kShexpDown},
          c);
      const Result r = Run(model, head, std::move(stages), std::move(source));
      float diff = MaxDiff(r.prefill, reference.prefill);
      for (std::size_t i = 0;
           i < std::min(r.decode.size(), reference.decode.size()); ++i)
        diff = std::max(diff, MaxDiff(r.decode[i], reference.decode[i]));
      std::cout << "routing,max_logit_difference," << diff << '\n';
      check(std::equal(r.ar_tokens.begin(), r.ar_tokens.begin() + kDecodeSteps,
                       reference.ar_tokens.begin()),
            "routing,first_ar_tokens");
      check(r.mtp_tokens == r.ar_tokens, "routing,mtp_equals_ar");
      // Output head + q/k/v + qkv/z + three routed + three shared.
      check(c.uploads == 12 && c.draft_heads == 1, "routing,upload_hooks");
      check(c.groups > 0 && c.linear > 0 && c.linear_with_layer == 0,
            "routing,linear_hooks");
      check(c.prompt_layers_used > 0 && c.moe_on_prompt_copy > 0,
            "routing,prompt_layer_hook");
    }
    return passed ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "stage_seam_test: " << failure.what() << '\n';
    return 1;
  }
}
