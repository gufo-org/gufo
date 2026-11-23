#include "src/models/inkling2/weights.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace gufo::models::inkling2 {
namespace {

using Dims = std::vector<std::uint64_t>;

/// What a tensor may be stored as.
enum class Kind {
  kF32,     // small tensors: always F32 in every artifact
  kMatrix,  // any supported type except NVFP4
  kExpert,  // any supported type, NVFP4 included
};

/// Which physical layers carry a tensor.
enum class Where { kAll, kSwa, kGlobal, kNotFirst };

struct Field {
  std::string_view suffix;  // after `blk.N.`
  TensorRef LayerWeights::* member;
  Kind kind;
  Where where;
  Dims (*shape)(const Inkling2Config&, bool is_global);
};

#define SHAPE(...) \
  [](const Inkling2Config& c, bool) -> Dims { return Dims{__VA_ARGS__}; }

// ggml order: ne[0] is the contiguous input dimension.
const Field kFields[] = {
    {"attn_norm.weight", &LayerWeights::attn_norm, Kind::kF32, Where::kSwa,
     SHAPE(c.n_embd)},
    {"attn_q_inp_norm.weight", &LayerWeights::attn_q_inp_norm, Kind::kF32,
     Where::kGlobal, SHAPE(c.n_embd)},
    {"attn_k_inp_norm.weight", &LayerWeights::attn_k_inp_norm, Kind::kF32,
     Where::kGlobal, SHAPE(c.n_embd)},
    {"attn_v_inp_norm.weight", &LayerWeights::attn_v_inp_norm, Kind::kF32,
     Where::kGlobal, SHAPE(c.n_embd)},
    {"attn_q.weight", &LayerWeights::attn_q, Kind::kMatrix, Where::kAll,
     SHAPE(c.n_embd, c.n_head* c.head_dim)},
    {"attn_k.weight", &LayerWeights::attn_k, Kind::kMatrix, Where::kAll,
     SHAPE(c.n_embd, c.n_head_kv* c.head_dim)},
    {"attn_v.weight", &LayerWeights::attn_v, Kind::kMatrix, Where::kAll,
     SHAPE(c.n_embd, c.n_head_kv* c.head_dim)},
    {"attn_gate.weight", &LayerWeights::attn_gate, Kind::kMatrix, Where::kAll,
     SHAPE(c.n_embd, c.n_head* c.head_dim)},
    {"attn_output.weight", &LayerWeights::attn_output, Kind::kMatrix,
     Where::kAll, SHAPE(c.n_head* c.head_dim, c.n_embd)},
    {"attn_rel.weight", &LayerWeights::attn_rel, Kind::kMatrix, Where::kAll,
     SHAPE(c.n_embd, c.rel_dim* c.n_head)},
    {"attn_sink_proj.weight", &LayerWeights::attn_sink_proj, Kind::kMatrix,
     Where::kAll, SHAPE(c.n_embd, c.n_head)},
    {"attn_q_clip.weight", &LayerWeights::attn_q_clip, Kind::kF32, Where::kAll,
     SHAPE(c.head_dim, c.n_head)},
    {"attn_rel_clip.weight", &LayerWeights::attn_rel_clip, Kind::kF32,
     Where::kAll, SHAPE(c.rel_dim, c.n_head)},
    {"attn_rel_bias.weight", &LayerWeights::attn_rel_bias, Kind::kF32,
     Where::kAll, SHAPE(c.rel_dim, c.rel_extent)},
    {"attn_k_conv.weight", &LayerWeights::attn_k_conv, Kind::kF32, Where::kAll,
     SHAPE(c.conv_kernel, c.n_head_kv* c.head_dim)},
    {"attn_v_conv.weight", &LayerWeights::attn_v_conv, Kind::kF32, Where::kAll,
     SHAPE(c.conv_kernel, c.n_head_kv* c.head_dim)},
    {"attn_k_conv_proj.weight", &LayerWeights::attn_k_conv_proj, Kind::kMatrix,
     Where::kGlobal, SHAPE(c.n_embd, c.conv_kernel* c.n_head_kv)},
    {"attn_v_conv_proj.weight", &LayerWeights::attn_v_conv_proj, Kind::kMatrix,
     Where::kGlobal, SHAPE(c.n_embd, c.conv_kernel* c.n_head_kv)},
    {"attn_post_conv.weight", &LayerWeights::attn_post_conv, Kind::kF32,
     Where::kAll, SHAPE(c.conv_kernel, c.n_embd)},
    {"ffn_post_conv.weight", &LayerWeights::ffn_post_conv, Kind::kF32,
     Where::kAll, SHAPE(c.conv_kernel, c.n_embd)},
    // Layer 0 reads the normalised embedding directly; global layers keep one
    // query per q/k/v input.
    {"attn_res_query.weight", &LayerWeights::attn_res_query, Kind::kF32,
     Where::kNotFirst,
     [](const Inkling2Config& c, bool is_global) -> Dims {
       return is_global ? Dims{c.n_embd, 3} : Dims{c.n_embd};
     }},
    {"ffn_res_query.weight", &LayerWeights::ffn_res_query, Kind::kF32,
     Where::kAll, SHAPE(c.n_embd)},
    {"ffn_norm.weight", &LayerWeights::ffn_norm, Kind::kF32, Where::kAll,
     SHAPE(c.n_embd)},
    {"ffn_gate_inp.weight", &LayerWeights::ffn_gate_inp, Kind::kF32,
     Where::kAll, SHAPE(c.n_embd, c.n_expert)},
    {"exp_probs_b.bias", &LayerWeights::exp_probs_b, Kind::kF32, Where::kAll,
     SHAPE(c.n_expert)},
    {"ffn_gate_up_exps.weight", &LayerWeights::ffn_gate_up_exps, Kind::kExpert,
     Where::kAll, SHAPE(c.n_embd, 2 * c.n_ff_expert, c.n_expert)},
    {"ffn_down_exps.weight", &LayerWeights::ffn_down_exps, Kind::kExpert,
     Where::kAll, SHAPE(c.n_ff_expert, c.n_embd, c.n_expert)},
};
// Per-expert scales sit beside the expert tensors.
const Field kScaleFields[] = {
    {"ffn_gate_up_exps.scale", &LayerWeights::ffn_gate_up_scale, Kind::kF32,
     Where::kAll, SHAPE(c.n_expert)},
    {"ffn_down_exps.scale", &LayerWeights::ffn_down_scale, Kind::kF32,
     Where::kAll, SHAPE(c.n_expert)},
};
#undef SHAPE

constexpr std::string_view kStreamNames[kStreams] = {"q", "k", "v", "ffn"};
constexpr std::string_view kLoopParts[] = {"down", "gain", "forget"};

TensorRef LoopController::* PartMember(std::size_t part) {
  constexpr TensorRef LoopController::* members[] = {
      &LoopController::down, &LoopController::gain, &LoopController::forget};
  return members[part];
}

std::string TypeLabel(GgmlType type) {
  const auto name = TypeName(type);
  return name == "UNKNOWN"
             ? "type " + std::to_string(static_cast<unsigned>(type))
             : std::string(name);
}

std::string ShapeText(std::span<const std::uint64_t> ne) {
  std::string text = "[";
  for (const auto d : ne) {
    text += (text.size() > 1 ? ", " : "") + std::to_string(d);
  }
  return text + "]";
}

class Binder {
public:
  explicit Binder(const core::GgufReader& reader) : reader_(reader) {}

  [[nodiscard]] const std::string& why() const { return why_; }

  /// Finds `name`, checks shape and type, fills `out`. A missing tensor is an
  /// error unless `optional`.
  bool Take(TensorRef& out, const std::string& name, Kind kind, const Dims& ne,
            bool optional = false) {
    const auto* info = reader_.FindTensor(name);
    if (info == nullptr) {
      return optional || Fail("missing tensor " + name);
    }
    const auto type = info->type;
    if (kind == Kind::kF32 ? (type != GgmlType::kF32 && type != GgmlType::kBF16)
                           : !IsSupportedType(type) ||
                                 (kind == Kind::kMatrix && type == kNvfp4)) {
      return Fail(
          name + " is " + TypeLabel(type) + ", " +
          (kind == Kind::kF32
               ? std::string("expected F32")
               : "not an allowed " +
                     std::string(kind == Kind::kExpert ? "expert" : "weight") +
                     " type"));
    }
    if (info->dimensions != ne) {
      return Fail(name + " has shape " + ShapeText(info->dimensions) +
                  ", expected " + ShapeText(ne));
    }
    out.type = type;
    out.data = info->data;
    out.size_bytes = info->size_bytes;
    out.n_dims = static_cast<std::uint32_t>(ne.size());
    std::ranges::copy(ne, out.ne.begin());
    if (!info->sized) {
      // Core could not size it; the extension sizes NVFP4 and nothing else.
      const auto bytes = Nvfp4Bytes(info->ElementCount());
      if (type != kNvfp4 || bytes == 0 || bytes > info->size_bytes) {
        return Fail(name + " (" + TypeLabel(type) + ") cannot be sized");
      }
      out.size_bytes = static_cast<std::size_t>(bytes);
    }
    if (kind == Kind::kF32 && type == GgmlType::kBF16) {
      // Small tensors are used as F32 everywhere: widen a BF16 copy once.
      const auto count = static_cast<std::size_t>(info->ElementCount());
      if (info->size_bytes < count * 2)
        return Fail(name + " is smaller than its shape");
      auto f = std::make_shared<std::vector<float>>(count);
      const auto* src = static_cast<const std::uint8_t*>(info->data);
      for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(src[2 * i]) << 16 |
            static_cast<std::uint32_t>(src[2 * i + 1]) << 24;
        std::memcpy(&(*f)[i], &bits, 4);
      }
      out.type = GgmlType::kF32;
      out.data = f->data();
      out.size_bytes = count * 4;
      widened_.push_back(std::move(f));
    }
    claimed_.insert(info->name);
    return true;
  }

  [[nodiscard]] std::vector<std::shared_ptr<const std::vector<float>>>
  TakeWidened() {
    return std::move(widened_);
  }

  /// Reports the first tensor of the file that nothing claimed.
  bool CheckAllClaimed() {
    for (const auto& tensor : reader_.GetTensors()) {
      if (!claimed_.contains(tensor.name)) {
        return Fail("unexpected tensor " + std::string(tensor.name));
      }
    }
    return true;
  }

  bool Fail(std::string message) {
    if (why_.empty()) {
      why_ = std::move(message);
    }
    return false;
  }

private:
  const core::GgufReader& reader_;
  std::unordered_set<std::string_view> claimed_;
  std::vector<std::shared_ptr<const std::vector<float>>> widened_;
  std::string why_;
};

std::string Blk(std::uint32_t il, std::string_view suffix) {
  return "blk." + std::to_string(il) + "." + std::string(suffix);
}

/// Row `row` of a 2-D F32 tensor as its own 1-D tensor.
TensorRef Row(const TensorRef& whole, std::uint64_t row) {
  const auto bytes = whole.ne[0] * sizeof(float);
  TensorRef out = whole;
  out.data = static_cast<const char*>(whole.data) + row * bytes;
  out.size_bytes = static_cast<std::size_t>(bytes);
  out.ne = {whole.ne[0], 1, 1};
  out.n_dims = 1;
  return out;
}

bool BindLayer(Binder& b, const Inkling2Config& c, std::uint32_t il,
               LayerWeights& layer) {
  layer.is_global = c.global_layer[il];
  layer.is_core = c.IsCore(il);
  const auto wanted = [&](Where where) {
    switch (where) {
      case Where::kAll:
        return true;
      case Where::kSwa:
        return !layer.is_global;
      case Where::kGlobal:
        return layer.is_global;
      case Where::kNotFirst:
        return il != 0;
    }
    return false;
  };
  for (const auto& f : kFields) {
    if (wanted(f.where) && !b.Take(layer.*f.member, Blk(il, f.suffix), f.kind,
                                   f.shape(c, layer.is_global))) {
      return false;
    }
  }
  const TensorRef* experts[] = {&layer.ffn_gate_up_exps, &layer.ffn_down_exps};
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& f = kScaleFields[i];
    const bool needed = experts[i]->type == kNvfp4;
    if (!b.Take(layer.*f.member, Blk(il, f.suffix), f.kind, f.shape(c, false),
                !needed)) {
      return false;
    }
  }
  return true;
}

/// Loop controllers and message queries of every core layer.
bool BindLoop(Binder& b, const Inkling2Config& c,
              std::array<LayerWeights, Inkling2Config::kPhysLayers>& layers) {
  const Dims shapes[] = {{2 * c.n_embd, c.gate_rank},
                         {c.gate_rank, c.n_embd},
                         {c.gate_rank, c.n_embd}};
  for (auto il = c.core_start; il < c.core_end; ++il) {
    auto& layer = layers[il];
    for (std::size_t s = 0; s < kStreams; ++s) {
      for (std::size_t part = 0; part < 3; ++part) {
        const auto name = std::string(kStreamNames[s]) + "_ctrl_" +
                          std::string(kLoopParts[part]);
        if (!b.Take(layer.loop[s].*PartMember(part),
                    Blk(il, "loop_" + name + ".weight"), Kind::kF32,
                    shapes[part])) {
          return false;
        }
      }
    }
    TensorRef packed;
    if (!b.Take(packed, Blk(il, "loop_msg_query.weight"), Kind::kF32,
                {c.n_embd, kStreams})) {
      return false;
    }
    for (std::uint64_t s = 0; s < kStreams; ++s) {
      layer.msg_query[s] = Row(packed, s);
    }
  }
  return true;
}

}  // namespace

std::optional<Inkling2Weights> Inkling2Weights::Bind(
    const core::GgufReader& reader, std::string* why) {
  Inkling2Weights w;
  Binder b(reader);
  const auto report = [&]() -> std::optional<Inkling2Weights> {
    if (why != nullptr) {
      *why = b.why();
    }
    return std::nullopt;
  };
  std::string config_why;
  if (!ParseConfig(reader, &w.config, &config_why)) {
    b.Fail(config_why);
    return report();
  }
  const auto& c = w.config;
  const bool ok =
      b.Take(w.token_embd, "token_embd.weight", Kind::kMatrix,
             {c.n_embd, c.n_vocab}) &&
      b.Take(w.token_embd_norm, "token_embd_norm.weight", Kind::kF32,
             {c.n_embd}) &&
      b.Take(w.output, "output.weight", Kind::kMatrix, {c.n_embd, c.n_vocab}) &&
      b.Take(w.output_norm, "output_norm.weight", Kind::kF32, {c.n_embd}) &&
      b.Take(w.output_res_query, "output_res_query.weight", Kind::kF32,
             {c.n_embd}) &&
      [&] {
        for (std::uint32_t il = 0; il < c.n_phys; ++il) {
          if (!BindLayer(b, c, il, w.layers[il])) {
            return false;
          }
        }
        return true;
      }() &&
      BindLoop(b, c, w.layers) && b.CheckAllClaimed();
  if (!ok) {
    return report();
  }
  w.widened = b.TakeWidened();
  return w;
}

void ForEachTensor(
    const Inkling2Weights& w,
    const std::function<void(const std::string&, const TensorRef&)>& fn) {
  const std::pair<const char*, const TensorRef*> globals[] = {
      {"token_embd.weight", &w.token_embd},
      {"token_embd_norm.weight", &w.token_embd_norm},
      {"output.weight", &w.output},
      {"output_norm.weight", &w.output_norm},
      {"output_res_query.weight", &w.output_res_query}};
  for (const auto& [name, tensor] : globals) {
    fn(name, *tensor);
  }
  for (std::uint32_t il = 0; il < w.layers.size(); ++il) {
    const auto& layer = w.layers[il];
    for (const auto fields : {std::span<const Field>(kFields),
                              std::span<const Field>(kScaleFields)}) {
      for (const auto& f : fields) {
        if (layer.*f.member) {
          fn(Blk(il, f.suffix), layer.*f.member);
        }
      }
    }
    for (std::size_t s = 0; s < kStreams && layer.is_core; ++s) {
      fn(Blk(il, "msg_query." + std::string(kStreamNames[s])),
         layer.msg_query[s]);
      for (std::size_t part = 0; part < 3; ++part) {
        fn(Blk(il, "loop." + std::string(kStreamNames[s]) + "." +
                       std::string(kLoopParts[part])),
           layer.loop[s].*PartMember(part));
      }
    }
  }
}

std::string Summarize(const Inkling2Weights& w) {
  struct Total {
    std::size_t count = 0;
    std::uint64_t bytes = 0;
  };
  std::map<std::string, Total> by_type;
  std::size_t tensors = 0;
  ForEachTensor(w, [&](const std::string&, const TensorRef& t) {
    auto& total = by_type[TypeLabel(t.type)];
    ++total.count;
    total.bytes += t.size_bytes;
    ++tensors;
  });
  std::ostringstream out;
  out << "layout inkling2_loop (24 blocks), " << tensors << " tensors\n";
  for (const auto& [type, total] : by_type) {
    out << "  " << type << ": " << total.count << " tensors, "
        << total.bytes / (1024.0 * 1024.0) << " MiB\n";
  }
  // Expert types per layer, as runs: "L0-19 NVFP4/NVFP4, L20-23 Q8_0/Q8_0".
  out << "  experts (gate_up/down):";
  for (std::size_t il = 0; il < w.layers.size();) {
    const auto& layer = w.layers[il];
    const auto label = TypeLabel(layer.ffn_gate_up_exps.type) + "/" +
                       TypeLabel(layer.ffn_down_exps.type);
    auto end = il;
    while (end + 1 < w.layers.size() &&
           TypeLabel(w.layers[end + 1].ffn_gate_up_exps.type) + "/" +
                   TypeLabel(w.layers[end + 1].ffn_down_exps.type) ==
               label) {
      ++end;
    }
    out << (il ? "," : "") << " L" << il << "-" << end << " " << label;
    il = end + 1;
  }
  out << "\n";
  return out.str();
}

}  // namespace gufo::models::inkling2
