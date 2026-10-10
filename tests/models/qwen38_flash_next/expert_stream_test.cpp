#include "src/models/qwen38_flash_next/expert_stream.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckByteSize() {
  Require(qfn::ParseByteSize("0") == 0, "0 rejected");
  Require(qfn::ParseByteSize("1024") == 1024, "plain bytes rejected");
  Require(qfn::ParseByteSize("1024b") == 1024, "trailing b rejected");
  Require(qfn::ParseByteSize("4G") == 4ULL << 30, "4G rejected");
  Require(qfn::ParseByteSize("4g") == 4ULL << 30, "lowercase 4g rejected");
  Require(qfn::ParseByteSize("512M") == 512ULL << 20, "512M rejected");
  Require(qfn::ParseByteSize("512 MiB") == 512ULL << 20, "spaced MiB rejected");
  Require(qfn::ParseByteSize("8 kib") == 8ULL << 10,
          "case-insensitive KiB rejected");
  Require(!qfn::ParseByteSize(""), "empty accepted");
  Require(!qfn::ParseByteSize("  "), "whitespace accepted");
  Require(!qfn::ParseByteSize("-1"), "negative accepted");
  Require(!qfn::ParseByteSize("1.5G"), "fractional accepted");
  Require(!qfn::ParseByteSize("G"), "unit without value accepted");
  Require(!qfn::ParseByteSize("1T"), "unknown unit accepted");
  Require(!qfn::ParseByteSize("1BB"), "double unit accepted");
  Require(!qfn::ParseByteSize("0x10"), "hex accepted");
  Require(!qfn::ParseByteSize("99999999999999999999999"), "overflow accepted");
  Require(!qfn::ParseByteSize("8G9"), "digits after unit accepted");
}

qfn::TensorRef Expert(GgmlType type, std::uint64_t cols, std::uint64_t rows,
                      std::uint64_t experts) {
  qfn::TensorRef t;
  // A non-null sentinel marks the tensor present; the planner only reads
  // shape and format.
  t.data = reinterpret_cast<const void*>(1);
  t.type = type;
  t.cols = cols;
  t.rows = rows;
  t.experts = experts;
  return t;
}

/// A Qwen3.8-Flash-Next-shaped layer: hidden 2560, expert_ff 640, 512
/// experts; the artifact's dominant class is gate/up Q4_K with down Q8_0,
/// and the UD outliers quantize one class differently.
qfn::LayerWeights Layer(GgmlType gate_up, GgmlType down) {
  qfn::LayerWeights l;
  l.ffn_gate_exps = Expert(gate_up, 2560, 640, 512);
  l.ffn_up_exps = Expert(gate_up, 2560, 640, 512);
  l.ffn_down_exps = Expert(down, 640, 2560, 512);
  return l;
}

void CheckPlan() {
  const auto q4k = GgmlType::kQ4_K;
  const auto q8 = GgmlType::kQ8_0;
  const auto q5_1 = GgmlType::kQ5_1;

  // Budget 0: disabled, no error.
  {
    std::vector<qfn::LayerWeights> layers(48, Layer(q4k, q8));
    std::string error;
    const auto plan =
        qfn::PlanExpertStreaming(layers, nullptr, 512, 10, 0, &error);
    Require(!plan.enabled(), "zero budget enabled streaming");
    Require(error.empty(), "zero budget reported an error");
  }
  // Uniform artifact: every layer streams; slots = budget / expert bytes /
  // layers.
  {
    std::vector<qfn::LayerWeights> layers(48, Layer(q4k, q5_1));
    const std::size_t expert_bytes = qfn::ExpertBytes(layers[0]);
    Require(expert_bytes > 0, "expert byte size lost");
    const std::size_t budget = expert_bytes * 48 * 20;  // 20 slots/layer
    std::string error;
    const auto plan =
        qfn::PlanExpertStreaming(layers, nullptr, 512, 10, budget, &error);
    Require(plan.enabled(), "uniform artifact did not stream");
    Require(plan.layers == 48, "streamed layer count wrong");
    Require(plan.slots_per_layer == 20, "slot sizing wrong");
    Require(plan.resident_expert_bytes == 0, "resident bytes reported");
    Require(error.empty(), "uniform plan reported an error");
  }
  // Mixed artifact: the dominant class streams, outliers stay resident.
  {
    std::vector<qfn::LayerWeights> layers(46, Layer(q4k, q5_1));
    layers.push_back(Layer(q8, q8));
    layers.push_back(Layer(q8, q8));
    const std::size_t outlier = layers[46].ffn_gate_exps.SizeBytes() +
                                layers[46].ffn_up_exps.SizeBytes() +
                                layers[46].ffn_down_exps.SizeBytes();
    std::string error;
    const auto plan =
        qfn::PlanExpertStreaming(layers, nullptr, 512, 10,
                                 qfn::ExpertBytes(layers[0]) * 46 * 16, &error);
    Require(plan.enabled(), "dominant class did not stream");
    Require(plan.layers == 46, "outlier layers joined the stream");
    Require(plan.resident_expert_bytes == 2 * outlier,
            "resident bytes for outliers wrong");
  }
  // A same-class MTP block joins the stream; a foreign class does not.
  {
    std::vector<qfn::LayerWeights> layers(4, Layer(q4k, q5_1));
    const auto mtp_same = Layer(q4k, q5_1);
    const auto mtp_other = Layer(q8, q8);
    std::string error;
    const auto same =
        qfn::PlanExpertStreaming(layers, &mtp_same, 512, 10,
                                 qfn::ExpertBytes(mtp_same) * 5 * 12, &error);
    Require(same.enabled(), "same-class MTP plan rejected");
    Require(same.layers == 5, "same-class MTP did not join the stream");
    const auto other =
        qfn::PlanExpertStreaming(layers, &mtp_other, 512, 10,
                                 qfn::ExpertBytes(layers[0]) * 4 * 12, &error);
    Require(other.layers == 4, "foreign MTP class streamed");
    Require(
        other.resident_expert_bytes == mtp_other.ffn_gate_exps.SizeBytes() +
                                           mtp_other.ffn_up_exps.SizeBytes() +
                                           mtp_other.ffn_down_exps.SizeBytes(),
        "foreign MTP not counted resident");
  }
  // Budget below one expert per streamed layer: rejected with a message.
  {
    std::vector<qfn::LayerWeights> layers(48, Layer(q4k, q5_1));
    std::string error;
    const auto plan = qfn::PlanExpertStreaming(
        layers, nullptr, 512, 10, qfn::ExpertBytes(layers[0]) * 47, &error);
    Require(!plan.enabled(), "sub-one-expert budget accepted");
    Require(!error.empty(), "sub-one-expert budget passed silently");
  }
  // Budget fits experts per layer but not one token's routed set (10):
  // rejected — the group planner cannot place a token.
  {
    std::vector<qfn::LayerWeights> layers(48, Layer(q4k, q5_1));
    std::string error;
    const auto plan = qfn::PlanExpertStreaming(
        layers, nullptr, 512, 10, qfn::ExpertBytes(layers[0]) * 48 * 9, &error);
    Require(!plan.enabled(), "nine-slot budget accepted for ten routed");
    Require(!error.empty(), "undersized group accepted silently");
  }
  // A budget covering all 512 experts per layer is rejected: full
  // residency is the cheaper plan at that size.
  {
    std::vector<qfn::LayerWeights> layers(48, Layer(q4k, q5_1));
    std::string error;
    const auto plan = qfn::PlanExpertStreaming(
        layers, nullptr, 512, 10, qfn::ExpertBytes(layers[0]) * 48 * 600,
        &error);
    Require(!plan.enabled(), "all-covering budget streamed");
    Require(!error.empty(), "all-covering budget passed silently");
  }
  // No streamable formats: rejected.
  {
    std::vector<qfn::LayerWeights> layers(
        4, Layer(GgmlType::kF32, GgmlType::kF32));
    std::string error;
    const auto plan =
        qfn::PlanExpertStreaming(layers, nullptr, 512, 10, 1 << 30, &error);
    Require(!plan.enabled(), "unsupported formats streamed");
    Require(!error.empty(), "unsupported formats passed silently");
  }
}

}  // namespace

int main() {
  CheckByteSize();
  CheckPlan();
  return 0;
}
