#include "src/models/qwen38_flash_next/probe.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/gyro/tq_types.hpp"

namespace {

namespace gyro = gufo::models::qwen38_flash_next::gyro;
using gufo::core::GgmlType;
using Tensor = std::pair<std::string, GgmlType>;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Independent GGUF fixture: each tensor has one 256-element row and enough
// payload for every tested storage format. The probe never reads its values.
std::vector<std::uint8_t> Fixture(const std::vector<Tensor>& tensors,
                                  std::optional<std::uint32_t> blocks,
                                  std::optional<std::uint32_t> nextn,
                                  const std::string& architecture) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{1} + blocks.has_value() + nextn.has_value());
  text("general.architecture");
  pod(std::uint32_t{8});
  text(architecture);
  for (const auto& [key, value] :
       {std::pair{"qwen4exp.block_count", blocks},
        std::pair{"qwen4exp.nextn_predict_layers", nextn}}) {
    if (value) {
      text(key);
      pod(std::uint32_t{4});
      pod(*value);
    }
  }
  std::uint64_t offset = 0;
  for (const auto& [name, type] : tensors) {
    text(name);
    pod(std::uint32_t{1});
    pod(std::uint64_t{256});
    pod(static_cast<std::uint32_t>(type));
    pod(offset);
    offset += 2048;
  }
  bytes.resize((bytes.size() + 31) / 32 * 32 + offset);
  return bytes;
}

// The text commands open models allowing tensors core cannot size, so both
// providers see the same bytes: one declines a layout it does not own rather
// than reading it with the wrong stride.
bool Probe(const std::vector<Tensor>& tensors,
           std::optional<std::uint32_t> blocks = 49,
           std::optional<std::uint32_t> nextn = 1,
           const std::string& architecture = "qwen4exp") {
  const auto bytes = Fixture(tensors, blocks, nextn, architecture);
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      bytes.data(), bytes.size(), &error, {.allow_unsized_tensors = true});
  Require(reader != nullptr, error.c_str());
  return gufo::models::qwen38_flash_next::ProbeFlashNextGguf(*reader);
}

bool GyroProbe(const std::vector<Tensor>& tensors,
               std::optional<std::uint32_t> blocks = 49,
               std::optional<std::uint32_t> nextn = 1) {
  const auto bytes = Fixture(tensors, blocks, nextn, "qwen4exp");
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      bytes.data(), bytes.size(), &error, {.allow_unsized_tensors = true});
  Require(reader != nullptr, error.c_str());
  return gufo::models::qwen38_flash_next::ProbeFlashNextGyroGguf(*reader);
}

void CheckExpertFormats() {
  for (const auto* role : {"gate", "up", "down"}) {
    const auto name = std::string("blk.47.ffn_") + role + "_exps.weight";
    for (const auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                            GgmlType::kQ5_1, GgmlType::kQ8_0}) {
      Require(Probe({{name, type}}), "supported expert format declined");
    }
    for (const auto type :
         {GgmlType::kIQ3_S, GgmlType::kIQ4_XS, GgmlType::kIQ4_NL,
          GgmlType::kF16, GgmlType::kQ2_K}) {
      Require(!Probe({{name, type}}), "unsupported trunk expert claimed");
    }
  }
  Require(!Probe({{"blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K},
                  {"blk.47.ffn_down_exps.weight", GgmlType::kIQ3_S}}),
          "unsupported later expert missed after a supported expert");
}

void CheckIgnoredTensors() {
  const std::vector<Tensor> trunk = {
      {"blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K},
      {"blk.47.ffn_down_exps.weight", GgmlType::kQ8_0}};
  for (const auto* name :
       {"blk.48.ffn_gate_exps.weight",  // Appended MTP layer.
        "blk.49.ffn_up_exps.weight",    // Beyond the model's layer count.
        "blk.4294967296.ffn_gate_exps.weight",
        "blk.00.ffn_gate_exps.weight",  // Noncanonical layer spelling.
        "blk.-1.ffn_gate_exps.weight", "blk.+0.ffn_gate_exps.weight",
        "blk..ffn_gate_exps.weight", "blk.", "other.blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_gate_exps.weight.extra", "blk.0.extra.ffn_gate_exps.weight",
        "per_layer_token_embd.weight"}) {
    auto tensors = trunk;
    tensors.emplace_back(name, GgmlType::kIQ4_NL);
    Require(Probe(tensors), "unused tensor incorrectly declined the trunk");
  }
  Require(Probe({{"blk.47.ffn_up_exps.weight", GgmlType::kIQ3_S}}, 48, 1),
          "appended layer was not excluded from block_count");
  Require(!Probe({{"blk.47.ffn_up_exps.weight", GgmlType::kIQ3_S}}, 48,
                 std::nullopt),
          "absent nextn metadata incorrectly excluded the final trunk layer");
}

// Gyro-M replaces every trunk expert tensor with TQ2_T trellis rows, a
// storage type core cannot size. The providers are complements on the trunk
// experts: a partly trellised file belongs to neither.
void CheckGyroExperts() {
  for (const auto* role : {"gate", "up", "down"}) {
    const auto name = std::string("blk.47.ffn_") + role + "_exps.weight";
    Require(!Probe({{name, gyro::kTQ2T}}), "trellis expert claimed by q4");
    Require(GyroProbe({{name, gyro::kTQ2T}}), "trellis expert declined");
    Require(!GyroProbe({{name, GgmlType::kQ4_K}}),
            "quantized expert claimed by the trellis provider");
  }
  Require(!GyroProbe({{"blk.0.ffn_gate_exps.weight", gyro::kTQ2T},
                      {"blk.47.ffn_down_exps.weight", GgmlType::kQ4_K}}),
          "half-trellised trunk claimed");
  // Appended MTP layers and unrelated tensors follow the trunk, as for q4.
  Require(GyroProbe({{"blk.0.ffn_gate_exps.weight", gyro::kTQ2T},
                     {"blk.48.ffn_gate_exps.weight", GgmlType::kQ4_K}}),
          "appended layer declined the trellis trunk");
  Require(!GyroProbe({}), "trellis probe claimed a file without experts");
}

void CheckMetadata() {
  Require(!Probe({}, 49, 1, "qwen3"), "unrelated architecture claimed");
  Require(!Probe({}, std::nullopt), "missing block_count claimed");
  Require(!Probe({}, 0, 0), "zero block_count claimed");
  Require(!Probe({}, 48, 48), "empty trunk claimed");
  Require(!Probe({}, 48, 49), "nextn count underflowed");
  // The probe chooses a provider; the loader still diagnoses missing tensors.
  Require(Probe({}), "probe unexpectedly required complete weights");
}

}  // namespace

int main() {
  CheckExpertFormats();
  CheckGyroExperts();
  CheckIgnoredTensors();
  CheckMetadata();
  std::cout << "PASS: Flash-Next provider probe\n";
}
