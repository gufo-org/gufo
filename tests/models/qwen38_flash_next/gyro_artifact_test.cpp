// The trellis provider against a real Gyro-M artifact: the probe claims it,
// flash-next-q4 declines it, and the trunk binds with TQ2_T experts and a
// validated block-Hadamard rotation. Only GGUF headers and tensor extents are
// read, so this needs neither a GPU nor a weight upload. It is skipped when
// QWEN38_FLASH_NEXT_GYRO_MODEL is unset.

#include <cstdlib>
#include <iostream>
#include <string>

#include <gufo/ext/gguf_reader.hpp>

#include "src/models/qwen38_flash_next/gyro/tq_types.hpp"
#include "src/models/qwen38_flash_next/probe.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace {

namespace gyro = gufo::models::qwen38_flash_next::gyro;
using gufo::core::GgufReader;
using gufo::models::qwen38_flash_next::Config;
using gufo::models::qwen38_flash_next::ModelWeights;
using gufo::models::qwen38_flash_next::ProbeFlashNextGguf;
using gufo::models::qwen38_flash_next::ProbeFlashNextGyroGguf;
using gufo::models::qwen38_flash_next::TensorRef;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// A trellis tensor is stored whole in 128-element blocks, and its rotation
/// vector must be the one for its reduction dimension: the kernels rotate the
/// activation rows, so a mismatched vector would silently change the product.
void CheckTrellis(const gufo::models::qwen38_flash_next::HadamardSigns& signs,
                  const TensorRef& t, const char* what) {
  const std::string name = std::string(what) + " " + std::string(t.name);
  Require(!t.empty(), name + " is missing");
  Require(gyro::IsTQ2T(t.type), name + " is not TQ2_T");
  Require(t.cols % gyro::kBlock == 0, name + " columns are not block aligned");
  Require(t.hadamard_signs != nullptr, name + " has no sign vector");
  Require(t.hadamard_signs == signs.For(t.cols),
          name + " sign vector is not the one for its columns");
  Require(gyro::EncodedSizeBytes(t.type, t.rows * t.experts * t.cols) != 0,
          name + " has no trellis row size");
}

}  // namespace

int main() {
  const char* path = std::getenv("QWEN38_FLASH_NEXT_GYRO_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cerr << "QWEN38_FLASH_NEXT_GYRO_MODEL is not set; skipping\n";
    return 77;
  }
  try {
    std::string error;
    const auto reader = GgufReader::OpenFile(
        path, &error, {.allow_unsized_tensors = true});
    Require(reader != nullptr, "open " + std::string(path) + ": " + error);
    Require(ProbeFlashNextGyroGguf(*reader),
            "the trellis provider declined its own artifact");
    Require(!ProbeFlashNextGguf(*reader),
            "flash-next-q4 claimed a trellis artifact");

    const auto config = Config::FromGguf(*reader, true, &error);
    Require(config.has_value(), error);

    const auto weights = ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), error);
    Require(weights->hadamard != nullptr,
            "the artifact stores no block-Hadamard rotation");
    Require(weights->layers.size() == config->num_layers,
            "trunk layer count changed");

    std::size_t rotated = 0;
    for (const auto& layer : weights->layers) {
      CheckTrellis(*weights->hadamard, layer.ffn_gate_exps, "gate");
      CheckTrellis(*weights->hadamard, layer.ffn_up_exps, "up");
      CheckTrellis(*weights->hadamard, layer.ffn_down_exps, "down");
      ++rotated;
    }
    std::cout << "PASS: Gyro-M artifact binds " << rotated
              << " trellis expert layers (" << config->num_experts
              << " experts, " << config->num_experts_used << " used, hidden "
              << config->hidden_size << ", ffn " << config->expert_ff << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
