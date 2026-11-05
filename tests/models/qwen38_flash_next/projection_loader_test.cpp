#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/projection.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void Put(std::string* out, const void* data, std::size_t bytes) {
  out->append(static_cast<const char*>(data), bytes);
}
void PutU32(std::string* out, std::uint32_t v) {
  Put(out, &v, 4);
}
void PutU64(std::string* out, std::uint64_t v) {
  Put(out, &v, 8);
}
void PutString(std::string* out, const std::string& s) {
  // GGUF strings carry a uint64 length prefix.
  PutU64(out, s.size());
  Put(out, s.data(), s.size());
}

struct TensorSpec {
  std::string name;
  std::vector<float> data;
  std::uint32_t ggml_type = 0;  // GGML_F32
};

// A minimal single-file GGUF v3 writer covering exactly what GgufReader
// parses for this test: 1-D tensors at the default 32-byte alignment and an
// optional controlvector.model_hint string.
std::string WriteGguf(const std::vector<TensorSpec>& tensors,
                      const std::string& model_hint = {}) {
  std::string data;
  std::string infos;
  for (const auto& t : tensors) {
    if (!data.empty()) {
      data.resize((data.size() + 31) / 32 * 32, '\0');
    }
    const std::uint64_t offset = data.size();
    PutString(&infos, t.name);
    PutU32(&infos, 1);  // n_dims
    PutU64(&infos, t.data.size());
    PutU32(&infos, t.ggml_type);
    PutU64(&infos, offset);
    Put(&data, t.data.data(), t.data.size() * sizeof(float));
  }
  std::string out;
  Put(&out, "GGUF", 4);
  PutU32(&out, 3);
  PutU64(&out, tensors.size());
  PutU64(&out, model_hint.empty() ? 0 : 1);
  if (!model_hint.empty()) {
    PutString(&out, "controlvector.model_hint");
    PutU32(&out, 8);  // string
    PutString(&out, model_hint);
  }
  out += infos;
  out.resize((out.size() + 31) / 32 * 32, '\0');
  out += data;
  return out;
}

std::filesystem::path TempFile(const std::string& blob, const char* tag) {
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string("gufo_projection_") + tag + ".gguf");
  std::ofstream file(path, std::ios::binary);
  file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  return path;
}

void TestLoadAndNormalize() {
  constexpr std::uint32_t kHidden = 64;
  std::vector<float> e0(kHidden, 0.0F);
  e0[0] = 3.0F;                            // norm 3, unit direction (1, 0, ...)
  std::vector<float> half(kHidden, 0.5F);  // norm 4
  std::vector<float> e1(kHidden, 0.0F);
  e1[1] = -2.0F;  // norm 2, unit direction (0, -1, 0, ...)
  const std::vector<TensorSpec> tensors = {
      {"direction.2", e0, 0},
      {"direction.3", half, 0},
      {"direction.4", std::vector<float>(kHidden, 0.0F), 0},  // zero: skip
      {"direction.5", e1, 0},
  };
  const auto path = TempFile(WriteGguf(tensors), "basic");
  qfn::ProjectionVector v;
  std::string error;
  std::string warning;
  const bool loaded =
      qfn::LoadControlVector(path.string(), /*num_layers=*/8, kHidden,
                             "qwen4exp", &v, &error, &warning);
  Check(loaded, "load succeeds" + (error.empty() ? "" : ": " + error));
  if (!loaded) {
    return;  // the checks below index the loaded tables
  }
  Check(error.empty(), "no error");
  Check(warning.empty(), "no warning without model_hint");
  Check(v.num_layers == 8 && v.hidden == kHidden, "geometry recorded");
  Check(v.first_layer == 1 && v.last_layer == 7, "full carried range");
  Check(v.scales.size() == 8, "one scale per layer");
  Check(v.scales[1] == 0.0F && v.scales[6] == 0.0F && v.scales[7] == 0.0F,
        "missing layers unsteered");
  Check(v.scales[2] == 3.0F, "scale is the normalized-away norm (3)");
  Check(v.scales[3] == 4.0F, "scale is the normalized-away norm (4)");
  Check(v.scales[4] == 0.0F, "a zero direction stays unsteered");
  Check(v.scales[5] == 2.0F, "scale is the normalized-away norm (2)");
  const float* row2 = v.directions.data() + 2 * kHidden;
  Check(row2[0] == 1.0F, "direction.2 normalized to e0");
  bool row2_tail_zero = true;
  for (std::uint32_t i = 1; i < kHidden; ++i) {
    row2_tail_zero = row2_tail_zero && row2[i] == 0.0F;
  }
  Check(row2_tail_zero, "direction.2 tail is zero");
  const float* row5 = v.directions.data() + 5 * kHidden;
  Check(row5[1] == -1.0F, "direction.5 normalized to -e1");
  const float* row3 = v.directions.data() + 3 * kHidden;
  Check(row3[7] == 0.125F, "direction.3 normalized to 0.5/4");
}

void TestHintWarning() {
  constexpr std::uint32_t kHidden = 16;
  std::vector<float> d(kHidden, 2.0F);  // norm 8
  const auto mismatch = TempFile(
      WriteGguf({{"direction.3", d, 0}}, "some-other-model"), "hint_mismatch");
  qfn::ProjectionVector v;
  std::string error;
  std::string warning;
  Check(qfn::LoadControlVector(mismatch.string(), 8, kHidden, "qwen4exp", &v,
                               &error, &warning),
        "mismatched hint still loads");
  Check(!warning.empty(), "mismatched hint warns");
  std::filesystem::remove(mismatch);

  const auto match =
      TempFile(WriteGguf({{"direction.3", d, 0}}, "qwen4exp"), "hint_match");
  warning.clear();
  Check(qfn::LoadControlVector(match.string(), 8, kHidden, "qwen4exp", &v,
                               &error, &warning),
        "matching hint loads");
  Check(warning.empty(), "matching hint is silent");
  std::filesystem::remove(match);
}

void TestRejections() {
  constexpr std::uint32_t kHidden = 16;
  std::vector<float> d(kHidden, 1.0F);
  qfn::ProjectionVector v;
  std::string error;

  Check(!qfn::LoadControlVector("/nonexistent/orcarouter.gguf", 8, kHidden,
                                "qwen4exp", &v, &error),
        "missing file rejected");
  Check(error.find("cannot open") != std::string::npos,
        "missing file names the path");

  const auto wrong_hidden =
      TempFile(WriteGguf({{"direction.3", d, 0}}), "wrong_hidden");
  error.clear();
  Check(!qfn::LoadControlVector(wrong_hidden.string(), 8, kHidden + 1,
                                "qwen4exp", &v, &error),
        "wrong hidden size rejected");
  Check(!error.empty(), "wrong hidden size explains itself");
  std::filesystem::remove(wrong_hidden);

  // F16 payload dressed as a direction: same byte size, wrong type.
  std::vector<float> raw(kHidden, 1.0F);
  const auto wrong_type =
      TempFile(WriteGguf({TensorSpec{"direction.3", raw, /*ggml_type=*/1}}),
               "wrong_type");
  error.clear();
  Check(!qfn::LoadControlVector(wrong_type.string(), 8, kHidden, "qwen4exp", &v,
                                &error),
        "non-F32 direction rejected");
  std::filesystem::remove(wrong_type);

  const auto empty = TempFile(WriteGguf({}), "empty");
  error.clear();
  Check(!qfn::LoadControlVector(empty.string(), 8, kHidden, "qwen4exp", &v,
                                &error),
        "no direction tensors rejected");
  std::filesystem::remove(empty);
}

}  // namespace

int main() {
  TestLoadAndNormalize();
  TestHintWarning();
  TestRejections();
  if (failures != 0) {
    std::printf("%d projection loader check(s) failed\n", failures);
    return 1;
  }
  std::printf("projection loader checks passed\n");
  return 0;
}
