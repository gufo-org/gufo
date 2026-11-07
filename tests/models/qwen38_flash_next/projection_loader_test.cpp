#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
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

// One directory per run, so concurrent build trees never share fixtures.
const std::filesystem::path& TempDir() {
  static const std::filesystem::path dir = [] {
    auto path =
        std::filesystem::temp_directory_path() /
        ("gufo_projection_test_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(path);
    return path;
  }();
  return dir;
}

std::filesystem::path TempFile(const std::string& blob, const char* tag) {
  const auto path = TempDir() / (std::string(tag) + ".gguf");
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
  Check(warning.find("model_hint") == std::string::npos,
        "no hint warning without model_hint");
  Check(warning.find("non-unit directions at layers 2, 3, 5") !=
            std::string::npos,
        "non-unit norms warn: " + warning);
  Check(warning.find("ignoring direction.4") != std::string::npos,
        "a zero direction warns: " + warning);
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
  std::filesystem::remove(path);
}

std::vector<float> Unit(std::uint32_t hidden, std::uint32_t axis) {
  std::vector<float> d(hidden, 0.0F);
  d[axis] = 1.0F;
  return d;
}

void TestIgnoredTensors() {
  constexpr std::uint32_t kHidden = 16;
  const auto path = TempFile(WriteGguf({
                                 {"direction.0", Unit(kHidden, 0), 0},
                                 {"direction.3", Unit(kHidden, 1), 0},
                                 {"direction.8", Unit(kHidden, 2), 0},
                             }),
                             "ignored");
  qfn::ProjectionVector v;
  std::string error;
  std::string warning;
  Check(qfn::LoadControlVector(path.string(), 8, kHidden, "qwen4exp", &v,
                               &error, &warning),
        "out-of-range directions still load the steerable ones");
  Check(warning.find("ignoring direction.0") != std::string::npos,
        "direction.0 warns: " + warning);
  Check(warning.find("ignoring direction.8") != std::string::npos,
        "direction.N past the last layer warns: " + warning);
  Check(warning.find("non-unit") == std::string::npos,
        "unit directions do not warn about their norm");
  Check(v.scales.size() == 8 && v.scales[3] == 1.0F, "direction.3 loads");
  std::filesystem::remove(path);
}

void TestSelectLayers() {
  constexpr std::uint32_t kHidden = 16;
  const auto path =
      TempFile(WriteGguf({{"direction.3", Unit(kHidden, 0), 0}}), "layers");
  qfn::ProjectionVector v;
  std::string error;
  Check(
      qfn::LoadControlVector(path.string(), 8, kHidden, "qwen4exp", &v, &error),
      "layer fixture loads");
  std::filesystem::remove(path);
  Check(qfn::SelectProjectionLayers("2-5", &v, &error) && v.first_layer == 2 &&
            v.last_layer == 5,
        "explicit range selected");
  Check(qfn::SelectProjectionLayers("7-7", &v, &error) && v.first_layer == 7 &&
            v.last_layer == 7,
        "single last layer selected");
  Check(qfn::SelectProjectionLayers("", &v, &error) && v.first_layer == 1 &&
            v.last_layer == 7,
        "empty spec selects the full range");
  for (const char* bad : {"0-3", "3-2", "1-8", "5-0", "2-5x", "-1-5", "4",
                          "a-b", "2-", "-3", " 2-5"}) {
    error.clear();
    Check(!qfn::SelectProjectionLayers(bad, &v, &error) && !error.empty(),
          std::string("range '") + bad + "' rejected");
  }
  Check(v.first_layer == 1 && v.last_layer == 7,
        "a rejected range leaves the selection unchanged");
  qfn::ProjectionVector unloaded;
  Check(!qfn::SelectProjectionLayers("", &unloaded, &error),
        "selection needs a loaded vector");
}

void TestFingerprint() {
  constexpr std::uint32_t kHidden = 16;
  const auto a =
      TempFile(WriteGguf({{"direction.3", Unit(kHidden, 0), 0}}), "print_a");
  const auto b =
      TempFile(WriteGguf({{"direction.3", Unit(kHidden, 1), 0}}), "print_b");
  qfn::ProjectionVector va;
  qfn::ProjectionVector vb;
  std::string error;
  Check(
      qfn::LoadControlVector(a.string(), 8, kHidden, "qwen4exp", &va, &error) &&
          qfn::LoadControlVector(b.string(), 8, kHidden, "qwen4exp", &vb,
                                 &error),
      "fingerprint fixtures load");
  std::filesystem::remove(a);
  std::filesystem::remove(b);
  const std::string full = qfn::ProjectionFingerprint(va);
  Check(full.size() == 64, "fingerprint is a SHA-256 hex digest");
  Check(full == qfn::ProjectionFingerprint(va), "fingerprint is stable");
  Check(full != qfn::ProjectionFingerprint(vb), "directions change it");
  Check(qfn::SelectProjectionLayers("3-3", &va, &error) &&
            full != qfn::ProjectionFingerprint(va),
        "the steered range changes it");
}

void TestHintWarning() {
  constexpr std::uint32_t kHidden = 16;
  const std::vector<float> d(kHidden, 0.25F);  // unit norm
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

  Check(!qfn::LoadControlVector("/nonexistent/projection.gguf", 8, kHidden,
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

  // Non-finite values and an F32-overflowing norm fail the load instead of
  // steering with NaN; the good layer beside them must not mask the error.
  const float inf = std::numeric_limits<float>::infinity();
  const float max = std::numeric_limits<float>::max();
  const struct {
    const char* tag;
    float value;
  } corrupt[] = {{"inf", inf},
                 {"neg_inf", -inf},
                 {"nan", std::numeric_limits<float>::quiet_NaN()},
                 {"overflow", max}};
  for (const auto& c : corrupt) {
    std::vector<float> bad(kHidden, c.value);
    const auto path = TempFile(
        WriteGguf({{"direction.2", d, 0}, {"direction.3", bad, 0}}), c.tag);
    error.clear();
    Check(!qfn::LoadControlVector(path.string(), 8, kHidden, "qwen4exp", &v,
                                  &error) &&
              error.find("direction.3") != std::string::npos,
          std::string("corrupt direction rejected: ") + c.tag + " (" + error +
              ")");
    std::filesystem::remove(path);
  }

  // A subnormal direction is valid: its F32 reciprocal would overflow, so it
  // must still normalize to a finite unit row.
  std::vector<float> tiny(kHidden, 0.0F);
  tiny[5] = std::numeric_limits<float>::denorm_min();
  const auto tiny_path =
      TempFile(WriteGguf({{"direction.3", tiny, 0}}), "tiny");
  error.clear();
  Check(qfn::LoadControlVector(tiny_path.string(), 8, kHidden, "qwen4exp", &v,
                               &error) &&
            v.directions[3 * kHidden + 5] == 1.0F,
        "subnormal direction normalizes to a unit row (" + error + ")");
  std::filesystem::remove(tiny_path);

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
  TestIgnoredTensors();
  TestSelectLayers();
  TestFingerprint();
  std::filesystem::remove_all(TempDir());
  if (failures != 0) {
    std::printf("%d projection loader check(s) failed\n", failures);
    return 1;
  }
  std::printf("projection loader checks passed\n");
  return 0;
}
