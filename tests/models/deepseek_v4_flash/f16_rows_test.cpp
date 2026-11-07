#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/resident_api.h"

namespace {
std::filesystem::path g_weight_file;

void RemoveWeightFile() {
  std::error_code ignored;
  std::filesystem::remove(g_weight_file, ignored);
}

void Check(bool success, const char* message) {
  if (!success) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct Shape {
  const char* name;
  uint32_t in;
  uint32_t out;
};

constexpr Shape kShapes[] = {{"ratio-4 compressor", 4096, 1024},
                             {"ratio-128 compressor", 4096, 512},
                             {"indexer compressor (router shape)", 4096, 256},
                             {"indexer query", 1024, 8192},
                             {"indexer weights", 4096, 64}};
constexpr uint32_t kHcStreams = 4, kHcEmbd = 4096, kHcMixOutputs = 24,
                   kHcMaxTokens = 1024, kHcWidth = kHcStreams * kHcEmbd;
constexpr Shape kHcMix = {"HC mix", kHcWidth, kHcMixOutputs};
constexpr uint32_t kMaxTokens = 4096;
constexpr uint64_t kGgufAlignment = 32;
constexpr int kDecodeBatchMode = 0;
constexpr int kVerifierBatchMode = 1;

struct Weights {
  uint64_t packed[2];
  uint64_t padded[2];
};

struct Tensor {
  explicit Tensor(uint64_t bytes) : data(ds4_gpu_tensor_alloc(bytes)) {
    Check(data != nullptr, "allocate tensor");
  }
  ~Tensor() { ds4_gpu_tensor_free(data); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ds4_gpu_tensor* data;
};

std::vector<uint8_t> Read(const Tensor& tensor, uint64_t bytes) {
  std::vector<uint8_t> out(bytes);
  Check(ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(tensor.data, 0, out.data(), bytes),
        "read output");
  return out;
}

void Poison(Tensor& tensor, uint64_t floats) {
  Check(ds4_gpu_tensor_fill_f32(tensor.data, NAN, floats), "poison output");
}

void Same(const char* route, const Shape& s, uint32_t n_tok,
          const Tensor& packed, const Tensor& padded, uint64_t floats) {
  const auto a = Read(packed, floats * sizeof(float));
  const auto b = Read(padded, floats * sizeof(float));
  for (uint64_t i = 0; i < floats; ++i) {
    float value;
    std::memcpy(&value, a.data() + i * sizeof(float), sizeof(value));
    if (!std::isfinite(value) ||
        std::memcmp(a.data() + i * sizeof(float), b.data() + i * sizeof(float),
                    sizeof(float)) != 0) {
      std::fprintf(stderr,
                   "FAIL: %s %s tokens=%u differs at %llu (packed %s)\n", route,
                   s.name, n_tok, (unsigned long long)i,
                   std::isfinite(value) ? "finite" : "not written");
      std::exit(2);
    }
  }
}

uint16_t RandomHalfBelowOne(std::mt19937& random) {
  constexpr uint32_t kMinExponent = 8, kExponents = 7;
  const uint32_t bits = random();
  return uint16_t(((bits & 1u) << 15) |
                  ((kMinExponent + (bits >> 1) % kExponents) << 10) |
                  ((bits >> 4) & 0x3ffu));
}

class Checker {
public:
  Checker(const void* map, uint64_t size, std::mt19937& random)
      : map_(map),
        size_(size),
        x_(uint64_t(kMaxTokens) * 4096 * sizeof(float)),
        x_h_(uint64_t(kMaxTokens) * 4096 * sizeof(uint16_t)),
        a0_(uint64_t(kMaxTokens) * 8192 * sizeof(float)),
        a1_(uint64_t(kMaxTokens) * 8192 * sizeof(float)),
        b0_(uint64_t(kMaxTokens) * 8192 * sizeof(float)),
        b1_(uint64_t(kMaxTokens) * 8192 * sizeof(float)),
        c0_(uint64_t(kHcMaxTokens) * kHcMix.out * sizeof(float)),
        c1_(uint64_t(kHcMaxTokens) * kHcMix.out * sizeof(float)) {
    std::uniform_real_distribution<float> values(-2.F, 2.F);
    std::vector<float> x(uint64_t(kMaxTokens) * 4096);
    for (float& v : x)
      v = values(random);
    std::vector<uint16_t> x_h(x.size());
    for (uint16_t& v : x_h)
      v = RandomHalfBelowOne(random);
    Check(
        ds4_gpu_tensor_write(x_.data, 0, x.data(), x.size() * sizeof(float)) &&
            ds4_gpu_tensor_write(x_h_.data, 0, x_h.data(),
                                 x_h.size() * sizeof(uint16_t)),
        "write activations");
  }

  void Single(const Shape& s, const Weights& w, uint32_t n) {
    const uint64_t floats = uint64_t(n) * s.out;
    Poison(a0_, floats);
    Poison(b0_, floats);
    Check(ds4_gpu_matmul_f16_tensor(a0_.data, map_, size_, w.packed[0], s.in,
                                    s.out, x_.data, n) &&
              ds4_gpu_matmul_f16_tensor(b0_.data, map_, size_, w.padded[0],
                                        s.in, s.out, x_.data, n),
          "f16 projection");
    Same("f16", s, n, a0_, b0_, floats);
  }

  void Pair(const Shape& s, const Weights& w, uint32_t n, bool narrow) {
    const uint64_t floats = uint64_t(n) * s.out;
    Poison(a0_, floats);
    Poison(a1_, floats);
    Poison(b0_, floats);
    Poison(b1_, floats);
    auto* entry = narrow ? ds4_gpu_matmul_f16_pair_narrow_tensor
                         : ds4_gpu_matmul_f16_pair_tensor;
    Check(entry(a0_.data, a1_.data, map_, size_, w.packed[0], w.packed[1], s.in,
                s.out, x_.data, n) &&
              entry(b0_.data, b1_.data, map_, size_, w.padded[0], w.padded[1],
                    s.in, s.out, x_.data, n),
          "f16 pair projection");
    const char* route = narrow ? "f16 narrow pair" : "f16 pair";
    Same(route, s, n, a0_, b0_, floats);
    Same(route, s, n, a1_, b1_, floats);
  }

  void Groups(const Shape& s, const Weights& w,
              const std::vector<uint32_t>& offsets) {
    const uint32_t groups = uint32_t(offsets.size() - 1);
    const uint32_t n = offsets.back();
    const uint64_t floats = uint64_t(n) * s.out;
    Poison(a0_, floats);
    Poison(b0_, floats);
    Check(ds4_gpu_matmul_f16_group_pairs_tensor(
              a0_.data, map_, size_, w.packed[0], s.in, s.out, x_.data, n,
              offsets.data(), groups) &&
              ds4_gpu_matmul_f16_group_pairs_tensor(
                  b0_.data, map_, size_, w.padded[0], s.in, s.out, x_.data, n,
                  offsets.data(), groups),
          "f16 grouped pairs");
    Same("f16 grouped pairs", s, n, a0_, b0_, floats);
  }

  void HcFused(const Weights& w, uint64_t scale, uint64_t base, uint32_t n) {
    const uint64_t out_floats = uint64_t(n) * kHcEmbd,
                   mix_floats = uint64_t(n) * kHcMix.out;
    for (Tensor* t : {&a0_, &b0_})
      Poison(*t, out_floats);
    for (Tensor* t : {&a1_, &b1_, &c0_, &c1_})
      Poison(*t, mix_floats);
    Check(
        ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
            a0_.data, a1_.data, c0_.data, x_.data, map_, size_, w.packed[0],
            scale, base, kHcEmbd, kHcStreams, n, 20, 1e-6F, 1e-6F) &&
            ds4_gpu_hc_norm_mix_split_weighted_sum_tensor(
                b0_.data, b1_.data, c1_.data, x_.data, map_, size_, w.padded[0],
                scale, base, kHcEmbd, kHcStreams, n, 20, 1e-6F, 1e-6F),
        "fused HC pre-block");
    Same("fused HC output", kHcMix, n, a0_, b0_, out_floats);
    Same("fused HC mix", kHcMix, n, a1_, b1_, mix_floats);
    Same("fused HC split", kHcMix, n, c0_, c1_, mix_floats);
  }

  void HalfInput(const Shape& s, const Weights& w, uint32_t n) {
    const uint64_t floats = uint64_t(n) * s.out;
    Poison(a0_, floats);
    Poison(b0_, floats);
    Check(
        ds4_gpu_matmul_f16_f16_input_tensor(a0_.data, map_, size_, w.packed[0],
                                            s.in, s.out, x_h_.data, n) &&
            ds4_gpu_matmul_f16_f16_input_tensor(
                b0_.data, map_, size_, w.padded[0], s.in, s.out, x_h_.data, n),
        "f16 half-input projection");
    Same("f16 half input", s, n, a0_, b0_, floats);
  }

private:
  const void* map_;
  uint64_t size_;
  Tensor x_, x_h_, a0_, a1_, b0_, b1_, c0_, c1_;
};
}  // namespace

int main() {
  if (!ds4_gpu_init())
    return 77;
  std::mt19937 random(1151);

  std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
  shapes.push_back(kHcMix);
  std::vector<Weights> weights;
  uint64_t size = 0;
  for (const Shape& s : shapes) {
    const uint64_t bytes = uint64_t(s.in) * s.out * sizeof(uint16_t);
    Weights w{};
    for (uint64_t* offset :
         {&w.packed[0], &w.packed[1], &w.padded[0], &w.padded[1]}) {
      *offset = size + kGgufAlignment;
      size = *offset + bytes;
    }
    weights.push_back(w);
  }
  const Weights& hc = weights.back();
  const uint64_t hc_scale = size + kGgufAlignment;
  const uint64_t hc_base = hc_scale + kGgufAlignment;
  size = hc_base + kHcMix.out * sizeof(float);

  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-ds4-f16-rows-" + std::to_string(random()));
  g_weight_file = path;
  std::atexit(RemoveWeightFile);
  {
    std::vector<uint16_t> file(size / sizeof(uint16_t), 0);
    for (size_t i = 0; i < weights.size(); ++i) {
      const Shape& s = shapes[i];
      const uint64_t count = uint64_t(s.in) * s.out;
      for (int k = 0; k < 2; ++k) {
        uint16_t* packed = file.data() + weights[i].packed[k] / 2;
        for (uint64_t j = 0; j < count; ++j)
          packed[j] = RandomHalfBelowOne(random);
        std::memcpy(file.data() + weights[i].padded[k] / 2, packed,
                    count * sizeof(uint16_t));
      }
    }
    const float scale[3] = {0.3F, -0.7F, 1.1F};
    std::memcpy(file.data() + hc_scale / 2, scale, sizeof(scale));
    std::uniform_real_distribution<float> bases(-0.5F, 0.5F);
    for (uint32_t j = 0; j < kHcMix.out; ++j) {
      const float b = bases(random);
      std::memcpy(file.data() + hc_base / 2 + 2 * j, &b, sizeof(b));
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()),
              std::streamsize(file.size() * sizeof(uint16_t)));
    Check(bool(out), "write weight file");
  }
  std::string error;
  auto map = gufo::platform::ReadOnlyMappedFile::Open(path, &error);
  Check(map.valid() && map.size() == size, "map weight file");
  Check(ds4_gpu_set_model_map_range(map.data(), size, 0, size, 0) &&
            ds4_gpu_set_model_fd(map.PlatformHandle()),
        "register weight file");
  Check(ds4_gpu_cache_model_range(map.data(), size, hc_scale, 3 * sizeof(float),
                                  "hc scale") &&
            ds4_gpu_cache_model_range(map.data(), size, hc_base,
                                      kHcMix.out * sizeof(float), "hc base"),
        "upload HC scale and base");
  for (size_t i = 0; i < weights.size(); ++i) {
    const Shape& s = shapes[i];
    const uint64_t bytes = uint64_t(s.in) * s.out * sizeof(uint16_t);
    for (int k = 0; k < 2; ++k) {
      Check(ds4_gpu_cache_model_range(map.data(), size, weights[i].packed[k],
                                      bytes, "packed") &&
                ds4_gpu_cache_model_f16_rows(map.data(), size,
                                             weights[i].padded[k], s.in, s.out,
                                             "padded"),
            "upload weights");
    }
  }
  Check(ds4_gpu_release_model_staging(), "finish weight upload");

  const Weights& w0 = weights[0];
  Check(ds4_gpu_cache_model_f16_rows(map.data(), size, w0.padded[0],
                                     kShapes[0].in, kShapes[0].out, "padded"),
        "repeating a padded upload is a no-op");
  Check(!ds4_gpu_cache_model_f16_rows(map.data(), size, w0.packed[0],
                                      kShapes[0].in, kShapes[0].out, "padded"),
        "a packed weight cannot be uploaded again with padded rows");
  Check(!ds4_gpu_cache_model_range(map.data(), size, w0.padded[0] + 4096, 64,
                                   "packed"),
        "a packed upload of padded bytes fails");
  {
    Tensor out(uint64_t(kShapes[0].out) * 2 * sizeof(float));
    Tensor x(uint64_t(kShapes[0].in) * sizeof(float));
    Check(!ds4_gpu_matmul_f16_tensor(out.data, map.data(), size, w0.padded[0],
                                     kShapes[0].in / 2, kShapes[0].out * 2,
                                     x.data, 1),
          "a padded weight read with another shape fails");
    Check(!ds4_gpu_matmul_f32_tensor(out.data, map.data(), size, w0.padded[0],
                                     kShapes[0].in, kShapes[0].out / 2, x.data,
                                     1),
          "an F32 read of padded bytes fails");
  }

  {
    Checker check(map.data(), size, random);
    for (int mode : {kDecodeBatchMode, kVerifierBatchMode}) {
      ds4_gpu_set_small_batch_mode(mode);
      for (uint32_t n : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 24u, 33u, 64u, 96u,
                         127u, 128u, 333u, 1000u, kHcMaxTokens})
        check.Single(kHcMix, hc, n);
      for (uint32_t n : {1u, 4u, 5u, 127u, 128u, 130u, 1000u})
        check.HcFused(hc, hc_scale, hc_base, n);
    }
    ds4_gpu_set_small_batch_mode(kDecodeBatchMode);
    std::printf(
        "F16 %s %ux%u: padded rows match packed rows on the projection "
        "and the fused pre-block\n",
        kHcMix.name, kHcMix.in, kHcMix.out);
    for (size_t i = 0; i < std::size(kShapes); ++i) {
      const Shape& s = kShapes[i];
      const Weights& w = weights[i];
      for (int mode : {kDecodeBatchMode, kVerifierBatchMode}) {
        ds4_gpu_set_small_batch_mode(mode);
        for (uint32_t n : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 24u, 33u, 64u, 96u,
                           65u, 128u, 136u, 333u, 512u, 1000u, 4096u})
          check.Single(s, w, n);
        for (uint32_t n : {1u, 2u, 4u, 8u, 64u, 128u, 256u, 333u, 4096u})
          check.Pair(s, w, n, false);
        for (uint32_t n = 2; n <= 8; ++n)
          check.Pair(s, w, n, true);
        for (const std::vector<uint32_t>& offsets :
             {std::vector<uint32_t>{0, 1, 2},
              {0, 3, 6, 9, 12},
              {0, 6, 12},
              {0, 2, 5},
              {0, 3, 4, 9, 10}})
          check.Groups(s, w, offsets);
        for (uint32_t n : {64u, 96u, 128u})
          check.HalfInput(s, w, n);
      }
      ds4_gpu_set_small_batch_mode(kDecodeBatchMode);
      std::printf(
          "F16 %s %ux%u: padded rows match packed rows on every route\n",
          s.name, s.in, s.out);
    }
  }
  ds4_gpu_cleanup();
  map.Close();
  std::puts("F16 padded-row weights pass");
}
