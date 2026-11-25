// Speed and output-identity bench for the Inkling 2 Mini vision encoder.
//
//   inkling2_vision_bench --mmproj GGUF [--sizes 640x480,1280x960]
//                         [--reps 5] [--load-reps 0]
//
// For every size: a deterministic synthetic RGB image is encoded once to warm
// up, then --reps times (each Encode + stream synchronize). Prints the median
// and minimum time, the token count and an FNV-1a hash of the output rows'
// bytes, so two builds can be compared for bit identity. --load-reps N also
// times N cold Load() calls.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/inkling2/vision/vision_encoder.hpp"

namespace {

using gufo::models::inkling2::vision::VisionEncoder;

double Now() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<uint8_t> MakeImage(int w, int h) {
  std::vector<uint8_t> rgb(size_t(w) * h * 3);
  uint32_t s = 0x9e3779b9u ^ uint32_t(w * 131 + h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      for (int c = 0; c < 3; c++) {
        s = s * 1664525u + 1013904223u;
        const int smooth = ((x * (c + 1) + y * (3 - c)) >> 2) & 255;
        rgb[(size_t(y) * w + x) * 3 + c] =
            uint8_t((smooth * 3 + int(s >> 24)) >> 2);
      }
  return rgb;
}

uint64_t Fnv(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mmproj, sizes = "448x448,640x480,1280x960,2560x2560";
  int reps = 5, load_reps = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--mmproj")
      mmproj = next();
    else if (k == "--sizes")
      sizes = next();
    else if (k == "--reps")
      reps = std::max(1, std::atoi(next().c_str()));
    else if (k == "--load-reps")
      load_reps = std::atoi(next().c_str());
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (mmproj.empty()) {
    std::fprintf(stderr, "usage: %s --mmproj GGUF [--sizes WxH,...]\n",
                 argv[0]);
    return 2;
  }
  for (int i = 0; i < load_reps; i++) {
    VisionEncoder e;
    std::string err;
    const double t0 = Now();
    if (!e.Load(mmproj, &err)) {
      std::fprintf(stderr, "load: %s\n", err.c_str());
      return 1;
    }
    std::printf("load %.1f ms\n", Now() - t0);
  }
  VisionEncoder enc;
  std::string err;
  if (!enc.Load(mmproj, &err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  hipStream_t s;
  if (hipStreamCreate(&s) != hipSuccess)
    return 1;
  size_t pos = 0;
  while (pos < sizes.size()) {
    size_t end = sizes.find(',', pos);
    if (end == std::string::npos)
      end = sizes.size();
    const std::string spec = sizes.substr(pos, end - pos);
    pos = end + 1;
    int w = 0, h = 0;
    if (std::sscanf(spec.c_str(), "%dx%d", &w, &h) != 2)
      continue;
    const auto img = MakeImage(w, h);
    VisionEncoder::Result r;
    std::vector<double> t;
    std::vector<float> host;
    uint64_t hash = 0;
    for (int i = 0; i <= reps; i++) {
      const double t0 = Now();
      if (!enc.Encode(img.data(), w, h, s, &r, &err)) {
        std::fprintf(stderr, "encode: %s\n", err.c_str());
        return 1;
      }
      hipStreamSynchronize(s);
      const double dt = Now() - t0;
      if (i > 0)
        t.push_back(dt);
      host.resize(size_t(r.n_tokens) * r.dim);
      hipMemcpy(host.data(), r.embd, host.size() * 4, hipMemcpyDeviceToHost);
      const uint64_t hh = Fnv(host.data(), host.size() * 4);
      if (i == 0)
        hash = hh;
      else if (hh != hash) {
        std::fprintf(stderr, "%s: output differs between runs\n", spec.c_str());
        return 1;
      }
    }
    std::sort(t.begin(), t.end());
    std::printf(
        "%-10s tokens %5d  median %8.2f ms  min %8.2f ms  hash %016llx\n",
        spec.c_str(), r.n_tokens, t[t.size() / 2], t[0],
        static_cast<unsigned long long>(hash));
    std::fflush(stdout);
  }
  hipStreamDestroy(s);
  return 0;
}
