// Compares Gufo's LiLiCorr drafter with llama.cpp on llama.cpp's own trunk
// features: reads a dump of tools/llama_parity/llama_dflash.cpp, injects the
// same features, drafts at the same anchors and reports, per drafted slot,
// top-1 agreement and KL of the drafter logits, candidate agreement, the
// largest lattice score difference, and the greedy walks.
//
//   nemotron35_bolt_drafter_probe --model MODEL.gguf --drafter DRAFT.gguf
//                                 --dump llama_dflash.bin

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/nemotron35_bolt/drafter.hpp"
#include "src/models/nemotron35_bolt/engine.hpp"

namespace nb = gufo::models::nemotron35_bolt;

namespace {

template<class T>
bool Read(std::ifstream& in, T* data, std::size_t count) {
  in.read(reinterpret_cast<char*>(data),
          static_cast<std::streamsize>(count * sizeof(T)));
  return static_cast<bool>(in);
}

double Kl(const float* p_logits, const float* q_logits, std::size_t n) {
  auto lse = [n](const float* v) {
    const float m = *std::max_element(v, v + n);
    double z = 0;
    for (std::size_t i = 0; i < n; ++i)
      z += std::exp(double(v[i]) - m);
    return m + std::log(z);
  };
  const double lp = lse(p_logits);
  const double lq = lse(q_logits);
  double kl = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = double(p_logits[i]) - lp;
    kl += std::exp(a) * (a - (double(q_logits[i]) - lq));
  }
  return kl;
}

std::size_t Argmax(const float* v, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, drafter_path, dump;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--model")
      model_path = argv[i + 1];
    else if (arg == "--drafter")
      drafter_path = argv[i + 1];
    else if (arg == "--dump")
      dump = argv[i + 1];
  }
  if (model_path.empty() || drafter_path.empty() || dump.empty()) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.gguf --drafter DRAFT.gguf --dump "
                 "llama_dflash.bin\n",
                 argv[0]);
    return 2;
  }
  std::ifstream in(dump, std::ios::binary);
  char magic[4] = {};
  std::uint32_t h[7] = {};
  in.read(magic, 4);
  if (!in || std::string(magic, 4) != "DFL1" || !Read(in, h, 7)) {
    std::fprintf(stderr, "not a llama_dflash dump\n");
    return 1;
  }
  const std::uint32_t n_tokens = h[0], n_feat = h[1] * h[2], n_anchor = h[3],
                      block = h[4], vocab = h[5], n_embd = h[6];
  std::vector<float> features(static_cast<std::size_t>(n_tokens) * n_feat);
  std::vector<float> projected(static_cast<std::size_t>(n_tokens) * n_embd);
  if (!Read(in, features.data(), features.size()) ||
      !Read(in, projected.data(), projected.size())) {
    std::fprintf(stderr, "truncated dump\n");
    return 1;
  }

  std::string error;
  nb::ModelOptions options;
  options.max_context = 4096;
  auto model = nb::Model::Load(model_path, options, &error);
  if (!model || !model->LoadDrafter(drafter_path, &error)) {
    std::fprintf(stderr, "load: %s\n", error.c_str());
    return 1;
  }
  const auto& dc = model->drafter()->config();
  if (dc.FeatureSize() != n_feat || dc.block_size != block ||
      model->config().vocab_size != vocab) {
    std::fprintf(stderr, "dump does not match the models\n");
    return 1;
  }
  auto session = model->CreateSession(4096, &error);
  if (!session || session->draft() == nullptr) {
    std::fprintf(stderr, "session: %s\n", error.c_str());
    return 1;
  }
  nb::DraftContext& draft = *session->draft();
  const std::uint32_t slots = dc.Slots();
  const std::uint32_t k = dc.lc_top_k;
  std::vector<float> llama_logits(static_cast<std::size_t>(block) * vocab);
  std::vector<float> llama_lattice(static_cast<std::size_t>(block) * n_embd);
  std::vector<float> logits(static_cast<std::size_t>(slots) * vocab);
  std::vector<double> kls;
  std::size_t top1 = 0, rows = 0, cand_same = 0, walk_same = 0;
  std::size_t slot_lists = 0, slot_sets = 0;
  std::vector<double> score_diffs;
  double max_score_diff = 0;
  std::uint32_t injected = 0;
  for (std::uint32_t a = 0; a < n_anchor; ++a) {
    std::int32_t meta[2];
    if (!Read(in, meta, 2) ||
        !Read(in, llama_logits.data(), llama_logits.size()) ||
        !Read(in, llama_lattice.data(), llama_lattice.size())) {
      std::fprintf(stderr, "truncated anchor %u\n", a);
      return 1;
    }
    const auto anchor = static_cast<std::uint32_t>(meta[0]);
    if (anchor > injected) {
      const std::span<const float> rows_in(
          features.data() + static_cast<std::size_t>(injected) * n_feat,
          static_cast<std::size_t>(anchor - injected) * n_feat);
      for (std::size_t off = 0; off < rows_in.size();
           off += static_cast<std::size_t>(512) * n_feat) {
        const std::size_t len =
            std::min<std::size_t>(rows_in.size() - off, 512ull * n_feat);
        if (!draft.InjectHost(
                rows_in.subspan(off, len),
                injected + static_cast<std::uint32_t>(off / n_feat), &error)) {
          std::fprintf(stderr, "inject: %s\n", error.c_str());
          return 1;
        }
      }
      injected = anchor;
    }
    nb::DraftLattice lattice;
    if (!draft.Draft(meta[1], anchor, &lattice, logits, &error)) {
      std::fprintf(stderr, "draft: %s\n", error.c_str());
      return 1;
    }
    // llama.cpp's lattice row s + 1 holds slot s: candidates, then scores.
    nb::DraftLattice reference;
    reference.slots = slots;
    reference.top_k = k;
    reference.scores.assign(lattice.scores.size(), 0.0F);
    bool same_candidates = true;
    for (std::uint32_t s = 0; s < slots; ++s) {
      const float* row =
          llama_lattice.data() + static_cast<std::size_t>(s + 1) * n_embd;
      for (std::uint32_t j = 0; j < k; ++j) {
        reference.candidates.push_back(static_cast<std::int32_t>(row[j]));
        same_candidates =
            same_candidates &&
            lattice.candidates[s * k + j] == static_cast<std::int32_t>(row[j]);
      }
      std::vector<std::int32_t> mine(lattice.candidates.begin() + s * k,
                                     lattice.candidates.begin() + (s + 1) * k);
      std::vector<std::int32_t> ref(reference.candidates.end() - k,
                                    reference.candidates.end());
      slot_lists += mine == ref;
      std::sort(mine.begin(), mine.end());
      std::sort(ref.begin(), ref.end());
      slot_sets += mine == ref;
      const std::size_t count = s == 0 ? k : k * k;
      const std::size_t base = s == 0 ? 0 : k + (s - 1) * k * k;
      for (std::size_t j = 0; j < count; ++j)
        reference.scores[base + j] = row[k + j];
      const float* ours = logits.data() + static_cast<std::size_t>(s) * vocab;
      const float* theirs =
          llama_logits.data() + static_cast<std::size_t>(s + 1) * vocab;
      kls.push_back(Kl(theirs, ours, vocab));
      top1 += Argmax(theirs, vocab) == Argmax(ours, vocab);
      ++rows;
    }
    // Scores of slots whose own and previous candidate lists agree.
    for (std::uint32_t s = 0; s < slots; ++s) {
      bool same = true;
      for (std::uint32_t q = (s == 0 ? 0 : s - 1); q <= s; ++q)
        for (std::uint32_t j = 0; j < k; ++j)
          same = same && lattice.candidates[q * k + j] ==
                             reference.candidates[q * k + j];
      if (!same)
        continue;
      const std::size_t count = s == 0 ? k : k * k;
      const std::size_t base = s == 0 ? 0 : k + (s - 1) * k * k;
      for (std::size_t j = 0; j < count; ++j)
        score_diffs.push_back(
            std::fabs(lattice.scores[base + j] - reference.scores[base + j]));
    }
    if (same_candidates) {
      ++cand_same;
      for (std::size_t j = 0; j < lattice.scores.size(); ++j)
        max_score_diff = std::max<double>(
            max_score_diff, std::fabs(lattice.scores[j] - reference.scores[j]));
    }
    const auto ours = lattice.Walk(slots);
    const auto theirs = reference.Walk(slots);
    walk_same += ours == theirs;
    std::printf("anchor %4u:", anchor);
    for (std::size_t j = 0; j < ours.size(); ++j)
      std::printf(" %d%s", ours[j], ours[j] == theirs[j] ? "" : "!");
    std::printf("%s\n", same_candidates ? "" : "  (candidates differ)");
  }
  std::sort(kls.begin(), kls.end());
  std::sort(score_diffs.begin(), score_diffs.end());
  if (!score_diffs.empty())
    std::printf(
        "lattice scores on agreeing slots: %zu, |diff| median %.3g "
        "p90 %.3g p99 %.3g max %.3g\n",
        score_diffs.size(), score_diffs[score_diffs.size() / 2],
        score_diffs[score_diffs.size() * 9 / 10],
        score_diffs[score_diffs.size() * 99 / 100], score_diffs.back());
  std::printf(
      "slots %zu: top-1 %zu/%zu, KL median %.2e p90 %.2e max %.2e\n"
      "slot candidates: same order %zu, same set %zu of %zu\n"
      "anchors %u: same candidates %zu, same walk %zu, max score diff %.3g\n",
      rows, top1, rows, kls[kls.size() / 2], kls[kls.size() * 9 / 10],
      kls.back(), slot_lists, slot_sets, rows, n_anchor, cand_same, walk_same,
      max_score_diff);
  return 0;
}
