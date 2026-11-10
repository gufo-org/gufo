#include <hip/hip_runtime.h>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "staging/models/glm5_next/kernels/rocm/executor.hpp"
#include "staging/tests/models/glm5_next/hip_test.hpp"

// End-to-end regression gate: replays each case's prompt tokens through
// Executor::Step, then the 24 greedy continuations, checking every sampled
// token exactly, the top-1 of both logits snapshots and the top-32 logit
// values within a fixed absolute margin. The fixture is this executor's own
// capture (GUFO_GLM5_NEXT_STEPS mode). Cross-engine semantics against the
// llama.cpp reference are established per-stage by the reference probe and
// a float64 weight-space simulation (see docs/models/glm-5.3-flash):
// the executor's dequantized F32 GEMMs are float64-faithful while the
// reference's quantized kernels deviate by ~0.08 nats at lm_head alone, and
// chaotic mHC residual accumulation moves ~5% of greedy argmaxes past any
// fixed logit margin. With GUFO_GLM5_NEXT_STEPS set the case instead runs
// self-capture: the executor samples greedily from its own logits and the
// top-32 lists of every position are written for the fixture refresh.
// Skips (77) unless GUFO_GLM5_NEXT_GGUF is set; the fixture path comes from
// GUFO_GLM5_NEXT_FIXTURE. Needs the GPU exclusively (the ~91 GiB resident
// artifact).
namespace {

namespace q = gufo::models::glm5_next::rocm;
namespace t = gufo::tests::glm5_next;

constexpr float kLogitMargin = 0.5F;
constexpr int kSelfGen = 24;

int failures = 0;

// Optional bisection dump (GUFO_GLM5_NEXT_STEPS): top-32 pairs at every
// prefill position and every greedy continuation state, in the same format
// as the reference probe side-output, for per-position comparison.
FILE* g_steps = nullptr;
const char* g_dbg_dir = nullptr;
bool g_dbg_done = false;
int g_dbg_step = 0;

// Points the executor dump at <dir>/<step:03d> for the single next Step.
void ArmDumpStep(q::Executor& exec) {
  if (g_dbg_dir == nullptr) {
    return;
  }
  char buf[16];
  std::snprintf(buf, sizeof(buf), "/%03d", g_dbg_step++);
  const std::string d = std::string(g_dbg_dir) + buf;
  static_cast<void>(mkdir(d.c_str(), 0755));
  exec.SetDebugDump(d);
}

void WriteTopK(FILE* f, const std::vector<float>& l) {
  constexpr std::size_t kTop = 32;
  std::vector<std::int32_t> idx(l.size());
  for (std::size_t i = 0; i < l.size(); ++i) {
    idx[i] = static_cast<std::int32_t>(i);
  }
  const std::size_t k = std::min(kTop, l.size());
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&l](std::int32_t a, std::int32_t b) {
                      return l[a] > l[b];
                    });
  for (std::size_t i = 0; i < k; ++i) {
    std::fprintf(f, "%s\n        [%d, %.9g]", i > 0 ? "," : "", idx[i], l[idx[i]]);
  }
}

void Report(const std::string& tag, bool ok) {
  std::printf("%-52s %s\n", tag.c_str(), ok ? "OK" : "FAIL");
  if (!ok) {
    ++failures;
  }
}

// Downloads the logits and reports the argmax plus (when `top` is set) the
// absolute deviation of the fixture's top-k entries from the recomputed
// values.
float CheckSnapshot(q::Executor& exec, const gufo::json::Value* top,
                    std::int32_t* argmax, std::vector<float>* logits) {
  t::CheckHip(hipMemcpy(logits->data(), exec.logits(),
                        logits->size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              "download logits");
  std::int32_t best = 0;
  float best_v = (*logits)[0];
  for (std::size_t i = 1; i < logits->size(); ++i) {
    if ((*logits)[i] > best_v) {
      best_v = static_cast<float>((*logits)[i]);
      best = static_cast<std::int32_t>(i);
    }
  }
  *argmax = best;
  float max_err = 0.0F;
  if (top != nullptr) {
    for (const gufo::json::Value& e : top->items()) {
      const std::size_t tok = e.items()[0].as_size();
      const double want = e.items()[1].as_double();
      const float got = (*logits)[tok];
      max_err = std::max(max_err, std::fabs(got - static_cast<float>(want)));
    }
  }
  return max_err;
}

void RunCase(q::Executor& exec, const gufo::json::Value& c) {
  const std::string name = c.member_str("name", "?");
  std::vector<std::int32_t> prompt;
  for (const gufo::json::Value& v : c.find("prompt_tokens")->items()) {
    prompt.push_back(static_cast<std::int32_t>(v.as_size()));
  }
  std::vector<std::int32_t> greedy;
  for (const gufo::json::Value& v : c.find("greedy_tokens")->items()) {
    greedy.push_back(static_cast<std::int32_t>(v.as_size()));
  }
  const gufo::json::Value* top_pre = c.find("logits_after_prefill");
  const gufo::json::Value* top_gen = c.find("logits_after_gen");

  const std::uint32_t cap =
      static_cast<std::uint32_t>(prompt.size() + greedy.size() + 2);
  std::string error;
  auto session = exec.CreateSession(cap, &error);
  if (session == nullptr) {
    Report(name + " create session", false);
    std::printf("  %s\n", error.c_str());
    return;
  }
  std::vector<float> logits(exec.model().config().vocab_size);
  std::int32_t next = -1;

  static int case_idx = 0;
  const int my_idx = case_idx++;
  const bool dump = g_steps != nullptr;
  // Trunk-anchor dump for the first case only; each Step re-arms the dump
  // directory with its own per-step subdirectory (<dir>/NNN).
  const bool dbg = g_dbg_dir != nullptr && my_idx == 0 && !g_dbg_done;
  if (dump) {
    std::fprintf(g_steps, "%s  {\"name\": \"%s\", \"prefill\": [\n",
                 my_idx > 0 ? ",\n" : "", name.c_str());
  }
  // GUFO_GLM5_NEXT_PREFILL_CHUNK=<k> replays the prompt through the chunked
  // Prefill path (Executor::kChunk batch) instead of per-token Step, which
  // must stay bit-identical: the same fixture gates catch any divergence.
  static const std::size_t prefill_chunk = [] {
    const char* v = std::getenv("GUFO_GLM5_NEXT_PREFILL_CHUNK");
    return v != nullptr && v[0] != '\0' ? std::strtoul(v, nullptr, 10) : 0U;
  }();
  if (prefill_chunk >= 2 && !dump && !dbg) {
    exec.Prefill(*session, prompt.data(), prompt.size());
  } else {
    bool first_pos = true;
    for (std::int32_t tok : prompt) {
      if (dbg) {
        ArmDumpStep(exec);
      }
      exec.Step(*session, tok);
      if (!dump) {
        continue;
      }
      CheckSnapshot(exec, nullptr, &next, &logits);
      std::fprintf(g_steps, "%s      [", first_pos ? "" : "],\n");
      first_pos = false;
      WriteTopK(g_steps, logits);
    }
  }
  if (dump) {
    // Self-capture mode: records the executor's own greedy continuation
    // (kSelfGen tokens plus the state after the last one). The fixture's
    // greedy chain and snapshot lists are produced by this path; the
    // `greedy` list from the file is ignored here.
    CheckSnapshot(exec, nullptr, &next, &logits);
    std::fprintf(g_steps, "]],\n      \"gen\": [\n      [");
    WriteTopK(g_steps, logits);
    for (int k = 1; k <= kSelfGen; ++k) {
      if (dbg) {
        ArmDumpStep(exec);
      }
      exec.Step(*session, next);
      CheckSnapshot(exec, nullptr, &next, &logits);
      std::fprintf(g_steps, "],\n      [");
      WriteTopK(g_steps, logits);
    }
    if (dbg && !g_dbg_done) {
      g_dbg_done = true;
      exec.SetDebugDump("");
    }
    std::fprintf(g_steps, "]]\n  }\n");
    return;
  }

  const float pre_err = CheckSnapshot(exec, top_pre, &next, &logits);
  Report(name + " prefill top-1", next == greedy[0]);
  Report(name + " prefill top-32 within margin", pre_err <= kLogitMargin);
  std::printf("  %s prefill max |dlogit| = %.4f\n", name.c_str(), pre_err);

  bool tokens_ok = true;
  for (std::size_t i = 0; i + 1 < greedy.size(); ++i) {
    if (dbg) {
      ArmDumpStep(exec);
    }
    exec.Step(*session, greedy[i]);
    CheckSnapshot(exec, nullptr, &next, &logits);
    if (next != greedy[i + 1]) {
      tokens_ok = false;
      std::printf("  %s step %zu: want %d got %d\n", name.c_str(), i,
                  greedy[i + 1], next);
    }
  }
  Report(name + " greedy tokens exact", tokens_ok);

  if (dbg) {
    ArmDumpStep(exec);
  }
  exec.Step(*session, greedy.back());
  if (dbg && !g_dbg_done) {
    g_dbg_done = true;
    exec.SetDebugDump("");
  }
  const float gen_err = CheckSnapshot(exec, top_gen, &next, &logits);
  Report(name + " final snapshot within margin", gen_err <= kLogitMargin);
  std::printf("  %s final max |dlogit| = %.4f\n", name.c_str(), gen_err);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GLM5_NEXT_GGUF");
  const char* fixture = std::getenv("GUFO_GLM5_NEXT_FIXTURE");
  if (path == nullptr || path[0] == '\0') {
    std::printf("SKIP: set GUFO_GLM5_NEXT_GGUF\n");
    return 77;
  }
  if (fixture == nullptr || fixture[0] == '\0') {
    std::printf("FAIL: set GUFO_GLM5_NEXT_FIXTURE\n");
    return 1;
  }
  std::ifstream in(fixture, std::ios::binary);
  if (!in) {
    std::printf("FAIL: cannot read fixture %s\n", fixture);
    return 1;
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  const gufo::json::Value doc = gufo::json::parse(buf.str());
  if (!doc.is_object() || doc.find("cases") == nullptr) {
    std::printf("FAIL: cannot parse fixture %s\n", fixture);
    return 1;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::printf("FAIL: open %s: %s\n", path, error.c_str());
    return 1;
  }
  const auto weights =
      gufo::models::glm5_next::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    std::printf("FAIL: bind: %s\n", error.c_str());
    return 1;
  }
  const auto model = q::DeviceModel::Upload(*weights, *reader, &error);
  if (model == nullptr) {
    std::printf("FAIL: upload: %s\n", error.c_str());
    return 1;
  }
  auto exec = q::Executor::Create(*model, &error);
  if (exec == nullptr) {
    std::printf("FAIL: executor: %s\n", error.c_str());
    return 1;
  }

  if (const char* steps_path = std::getenv("GUFO_GLM5_NEXT_STEPS");
      steps_path != nullptr && steps_path[0] != '\0') {
    g_steps = std::fopen(steps_path, "w");
    if (g_steps == nullptr) {
      std::printf("FAIL: cannot open steps file %s\n", steps_path);
      return 1;
    }
    std::fprintf(g_steps, "{\"cases\":[\n");
  }

  // Optional executor trunk-anchor dump for bisection against a reference
  // (GUFO_GLM5_NEXT_DBG_DUMP): raw f32 files for the first Step only.
  if (const char* dbg_dir = std::getenv("GUFO_GLM5_NEXT_DBG_DUMP");
      dbg_dir != nullptr && dbg_dir[0] != '\0') {
    static_cast<void>(mkdir(dbg_dir, 0755));
    g_dbg_dir = dbg_dir;
  }

  for (const gufo::json::Value& c : doc.find("cases")->items()) {
    RunCase(*exec, c);
  }
  if (g_steps != nullptr) {
    std::fprintf(g_steps, "]\n}\n");
    std::fclose(g_steps);
  }
  if (failures != 0) {
    std::printf("FAILURES: %d\n", failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
