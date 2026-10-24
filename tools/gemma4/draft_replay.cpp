// Offline replay of the Gemma 4 calibrated draft policy over --draft-trace
// records, through the production DraftCalibration and CalibratedChain.
//
//   gemma4_draft_replay validate [--experts] [--min-drafts N] TRACE.jsonl
//     Rebuilds the shared calibration from the recorded observations and
//     checks every recorded estimate, decision margin, stop decision, sibling
//     estimate and expected token count of single-session cycles bit for
//     bit. Exit 1 on any mismatch.
//
//   gemma4_draft_replay simulate [--experts] [--replicas N] [--seed S]
//                                TRACE.jsonl...
//     Evaluates draft policies on complete chains: cycles of a run with
//     --draft-policy fixed --draft-tokens 15 whose 15 drafter proposals were
//     all verified (no copies). Each row's acceptance label is conditional on
//     the rows before it and each runner-up carries its sibling hit
//     probability, so a policy's expected tokens per cycle are exact given
//     its decisions. Costs come from the policy's price tables. Policies
//     learn online from outcomes drawn from the labels (averaged over
//     replicas): the production rule (with and without siblings, and
//     frozen at its prior), position-aware and forgetting variants, fixed
//     lengths 0..15 and a per-cycle oracle over fixed lengths. The
//     production rule is also decided through a replica of its chain
//     arithmetic, which must agree on every chain (exit 1 otherwise).
//     --decide-scale S prices verification for the policies' decisions and
//     --score-scale S for the time each cycle is scored at (both default 1,
//     the table): S adds (S - 1) times the depth-0 verification row at every
//     depth, the projections of a target S times as heavy as the table's.
//     Prints one JSON object.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/gemma4/draft_policy.hpp"

namespace g4 = gufo::models::gemma4;
using gufo::json::Value;

namespace {

std::vector<float> Floats(const Value& v, const char* key) {
  std::vector<float> out;
  if (const Value* a = v.find(key); a != nullptr && a->is_array()) {
    for (const Value& x : a->items()) {
      out.push_back(static_cast<float>(x.as_double()));
    }
  }
  return out;
}

std::vector<bool> Bools(const Value& v, const char* key) {
  std::vector<bool> out;
  if (const Value* a = v.find(key); a != nullptr && a->is_array()) {
    for (const Value& x : a->items()) {
      out.push_back(x.as_bool());
    }
  }
  return out;
}

std::vector<Value> Read(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  std::vector<Value> records;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) {
      records.push_back(gufo::json::parse(line));
    }
  }
  return records;
}

struct Options {
  bool experts{false};
  std::uint32_t min_drafts{1};
  int replicas{8};
  std::uint64_t seed{1};
  float decide_scale{1.0F};
  float score_scale{1.0F};
  std::vector<std::string> files;
};

/// The table's costs at `context` with verification scaled as for a target
/// `scale` times as heavy (Options::decide_scale, score_scale).
g4::DraftCosts CostsAt(std::uint32_t context, bool experts, float scale) {
  g4::DraftCosts costs = g4::DraftCostsAt(context, experts);
  if (scale != 1.0F) {
    const g4::DraftCosts flat = g4::DraftCostsAt(0, experts);
    for (std::size_t r = 0; r < costs.verify.size(); ++r) {
      costs.verify[r] += (scale - 1.0F) * flat.verify[r];
    }
  }
  return costs;
}

// ---------------------------------------------------------------- validate

int Validate(const Options& opt) {
  std::array<g4::DraftCalibration, 4> tables;  // as Session::Calibration
  std::size_t cycles = 0;
  std::size_t checked = 0;
  std::size_t mismatches = 0;
  const auto mismatch = [&](std::size_t line, const std::string& what) {
    if (++mismatches <= 20) {
      std::cerr << "cycle " << line << ": " << what << '\n';
    }
  };
  for (const std::string& file : opt.files) {
    for (const Value& r : Read(file)) {
      ++cycles;
      const bool sampled = r.find("sampled")->as_bool();
      g4::DraftCalibration& drafts = tables[sampled ? 1 : 0];
      g4::DraftCalibration& siblings = tables[sampled ? 3 : 2];
      const std::vector<float> signals = Floats(r, "signals");
      const std::vector<bool> observed = Bools(r, "observed");
      const Value* stop = r.find("stop");
      const bool alone = r.member_size("sessions") == 1;
      if (alone && (!signals.empty() || (stop && !stop->is_null()))) {
        ++checked;
        const g4::DraftCosts costs = g4::DraftCostsAt(
            static_cast<std::uint32_t>(r.member_size("position")), opt.experts);
        g4::CalibratedChain chain(
            drafts, costs, opt.min_drafts,
            static_cast<std::uint32_t>(r.member_size("step_cap")));
        const std::vector<float> estimates = Floats(r, "estimates");
        const std::vector<float> margins = Floats(r, "margins");
        for (std::size_t i = 0; i < signals.size(); ++i) {
          if (drafts.Estimate(signals[i]) != estimates[i]) {
            mismatch(cycles, "estimate " + std::to_string(i));
          }
          if (!chain.Include(signals[i])) {
            mismatch(cycles, "draft " + std::to_string(i) + " not included");
          }
          const auto d = chain.LastDecision();
          if ((d.compared ? d.gain - d.cost : 0.0F) != margins[i]) {
            mismatch(cycles, "margin " + std::to_string(i));
          }
        }
        if (stop && !stop->is_null()) {
          const auto signal = static_cast<float>(stop->member_double("signal"));
          if (chain.Include(signal)) {
            mismatch(cycles, "stop not reproduced");
          }
          const auto d = chain.LastDecision();
          if (d.gain != static_cast<float>(stop->member_double("gain")) ||
              d.cost != static_cast<float>(stop->member_double("cost"))) {
            mismatch(cycles, "stop gain/cost");
          }
        }
        const std::vector<float> sibling_estimates =
            Floats(r, "sibling_estimates");
        const std::vector<float> up_signals = Floats(r, "runner_up_signals");
        auto rows = static_cast<std::uint32_t>(r.member_size("rows") -
                                               sibling_estimates.size());
        for (std::size_t i = 0; i < sibling_estimates.size(); ++i) {
          const float hit = siblings.Estimate(up_signals[i]);
          if (hit != sibling_estimates[i]) {
            mismatch(cycles, "sibling estimate " + std::to_string(i));
          }
          if (!chain.IncludeSibling(static_cast<std::uint32_t>(i), rows++,
                                    hit)) {
            mismatch(cycles, "sibling " + std::to_string(i) + " not included");
          }
        }
        if (chain.Expected() !=
            static_cast<float>(r.member_double("expected"))) {
          mismatch(cycles, "expected tokens");
        }
      }
      for (std::size_t j = 0; j < observed.size(); ++j) {
        drafts.Observe(signals[j], observed[j]);
      }
      if (const Value* s = r.find("sibling_observed"); s && !s->is_null()) {
        const std::vector<float> up_signals = Floats(r, "runner_up_signals");
        siblings.Observe(up_signals[r.member_size("accepted")], s->as_bool());
      }
    }
  }
  Value out = Value::object();
  out["cycles"] = cycles;
  out["checked"] = checked;
  out["mismatches"] = mismatches;
  std::cout << out.dump() << '\n';
  return mismatches == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- simulate

constexpr std::size_t kChain = 15;
constexpr std::size_t kMaxRows = 16;  // a lone cycle's verification rows

struct Chain {
  std::uint32_t position{0};
  bool sampled{false};
  std::array<float, kChain> signal{};
  std::array<float, kChain> accept{};  // conditional acceptance labels
  std::array<bool, kChain> has_up{};
  std::array<float, kChain> up_signal{};
  std::array<float, kChain> up_accept{};  // sibling hit after a rejection
};

/// Acceptance estimates a policy decides with, and what it learns from.
class Estimator {
public:
  virtual ~Estimator() = default;
  [[nodiscard]] virtual float Draft(float signal, std::size_t position,
                                    bool sampled) const = 0;
  virtual void ObserveDraft(float signal, std::size_t position, bool sampled,
                            bool accepted) = 0;
  [[nodiscard]] float Sibling(float signal, bool sampled) const {
    return siblings_[sampled ? 1 : 0].Estimate(signal);
  }
  void ObserveSibling(float signal, bool sampled, bool hit) {
    if (learn_) {
      siblings_[sampled ? 1 : 0].Observe(signal, hit);
    }
  }

protected:
  explicit Estimator(bool learn) : learn_(learn) {}
  bool learn_;

private:
  std::array<g4::DraftCalibration, 2> siblings_;
};

/// The production tables (frozen: never learns, the prior alone).
class Production final : public Estimator {
public:
  explicit Production(bool learn) : Estimator(learn) {}
  [[nodiscard]] float Draft(float signal, std::size_t, bool sampled) const override {
    return tables_[sampled ? 1 : 0].Estimate(signal);
  }
  void ObserveDraft(float signal, std::size_t, bool sampled,
                    bool accepted) override {
    if (learn_) {
      tables_[sampled ? 1 : 0].Observe(signal, accepted);
    }
  }
  [[nodiscard]] const g4::DraftCalibration& table(bool sampled) const {
    return tables_[sampled ? 1 : 0];
  }

private:
  std::array<g4::DraftCalibration, 2> tables_;
};

/// Bins per draft position group (0, 1, 2-3, 4-7, 8+), each group shrunk
/// toward the shallower group's estimate with `weight` observations, the
/// first toward the bin midpoint (group 0 is then the production table).
/// `half_life` > 0 forgets observations exponentially (per observation).
class Grouped final : public Estimator {
public:
  Grouped(float weight, bool by_position, float half_life)
      : Estimator(true),
        weight_(weight),
        by_position_(by_position),
        decay_(half_life > 0.0F ? std::exp2(-1.0F / half_life) : 1.0F) {}
  [[nodiscard]] float Draft(float signal, std::size_t position,
                            bool sampled) const override {
    const std::size_t bin = g4::DraftCalibration::Bin(signal);
    const auto& t = tables_[sampled ? 1 : 0];
    float estimate = (static_cast<float>(bin) + 0.5F) /
                     static_cast<float>(g4::DraftCalibration::kBins);
    for (std::size_t g = 0; g <= Group(position); ++g) {
      estimate = (t.accepted[g][bin] + weight_ * estimate) /
                 (t.seen[g][bin] + weight_);
    }
    return estimate;
  }
  void ObserveDraft(float signal, std::size_t position, bool sampled,
                    bool accepted) override {
    auto& t = tables_[sampled ? 1 : 0];
    if (decay_ < 1.0F) {
      for (std::size_t g = 0; g < kGroups; ++g) {
        for (std::size_t b = 0; b < g4::DraftCalibration::kBins; ++b) {
          t.accepted[g][b] *= decay_;
          t.seen[g][b] *= decay_;
        }
      }
    }
    const std::size_t bin = g4::DraftCalibration::Bin(signal);
    const std::size_t g = Group(position);
    t.accepted[g][bin] += accepted ? 1.0F : 0.0F;
    t.seen[g][bin] += 1.0F;
  }

private:
  static constexpr std::size_t kGroups = 5;
  [[nodiscard]] std::size_t Group(std::size_t position) const {
    if (!by_position_) {
      return 0;
    }
    return position == 0 ? 0 : position == 1 ? 1 : position < 4 ? 2
                           : position < 8    ? 3
                                             : 4;
  }
  struct Tables {
    std::array<std::array<float, g4::DraftCalibration::kBins>, kGroups>
        accepted{};
    std::array<std::array<float, g4::DraftCalibration::kBins>, kGroups> seen{};
  };
  float weight_;
  bool by_position_;
  float decay_;
  std::array<Tables, 2> tables_;
};

/// CalibratedChain's arithmetic for a lone cycle, with the estimates given.
class ReplicaChain {
public:
  ReplicaChain(const g4::DraftCosts& costs, std::uint32_t min_drafts,
               std::uint32_t cap)
      : costs_(costs), min_drafts_(std::max(1U, min_drafts)), cap_(cap) {}
  bool Include(float accept) {
    const std::uint32_t index = kept_;
    if (index >= cap_) {
      return false;
    }
    steps_ = index + 1;
    const float survival = survival_ * accept;
    if (index >= min_drafts_) {
      const float current = 0.0F + costs_.draft[index + 1] + Verify(index + 1);
      const float next =
          0.0F + costs_.draft[std::min(index + 2, cap_)] + Verify(index + 2);
      if (survival * current < (0.0F + expected_) * (next - current)) {
        return false;
      }
    }
    reach_[kept_] = survival_;
    accept_[kept_] = accept;
    survival_ = survival;
    expected_ += survival;
    ++kept_;
    return true;
  }
  bool IncludeSibling(std::uint32_t depth, std::uint32_t rows, float hit) {
    if (depth >= kept_) {
      return false;
    }
    const float gain = reach_[depth] * (1.0F - accept_[depth]) * hit;
    const float current = 0.0F + costs_.draft[steps_] + Verify(rows);
    const float next = current - Verify(rows) + Verify(rows + 1);
    if (gain * current < (0.0F + expected_) * (next - current)) {
      return false;
    }
    expected_ += gain;
    return true;
  }
  [[nodiscard]] std::uint32_t kept() const { return kept_; }
  [[nodiscard]] std::uint32_t steps() const { return steps_; }

private:
  [[nodiscard]] float Verify(std::uint32_t rows) const {
    return costs_.verify[std::min<std::size_t>(rows, costs_.verify.size() - 1)];
  }
  const g4::DraftCosts& costs_;
  std::uint32_t min_drafts_;
  std::uint32_t cap_;
  std::uint32_t kept_{0};
  std::uint32_t steps_{0};
  std::array<float, 16> reach_{};
  std::array<float, 16> accept_{};
  float survival_{1.0F};
  float expected_{1.0F};
};

struct Tally {
  double tokens{0.0};
  double ms{0.0};
  double drafts{0.0};
  double siblings{0.0};
  double cycles{0.0};
  void Add(double t, double m, double d, double s = 0.0) {
    tokens += t;
    ms += m;
    drafts += d;
    siblings += s;
    cycles += 1.0;
  }
  [[nodiscard]] Value Json() const {
    Value v = Value::object();
    v["tokens_per_s"] = ms > 0.0 ? 1000.0 * tokens / ms : 0.0;
    v["tokens_per_cycle"] = cycles > 0.0 ? tokens / cycles : 0.0;
    v["drafts_per_cycle"] = cycles > 0.0 ? drafts / cycles : 0.0;
    v["siblings_per_cycle"] = cycles > 0.0 ? siblings / cycles : 0.0;
    return v;
  }
};

/// One cycle under a policy: drafts, siblings, expected tokens and priced
/// time; then learning from outcomes drawn from the labels.
struct Outcome {
  std::uint32_t drafts{0};
  std::uint32_t siblings{0};
};
Outcome RunCycle(Estimator& estimator, const Chain& c,
                 const g4::DraftCosts& costs, const Options& opt,
                 bool with_siblings, std::mt19937_64* rng, Tally* tally,
                 const g4::DraftCosts* score = nullptr) {
  ReplicaChain chain(costs, opt.min_drafts, kChain);
  while (chain.kept() < kChain &&
         chain.Include(estimator.Draft(c.signal[chain.kept()], chain.kept(),
                                       c.sampled))) {
  }
  const std::uint32_t k = chain.kept();
  auto rows = k + 1;
  std::uint32_t siblings = 0;
  for (std::uint32_t i = 0; with_siblings && i < k && c.has_up[i] &&
                            rows < kMaxRows;
       ++i) {
    if (!chain.IncludeSibling(i, rows, estimator.Sibling(c.up_signal[i],
                                                         c.sampled))) {
      break;
    }
    ++rows;
    ++siblings;
  }
  double survival = 1.0;
  double tokens = 1.0;
  for (std::uint32_t j = 0; j < k; ++j) {
    if (j < siblings) {
      tokens += survival * (1.0 - c.accept[j]) * c.up_accept[j];
    }
    survival *= c.accept[j];
    tokens += survival;
  }
  const std::uint32_t steps = k < kChain ? k + 1 : k;
  const g4::DraftCosts& scored = score != nullptr ? *score : costs;
  tally->Add(tokens, scored.draft[steps] + scored.verify[rows], k, siblings);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (std::uint32_t j = 0; j < k; ++j) {
    const bool accepted = uniform(*rng) < c.accept[j];
    estimator.ObserveDraft(c.signal[j], j, c.sampled, accepted);
    if (!accepted) {
      if (j < siblings) {
        estimator.ObserveSibling(c.up_signal[j], c.sampled,
                                 uniform(*rng) < c.up_accept[j]);
      }
      break;
    }
  }
  return {k, siblings};
}

/// Estimates against labels, each row weighted by the probability that a
/// chain reaches it (the rows a policy actually decides on).
struct Fit {
  std::array<double, kChain> pos_est{}, pos_label{}, pos_count{};
  std::array<double, g4::DraftCalibration::kBins> bin_est{}, bin_label{},
      bin_count{};
  double brier{0.0}, n{0.0};
  void Add(std::size_t position, float signal, double e, double a,
           double reach) {
    pos_est[position] += reach * e;
    pos_label[position] += reach * a;
    pos_count[position] += reach;
    const std::size_t bin = g4::DraftCalibration::Bin(signal);
    bin_est[bin] += reach * e;
    bin_label[bin] += reach * a;
    bin_count[bin] += reach;
    brier += reach * (e - a) * (e - a);
    n += reach;
  }
  [[nodiscard]] Value Json() const {
    Value v = Value::object();
    v["reach_weighted_brier"] = n > 0 ? brier / n : 0.0;
    Value bins = Value::array();
    for (std::size_t b = 0; b < bin_est.size(); ++b) {
      Value x = Value::object();
      x["bin"] = b;
      x["weight"] = bin_count[b];
      x["mean_estimate"] = bin_count[b] > 0 ? bin_est[b] / bin_count[b] : 0.0;
      x["mean_label"] = bin_count[b] > 0 ? bin_label[b] / bin_count[b] : 0.0;
      bins.push_back(std::move(x));
    }
    v["bins"] = std::move(bins);
    Value pos = Value::array();
    for (std::size_t j = 0; j < kChain; ++j) {
      Value x = Value::object();
      x["position"] = j;
      x["weight"] = pos_count[j];
      x["mean_estimate"] = pos_count[j] > 0 ? pos_est[j] / pos_count[j] : 0.0;
      x["mean_label"] = pos_count[j] > 0 ? pos_label[j] / pos_count[j] : 0.0;
      pos.push_back(std::move(x));
    }
    v["positions"] = std::move(pos);
    return v;
  }
};

int Simulate(const Options& opt) {
  std::vector<Chain> chains;
  std::size_t skipped = 0;
  for (const std::string& file : opt.files) {
    for (const Value& r : Read(file)) {
      const std::vector<float> s = Floats(r, "proposal_signals");
      const std::vector<float> a = Floats(r, "row_acceptance");
      const std::vector<float> us = Floats(r, "runner_up_signals");
      const std::vector<float> ua = Floats(r, "runner_up_acceptance");
      const Value* ups = r.find("runner_ups");
      if (r.member_size("copied") != 0 || s.size() != kChain ||
          a.size() != kChain || us.size() != kChain || ua.size() != kChain ||
          ups == nullptr || ups->size() != kChain) {
        ++skipped;
        continue;
      }
      Chain c;
      c.position = static_cast<std::uint32_t>(r.member_size("position"));
      c.sampled = r.find("sampled")->as_bool();
      for (std::size_t j = 0; j < kChain; ++j) {
        c.signal[j] = s[j];
        c.accept[j] = a[j];
        c.has_up[j] = ups->items()[j].as_double() >= 0.0;
        c.up_signal[j] = us[j];
        c.up_accept[j] = ua[j];
      }
      chains.push_back(c);
    }
  }
  Value out = Value::object();
  out["chains"] = chains.size();
  out["skipped_cycles"] = skipped;
  out["replicas"] = opt.replicas;

  // Fixed lengths (no siblings, as --draft-policy fixed) and the per-cycle
  // oracle over them.
  std::vector<Tally> fixed(kChain + 1);
  Tally oracle;
  std::size_t replica_mismatches = 0;
  {
    Production check(true);
    std::mt19937_64 rng(opt.seed);
    Tally ignored;
    for (const Chain& c : chains) {
      const g4::DraftCosts costs =
          CostsAt(c.position, opt.experts, opt.decide_scale);
      const g4::DraftCosts scored =
          CostsAt(c.position, opt.experts, opt.score_scale);
      double survival = 1.0;
      double tokens = 1.0;
      std::size_t best = 0;
      double best_rate = 0.0;
      for (std::size_t k = 0; k <= kChain; ++k) {
        if (k > 0) {
          survival *= c.accept[k - 1];
          tokens += survival;
        }
        const double ms = scored.draft[k] + scored.verify[k + 1];
        fixed[k].Add(tokens, ms, static_cast<double>(k));
        if (tokens / ms > best_rate) {
          best_rate = tokens / ms;
          best = k;
        }
      }
      double t = 1.0;
      double sv = 1.0;
      for (std::size_t j = 0; j < best; ++j) {
        sv *= c.accept[j];
        t += sv;
      }
      oracle.Add(t, scored.draft[best] + scored.verify[best + 1],
                 static_cast<double>(best));
      // The replica decides as production does.
      g4::CalibratedChain production(check.table(c.sampled), costs,
                                     opt.min_drafts, kChain);
      std::uint32_t k = 0;
      while (k < kChain && production.Include(c.signal[k])) {
        ++k;
      }
      if (RunCycle(check, c, costs, opt, false, &rng, &ignored).drafts != k) {
        ++replica_mismatches;
      }
    }
  }
  out["replica_mismatches"] = replica_mismatches;
  Value fixed_json = Value::array();
  for (std::size_t k = 0; k <= kChain; ++k) {
    Value x = fixed[k].Json();
    x["drafts"] = k;
    fixed_json.push_back(std::move(x));
  }
  out["fixed"] = std::move(fixed_json);
  out["oracle_per_cycle"] = oracle.Json();

  struct Policy {
    std::string name;
    std::function<std::unique_ptr<Estimator>()> make;
    bool siblings;
  };
  std::vector<Policy> policies = {
      {"production", [] { return std::make_unique<Production>(true); }, true},
      {"production_no_siblings",
       [] { return std::make_unique<Production>(true); }, false},
      {"production_frozen", [] { return std::make_unique<Production>(false); },
       true},
  };
  for (const float w : {4.0F, 8.0F, 16.0F, 32.0F}) {
    policies.push_back({"position_w" + std::to_string(static_cast<int>(w)),
                        [w] { return std::make_unique<Grouped>(w, true, 0.0F); },
                        true});
  }
  for (const float h : {128.0F, 512.0F, 2048.0F}) {
    policies.push_back(
        {"recency_h" + std::to_string(static_cast<int>(h)),
         [h] { return std::make_unique<Grouped>(8.0F, false, h); }, true});
    policies.push_back(
        {"position_w8_recency_h" + std::to_string(static_cast<int>(h)),
         [h] { return std::make_unique<Grouped>(8.0F, true, h); }, true});
  }
  Value results = Value::object();
  for (const Policy& policy : policies) {
    Tally tally;
    Fit fit;
    for (int replica = 0; replica < opt.replicas; ++replica) {
      std::mt19937_64 rng(opt.seed + static_cast<std::uint64_t>(replica));
      const auto estimator = policy.make();
      for (const Chain& c : chains) {
        const g4::DraftCosts costs =
            CostsAt(c.position, opt.experts, opt.decide_scale);
        const g4::DraftCosts scored =
            CostsAt(c.position, opt.experts, opt.score_scale);
        if (replica == 0) {
          double reach = 1.0;
          for (std::size_t j = 0; j < kChain; ++j) {
            fit.Add(j, c.signal[j], estimator->Draft(c.signal[j], j, c.sampled),
                    c.accept[j], reach);
            reach *= c.accept[j];
          }
        }
        RunCycle(*estimator, c, costs, opt, policy.siblings, &rng, &tally,
                 &scored);
      }
    }
    Value v = tally.Json();
    v["fit"] = fit.Json();
    results[policy.name] = std::move(v);
  }
  out["policies"] = std::move(results);
  std::cout << out.dump() << '\n';
  return replica_mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: gemma4_draft_replay validate|simulate [--experts] "
                   "[--min-drafts N] [--replicas N] [--seed S] "
                   "[--decide-scale S] [--score-scale S] TRACE...\n";
      return 2;
    }
    const std::string mode = argv[1];
    Options opt;
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--experts") {
        opt.experts = true;
      } else if (arg == "--min-drafts" && i + 1 < argc) {
        opt.min_drafts = static_cast<std::uint32_t>(std::stoul(argv[++i]));
      } else if (arg == "--replicas" && i + 1 < argc) {
        opt.replicas = std::stoi(argv[++i]);
      } else if (arg == "--decide-scale" && i + 1 < argc) {
        opt.decide_scale = std::stof(argv[++i]);
      } else if (arg == "--score-scale" && i + 1 < argc) {
        opt.score_scale = std::stof(argv[++i]);
      } else if (arg == "--seed" && i + 1 < argc) {
        opt.seed = std::stoull(argv[++i]);
      } else {
        opt.files.push_back(arg);
      }
    }
    if (mode == "validate") {
      return Validate(opt);
    }
    if (mode == "simulate") {
      return Simulate(opt);
    }
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
