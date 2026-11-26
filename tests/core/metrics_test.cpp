#include "src/core/metrics.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace metrics = gufo::core::metrics;

namespace {

void Require(bool value, const char* what) {
  if (!value) {
    std::cerr << "metrics check failed: " << what << '\n';
    std::exit(1);
  }
}

template<class Call>
bool ThrowsInvalid(Call&& call) {
  try {
    call();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

std::string Ostream(double value) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return out.str();
}

void TestRegistrationAndRender() {
  metrics::Registry registry;
  auto counter = registry.Add<metrics::Counter>("a_total", "Counts\nthings");
  auto seconds = registry.Add<metrics::FloatCounter>("b_seconds_total", "s");
  auto high = registry.Add<metrics::MaxCounter>("c_max", "m");
  auto bytes = registry.Add<metrics::UintGauge>("d_bytes", "b");
  auto ratio = registry.Add<metrics::Gauge>("e_ratio", "r");
  auto level = registry.Add<metrics::IntGauge>("f_level", "l");
  const double bounds[] = {0.5, 1.0};
  auto latency =
      registry.AddFamily<metrics::Histogram>("g_seconds", "g", bounds);
  auto ok = latency.Add({{"status", "ok"}});
  counter.Add(2);
  counter.Add(3);
  seconds.Add(0.5);
  seconds.Add(0.25);
  high.Observe(7);
  high.Observe(3);
  bytes.Set(4096);
  ratio.Set(0.1);
  level.Add(5);
  level.Add(-8);
  ok.Observe(0.25);
  ok.Observe(2.0);
  Require(registry.Render() ==
              "# HELP a_total Counts\\nthings\n# TYPE a_total counter\n"
              "a_total 5\n"
              "# HELP b_seconds_total s\n# TYPE b_seconds_total counter\n"
              "b_seconds_total 0.75\n"
              "# HELP c_max m\n# TYPE c_max counter\nc_max 7\n"
              "# HELP d_bytes b\n# TYPE d_bytes gauge\nd_bytes 4096\n"
              "# HELP e_ratio r\n# TYPE e_ratio gauge\n"
              "e_ratio 0.10000000000000001\n"
              "# HELP f_level l\n# TYPE f_level gauge\nf_level -3\n"
              "# HELP g_seconds g\n# TYPE g_seconds histogram\n"
              "g_seconds_bucket{status=\"ok\",le=\"0.5\"} 1\n"
              "g_seconds_bucket{status=\"ok\",le=\"1\"} 1\n"
              "g_seconds_bucket{status=\"ok\",le=\"+Inf\"} 2\n"
              "g_seconds_sum{status=\"ok\"} 2.25\n"
              "g_seconds_count{status=\"ok\"} 2\n",
          "exposition text");
}

void TestRegistrationErrors() {
  metrics::Registry registry;
  const auto family = registry.AddFamily<metrics::Counter>("x_total", "x");
  Require(ThrowsInvalid([&] { registry.Add<metrics::Gauge>("x_total", "x"); }),
          "duplicate family name throws");
  Require(ThrowsInvalid([&] { registry.Add<metrics::Gauge>("bad name", "x"); }),
          "invalid name throws");
  const double bounds[] = {2.0, 1.0};
  Require(ThrowsInvalid([&] {
            registry.AddFamily<metrics::Histogram>("h", "h", bounds);
          }),
          "unsorted bounds throw");
  Require(
      ThrowsInvalid([&] { registry.AddFamily<metrics::Histogram>("h2", "h"); }),
      "a histogram without bounds throws");
  family.Add({{"k", "v"}});
  Require(ThrowsInvalid([&] { family.Add({{"k", "v"}}); }),
          "duplicate label set throws");
  Require(ThrowsInvalid([&] { family.Add({{"1k", "v"}}); }),
          "invalid label throws");
  std::string message;
  try {
    family.Add({{"k", "v"}});
  } catch (const std::invalid_argument& error) {
    message = error.what();
  }
  Require(message.find("x_total{k=\"v\"}") != std::string::npos,
          "a series error names the family and labels");
  metrics::Counter unregistered;
  unregistered.Add(1);
}

void TestVisibility() {
  metrics::Registry registry;
  auto shown = registry.Add<metrics::Counter>("shown_total", "s");
  const auto optional =
      registry.AddFamily<metrics::UintGauge>("opt_bytes", "o");
  optional.Add().Set(9);
  optional.visibility().Set(false);
  Require(registry.Render() ==
              "# HELP shown_total s\n# TYPE shown_total counter\n"
              "shown_total 0\n",
          "a hidden family is absent");
  optional.visibility().Set(true);
  Require(registry.Render().find("opt_bytes 9\n") != std::string::npos,
          "a shown family returns");
}

void TestFloatFormatMatchesOstream() {
  metrics::Registry registry;
  auto gauge = registry.Add<metrics::Gauge>("v", "v");
  const auto check = [&](double value) {
    gauge.Set(value);
    const std::string expected =
        "# HELP v v\n# TYPE v gauge\nv " + Ostream(value) + "\n";
    if (registry.Render() != expected) {
      std::cerr << "got " << registry.Render() << "want " << expected;
      Require(false, "float text matches ostream");
    }
  };
  for (const double value :
       {0.0, -0.0, 1.0, 0.1, 1e-9, 1234.5678, 1e16, 1e17, 1e20, 1e300, 5e-324,
        std::numeric_limits<double>::max(), std::numeric_limits<double>::min(),
        0.0001, 0.00001, 123456.789}) {
    check(value);
    check(-value);
  }
  std::mt19937_64 random(12345);
  for (int i = 0; i < 100000; ++i) {
    const double value = std::bit_cast<double>(random());
    if (std::isfinite(value))
      check(value);
  }
  for (int i = 0; i < 100000; ++i) {
    const double scale = std::pow(10.0, static_cast<int>(random() % 40) - 20);
    check(static_cast<double>(random() % 1000000) * scale);
  }
}

void TestConcurrentUpdates() {
  metrics::Registry registry;
  auto count = registry.Add<metrics::Counter>("n_total", "n");
  auto seconds = registry.Add<metrics::FloatCounter>("s_total", "s");
  auto level = registry.Add<metrics::IntGauge>("l", "l");
  constexpr int kThreads = 8;
  constexpr int kPerThread = 50000;
  std::vector<std::jthread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([=] {
      for (int i = 0; i < kPerThread; ++i) {
        count.Add(1);
        seconds.Add(0.25);
        level.Add(1);
        level.Add(-1);
      }
    });
  }
  workers.clear();
  const std::string text = registry.Render();
  Require(text.find("n_total 400000\n") != std::string::npos, "no lost adds");
  Require(text.find("s_total 100000\n") != std::string::npos,
          "no lost float adds");
  Require(text.find("\nl 0\n") != std::string::npos, "gauge balances");
}

}  // namespace

int main() {
  TestRegistrationAndRender();
  TestRegistrationErrors();
  TestVisibility();
  TestFloatFormatMatchesOstream();
  TestConcurrentUpdates();
  std::cout << "Metrics wrapper checks passed.\n";
}
