#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen_image_21/hip/runtime.hpp"
#include "src/models/qwen_image_21/lora.hpp"
#include "src/models/qwen_image_21/weights.hpp"

namespace {

using namespace gufo::models::qwen_image_21;
using Terms = std::map<std::string, std::vector<LoraTerm>, std::less<>>;
using Hashes = std::map<std::string, std::string>;

std::pair<std::string, std::filesystem::path> Spec(std::string_view value) {
  const auto equals = value.find('=');
  if (equals == 0 || equals == std::string_view::npos ||
      equals + 1 == value.size())
    throw std::invalid_argument("LoRA options expect NAME=PATH");
  return {std::string(value.substr(0, equals)),
          std::filesystem::path(value.substr(equals + 1))};
}

LoraAdapter Load(const Weights& weights, const std::string& name,
                 const std::filesystem::path& path) {
  LoraAdapter adapter(path, [&](std::string_view tensor) {
    return weights.Contains(tensor) ? weights.Get(tensor).shape
                                    : std::vector<int>{};
  });
  std::cout << "lora " << name << " format=" << adapter.format()
            << " targets=" << adapter.patches().size()
            << " rank=" << adapter.min_rank() << '-' << adapter.max_rank()
            << '\n';
  return adapter;
}

Terms Select(const LoraAdapter* adapter, float strength) {
  Terms terms;
  if (adapter)
    for (const auto& patch : adapter->patches())
      terms[patch.target].push_back({&patch, strength});
  return terms;
}

// Same sequence as a per-request switch: drop, rebuild changed targets.
// Returns the time and the number of rebuilt weights.
std::pair<double, std::size_t> Switch(hip::Runtime& rt, Terms terms) {
  const auto start = std::chrono::steady_clock::now();
  const auto names = rt.SetLoras(std::move(terms));
  for (const auto& name : names)
    rt.Weight(name);
  rt.Synchronize();
  return {std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count(),
          names.size()};
}

// Row-major values of a weight; PackDense16 stores 16-column panels, each
// row-major: packed[(c / 16) * rows * 16 + r * 16 + c % 16] = w[r][c].
std::vector<float> Unpacked(const hip::Matrix& matrix,
                            const std::vector<float>& values) {
  if (!matrix.packed)
    return values;
  const auto rows = static_cast<std::size_t>(matrix.rows);
  const auto cols = static_cast<std::size_t>(matrix.cols);
  if (cols % 16 != 0 || values.size() != rows * cols)
    throw std::logic_error("unexpected packed weight layout");
  std::vector<float> result(values.size());
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      result[r * cols + c] = values[c / 16 * rows * 16 + r * 16 + c % 16];
  return result;
}

Hashes Hash(hip::Runtime& rt, const std::set<std::string>& names) {
  Hashes hashes;
  for (const auto& name : names) {
    const auto values = rt.Download(rt.Weight(name));
    hashes[name] = gufo::crypto::Sha256Hex(
        std::span(reinterpret_cast<const std::uint8_t*>(values.data()),
                  values.size() * sizeof(float)));
  }
  return hashes;
}

bool Report(std::string_view check, const Hashes& actual,
            const Hashes& expected, const std::set<std::string>& names) {
  std::size_t mismatches = 0;
  for (const auto& name : names)
    if (actual.at(name) != expected.at(name)) {
      if (mismatches++ == 0)
        std::cout << "first mismatch " << name << '\n';
    }
  std::cout << (mismatches == 0 ? "PASS " : "FAIL ") << check
            << " tensors=" << names.size() << " mismatches=" << mismatches
            << '\n';
  return mismatches == 0;
}

int SwitchCheck(const Weights& weights, std::string_view first,
                std::string_view second, float strength) {
  const auto [name_a, path_a] = Spec(first);
  const auto [name_b, path_b] = Spec(second);
  const auto a = Load(weights, name_a, path_a);
  const auto b = Load(weights, name_b, path_b);
  std::set<std::string> names, only_a;
  for (const auto& patch : a.patches())
    names.insert(patch.target);
  only_a = names;
  for (const auto& patch : b.patches()) {
    names.insert(patch.target);
    only_a.erase(patch.target);
  }
  Hashes base;
  {
    hip::Runtime fresh(weights);
    base = Hash(fresh, names);
  }
  std::set<std::string> targets_a, targets_b;
  for (const auto& patch : a.patches())
    targets_a.insert(patch.target);
  for (const auto& patch : b.patches())
    targets_b.insert(patch.target);
  const auto count = [](std::string_view check, std::size_t actual,
                        std::size_t expected) {
    std::cout << (actual == expected ? "PASS " : "FAIL ") << check
              << " rebuilt=" << actual << " expected=" << expected << '\n';
    return actual == expected;
  };
  hip::Runtime rt(weights);
  // A cold runtime builds every adapter target during the switch.
  const auto none_to_a = Switch(rt, Select(&a, strength));
  bool pass = count("cold_switch_builds_targets", none_to_a.second,
                    targets_a.size());
  const auto with_a = Hash(rt, names);
  const auto a_to_b = Switch(rt, Select(&b, strength));
  const auto with_b = Hash(rt, names);
  std::size_t changed = 0;
  for (const auto& patch : b.patches())
    changed += with_b.at(patch.target) != base.at(patch.target);
  const auto b_to_a = Switch(rt, Select(&a, strength));
  const auto again = Hash(rt, names);
  // Adding B to A rebuilds only B's targets; A-only weights stay as they are.
  auto both = Select(&a, strength);
  for (auto& [target, terms] : Select(&b, strength))
    both[target].insert(both[target].end(), terms.begin(), terms.end());
  const auto a_to_ab = Switch(rt, std::move(both));
  const auto with_ab = Hash(rt, names);
  pass = count("a_to_ab_rebuilds_b_targets", a_to_ab.second,
               targets_b.size()) &&
         pass;
  const auto ab_to_none = Switch(rt, {});
  const auto none = Hash(rt, names);
  pass = Report("b_restores_a_only_targets", with_b, base, only_a) && pass;
  pass = Report("a_b_a_identical", again, with_a, names) && pass;
  pass = Report("ab_keeps_a_only_targets", with_ab, with_a, only_a) && pass;
  pass = Report("none_matches_fresh_runtime", none, base, names) && pass;
  std::cout << "b_changed_targets=" << changed << '/' << b.patches().size()
            << '\n'
            << "switch_ms none_to_a=" << none_to_a.first
            << " a_to_b=" << a_to_b.first << " b_to_a=" << b_to_a.first
            << " a_to_ab=" << a_to_ab.first
            << " ab_to_none=" << ab_to_none.first << '\n';
  return pass ? 0 : 1;
}

int Dump(const std::filesystem::path& model, const Weights& weights,
         const std::filesystem::path& output, std::string_view spec,
         float strength, const std::vector<std::string>& filters) {
  const auto [name, path] = Spec(spec);
  const auto adapter = Load(weights, name, path);
  const auto wanted = [&](std::string_view target) {
    if (filters.empty())
      return true;
    for (const auto& filter : filters)
      if (target.find(filter) != std::string_view::npos)
        return true;
    return false;
  };
  std::set<std::string> targets;
  for (const auto& patch : adapter.patches())
    if (wanted(patch.target))
      targets.insert(patch.target);
  // Filters also select unpatched tensors, which must stay equal to base.
  // Both are matched by their full `transformer.` name.
  if (!filters.empty()) {
    const auto patched = targets.size();
    const auto manifest = ReadJson(
        model / "transformer/diffusion_pytorch_model.safetensors.index.json");
    for (const auto& [tensor, file] : manifest.find("weight_map")->members()) {
      (void)file;
      if (const auto target = "transformer." + tensor; wanted(target))
        targets.insert(target);
    }
    if (targets.size() == patched)
      std::cerr << "warning: --tensor selects no unpatched tensor; the "
                   "unpatched control is skipped\n";
  }
  if (targets.empty())
    throw std::invalid_argument("no tensor selected");
  std::filesystem::create_directories(output);
  hip::Runtime rt(weights);
  rt.SetLoras(Select(&adapter, strength));
  std::ofstream index(output / "index.json");
  index << '{';
  const char* separator = "\n";
  for (const auto& target : targets) {
    const auto matrix = rt.Weight(target);
    const auto values = Unpacked(matrix, rt.Download(matrix));
    std::ofstream stream(output / (target + ".f32"), std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(float));
    if (!stream)
      throw std::runtime_error("cannot write " + target);
    index << separator << "  \"" << target << "\": [" << matrix.rows << ", "
          << matrix.cols << ']';
    separator = ",\n";
    std::cout << target << ' ' << matrix.rows << 'x' << matrix.cols << '\n';
  }
  index << "\n}\n";
  if (!index)
    throw std::runtime_error("cannot write index.json");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const auto usage = [] {
    std::cerr << "usage: qwen_image_21_lora_dump MODEL_DIR OUTPUT_DIR "
                 "--lora NAME=PATH [--scale S] [--tensor SUBSTR]...\n"
                 "       qwen_image_21_lora_dump MODEL_DIR --switch-check "
                 "NAME_A=PATH_A NAME_B=PATH_B [--scale S]\n";
    return 2;
  };
  if (argc < 4)
    return usage();
  try {
    const std::filesystem::path model(argv[1]);
    const bool check = std::string_view(argv[2]) == "--switch-check";
    std::string lora, check_a, check_b;
    std::vector<std::string> filters;
    float strength = 1;
    int i = 3;
    if (check) {
      if (argc < 5)
        return usage();
      check_a = argv[3];
      check_b = argv[4];
      i = 5;
    }
    for (; i < argc; ++i) {
      const std::string_view option(argv[i]);
      if (!check && option == "--lora" && i + 1 < argc)
        lora = argv[++i];
      else if (option == "--scale" && i + 1 < argc)
        strength = std::stof(argv[++i]);
      else if (!check && option == "--tensor" && i + 1 < argc)
        filters.emplace_back(argv[++i]);
      else
        return usage();
    }
    if (!check && lora.empty())
      return usage();
    const Weights weights(model);
    return check ? SwitchCheck(weights, check_a, check_b, strength)
                 : Dump(model, weights, argv[2], lora, strength, filters);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
