#include <cstdint>
#include <gufo/ext/families/qwen/models/flash_next/ngram.hpp>
#include <gufo/ext/families/qwen/models/flash_next/runner.hpp>
#include <gufo/ext/families/qwen/models/flash_next/stages.hpp>
#include <gufo/ext/families/qwen/models/flash_next/weight_source.hpp>
#include <gufo/ext/families/qwen/text/tokenizer.hpp>
#include <gufo/ext/registry.hpp>
#include <memory>
#include <string>
#include <typeinfo>

#include "gufo/ext/generated/module_dependent_fixture_constants.hpp"

namespace {
namespace flash_next = gufo::ext::qwen::flash_next;
namespace qfn = gufo::models::qwen38_flash_next;

constexpr const char* kRefused = "module weight source refused";
bool g_called = false;

bool Refuse(std::string* error) {
  g_called = true;
  if (error != nullptr)
    *error = kRefused;
  return false;
}

class FixtureStages final : public flash_next::Stages {};

// Every entry fails with a marker, so the runner reports the module's error.
class FixtureWeights final : public flash_next::WeightSource {
public:
  bool Probe(const std::string&) const override { return true; }
  bool Open(const std::string&, const std::string&,
            std::string* error) override {
    return Refuse(error);
  }
  bool BindTrunk(flash_next::ModelWeights*, std::string* error) override {
    return Refuse(error);
  }
  bool BindMtp(const flash_next::Config&, flash_next::MtpWeights*,
               std::string* error) override {
    return Refuse(error);
  }
  std::unique_ptr<gufo::tokenization::QwenTokenizer> Tokenizer(
      const flash_next::Config&, std::string* error) override {
    Refuse(error);
    return nullptr;
  }
  std::unique_ptr<flash_next::NgramTable> Ngram(const flash_next::ModelWeights&,
                                                std::string* error) override {
    Refuse(error);
    return nullptr;
  }
  std::span<const gufo::core::GgufMappedRegion> Regions(bool) const override {
    return {};
  }
  std::string ModelName() const override { return "module-dependent-fixture"; }
  std::uint32_t NativeContextLength(const std::string&) const override {
    return 4096;
  }
};
}  // namespace

// Registers only when the executable's runner dispatches into the module's
// subclasses and their RTTI agrees across the boundary.
void module_dependent_fixture_hip_register(gufo::ext::Registry& registry) {
  std::unique_ptr<flash_next::Stages> stages =
      std::make_unique<FixtureStages>();
  const bool rtti = dynamic_cast<FixtureStages*>(stages.get()) != nullptr &&
                    typeid(*stages) == typeid(FixtureStages);
  gufo::ext::TextRunnerOptions options;
  options.model_path = "/nonexistent/module-dependent-fixture";
  std::string error;
  const auto runner = qfn::CreateQwenFlashNextTextRunner(
      options, std::move(stages), std::make_unique<FixtureWeights>(), &error);
  if (rtti && runner == nullptr && g_called &&
      error.find(kRefused) != std::string::npos) {
    registry.AddProvider(
        gufo::ext::generated::extension_module_dependent_fixture::
            kProvider_module_dependent_fixture_hip);
  }
}
