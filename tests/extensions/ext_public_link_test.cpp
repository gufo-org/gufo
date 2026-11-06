// An external extension's view of the public link targets: only public headers,
// and only gufo_ext_runtime with the gufo_ext_* contracts on the link line. A
// public service whose implementation drops out of them fails to link here.
#include <cstdint>
#include <cstdlib>
#include <gufo/ext/cache.hpp>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/sha256.hpp>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/experimental/host_allocations.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/transfer_pool.hpp>
#endif

// Linking the implementation must not publish the private headers behind it.
#if __has_include("src/core/sampling_logit_row.hpp")
#error "a private header is reachable through the public link targets"
#endif

namespace {

void Check(bool condition, const char* what) {
  if (!condition) {
    std::cerr << "failed: " << what << '\n';
    std::exit(1);
  }
}

template<typename T>
void Referenced(T value) {
  volatile T sink = value;
  Check(sink != nullptr, "referenced symbol");
}

}  // namespace

int main() {
  using namespace gufo;

  Check(crypto::Sha256Hex(std::span<const std::uint8_t>{}) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256");

  std::string error;
  Check(!core::GgufReader::OpenMemory("", 0, &error), "gguf reader");
  Referenced(&core::GgufIdentityHex);

  const float source = 1.5f;
  float row = 0.f;
  Check(quant::DequantizeRow(core::GgmlType::kF32, &source, &row, 1) &&
            row == source,
        "dequantization");

  Check(sampling::JsonConstraint::Object() != nullptr, "json constraint");
  sampling::SamplerState sampler;
  sampling::SamplerState copy = sampler;
  (void)copy;

  const std::uint8_t not_an_image = 0;
  bool rejected = false;
  try {
    (void)core::DecodeImage(std::span(&not_an_image, 1));
  } catch (const std::exception&) {
    rejected = true;
  }
  Check(rejected, "image decoding");

  const cache::InputIdentity identity;
  Check(identity.TokenCount() == 0, "cache identity");
  Referenced(&cache::InputIdentity::Through);

#if defined(ENGINE_ENABLE_HIP)
  // These need a device to run, so the test only resolves them.
  Referenced(&hip::WeightUpload::Create);
  Referenced(&hip::TransferPool::TryAcquire);
  Referenced(&hip::CreateHipComponentCacheResources);
  auto allocate = static_cast<hipError_t (hip::HostAllocations::*)(
      void**, std::size_t, unsigned)>(&hip::HostAllocations::Allocate);
  Referenced(allocate);
#endif
  return 0;
}
