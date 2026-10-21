#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

struct UploadFault {
  const float* source{};
  void* destination{};
  hipStream_t stream{};
  bool uploaded{};
  bool injected{};
  std::atomic<bool> delay_finished{false};
} fault;

void DelayUpload(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fault.delay_finished = true;
}

void Check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

}  // namespace

extern "C" hipError_t __real_hipMemcpyAsync(void*, const void*, std::size_t,
                                            hipMemcpyKind, hipStream_t);

extern "C" hipError_t __wrap_hipMemcpyAsync(void* destination,
                                            const void* source,
                                            std::size_t bytes,
                                            hipMemcpyKind kind,
                                            hipStream_t stream) {
  if (fault.source && source == fault.source && kind == hipMemcpyHostToDevice) {
    fault.destination = destination;
    fault.stream = stream;
    const auto status = hipLaunchHostFunc(stream, DelayUpload, nullptr);
    if (status != hipSuccess)
      return status;
    fault.uploaded = true;
  } else if (fault.uploaded && !fault.injected &&
             kind == hipMemcpyDeviceToDevice && stream == fault.stream) {
    // Fail the first KV write after the feature upload was queued. This is a
    // recoverable API failure; the GPU and its queued work remain healthy.
    fault.injected = true;
    return hipErrorOutOfMemory;
  }
  return __real_hipMemcpyAsync(destination, source, bytes, kind, stream);
}

int main(int argc, const char* const* argv) {
  const char* base_path =
      argc > 1 ? argv[1] : std::getenv("GUFO_QWEN27B_MODEL");
  const char* draft_path =
      argc > 2 ? argv[2] : std::getenv("GUFO_QWEN27B_DFLASH_MODEL");
  if (base_path == nullptr || draft_path == nullptr) {
    std::cerr << "pass target and DFlash2 GGUF paths\n";
    return 77;
  }
  try {
    std::string error;
    auto owner = gufo::core::GgufReader::OpenFile(base_path, &error);
    Check(owner != nullptr, error.c_str());
    std::shared_ptr<const gufo::core::GgufReader> reader(std::move(owner));
    auto target = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Check(target != nullptr, error.c_str());
    auto backend = gufo::hip::QwenDFlashGpuDraftBackend::CreateFromGguf(
        draft_path, target, {.max_context = 32}, &error);
    Check(backend != nullptr, error.c_str());
    const std::vector<gufo::tokenization::TokenId> tokens{1, 2, 3, 4};
    const auto width = target->GetConfig().hidden_size *
                       backend->TargetHiddenLayerIds().size();
    std::vector<float> original(tokens.size() * width);
    for (std::size_t i = 0; i < original.size(); ++i)
      original[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 32;
    const auto prime = [&](std::span<const float> features) {
      return backend->PrimeTargetContext({
          .prompt_tokens = tokens,
          .prompt_hidden_states = features,
          .hidden_size = width,
          .first_token = 5,
      });
    };
    Check(prime(original), "warm context injection failed");

    bool all_preserved = true;
    for (bool pinned : {false, true}) {
      std::vector<float> pageable(original);
      gufo::hip::HostAllocations allocation;
      struct DrainBeforeRelease {
        ~DrainBeforeRelease() {
          if (fault.stream)
            (void)hipStreamSynchronize(fault.stream);
          fault.stream = nullptr;
        }
      } drain;
      float* source = pageable.data();
      if (pinned) {
        Check(allocation.Allocate(&source, original.size() * sizeof(float)) ==
                  hipSuccess,
              "pinned allocation failed");
        std::ranges::copy(original, source);
      }
      fault.source = source;
      fault.uploaded = false;
      fault.injected = false;
      fault.delay_finished = false;
      const bool accepted = prime({source, original.size()});
      const bool still_delayed = !fault.delay_finished;
      // The caller owns the input again after PrimeTargetContext returns,
      // including when it reports failure.
      std::fill_n(source, original.size(), 17.0F);
      const auto stream = fault.stream;
      fault.source = nullptr;
      Check(hipStreamSynchronize(stream) == hipSuccess, "drain failed");
      std::vector<float> uploaded(original.size());
      Check(hipMemcpy(uploaded.data(), fault.destination,
                      uploaded.size() * sizeof(float),
                      hipMemcpyDeviceToHost) == hipSuccess,
            "upload readback failed");
      Check(fault.injected && !accepted, "recoverable fault was not exercised");
      const bool preserved = uploaded == original;
      std::cout << (pinned ? "pinned" : "pageable")
                << " upload_pending_at_return=" << still_delayed
                << " source_bytes_preserved=" << preserved << '\n';
      all_preserved &= preserved;
      fault.uploaded = false;
      Check(prime(original), "context injection did not recover");
    }
    Check(all_preserved, "failed injection returned while reading caller data");
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
