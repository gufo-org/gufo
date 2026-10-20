#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Prompt hidden buffer test failure: " << message << '\n';
    std::abort();
  }
}

bool IsPinned(const float* data) {
  hipPointerAttribute_t attributes{};
  if (hipPointerGetAttributes(&attributes, data) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return attributes.type == hipMemoryTypeHost;
}

void Fill(gufo::hip::PromptHiddenBuffer& buffer, std::size_t from) {
  float* data = buffer.Data();
  for (std::size_t i = from; i < buffer.Size(); ++i) {
    data[i] = static_cast<float>(i);
  }
}

bool Holds(const gufo::hip::PromptHiddenBuffer& buffer) {
  const auto view = buffer.View();
  for (std::size_t i = 0; i < view.size(); ++i) {
    if (view[i] != static_cast<float>(i)) {
      return false;
    }
  }
  return true;
}

// A prompt within the bound captures into pinned memory and keeps its
// capacity across prompts.
void TestPinnedWithinBound() {
  gufo::hip::PromptHiddenBuffer buffer;
  buffer.Reserve(4096, 8192);
  buffer.Resize(1024);
  Fill(buffer, 0);
  buffer.Resize(4096);
  Fill(buffer, 1024);
  Check(IsPinned(buffer.Data()), "bounded capture must be pinned");
  Check(Holds(buffer), "bounded capture must keep its rows");
  const float* first = buffer.Data();
  buffer.Clear();
  Check(buffer.Size() == 0, "clear must empty the capture");
  buffer.Reserve(2048, 8192);
  Check(buffer.Data() == first, "a smaller prompt must reuse the capacity");
}

// Growth inside the bound keeps the captured rows.
void TestPinnedGrowth() {
  gufo::hip::PromptHiddenBuffer buffer;
  buffer.Reserve(1024, 8192);
  buffer.Resize(1024);
  Fill(buffer, 0);
  buffer.Resize(6000);
  Fill(buffer, 1024);
  Check(IsPinned(buffer.Data()), "growth inside the bound must stay pinned");
  Check(Holds(buffer), "growth must keep the captured rows");
}

// A larger request uses pageable memory, and growth past the bound moves the
// captured rows there.
void TestPageableBeyondBound() {
  gufo::hip::PromptHiddenBuffer reserved;
  reserved.Reserve(16384, 8192);
  reserved.Resize(16384);
  Fill(reserved, 0);
  Check(!IsPinned(reserved.Data()),
        "a request past the bound must be pageable");
  Check(Holds(reserved), "pageable capture must keep its rows");

  gufo::hip::PromptHiddenBuffer grown;
  grown.Reserve(4096, 8192);
  grown.Resize(4096);
  Fill(grown, 0);
  grown.Resize(12288);
  Fill(grown, 4096);
  Check(!IsPinned(grown.Data()), "growth past the bound must be pageable");
  Check(Holds(grown), "moving to pageable memory must keep the rows");

  grown.Clear();
  grown.Reserve(4096, 8192);
  grown.Resize(4096);
  Check(IsPinned(grown.Data()), "the next bounded prompt must be pinned again");
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (const int gate = gufo::test::GateHipDevice(
          gufo::test::HipDeviceRequirement::kOptional, "prompt hidden buffer");
      gate != gufo::test::kHipTestSuccess) {
    return gate;
  }
  TestPinnedWithinBound();
  TestPinnedGrowth();
  TestPageableBeyondBound();
  std::cout << "Prompt hidden buffer tests passed.\n";
  return EXIT_SUCCESS;
#else
  std::cerr << "HIP disabled; prompt hidden buffer test unavailable.\n";
  return EXIT_FAILURE;
#endif
}
