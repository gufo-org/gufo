#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/arch.hpp"
#include "src/models/minimax_h3/sampling.hpp"

namespace {

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool Allocate(std::size_t bytes) {
    return hipMalloc(&data_, bytes) == hipSuccess;
  }
  [[nodiscard]] void* data() const noexcept { return data_; }

private:
  void* data_{nullptr};
};

class Stream {
public:
  Stream() = default;
  ~Stream() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  bool Create() {
    return hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) ==
           hipSuccess;
  }
  [[nodiscard]] hipStream_t get() const noexcept { return stream_; }

private:
  hipStream_t stream_{nullptr};
};

bool CheckTarget() {
  int device = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) {
    return false;
  }
  return gufo::hip::IsRdna35(properties.gcnArchName) &&
         properties.integrated != 0;
}

bool CopyToDevice(void* destination, const void* source, std::size_t bytes,
                  hipStream_t stream) {
  return hipMemcpyAsync(destination, source, bytes, hipMemcpyHostToDevice,
                        stream) == hipSuccess;
}

bool RunEulerCase(float sigma_from_timestep, float ratio,
                  std::array<float, 7>* output) {
  constexpr std::array<float, 7> sample = {-7.0F, -8.0F, 10.0F, 20.0F,
                                           30.0F, 40.0F, -9.0F};
  constexpr std::array<float, 4> last = {1.003F, -2.007F, 0.501F, 4.009F};
  constexpr std::array<float, 4> previous = {0.499F, -1.011F, 0.249F,
                                             5.013F};

  DeviceBuffer device_sample;
  DeviceBuffer device_last;
  DeviceBuffer device_previous;
  Stream stream;
  if (!device_sample.Allocate(sizeof(sample)) ||
      !device_last.Allocate(sizeof(last)) ||
      !device_previous.Allocate(sizeof(previous)) || !stream.Create() ||
      !CopyToDevice(device_sample.data(), sample.data(), sizeof(sample),
                    stream.get()) ||
      !CopyToDevice(device_last.data(), last.data(), sizeof(last),
                    stream.get()) ||
      !CopyToDevice(device_previous.data(), previous.data(), sizeof(previous),
                    stream.get())) {
    return false;
  }

  std::string error;
  if (!gufo::minimax_h3::HipEulerUpdate(device_sample.data(), sample.size(), 2,
                                         device_last.data(),
                                         device_previous.data(), last.size(),
                                         sigma_from_timestep, ratio, 0.0F,
                                         stream.get(), &error)) {
    std::cerr << error << '\n';
    return false;
  }
  if (hipMemcpyAsync(output->data(), device_sample.data(), sizeof(sample),
                     hipMemcpyDeviceToHost, stream.get()) != hipSuccess ||
      hipStreamSynchronize(stream.get()) != hipSuccess) {
    return false;
  }
  return true;
}

bool RunEulerTrajectory(
    bool video,
    const std::array<std::array<std::uint32_t, 4>, 4>& expected_bits) {
  constexpr std::array<float, 4> initial = {0.125F, -1.5F, 3.25F, 7.0F};
  constexpr std::array<std::array<float, 4>, 4> velocities = {{
      {0.75F, -0.5F, 1.25F, -2.0F},
      {-0.25F, 0.625F, -1.5F, 0.75F},
      {1.125F, -0.875F, 0.5F, -0.375F},
      {-0.625F, 1.75F, -0.125F, 0.5F},
  }};
  std::string error;
  auto schedule = gufo::minimax_h3::BuildServingSchedule(4, &error);
  auto plan = schedule.has_value()
                  ? gufo::minimax_h3::BuildEulerPlan(*schedule, 1, &error)
                  : std::nullopt;
  if (!plan.has_value()) {
    std::cerr << error << '\n';
    return false;
  }

  DeviceBuffer device_sample;
  DeviceBuffer device_velocity;
  Stream stream;
  if (!device_sample.Allocate(sizeof(initial)) ||
      !device_velocity.Allocate(sizeof(initial)) || !stream.Create() ||
      !CopyToDevice(device_sample.data(), initial.data(), sizeof(initial),
                    stream.get())) {
    return false;
  }
  std::array<float, 4> output{};
  for (std::size_t step = 0; step < velocities.size(); ++step) {
    const auto& item = (*plan)[step];
    if (!CopyToDevice(device_velocity.data(), velocities[step].data(),
                      sizeof(velocities[step]), stream.get()) ||
        !gufo::minimax_h3::HipEulerUpdate(
            device_sample.data(), initial.size(), 0, device_velocity.data(),
            device_velocity.data(), initial.size(),
            video ? item.video_sigma_from_timestep
                  : item.audio_sigma_from_timestep,
            video ? item.video_ratio : item.audio_ratio, 0.0F, stream.get(),
            &error) ||
        hipMemcpyAsync(output.data(), device_sample.data(), sizeof(output),
                       hipMemcpyDeviceToHost, stream.get()) != hipSuccess ||
        hipStreamSynchronize(stream.get()) != hipSuccess) {
      std::cerr << error << '\n';
      return false;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
      if (std::bit_cast<std::uint32_t>(output[index]) !=
          expected_bits[step][index]) {
        std::cerr << "MiniMax H3 " << (video ? "video" : "audio")
                  << " trajectory differs from Diffusers at step " << step
                  << ", element " << index << ": got 0x" << std::hex
                  << std::bit_cast<std::uint32_t>(output[index])
                  << ", expected 0x" << expected_bits[step][index] << std::dec
                  << '\n';
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckTarget()) {
    std::cerr << "MiniMax H3 sampling HIP test requires an integrated "
                 "RDNA 3.5 GPU (gfx1151 or gfx1150)\n";
    return 77;
  }
  std::array<float, 7> first{};
  std::array<float, 7> second{};
  constexpr float kSigmaFromTimestep = 0.8648648858070374F;
  constexpr float kRatio = 0.9765625F;
  if (!RunEulerCase(kSigmaFromTimestep, kRatio, &first) ||
      !RunEulerCase(kSigmaFromTimestep, kRatio, &second)) {
    std::cerr << "MiniMax H3 HIP Euler execution failed\n";
    return 1;
  }
  constexpr std::array<std::uint32_t, 7> expected_bits = {
      0xc0e00000U, 0xc1000000U, 0x41205347U, 0x419facafU,
      0x41f014ccU, 0x42205337U, 0xc1100000U};
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(first[index]) != expected_bits[index]) {
      std::cerr << "MiniMax H3 HIP Euler output differs from the Diffusers "
                   "float32 oracle at "
                << index << ": got 0x" << std::hex
                << std::bit_cast<std::uint32_t>(first[index])
                << ", expected 0x" << expected_bits[index] << std::dec << '\n';
      return 1;
    }
  }
  if (std::memcmp(first.data(), second.data(), sizeof(first)) != 0) {
    std::cerr << "MiniMax H3 HIP Euler output is not byte stable\n";
    return 1;
  }
  constexpr std::array<std::array<std::uint32_t, 4>, 4>
      video_trajectory_bits = {{
          {0x3e14c1baU, 0xbfc1bad0U, 0x40522984U, 0x40de4531U},
          {0x3e07fbbdU, 0xbfbdbcf1U, 0x404d5f46U, 0x40df77c1U},
          {0x3e8ae24dU, 0xbfcb85ceU, 0x40514f85U, 0x40ddfdaaU},
          {0xbe6a3b66U, 0xbe4294d8U, 0x404ae91fU, 0x40eaca77U},
      }};
  constexpr std::array<std::array<std::uint32_t, 4>, 4>
      audio_trajectory_bits = {{
          {0x3e4cccceU, 0xbfc66666U, 0x40580000U, 0x40d99999U},
          {0x3e266668U, 0xbfba6667U, 0x4049999aU, 0x40dd3332U},
          {0x3ee33333U, 0xbfd66668U, 0x4051999aU, 0x40da3332U},
          {0x3e066666U, 0xbf4cccd0U, 0x404d999aU, 0x40e23332U},
      }};
  if (!RunEulerTrajectory(true, video_trajectory_bits) ||
      !RunEulerTrajectory(false, audio_trajectory_bits)) {
    return 1;
  }
  std::string error;
  void* const invalid_device_pointer =
      reinterpret_cast<void*>(std::uintptr_t{1});
  if (gufo::minimax_h3::HipEulerUpdate(
          invalid_device_pointer, 1, 0, invalid_device_pointer,
          invalid_device_pointer, 1, 0.0F, 0.5F, 0.0F, nullptr, &error)) {
    std::cerr << "MiniMax H3 HIP Euler accepted a zero-sigma step\n";
    return 1;
  }
  if (gufo::minimax_h3::HipEulerUpdate(
          invalid_device_pointer, 1, 1, invalid_device_pointer,
          invalid_device_pointer, 1, 0.25F, 0.5F, 0.0F, nullptr, &error)) {
    std::cerr << "MiniMax H3 HIP Euler accepted an out-of-bounds range\n";
    return 1;
  }
  std::cout << "MiniMax H3 GPU-state Euler checks passed on RDNA 3.5\n";
  return 0;
}
