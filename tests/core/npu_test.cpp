// The NPU API without an NPU: the sysfs probe, instruction patching, dispatch
// libraries, and a clean refusal when no device can open.
#include "src/core/npu/npu.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace npu = gufo::npu;

namespace {

void Expect(bool ok, const std::string& what) {
  if (!ok)
    throw std::runtime_error(what);
}

void Write(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << text;
}

template<typename F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const npu::Error&) {
    return true;
  }
  return false;
}

// accel0: another driver's accel node (ignored); accel1: an XDNA NPU.
void ProbeFindsOnlyXdna(const fs::path& root) {
  fs::remove_all(root);
  const fs::path drivers = root / "bus" / "pci" / "drivers";
  fs::create_directories(drivers / "amdxdna");
  fs::create_directories(drivers / "other");
  for (const char* node : {"accel0", "accel1"}) {
    const fs::path pci = root / "devices" / node;
    fs::create_directories(pci);
    fs::create_directories(root / "class" / "accel" / node);
    fs::create_directory_symlink(pci,
                                 root / "class" / "accel" / node / "device");
  }
  fs::create_directory_symlink(drivers / "other",
                               root / "devices" / "accel0" / "driver");
  fs::create_directory_symlink(drivers / "amdxdna",
                               root / "devices" / "accel1" / "driver");
  Write(root / "devices" / "accel1" / "vendor", "0x1022\n");
  Write(root / "devices" / "accel1" / "device", "0x17f0\n");
  Write(root / "devices" / "accel1" / "fw_version", "1.1.2.65\n");

  const auto found = npu::ProbeDevices(root);
  Expect(found.size() == 1,
         "probe found " + std::to_string(found.size()) + " NPUs, expected 1");
  Expect(found[0].node == "accel1", "probe named " + found[0].node);
  Expect(found[0].driver == "amdxdna", "probe driver " + found[0].driver);
  Expect(found[0].pci_id == "1022:17f0", "probe PCI id " + found[0].pci_id);
  Expect(found[0].firmware == "1.1.2.65",
         "probe firmware " + found[0].firmware);
  Expect(npu::ProbeDevices(root / "missing").empty(),
         "probe of a missing sysfs found NPUs");
}

void DispatchAndPatch(const fs::path& library) {
  const auto dispatch = npu::DispatchLibrary::Open(library, 2);
  const std::array<std::int32_t, 2> at0{0, 5}, at1{1, 5}, at9{9, 5};
  const auto s0 = dispatch.Generate(at0);
  const std::vector<std::uint32_t> base(s0.begin(), s0.end());
  const auto s1 = dispatch.Generate(at1);
  const auto patches = npu::DiffInstructions(
      base, std::vector<std::uint32_t>(s1.begin(), s1.end()));
  Expect(patches.size() == 1 && patches[0].index == 1 &&
             patches[0].delta == 0x200u,
         "token patch");

  // Patching the id-0 stream for token 9 must equal generating it.
  std::vector<std::uint32_t> patched = base;
  for (const auto& p : patches)
    patched[p.index] += 9u * p.delta;
  const auto s9 = dispatch.Generate(at9);
  Expect(std::equal(patched.begin(), patched.end(), s9.begin(), s9.end()),
         "patched stream differs");

  Expect(
      Throws([&] { (void)dispatch.Generate(std::array<std::int32_t, 1>{0}); }),
      "scalar count mismatch was accepted");
  Expect(Throws([&] {
           (void)npu::DispatchLibrary::Open(library, 0, "dispatch_empty")
               .Generate({});
         }),
         "an empty stream was accepted");
  Expect(Throws([&] {
           (void)npu::DispatchLibrary::Open(library, 2, "dispatch_missing");
         }),
         "a missing symbol was accepted");
  Expect(Throws([&] {
           (void)npu::DiffInstructions(base, std::vector<std::uint32_t>(3));
         }),
         "streams of different length were diffed");
}

// Without an NPU, Open reports an npu::Error.
void OpenWithoutNpu() {
  if (!npu::ProbeDevices().empty())
    return;
  Expect(Throws([] { (void)npu::Device::Open(); }),
         "opened an NPU on a machine without one");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Expect(argc == 3, "usage: npu_test <dispatch fixture> <scratch sysfs dir>");
    ProbeFindsOnlyXdna(argv[2]);
    DispatchAndPatch(argv[1]);
    OpenWithoutNpu();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
