#include "src/core/diagnostics/fingerprint.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace gufo::diagnostics {

namespace {

// Standard SHA-256 implementation (FIPS 180-4 / RFC 6234)
class Sha256 {
public:
  Sha256() { Reset(); }

  void Reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    count_ = 0;
  }

  void Update(std::string_view data) {
    for (const char ch : data) {
      buffer_[count_ % 64] = static_cast<std::uint8_t>(ch);
      ++count_;
      if (count_ % 64 == 0) {
        Transform(buffer_.data());
      }
    }
  }

  void UpdateBytes(const std::uint8_t* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
      buffer_[count_ % 64] = data[i];
      ++count_;
      if (count_ % 64 == 0) {
        Transform(buffer_.data());
      }
    }
  }

  void Final(std::array<std::uint8_t, 32>& digest) {
    const std::uint64_t total_bits = count_ * 8;
    // Pad with 0x80
    const std::uint8_t pad80 = 0x80;
    UpdateBytes(&pad80, 1);
    // Pad with zeros until 56 bytes mod 64
    const std::uint8_t pad00 = 0x00;
    while (count_ % 64 != 56) {
      UpdateBytes(&pad00, 1);
    }
    // Append 64-bit length in big-endian
    std::array<std::uint8_t, 8> length_bytes{};
    for (std::size_t i = 0; i < 8; ++i) {
      length_bytes[i] =
          static_cast<std::uint8_t>((total_bits >> (56 - (i * 8))) & 0xFF);
    }
    UpdateBytes(length_bytes.data(), 8);

    for (std::size_t i = 0; i < 8; ++i) {
      digest[i * 4] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFF);
      digest[(i * 4) + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFF);
      digest[(i * 4) + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFF);
      digest[(i * 4) + 3] = static_cast<std::uint8_t>(state_[i] & 0xFF);
    }
  }

private:
  static constexpr std::array<std::uint32_t, 64> K = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  static std::uint32_t RotR(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32 - n));
  }

  void Transform(const std::uint8_t* chunk) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(chunk[i * 4]) << 24) |
             (static_cast<std::uint32_t>(chunk[(i * 4) + 1]) << 16) |
             (static_cast<std::uint32_t>(chunk[(i * 4) + 2]) << 8) |
             static_cast<std::uint32_t>(chunk[(i * 4) + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + K[i] + w[i];
      const std::uint32_t s0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t count_{0};
};

std::string EscapeJsonStr(std::string_view str) {
  std::ostringstream oss;
  for (const char character : str) {
    if (character == '"') {
      oss << "\\\"";
    } else if (character == '\\') {
      oss << "\\\\";
    } else if (character == '\n') {
      oss << "\\n";
    } else if (character == '\t') {
      oss << "\\t";
    } else {
      oss << character;
    }
  }
  return oss.str();
}

}  // namespace

std::string ComputeSha256Hex(std::string_view data) {
  Sha256 sha;
  sha.Update(data);
  std::array<std::uint8_t, 32> digest{};
  sha.Final(digest);

  std::ostringstream oss;
  for (const auto byte : digest) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(byte);
  }
  return oss.str();
}

std::string CanonicalFingerprint::CanonicalJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"cppStandard\": \"" << EscapeJsonStr(cpp_standard) << "\",\n";
  oss << "  \"cpuArchitecture\": \"" << EscapeJsonStr(cpu_architecture)
      << "\",\n";
  oss << "  \"cpuLogicalCores\": " << cpu_logical_cores << ",\n";
  oss << "  \"cpuModel\": \"" << EscapeJsonStr(cpu_model) << "\",\n";
  oss << "  \"cpuPhysicalCores\": " << cpu_physical_cores << ",\n";
  oss << "  \"cxxCompiler\": \"" << EscapeJsonStr(cxx_compiler) << "\",\n";
  oss << "  \"gpuArchitecture\": \"" << EscapeJsonStr(gpu_architecture)
      << "\",\n";
  oss << "  \"gpuComputeUnits\": " << gpu_compute_units << ",\n";
  oss << "  \"gpuDriver\": \"" << EscapeJsonStr(gpu_driver) << "\",\n";
  oss << "  \"gpuName\": \"" << EscapeJsonStr(gpu_name) << "\",\n";
  oss << "  \"gpuPciId\": \"" << EscapeJsonStr(gpu_pci_id) << "\",\n";
  oss << "  \"kernelRelease\": \"" << EscapeJsonStr(kernel_release) << "\",\n";
  oss << "  \"memoryTotalBytes\": " << memory_total_bytes << ",\n";
  oss << "  \"memoryType\": \"" << EscapeJsonStr(memory_type) << "\",\n";
  oss << "  \"rocmVersion\": \"" << EscapeJsonStr(rocm_version) << "\",\n";
  oss << "  \"schemaVersion\": \"" << EscapeJsonStr(schema_version) << "\"\n";
  oss << "}";
  return oss.str();
}

std::string CanonicalFingerprint::ComputeFingerprintId() const {
  return ComputeSha256Hex(CanonicalJson());
}

MachineFingerprint GenerateMachineFingerprint(
    const SystemInventory& inventory) {
  MachineFingerprint fp;
  fp.canonical.cpu_model = inventory.cpu.model_name;
  fp.canonical.cpu_architecture = inventory.cpu.architecture;
  fp.canonical.cpu_logical_cores = inventory.cpu.logical_cores;
  fp.canonical.cpu_physical_cores = inventory.cpu.physical_cores;
  fp.canonical.memory_total_bytes = inventory.memory.total_bytes;
  fp.canonical.memory_type = inventory.memory.memory_type;
  fp.canonical.gpu_name = inventory.gpu.name;
  fp.canonical.gpu_architecture = inventory.gpu.architecture;
  fp.canonical.gpu_compute_units = inventory.gpu.compute_units;
  fp.canonical.gpu_driver = inventory.gpu.driver_name;
  fp.canonical.gpu_pci_id = inventory.gpu.pci_id;
  fp.canonical.kernel_release = inventory.toolchain.kernel_release;
  fp.canonical.rocm_version = inventory.toolchain.rocm_version;
  fp.canonical.cxx_compiler = inventory.toolchain.cxx_compiler;
  fp.canonical.cpp_standard = inventory.toolchain.cpp_standard;

  fp.os_name = inventory.toolchain.os_name;
  fp.target_platform = inventory.toolchain.target_platform;

  fp.fingerprint_id = fp.canonical.ComputeFingerprintId();
  return fp;
}

std::string MachineFingerprint::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"schemaVersion\": \"" << EscapeJsonStr(canonical.schema_version)
      << "\",\n";
  oss << "  \"fingerprintId\": \"" << EscapeJsonStr(fingerprint_id) << "\",\n";
  oss << "  \"canonical\": " << canonical.CanonicalJson() << "\n";
  oss << "}\n";
  return oss.str();
}

std::string MachineFingerprint::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Machine Fingerprint ===\n";
  oss << "Fingerprint ID      : " << fingerprint_id << "\n";
  oss << "Target Platform     : " << target_platform << " (" << os_name << " "
      << canonical.kernel_release << ")\n";
  oss << "CPU Model           : " << canonical.cpu_model << " ("
      << canonical.cpu_logical_cores << " threads)\n";
  oss << "GPU Architecture    : " << canonical.gpu_name << " ["
      << canonical.gpu_architecture << ", " << canonical.gpu_compute_units
      << " CUs]\n";
  oss << "Pinned Toolchain    : ROCm " << canonical.rocm_version << "\n";
  return oss.str();
}

}  // namespace gufo::diagnostics
