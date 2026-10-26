#include "src/core/gguf_identity.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/env.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::size_t kWindow = 4096;

/// Minimal GGUF v3 image: one string metadata entry and two tensors whose
/// payloads span `first_bytes` and `second_bytes`.
std::vector<std::uint8_t> BuildImage(std::string_view name_value,
                                     std::string_view first_tensor_name,
                                     std::size_t first_bytes,
                                     std::size_t second_bytes) {
  std::vector<std::uint8_t> buffer;
  auto append = [&buffer](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
  };
  auto pod = [&append](auto value) { append(&value, sizeof(value)); };
  auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    append(value.data(), value.size());
  };

  append("GGUF", 4);
  pod(static_cast<std::uint32_t>(3));
  pod(static_cast<std::uint64_t>(2));  // tensor count
  pod(static_cast<std::uint64_t>(1));  // metadata count
  string("general.name");
  pod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
  string(name_value);

  string(first_tensor_name);
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(first_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(0));
  string("second");
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(second_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(first_bytes));

  while (buffer.size() % 32 != 0) {
    buffer.push_back(0);
  }
  const std::size_t payload_base = buffer.size();
  buffer.resize(payload_base + first_bytes + second_bytes);
  for (std::size_t index = 0; index < first_bytes + second_bytes; ++index) {
    buffer[payload_base + index] = static_cast<std::uint8_t>(index * 7U + 3U);
  }
  return buffer;
}

std::size_t PayloadBase(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return static_cast<std::size_t>(
      static_cast<const std::uint8_t*>(reader->GetTensors()[0].data) -
      image.data());
}

std::string Identity(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return gufo::core::GgufIdentityHex(*reader);
}

void TestDeterministic() {
  const auto image = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const std::string first = Identity(image);
  Expect(first.size() == 64, "identity is 64 hex characters");
  Expect(first == Identity(image), "identity is deterministic");
  Expect(first ==
             Identity(BuildImage("model", "first", 64 * kWindow, 2 * kWindow)),
         "identity depends on bytes, not on the buffer instance");
}

void TestHeaderChangesIdentity() {
  const auto base = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  Expect(Identity(base) != Identity(BuildImage("model-v2", "first",
                                               64 * kWindow, 2 * kWindow)),
         "metadata change alters identity");
  Expect(Identity(base) != Identity(BuildImage("model", "renamed", 64 * kWindow,
                                               2 * kWindow)),
         "tensor name change alters identity");
  Expect(Identity(base) !=
             Identity(BuildImage("model", "first", 64 * kWindow, 4 * kWindow)),
         "tensor extent change alters identity");
}

void TestEveryPayloadByte() {
  const std::size_t first_bytes = 64 * kWindow;
  auto image = BuildImage("model", "first", first_bytes, 2 * kWindow);
  const std::string base = Identity(image);
  const std::size_t payload = PayloadBase(image);

  auto flipped = [&](std::size_t offset) {
    auto copy = image;
    copy[payload + offset] ^= 0xFFU;
    return Identity(copy);
  };

  Expect(flipped(0) != base, "start bytes are hashed");
  Expect(flipped(first_bytes - 1) != base, "end bytes are hashed");
  const std::size_t middle = ((first_bytes - kWindow) / 2) & ~(kWindow - 1);
  Expect(flipped(middle + 17) != base, "middle bytes are hashed");
  Expect(flipped(kWindow + 100) != base,
         "bytes outside the old sample windows alter identity");

  // The short second tensor is hashed whole.
  Expect(flipped(first_bytes + kWindow + 5) != base,
         "small tensors are hashed in full");
}

void TestFileDigestCache() {
  char temp[] = "/tmp/gufo-identity-XXXXXX";
  Expect(mkdtemp(temp) != nullptr, "create cache test directory");
  const std::filesystem::path root(temp);
  const char* old_cache = std::getenv("XDG_CACHE_HOME");
  const std::string saved_cache = old_cache ? old_cache : "";
  const bool had_cache = old_cache != nullptr;
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);
  auto bytes = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const auto path = root / "model.gguf";
  auto write = [&] {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Expect(out.good(), "write GGUF fixture");
  };
  write();
  auto read = [&] {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
    Expect(reader != nullptr, "open file: " + error);
    return gufo::core::GgufIdentityHex(*reader);
  };
  const auto first = read();
  Expect(first == Identity(bytes), "file and memory full digests agree");
  Expect(read() == first, "cached digest agrees with initial scan");
  const auto cache = root / "gufo" / "gguf-sha256-v1";
  Expect(std::distance(std::filesystem::directory_iterator(cache),
                       std::filesystem::directory_iterator{}) == 1,
         "one cached full digest exists");
  struct stat before{};
  Expect(stat(path.c_str(), &before) == 0, "stat original file");
  bytes[PayloadBase(bytes) + kWindow + 100] ^= 0xff;
  write();
  const timespec times[] = {before.st_atim, before.st_mtim};
  Expect(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0,
         "restore original mtime");
  Expect(read() != first,
         "same-size edit with restored mtime invalidates cache");
  Expect(read() == Identity(bytes),
         "edited file still matches full memory digest");
  for (const auto& entry : std::filesystem::directory_iterator(cache)) {
    std::ofstream(entry.path()) << "broken";
  }
  Expect(read() == Identity(bytes), "malformed cached digest is recomputed");
  if (had_cache)
    gufo::platform::SetEnv("XDG_CACHE_HOME", saved_cache);
  else
    gufo::platform::UnsetEnv("XDG_CACHE_HOME");
  std::filesystem::remove_all(root);
}

void TestPipelinedFileHash() {
  const auto bytes = BuildImage("pipeline", "first", 40 * 1024 * 1024, 8194);
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-identity-pipeline-" + std::to_string(getpid()));
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
  Expect(reader != nullptr, "pipelined fixture opens");
  Expect(gufo::core::GgufIdentityHex(*reader) == Identity(bytes),
         "parallel disk reads hash all chunks and the unaligned tail in order");
  std::filesystem::remove(path);
}

void TestSha256() {
  const std::string text = "abc";
  gufo::crypto::Sha256Hasher hash;
  for (unsigned char byte : text)
    hash.Update({&byte, 1});
  Expect(hash.FinishHex() ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "incremental SHA-256 matches independent standard vector");
  Expect(gufo::crypto::Sha256Hex({}) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "empty SHA-256 matches standard vector");
}

}  // namespace

int main() {
  TestSha256();
  TestFileDigestCache();
  TestPipelinedFileHash();
  TestDeterministic();
  TestHeaderChangesIdentity();
  TestEveryPayloadByte();
  std::cout << "gguf_identity_test passed\n";
  return 0;
}
