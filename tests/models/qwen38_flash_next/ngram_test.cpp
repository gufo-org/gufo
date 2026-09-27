// PLE n-gram hashing and disk row reads, checked without the model.
#include "src/models/qwen38_flash_next/ngram.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {
std::unique_ptr<q::NgramTable> OpenTable(const std::filesystem::path& path,
                                         std::uint64_t offset,
                                         std::uint64_t rows, std::uint32_t dim,
                                         gufo::core::GgmlType type,
                                         std::string* error) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  auto table = q::NgramTable::Open(fd, offset, rows, dim, type, error);
  if (fd >= 0)
    ::close(fd);
  return table;
}

q::Config FlashNextPle() {
  q::Config c;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 248044;
  c.ple_multipliers = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
  c.ple_head_offsets = {0,         20000003,  40000026,  60000059,
                        80000106,  100000165, 120000228, 140000297,
                        160000374, 180000455, 200000548, 220000655,
                        240000802, 260000955, 280001114, 300001275};
  c.ple_head_vocab = {20000003, 20000023, 20000033, 20000047,
                      20000059, 20000063, 20000069, 20000077,
                      20000081, 20000093, 20000107, 20000147,
                      20000153, 20000159, 20000161, 20000171};
  c.ple_rows = 320001446;
  return c;
}

int failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

// Expected rows computed independently (Python, llama.cpp hash) for
// [<|im_start|>, "user", "\n", EOS, "Hello", ","]. The EOS at position 3
// cuts the window: position 4 hashes (9707, EOS, EOS), position 5
// (11, 9707, EOS).
void TestHash() {
  const q::Config c = FlashNextPle();
  const std::array<std::int32_t, 6> tokens{151644, 872, 198, 248044, 9707, 11};
  const std::array<std::array<std::uint32_t, 16>, 6> expected{{
      {9213577, 25503567, 59963735, 67280845, 97263283, 108604657, 126879742,
       153604170, 163991649, 192963728, 210734887, 224850479, 244505954,
       265606139, 292960741, 312140635},
      {6954764, 37428000, 54939688, 74004108, 95568253, 116575022, 138540160,
       142009604, 165349151, 193458441, 202840010, 231812024, 252625381,
       274082424, 294711253, 318927564},
      {6746895, 31942778, 54550641, 78212738, 81361934, 102413837, 123993646,
       146103739, 175246506, 190873805, 206877969, 222022136, 242132242,
       262461054, 282619366, 303774792},
      {16849591, 37537304, 42290399, 73883125, 88405327, 114186760, 123740665,
       158125394, 169423291, 194333347, 208742846, 238789911, 242351884,
       266971822, 295413699, 319385148},
      {16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017,
       152932897, 169641436, 182022480, 209277433, 237891023, 256841529,
       277007269, 290665954, 300984276},
      {18158303, 36390029, 45652312, 70783524, 98191154, 114024957, 127804916,
       159566246, 175132467, 192583600, 217919476, 237199054, 259336807,
       261799367, 289359313, 307699687},
  }};

  // Whole batch at once.
  q::NgramHistory history;
  std::vector<std::uint32_t> rows(tokens.size() * 16);
  q::HashNgramRows(c, history, tokens, rows);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    for (std::size_t h = 0; h < 16; ++h) {
      Check(rows[i * 16 + h] == expected[i][h], "batched hash row");
    }
  }
  Check(history.prev[0] == 11 && history.prev[1] == 9707, "history tail");

  // One token at a time must agree with the batch.
  q::NgramHistory step;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    std::array<std::uint32_t, 16> one{};
    q::HashNgramRows(c, step, std::span<const std::int32_t>(&tokens[i], 1),
                     one);
    Check(one == expected[i], "single-step hash row");
  }
}

// A BF16 table where row r holds r + i/1000 in column i; reads through the
// direct-I/O path must return exactly those values, duplicates included.
void TestTable() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_test.bin";
  const std::uint32_t dim = 160;
  const std::uint64_t rows = 2049;
  const std::uint64_t offset = 4096 + 90;  // deliberately unaligned
  {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> pad(offset, 0);
    out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    for (std::uint64_t r = 0; r < rows; ++r) {
      for (std::uint32_t i = 0; i < dim; ++i) {
        const float v = static_cast<float>(r) + static_cast<float>(i) / 1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        const std::uint16_t bf = static_cast<std::uint16_t>(bits >> 16);
        out.write(reinterpret_cast<const char*>(&bf), 2);
      }
    }
  }
  std::string error;
  auto table =
      OpenTable(path, offset, rows, dim, gufo::core::GgmlType::kBF16, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    const std::vector<std::uint32_t> ids{7, rows - 1, 7, 0, 128, rows - 1};
    std::vector<float> out(ids.size() * dim);
    Check(table->Read(ids, out), "table read");
    for (std::size_t j = 0; j < ids.size(); ++j) {
      for (std::uint32_t i = 0; i < dim; i += 37) {
        const float v =
            static_cast<float>(ids[j]) + static_cast<float>(i) / 1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        bits &= 0xFFFF0000U;
        float expected = 0.0F;
        std::memcpy(&expected, &bits, 4);
        Check(out[j * dim + i] == expected, "table row value");
      }
    }
    const std::vector<std::uint32_t> bad{rows};
    Check(!table->Read(bad, out), "out-of-range row rejected");
    const auto expected = out;
    std::fill(out.begin(), out.end(), -1.0F);
    Check(table->StartRead(ids, out), "asynchronous gather starts");
    Check(!table->StartRead(ids, out), "overlapping gather rejected");
    Check(table->WaitRead(), "asynchronous gather completes");
    Check(out == expected, "asynchronous rows and duplicates match");
    Check(!table->WaitRead(), "completed gather cannot be consumed twice");
    Check(table->Read({}, {}), "empty gather completes");
    Check(!table->StartRead(ids, std::span<float>(out).first(dim)),
          "short output rejected");
    Check(table->Read(ids, out), "reader remains usable after rejection");
    // More distinct rows than the reader's batching threshold, shuffled
    // with duplicates. File-order reads must preserve every output slot.
    std::vector<std::uint32_t> wide_ids(rows * 2 + 17);
    for (std::size_t i = 0; i < wide_ids.size(); ++i) {
      wide_ids[i] = static_cast<std::uint32_t>((i * 109 + 19) % rows);
    }
    constexpr std::size_t guard = 16;
    const std::size_t wide_size = wide_ids.size() * dim;
    std::vector<float> wide(wide_size + guard, -123.0F);
    Check(table->Read(wide_ids, std::span<float>(wide).first(wide_size)),
          "batched gather completes");
    for (std::size_t j = 0; j < wide_ids.size(); ++j) {
      for (std::uint32_t i = 0; i < dim; ++i) {
        const float v =
            static_cast<float>(wide_ids[j]) + static_cast<float>(i) / 1000.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        bits &= 0xFFFF0000U;
        float expected_value = 0.0F;
        std::memcpy(&expected_value, &bits, 4);
        Check(wide[j * dim + i] == expected_value, "batched output slot");
      }
    }
    for (std::size_t i = wide_size; i < wide.size(); ++i) {
      Check(wide[i] == -123.0F, "batched output guard");
    }
    std::fill(out.begin(), out.end(), -1.0F);
    Check(table->StartRead(ids, out), "final gather starts");
    table.reset();
    Check(out == expected, "destruction drains the outstanding gather");

    // Valid row metadata with a truncated backing file must fail at the
    // read boundary, then allow another gather to use the same readers.
    auto truncated = OpenTable(path, offset, rows + 1, dim,
                               gufo::core::GgmlType::kBF16, &error);
    Check(truncated != nullptr, error.c_str());
    if (truncated) {
      Check(truncated->StartRead(bad, out), "missing row gather starts");
      Check(!truncated->WaitRead(), "truncated row read fails");
      Check(truncated->Read(ids, out), "reader recovers after failed I/O");
      Check(out == expected, "failed I/O does not corrupt later rows");
      wide_ids[333] = rows;
      Check(!truncated->Read(wide_ids, std::span<float>(wide).first(wide_size)),
            "failed batched read drains outstanding jobs");
      Check(truncated->Read(ids, out), "reader recovers after batched failure");
      Check(out == expected, "batched failure preserves subsequent reads");
    }
  }
  std::filesystem::remove(path);
}

// The production IQ4_NL format must decode identically on a cold read, a
// cached gather and a mixed cached/disk gather, including duplicate slots.
void TestQuantizedRows() {
  const auto path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_iq4.bin";
  constexpr std::uint32_t dim = 160;
  constexpr std::uint32_t rows = 129;
  constexpr std::uint64_t offset = 4096 + 90;
  constexpr std::array<int, 16> values{
      -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  {
    std::ofstream file(path, std::ios::binary);
    file.seekp(offset);
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t block = 0; block < dim / 32; ++block) {
        // Exactly representable half scales: 0.5 and -2.
        const std::uint16_t scale = row % 2 == 0 ? 0x3800 : 0xC000;
        file.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        for (std::uint32_t col = 0; col < 16; ++col) {
          const auto packed =
              static_cast<std::uint8_t>(((row + block + col) % 16) |
                                        (((row + block + col + 7) % 16) << 4));
          file.put(static_cast<char>(packed));
        }
      }
    }
    Check(file.good(), "IQ4_NL fixture written");
  }
  std::string error;
  auto table =
      OpenTable(path, offset, rows, dim, gufo::core::GgmlType::kIQ4_NL, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    const auto gather = [&](std::span<const std::uint32_t> ids) {
      std::vector<float> out(ids.size() * dim + 16, -123.0F);
      Check(table->Read(ids, std::span(out).first(ids.size() * dim)),
            "IQ4_NL gather completes");
      for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::uint32_t col = 0; col < dim; ++col) {
          const auto index =
              (ids[i] + col / 32 + col % 16 + (col % 32 >= 16 ? 7 : 0)) % 16;
          const float scale = ids[i] % 2 == 0 ? 0.5F : -2.0F;
          Check(out[i * dim + col] == scale * values[index],
                "IQ4_NL exact decoded value");
        }
      }
      for (std::size_t i = ids.size() * dim; i < out.size(); ++i) {
        Check(out[i] == -123.0F, "IQ4_NL output guard");
      }
    };
    const std::array<std::uint32_t, 6> warm{0, 7, 128, 7, 0, 32};
    gather(warm);
    gather(warm);
    const std::array<std::uint32_t, 7> mixed{128, 3, 7, 64, 3, 32, 1};
    gather(mixed);
    // The cached path must not need the backing file after successful reads.
    std::filesystem::resize_file(path, 0);
    gather(warm);
    gather(mixed);
  }
  table.reset();
  std::filesystem::remove(path);
}

// The production Q8_0 format must use the canonical 32/34-byte row layout on
// cold, cached, asynchronous and batched reads. The unaligned offset and
// repeated/interior/final row IDs catch an incorrect row stride immediately.
void TestQ8Rows() {
  const auto path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_q8.bin";
  constexpr std::uint32_t dim = 160;
  constexpr std::uint32_t rows = 129;
  constexpr std::uint64_t offset = 4096 + 90;
  constexpr std::size_t blocks = dim / 32;
  const auto code = [](std::uint32_t row, std::uint32_t block,
                       std::uint32_t col) {
    return static_cast<std::int8_t>(
        static_cast<int>((row * 17 + block * 29 + col * 13) % 255) - 127);
  };
  const auto scale = [](std::uint32_t row) {
    return row % 2 == 0 ? 0.5F : -2.0F;
  };
  {
    std::ofstream file(path, std::ios::binary);
    file.seekp(offset);
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t block = 0; block < blocks; ++block) {
        gufo::quant::block_q8_0 value{};
        value.d = row % 2 == 0 ? 0x3800 : 0xC000;
        for (std::uint32_t col = 0; col < 32; ++col) {
          value.qs[col] = code(row, block, col);
        }
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
    Check(file.good(), "Q8_0 fixture written");
  }
  std::string error;
  auto table =
      OpenTable(path, offset, rows, dim, gufo::core::GgmlType::kQ8_0, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    Check(table->RowBytes() == 170, "Q8_0 row geometry");
    const auto expected = [&](std::uint32_t row, std::uint32_t col) {
      return scale(row) * static_cast<float>(code(row, col / 32, col % 32));
    };
    const auto gather = [&](std::span<const std::uint32_t> ids,
                            const char* label) {
      std::vector<float> out(ids.size() * dim + 16, -123.0F);
      Check(table->Read(ids, std::span(out).first(ids.size() * dim)), label);
      for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::uint32_t col = 0; col < dim; ++col) {
          Check(out[i * dim + col] == expected(ids[i], col),
                "Q8_0 exact decoded value");
        }
      }
      for (std::size_t i = ids.size() * dim; i < out.size(); ++i) {
        Check(out[i] == -123.0F, "Q8_0 output guard");
      }
    };

    const std::array<std::uint32_t, 8> warm{0,  1,   7,        63,
                                            64, 127, rows - 2, rows - 1};
    gather(warm, "Q8_0 cold gather");
    gather(warm, "Q8_0 cached gather");
    const std::array<std::uint32_t, 7> mixed{rows - 1, 3, 64, 0,
                                             rows - 1, 2, 128};
    gather(mixed, "Q8_0 mixed gather");

    std::vector<float> expected_async(warm.size() * dim);
    Check(table->Read(warm, expected_async), "Q8_0 async baseline");
    std::vector<float> async(warm.size() * dim + 16, -123.0F);
    Check(table->StartRead(warm, std::span(async).first(warm.size() * dim)),
          "Q8_0 asynchronous gather starts");
    Check(!table->StartRead(warm, std::span(async).first(warm.size() * dim)),
          "Q8_0 overlapping gather rejected");
    Check(table->WaitRead(), "Q8_0 asynchronous gather completes");
    for (std::size_t i = 0; i < expected_async.size(); ++i) {
      Check(async[i] == expected_async[i], "Q8_0 asynchronous value");
    }
    for (std::size_t i = expected_async.size(); i < async.size(); ++i) {
      Check(async[i] == -123.0F, "Q8_0 asynchronous output guard");
    }

    std::vector<std::uint32_t> wide_ids(rows * 8 + 17);
    for (std::size_t i = 0; i < wide_ids.size(); ++i) {
      wide_ids[i] = static_cast<std::uint32_t>((i * 109 + 19) % rows);
    }
    gather(wide_ids, "Q8_0 batched gather");

    auto invalid = OpenTable(path, offset, rows, dim - 1,
                             gufo::core::GgmlType::kQ8_0, &error);
    Check(!invalid, "Q8_0 rejects a non-block-aligned row dimension");
    invalid.reset();

    // Every warm row remains encoded in the bounded cache, so a truncated
    // backing file must not invalidate an already-read gather.
    std::filesystem::resize_file(path, 0);
    gather(warm, "Q8_0 cached gather after truncation");
    const std::array<std::uint32_t, 1> missing{rows};
    std::vector<float> missing_out(dim);
    Check(!table->Read(missing, missing_out), "Q8_0 missing row rejected");
  }
  table.reset();
  std::filesystem::remove(path);
}

// Widely spaced rows exercise cache eviction, including simultaneous readers
// whose compressed rows map to the same bounded-cache slot.
void TestDistantRows() {
  const auto path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_distant.bin";
  constexpr std::uint32_t dim = 160;
  constexpr std::uint32_t stride = 65536;
  constexpr std::uint64_t offset = 4096 + 90;
  constexpr std::uint32_t last = 7 * stride;
  {
    std::ofstream file(path, std::ios::binary);
    for (std::uint32_t r = 0; r < 8; ++r) {
      file.seekp(offset + static_cast<std::uint64_t>(r * stride) * dim * 2);
      for (std::uint32_t col = 0; col < dim; ++col) {
        const float value = static_cast<float>(r) + 1.25F +
                            static_cast<float>(col % 16) / 16.0F;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const auto bf = static_cast<std::uint16_t>(bits >> 16);
        file.write(reinterpret_cast<const char*>(&bf), sizeof(bf));
      }
    }
    Check(file.good(), "sparse row fixture written");
  }
  std::string error;
  auto table = OpenTable(path, offset, last + 2, dim,
                         gufo::core::GgmlType::kBF16, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    std::vector<std::uint32_t> ids(80);
    std::vector<float> out(ids.size() * dim + 16, -123.0F);
    for (std::uint32_t round = 0; round < 4; ++round) {
      for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<std::uint32_t>((i * 3 + round) % 8) * stride;
      }
      Check(table->Read(ids, std::span<float>(out).first(ids.size() * dim)),
            "colliding row gather completes");
      for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::uint32_t col = 0; col < dim; ++col) {
          const float expected = static_cast<float>(ids[i] / stride) + 1.25F +
                                 static_cast<float>(col % 16) / 16.0F;
          Check(out[i * dim + col] == expected,
                "evicted row retains exact bytes");
        }
      }
      const std::array<std::uint32_t, 1> missing{last + 1};
      Check(!table->Read(missing, out), "incomplete row is not cached");
    }
    for (std::size_t i = ids.size() * dim; i < out.size(); ++i) {
      Check(out[i] == -123.0F, "colliding gather output guard");
    }
  }
  table.reset();
  std::filesystem::remove(path);
}

void TestReplacedPath() {
  char name[] = "/tmp/qwen-ngram-bound-XXXXXX";
  const int fd = ::mkstemp(name);
  Check(fd >= 0, "create bound table");
  if (fd < 0)
    return;
  const std::uint16_t value = 0x3f80;  // BF16 1.0
  Check(::write(fd, &value, sizeof(value)) == sizeof(value), "write bound row");
  ::unlink(name);
  {
    std::ofstream replacement(name, std::ios::binary);
    const std::uint16_t other = 0x4000;
    replacement.write(reinterpret_cast<const char*>(&other), sizeof(other));
  }
  std::string error;
  auto table =
      q::NgramTable::Open(fd, 0, 1, 1, gufo::core::GgmlType::kBF16, &error);
  ::close(fd);
  Check(table != nullptr, "open original descriptor after path replacement");
  std::array<std::uint32_t, 1> ids{0};
  std::array<float, 1> output{};
  if (table)
    Check(table->Read(ids, output) && output[0] == 1.0F,
          "PLE reads original weights after replacement and descriptor close");
  std::filesystem::remove(name);
}

}  // namespace

int main() {
  TestReplacedPath();
  TestHash();
  TestTable();
  TestQuantizedRows();
  TestQ8Rows();
  TestDistantRows();
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
