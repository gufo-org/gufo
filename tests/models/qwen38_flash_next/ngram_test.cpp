// PLE n-gram hashing and disk row reads, checked without the model.
#include "src/models/qwen38_flash_next/ngram.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "tests/support/temporary_directory.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {
std::unique_ptr<q::NgramTable> OpenTable(const std::filesystem::path& path,
                                         std::uint64_t offset,
                                         std::uint64_t rows, std::uint32_t dim,
                                         gufo::core::GgmlType type,
                                         std::string* error) {
  const auto file = gufo::platform::OpenReadFile(path);
  auto table = q::NgramTable::Open(file, offset, rows, dim, type, error);
  gufo::platform::CloseFile(file);
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
// platform read path must return exactly those values, duplicates included.
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
  auto mapped = gufo::platform::ReadOnlyMappedFile::Open(path, &error);
  Check(mapped.valid(), "table file remains mapped during row reads");
  auto table = q::NgramTable::Open(mapped.PlatformHandle(), offset, rows, dim,
                                   gufo::core::GgmlType::kBF16, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    table->ReserveGather(32769);
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
    bool reserve_rejected = false;
    try {
      table->ReserveGather(65536);
    } catch (const std::logic_error&) {
      reserve_rejected = true;
    }
    Check(reserve_rejected, "scratch reservation rejects an active gather");
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
    // Reuse deduplication scratch across wide repetitive gathers, a tiny
    // decode, and a larger gather. Old first-occurrence indices must never
    // survive a change in input order or active table size.
    for (const std::size_t count : {32768U, 3U, 32769U}) {
      std::vector<std::uint32_t> repeated(count);
      for (std::size_t i = 0; i < count; ++i)
        repeated[i] = static_cast<std::uint32_t>((i * 197 + count) % 17);
      std::vector<float> gathered(count * dim + guard, -123.0F);
      Check(table->Read(repeated, std::span(gathered).first(count * dim)),
            "reused gather scratch completes");
      for (std::size_t j = 0; j < count; ++j) {
        for (std::uint32_t i = 0; i < dim; i += 31) {
          const float value =
              static_cast<float>(repeated[j]) + static_cast<float>(i) / 1000.0F;
          std::uint32_t bits = 0;
          std::memcpy(&bits, &value, sizeof(bits));
          bits &= 0xFFFF0000U;
          float expected_value = 0;
          std::memcpy(&expected_value, &bits, sizeof(expected_value));
          Check(gathered[j * dim + i] == expected_value,
                "reused scratch preserves duplicate output slots");
        }
      }
      Check(std::all_of(gathered.end() - guard, gathered.end(),
                        [](float value) { return value == -123.0F; }),
            "reused scratch preserves output guard");
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
  mapped.Close();
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

// A Q8_0 table (the source converter's format) decodes exactly through the
// same cold, cached and mixed gathers.
void TestQ8Rows() {
  const auto path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_q8.bin";
  constexpr std::uint32_t dim = 160;
  constexpr std::uint32_t rows = 129;
  constexpr std::uint64_t offset = 4096 + 90;
  const auto quant = [](std::uint32_t row, std::uint32_t col) {
    return static_cast<int>((row * 7 + col * 3) % 255) - 127;
  };
  {
    std::ofstream file(path, std::ios::binary);
    file.seekp(offset);
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t block = 0; block < dim / 32; ++block) {
        // Exactly representable half scales: 0.5 and -2.
        const std::uint16_t scale = row % 2 == 0 ? 0x3800 : 0xC000;
        file.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
        for (std::uint32_t col = 0; col < 32; ++col)
          file.put(static_cast<char>(
              static_cast<std::int8_t>(quant(row, block * 32 + col))));
      }
    }
    Check(file.good(), "Q8_0 fixture written");
  }
  std::string error;
  auto table =
      OpenTable(path, offset, rows, dim, gufo::core::GgmlType::kQ8_0, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    const auto gather = [&](std::span<const std::uint32_t> ids) {
      std::vector<float> out(ids.size() * dim + 16, -123.0F);
      Check(table->Read(ids, std::span(out).first(ids.size() * dim)),
            "Q8_0 gather completes");
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const float scale = ids[i] % 2 == 0 ? 0.5F : -2.0F;
        for (std::uint32_t col = 0; col < dim; ++col)
          Check(out[i * dim + col] == scale * quant(ids[i], col),
                "Q8_0 exact decoded value");
      }
      for (std::size_t i = ids.size() * dim; i < out.size(); ++i)
        Check(out[i] == -123.0F, "Q8_0 output guard");
    };
    const std::array<std::uint32_t, 6> warm{0, 7, 128, 7, 0, 32};
    gather(warm);
    gather(warm);
    const std::array<std::uint32_t, 7> mixed{128, 3, 7, 64, 3, 32, 1};
    gather(mixed);
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
  TemporaryDirectory temp("qwen-ngram-bound");
  const auto bound = temp.path() / "bound";
  const auto next = temp.path() / "next";
  {
    std::ofstream initial(bound, std::ios::binary);
    const std::uint16_t value = 0x3f80;  // BF16 1.0
    initial.write(reinterpret_cast<const char*>(&value), sizeof(value));
    Check(initial.good(), "write bound row");
  }
  const auto file = gufo::platform::OpenReadFile(bound);
  Check(gufo::platform::FileValid(file), "open bound table");
  if (!gufo::platform::FileValid(file))
    return;
  {
    std::ofstream replacement(next, std::ios::binary);
    const std::uint16_t other = 0x4000;
    replacement.write(reinterpret_cast<const char*>(&other), sizeof(other));
  }
  std::error_code ec;
  gufo::platform::Directory dir;
  Check(dir.Open(temp.path(), ec) &&
            gufo::platform::Rename(dir, "bound", dir, "retired", ec) &&
            gufo::platform::Rename(dir, "next", dir, "bound", ec),
        "replace bound path while descriptor stays open");
  std::string error;
  auto table =
      q::NgramTable::Open(file, 0, 1, 1, gufo::core::GgmlType::kBF16, &error);
  gufo::platform::CloseFile(file);
  Check(table != nullptr, "open original descriptor after path replacement");
  std::array<std::uint32_t, 1> ids{0};
  std::array<float, 1> output{};
  if (table)
    Check(table->Read(ids, output) && output[0] == 1.0F,
          "PLE reads original weights after replacement and descriptor close");
}

// Gathers attribute every row to the sequence that requested it: copied from
// a duplicate in the same gather, a row-cache hit or a disk read.
void TestCounters() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "qwen38_ngram_counters.bin";
  const std::uint32_t dim = 4;
  const std::uint64_t rows = 4096;
  {
    std::ofstream out(path, std::ios::binary);
    const std::vector<char> zeros(rows * dim * 2, 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  }
  std::string error;
  auto table =
      OpenTable(path, 0, rows, dim, gufo::core::GgmlType::kBF16, &error);
  Check(table != nullptr, error.c_str());
  if (table) {
    const auto balanced = [](const q::NgramTotals& t) {
      return t.rows == t.dedup_rows + t.cache_hits + t.disk_reads;
    };
    q::NgramCounters first, second;
    const std::vector<std::uint32_t> ids{5, 6, 5, 9, 6, 9, 10};
    std::vector<float> out(ids.size() * dim);
    const std::array owners{q::NgramGatherOwner{4, &first},
                            q::NgramGatherOwner{3, &second}};
    Check(table->StartRead(ids, out, owners) && table->WaitRead(),
          "owned gather completes");
    const auto a = first.Load(), b = second.Load();
    Check(a.rows == 4 && a.dedup_rows == 1 && a.cache_hits == 0 &&
              a.disk_reads == 3,
          "first owner pays its own reads");
    Check(b.rows == 3 && b.dedup_rows == 2 && b.cache_hits == 0 &&
              b.disk_reads == 1,
          "second owner copies rows the first one read");

    const std::array whole{q::NgramGatherOwner{ids.size(), &first}};
    Check(table->Read(ids, out) && table->StartRead(ids, out, whole) &&
              table->WaitRead(),
          "repeated gather completes");
    const auto repeated = first.Load();
    Check(repeated.rows == 11 && repeated.dedup_rows == 4 &&
              repeated.cache_hits + repeated.disk_reads == 7 &&
              repeated.disk_reads >= 3,
          "repeated rows hit the row cache");
    Check(balanced(repeated) && second.Load().rows == 3,
          "unowned gathers are not counted");

    // Large gathers read on the worker pool.
    std::vector<std::uint32_t> wide(3000);
    for (std::size_t i = 0; i < wide.size(); ++i)
      wide[i] = static_cast<std::uint32_t>((i * 7) % 2048 + 1024);
    std::vector<float> wide_out(wide.size() * dim);
    q::NgramCounters pooled;
    const std::array pooled_owner{q::NgramGatherOwner{wide.size(), &pooled}};
    Check(table->StartRead(wide, wide_out, pooled_owner) && table->WaitRead(),
          "pooled gather completes");
    const auto p = pooled.Load();
    Check(p.rows == wide.size() && p.dedup_rows == wide.size() - 2048 &&
              balanced(p),
          "pooled gather balances");

    const std::array short_owner{q::NgramGatherOwner{ids.size() - 1, &first}};
    Check(!table->StartRead(ids, out, short_owner),
          "owners must cover the gather");
    Check(first.Load().rows == repeated.rows, "rejected gather is uncounted");
  }
  table.reset();
  std::filesystem::remove(path);
}
}  // namespace

int main() {
  TestReplacedPath();
  TestHash();
  TestTable();
  TestQuantizedRows();
  TestQ8Rows();
  TestDistantRows();
  TestCounters();
  if (failures != 0) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
