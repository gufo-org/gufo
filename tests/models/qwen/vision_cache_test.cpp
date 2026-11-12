#include <webp/encode.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/families/qwen/vision/admission.hpp"
#include "src/families/qwen/vision/image_cache.hpp"
#include "src/families/qwen/vision/prompt.hpp"

namespace {
constexpr std::size_t kTrackedPixelBytes = 256 * 256 * 3;
thread_local std::barrier<>* before_pixels{};
thread_local gufo::models::qwen::vision::AdmissionBudget* allocation_budget{};
thread_local std::array<void*, 4> pixel_allocations{};
thread_local bool fail_owner{}, owner_failure_triggered{};
}  // namespace

// Only the two focused tests below arm these controls. They observe real
// vector allocations and the final committed-owner allocation, without a
// production test hook or decoder replacement.
void* operator new(std::size_t bytes) {
  if (allocation_budget && fail_owner &&
      std::ranges::any_of(pixel_allocations, [](void* p) { return p; }) &&
      allocation_budget->PendingBytes() == 0) {
    fail_owner = false;
    owner_failure_triggered = true;
    throw std::bad_alloc();
  }
  if (bytes == kTrackedPixelBytes && before_pixels) {
    auto* rendezvous = std::exchange(before_pixels, nullptr);
    rendezvous->arrive_and_wait();
  }
  void* memory = std::malloc(bytes ? bytes : 1);
  if (!memory)
    throw std::bad_alloc();
  if (allocation_budget && bytes == kTrackedPixelBytes) {
    auto slot = std::ranges::find(pixel_allocations, nullptr);
    assert(slot != pixel_allocations.end());
    *slot = memory;
  }
  return memory;
}
void operator delete(void* memory) noexcept {
  if (memory && allocation_budget) {
    auto slot = std::ranges::find(pixel_allocations, memory);
    if (slot != pixel_allocations.end()) {
      assert(allocation_budget->Outstanding() >= kTrackedPixelBytes);
      *slot = nullptr;
    }
  }
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  ::operator delete(memory);
}

namespace {
using namespace gufo::models::qwen::vision;
using ImageDetail = gufo::tokenization::ChatMessage::ImageDetail;
constexpr auto kAuto = ImageDetail::kAuto;

std::shared_ptr<const std::vector<std::uint8_t>> Encoded(
    std::uint8_t value, std::uint32_t width = 256, std::uint32_t height = 256) {
  const std::vector<std::uint8_t> rgb(width * height * 3, value);
  std::uint8_t* encoded = nullptr;
  const auto size =
      WebPEncodeLosslessRGB(rgb.data(), width, height, width * 3, &encoded);
  assert(size > 0);
  std::vector<std::uint8_t> bytes(encoded, encoded + size);
  WebPFree(encoded);
  return std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
}

void TestReuseAndEviction() {
  const auto red = Encoded(17), blue = Encoded(42), green = Encoded(123);
  constexpr std::size_t bytes = 256 * 256 * 3;
  ImageCache cache(2 * (bytes + std::max({red->capacity(), blue->capacity(),
                                          green->capacity()})));
  auto first = cache.Get(red, kAuto);
  const auto second = cache.Get(blue, kAuto);
  // Fresh request buffers containing identical bytes must share the result.
  const auto copy = std::make_shared<const std::vector<std::uint8_t>>(*red);
  assert(cache.Get(copy, kAuto) == first);
  auto stats = cache.GetStats();
  assert(stats.hits == 1 && stats.misses == 2 && stats.entries == 2 &&
         stats.pixel_bytes == 2 * bytes &&
         stats.encoded_bytes == red->capacity() + blue->capacity());
  // Red was just touched, so blue is the victim. Eviction cannot invalidate
  // the request's pixels, nor retain a second copy when red hits again.
  (void)cache.Get(green, kAuto);
  assert(cache.Get(red, kAuto) == first);
  assert(second->pixels == std::vector<std::uint8_t>(bytes, 42));
  const auto replaced = cache.Get(blue, kAuto);
  assert(replaced != second && replaced->pixels == second->pixels);
  stats = cache.GetStats();
  assert(stats.evictions == 2 && stats.entries == 2 &&
         stats.pixel_bytes == 2 * bytes);
  // Oversized entries bypass retention and do not evict useful small ones.
  (void)cache.Get(Encoded(9, 768), kAuto);
  assert(cache.GetStats().pixel_bytes == 2 * bytes);
  assert(cache.GetStats().evictions == 2);
  // Encoded owners count against the same budget, not only RGB storage.
  ImageCache pixel_only_budget(bytes);
  (void)pixel_only_budget.Get(red, kAuto);
  assert(pixel_only_budget.GetStats().entries == 0);
  ImageCache lifetime(bytes + std::max(red->capacity(), blue->capacity()));
  auto encoded_owner = Encoded(17);
  const std::weak_ptr<const std::vector<std::uint8_t>> weak = encoded_owner;
  const auto retained_pixels = lifetime.Get(encoded_owner, kAuto);
  encoded_owner.reset();
  assert(!weak.expired());
  (void)lifetime.Get(blue, kAuto);
  assert(weak.expired() && retained_pixels->pixels.front() == 17);
  ImageCache disabled(0);
  const auto a = disabled.Get(red, kAuto), b = disabled.Get(red, kAuto);
  assert(a != b && a->pixels == b->pixels);
  assert(disabled.GetStats().entries == 0);
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool rejected = false;
    try {
      (void)cache.Get(std::make_shared<const std::vector<std::uint8_t>>(
                          std::vector<std::uint8_t>{1, 2, 3}),
                      kAuto);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
  assert(cache.GetStats().entries == 2);
}

void TestConcurrentMisses() {
  ImageCache cache;
  const auto bytes = Encoded(57);
  std::array<std::shared_ptr<const gufo::core::Image>, 8> results;
  std::barrier start(8);
  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < results.size(); ++i) {
    workers.emplace_back([&, i] {
      start.arrive_and_wait();
      for (int repeat = 0; repeat < 16; ++repeat)
        results[i] = cache.Get(bytes, kAuto);
    });
  }
  for (auto& worker : workers)
    worker.join();
  for (const auto& result : results)
    assert(result == results.front());
  const auto stats = cache.GetStats();
  assert(stats.entries == 1 && stats.pixel_bytes == 256 * 256 * 3 &&
         stats.hits + stats.misses == 8 * 16);
}

void TestAdmittedConcurrentMisses() {
  const auto encoded = Encoded(57);
  const auto plan = PlanImagePreparation(*encoded, kAuto);
  // Two independently admitted decoders can coexist until the cache chooses
  // one result. These are logical byte bounds, not physical allocations.
  const auto capacity =
      CheckedMultiply(2, CheckedAdd(plan.scratch_bytes, plan.pixel_bytes));
  const auto budget =
      std::make_shared<AdmissionBudget>([capacity] { return capacity; });
  std::array<std::shared_ptr<const gufo::core::Image>, 2> results;
  {
    ImageCache cache;
    std::barrier rendezvous(2);
    std::array<std::thread, 2> workers;
    for (std::size_t i = 0; i < workers.size(); ++i)
      workers[i] = std::thread([&, i] {
        // Both misses must enter decoding before either can publish pixels.
        before_pixels = &rendezvous;
        results[i] = cache.Get(encoded, kAuto, budget);
        assert(!before_pixels);
      });
    for (auto& worker : workers)
      worker.join();
    assert(results[0] == results[1]);
    assert(cache.GetStats().misses == 2 && cache.GetStats().entries == 1);
    assert(!budget->PendingBytes() &&
           budget->Outstanding() == kTrackedPixelBytes);
  }
  results[0].reset();
  assert(budget->Outstanding() == kTrackedPixelBytes);
  results[1].reset();
  assert(!budget->Outstanding());
}

void TestOwnerAllocationFailure() {
  const auto encoded = Encoded(17);
  const auto plan = PlanImagePreparation(*encoded, kAuto);
  const auto capacity = CheckedAdd(plan.scratch_bytes, plan.pixel_bytes);
  const auto budget =
      std::make_shared<AdmissionBudget>([capacity] { return capacity; });
  auto cache = std::make_unique<ImageCache>();
  allocation_budget = budget.get();
  fail_owner = true;
  owner_failure_triggered = false;
  bool failed = false;
  try {
    (void)cache->Get(encoded, kAuto, budget);
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  assert(failed && owner_failure_triggered && !fail_owner);
  assert(std::ranges::all_of(pixel_allocations, [](void* p) { return !p; }));
  assert(!budget->Outstanding() && !budget->PendingBytes());
  assert(cache->GetStats().entries == 0);
  (void)cache->Get(encoded, kAuto, budget);
  assert(cache->GetStats().entries == 1 &&
         budget->Outstanding() == kTrackedPixelBytes);
  cache.reset();
  assert(std::ranges::all_of(pixel_allocations, [](void* p) { return !p; }));
  allocation_budget = nullptr;
  assert(!budget->Outstanding());
}

void TestAdmissionRefusalAndHits() {
  const auto red = Encoded(17), blue = Encoded(42);
  const auto plan = PlanImagePreparation(*red, kAuto);
  const auto required = plan.scratch_bytes + plan.pixel_bytes;
  std::size_t available = required - 1;
  const auto budget =
      std::make_shared<AdmissionBudget>([&] { return available; });
  ImageCache cache;
  bool refused = false;
  try {
    (void)cache.Get(red, kAuto, budget);
  } catch (const gufo::core::ResourceAdmissionError&) {
    refused = true;
  }
  assert(refused && !budget->Outstanding() && !budget->PendingBytes());
  assert(cache.GetStats().entries == 0);
  available = required;
  const auto pixels = cache.Get(red, kAuto, budget);
  assert(budget->Outstanding() == kTrackedPixelBytes &&
         !budget->PendingBytes());
  // A committed cache hit needs no new pixel or decoder allocation.
  available = 0;
  assert(cache.Get(red, kAuto, budget) == pixels);
  refused = false;
  try {
    (void)cache.Get(blue, kAuto, budget);
  } catch (const gufo::core::ResourceAdmissionError&) {
    refused = true;
  }
  assert(refused && cache.GetStats().entries == 1);
  assert(budget->Outstanding() == kTrackedPixelBytes &&
         !budget->PendingBytes());
  assert(cache.Get(red, kAuto, budget) == pixels);
}

void EqualPrompts(const Prompt& a, const Prompt& b) {
  assert(a.tokens == b.tokens && a.rope == b.rope &&
         a.stable_prefix_tokens == b.stable_prefix_tokens &&
         a.cache_identity == b.cache_identity &&
         a.images.size() == b.images.size());
  for (std::size_t i = 0; i < a.images.size(); ++i) {
    const auto& x = a.images[i];
    const auto& y = b.images[i];
    assert(x.grid == y.grid && x.prefix_identity == y.prefix_identity &&
           x.pixels->width == y.pixels->width &&
           x.pixels->height == y.pixels->height &&
           x.pixels->pixels == y.pixels->pixels);
  }
}

auto MakeTokenizer() {
  using namespace gufo::tokenization;
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  // Independent literal framing fixture, not derived from production constants.
  std::unordered_map<std::string, TokenId> specials;
  for (const auto* token : {"<|im_start|>", "<|im_end|>", "<|vision_start|>",
                            "<|vision_end|>", "<|image_pad|>"}) {
    specials.emplace(token, static_cast<TokenId>(vocab.size()));
    vocab.emplace_back(token);
  }
  auto tokenizer = QwenTokenizer::CreateFromVocabulary(vocab, {}, specials);
  assert(tokenizer);
  return tokenizer;
}

void TestPromptIdentities() {
  using namespace gufo::tokenization;
  const auto tokenizer = MakeTokenizer();
  const auto red = Encoded(17, 48, 32);
  const auto blue = Encoded(42);
  std::vector<ChatMessage> messages{{ChatRole::kUser, "beforeafter"}};
  messages.front().images = {{6, red}, {6, blue}};
  ChatTemplateOptions options;
  options.enable_thinking = false;
  ImageCache cache;
  const auto prepare = [&](ImageCache* reuse,
                           std::string_view encoder = "encoder") {
    return Prepare(*tokenizer, messages, {}, options, encoder, 8192, reuse);
  };
  const auto uncached = prepare(nullptr);
  const auto cold = prepare(&cache);
  const auto warm = prepare(&cache);
  EqualPrompts(uncached, cold);
  EqualPrompts(uncached, warm);
  // Independent Python hashlib fixture for the original v1 identity stream:
  // literal version + "encoder", LE32 grids (13,7,10)/(85,8,8), constant RGB.
  const std::array<std::uint8_t, 32> golden{
      0x6b, 0x94, 0x89, 0x3a, 0xa0, 0xcc, 0x67, 0x94, 0x82, 0xf7, 0xb7,
      0xa6, 0x60, 0xf3, 0xf1, 0xa1, 0x8a, 0xaf, 0x7a, 0x69, 0xd2, 0x64,
      0x8b, 0x94, 0x29, 0x0b, 0xf9, 0x22, 0x6d, 0x41, 0xdb, 0x6b};
  assert(std::ranges::equal(uncached.cache_identity, golden));
  assert(cold.images[0].pixels == warm.images[0].pixels);
  assert(warm.images[0].pixels->width == 320 &&
         warm.images[0].pixels->height == 224);
  assert(cache.GetStats().hits == 2 && cache.GetStats().misses == 2);
  const auto other_encoder = prepare(&cache, "another encoder");
  assert(other_encoder.cache_identity != warm.cache_identity &&
         other_encoder.images[0].pixels == warm.images[0].pixels);
  std::swap(messages.front().images[0].bytes, messages.front().images[1].bytes);
  const auto reordered = prepare(&cache);
  EqualPrompts(prepare(nullptr), reordered);
  assert(reordered.cache_identity != warm.cache_identity &&
         reordered.images[0].pixels == warm.images[1].pixels);
  messages.front().images[0].offset = 0;
  const auto moved = prepare(&cache);
  EqualPrompts(prepare(nullptr), moved);
  assert(moved.cache_identity != reordered.cache_identity);
  messages.front().images[0].bytes = red;
  const auto changed = prepare(&cache);
  EqualPrompts(prepare(nullptr), changed);
  assert(changed.cache_identity != moved.cache_identity);
  // A text-only follow-up still refers to the same shared historical pixels.
  messages.emplace_back(ChatRole::kAssistant, "Acknowledged.");
  messages.emplace_back(ChatRole::kUser, "Continue.");
  const auto continued = prepare(&cache);
  EqualPrompts(prepare(nullptr), continued);
  assert(continued.cache_identity == changed.cache_identity &&
         continued.images[0].pixels == changed.images[0].pixels);
  // The encoder requirement remains enforced even when the pixels are cached.
  bool rejected = false;
  try {
    (void)prepare(&cache, "");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void TestDetailKeysCache() {
  const auto large = Encoded(7, 1024, 768);
  ImageCache cache;
  const auto low = cache.Get(large, ImageDetail::kLow);
  const auto automatic = cache.Get(large, kAuto);
  assert(automatic->width == 1024 && automatic->height == 768);
  assert(low->width == 512 && low->height == 384);
  assert(cache.GetStats().entries == 2 && cache.GetStats().misses == 2);
  assert(cache.Get(large, ImageDetail::kLow) == low);
  assert(cache.Get(large, kAuto) == automatic);
  assert(cache.GetStats().entries == 2 && cache.GetStats().hits == 2);
}

void TestDetailPrompts() {
  using namespace gufo::tokenization;
  const auto tokenizer = MakeTokenizer();
  ChatTemplateOptions options;
  options.enable_thinking = false;
  ImageCache cache;
  const auto prepare =
      [&](const std::shared_ptr<const std::vector<std::uint8_t>>& image,
          ImageDetail detail, ImageCache* reuse) {
        std::vector<ChatMessage> messages{{ChatRole::kUser, "look"}};
        messages.front().images = {{4, image, detail}};
        return Prepare(*tokenizer, messages, {}, options, "encoder", 8192,
                       reuse);
      };
  const auto large = Encoded(7, 1024, 768);
  const auto low = prepare(large, ImageDetail::kLow, &cache);
  const auto automatic = prepare(large, kAuto, &cache);
  EqualPrompts(prepare(large, ImageDetail::kLow, nullptr), low);
  EqualPrompts(prepare(large, kAuto, nullptr), automatic);
  assert(low.images[0].grid != automatic.images[0].grid);
  assert(low.images[0].grid.height <= 16 && low.images[0].grid.width <= 16);
  assert(low.tokens.size() < automatic.tokens.size());
  assert(low.cache_identity != automatic.cache_identity);
  // Images below the low limit resize the same for both details, so their
  // prompts share one identity.
  const auto small = Encoded(9, 320, 240);
  const auto small_low = prepare(small, ImageDetail::kLow, &cache);
  const auto small_auto = prepare(small, kAuto, &cache);
  EqualPrompts(small_low, small_auto);
}

void TestAdmittedPromptAndCacheOwnership() {
  using namespace gufo::tokenization;
  const auto tokenizer = MakeTokenizer();
  const auto red = Encoded(17), blue = Encoded(42);
  const auto red_plan = PlanImagePreparation(*red, kAuto);
  const auto blue_plan = PlanImagePreparation(*blue, kAuto);
  // One previous image can remain retained while the replacement is prepared.
  const auto capacity = CheckedAdd(
      std::max(CheckedAdd(red_plan.scratch_bytes, red_plan.pixel_bytes),
               CheckedAdd(blue_plan.scratch_bytes, blue_plan.pixel_bytes)),
      kTrackedPixelBytes);
  const auto budget =
      std::make_shared<AdmissionBudget>([capacity] { return capacity; });
  const auto other_budget =
      std::make_shared<AdmissionBudget>([capacity] { return capacity; });
  std::vector<ChatMessage> messages{{ChatRole::kUser, "look"}};
  messages.front().images = {{4, red}};
  ChatTemplateOptions options;
  options.enable_thinking = false;
  std::shared_ptr<const gufo::core::Image> retained;
  {
    ImageCache cache(kTrackedPixelBytes +
                     std::max(red->capacity(), blue->capacity()));
    const auto prepare = [&] {
      return Prepare(*tokenizer, messages, {}, options, "encoder", 8192, &cache,
                     budget);
    };
    // Token counting discards its Prompt immediately; preprocessing reuse
    // still owns the pixels and their committed charge.
    assert(prepare().tokens.size() > 0);
    assert(!budget->PendingBytes() &&
           budget->Outstanding() == kTrackedPixelBytes);
    {
      const auto prompt = prepare();
      retained = prompt.images.front().pixels;
      assert(prompt.memory_admission.lock() == budget);
      assert(budget->Outstanding() == kTrackedPixelBytes);
    }
    assert(cache.GetStats().hits == 1);
    // A later invalid attachment cannot strand the earlier cache image's
    // transient scratch or drop its retained pixel charge.
    messages.front().images.push_back(
        {4, std::make_shared<const std::vector<std::uint8_t>>(
                std::vector<std::uint8_t>{1, 2, 3})});
    bool failed = false;
    try {
      (void)prepare();
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    assert(failed && !budget->PendingBytes() &&
           budget->Outstanding() == kTrackedPixelBytes &&
           cache.GetStats().entries == 1);
    messages.front().images.pop_back();
    // Eviction removes only cache ownership; a copied image remains charged.
    (void)cache.Get(blue, kAuto, budget);
    assert(budget->Outstanding() == 2 * kTrackedPixelBytes);
    retained.reset();
    assert(budget->Outstanding() == kTrackedPixelBytes);
    // A different coordinator and the conservative path cannot borrow an
    // image while claiming it is owned by their own admission budget.
    const auto other = cache.Get(blue, kAuto, other_budget);
    assert(!budget->Outstanding() &&
           other_budget->Outstanding() == kTrackedPixelBytes);
    assert(cache.Get(blue, kAuto) != other);
    assert(other_budget->Outstanding() == kTrackedPixelBytes);
  }
  assert(!budget->Outstanding() && !other_budget->Outstanding());

  // Uncached prompts retain the same per-image ownership contract.
  {
    const auto prompt = Prepare(*tokenizer, messages, {}, options, "encoder",
                                8192, nullptr, budget);
    retained = prompt.images.front().pixels;
    assert(budget->Outstanding() == kTrackedPixelBytes);
  }
  assert(budget->Outstanding() == kTrackedPixelBytes);
  retained.reset();
  assert(!budget->Outstanding());
}
}  // namespace

int main(int argc, char** argv) {
  TestReuseAndEviction();
  TestConcurrentMisses();
  TestAdmittedConcurrentMisses();
  TestOwnerAllocationFailure();
  TestAdmissionRefusalAndHits();
  TestPromptIdentities();
  TestDetailKeysCache();
  TestDetailPrompts();
  TestAdmittedPromptAndCacheOwnership();
  if (argc == 2 && std::string_view(argv[1]) == "--capacity-probe") {
    ImageCache cache;
    for (std::uint8_t i = 0; i < 22; ++i)
      (void)cache.Get(Encoded(i, 1920, 1088), kAuto);
    const auto stats = cache.GetStats();
    assert(stats.entries == 21 && stats.pixel_bytes == 21 * 1920 * 1088 * 3 &&
           stats.evictions == 1 &&
           stats.pixel_bytes + stats.encoded_bytes <=
               ImageCache::kDefaultCapacityBytes);
    std::cout << "{\"entries\":" << stats.entries
              << ",\"pixel_bytes\":" << stats.pixel_bytes
              << ",\"encoded_bytes\":" << stats.encoded_bytes
              << ",\"misses\":" << stats.misses
              << ",\"evictions\":" << stats.evictions
              << ",\"capacity_bytes\":" << ImageCache::kDefaultCapacityBytes
              << "}\n";
  } else {
    assert(argc == 1);
  }
}
