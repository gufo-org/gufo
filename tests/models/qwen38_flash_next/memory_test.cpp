#include "src/models/qwen38_flash_next/memory.hpp"

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

#include "src/models/qwen38_flash_next/copy_width_admission.hpp"
#include "src/models/qwen38_flash_next/execution_admission.hpp"
#include "src/models/qwen38_flash_next/snapshot_admission.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::SessionMode;
namespace vision = gufo::models::qwen::vision;

namespace {
void Require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

qfn::Config SyntheticConfig() {
  qfn::Config c;
  c.num_layers = 48;
  c.hidden_size = 2560;
  c.vocab_size = 248320;
  c.hc_count = 4;
  c.full_attention_interval = 4;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}

void SessionComponents() {
  // Fixed byte oracle for a full-context, seven-draft-token configuration.
  // No HIP type, model weights or multi-GiB physical allocation is involved.
  auto plan =
      qfn::PlanSessionMemory(SyntheticConfig(), SessionMode::kSpeculative,
                             262144, 4224, 8, 8, 7, 20, 2720);
  Require(plan.trunk_kv_bytes == 6442450944ULL, "twelve full-context KV pairs");
  Require(plan.trunk_index_bytes == 251661312, "trunk raw and pooled indexes");
  Require(plan.recurrent_bytes == 118038528, "linear and PLE mutable state");
  Require(plan.rollback_bytes == 153953280, "seven exact rollback frontiers");
  Require(plan.mtp_kv_bytes == 536870912, "predictor retains its own KV");
  Require(plan.mtp_index_bytes == 20971776, "predictor raw and pooled index");
  Require(plan.mtp_hidden_bytes == 368640,
          "eight trunk rows plus predictor row");
  Require(plan.mtp_draft_table_bytes == 2793472, "private Q8_0 draft table");
  Require(plan.control_bytes == 20 && plan.vision_descriptor_bytes == 0,
          "executor control excludes the model-owned vision descriptor");
  plan.vision_descriptor_bytes = 64;
  Require(plan.TotalBytes() == 7527108948ULL,
          "production components match the independent base claim");
  Require(4 * plan.TotalBytes() == 30108435792ULL,
          "four sessions preserve the full context");
}

void ModesAndGrowth() {
  auto c = SyntheticConfig();
  const auto ar = qfn::PlanSessionMemory(c, SessionMode::kAutoregressive,
                                         262144, 4224, 1, 0, 0, 20, 2720);
  Require(ar.TotalBytes() == 6812150804ULL,
          "AR has no predictor, draft table, hidden verification or rollback");
  Require(ar.rollback_bytes == 0 && ar.mtp_kv_bytes == 0 &&
              ar.mtp_index_bytes == 0 && ar.mtp_hidden_bytes == 0 &&
              ar.mtp_draft_table_bytes == 0,
          "loaded predictor weights do not make an AR slot speculative");
  const auto narrow = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                             262144, 4224, 8, 8, 7, 20, 2720);
  const auto wide = qfn::PlanSessionMemory(c, SessionMode::kSpeculative, 262144,
                                           4224, 8, 64, 7, 20, 2720);
  Require(wide.TotalBytes() - narrow.TotalBytes() == 2293760 &&
              wide.rollback_bytes == narrow.rollback_bytes,
          "wide verification only grows private retained hidden rows here");
  const auto no_subset = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                                262144, 4224, 8, 8, 7, 20, 0);
  Require(no_subset.mtp_draft_table_bytes == 0 &&
              narrow.TotalBytes() - no_subset.TotalBytes() == 2793472,
          "full-vocabulary drafts do not allocate the private subset table");
  c.ple_layer = -1;
  const auto no_ple = qfn::PlanSessionMemory(c, SessionMode::kSpeculative,
                                             262144, 4224, 8, 8, 7, 20, 2720);
  Require(narrow.TotalBytes() - no_ple.TotalBytes() == 2949120,
          "PLE current state and seven rollback histories are counted once");
  Require(qfn::SessionIndexerCapacity(c, 4224, 262144) == 8192 &&
              qfn::SessionIndexerCapacity(c, 16, 1024) == 1024,
          "indexer retention follows production ring bounds");
  Require(
      qfn::GdnRollbackRowFloats(0, 16, 48, 128) == 786432 &&
          qfn::GdnRollbackRowFloats(1, 16, 48, 128) == 8288,
      "first frontier is full state, subsequent frontiers are exact deltas");
}

void CopyWidthSelection() {
  const auto config = SyntheticConfig();
  const auto estimate = [&](std::uint32_t rows) {
    auto session = qfn::PlanSessionMemory(config, SessionMode::kSpeculative,
                                          262144, 4224, 8, rows, 7, 20, 2720);
    session.vision_descriptor_bytes = 64;
    return qfn::CopyWidthMemory{
        session.TotalBytes(),
        qfn::PlanDeferredScratchBytes(config, true, 8, 8, rows, false, false, 0,
                                      7)};
  };
  const auto narrow = estimate(8);
  const auto wide = estimate(64);
  Require(narrow.per_session_bytes == 7527108948ULL &&
              narrow.deferred_bytes == 418013184ULL &&
              wide.per_session_bytes == 7529402708ULL &&
              wide.deferred_bytes == 2321955840ULL,
          "production formulas cover both copy widths");
  Require(
      wide.deferred_bytes - narrow.deferred_bytes == 1903942656ULL &&
          4 * (wide.per_session_bytes - narrow.per_session_bytes) == 9175040,
      "optional deferred and initially committed width costs are distinct");

  constexpr std::size_t device = 40497627136ULL;
  constexpr std::size_t host_available = 38883639296ULL;
  constexpr std::size_t runtime = 34588672000ULL;
  constexpr std::size_t cache = 4294967296ULL;
  constexpr std::size_t margin = 536870912ULL;
  Require(host_available - cache == runtime,
          "host ceiling retains its existing OS reserve");
  const auto device_gate = vision::PlanStartupMemory(
      device, wide.deferred_bytes, wide.per_session_bytes, 0, true);
  Require(device_gate.Fits(4), "64 rows fit the text/device gate");
  std::vector<std::uint32_t> evaluated;
  const auto choose = [&] {
    evaluated.clear();
    return qfn::SelectOnDemandCopyWidth([&](std::uint32_t rows) {
      evaluated.push_back(rows);
      return estimate(rows);
    });
  };
  const auto ordinary = choose();
  Require(ordinary.rows == 8 && evaluated == std::vector<std::uint32_t>{8} &&
              ordinary.memory.per_session_bytes == narrow.per_session_bytes &&
              ordinary.memory.deferred_bytes == narrow.deferred_bytes,
          "on-demand installs and estimates only ordinary verification before "
          "sessions allocate");
  Require(
      vision::PlanStartupMemory(device, ordinary.memory.deferred_bytes,
                                ordinary.memory.per_session_bytes, 0, true)
          .Fits(4),
      "unfunded optional acceleration cannot become a text startup refusal");
  const auto needed =
      4 * wide.per_session_bytes + wide.deferred_bytes + cache + margin;
  Require(needed > runtime && needed <= device,
          "device admission alone can enable a costly optional width");
  // Even an observer returning enough startup headroom for every preference
  // cannot establish an enduring runtime grant for optional wider execution.
  vision::AdmissionBudget affordable([&] { return needed; });
  auto startup_capacity = affordable.Reserve(needed);
  Require(choose().rows == 8 && evaluated == std::vector<std::uint32_t>{8},
          "affordable startup memory does not re-enable permanent wide work");
  startup_capacity = {};
  const auto text_short =
      4 * narrow.per_session_bytes + narrow.deferred_bytes - 1;
  Require(ordinary.rows == 8 && !vision::PlanStartupMemory(
                                     text_short, ordinary.memory.deferred_bytes,
                                     ordinary.memory.per_session_bytes, 0, true)
                                     .Fits(4),
          "fallback does not bypass genuine text-state insufficiency");
  Require(!vision::PlanStartupMemory(device, ordinary.memory.deferred_bytes,
                                     ordinary.memory.per_session_bytes, 0, true)
               .Fits(std::numeric_limits<std::size_t>::max()),
          "the unchanged text gate rejects overflowing session requirements");
  Require(qfn::PlanDeferredScratchBytes(config, true, 8, 8, 64, false, false, 7,
                                        7) == narrow.deferred_bytes &&
              qfn::PlanDeferredScratchBytes(config, true, 8, 8, 64, true, true,
                                            7, 7) == 236077056ULL,
          "committed wide, batch and checkpoint buffers leave only real "
          "remaining obligations");
}

template<class Action>
void Refused(Action action, std::string_view message) {
  try {
    action();
  } catch (const gufo::core::ResourceAdmissionError&) {
    return;
  }
  Require(false, message);
}

void WarmedCopyWidthAdmission() {
  const auto config = SyntheticConfig();
  const auto wide =
      qfn::PlanDeferredScratchBytes(config, true, 8, 8, 64, true, true, 0, 7);
  const auto ordinary = qfn::SelectOnDemandCopyWidth([&](std::uint32_t rows) {
    return qfn::CopyWidthMemory{
        qfn::PlanSessionMemory(config, SessionMode::kSpeculative, 262144, 4224,
                               8, rows, 7, 20, 2720)
            .TotalBytes(),
        qfn::PlanDeferredScratchBytes(config, true, 8, 8, rows, true, true, 0,
                                      7)};
  });
  constexpr std::size_t pending_with_wide = 2140019904ULL;
  const auto other_pending = pending_with_wide - wide;
  Require(other_pending == 192,
          "retain the unexplained pending difference without crediting it");
  const auto pending_ordinary = ordinary.memory.deferred_bytes + other_pending;
  Require(ordinary.rows == 8 && pending_ordinary == 236077248 &&
              pending_with_wide - pending_ordinary == 1903942656ULL,
          "only prevented wide segments leave the warmed execution obligation");

  // Injected runtime headroom and image bytes, not a reconstruction from an
  // asynchronous host-memory sample. No physical GiB allocations are made.
  constexpr std::size_t headroom = std::size_t{512} << 20;
  constexpr std::size_t image_bytes = std::size_t{64} << 20;
  vision::AdmissionBudget before([] { return headroom; }, pending_with_wide);
  Refused([&] { (void)before.Reserve(image_bytes); },
          "unmaterialized wide execution can exhaust image admission");
  Require(before.Capacity() == 0 && !before.Outstanding(),
          "refusal neither allocates nor leaks an image reservation");
  vision::AdmissionBudget after([] { return headroom; }, pending_ordinary);
  {
    auto image = after.Reserve(image_bytes);
    Require(after.ProtectedBytes() == pending_ordinary &&
                after.PendingBytes() == image_bytes,
            "ordinary execution remains protected while an image is admitted");
    auto shared =
        after.ReserveShared(headroom - pending_ordinary - image_bytes);
    Refused([&] { (void)after.Reserve(1); },
            "images and shared growth still cannot spend execution capacity");
  }
  Require(after.ProtectedBytes() == pending_ordinary && !after.Outstanding(),
          "cancellation releases optional work without weakening execution");
  auto later = after.Reserve(image_bytes);
  Require(after.Outstanding() == image_bytes,
          "refusal and cancellation leave subsequent image admission healthy");
}

void SnapshotAdmissionOwnership() {
  // The accepted 600 covers text and deferred execution, including 200 for
  // mutable-state spares. Optional future cache growth is not an obligation.
  std::size_t free = 1000;
  auto budget =
      std::make_shared<vision::AdmissionBudget>([&] { return free; }, 600);
  qfn::SnapshotAdmission snapshots(budget, 200);
  auto image = budget->Reserve(250);
  auto first = snapshots.Reserve(100, true);
  auto second = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 250,
          "two mutable blocks use protected scratch without a second charge");
  auto third = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 350,
          "additional live mutable copies compete with images");
  Refused(
      [&] { (void)snapshots.Reserve(51, false); },
      "aggregate image and private copies cannot exceed available capacity");
  auto extra = snapshots.Reserve(50, false);
  Require(budget->AccountedBytes() == 1000, "exact aggregate boundary fits");
  Refused([&] { (void)budget->Reserve(1); },
          "images cannot take an admitted private snapshot's bytes");
  free -= 100;  // Synchronized physical allocation precedes conversion.
  auto retained = snapshots.Commit(third);
  auto spare = std::move(retained);
  Require(budget->Outstanding() == 400,
          "moving a released allocation into retained spares keeps its charge");
  extra = {};
  Require(budget->Outstanding() == 350,
          "only the genuinely released allocation returns capacity");
  auto reused = std::move(spare);
  Require(budget->Outstanding() == 350,
          "reusing retained storage neither releases nor reserves twice");
  first = {};
  auto replacement = snapshots.Reserve(100, true);
  Require(budget->Outstanding() == 350,
          "freed protected allocation makes its existing allowance reusable");
  reused.BeforeRelease();
  free += 100;
  reused = {};
  Require(
      budget->Outstanding() == 250,
      "freeing the optional physical allocation releases its shared ticket");
  Refused(
      [&] {
        (void)snapshots.Reserve(std::numeric_limits<std::size_t>::max(), false);
      },
      "overflow-sized optional copies fail without changing ownership");
  Require(budget->Outstanding() == 250,
          "failed growth leaves existing images and spare claims intact");
}

void SnapshotAdmissionFailuresAndModes() {
  std::size_t free = 100;
  auto budget =
      std::make_shared<vision::AdmissionBudget>([&] { return free; }, 60);
  qfn::SnapshotAdmission snapshots(budget, 20);
  for (const bool mutable_state : {false, true}) {
    try {
      auto pending = snapshots.Reserve(20, mutable_state);
      // Inject failure after admission, before allocation/conversion.
      throw std::bad_alloc();
    } catch (const std::bad_alloc&) {
    }
    Require(budget->Outstanding() == 0,
            "failed allocation rolls back its unmaterialized ticket");
  }
  auto protected_block = snapshots.Reserve(20, true);
  auto active = snapshots.Reserve(40, true);
  Refused([&] { (void)snapshots.Reserve(1, false); },
          "failure and retry preserve the protected-spare bound");
  active = {};
  auto pending_copy = snapshots.Reserve(40, false);
  free -= 40;
  auto copy = snapshots.Commit(pending_copy);
  Require(budget->Outstanding() == 40,
          "successful conversion retains capacity through asynchronous use");
  copy.BeforeRelease();
  free += 40;
  copy = {};
  Require(budget->Outstanding() == 0,
          "completion and release restore capacity for later requests");
  auto image = budget->Reserve(40);
  Refused([&] { (void)snapshots.Reserve(21, true); },
          "changing mutable size cannot enlarge the protected spare allowance");
  qfn::SnapshotAdmission conservative(nullptr, 20);
  Require(
      !conservative.Reserve(1000, false) && !conservative.Reserve(1000, true),
      "conservative snapshot behavior remains unchanged");
}

void LazyExecutionMaterialization() {
  std::size_t free = 200;
  auto budget =
      std::make_shared<vision::AdmissionBudget>([&] { return free; }, 150);
  qfn::ExecutionCommitment execution(budget);
  free -= 60;
  execution.SetCommitted(60);
  Require(budget->ProtectedBytes() == 90,
          "initial text storage leaves lazy rollback and scratch outstanding");
  auto image = budget->Reserve(50);
  Refused([&] { (void)budget->Reserve(1); },
          "optional work cannot consume accepted but unmaterialized rollback");
  free -= 40;
  execution.SetCommitted(100);
  Require(
      budget->ProtectedBytes() == 50,
      "successful lazy rollback is reflected once in the fresh observation");
  Refused([&] { (void)budget->Reserve(1); },
          "materializing rollback creates no extra optional capacity");
  // Model partial trimming: restore its obligation before the physical free.
  execution.SetCommitted(80);
  Refused([&] { (void)budget->Reserve(1); },
          "trim cannot lend a still-live row to image work");
  free += 20;
  Refused(
      [&] { (void)budget->Reserve(1); },
      "freed rollback remains available for accepted future text execution");
  image = {};
  // One row succeeds, a later allocation fails. Only the completed row is
  // materialized; the remainder is still protected without a physical claim.
  free -= 10;
  execution.SetCommitted(90);
  try {
    throw std::bad_alloc();
  } catch (const std::bad_alloc&) {
  }
  Require(execution.Bytes() == 90 && budget->ProtectedBytes() == 60,
          "partial allocation failure preserves actual execution ownership");
  execution.Release();
  Refused([&] { (void)budget->Reserve(1); },
          "reset restores all future execution before freeing its buffers");
  free += 90;
  auto later_image = budget->Reserve(50);
  Require(budget->ProtectedBytes() == 150 && budget->Outstanding() == 50,
          "reset and failed growth leave later requests healthy");
  qfn::ExecutionCommitment conservative;
  conservative.SetCommitted(1000);
  Require(conservative.Bytes() == 0,
          "default execution allocation semantics are unchanged");
}

void ProtectedSnapshotMaterialization() {
  std::size_t free = 100;
  auto budget =
      std::make_shared<vision::AdmissionBudget>([&] { return free; }, 60);
  qfn::SnapshotAdmission snapshots(budget, 20);
  auto pending = snapshots.Reserve(20, true);
  free -= 20;
  auto spare = snapshots.Commit(pending);
  Require(budget->ProtectedBytes() == 40 && budget->PendingBytes() == 0,
          "committed protected spare is reflected in free memory, not "
          "subtracted twice as unmaterialized execution");
  auto image = budget->Reserve(40);
  Refused([&] { (void)budget->Reserve(1); },
          "materializing accepted execution does not create new headroom");
  auto reused = spare;
  spare = {};
  Require(budget->ProtectedBytes() == 40,
          "retained or shared spare owns its execution commitment");
  // Restore the obligation first, then release its physical memory. Queries
  // between these operations are conservative and cannot lend it to images.
  reused.BeforeRelease();
  Refused([&] { (void)budget->Reserve(1); },
          "pending execution returns before protected physical storage frees");
  free += 20;
  reused = {};
  Require(budget->ProtectedBytes() == 60,
          "last protected owner restores its complete execution obligation");
  image = {};
  auto again = snapshots.Reserve(20, true);
  free -= 20;
  auto second = snapshots.Commit(again);
  second.BeforeRelease();
  free += 20;
  second = {};
  Require(budget->ProtectedBytes() == 60 && !budget->Outstanding(),
          "repeated spare lifetimes neither leak nor duplicate reservations");

  gufo::cache::ResourceReservation no_optional_ticket;
  free -= 10;
  auto checkpoint = snapshots.Commit(no_optional_ticket, 10);
  Require(budget->ProtectedBytes() == 50,
          "checkpoint slot zero materializes its accepted deferred allowance");
  checkpoint.BeforeRelease();
  free += 10;
  checkpoint = {};
  Require(budget->ProtectedBytes() == 60,
          "checkpoint release restores only its own deferred allowance");
}

void ConcurrentSnapshotAdmission() {
  auto budget =
      std::make_shared<vision::AdmissionBudget>([] { return 100; }, 80);
  qfn::SnapshotAdmission snapshots(budget, 0);
  std::barrier rendezvous(3);
  std::atomic<unsigned> admitted{}, refused{};
  const auto contender = [&](bool image) {
    rendezvous.arrive_and_wait();
    gufo::cache::ResourceReservation reservation;
    try {
      reservation = image ? budget->Reserve(20) : snapshots.Reserve(20, false);
      ++admitted;
    } catch (const gufo::core::ResourceAdmissionError&) {
      ++refused;
    }
    // Hold the winner's reservation until both admission attempts completed.
    rendezvous.arrive_and_wait();
  };
  std::thread image(contender, true);
  std::thread copy(contender, false);
  rendezvous.arrive_and_wait();
  rendezvous.arrive_and_wait();
  image.join();
  copy.join();
  Require(
      admitted == 1 && refused == 1 && budget->Outstanding() == 0,
      "concurrent images and private snapshots cannot spend the same bytes");
}
}  // namespace

int main() {
  SessionComponents();
  ModesAndGrowth();
  CopyWidthSelection();
  WarmedCopyWidthAdmission();
  SnapshotAdmissionOwnership();
  SnapshotAdmissionFailuresAndModes();
  LazyExecutionMaterialization();
  ProtectedSnapshotMaterialization();
  ConcurrentSnapshotAdmission();
  std::cout << "Flash-Next production memory components passed\n";
}
