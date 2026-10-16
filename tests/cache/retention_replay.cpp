// CPU policy replay, with measured aggregate payload sizes and no numerical
// claims. Input is produced by retention_trace.py from the pinned RFC archive.
#include <cassert>
#include <fstream>
#include <iostream>

#include "src/cache/retention.hpp"
#include "tests/cache/fake_adapter.hpp"
using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {
struct Empty final : MutationGuard {
  void BeforeOverwrite(ComponentId, Rows, Rows) override {}
  void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
};
struct Log final : RetentionEventSink {
  std::ofstream file;
  std::size_t request{}, admitted{}, refused{}, removed{}, peak{},
      payload_peak{};
  explicit Log(const char* path) : file(path) {
    if (!file)
      throw std::runtime_error("cannot open event log");
    file << "request,sequence,action,reason,id,boundary,purpose,rank,last_used,"
            "freed,records,ledger_bytes,retained_payload_bytes\n";
  }
  void Emit(const RetentionEvent& e) noexcept override {
    admitted += e.action == RetentionAction::kAdmitted;
    refused += e.action == RetentionAction::kRefused;
    removed += e.action == RetentionAction::kRemoved &&
               e.reason != RetentionReason::kShutdown;
    peak = std::max(peak, e.ledger_bytes);
    payload_peak = std::max(payload_peak, e.retained_payload_bytes);
    file << request << ',' << e.sequence << ',' << static_cast<int>(e.action)
         << ',' << static_cast<int>(e.reason) << ',' << e.checkpoint.value
         << ',' << e.boundary << ',' << static_cast<int>(e.purpose) << ','
         << e.rank << ',' << e.last_used << ',' << e.unique_bytes_freed << ','
         << e.retained_records << ',' << e.ledger_bytes << ','
         << e.retained_payload_bytes << '\n';
  }
};
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  std::size_t fixed{}, row{}, budget{}, requests{};
  if (!(std::cin >> fixed >> row >> budget >> requests))
    return 2;
  const std::array<ComponentDescriptor, 2> descriptors{
      {{{1}, 1, ComponentKind::kAppendRows, row, 2048, 0},
       {{2}, 1, ComponentKind::kPrivateState, 0, 0, fixed}}};
  ResourceLedger ledger({budget, budget, 0, 0});
  Log log(argv[1]);
  std::size_t final_payload{}, final_records{}, lifetimes{}, required_splits{},
      optional_splits{};
  {
    FakeAdapter adapter;
    Empty guard;
    auto slot = adapter.CreateSlot(guard);
    PrefixIndex index(ledger);
    index.Register({1}, descriptors);
    auto policy = std::make_unique<RetentionPolicy>(ledger, index, &log);
    std::uint64_t lifetime = ~std::uint64_t{0};
    for (log.request = 0; log.request < requests; ++log.request) {
      std::uint64_t next{};
      std::size_t count{}, stable{};
      if (!(std::cin >> next >> count >> stable))
        return 2;
      std::vector<Token> tokens(count);
      for (auto& t : tokens)
        if (!(std::cin >> t))
          return 2;
      if (next != lifetime) {
        if (lifetimes++) {
          policy.reset();
          policy = std::make_unique<RetentionPolicy>(ledger, index, &log);
        }
        lifetime = next;
      }
      auto lookup = index.Lookup({{1}, tokens, InputIdentity(count), stable});
      Rows reused = 0;
      CheckpointId source;
      auto history =
          lookup.selected
              ? ExecutionHistory::Restored(ledger, *lookup.selected->checkpoint)
              : ExecutionHistory::Cold(ledger, descriptors, {1});
      if (lookup.selected) {
        reused = lookup.selected->boundary;
        source = lookup.selected->checkpoint->Id();
        policy->Touch(source);
      }
      std::vector<Rows> retained;
      for (const auto& c : lookup.candidates)
        retained.push_back(c.boundary);
      const auto common =
          index.CommonPrefixTokens({{1}, tokens, InputIdentity(count)});
      const auto candidates =
          PlanCaptures(adapter, *slot,
                       {count, reused, stable, common, true, true, retained});
      // Candidate handles are selection readers, not retention owners. The
      // selected source remains protected by its ID throughout this request;
      // release lookup pins before admission so victims can reclaim storage.
      lookup = {};
      for (const auto& candidate : candidates) {
        if (candidate.splits_pass) {
          if (candidate.required)
            ++required_splits;
          else
            ++optional_splits;
        }
        const auto prefix =
            std::span<const Token>(tokens).first(candidate.boundary);
        const auto purpose = candidate.purpose == CheckpointPurpose::kLearned
                                 ? RetentionPurpose::kBranchPoint
                             : candidate.purpose == CheckpointPurpose::kGrid
                                 ? RetentionPurpose::kHistory
                             : candidate.boundary == count && stable < count
                                 ? RetentionPurpose::kRetry
                                 : RetentionPurpose::kContinuation;
        const std::vector<ComponentPosition> positions{{{1}, prefix.size()},
                                                       {{2}, prefix.size()}};
        RetentionRequest request{
            prefix, {1}, {}, purpose, std::min<Rows>(stable, prefix.size()),
            source};
        request.new_payload_bytes = history.NewPayloadBytes(positions);
        (void)policy->Admit(request, [&] {
          return history.Capture(
              {prefix,
               InputIdentity(prefix.size()),
               {{{1}, prefix.size()}, {{2}, prefix.size()}},
               candidate.purpose,
               3},
              [&](const PayloadRequest& r) {
                auto pool =
                    ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                        .Convert();
                auto charge = pool.ReserveBacking(r.category).Convert();
                return Payload::Committed(std::move(charge),
                                          std::make_shared<int>(0));
              });
        });
      }
    }
    final_payload = policy->RetainedPayloadBytes();
    final_records = policy->Size();
  }
  assert(ledger.Snapshot().total_bytes == 0);
  log.file.flush();
  if (!log.file)
    return 1;
  std::cout << "{\"requests\":" << requests << ",\"admitted\":" << log.admitted
            << ",\"refused\":" << log.refused << ",\"evicted\":" << log.removed
            << ",\"peak_ledger_bytes\":" << log.peak
            << ",\"peak_retained_payload_bytes\":" << log.payload_peak
            << ",\"required_splits\":" << required_splits
            << ",\"optional_splits\":" << optional_splits
            << ",\"final_records\":" << final_records
            << ",\"final_payload_bytes\":" << final_payload << "}\n";
}
