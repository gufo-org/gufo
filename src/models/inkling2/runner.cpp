// Gufo text runner for Inkling 2 Mini: component cache (cache_adapter.*,
// state_layout.*) with prefix reuse, RAM and disk cache, --sessions N with
// per-session memory claims, layered prefill, batched decode across sessions.
// Speculative decoding: DSpark (--dspark-model), verified exactly (see
// DecodeStep).
//
// M5: image input. The mmproj (--mmproj) is encoded by VisionEncoder; the
// prompt carries <|content_image|>, one placeholder token per image row,
// <|end_message|>, and Prefill injects the rows in place of the placeholders'
// embeddings (Engine::Forward InputRows). The cache identity carries the
// content hash of every image.
#include "src/models/inkling2/runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/experimental/admission_query.hpp>
#include <gufo/ext/experimental/state_memory.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/prompt_lookup.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/sha256.hpp>
#include <gufo/ext/transfer_pool.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "src/models/inkling2/cache_adapter.hpp"
#include "src/models/inkling2/dspark_policy.hpp"
#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/image_cache.hpp"
#include "src/models/inkling2/text/gguf_meta.hpp"
#include "src/models/inkling2/text/gufo_adapter.hpp"
#include "src/models/inkling2/vision/vision_encoder.hpp"

namespace gufo::models::inkling2 {
namespace {

namespace g = ::gufo::ext;
namespace ga = ::inkling2::text::gufo_adapter;

constexpr std::string_view kStateAbi = "inkling2-hip-m2-v2";
/// Layer groups one pass may be split into (48 visits; the host caps it by pass
/// length).
constexpr std::size_t kMaxLayerGroups = 16;
/// Prompt tokens of a pass while another request could join or decodes: an
/// arrival waits for at most one such pass (about 0.9 s). A prompt nobody can
/// join takes the engine's wider passes (Engine::kUbatch).
constexpr std::size_t kSharedPassTokens = 2048;
constexpr std::string_view kModelId = "inkling-2-mini";

struct Inkling2PromptContext;

class Inkling2State final : public g::TextRunnerState {
public:
  Inkling2State(Inkling2Slot& slot, cache::Adapter& adapter)
      : slot_(&slot),
        session_(&slot.session()),
        adapter_(&adapter),
        block(slot.block) {}
  void Invalidate() noexcept override { (void)adapter_->Invalidate(*slot_); }
  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] Inkling2Slot& slot() const { return *slot_; }
  [[nodiscard]] Inkling2CacheAdapter& adapter() const {
    return static_cast<Inkling2CacheAdapter&>(*adapter_);
  }
  /// Where the generated text stands inside the model's block structure (see
  /// Piece()). Lives in the slot: it is part of a checkpoint's metadata.
  Block& block;
  /// A reasoning budget closed the thought: no second thinking block may open
  /// (the model reopens one right after the forced close). Per request.
  bool budget_closed{false};
  /// A thought of this turn closed. With a call required no second one may
  /// open (Select): a model that decides against the call thinks again
  /// instead, past the one reasoning span the parsers know, until the token
  /// limit. With tools it may call, a second thought often plans the call, so
  /// it stays. Per request.
  bool thought_closed{false};
  /// A call is required: after a header the model writes a thought (unless
  /// a budget closed it) or one declared function name
  /// (the tokens the template writes for it), then the invoke token (see
  /// Select). Token ids of each declared name (of every request that may
  /// call one; see Piece()), as the template writes it, and the name tokens
  /// written since the last header.
  bool call_required{false};
  std::vector<std::vector<g::TextRunnerToken>> call_names;
  std::vector<g::TextRunnerToken> call_prefix;
  /// The request's images (set by SetPromptContext; null for a text-only
  /// prompt).
  std::shared_ptr<const Inkling2PromptContext> prompt;
  /// DSpark: the token a cycle selected after its last kept row (drawing from
  /// the request's sampler) but did not emit; the next cycle emits it first,
  /// and SelectNext returns it. Per request.
  std::optional<g::TextDecodeSelection> pending;
  DsparkBackoff backoff;
  /// Copy drafts: the request's committed tokens (prompt and generated, valid
  /// when `history_ok`) and their index. They only propose: every token is
  /// still the one verification selects. Per request.
  std::vector<std::int32_t> history;
  bool history_ok{false};
  speculative::PromptLookup<std::int32_t> lookup;
  CopyBackoff copy_backoff;
  /// Wide copies (a request decoding alone): `copy_run` when the last cycle's
  /// copied proposals were all kept (the next cycle copies first);
  /// `copy_rows` the rows a copy may verify (kNarrowRows, doubling to
  /// Engine::kMaxVerifyRows after two whole copies of that width in a row,
  /// back to kNarrowRows after any rejected copy).
  bool copy_run{false};
  int copy_rows{0};
  int copy_streak{0};
  [[nodiscard]] std::shared_ptr<mem::StateAccount> state_account()
      const override {
    return session_->account();
  }
  /// Polled between a prefill pass's visits (Engine::Forward).
  void SetCancellationCheck(const CancellationCheck& check) override {
    cancelled = check;
  }
  CancellationCheck cancelled;
  [[nodiscard]] g::TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->allocated_bytes(),
            .temporary_scratch_bytes = 0};
  }

private:
  Inkling2Slot* slot_;
  Session* session_;
  cache::Adapter* adapter_;
};

/// Threads that stay up between decode steps for the requests' token
/// selections (a host sampler pass over the vocabulary per sampled row).
/// Run(n, f) calls f(0 .. n - 1) once each: the caller starts at once and the
/// workers take the indices left as they wake, so short work never waits for
/// a thread. One Run at a time; a second caller runs its indices alone.
class LanePool {
public:
  explicit LanePool(unsigned threads) {
    for (unsigned i = 0; i < threads; ++i)
      threads_.emplace_back([this] { Work(); });
  }
  ~LanePool() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_)
      t.join();
  }
  LanePool(const LanePool&) = delete;
  LanePool& operator=(const LanePool&) = delete;

  template<class F>
  void Run(std::size_t n, F&& f) {
    std::unique_lock owner(owner_, std::try_to_lock);
    if (n < 2 || !owner.owns_lock() || threads_.empty()) {
      for (std::size_t k = 0; k < n; ++k)
        f(k);
      return;
    }
    struct alignas(128) Cursor {
      std::atomic<std::size_t> next{0};
    } cursor;
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto share = [&] {
      for (;;) {
        const std::size_t k = cursor.next.fetch_add(1);
        if (k >= n)
          return;
        try {
          f(k);
        } catch (...) {
          const std::lock_guard lock(failure_mutex);
          if (!failure)
            failure = std::current_exception();
        }
      }
    };
    using Share = decltype(share);
    {
      const std::lock_guard lock(mutex_);
      invoke_ = [](const void* self) { (*static_cast<const Share*>(self))(); };
      argument_ = &share;
      open_ = true;
      ++generation_;
    }
    wake_.notify_all();
    share();
    {
      // workers that wake from here on find the run closed
      std::unique_lock lock(mutex_);
      open_ = false;
      done_.wait(lock, [&] { return active_ == 0; });
    }
    if (failure)
      std::rethrow_exception(failure);
  }

private:
  void Work() {
    std::uint64_t seen = 0;
    std::unique_lock lock(mutex_);
    for (;;) {
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_)
        return;
      seen = generation_;
      if (!open_)
        continue;
      const auto invoke = invoke_;
      const void* argument = argument_;
      ++active_;
      lock.unlock();
      invoke(argument);
      lock.lock();
      if (--active_ == 0)
        done_.notify_one();
    }
  }

  std::mutex owner_, mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> threads_;
  void (*invoke_)(const void*){nullptr};
  const void* argument_{nullptr};
  std::size_t active_{0};
  std::uint64_t generation_{0};
  bool open_{false}, stop_{false};
};

Inkling2State& Require(g::TextRunnerState& s) {
  auto* p = dynamic_cast<Inkling2State*>(&s);
  if (p == nullptr)
    throw std::logic_error("text runner state is not Inkling 2 Mini");
  return *p;
}
const Inkling2State& Require(const g::TextRunnerState& s) {
  const auto* p = dynamic_cast<const Inkling2State*>(&s);
  if (p == nullptr)
    throw std::logic_error("text runner state is not Inkling 2 Mini");
  return *p;
}

/// One image of a request: its decoded pixels and, once a prefill needs them,
/// its encoder rows (both shared through the runner's ImageCache).
struct ImageEntry {
  std::shared_ptr<const DecodedImage> image;
  std::size_t first = 0;  // prompt index of the first placeholder token
  std::size_t count = 0;  // rows = placeholder tokens
  // Set by Inkling2Runner::EnsureRows (under `mutex`).
  std::mutex mutex;
  std::shared_ptr<const ImageRows> rows;
};

/// What a request tells the runner beyond its tokens (SetPromptContext).
struct Inkling2PromptContext final : g::TextPromptContext {
  bool call_required{false};
  std::vector<std::vector<g::TextRunnerToken>> call_names;
  std::vector<std::unique_ptr<ImageEntry>> images;
  /// The prompt's tokens, for copy drafts (a request whose whole prompt is
  /// restored from the cache never prefills).
  std::vector<std::int32_t> tokens;
  /// Tool names and the call requirement come from the request, not its
  /// tokens. Image rows sit at token positions. Copy drafts only propose, so
  /// another tokenization of the same text costs drafts, never output (and
  /// prefill replaces the copy history with the tokens it ran).
  [[nodiscard]] bool TokenizationIndependent() const override {
    return images.empty();
  }
};

std::vector<std::int32_t> ToIds(std::span<const g::TextRunnerToken> t) {
  return {t.begin(), t.end()};
}

class Inkling2Runner final : public g::HipTextModelRunner {
public:
  Inkling2Runner(std::shared_ptr<Engine> engine,
                 std::unique_ptr<::inkling2::text::Tokenizer> tok,
                 std::uint32_t max_context, cache::Identity identity,
                 bool durable,
                 std::unique_ptr<vision::VisionEncoder> vision = nullptr,
                 std::string vision_identity = {}, int max_draft = 0)
      : engine_(std::move(engine)),
        tok_(std::move(tok)),
        max_context_(max_context),
        identity_(std::move(identity)),
        durable_(durable),
        vision_(std::move(vision)),
        vision_identity_(std::move(vision_identity)),
        max_draft_(static_cast<std::size_t>(
            std::clamp(max_draft, 0, Drafter::kMaxDraft))) {
    namespace d = ::inkling2::text::dialect;
    const auto id = [&](std::string_view spelling) {
      const auto t = tok_->FindSpecial(spelling);
      return t ? static_cast<g::TextRunnerToken>(*t)
               : g::TextRunnerToken{0xFFFFFFFFu};
    };
    t_header_ = id(d::kHeader);
    t_text_ = id(d::kContentText);
    t_thinking_ = id(d::kThinking);
    t_end_message_ = id(d::kEndMessage);
    t_invoke_ = id(d::kInvokeJson);
  }

  // ---- descriptor, claim, plans ----
  [[nodiscard]] g::TextRunnerDescriptor Descriptor() const override {
    g::TextRunnerDescriptor d;
    d.model_id = std::string(kModelId);
    d.state_abi = std::string(kStateAbi);
    d.max_context = max_context_;
    d.capabilities.supports_vision = vision_ != nullptr;
    d.capabilities.incremental_prefill = true;
    d.capabilities.final_token_advance_required = false;
    d.capabilities.incremental_text_is_exact =
        true;  // the streamed pieces are the text (Piece())
    d.capabilities.prefix_reuse = true;
    d.capabilities.in_pass_checkpoint = true;
    d.capabilities.max_in_pass_checkpoints = Engine::kMaxPassCheckpoints;
    // A lone decoding request runs DecodeStep (with DSpark several tokens a
    // step; greedy rows' argmax on the device); concurrent ones keep batched
    // one-token steps.
    d.capabilities.multi_token_decode = engine_->batch_capable();
    d.capabilities.batched_multi_token_decode = engine_->batch_capable();
    d.capabilities.batched_multi_token_decode_max_width =
        Engine::kMaxDecodeBatch;
    sampling::TextSamplingValues
        v;  // Unsloth/OpenAI defaults: temperature 1, no truncation
    v.temperature = 1.0F;
    v.top_k = 0;
    v.top_p = 1.0F;
    v.min_p = 0.0F;
    d.sampling_defaults = sampling::TextSamplingDefaults{v, v, v};
    // A pass of one engine ubatch takes seconds on the M1 kernels: split it by
    // visits so other sessions keep decoding. The engine splits at visit
    // boundaries (Engine::Forward, PrefillYield) and the arithmetic of every
    // visit is unchanged.
    // Every decoding session may step between two groups: a batched step
    // costs little more than a lone one (inkling2_spec_test checks that
    // decoding inside a yield equals decoding alone).
    d.scheduling = g::TextSchedulingProfile{
        .pass_tokens = kSharedPassTokens,
        .max_layer_groups = kMaxLayerGroups,
        .max_yield_decode_width = Engine::kMaxDecodeBatch};
    return d;
  }
  [[nodiscard]] g::TextRunnerResourceClaim ResourceClaim() const override {
    g::TextRunnerResourceClaim c;
    c.resident_weights_bytes = engine_->weight_bytes();
    // One session = its device KV (8 global visits x max_context rows + 40
    // rings) + conv history. The host multiplies by --sessions and checks the
    // product against state_capacity_bytes, so a
    // --sessions/--context pair that cannot fit is refused at load with a
    // message that names both.
    c.per_request_state_bytes = engine_->session_bytes();
    // Activations and MoE staging are shared by all sessions and already
    // allocated by the engine.
    c.temporary_scratch_bytes = 0;
    if (vision_) {  // the encoder's weights (~0.65 GB as f16) and its workspace
                    // for a 4096-token image
      c.resident_weights_bytes =
          c.resident_weights_bytes.value_or(0) + vision_->device_bytes();
      c.temporary_scratch_bytes = c.temporary_scratch_bytes.value_or(0) +
                                  kVisionScratchBytes + ImageCache::kRowsBytes;
    }
    if (const auto memory = gufo::hip::QueryAdmissionMemory())
      c.state_capacity_bytes = memory->free_bytes;
    // RAM cache budget (--cache-ram-bytes caps it) and the host memory kept
    // free above it.
    c.retained_snapshot_capacity_bytes = g::HostSnapshotBudgetBytes();
    c.retained_snapshot_ceiling_bytes = g::HostSnapshotCeilingBytes();
    c.requires_device_runtime_lock = true;
    c.demand_mapped_state_bytes = engine_->demand_mapped_bytes();
    return c;
  }
  [[nodiscard]] std::vector<g::TextExecutionPlan> SupportedPlans()
      const override {
    // Batched decode: one weight pass for up to Engine kMaxDecodeBatch
    // sessions (AdvanceBatch).
    std::vector<g::TextExecutionPlan> plans{
        {.kind = g::TextExecutionPlanKind::kSerial, .physical_width = 1}};
    if (engine_->batch_capable())
      for (std::size_t w = 2; w <= Engine::kMaxDecodeBatch; ++w)
        plans.push_back(
            {.kind = g::TextExecutionPlanKind::kBatched, .physical_width = w});
    return plans;
  }

  // ---- block framing ----
  // The model writes a turn as blocks:
  // <|message_model|><|content_thinking|>R<|end_message|> then either
  // <|message_model|><|content_text|>A<|end_message|> or a tool block
  // <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>.
  // Gufo's response parsers know one reasoning span and "content", and
  // v0.35.0's tool grammar can name a function only once, so the framing the
  // host does not need is not streamed: the text reads
  // "<|message_model|><|content_thinking|>R<|end_message|>" (reasoning
  // open/close, see ReasoningDelimiters), then A for a text block, or
  // "<|content_invoke_tool_json|>{...}
  // <|end_message|>" for a call (stream_dialect.hpp). The decision uses only
  // committed state (Advance), so previewing a token twice is harmless.
  // Decode() stays literal.
  //
  // A grammar sees each token's own bytes, so it never allows <|content_text|>:
  // under a response format the model writes its answer right after the header
  // (unless a thought came first). An ordinary token after a header is
  // therefore a function name only when it starts a declared one; otherwise it
  // opens a text block.
  [[nodiscard]] bool StartsName(const Inkling2State& st,
                                g::TextRunnerToken token) const {
    return std::any_of(
        st.call_names.begin(), st.call_names.end(),
        [&](const auto& name) { return !name.empty() && name[0] == token; });
  }
  [[nodiscard]] std::string Piece(const Inkling2State& st,
                                  g::TextRunnerToken token) const {
    if (token == t_header_)
      return {};
    switch (st.block) {
      case Block::kAfterHeader:
        if (token == t_text_)
          return {};
        if (token == t_thinking_)
          return std::string(::inkling2::text::dialect::kHeader) +
                 std::string(tok_->TokenBytes(token));
        // The invoke token right after a header (a call written without the
        // first function name): the call starts here. The first token of a
        // function name is written again inside the JSON.
        if (token == t_invoke_)
          return std::string(tok_->TokenBytes(token));
        if (token == t_end_message_ || StartsName(st, token))
          return {};  // an empty block, or a name
        break;
      case Block::kTool:
        return token == t_invoke_ ? std::string(tok_->TokenBytes(token))
                                  : std::string{};
      case Block::kText:
        if (token == t_end_message_)
          return {};
        break;
      default:
        break;
    }
    return std::string(tok_->TokenBytes(token));
  }
  [[nodiscard]] Block NextBlock(const Inkling2State& st,
                                g::TextRunnerToken token) const {
    if (token == t_header_)
      return Block::kAfterHeader;
    switch (st.block) {
      case Block::kAfterHeader:
        if (token == t_text_)
          return Block::kText;
        if (token == t_thinking_)
          return Block::kThinking;
        if (token == t_invoke_)
          return Block::kToolBody;
        if (token == t_end_message_)
          return Block::kIdle;
        return StartsName(st, token) ? Block::kTool : Block::kText;
      case Block::kTool:
        if (token == t_invoke_)
          return Block::kToolBody;
        return token == t_end_message_ ? Block::kIdle : st.block;
      default:
        return token == t_end_message_ ? Block::kIdle : st.block;
    }
  }

  // ---- text front-end (src/text) ----
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        static_cast<std::uint32_t>(tok_->VocabSize()),
        [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              std::string(tok_->TokenBytes(id)), tok_->IsStop(id)};
        });
  }
  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(::inkling2::text::dialect::kId);
  }
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return ga::BudgetTags();
  }
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    // The streamed text carries the open framing and one <|end_message|>
    // closing the block (see Piece(); a text block's own header is dropped, so
    // no <|content_text|> follows the close).
    // A turn is a list of blocks: the model can think again after an answer
    // block (text, thought, text). The later thought streams the same opener
    // and joins the reasoning, as llama.cpp, vLLM and SGLang return it.
    return {std::string(::inkling2::text::dialect::kReasoningOpen),
            std::string(::inkling2::text::dialect::kEndMessage),
            /*repeats=*/true};
  }
  /// Grammars see the tokens' own bytes. The prompt already wrote the
  /// thought's header, so the model opens it with <|content_thinking|> alone
  /// (the streamed opener would make it write a second header, or skip the
  /// thought), and closes it with the answer block's framing, which the
  /// streamed text drops. A call written after the thought ends the phase at
  /// its header.
  [[nodiscard]] std::optional<ReasoningMarkup> ReasoningGrammarMarkers(
      const g::ChatRequest&) const override {
    return ReasoningMarkup{
        std::string(::inkling2::text::dialect::kThinking),
        std::string(::inkling2::text::dialect::kReasoningClose)};
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(tok_->VocabSize());
  }
  [[nodiscard]] std::vector<g::TextRunnerToken> Tokenize(
      std::string_view text) const override {
    const auto ids = tok_->Encode(text, true);
    return {ids.begin(), ids.end()};
  }
  [[nodiscard]] std::optional<std::vector<g::TextRunnerToken>>
  RenderAndTokenize(const g::ChatRequest& request) const override {
    return ga::RenderAndTokenize(*tok_, request);
  }
  /// The cache snapshots the prompt before the assistant framing this runner
  /// appends when reasoning is off (<|message_model|><|content_text|> ...): the
  /// next turn writes those tokens as part of the assistant message, so a
  /// checkpoint that includes them can never be reused.
  /// The chat template's text for /apply-template: the renderer PreparePrompt
  /// tokenizes, control tokens spelled out.
  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const g::ChatRequest& request) const override {
    ::inkling2::text::Rendered rendered;
    std::string err;
    if (!::inkling2::text::RenderChat(
            ga::ToMessages(request), ga::ToTools(request),
            ga::ToRenderOptions(request), &rendered, &err))
      throw std::invalid_argument(err);
    return rendered.Text();
  }

  [[nodiscard]] std::optional<g::TextPreparedPrompt> PreparePrompt(
      const g::ChatRequest& request) const override {
    // Images: decode them all first (the row count of each sets how many
    // placeholders the template writes), in the order the template renders them
    // (message order, then part order).
    std::vector<std::unique_ptr<ImageEntry>> images;
    for (const auto& message : request.messages) {
      for (const auto& part : message.images) {
        if (!vision_)
          throw std::invalid_argument(
              "image input requires the model's mmproj (--mmproj)");
        if (!part.bytes || part.bytes->empty())
          throw std::invalid_argument("empty image");
        auto entry = std::make_unique<ImageEntry>();
        const bool low = part.detail == g::ChatMessage::ImageDetail::kLow;
        entry->image = image_cache_.Decoded(part.bytes, low, [&] {
          auto out = std::make_shared<DecodedImage>();
          // Gufo's decoder (PNG, JPEG, WebP; display orientation, RGB8), the
          // one that validated the upload.
          try {
            const core::Image decoded = core::DecodeImage(*part.bytes);
            out->pixels.w = static_cast<int>(decoded.width);
            out->pixels.h = static_cast<int>(decoded.height);
            out->pixels.rgb = decoded.pixels;
            // detail: low caps the longer side at 512 pixels (at most 13 x 13
            // = 169 rows), as Gufo's Qwen path does.
            if (low)
              vision::ShrinkToFit(&out->pixels, kLowDetailMaxSide);
          } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("image decode failed: ") +
                                        e.what());
          }
          crypto::Sha256Hasher h;
          const std::uint64_t dims[2] = {
              static_cast<std::uint64_t>(out->pixels.w),
              static_cast<std::uint64_t>(out->pixels.h)};
          h.Update({reinterpret_cast<const std::uint8_t*>(dims), sizeof dims});
          h.Update(out->pixels.rgb);
          out->digest = h.Finish();
          return std::shared_ptr<const DecodedImage>(std::move(out));
        });
        entry->count =
            static_cast<std::size_t>(vision::VisionEncoder::CountTokens(
                entry->image->pixels.w, entry->image->pixels.h));
        images.push_back(std::move(entry));
      }
    }
    std::vector<std::size_t> counts;
    for (const auto& e : images)
      counts.push_back(e->count);
    ::inkling2::text::PromptTokens pt;
    auto tokens = ga::RenderAndTokenize(*tok_, request, &counts, &pt);
    if (!tokens)
      return std::nullopt;
    const std::size_t framing = ga::GenerationFramingTokens(request);
    const std::size_t prefix =
        framing != 0 && tokens->size() > framing ? tokens->size() - framing : 0;
    auto context = std::make_shared<Inkling2PromptContext>();
    if (max_draft_ > 0)
      context->tokens.assign(tokens->begin(), tokens->end());
    context->call_required = ga::CallRequired(request);
    if (!request.tools.empty() &&
        request.tool_choice != g::ChatRequest::ToolChoice::kNone) {
      for (const auto& tool : request.tools) {
        std::vector<::inkling2::text::TokenId> ids;
        tok_->EncodeOrdinary(tool.name, &ids);
        context->call_names.emplace_back(ids.begin(), ids.end());
      }
    }
    if (!images.empty()) {
      // The placeholders of image k are the next images[k]->count entries of
      // pt.image_positions.
      std::size_t at = 0;
      for (auto& e : images) {
        if (at + e->count > pt.image_positions.size())
          throw std::logic_error("image placeholders lost");
        e->first = pt.image_positions[at];
        at += e->count;
      }
      if (at != pt.image_positions.size())
        throw std::logic_error("image placeholders left over");
      // Cache identity: SHA-256 over the encoder identity and, per image, its
      // position, row count and pixels. The tokens (placeholders) are a
      // separate key, so a prefix reuses state only when its images are the
      // same pixels at the same positions. A boundary at or before image k's
      // first row keeps the identity of the images before it (see
      // PrefixInputIdentity).
      crypto::Sha256Hasher running;
      const auto feed = [&](const void* p, std::size_t n) {
        running.Update({static_cast<const std::uint8_t*>(p), n});
      };
      feed(vision_identity_.data(), vision_identity_.size());
      for (const auto& e : images) {
        const auto before = running.Digest();
        context->cache_prefixes.push_back(
            {e->first,
             e == images.front()
                 ? std::vector<std::uint8_t>{}
                 : std::vector<std::uint8_t>(before.begin(), before.end())});
        const std::uint64_t pos[2] = {e->first, e->count};
        feed(pos, sizeof pos);
        feed(e->image->digest.data(), e->image->digest.size());
      }
      const auto full = running.Finish();
      context->cache_identity.assign(full.begin(), full.end());
      context->images = std::move(images);
    }
    return g::TextPreparedPrompt{std::move(*tokens), std::move(context),
                                 prefix};
  }
  /// Counts without GPU work: the images are decoded for their row counts,
  /// never encoded. (The base class counts through its own PreparePrompt, which
  /// refuses images.)
  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const g::ChatRequest& request) const override {
    const auto prompt = PreparePrompt(request);
    if (!prompt)
      return std::nullopt;
    return prompt->tokens.size();
  }
  void SetPromptContext(
      g::TextRunnerState& state,
      std::shared_ptr<const g::TextPromptContext> context) const override {
    const auto* c = dynamic_cast<const Inkling2PromptContext*>(context.get());
    auto& st = Require(state);
    st.prompt =
        c != nullptr && !c->images.empty()
            ? std::static_pointer_cast<const Inkling2PromptContext>(context)
            : nullptr;
    st.budget_closed = false;
    st.thought_closed = false;
    st.call_required = c != nullptr && c->call_required;
    st.call_names = c != nullptr ? c->call_names : decltype(st.call_names){};
    st.call_prefix.clear();
    st.pending.reset();
    st.backoff = {};
    st.copy_backoff = {};
    st.copy_run = false;
    st.copy_rows = 0;
    st.copy_streak = 0;
    st.history_ok = c != nullptr && !c->tokens.empty();
    if (st.history_ok)
      st.history = c->tokens;
    else
      st.history.clear();
    st.lookup.Clear();
    st.lookup.BeginRequest();
  }
  [[nodiscard]] g::InitialOutputState InitialOutputState(
      const g::ChatRequest&) const override {
    // The prompt ends with the assistant header and the model chooses the
    // block: a thought, an answer or a call. No effort level turns thinking
    // off (level 0 is the minimum, per the model card), so reasoning "off"
    // only lowers the effort and the model may still think.
    return g::InitialOutputState::kAuto;
  }
  [[nodiscard]] std::string Decode(
      std::span<const g::TextRunnerToken> tokens) const override {
    return tok_->Decode(std::span<const ::inkling2::text::TokenId>(
        tokens.data(), tokens.size()));
  }

  // ---- state ----
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t) const override {
    return MakeResources(ledger, std::nullopt);
  }
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources>
  CreateComponentCacheResourcesForSessions(
      cache::ResourceLedger& ledger, std::size_t,
      std::size_t sessions) const override {
    // Several sessions prefill in each other's yields.
    if (sessions > 1)
      engine_->ReserveYieldSpill();
    return MakeResources(ledger, gufo::hip::PlanComponentTransfers(sessions));
  }
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources> MakeResources(
      cache::ResourceLedger& ledger,
      std::optional<gufo::hip::ComponentTransferPlan> transfers) const {
    gufo::hip::ComponentCacheHooks hooks;
    hooks.host_ceiling = [] { return g::HostSnapshotCeilingBytes(); };
    hooks.warn = [](std::string_view m) {
      std::fprintf(stderr, "[inkling2 cache] %.*s\n", int(m.size()), m.data());
    };
    hooks.info = [](std::string_view m) {
      std::fprintf(stderr, "[inkling2 cache] %.*s\n", int(m.size()), m.data());
    };
    if (transfers) {
      hooks.transfer_streams = transfers->native_streams;
      hooks.transfer_leases = transfers->leases;
    }
    return gufo::hip::CreateHipComponentCacheResources(
        std::make_unique<Inkling2CacheAdapter>(engine_, identity_, durable_,
                                               engine_->ubatch()),
        ledger, std::move(hooks));
  }
  [[nodiscard]] std::unique_ptr<g::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<Inkling2State>(static_cast<Inkling2Slot&>(slot),
                                           adapter);
  }
  void ReconcileComponentState(g::TextRunnerState& state,
                               std::size_t position) const override {
    // The adapter's Validate set the frontier; the runner only checks it agrees
    // with the cache.
    if (Require(state).session().position() != position)
      throw std::logic_error(
          "Inkling 2 Mini restored position does not match the cache frontier");
  }
  [[nodiscard]] bool HasDecodeFrontier(
      const g::TextRunnerState& state) const override {
    return Require(state).session().has_logits();
  }

  // ---- execution ----
  [[nodiscard]] g::TextPrefillStep Prefill(
      g::TextRunnerState& state, std::span<const g::TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& st = Require(state);
    if (offset != st.session().position())
      throw std::logic_error(
          "Inkling 2 Mini prefill offset does not match the session position");
    if (offset >= prompt.size())
      throw std::logic_error("Inkling 2 Mini prefill has no input");
    const std::size_t consumed = std::min<std::size_t>(
        {prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1),
         kMaxStepTokens});
    const auto ids = ToIds(prompt.subspan(offset, consumed));
    // Layered prefill: the host asks for `groups` groups of visits with a
    // decode round between.
    // A cancelled request stops the pass at the next visit (tens of
    // milliseconds) instead of finishing it.
    PrefillYield yield{st.prefill_yield_groups(), [&st] { st.YieldPrefill(); },
                       st.cancelled};
    // Image rows inside this chunk replace the placeholders' embeddings
    // (encoded on first use).
    std::vector<InputRows> rows;
    if (st.prompt) {
      for (const auto& e : st.prompt->images) {
        const std::size_t lo = std::max(e->first, offset),
                          hi = std::min(e->first + e->count, offset + consumed);
        if (lo >= hi)
          continue;
        EnsureRows(*e);
        rows.push_back({lo - offset, hi - lo,
                        e->rows->device + (lo - e->first) * kVisionDim});
      }
    }
    engine_->Forward(st.session(), ids, {},
                     yield.groups >= 2 || yield.cancelled ? &yield : nullptr,
                     rows);
    // The prompt may end inside the assistant's block (reasoning off: header +
    // text token).
    st.block = Block::kIdle;
    const auto done = offset + consumed;
    if (done == prompt.size() && max_draft_ > 0) {
      st.history.assign(prompt.begin(), prompt.end());
      st.history_ok = true;
    }
    if (done == prompt.size()) {
      if (done >= 1 && prompt[done - 1] == t_header_)
        st.block = Block::kAfterHeader;
      if (done >= 2 && prompt[done - 2] == t_header_ &&
          prompt[done - 1] == t_text_)
        st.block = Block::kText;
    }
    return {.consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size()};
  }

  [[nodiscard]] Block BlockAt(std::span<const g::TextRunnerToken> prompt,
                              std::size_t pos) const {
    if (pos >= 2 && prompt[pos - 2] == t_header_ && prompt[pos - 1] == t_text_)
      return Block::kText;
    if (pos >= 1 && prompt[pos - 1] == t_header_)
      return Block::kAfterHeader;
    return Block::kIdle;
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const g::TextRunnerState& state,
      std::span<const g::TextRunnerToken> prompt, std::size_t offset,
      std::size_t max_input_tokens, std::size_t boundary) const override {
    const auto& st = Require(state);
    if (offset != st.session().position() || offset >= prompt.size())
      return std::nullopt;
    const std::size_t consumed = std::min<std::size_t>(
        {prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1),
         kMaxStepTokens});
    if (!engine_->CanCaptureInPass(offset, consumed, boundary))
      return std::nullopt;
    return engine_->PrefillCheckpointScratchBytes();
  }

  [[nodiscard]] g::TextPrefillStep PrefillThrough(
      g::TextRunnerState& state, std::span<const g::TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens,
      std::span<const std::size_t> boundaries,
      const InPassConsumer& consume) const override {
    auto& st = Require(state);
    if (!consume || boundaries.empty() ||
        boundaries.size() > Engine::kMaxPassCheckpoints ||
        offset != st.session().position() || offset >= prompt.size())
      throw std::logic_error(
          "invalid Inkling 2 Mini in-pass checkpoint request");
    const std::size_t consumed = std::min<std::size_t>(
        {prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1),
         kMaxStepTokens});
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
      if ((i > 0 && boundaries[i] <= boundaries[i - 1]) ||
          !engine_->CanCaptureInPass(offset, consumed, boundaries[i]))
        throw std::logic_error(
            "invalid Inkling 2 Mini in-pass checkpoint boundary");
    }
    const auto ids = ToIds(prompt.subspan(offset, consumed));
    PrefillYield yield{st.prefill_yield_groups(), [&st] { st.YieldPrefill(); },
                       st.cancelled};
    std::vector<InputRows> rows;
    if (st.prompt) {
      for (const auto& e : st.prompt->images) {
        const std::size_t lo = std::max(e->first, offset),
                          hi = std::min(e->first + e->count, offset + consumed);
        if (lo >= hi)
          continue;
        EnsureRows(*e);
        rows.push_back({lo - offset, hi - lo,
                        e->rows->device + (lo - e->first) * kVisionDim});
      }
    }
    std::vector<Block> blocks;
    blocks.reserve(boundaries.size());
    for (const std::size_t b : boundaries)
      blocks.push_back(BlockAt(prompt, b));
    double capture_ms = 0.0;
    st.adapter().PrefillThrough(
        st.slot(), ids, boundaries, blocks,
        [&](std::size_t index) {
          const auto t0 = std::chrono::steady_clock::now();
          consume(index);
          capture_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        },
        yield.groups >= 2 || yield.cancelled ? &yield : nullptr, rows);
    st.block = Block::kIdle;
    const auto done = offset + consumed;
    if (done == prompt.size() && max_draft_ > 0) {
      st.history.assign(prompt.begin(), prompt.end());
      st.history_ok = true;
    }
    if (done == prompt.size())
      st.block = BlockAt(prompt, done);
    return {.consumed_tokens = consumed,
            .decode_ready = done == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  [[nodiscard]] g::TextDecodeSelection SelectNext(
      g::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& st = Require(state);
    if (st.pending) {  // a DSpark cycle drew it already
      auto sel = std::move(*st.pending);
      st.pending.reset();
      return sel;
    }
    const auto& raw = st.session().logits();
    if (raw.empty() && st.session().position() < max_context_)
      throw std::runtime_error("Inkling 2 Mini token selection has no logits");
    return Select(st, sampler, raw, st.session().position());
  }
  /// Whether Inkling's runner-level masks (`budget_closed`, `call_required`)
  /// allow `tok` in the current block state.
  [[nodiscard]] bool RunnerAllowsToken(const Inkling2State& st,
                                       g::TextRunnerToken tok) const {
    if (st.budget_closed && st.block == Block::kAfterHeader &&
        tok == t_thinking_)
      return false;
    if (st.call_required &&
        (st.block == Block::kAfterHeader || st.block == Block::kTool)) {
      const auto& prefix = st.call_prefix;
      const std::size_t V = engine_->config().n_vocab_text;
      bool any_allowed = false;
      auto check = [&](g::TextRunnerToken t) {
        if (t < V) {
          any_allowed = true;
          if (t == tok)
            return true;
        }
        return false;
      };
      if (prefix.empty() && check(t_invoke_))
        return true;
      if (prefix.empty() && st.block == Block::kAfterHeader &&
          !st.budget_closed && !st.thought_closed && check(t_thinking_))
        return true;
      for (const auto& name : st.call_names) {
        if (name.size() < prefix.size() ||
            !std::equal(prefix.begin(), prefix.end(), name.begin()))
          continue;
        if (check(name.size() == prefix.size() ? t_invoke_
                                               : name[prefix.size()]))
          return true;
      }
      return !any_allowed;
    }
    return true;
  }

  /// Applies Inkling's runner-level token masks (`budget_closed`,
  /// `call_required`) to `raw`; returns `raw` unchanged when no mask applies.
  [[nodiscard]] std::span<const float> ApplyRunnerMasks(
      const Inkling2State& st, std::span<const float> raw,
      std::vector<float>& masked) const {
    masked.clear();
    if (st.budget_closed && st.block == Block::kAfterHeader) {
      masked.assign(raw.begin(), raw.end());
      if (t_thinking_ < masked.size())
        masked[t_thinking_] = -std::numeric_limits<float>::infinity();
    }
    if (st.call_required &&
        (st.block == Block::kAfterHeader || st.block == Block::kTool)) {
      std::vector<g::TextRunnerToken> allowed;
      const auto& prefix = st.call_prefix;
      if (prefix.empty() && t_invoke_ < raw.size())
        allowed.push_back(t_invoke_);  // a call without the first name
      if (prefix.empty() && st.block == Block::kAfterHeader &&
          !st.budget_closed && !st.thought_closed && t_thinking_ < raw.size())
        allowed.push_back(t_thinking_);  // a thought before the call
      for (const auto& name : st.call_names) {
        if (name.size() < prefix.size() ||
            !std::equal(prefix.begin(), prefix.end(), name.begin()))
          continue;
        allowed.push_back(name.size() == prefix.size() ? t_invoke_
                                                       : name[prefix.size()]);
      }
      if (!allowed.empty()) {
        masked.assign(raw.size(), -std::numeric_limits<float>::infinity());
        for (const auto t : allowed)
          if (t < raw.size())
            masked[t] = raw[t];
      }
    }
    return masked.empty() ? raw : std::span<const float>(masked);
  }
  /// SelectNext's choice from the logits `raw` of the token before
  /// `position`, for the block state the committed tokens left.
  [[nodiscard]] g::TextDecodeSelection Select(Inkling2State& st,
                                              sampling::SamplerState& sampler,
                                              std::span<const float> raw,
                                              std::size_t position) const {
    if (position >= max_context_)
      return {.stop = true, .piece = {}};
    if (sampler.ForcedToken() && st.block == Block::kThinking)
      st.budget_closed = true;
    std::vector<float> masked;
    const std::span<const float> logits = ApplyRunnerMasks(st, raw, masked);
    const auto token = static_cast<g::TextRunnerToken>(sampler.Sample(logits));
    const bool stop = st.stop_at_eos() && tok_->IsStop(token);
    return {.stop = stop,
            .token = token,
            .piece = stop ? std::string{} : Piece(st, token),
            .logprobs = sampling::CollectLogprobs(
                raw, token,
                sampler.config().logprobs)};  // the model's own distribution
  }
  void Advance(g::TextRunnerState& state,
               g::TextRunnerToken token) const override {
    auto& st = Require(state);
    const std::int32_t id = static_cast<std::int32_t>(token);
    engine_->Decode(st.session(), id);
    CommitToken(st, token);
  }
  void CommitToken(Inkling2State& st, g::TextRunnerToken token) const {
    if (st.history_ok)
      st.history.push_back(static_cast<std::int32_t>(token));
    if (st.block == Block::kThinking && token == t_end_message_)
      st.thought_closed = true;
    if (token == t_header_)
      st.call_prefix.clear();
    else if ((st.block == Block::kAfterHeader || st.block == Block::kTool) &&
             token != t_invoke_)
      st.call_prefix.push_back(token);
    st.block = NextBlock(st, token);
  }
  /// One token for each of several sessions in a single weight pass
  /// (Engine::DecodeBatch).
  void AdvanceBatch(
      std::span<const g::TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > Engine::kMaxDecodeBatch) {
      g::TextModelRunner::AdvanceBatch(advances);
      return;
    }
    std::vector<Inkling2State*> states;
    std::vector<Session*> sessions;
    std::vector<std::int32_t> ids;
    for (const auto& a : advances) {
      states.push_back(&Require(a.state.get()));
      sessions.push_back(&states.back()->session());
      ids.push_back(static_cast<std::int32_t>(a.token));
    }
    try {
      engine_->DecodeBatch(sessions, ids);
    } catch (...) {
      // A failed step leaves the sessions unchanged or half-advanced: report it
      // to every lane.
      for (const auto& a : advances) {
        if (a.failure != nullptr)
          *a.failure = std::current_exception();
        else
          throw;
      }
      return;
    }
    for (std::size_t i = 0; i < advances.size(); ++i)
      CommitToken(*states[i], advances[i].token);
  }
  /// Runs f(0 .. n - 1), of which `sampled` make a host sampler pass over
  /// the vocabulary. Up to kSerialRows of those run one after another on this
  /// thread, which is warm and has the sampler's own threads to itself (0.4
  /// ms a row; a selection thread woken for one row takes about 1 ms);
  /// more are spread over the selection threads.
  static constexpr std::size_t kSerialRows = 3;
  /// Rows of an ordinary DSpark verification (the anchor and seven drafts).
  static constexpr int kNarrowRows = Drafter::kMaxDraft + 1;
  template<class F>
  void ForEachLane(std::size_t n, std::size_t sampled, F&& f) const {
    if (sampled <= kSerialRows) {
      for (std::size_t k = 0; k < n; ++k)
        f(k);
      return;
    }
    lane_pool_.Run(n, f);
  }

  /// How a request's verified rows are selected and its drafts proposed.
  /// Plain greedy requests take each row's argmax from the device (the token
  /// Select would pick) and copy only the kept row's logits. Requests that
  /// sample with temperature only get coupled proposals: the drafter draws
  /// each with the uniform the request's sampler will use for that row.
  struct RowMode {
    bool greedy = false, coupled = false, gpu_linear = false;
    Drafter::Coupled cp;
  };
  [[nodiscard]] RowMode ModeOf(const Inkling2State& st,
                               const sampling::SamplerState& working) const {
    RowMode m;
    const auto& cfg = working.config();
    // Device argmax fast path for all greedy requests without penalties or
    // logprobs; constrained / budget_closed / call_required rows lazily read
    // logits only if the argmax is forbidden.
    // (temperature 0 only: CanSelectArgmax refuses a top_k = 1 request with a
    // temperature, which would read every row back on its own)
    m.greedy =
        cfg.temperature == 0.0F && !cfg.penalties_enabled() && !cfg.logprobs;
    m.coupled = !m.greedy && cfg.temperature > 0 && !cfg.constraint &&
                !working.ForcedToken() && cfg.top_k == 0 && cfg.top_p == 1 &&
                !(cfg.min_p > 0 && cfg.min_keep > 1);
    m.gpu_linear = m.coupled && cfg.temperature >= 0.1F &&
                   !cfg.reasoning_budget && !cfg.penalties_enabled() &&
                   cfg.min_p == 0.0F && !cfg.logprobs && !st.budget_closed &&
                   !st.call_required;
    if (m.coupled) {
      m.cp.temperature = cfg.temperature;
      m.cp.min_p = cfg.min_p;
      auto peek = working;  // the draws the rows' selections will make
      for (float& u : m.cp.u)
        u = static_cast<float>(peek.Uniform());
    }
    return m;
  }

  /// Copy drafts (prompt lookup): when the request's last tokens repeat an
  /// earlier stretch of its context (speculative::PromptLookup's rules: 12
  /// matching generated tokens, 24 for sampled requests, 5 where the last
  /// accepted copy left off), up to `limit` of the tokens that followed that
  /// stretch are proposed in place of the drafter's. Called after the anchor
  /// was committed. Returns the number of proposals; `start` is the index of
  /// their source and `conf` their calibration bins.
  int ProposeCopy(Inkling2State& st, bool sampled, std::size_t limit,
                  std::int32_t* drafts, float* conf, std::size_t* start) const {
    using Lookup = speculative::PromptLookup<std::int32_t>;
    const std::size_t pos = st.session().position();
    // Sampled requests only: the drafter's coupled draw misses about a fifth
    // of the tokens of a copied stretch, the copied token almost none. Greedy
    // requests gain nothing (the drafter copies as well) and lose where a
    // match's continuation differs.
    if (!sampled || limit == 0 || !st.history_ok ||
        st.history.size() != pos + 1)
      return 0;
    if (!st.lookup.Step(pos))
      return 0;  // a request's first cycle copies nothing
    st.lookup.Extend(st.history, sampled);
    if (!st.copy_backoff.Copy())
      return 0;
    const std::size_t minimum =
        sampled ? Lookup::kSampledMinMatch : Lookup::kMinMatch;
    const auto m = st.lookup.Find(st.history, {}, minimum);
    if (m.length == 0)
      return 0;
    const std::size_t have =
        std::min(limit, st.lookup.IndexedTokens() - m.start);
    int n = 0;
    for (; static_cast<std::size_t>(n) < have; ++n) {
      const std::int32_t t = st.history[m.start + static_cast<std::size_t>(n)];
      if (tok_->IsStop(static_cast<g::TextRunnerToken>(t)))
        break;  // a stop is never kept as a draft
      drafts[n] = t;
      // an established match, or the shorter one that resumes a copy
      conf[n] = m.length >= minimum ? 0.95F : 0.85F;
    }
    *start = m.start;
    return n;
  }

  /// Rows a wide copy may verify now (the anchor included).
  static int CopyRows(const Inkling2State& st) {
    return std::max(st.copy_rows, kNarrowRows);
  }

  /// Copied proposals that continue the request's context plus `tail` (the
  /// cycle's drafts after the committed anchor), up to `limit`, written to
  /// `drafts`: speculative::PromptLookup's match rules (greedy requests).
  /// Returns their count; `start` is the index of their source. Called
  /// after the anchor was committed and lookup.Step / Extend ran.
  int CopyAfter(Inkling2State& st, std::span<const std::int32_t> tail,
                std::size_t limit, std::int32_t* drafts,
                std::size_t* start) const {
    using Lookup = speculative::PromptLookup<std::int32_t>;
    if (limit == 0 || !st.copy_backoff.Copy())
      return 0;
    const auto m = st.lookup.Find(st.history, tail, Lookup::kMinMatch);
    if (m.length == 0)
      return 0;
    const std::size_t have =
        std::min(limit, st.lookup.IndexedTokens() - m.start);
    int n = 0;
    for (; static_cast<std::size_t>(n) < have; ++n) {
      const std::int32_t t = st.history[m.start + static_cast<std::size_t>(n)];
      if (tok_->IsStop(static_cast<g::TextRunnerToken>(t)))
        break;
      drafts[n] = t;
    }
    *start = m.start;
    return n;
  }

  /// The anchor of a cycle: the pending token, or a selection from the
  /// session's logits. A non-stop anchor is accepted and committed.
  [[nodiscard]] g::TextDecodeSelection TakeAnchor(
      Inkling2State& st, sampling::SamplerState& working) const {
    Session& ss = st.session();
    g::TextDecodeSelection anchor;
    if (st.pending) {
      anchor = std::move(*st.pending);
      st.pending.reset();
    } else {
      if (ss.logits().empty() && ss.position() < max_context_)
        throw std::runtime_error(
            "Inkling 2 Mini token selection has no logits");
      anchor = Select(st, working, ss.logits(), ss.position());
    }
    if (!anchor.stop) {
      working.Accept(anchor.token);
      CommitToken(st, anchor.token);
    }
    return anchor;
  }

  /// Selects a session's verified rows (the anchor's row and `width` draft
  /// rows from position `pos`; `out` its rows' logits when not greedy, `amax`
  /// its rows' device argmax when greedy). Each row's token is selected
  /// exactly as SelectNext would select it after the tokens before it (same
  /// masks, the request's sampler on `working`), and a proposal is kept while
  /// it equals that selection; the first selection that differs, or follows
  /// the last kept row, becomes the next anchor (`st.pending`). Returns the
  /// rows kept (at least 1).
  std::size_t Resolve(Inkling2State& st, sampling::SamplerState& working,
                      g::TextDecodeStep& step, std::size_t& budget,
                      std::size_t pos, int width, const std::int32_t* drafts,
                      std::span<const float> out, const std::int32_t* amax,
                      bool greedy, std::size_t row0 = 0,
                      const std::int32_t* gpu_sampled = nullptr,
                      const std::uint64_t* gpu_draws = nullptr) const {
    const auto& cfg = working.config();
    const std::size_t V = engine_->config().n_vocab_text;
    std::size_t keep = 1;
    // Sampling with temperature only (no min-p, T >= 0.1) draws exactly one
    // uniform per token: every capped weight is positive, so there are always
    // several entries. The rows' draws are then known in advance and the
    // host sampler runs on all rows at once; tokens, draws and the published
    // RNG state are those of the rows one after another.
    const bool parallel =
        !greedy && gpu_sampled == nullptr && width > 0 &&
        cfg.temperature >= 0.1F && !cfg.constraint && !working.ForcedToken() &&
        !cfg.reasoning_budget && !cfg.penalties_enabled() && cfg.top_k == 0 &&
        cfg.top_p == 1 && cfg.min_p == 0 && !st.budget_closed &&
        !st.call_required &&
        pos + static_cast<std::size_t>(width) + 1 < max_context_;
    std::array<std::uint64_t, Engine::kMaxVerifyRows + 1> draws{};
    std::array<g::TextRunnerToken, Engine::kMaxVerifyRows> sampled{};
    if (parallel) {
      auto ahead = working;
      for (int j = 0; j <= width; ++j) {
        draws[j] = ahead.rng_state();
        (void)ahead.Uniform();
      }
      draws[width + 1] = ahead.rng_state();
      const auto sample_row = [&](int j) {
        sampling::SamplerState s = working;
        s.SetRngState(draws[j]);
        return static_cast<g::TextRunnerToken>(
            s.Sample(out.subspan(static_cast<std::size_t>(j) * V, V)));
      };
      // (rows of a request that is itself one of ForEachLane's lanes run
      // here, one after another: the pool takes one Run at a time)
      ForEachLane(static_cast<std::size_t>(width) + 1,
                  static_cast<std::size_t>(width) + 1, [&](std::size_t j) {
                    sampled[j] = sample_row(static_cast<int>(j));
                  });
    }
    int used = 0;  // rows whose selection drew (parallel / gpu_sampled)
    for (int j = 0; j <= width; ++j) {
      const std::size_t at = pos + static_cast<std::size_t>(j) + 1;
      g::TextDecodeSelection sel;
      if (gpu_sampled != nullptr && at < max_context_) {
        const auto tk = static_cast<g::TextRunnerToken>(gpu_sampled[j]);
        const bool stop = st.stop_at_eos() && tok_->IsStop(tk);
        sel = {.stop = stop,
               .token = tk,
               .piece = stop ? std::string{} : Piece(st, tk)};
        used = j + 1;
      } else if (parallel) {
        const auto tk = sampled[j];
        const auto row = out.subspan(static_cast<std::size_t>(j) * V, V);
        const bool stop = st.stop_at_eos() && tok_->IsStop(tk);
        sel = {.stop = stop,
               .token = tk,
               .piece = stop ? std::string{} : Piece(st, tk),
               .logprobs = sampling::CollectLogprobs(row, tk, cfg.logprobs)};
        used = j + 1;
      } else if (greedy) {
        const auto forced = working.ForcedToken();
        if (forced) {
          if (st.block == Block::kThinking)
            st.budget_closed = true;
          const auto tk = static_cast<g::TextRunnerToken>(*forced);
          const bool stop =
              at >= max_context_ || (st.stop_at_eos() && tok_->IsStop(tk));
          sel = {.stop = stop,
                 .token = tk,
                 .piece = stop ? std::string{} : Piece(st, tk)};
        } else if (at >= max_context_) {
          sel = {.stop = true, .piece = {}};
        } else if (working.CanSelectArgmax(
                       static_cast<sampling::TokenId>(amax[j]), false) &&
                   RunnerAllowsToken(
                       st, static_cast<g::TextRunnerToken>(amax[j]))) {
          const auto tk = static_cast<g::TextRunnerToken>(amax[j]);
          const bool stop = st.stop_at_eos() && tok_->IsStop(tk);
          sel = {.stop = stop,
                 .token = tk,
                 .piece = stop ? std::string{} : Piece(st, tk)};
        } else {
          std::span<const float> row;
          engine_->ReadVerifyRows(row0 + static_cast<std::size_t>(j), 1, &row);
          sel = Select(st, working, row, at);
        }
      } else {
        sel = Select(st, working, out.subspan(j * V, V), at);
      }
      if (sel.stop) {
        step.stop = true;
        step.terminal_logprobs = std::move(sel.logprobs);
        break;
      }
      if (j < width && budget > 0 &&
          sel.token == static_cast<g::TextRunnerToken>(drafts[j])) {
        working.Accept(sel.token);
        CommitToken(st, sel.token);
        step.selections.push_back(std::move(sel));
        --budget;
        ++keep;
        continue;
      }
      st.pending = std::move(sel);
      break;
    }
    if (gpu_sampled != nullptr && gpu_draws != nullptr)
      working.SetRngState(gpu_draws[used]);
    else if (parallel)
      working.SetRngState(draws[used]);
    return keep;
  }

  /// DSpark cycle: the anchor (the pending token, or a selection from the
  /// session's logits) is emitted, the drafter proposes up to seven tokens
  /// after it, and one Engine::Verify step evaluates the anchor and the first
  /// W proposals (W from ChooseDsparkWidth); Resolve keeps the proposals the
  /// request's own selections confirm. So every emitted token and draw is the
  /// one plain decoding gives.
  [[nodiscard]] g::TextDecodeStep DecodeStep(
      g::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!engine_->batch_capable())
      return g::TextModelRunner::DecodeStep(state, max_tokens, sampler);
    g::TextDecodeStep step;
    if (max_tokens == 0)
      return step;
    auto& st = Require(state);
    Session& ss = st.session();
    sampling::SamplerState working(sampler);
    g::TextDecodeSelection anchor = TakeAnchor(st, working);
    if (anchor.stop) {
      step.stop = true;
      step.terminal_logprobs = std::move(anchor.logprobs);
      sampler.CopyDrawStateFrom(working);
      return step;
    }
    const auto a = anchor.token;
    step.selections.push_back(std::move(anchor));
    std::size_t budget = max_tokens - 1;

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    std::int32_t drafts[Engine::kMaxVerifyRows] = {};
    float conf[Engine::kMaxVerifyRows] = {};
    int n = 0;
    const RowMode mode = ModeOf(st, working);
    const bool greedy = mode.greedy;
    const bool sampled = working.config().temperature > 0;
    const std::size_t rows_room = engine_->VerifyRows(ss) - 1;
    // Copies of earlier context (prompt lookup). While a copy keeps being
    // accepted, the next cycle copies first, up to CopyRows() rows in one
    // wide verification (this request decodes alone). Otherwise sampled
    // requests copy first within the drafter's width, and greedy ones append
    // a copy to the drafter's proposals when the context they extend
    // continues an earlier stretch.
    std::size_t copy_start = 0;
    int copy_width = 0;   // copied proposals verified
    int draft_width = 0;  // the drafter's proposals before them
    bool copy_only = false;
    bool copy_conf = false;  // conf[] holds the copied proposals' bins
    bool lookup_ready = false;
    if (budget > 0 && max_draft_ > 0 && st.history_ok &&
        st.history.size() == ss.position() + 1 &&
        st.lookup.Step(ss.position())) {
      st.lookup.Extend(st.history, sampled);
      lookup_ready = true;
    }
    if (lookup_ready && st.copy_run) {
      const int c = CopyAfter(
          st, {},
          std::min<std::size_t>(
              {static_cast<std::size_t>(CopyRows(st) - 1), budget, rows_room}),
          drafts, &copy_start);
      if (c > 0) {
        copy_width = c;
        copy_only = true;
      }
    }
    if (copy_width == 0 && lookup_ready && sampled) {
      const int c = ProposeCopy(
          st, sampled,
          std::min<std::size_t>(
              {static_cast<std::size_t>(max_draft_), budget, rows_room}),
          drafts, conf, &copy_start);
      if (c > 0) {
        const std::lock_guard lock(dsp_mutex_);
        copy_width = ChooseDsparkWidth(conf, c, cal_copy_sampled_, costs_, 0.0);
        copy_only = copy_width > 0;
        copy_conf = copy_only;
      }
    }
    const bool copied = copy_width > 0;
    DsparkCalibration& cal = copy_only && sampled ? cal_copy_sampled_
                             : mode.coupled       ? cal_coupled_
                                                  : cal_;
    if (!copied && budget > 0 && engine_->DraftReady(ss) &&
        st.backoff.Draft()) {
      n = static_cast<int>(std::min<std::size_t>(
          {static_cast<std::size_t>(max_draft_), budget, rows_room}));
      if (n > 0)  // stop once the width policy would not verify another
        n = engine_->Propose(
            ss, static_cast<std::int32_t>(a), n, drafts, conf,
            [&](int i, const float* p) {
              const std::lock_guard lock(dsp_mutex_);
              return ChooseDsparkWidth(p, i, cal, costs_) == i;
            },
            mode.coupled ? &mode.cp : nullptr);
    }
    const auto t1 = Clock::now();
    int width = 0;
    {
      const std::lock_guard lock(dsp_mutex_);
      if (n > 0 && !copied)
        costs_.ObserveDraft(
            n, std::chrono::duration<double, std::milli>(t1 - t0).count());
      width = copied  ? copy_width
              : n > 0 ? ChooseDsparkWidth(conf, n, cal, costs_)
                      : 0;
    }
    if (!copied && lookup_ready && greedy) {
      // the drafter's proposals, then the copy their context continues
      const std::size_t cap = std::min<std::size_t>(
          {static_cast<std::size_t>(CopyRows(st) - 1), budget, rows_room});
      if (cap > static_cast<std::size_t>(width)) {
        // (the anchor is already in the history)
        std::int32_t tail[Engine::kMaxVerifyRows];
        std::copy(drafts, drafts + width, tail);
        const int c = CopyAfter(st, std::span<const std::int32_t>(tail, width),
                                cap - static_cast<std::size_t>(width),
                                drafts + width, &copy_start);
        if (c > 0) {
          draft_width = width;
          copy_width = c;
          width += c;
        }
      }
    }
    std::vector<std::int32_t> rows{static_cast<std::int32_t>(a)};
    rows.insert(rows.end(), drafts, drafts + width);
    std::span<const float> out;
    std::int32_t amax[Engine::kMaxVerifyRows] = {};
    std::int32_t gpu_samp[Engine::kMaxVerifyRows] = {};
    std::array<std::uint64_t, Engine::kMaxVerifyRows + 1> gpu_draws{};
    const std::size_t pos = ss.position();
    const bool use_gpu_linear =
        mode.gpu_linear &&
        pos + static_cast<std::size_t>(width) + 1 < max_context_;
    std::array<Engine::LinearSample, Engine::kMaxVerifyRows> ls{};
    if (use_gpu_linear) {
      auto ahead = working;
      const auto& scfg = working.config();
      for (int j = 0; j <= width; ++j) {
        gpu_draws[j] = ahead.rng_state();
        ls[j] = {.temperature = scfg.temperature,
                 .min_p = scfg.min_p,
                 .u = ahead.Uniform()};
      }
      gpu_draws[width + 1] = ahead.rng_state();
    }
    if (width == 0) {
      // Nothing to verify: a plain decode step (logits left on the device
      // when greedy or GPU-sampled).
      if (greedy) {
        engine_->Step(ss, static_cast<std::int32_t>(a), amax);
      } else if (use_gpu_linear) {
        const std::int32_t id = static_cast<std::int32_t>(a);
        Session* const sp = &ss;
        const bool host = false;
        engine_->DecodeBatchRows(
            std::span<Session* const>(&sp, 1),
            std::span<const std::int32_t>(&id, 1),
            std::span<const bool>(&host, 1), nullptr, nullptr,
            std::span<const Engine::LinearSample>(ls.data(), 1), gpu_samp);
      } else {
        // the row's logits through the pinned buffer (the session keeps its
        // copy on the device)
        const std::int32_t id = static_cast<std::int32_t>(a);
        Session* const sp = &ss;
        const bool host = true;
        engine_->DecodeBatchRows(std::span<Session* const>(&sp, 1),
                                 std::span<const std::int32_t>(&id, 1),
                                 std::span<const bool>(&host, 1), &out,
                                 nullptr);
      }
    } else {
      engine_->Verify(ss, rows, (greedy || use_gpu_linear) ? nullptr : &out,
                      greedy ? amax : nullptr,
                      use_gpu_linear ? std::span<const Engine::LinearSample>(
                                           ls.data(), rows.size())
                                     : std::span<const Engine::LinearSample>{},
                      use_gpu_linear ? gpu_samp : nullptr);
    }
    const auto t2 = Clock::now();
    const std::size_t keep =
        Resolve(st, working, step, budget, pos, width, drafts, out, amax,
                greedy, 0, use_gpu_linear ? gpu_samp : nullptr,
                use_gpu_linear ? gpu_draws.data() : nullptr);
    if (width > 0)
      engine_->CommitVerify(ss, keep, {});  // lazy: keep logits on device
    sampler.CopyDrawStateFrom(working);
    const auto t3 = Clock::now();
    const int kept = static_cast<int>(keep) - 1;  // proposals kept
    const int drafted = copied ? 0 : width - copy_width;
    {
      const std::lock_guard lock(dsp_mutex_);
      if (rows.size() <= static_cast<std::size_t>(DsparkCosts::kMaxRows))
        costs_.ObserveVerify(
            static_cast<int>(rows.size()),
            std::chrono::duration<double, std::milli>(t2 - t1).count());
      const int binned = copy_only ? (copy_conf ? width : 0) : drafted;
      for (int j = 0; j < binned && j <= kept; ++j)
        cal.Observe(conf[j], j + 1 <= kept);
      const double cycle =
          std::chrono::duration<double, std::milli>(t3 - t0).count();
      // what a plain step costs requests of this kind (host sampling included)
      double& plain = greedy ? plain_greedy_ms_ : plain_sampled_ms_;
      if (n == 0 && !copied)
        plain += 0.1 * (cycle - plain);
      else if (!copied && copy_width == 0)
        st.backoff.Observe(static_cast<int>(keep), cycle, plain);
    }
    if (width > 0) {
      step.draft_rounds = 1;
      step.draft_tokens = static_cast<std::size_t>(width);
      step.draft_accepted_tokens = keep - 1;
    }
    if (copy_width > 0) {
      const int copy_kept = std::max(0, kept - draft_width);
      step.lookup_tokens = static_cast<std::size_t>(copy_width);
      step.lookup_accepted_tokens = static_cast<std::size_t>(copy_kept);
      st.lookup.Observe(copy_start, static_cast<std::size_t>(copy_width),
                        static_cast<std::size_t>(copy_kept));
      st.copy_backoff.Observe(copy_width, copy_kept);
      const bool whole = kept == width;
      st.copy_run = whole && !step.stop;
      if (!whole) {
        st.copy_rows = kNarrowRows;
        st.copy_streak = 0;
      } else if (copy_only && width + 1 >= CopyRows(st) &&
                 ++st.copy_streak >= 2) {
        st.copy_rows = std::min(2 * CopyRows(st),
                                static_cast<int>(Engine::kMaxVerifyRows));
        st.copy_streak = 0;
      }
    } else {
      st.copy_run = false;
    }
    return step;
  }

  /// Several requests: their anchors (the pending token, or a selection from
  /// the session's logits) are emitted, each drafts after its anchor, and one
  /// Engine::VerifyBatch step evaluates every anchor and the proposals the
  /// joint width policy keeps (ChooseDsparkBatchWidths: rows go to the
  /// drafts most likely to survive while they pay for their share of the
  /// step); Resolve then keeps, per request, the proposals its own
  /// selections confirm. Without proposals the step is a batched one-token
  /// decode. Every row has the arithmetic of the request's lone one-token
  /// decode, so each request's tokens and draws are those of plain decoding.
  [[nodiscard]] std::vector<g::TextDecodeStep> DecodeBatch(
      std::span<const g::TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || decodes.size() > Engine::kMaxDecodeBatch ||
        !engine_->batch_capable())
      return g::TextModelRunner::DecodeBatch(decodes);
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    std::vector<g::TextDecodeStep> steps(decodes.size());
    std::vector<sampling::SamplerState> working;
    working.reserve(decodes.size());
    struct Lane {
      std::size_t i;  // index into decodes
      Inkling2State* st;
      Session* ss;
      std::int32_t anchor;
      std::size_t budget;
      std::size_t pos = 0;
      RowMode mode;
      int n = 0, width = 0;
      std::int32_t drafts[Drafter::kMaxDraft] = {};
      float conf[Drafter::kMaxDraft] = {};
      bool copied = false, sampled = false;  // proposals copied (prompt lookup)
      std::size_t copy_start = 0;
    };
    std::vector<Lane> lanes;
    const auto cal_of = [this](const Lane& l) -> DsparkCalibration* {
      return l.copied         ? (l.sampled ? &cal_copy_sampled_ : &cal_copy_)
             : l.mode.coupled ? &cal_coupled_
                              : &cal_;
    };
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      working.emplace_back(decodes[i].sampler.get());
      auto& step = steps[i];
      step.execution_plan = {.kind = g::TextExecutionPlanKind::kBatched,
                             .physical_width = decodes.size()};
      try {
        auto& st = Require(decodes[i].state.get());
        g::TextDecodeSelection anchor = TakeAnchor(st, working[i]);
        if (anchor.stop || decodes[i].max_tokens == 0) {
          step.stop = anchor.stop;
          step.terminal_logprobs = std::move(anchor.logprobs);
          continue;
        }
        Lane l{.i = i,
               .st = &st,
               .ss = &st.session(),
               .anchor = static_cast<std::int32_t>(anchor.token),
               .budget = decodes[i].max_tokens - 1};
        step.selections.push_back(std::move(anchor));
        lanes.push_back(l);
      } catch (...) {
        step.failure = std::current_exception();
      }
    }
    const int B = static_cast<int>(lanes.size());
    // Drafting: each lane proposes after its anchor, stopping once a further
    // row would cost more of the step than the tokens it is expected to add.
    bool drafted = false, neural = false;
    double draft_ms = 0;
    if (B >= 2 && engine_->has_drafter()) {
      bool draft = false;
      {
        const std::lock_guard lock(dsp_mutex_);
        draft = batch_backoff_[B].Draft();
      }
      std::vector<Engine::ProposeRequest> reqs;
      std::vector<Lane*> owners;
      for (auto& l : lanes) {
        try {
          l.mode = ModeOf(*l.st, working[l.i]);
          l.sampled = working[l.i].config().temperature > 0;
          if (l.budget == 0)
            continue;
          const int room = static_cast<int>(Engine::kMaxBatchVerifyRows) - B;
          const int limit = static_cast<int>(std::min<std::size_t>(
              {max_draft_, l.budget, engine_->VerifyRows(*l.ss) - 1,
               static_cast<std::size_t>(std::max(room, 0))}));
          if (limit <= 0)
            continue;
          // copied proposals first (they cost nothing)
          l.n = ProposeCopy(*l.st, l.sampled, static_cast<std::size_t>(limit),
                            l.drafts, l.conf, &l.copy_start);
          if (l.n > 0) {
            l.copied = true;
            drafted = true;
            continue;
          }
          if (!draft || !engine_->DraftReady(*l.ss))
            continue;
          l.n = limit;
          const DsparkCalibration* cal = l.mode.coupled ? &cal_coupled_ : &cal_;
          reqs.push_back({.session = l.ss,
                          .anchor = l.anchor,
                          .n = l.n,
                          .out = l.drafts,
                          .confidence = l.conf,
                          // keep drafting while the first i drafts' survival
                          // pays for the extra rows' share of a plain step
                          .more =
                              [this, cal, B](int i, const float* p) {
                                const std::lock_guard lock(dsp_mutex_);
                                double sv = 1;
                                for (int k = 0; k < i; ++k)
                                  sv *= cal->Estimate(p[k]);
                                const double per_token =
                                    costs_batch_.Verify(B) / B;
                                const double extra =
                                    costs_batch_.Verify(std::min(
                                        B + i, DsparkCosts::kMaxRows)) -
                                    costs_batch_.Verify(std::min(
                                        B + i - 1, DsparkCosts::kMaxRows));
                                return sv * per_token > extra;
                              },
                          .coupled = l.mode.coupled ? &l.mode.cp : nullptr});
          owners.push_back(&l);
        } catch (...) {
          steps[l.i].failure = std::current_exception();
          l.n = -1;  // dropped below
        }
      }
      if (!reqs.empty()) {
        const auto ts = Clock::now();
        try {
          engine_->ProposeBatch(reqs);
          for (std::size_t k = 0; k < reqs.size(); ++k) {
            owners[k]->n = reqs[k].done;
            drafted = drafted || reqs[k].done > 0;
            neural = neural || reqs[k].done > 0;
          }
        } catch (...) {
          for (auto* o : owners)
            o->n = 0;  // decode without drafts
        }
        draft_ms = std::chrono::duration<double, std::milli>(Clock::now() - ts)
                       .count();
      }
      std::erase_if(lanes, [](const Lane& l) { return l.n < 0; });
    }
    const int nl = static_cast<int>(lanes.size());
    if (drafted && nl >= 1) {
      const std::lock_guard lock(dsp_mutex_);
      const float* p[Engine::kMaxDecodeBatch];
      int n[Engine::kMaxDecodeBatch], w[Engine::kMaxDecodeBatch];
      const DsparkCalibration* c[Engine::kMaxDecodeBatch];
      for (int k = 0; k < nl; ++k) {
        p[k] = lanes[k].conf;
        n[k] = lanes[k].n;
        c[k] = cal_of(lanes[k]);
      }
      ChooseDsparkBatchWidths(nl, p, n, c, costs_batch_, draft_ms, w);
      for (int k = 0; k < nl; ++k)
        lanes[k].width = w[k];
    }
    int extra = 0;
    for (const auto& l : lanes)
      extra += l.width;
    const auto t1 = Clock::now();
    if (extra == 0) {
      // A batched one-token decode of the anchors; each request's next token
      // is selected from its new logits and kept as the next anchor.
      std::vector<Session*> sessions;
      std::vector<std::int32_t> ids;
      for (const auto& l : lanes) {
        sessions.push_back(l.ss);
        ids.push_back(l.anchor);
      }
      bool ok = !sessions.empty();
      // greedy and GPU-linear rows avoid D2H logits; only host-sampled rows
      // come to the host
      std::vector<RowMode> modes(lanes.size());
      bool host[Engine::kMaxDecodeBatch] = {};
      bool gpu_lin[Engine::kMaxDecodeBatch] = {};
      std::int32_t amax[Engine::kMaxDecodeBatch] = {};
      std::int32_t gpu_samp[Engine::kMaxDecodeBatch] = {};
      Engine::LinearSample ls[Engine::kMaxDecodeBatch] = {};
      std::uint64_t next_rng[Engine::kMaxDecodeBatch] = {};
      bool any_greedy = false, any_gpu_lin = false;
      std::span<const float> out;
      const std::size_t V = engine_->config().n_vocab_text;
      if (ok) {
        try {
          for (std::size_t k = 0; k < lanes.size(); ++k) {
            modes[k] = ModeOf(*lanes[k].st, working[lanes[k].i]);
            gpu_lin[k] = modes[k].gpu_linear &&
                         lanes[k].ss->position() + 1 < max_context_;
            host[k] = !modes[k].greedy && !gpu_lin[k];
            any_greedy = any_greedy || modes[k].greedy;
            any_gpu_lin = any_gpu_lin || gpu_lin[k];
            if (gpu_lin[k]) {
              auto ahead = working[lanes[k].i];
              const auto& scfg = ahead.config();
              ls[k] = {.temperature = scfg.temperature,
                       .min_p = scfg.min_p,
                       .u = ahead.Uniform()};
              next_rng[k] = ahead.rng_state();
            }
          }
          engine_->DecodeBatchRows(
              sessions, ids, std::span<const bool>(host, lanes.size()), &out,
              any_greedy ? amax : nullptr,
              any_gpu_lin
                  ? std::span<const Engine::LinearSample>(ls, lanes.size())
                  : std::span<const Engine::LinearSample>{},
              any_gpu_lin ? gpu_samp : nullptr);
        } catch (...) {
          for (const auto& l : lanes)
            steps[l.i].failure = std::current_exception();
          ok = false;
        }
      }
      if (ok) {
        // each request's selection on its own thread (host sampling over the
        // vocabulary; the requests' states are disjoint)
        std::size_t host_rows = 0;
        for (std::size_t k = 0; k < lanes.size(); ++k)
          host_rows += host[k] ? 1 : 0;
        ForEachLane(lanes.size(), host_rows, [&](std::size_t k) {
          const auto& l = lanes[k];
          try {
            const std::size_t at = l.ss->position();
            if (gpu_lin[k]) {
              const auto tk = static_cast<g::TextRunnerToken>(gpu_samp[k]);
              const bool stop = at >= max_context_ ||
                                (l.st->stop_at_eos() && tok_->IsStop(tk));
              l.st->pending = g::TextDecodeSelection{
                  .stop = stop,
                  .token = tk,
                  .piece = stop ? std::string{} : Piece(*l.st, tk)};
              working[l.i].SetRngState(next_rng[k]);
            } else if (modes[k].greedy) {
              const auto forced = working[l.i].ForcedToken();
              if (forced) {
                if (l.st->block == Block::kThinking)
                  l.st->budget_closed = true;
                const auto tk = static_cast<g::TextRunnerToken>(*forced);
                const bool stop = at >= max_context_ ||
                                  (l.st->stop_at_eos() && tok_->IsStop(tk));
                l.st->pending = g::TextDecodeSelection{
                    .stop = stop,
                    .token = tk,
                    .piece = stop ? std::string{} : Piece(*l.st, tk)};
              } else if (at >= max_context_) {
                l.st->pending =
                    g::TextDecodeSelection{.stop = true, .piece = {}};
              } else if (working[l.i].CanSelectArgmax(
                             static_cast<sampling::TokenId>(amax[k]), false) &&
                         RunnerAllowsToken(
                             *l.st, static_cast<g::TextRunnerToken>(amax[k]))) {
                const auto tk = static_cast<g::TextRunnerToken>(amax[k]);
                const bool stop = l.st->stop_at_eos() && tok_->IsStop(tk);
                l.st->pending = g::TextDecodeSelection{
                    .stop = stop,
                    .token = tk,
                    .piece = stop ? std::string{} : Piece(*l.st, tk)};
              } else {
                std::span<const float> row;
                engine_->ReadVerifyRows(k, 1, &row);
                l.st->pending = Select(*l.st, working[l.i], row, at);
              }
            } else {
              l.st->pending =
                  Select(*l.st, working[l.i], out.subspan(k * V, V), at);
            }
          } catch (...) {
            steps[l.i].failure = std::current_exception();
          }
        });
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t1)
                .count();
        const std::lock_guard lock(dsp_mutex_);
        if (nl >= 2)
          costs_batch_.ObserveVerify(nl, ms);
        // what a plain step of nl requests costs (verification steps of nl
        // rows from fewer requests are cheaper and share costs_batch_)
        double& plain = plain_batch_ms_[static_cast<std::size_t>(nl)];
        plain = plain > 0 ? plain + 0.1 * (ms - plain) : ms;
        if (neural)  // drafting that bought nothing
          batch_backoff_[B].Observe(nl, ms + draft_ms, plain / std::max(nl, 1),
                                    1.0);
      }
    } else {
      std::vector<Session*> sessions;
      std::vector<std::size_t> counts;
      std::vector<std::int32_t> rows;
      std::vector<std::size_t> first(lanes.size()), keeps(lanes.size(), 0);
      std::vector<bool> use_gpu_lin(lanes.size(), false);
      std::vector<std::array<std::uint64_t, Engine::kMaxVerifyRows + 1>>
          lane_draws(lanes.size());
      std::array<Engine::LinearSample, Engine::kMaxVerifyRows> ls{};
      bool any_greedy = false, any_host = false, any_gpu_lin = false;
      for (std::size_t k = 0; k < lanes.size(); ++k) {
        auto& l = lanes[k];
        first[k] = rows.size();
        sessions.push_back(l.ss);
        counts.push_back(static_cast<std::size_t>(1 + l.width));
        rows.push_back(l.anchor);
        rows.insert(rows.end(), l.drafts, l.drafts + l.width);
        l.pos = l.ss->position();
        use_gpu_lin[k] =
            l.mode.gpu_linear &&
            l.pos + static_cast<std::size_t>(l.width) + 1 < max_context_;
        if (use_gpu_lin[k]) {
          auto ahead = working[l.i];
          const auto& scfg = ahead.config();
          for (int j = 0; j <= l.width; ++j) {
            lane_draws[k][j] = ahead.rng_state();
            ls[first[k] + static_cast<std::size_t>(j)] = {
                .temperature = scfg.temperature,
                .min_p = scfg.min_p,
                .u = ahead.Uniform()};
          }
          lane_draws[k][l.width + 1] = ahead.rng_state();
          any_gpu_lin = true;
        } else if (l.mode.greedy) {
          any_greedy = true;
        } else {
          any_host = true;
        }
      }
      std::span<const float> out;
      std::int32_t amax[Engine::kMaxVerifyRows] = {};
      std::int32_t gpu_samp[Engine::kMaxVerifyRows] = {};
      bool ok = true;
      try {
        engine_->VerifyBatch(
            sessions, counts, rows, any_host ? &out : nullptr,
            any_greedy ? amax : nullptr,
            any_gpu_lin
                ? std::span<const Engine::LinearSample>(ls.data(), rows.size())
                : std::span<const Engine::LinearSample>{},
            any_gpu_lin ? gpu_samp : nullptr);
      } catch (...) {
        for (const auto& l : lanes)
          steps[l.i].failure = std::current_exception();
        ok = false;
      }
      const std::size_t V = engine_->config().n_vocab_text;
      const auto rows_of = [&](std::size_t k) {
        const auto& l = lanes[k];
        return (l.mode.greedy || use_gpu_lin[k])
                   ? std::span<const float>{}
                   : out.subspan(first[k] * V,
                                 static_cast<std::size_t>(1 + l.width) * V);
      };
      int emitted = 0;
      if (ok) {
        // selections on a thread per request (disjoint states), then the
        // commits in order
        std::size_t host_rows = 0;
        for (std::size_t k = 0; k < lanes.size(); ++k)
          host_rows += (lanes[k].mode.greedy || use_gpu_lin[k])
                           ? 0
                           : static_cast<std::size_t>(1 + lanes[k].width);
        ForEachLane(lanes.size(), host_rows, [&](std::size_t k) {
          auto& l = lanes[k];
          try {
            keeps[k] = Resolve(*l.st, working[l.i], steps[l.i], l.budget, l.pos,
                               l.width, l.drafts, rows_of(k), amax + first[k],
                               l.mode.greedy, first[k],
                               use_gpu_lin[k] ? (gpu_samp + first[k]) : nullptr,
                               use_gpu_lin[k] ? lane_draws[k].data() : nullptr);
          } catch (...) {
            steps[l.i].failure = std::current_exception();
          }
        });
        // one commit for every lane that resolved (failed lanes keep 0); keep
        // logits lazily on device since Resolve already populated st.pending
        std::vector<Session*> cs;
        std::vector<std::size_t> ck;
        std::vector<std::span<const float>> cl;
        for (std::size_t k = 0; k < lanes.size(); ++k)
          if (keeps[k] > 0) {
            cs.push_back(lanes[k].ss);
            ck.push_back(keeps[k]);
            cl.emplace_back();
          }
        bool committed = true;
        try {
          engine_->CommitVerifyBatch(cs, ck, cl);
        } catch (...) {
          committed = false;
          for (std::size_t k = 0; k < lanes.size(); ++k)
            if (keeps[k] > 0)
              steps[lanes[k].i].failure = std::current_exception();
        }
        for (std::size_t k = 0; k < lanes.size() && committed; ++k) {
          auto& l = lanes[k];
          const std::size_t keep = keeps[k];
          if (keep == 0)
            continue;  // failed above
          try {
            emitted += static_cast<int>(keep);
            if (l.width > 0) {
              steps[l.i].draft_rounds = 1;
              steps[l.i].draft_tokens = static_cast<std::size_t>(l.width);
              steps[l.i].draft_accepted_tokens = keep - 1;
            }
            if (l.copied && l.width > 0) {
              steps[l.i].lookup_tokens = static_cast<std::size_t>(l.width);
              steps[l.i].lookup_accepted_tokens = keep - 1;
              l.st->lookup.Observe(l.copy_start,
                                   static_cast<std::size_t>(l.width), keep - 1);
              l.st->copy_backoff.Observe(l.width, static_cast<int>(keep) - 1);
            }
            const std::lock_guard lock(dsp_mutex_);
            DsparkCalibration& cal = *cal_of(l);
            for (int j = 0; j < l.width && j < static_cast<int>(keep); ++j)
              cal.Observe(l.conf[j], j + 1 < static_cast<int>(keep));
          } catch (...) {
            steps[l.i].failure = std::current_exception();
          }
        }
      }
      if (ok) {
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t1)
                .count();
        const std::lock_guard lock(dsp_mutex_);
        costs_batch_.ObserveVerify(static_cast<int>(rows.size()), ms);
        const double plain = plain_batch_ms_[static_cast<std::size_t>(nl)];
        batch_backoff_[B].Observe(
            emitted,
            std::chrono::duration<double, std::milli>(Clock::now() - t0)
                .count(),
            (plain > 0 ? plain : costs_batch_.Verify(nl)) / nl, 1.0);
      }
    }
    for (std::size_t i = 0; i < decodes.size(); ++i)
      decodes[i].sampler.get().CopyDrawStateFrom(working[i]);
    return steps;
  }

  [[nodiscard]] std::optional<g::TextDecodeSelection> PreviewFirstToken(
      g::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }
  /// The encoder's workspace grows to the largest image seen (up to ~1 GiB)
  /// and is needed only while one is encoded.
  void Idle() const override {
    if (!vision_)
      return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    vision_->ReleaseWorkspace();
  }
  [[nodiscard]] std::size_t CheckpointPosition(
      const g::TextRunnerState& state) const override {
    return Require(state).session().position();
  }

private:
  static constexpr std::size_t kMaxStepTokens = 4096;
  static constexpr std::size_t kVisionDim = 2304;
  static constexpr int kLowDetailMaxSide = 512;
  static constexpr std::size_t kVisionScratchBytes = std::size_t{1} << 30;

  /// The image's encoder rows: from the image cache, or encoded once (the
  /// encoder is single-threaded) into rows of their own.
  void EnsureRows(ImageEntry& e) const {
    std::lock_guard<std::mutex> entry_lock(e.mutex);
    if (e.rows != nullptr)
      return;
    if ((e.rows = image_cache_.FindRows(e.image->digest)) != nullptr) {
      if (e.rows->count != e.count)
        throw std::logic_error(
            "inkling2: cached image rows do not match the prompt");
      return;
    }
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::shared_lock<std::shared_mutex> gate(
        CaptureGate());  // no graph capture while image kernels run
    vision::VisionEncoder::Result r;
    std::string why;
    if (vision_stream_ == nullptr &&
        hipStreamCreate(&vision_stream_) != hipSuccess)
      throw std::runtime_error("inkling2: cannot create the vision stream");
    const auto& px = e.image->pixels;
    if (!vision_->Encode(px.rgb.data(), px.w, px.h, vision_stream_, &r, &why))
      throw std::runtime_error("inkling2: image encoding failed: " + why);
    if (static_cast<std::size_t>(r.n_tokens) != e.count ||
        r.dim != static_cast<int>(kVisionDim))
      throw std::logic_error(
          "inkling2: image encoder row count does not match the prompt");
    auto rows = std::make_shared<ImageRows>();
    if (hipMalloc(&rows->device, e.count * kVisionDim * sizeof(float)) !=
        hipSuccess) {
      rows->device = nullptr;
      throw std::runtime_error("inkling2: out of device memory for image rows");
    }
    rows->count = e.count;
    if (hipMemcpyAsync(rows->device, r.embd,
                       e.count * kVisionDim * sizeof(float),
                       hipMemcpyDeviceToDevice, vision_stream_) != hipSuccess ||
        hipStreamSynchronize(vision_stream_) != hipSuccess)
      throw std::runtime_error("inkling2: image rows copy failed");
    image_cache_.PutRows(e.image->digest, rows);
    e.rows = std::move(rows);
  }

  std::shared_ptr<Engine> engine_;
  std::unique_ptr<::inkling2::text::Tokenizer> tok_;
  std::uint32_t max_context_;
  cache::Identity identity_;
  bool durable_;
  std::unique_ptr<vision::VisionEncoder> vision_;
  std::string vision_identity_;
  std::size_t max_draft_ = 0;  // proposals per request and cycle
  // threads for the requests' selections (up to eight rows a step)
  mutable LanePool lane_pool_{
      std::min(7U, std::max(1U, std::thread::hardware_concurrency()) - 1)};
  mutable std::mutex vision_mutex_;
  mutable ImageCache image_cache_;
  mutable hipStream_t vision_stream_ = nullptr;
  g::TextRunnerToken t_header_{}, t_text_{}, t_thinking_{}, t_end_message_{},
      t_invoke_{};
  // DSpark width policy, shared by the requests
  mutable std::mutex dsp_mutex_;
  mutable DsparkCalibration cal_, cal_coupled_;
  // copied proposals (prompt lookup) of greedy and of sampled requests
  mutable DsparkCalibration cal_copy_, cal_copy_sampled_;
  mutable double plain_greedy_ms_ = 20.2, plain_sampled_ms_ = 20.8;
  mutable DsparkCosts costs_;
  // batched cycles: costs by total rows (several sessions' rows; plain batched
  // steps too) and a back-off per batch width
  mutable DsparkCosts costs_batch_ = DsparkCosts::Batched();
  mutable std::array<DsparkBackoff, Engine::kMaxDecodeBatch + 1>
      batch_backoff_{};
  // measured plain batched steps by request count (0: none yet)
  mutable std::array<double, Engine::kMaxDecodeBatch + 1> plain_batch_ms_{};
};

/// The layout of the identity text (schema line, artifact id,
/// tokenizer/template hash, state ABI) follows Gufo's
/// src/models/qwen/continuation_identity.cpp (MIT, v0.35.0, 0260e11).
///
/// Canonical compatibility identity of the cache adapter: everything that
/// changes what a restored checkpoint means. With a known artifact fingerprint
/// (disk cache) it is stable across processes; otherwise it names this
/// process's engine, so a RAM-only cache can never leak across loads.
cache::Identity BuildIdentity(const ext::TextRunnerOptions& options,
                              const core::GgufReader& reader,
                              const Engine& engine,
                              const std::string& fingerprint,
                              const std::string& draft_fingerprint,
                              std::string* error) {
  const auto& c = engine.config();
  ::inkling2::text::TokenizerData tokenizer;
  std::string why;
  if (!::inkling2::text::ReadTokenizerData(reader, &tokenizer, &why)) {
    if (error)
      *error = "inkling2: tokenizer metadata: " + why;
    return {};
  }
  crypto::Sha256Hasher hasher;
  const auto feed = [&](std::string_view text) {
    const std::uint64_t n = text.size();
    hasher.Update({reinterpret_cast<const std::uint8_t*>(&n), sizeof n});
    hasher.Update(
        {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
  };
  feed(tokenizer.model);
  feed(tokenizer.pre);
  for (const auto& t : tokenizer.tokens)
    feed(t);
  for (const auto t : tokenizer.token_types)
    feed(std::to_string(t));
  for (const auto& m : tokenizer.merges)
    feed(m);
  feed(tokenizer.chat_template);
  std::ostringstream id;
  id << "schema=inkling2-component-cache-v1\n";
  if (!fingerprint.empty())
    id << "artifact_id=" << core::kGgufIdentityScheme << ':' << fingerprint
       << '\n';
  else
    id << "artifact_id=process:" << options.model_path << ':'
       << static_cast<const void*>(&engine) << '\n';
  id << "tokenizer_template_sha256=" << hasher.FinishHex() << '\n'
     << "template_impl=inkling2-cpp-renderer-v1\n"
     << "state_abi=" << kStateAbi << '\n'
     << "kv_dtype=f16\nconv_dtype=f32\nlogits_dtype=f32\nlogits_vocab="
     << c.n_vocab_text << '\n'
     << "loop=phys:" << c.n_phys << " visits:" << c.n_visits
     << " core:" << c.core_start << '-' << c.core_end
     << " passes:" << c.n_passes << " swa_window:" << c.n_swa
     << " ring_rows:" << engine.ring_rows() << " chunk_rows:" << kRowsPerChunk
     << '\n'
     << "numerics=ubatch:" << engine.ubatch() << " softcap:" << c.logit_softcap
     << " log_scale:" << c.log_scale_floor << '/' << c.log_scale_alpha
     << " rms_eps:" << c.rms_eps << '\n'
     << "adapters=none\n";
  // A drafter's context rows are part of the checkpoints.
  if (engine.has_drafter())
    id << "speculative=dspark:"
       << (draft_fingerprint.empty() ? "process" : draft_fingerprint) << '\n';
  else
    id << "speculative=none\n";
  const auto text = id.str();
  return {text.begin(), text.end()};
}

/// The context the model was trained for (1,048,576 for Inkling 2 Mini): the
/// default when the command line names none.
std::uint32_t NativeContext(const core::GgufReader& reader) {
  const auto arch =
      reader.GetMetadataString("general.architecture").value_or("inkling2");
  return reader.GetMetadataUint32(std::string(arch) + ".context_length")
      .value_or(1048576);
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateInkling2TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    std::shared_ptr<const core::GgufReader> reader = options.reader;
    if (!reader) {
      core::GgufOpenOptions open;
      open.allow_unsized_tensors = true;
      std::string why;
      std::shared_ptr<core::GgufReader> opened =
          core::GgufReader::OpenFile(options.model_path, &why, open);
      if (!opened) {
        if (error)
          *error = "inkling2: cannot open " + options.model_path + ": " + why;
        return nullptr;
      }
      reader = std::move(opened);
    }
    const bool dspark =
        options.speculative.backend == ext::TextSpeculativeBackend::kDSpark;
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
        !dspark) {
      if (error)
        *error = "inkling2: the only speculative backend is dspark";
      return nullptr;
    }
    if (dspark && options.speculative.draft_model_path.empty()) {
      if (error)
        *error = "inkling2: DSpark needs --dspark-model";
      return nullptr;
    }
    const std::uint32_t max_context =
        options.max_context ? options.max_context : NativeContext(*reader);
    std::string why;
    auto tok = ::inkling2::text::Tokenizer::CreateFromGguf(*reader, &why);
    if (!tok) {
      if (error)
        *error = "inkling2: tokenizer: " + why;
      return nullptr;
    }
    auto engine = Engine::Create(reader, max_context, &why);
    if (!engine) {
      if (error)
        *error = why;
      return nullptr;
    }
    if (dspark &&
        !engine->AttachDrafter(options.speculative.draft_model_path, &why)) {
      if (error)
        *error = "inkling2: " + why;
      return nullptr;
    }
    if (max_context > engine->config().n_ctx_train)
      std::fprintf(stderr,
                   "inkling2: warning: context %u exceeds the model's training "
                   "context %u\n",
                   max_context, engine->config().n_ctx_train);
    // A disk cache needs an identity that survives the process: the hash of the
    // whole artifact (computed once, cached privately by Gufo by
    // inode/size/mtime), as the in-tree runners do.
    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (!options.disk_cache.directory.empty() && fingerprint.empty()) {
      try {
        fingerprint = core::GgufIdentityHex(*reader);
      } catch (const std::exception& e) {
        if (error)
          *error = std::string("inkling2: disk cache fingerprinting failed: ") +
                   e.what();
        return nullptr;
      }
    }
    std::string draft_fingerprint;
    if (dspark && !fingerprint.empty()) {
      try {
        core::GgufOpenOptions draft_open;
        draft_open.allow_unsized_tensors = true;
        const auto draft = core::GgufReader::OpenFile(
            options.speculative.draft_model_path, &why, draft_open);
        if (!draft) {
          if (error)
            *error = "inkling2: drafter: " + why;
          return nullptr;
        }
        draft_fingerprint = core::GgufIdentityHex(*draft);
      } catch (const std::exception& e) {
        if (error)
          *error = std::string("inkling2: drafter fingerprinting failed: ") +
                   e.what();
        return nullptr;
      }
    }
    const auto identity = BuildIdentity(options, *reader, *engine, fingerprint,
                                        draft_fingerprint, &why);
    if (identity.empty()) {
      if (error)
        *error = why;
      return nullptr;
    }
    const bool durable = !fingerprint.empty();
    std::unique_ptr<vision::VisionEncoder> vision;
    std::string vision_identity;
    if (!options.vision_model_path.empty()) {
      vision = std::make_unique<vision::VisionEncoder>();
      if (!vision->Load(options.vision_model_path, &why)) {
        if (error)
          *error = "inkling2: mmproj " + options.vision_model_path + ": " + why;
        return nullptr;
      }
      vision_identity = "inkling2-vision-v1:" +
                        crypto::Sha256FileHex(options.vision_model_path);
    }
    // --draft-tokens (the provider's default: the drafter's whole block)
    const int max_draft =
        dspark ? static_cast<int>(options.speculative.max_draft_tokens) : 0;
    return std::make_shared<Inkling2Runner>(
        std::shared_ptr<Engine>(std::move(engine)), std::move(tok), max_context,
        identity, durable, std::move(vision), std::move(vision_identity),
        max_draft);
  } catch (const std::exception& e) {
    if (error)
      *error = std::string("inkling2: ") + e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::inkling2
