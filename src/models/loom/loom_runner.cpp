// Loom model packages as Gufo text models.
//
// A fetched Loom package (model.json, its compiled programs and parameters)
// runs through Gufo's ordinary step contract, TextModelRunner. Gufo's server,
// scheduler, component cache, sampler policy and parsers stay in charge; Loom
// supplies the model's kernels, chat program, tokenizer, output parser and
// drafter through libloom_text_embed:
//
// - Each Gufo session slot is one resident Loom row. Gufo decides which
//   sessions step together; one decode batch is one packed Loom epoch, and a
//   package with a drafter decodes several verified tokens per step.
// - Gufo's component cache keeps Loom row checkpoints: a capture pins the
//   row's exact state on device (pages shared copy-on-write), a restore puts
//   it into any row, and evicting the cache entry releases it.
// - Output is read with the package's own chat program, as loom-serve reads
//   it, and streamed to Gufo in one canonical markup: reasoning inside
//   <think>...</think>, answers as text, tool calls as <tool_call> JSON
//   blocks. Gufo's parsers then serve every endpoint for every architecture.
//
// Sampling runs on device with the package's sampler (temperature, top-k,
// top-p, seed). Gufo's grammar constraints, logprobs, min-p and penalties need
// the logit row, which a package that declares "llm.logits" returns when it is
// loaded with the capability (always, when it declares it): a request that
// needs them is marked "logits": true for the package, every selection of its
// row leaves the full row, and the token is chosen here with Gufo's
// SamplerState and fed back through the epoch (decode_batch_tokens). Every
// other request, and every package without the capability, keeps the device
// path unchanged.

#include "loom_cache.hpp"
#include "loom_chat.hpp"
#include "loom_dialect.hpp"
#include "loom_source.hpp"

namespace gufo::loom {
namespace {

// ---------------------------------------------------------------- state ---

class LoomState final : public ext::TextRunnerState {
public:
  LoomState(LoomSlot& slot, loom_text_embed_decoder_t* decoder)
      : slot(&slot), decoder(decoder) {}
  ~LoomState() override { loom_text_embed_decoder_free(decoder); }
  void Invalidate() noexcept override {
    slot->position = 0;
    slot->deferred = -1;
    slot->stale = true;
    ResetHost();
  }
  [[nodiscard]] std::uint32_t row() const { return slot->row(); }
  [[nodiscard]] std::size_t position() const { return slot->position; }

  LoomSlot* slot;
  // Per request: output reader, streaming detokenizer, the bound sampling,
  // the selection whose piece was produced, and whether anything was.
  std::shared_ptr<const LoomPromptContext> context;
  loom_text_embed_decoder_t* decoder;
  std::string bound;
  std::optional<std::pair<std::size_t, std::int32_t>> piece_key;
  std::string piece;
  std::size_t generated{0};
  bool produced{false};
  bool unconstrained{false};
  bool flushed{false};
  // Whether the streamed output is inside reasoning, and its token count.
  bool reasoning{false};
  std::size_t reasoning_tokens{0};
  bool budget_closed{false};
  // Whether any canonical output text was streamed yet.
  bool emitted{false};

  // Host sampling (the logits capability). `flag` is what the row's bound
  // sampling asks of the package's next epoch; `sticky` keeps it on once a
  // constraint restricts; `hosted` says the request samples on the host when
  // it needs to.
  bool flag{false};
  bool sticky{false};
  bool hosted{false};
  bool hosted_known{false};
  // The next selection when the host already chose it (from the row's
  // logits, right after the epoch that produced them).
  struct HostNext {
    TextRunnerToken token{0};
    bool stop{false};
    std::uint64_t rng{0};
    std::optional<sampling::TokenLogprobs> logprobs;
  };
  std::optional<HostNext> next;
  // SelectNext's sampler, for the Advance that follows it.
  std::unique_ptr<sampling::SamplerState> snapshot;
  std::vector<float> logits;
  // The last raw bytes the request wrote: a lazy tool grammar restricts only
  // once the call marker is in it.
  std::string tail;

  void ResetHost() {
    flag = sticky = hosted = hosted_known = false;
    tail.clear();
    next.reset();
    snapshot.reset();
  }
};

LoomState& Require(ext::TextRunnerState& state) {
  auto* loom = dynamic_cast<LoomState*>(&state);
  if (loom == nullptr)
    throw std::invalid_argument("state does not belong to the Loom runner");
  return *loom;
}

// --------------------------------------------------------------- runner ---

class LoomRunner final : public ext::TextModelRunner {
public:
  LoomRunner(std::shared_ptr<LoomModel> model, std::uint32_t max_context,
             bool disk_cache, std::size_t ram_cache_bytes)
      : model_(std::move(model)),
        max_context_(
            max_context == 0
                ? model_->info().context_capacity
                : std::min(max_context, model_->info().context_capacity)),
        rows_(std::max<std::uint32_t>(model_->info().rows, 1)),
        width_(std::max<std::uint32_t>(
            1,
            std::min<std::uint32_t>(model_->info().max_epoch_spans, rows_))) {
    ram_cache_bytes_ = ram_cache_bytes;
    if (disk_cache) {
      // A disk tier persists what the RAM tier holds, so entries become
      // portable checkpoint images instead of named device checkpoints.
      loom_text_embed_image_layout_t layout{};
      loom_text_embed_image_layout(model_->embed(), &layout);
      const auto rows =
          ImageLayout::RowsPerPage(layout.page_bytes, layout.block_size);
      if (layout.supported && layout.page_bytes > 0 && rows)
        images_ = ImageLayout{.block_size = layout.block_size,
                              .fixed_bytes = layout.fixed_bytes,
                              .page_bytes = layout.page_bytes,
                              .rows_per_page = *rows};
      else
        std::fprintf(stderr,
                     "loom: this preset's state cannot be saved as images; "
                     "the prompt cache stays in device memory\n");
    }
    close_tokens_ = DiscoverReasoningClose();
    const char* source = close_tokens_.empty() ? "probe" : "template";
    if (close_tokens_.empty())
      ProbeReasoningClose();
    std::lock_guard lock(model_->mutex());
    // Every session's KV comes from one pool that Loom pages itself, sized
    // for each session's full context (LoomModel).
    loom_text_embed_memory(model_->embed(), &memory_);
    const auto gib = [](std::uint64_t bytes) {
      return static_cast<double>(bytes) / static_cast<double>(1ull << 30);
    };
    std::fprintf(
        stderr,
        "loom: device memory %.1f GiB, KV pool %.1f GiB of it (%llu tokens "
        "shared by %u sessions)\n",
        gib(memory_.device_bytes), gib(KvBytes()),
        static_cast<unsigned long long>(memory_.state_tokens), rows_);
    std::fprintf(stderr, "loom: reasoning close %s (%s)\n",
                 close_tokens_.empty()
                     ? "unknown; reasoning budgets are not enforced"
                     : ("'" + DecodeLocked(close_tokens_) + "'").c_str(),
                 source);
    if (model_->logits().enabled) {
      // Host sampling applies Gufo's grammars to the model's own tokens, so
      // they must be written in the dialect the model speaks, which the
      // package's llm_config.json names. Without it, tool grammars stay off.
      if (model_->chat_config()) {
        native_ = NativeDialectOf(*model_->chat_config());
        reasoning_ok_ = true;
      } else {
        reasoning_ok_ = close_tokens_.empty() ||
                        Trimmed(DecodeLocked(close_tokens_)) == "</think>";
      }
      std::fprintf(stderr,
                   "loom: logits on (%u floats per row); native tool calls: "
                   "%s%s; reasoning markers: %s\n",
                   model_->logits().width,
                   native_.tools ? "grammar available" : "no grammar",
                   native_.known ? (" (" + native_.call_format + ")").c_str()
                                 : " (undeclared)",
                   !reasoning_ok_ ? "differ: grammars stay off"
                   : native_.reasoning_open.empty()
                       ? "none"
                       : ("'" + native_.reasoning_open + "' ... '" +
                          native_.reasoning_close + "'")
                             .c_str());
    }
  }

  // The grammar's view of the model's own tokens (see loom_dialect.hpp):
  // only for a logits load, which is the only one with grammars.
  [[nodiscard]] std::optional<sampling::JsonConstraint::ToolSyntax>
  ToolGrammarSyntax(const ext::ChatRequest&) const override {
    return model_->logits().enabled ? native_.tools : std::nullopt;
  }

  [[nodiscard]] std::optional<ReasoningMarkup> ReasoningGrammarMarkers(
      const ext::ChatRequest& request) const override {
    if (!model_->logits().enabled || !native_.known)
      return std::nullopt;
    if (native_.reasoning_open.empty())
      return ReasoningMarkup{"", "</think>"};  // no reasoning phase
    std::string close = native_.reasoning_close + native_.after_reasoning;
    if (request.response_format)
      close += native_.answer_channel;
    return ReasoningMarkup{native_.reasoning_open, close};
  }

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->info().model_id,
        .state_abi = "loom-text-2",
        .max_context = max_context_,
        .capabilities =
            {
                .supports_vision = model_->info().max_images > 0,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                // Every decode step goes through DecodeBatch, which sees the
                // next selection before publishing (end of output).
                .multi_token_decode = true,
                .batched_multi_token_decode = width_ > 1,
                .batched_multi_token_decode_max_width = width_,
                .prefix_reuse = true,
            },
        .scheduling =
            {
                .pass_tokens =
                    std::max<std::size_t>(model_->info().prefill_capacity, 1),
                .max_layer_groups = 1,
                .max_yield_decode_width = 1,
            },
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    // From what Loom committed (loom_text_embed_memory): the pool holds every
    // session's full context, so each claims its share of the KV; the host
    // cache holds only checkpoint names and its own metadata.
    return {
        .resident_weights_bytes =
            memory_.device_bytes - std::min(memory_.device_bytes, KvBytes()),
        .per_request_state_bytes = KvBytes() / rows_,
        .temporary_scratch_bytes = 0,
        // Image blocks fill ImageCacheBytes; the cache's own metadata,
        // restore bookkeeping and staging take the rest.
        .retained_snapshot_capacity_bytes =
            images_ ? ImageCacheBytes() + MetadataBytes() * 8 : MetadataBytes(),
        .retained_snapshot_ceiling_bytes =
            images_ ? ImageCacheBytes() + MetadataBytes() * 8
                    : MetadataBytes()};
  }

  // The host cache's metadata: its fixed share, and the token history each
  // retained checkpoint keeps (Gufo charges about four bytes per token), so
  // long contexts in every session do not exhaust it.
  [[nodiscard]] std::size_t MetadataBytes() const {
    return kHostCacheBytes + kCheckpoints * std::size_t{max_context_} * 8;
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans()
      const override {
    std::vector<ext::TextExecutionPlan> plans{
        {.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= width_; ++width)
      plans.push_back({.kind = ext::TextExecutionPlanKind::kBatched,
                       .physical_width = width});
    return plans;
  }

  // Whether tool requests get a grammar in this package's own call syntax
  // (logits load only).
  [[nodiscard]] bool NativeToolGrammar() const {
    return native_.tools.has_value();
  }

  [[nodiscard]] bool IsStopToken(std::uint32_t id) const {
    std::lock_guard lock(model_->mutex());
    return loom_text_embed_is_stop_token(model_->embed(),
                                         static_cast<std::int32_t>(id)) != 0;
  }

  [[nodiscard]] std::uint32_t VocabularySize() const override {
    std::lock_guard lock(model_->mutex());
    return static_cast<std::uint32_t>(
        loom_text_embed_vocabulary_size(model_->embed()));
  }

  // Gufo binds a grammar to every tool request; the package samples on
  // device, so the grammar is released (see Unconstrain).
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    // The package's stop token ends a response: grammars admit it once they
    // are complete (host sampling only; the device path never sees a grammar).
    return std::make_shared<const sampling::ConstraintVocabulary>(
        VocabularySize(), [this](std::uint32_t id) {
          const bool stop = IsStopToken(id);
          return sampling::ConstraintVocabulary::Piece{
              stop ? std::string()
                   : Decode(std::span<const TextRunnerToken>(&id, 1)),
              stop};
        });
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::lock_guard lock(model_->mutex());
    std::vector<std::int32_t> tokens(text.size() + 16);
    std::size_t count = 0;
    model_->Check(
        loom_text_embed_tokenize(model_->embed(), text.data(), text.size(), 1,
                                 tokens.data(), tokens.size(), &count),
        "tokenize");
    return std::vector<TextRunnerToken>(tokens.begin(), tokens.begin() + count);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    auto prepared = PreparePrompt(request);
    if (!prepared)
      return std::nullopt;
    return std::move(prepared->tokens);
  }

  // The package's chat program renders the prompt; the request's output
  // reader travels with it to the state (SetPromptContext).
  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PreparePrompt(
      const ext::ChatRequest& request) const override {
    return PrepareWith(request, true);
  }

  // Renders `request` with the package's chat program; with `next_turn`, also
  // renders the next turn to find the stable cache prefix. Gufo asks for the
  // initial output state right after preparing the same request, so the
  // prompt's reasoning state is remembered by body: each render of a long
  // conversation tokenizes all of it under the model mutex.
  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PrepareWith(
      const ext::ChatRequest& request, bool next_turn) const {
    std::string body = ChatBody(model_->info().model_id, request);
    const auto key = BodyKey(body);
    std::unique_lock lock(model_->mutex());
    std::vector<std::int32_t> tokens(model_->info().context_capacity);
    std::size_t count = 0;
    loom_text_embed_request_t* handle = nullptr;
    int status = loom_text_embed_chat_prepare(model_->embed(), body.data(),
                                              body.size(), tokens.data(),
                                              tokens.size(), &count, &handle);
    if (status != 0 && request.reasoning.effort.has_value()) {
      // A package without this effort level: Gufo's other runners ignore
      // an effort they do not support, so render without it.
      body = ChatBody(model_->info().model_id, request, false);
      status = loom_text_embed_chat_prepare(model_->embed(), body.data(),
                                            body.size(), tokens.data(),
                                            tokens.size(), &count, &handle);
    }
    if (status != 0)
      throw std::invalid_argument(std::string("loom: ") +
                                  loom_text_embed_error(model_->embed()));
    auto context = std::make_shared<LoomPromptContext>(model_, handle);
    tokens.resize(count);
    context->opens_reasoning = OpensReasoningLocked(tokens);
    RememberReasoningLocked(key, context->opens_reasoning);
    // Automatic tool use stays on the device where a lazy grammar cannot
    // follow the model (NativeDialect::auto_on_device).
    const bool lazy_ambiguous =
        native_.auto_on_device &&
        request.tool_choice == ext::ChatRequest::ToolChoice::kAuto;
    context->constraint_ok =
        model_->logits().enabled && reasoning_ok_ &&
        (request.tools.empty() || (native_.tools && !lazy_ambiguous));
    if (native_.tools)
      context->lazy_open = native_.tools->open;
    context->lazy_tools =
        !request.tools.empty() && !request.response_format &&
        request.tool_choice == ext::ChatRequest::ToolChoice::kAuto;
    if (request.reasoning.budget_tokens && !close_tokens_.empty())
      context->reasoning_budget = request.reasoning.budget_tokens;
    if (const std::size_t media =
            loom_text_embed_request_media(handle, nullptr, nullptr, 0)) {
      std::vector<std::uint32_t> offsets(media), counts(media);
      loom_text_embed_request_media(handle, offsets.data(), counts.data(),
                                    media);
      // Image rows are placeholders in token space: each prefix carries the
      // identity of the images before it, so cached state never crosses
      // different images.
      std::vector<std::uint8_t> identity;
      std::size_t index = 0;
      for (const auto& message : request.messages) {
        for (const auto& image : message.images) {
          if (index < media) {
            context->cache_prefixes.push_back(
                {.token_count = offsets[index], .identity = identity});
            context->images.emplace_back(offsets[index], counts[index]);
          }
          crypto::Sha256Hasher hasher;
          hasher.Update(std::span<const std::uint8_t>(*image.bytes));
          const auto digest = hasher.Finish();
          identity.insert(identity.end(), digest.begin(), digest.end());
          ++index;
        }
      }
      context->cache_identity = identity;
    }
    // The next turn renders this conversation plus the answer and a new user
    // message. Where that rendering departs from this prompt, the assistant
    // framing begins: Gufo checkpoints before it so the next turn reuses all
    // earlier work, whatever the package's template.
    std::size_t stable = 0;
    // Other sessions decode between the two renders: each of a long
    // conversation holds the model mutex for tens of milliseconds.
    lock.unlock();
    if (next_turn && !request.messages.empty()) {
      ext::ChatRequest next = request;
      next.messages.emplace_back(ext::ChatRole::kAssistant, "\x01");
      next.messages.emplace_back(ext::ChatRole::kUser, "\x01");
      const std::string next_body = ChatBody(model_->info().model_id, next);
      std::vector<std::int32_t> next_tokens(model_->info().context_capacity);
      std::size_t next_count = 0;
      lock.lock();
      if (loom_text_embed_chat_prepare(model_->embed(), next_body.data(),
                                       next_body.size(), next_tokens.data(),
                                       next_tokens.size(), &next_count,
                                       nullptr) == 0) {
        const std::size_t limit = std::min(count, next_count);
        while (stable < limit && tokens[stable] == next_tokens[stable])
          ++stable;
      }
    }
    return ext::TextPreparedPrompt{
        .tokens = std::vector<TextRunnerToken>(tokens.begin(),
                                               tokens.begin() + count),
        .context = std::move(context),
        .cache_prefix_tokens = stable < count ? stable : 0,
    };
  }

  void SetPromptContext(
      ext::TextRunnerState& state,
      std::shared_ptr<const ext::TextPromptContext> context) const override {
    auto& loom = Require(state);
    auto loom_context =
        std::dynamic_pointer_cast<const LoomPromptContext>(context);
    if (context != nullptr && loom_context == nullptr)
      throw std::invalid_argument("prompt context is not from the Loom runner");
    // Called again after a restore within the same request.
    if (loom_context == loom.context && !loom.produced)
      return;
    std::lock_guard lock(model_->mutex());
    loom.context = std::move(loom_context);
    loom.bound.clear();
    loom.piece_key.reset();
    loom.generated = 0;
    loom.produced = false;
    loom.unconstrained = false;
    loom.flushed = false;
    loom.reasoning = loom.context != nullptr && loom.context->opens_reasoning;
    loom.reasoning_tokens = 0;
    loom.budget_closed = false;
    loom.emitted = false;
    loom.ResetHost();
    model_->Check(loom_text_embed_decoder_reset(loom.decoder), "decoder");
  }

  // Reasoning budgets are enforced here, with the package's own close marker
  // (any tokenizer); Gufo's token-tag budget is not used.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"", "", {}};
  }
  // Where the package's chat program leaves the prompt; usage accounting
  // counts reasoning from a prompt-opened reasoning block.
  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ext::ChatRequest& request) const override {
    // Gufo also asks with no messages (its defaults at load); some chat
    // programs refuse a chat without a user turn, so render a placeholder.
    std::optional<ext::ChatRequest> placeholder;
    if (request.messages.empty()) {
      placeholder = request;
      placeholder->messages.emplace_back(ext::ChatRole::kUser, ".");
    }
    const auto& rendered = placeholder ? *placeholder : request;
    std::optional<bool> opens =
        RecalledReasoning(BodyKey(ChatBody(model_->info().model_id, rendered)));
    if (!opens) {
      const auto prepared = PrepareWith(rendered, false);
      const auto* context =
          prepared
              ? dynamic_cast<const LoomPromptContext*>(prepared->context.get())
              : nullptr;
      opens = context != nullptr && context->opens_reasoning;
    }
    // A model that always reasons starts in reasoning even when the request
    // turned it off: a grammar must leave it room to reason and frame its
    // answer (harmony's final channel) instead of forcing the answer first.
    // Only requests that build such a grammar: a plain turn may skip
    // reasoning, and its answer must then parse as content.
    const bool constrained =
        request.response_format != nullptr ||
        request.tool_choice == ext::ChatRequest::ToolChoice::kRequired ||
        !request.forced_tool_name.empty();
    const bool always = model_->logits().enabled && native_.known &&
                        native_.always_reasons &&
                        !native_.reasoning_open.empty() && constrained;
    return *opens || always ? ext::InitialOutputState::kThought
                            : ext::InitialOutputState::kAuto;
  }

  using BodyDigest = std::array<std::uint8_t, 32>;

  static BodyDigest BodyKey(std::string_view body) {
    crypto::Sha256Hasher hasher;
    hasher.Update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(body.data()), body.size()));
    return hasher.Finish();
  }

  void RememberReasoningLocked(const BodyDigest& key, bool opens) const {
    std::erase_if(recent_reasoning_,
                  [&](const auto& entry) { return entry.first == key; });
    if (recent_reasoning_.size() >= kRecentPrompts)
      recent_reasoning_.erase(recent_reasoning_.begin());
    recent_reasoning_.emplace_back(key, opens);
  }

  std::optional<bool> RecalledReasoning(const BodyDigest& key) const {
    std::lock_guard lock(model_->mutex());
    for (const auto& [digest, opens] : recent_reasoning_) {
      if (digest == key)
        return opens;
    }
    return std::nullopt;
  }

  // Whether the prompt ends inside an open "<think>" block.
  bool OpensReasoningLocked(const std::vector<std::int32_t>& tokens) const {
    if (tokens.empty())
      return false;
    std::string text = DecodeLocked(std::vector<std::int32_t>(
        tokens.end() - std::min<std::size_t>(tokens.size(), 4), tokens.end()));
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    static constexpr std::string_view kOpen = "<think>";
    return text.size() >= kOpen.size() &&
           text.compare(text.size() - kOpen.size(), kOpen.size(), kOpen) == 0;
  }

  // The canonical markup the output reader streams, for every package.
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    return {"<think>", "</think>"};
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::lock_guard lock(model_->mutex());
    return DecodeLocked(
        std::vector<std::int32_t>(tokens.begin(), tokens.end()));
  }

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t) const override {
    // One ledger block per retainable checkpoint: when Gufo's cache holds
    // kCheckpoints entries, a new capture makes it evict one first.
    if (images_)
      return CreateImageCacheResources(ledger);
    struct Block {
      cache::ResourceCharge pool;
    };
    auto blocks = std::make_shared<std::vector<Block>>();
    for (std::size_t i = 0; i < kCheckpoints; ++i)
      blocks->push_back(
          {ledger.Reserve(cache::ResourceCategory::kBackingFree, 64)
               .Convert()});
    auto resources = std::make_unique<ext::ComponentCacheResources>();
    resources->owner = blocks;
    resources->adapter =
        std::make_unique<LoomAdapter>(model_, rows_, std::nullopt);
    resources->stream = [] { return std::make_unique<NullStream>(); };
    resources->allocate = [blocks,
                           model = model_](const cache::PayloadRequest& request)
        -> cache::ComponentCaptureBuffer {
      for (auto& block : *blocks) {
        if (request.bytes > kMarkerBytes ||
            block.pool.Info().category != cache::ResourceCategory::kBackingFree)
          continue;
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          auto payload = std::make_shared<EndpointPayload>();
          payload->model = model;
          const std::span<std::byte> bytes(payload->bytes);
          return {bytes,
                  [reservation, payload] {
                    // Gufo reads a payload through its owner: alias the
                    // bytes, keeping the whole payload (and checkpoint) alive.
                    return cache::Payload::Committed(
                        reservation->Convert(),
                        std::shared_ptr<const void>(payload,
                                                    payload->bytes.data()));
                  },
                  payload.get()};
        } catch (const std::bad_alloc&) {
        }
      }
      throw cache::ResourceExhausted();
    };
    resources->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    return resources;
  }

  // Image mode: payloads are host bytes charged to Gufo's ledger.
  std::unique_ptr<ext::ComponentCacheResources> CreateImageCacheResources(
      cache::ResourceLedger& ledger) const {
    auto resources = std::make_unique<ext::ComponentCacheResources>();
    resources->adapter = std::make_unique<LoomAdapter>(model_, rows_, images_);
    resources->stream = [] { return std::make_unique<NullStream>(); };
    // The budget is committed up front as ledger blocks (private-state
    // blocks and page-chunk blocks), as Gufo's cache expects; each capture
    // claims a free block and allocates its host bytes.
    struct Block {
      cache::ResourceCharge pool;
      std::size_t bytes;
      bool private_state;
    };
    const std::size_t budget = ImageCacheBytes();
    const std::size_t private_bytes = images_->private_bytes();
    const std::size_t chunk_bytes = kPagesPerChunk * images_->page_bytes;
    const std::size_t private_blocks = std::clamp<std::size_t>(
        budget / 4 / std::max<std::size_t>(private_bytes, 1), 2, kCheckpoints);
    const std::size_t chunk_blocks =
        budget > private_blocks * private_bytes
            ? (budget - private_blocks * private_bytes) / chunk_bytes
            : 0;
    auto blocks = std::make_shared<std::vector<Block>>();
    for (std::size_t i = 0; i < private_blocks + chunk_blocks; ++i) {
      const bool private_state = i < private_blocks;
      const std::size_t bytes = private_state ? private_bytes : chunk_bytes;
      blocks->push_back(
          {ledger.Reserve(cache::ResourceCategory::kBackingFree, bytes)
               .Convert(),
           bytes, private_state});
    }
    resources->owner = blocks;
    resources->allocate = [blocks](const cache::PayloadRequest& request)
        -> cache::ComponentCaptureBuffer {
      const bool private_state =
          request.category == cache::ResourceCategory::kPrivateState;
      for (auto& block : *blocks) {
        if (block.private_state != private_state ||
            request.bytes > block.bytes ||
            block.pool.Info().category != cache::ResourceCategory::kBackingFree)
          continue;
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          auto data = std::make_shared<std::vector<std::byte>>(request.bytes);
          const std::span<std::byte> bytes(*data);
          return {bytes,
                  [reservation, data] {
                    return cache::Payload::Committed(
                        reservation->Convert(),
                        std::shared_ptr<const void>(data, data->data()));
                  },
                  data.get()};
        } catch (const std::bad_alloc&) {
        }
      }
      throw cache::ResourceExhausted();
    };
    resources->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    return resources;
  }

  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot& slot) const override {
    std::lock_guard lock(model_->mutex());
    loom_text_embed_decoder_t* decoder = nullptr;
    model_->Check(loom_text_embed_decoder_open(model_->embed(), &decoder),
                  "decoder");
    return std::make_unique<LoomState>(dynamic_cast<LoomSlot&>(slot), decoder);
  }

  // After a restore the row holds the checkpoint's state at `position`.
  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    auto& loom = Require(state);
    std::lock_guard lock(model_->mutex());
    loom.slot->position = position;
    if (position == 0) {
      ResetRowLocked(loom);
      return;
    }
    loom.slot->stale = false;
  }

  // Runs a row reservation (0: reserved, 1: no capacity). When the device
  // pool cannot back it, retained checkpoints are released oldest first (their
  // cache entries then miss and re-prefill) and it is tried again, as
  // loom-serve's prefix cache gives way under pressure.
  template<class Reserve>
  bool ReserveLocked(Reserve&& reserve) const {
    for (;;) {
      if (model_->Check(reserve(), "reserve", 1) == 0)
        return true;
      if (!model_->ReleaseOldest())
        return false;
    }
  }

  [[nodiscard]] ext::TextPrefillStep Prefill(
      ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& loom = Require(state);
    if (offset != loom.position() || offset >= prompt.size())
      throw std::invalid_argument("loom prefill must continue the row");
    const std::size_t count = std::min(
        prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1));
    const bool last = offset + count == prompt.size();
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    // The pool holds sessions x --context tokens: a row's slack stays within
    // its own share, or concurrent rows run the pool dry.
    const std::uint64_t extent =
        std::min<std::uint64_t>(max_context_, prompt.size() + kExtentSlack);
    if (offset == 0 || loom.slot->stale) {
      if (offset != 0)
        throw std::logic_error("loom row lost its state");
      ResetRowLocked(loom);
      if (!ReserveLocked([&] {
            return loom_text_embed_row_begin(embed, loom.row(), nullptr, extent,
                                             1);
          }))
        throw cache::ResourceExhausted();
    } else if (const std::uint64_t actual =
                   loom_text_embed_row_position(embed, loom.row()) +
                   (loom.slot->deferred >= 0 ? 1 : 0);
               actual != offset) {
      // The row cannot be rewound: its state must end exactly where Gufo
      // continues the prompt.
      throw std::logic_error("loom row holds " + std::to_string(actual) +
                             " tokens, continued at " + std::to_string(offset));
    } else if (loom_text_embed_row_extent(embed, loom.row()) < extent) {
      if (!ReserveLocked([&] {
            return loom_text_embed_row_extend(embed, loom.row(), extent);
          }))
        throw cache::ResourceExhausted();
      loom.bound.clear();
    }
    if (loom.context != nullptr && loom.context->has_images()) {
      // Image prompts run through the request's checked input, whose rows
      // at image positions are projected features.
      if (!loom.context->projected) {
        model_->Check(loom_text_embed_request_project(loom.context->request),
                      "project images");
        loom.context->projected = true;
      }
      if (loom.slot->deferred >= 0) {
        const std::int32_t token =
            static_cast<std::int32_t>(loom.slot->deferred);
        model_->Check(loom_text_embed_row_prefill(embed, loom.row(), &token, 1),
                      "prefill");
      }
      const std::size_t now = count - (last ? 1 : 0);
      const std::size_t step =
          std::max<std::size_t>(model_->info().prefill_capacity, 1);
      for (std::size_t done = 0; done < now;) {
        const std::size_t chunk = std::min(step, now - done);
        model_->Check(loom_text_embed_row_prefill_request(
                          embed, loom.row(), loom.context->request,
                          offset + done, chunk, 0),
                      "prefill");
        done += chunk;
      }
      loom.slot->deferred =
          last ? static_cast<std::int64_t>(prompt.back()) : -1;
      loom.slot->position += count;
      return {.consumed_tokens = count, .decode_ready = last};
    }
    // A deferred token from the last prefill (or a restored checkpoint)
    // enters state first; this prompt's final token waits for its sampler.
    std::vector<std::int32_t> input;
    if (loom.slot->deferred >= 0)
      input.push_back(static_cast<std::int32_t>(loom.slot->deferred));
    input.insert(input.end(), prompt.begin() + offset,
                 prompt.begin() + offset + count - (last ? 1 : 0));
    loom.slot->deferred = last ? static_cast<std::int64_t>(prompt.back()) : -1;
    if (!input.empty())
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), input.data(),
                                                input.size()),
                    "prefill");
    loom.slot->position += count;
    return {.consumed_tokens = count, .decode_ready = last};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& loom = Require(state);
    std::lock_guard lock(model_->mutex());
    PrepareLocked(loom, sampler);
    if (const auto forced = sampler.ForcedToken())
      return {.token = *forced, .piece = PieceLocked(loom, *forced)};
    auto selection = SelectionLocked(loom, &sampler);
    // The Advance that follows chooses the next epoch's flag from this
    // sampler once the selected token is accepted.
    if (loom.hosted)
      loom.snapshot = std::make_unique<sampling::SamplerState>(sampler);
    return selection;
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    auto& loom = Require(state);
    std::lock_guard lock(model_->mutex());
    if (loom.hosted && loom.snapshot) {
      AdvanceHostedLocked(loom, token);
      return;
    }
    AdvanceLocked(loom, token);
  }

  void AdvanceBatch(
      std::span<const ext::TextRunnerAdvance> advances) const override {
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    std::vector<std::size_t> packed;
    std::vector<std::uint32_t> rows;
    for (std::size_t i = 0; i < advances.size(); ++i) {
      try {
        auto& loom = Require(advances[i].state.get());
        if (loom.hosted && loom.snapshot) {
          AdvanceHostedLocked(loom, advances[i].token);
          continue;
        }
        if (loom.slot->deferred >= 0 ||
            static_cast<TextRunnerToken>(loom_text_embed_row_token(
                embed, loom.row())) != advances[i].token) {
          AdvanceLocked(loom, advances[i].token);
          continue;
        }
        EnsureExtentLocked(loom, 2);
        packed.push_back(i);
        rows.push_back(loom.row());
      } catch (...) {
        if (advances[i].failure == nullptr)
          throw;
        *advances[i].failure = std::current_exception();
      }
    }
    if (rows.empty())
      return;
    try {
      model_->Check(
          loom_text_embed_decode_batch(embed, rows.data(), rows.size()),
          "decode");
      for (const auto i : packed)
        CommitLocked(Require(advances[i].state.get()));
    } catch (...) {
      for (const auto i : packed) {
        if (advances[i].failure == nullptr)
          throw;
        *advances[i].failure = std::current_exception();
      }
    }
  }

  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    const ext::TextRunnerDecode decode{state, max_tokens, sampler};
    auto steps = DecodeBatch(std::span(&decode, 1));
    if (steps.front().failure)
      std::rethrow_exception(steps.front().failure);
    return std::move(steps.front());
  }

  // One packed epoch for every session Gufo batches; drafted when the
  // package has a drafter. After the epoch each row's next selection is
  // known, so a response that ends here releases its withheld output (tool
  // calls) with its last token. Requests that sample on the host (logits)
  // run their own undrafted epoch with the host's token.
  [[nodiscard]] std::vector<ext::TextDecodeStep> DecodeBatch(
      std::span<const ext::TextRunnerDecode> decodes) const override {
    std::vector<ext::TextDecodeStep> steps(decodes.size());
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    const std::uint32_t depth = model_->info().draft_depth;
    std::vector<std::size_t> active;
    std::vector<std::uint32_t> rows, limits;
    // Rows whose token the host chose: index, token, the request's sampler
    // after accepting it.
    struct Hosted {
      std::size_t index;
      std::int32_t token;
      std::unique_ptr<sampling::SamplerState> probe;
    };
    std::vector<Hosted> hosted;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      try {
        auto& loom = Require(decodes[i].state.get());
        const std::size_t remaining = decodes[i].max_tokens;
        if (remaining == 0)
          continue;
        auto& sampler = decodes[i].sampler.get();
        PrepareLocked(loom, sampler);
        if (CloseReasoningLocked(loom, steps[i], remaining))
          continue;
        if (const auto forced = sampler.ForcedToken()) {
          // Gufo's reasoning budget forces this token in place of the
          // row's own selection.
          steps[i].selections.push_back(
              {.token = *forced, .piece = PieceLocked(loom, *forced)});
          AdvanceLocked(loom, *forced);
          FinishLocked(loom, steps[i], remaining);
          continue;
        }
        auto selection = SelectionLocked(loom, &sampler);
        if (selection.stop) {
          steps[i].stop = true;
          steps[i].terminal_logprobs = std::move(selection.logprobs);
          continue;
        }
        if (loom.hosted) {
          // The epoch that consumes this token leaves the logits of the
          // next selection when the host will need them.
          EnsureExtentLocked(loom, 2);
          auto probe = std::make_unique<sampling::SamplerState>(sampler);
          probe->Accept(static_cast<sampling::TokenId>(selection.token));
          loom.flag = NeedLocked(loom, sampler.config(), *probe);
          BindLocked(loom, sampler.config());
          hosted.push_back({i, static_cast<std::int32_t>(selection.token),
                            std::move(probe)});
          steps[i].selections.push_back(std::move(selection));
          continue;
        }
        const std::uint32_t limit =
            depth == 0 || remaining < 2
                ? 0
                : static_cast<std::uint32_t>(
                      std::min<std::size_t>(depth + 1, remaining));
        EnsureExtentLocked(loom, limit ? depth + 2 : 2);
        steps[i].selections.push_back(std::move(selection));
        active.push_back(i);
        rows.push_back(loom.row());
        limits.push_back(limit);
      } catch (...) {
        steps[i].failure = std::current_exception();
      }
    }
    if (!hosted.empty()) {
      try {
        std::vector<std::uint32_t> hosted_rows;
        std::vector<std::int32_t> hosted_tokens;
        for (const auto& h : hosted) {
          hosted_rows.push_back(Require(decodes[h.index].state.get()).row());
          hosted_tokens.push_back(h.token);
        }
        model_->Check(loom_text_embed_decode_batch_tokens(
                          embed, hosted_rows.data(), hosted_rows.size(),
                          hosted_tokens.data()),
                      "decode");
        for (auto& h : hosted) {
          auto& loom = Require(decodes[h.index].state.get());
          CommitLocked(loom);
          if (loom.flag)
            SampleNextLocked(loom, *h.probe);
          else
            loom.next.reset();
          FinishLocked(loom, steps[h.index], decodes[h.index].max_tokens);
        }
      } catch (...) {
        const auto failure = std::current_exception();
        for (const auto& h : hosted)
          steps[h.index].failure = failure;
      }
    }
    if (active.empty())
      return steps;
    try {
      const bool drafted =
          std::any_of(limits.begin(), limits.end(), [](auto l) { return l; });
      if (!drafted) {
        model_->Check(
            loom_text_embed_decode_batch(embed, rows.data(), rows.size()),
            "decode");
        for (std::size_t k = 0; k < active.size(); ++k) {
          auto& loom = Require(decodes[active[k]].state.get());
          CommitLocked(loom);
          FinishLocked(loom, steps[active[k]], decodes[active[k]].max_tokens);
        }
        return steps;
      }
      std::vector<std::int32_t> tokens(16 * rows.size());
      std::vector<std::uint32_t> counts(rows.size()), consumed(rows.size());
      model_->Check(loom_text_embed_verify_batch(
                        embed, rows.data(), rows.size(), limits.data(),
                        tokens.data(), counts.data(), consumed.data()),
                    "verify");
      for (std::size_t k = 0; k < active.size(); ++k) {
        auto& step = steps[active[k]];
        auto& loom = Require(decodes[active[k]].state.get());
        CommitLocked(loom);
        // Every output but the last entered state; the last stays pending.
        for (std::uint32_t j = 0; j + 1 < counts[k]; ++j) {
          const auto token = static_cast<TextRunnerToken>(tokens[16 * k + j]);
          step.selections.push_back(
              {.token = token, .piece = PieceLocked(loom, token)});
          CommitLocked(loom);
        }
        if (limits[k]) {
          step.draft_rounds = 1;
          step.draft_tokens = depth;
          step.draft_accepted_tokens = counts[k] ? counts[k] - 1 : 0;
        }
        FinishLocked(loom, step, decodes[active[k]].max_tokens);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto i : active)
        steps[i].failure = failure;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    return dynamic_cast<const LoomState&>(state).position();
  }

private:
  std::string DecodeLocked(const std::vector<std::int32_t>& tokens) const {
    if (tokens.empty())
      return {};
    std::string text(16 * tokens.size() + 64, '\0');
    for (int attempt = 0;; ++attempt) {
      std::size_t size = 0;
      if (loom_text_embed_detokenize(model_->embed(), tokens.data(),
                                     tokens.size(), text.data(), text.size(),
                                     &size) == 0) {
        text.resize(size);
        return text;
      }
      if (attempt == 6 ||
          std::string_view(loom_text_embed_error(model_->embed()))
                  .find("RESOURCE_EXHAUSTED") == std::string_view::npos)
        model_->Check(1, "detokenize");
      text.resize(text.size() * 4);
    }
  }

  static std::string Trimmed(std::string text) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && space(text.back()))
      text.pop_back();
    std::size_t at = 0;
    while (at < text.size() && space(text[at]))
      ++at;
    return text.substr(at);
  }

  // Releases the request's grammar once: the package samples on device, so
  // Gufo's token masks cannot apply; the package's parser reads the output.
  static void Unconstrain(LoomState& loom, sampling::SamplerState& sampler) {
    if (!loom.unconstrained) {
      sampler = sampler.WithoutConstraint();
      loom.unconstrained = true;
    }
  }

  void ResetRowLocked(LoomState& loom) const {
    loom_text_embed_row_end(model_->embed(), loom.row());
    model_->Check(loom_text_embed_row_reset(model_->embed(), loom.row()),
                  "reset");
    loom.slot->position = 0;
    loom.slot->deferred = -1;
    loom.slot->stale = false;
    ++loom.slot->generation;
    loom.bound.clear();
    loom.ResetHost();
  }

  // --------------------------------------------------- host sampling ---

  [[nodiscard]] bool CanConstrain(const LoomState& loom) const {
    return model_->logits().enabled && loom.context != nullptr &&
           loom.context->constraint_ok;
  }

  // Controls the device sampler does not implement: they need the logit row.
  [[nodiscard]] static bool StaticNeed(const sampling::SamplingConfig& config) {
    return config.logprobs.has_value() || config.penalties_enabled() ||
           config.min_p > 0.0F;
  }

  // Whether this request samples on the host when it needs to: logits are on
  // and it has a grammar this package can take, or controls the device lacks.
  bool HostedLocked(LoomState& loom,
                    const sampling::SamplingConfig& config) const {
    if (!loom.hosted_known) {
      loom.hosted_known = true;
      loom.hosted = model_->logits().enabled &&
                    ((config.constraint != nullptr && CanConstrain(loom)) ||
                     StaticNeed(config));
    }
    return loom.hosted;
  }

  // Whether the selection after `state` (the request's sampler with every
  // consumed token accepted) must be chosen from the logit row. A lazy tool
  // grammar admits everything until a call starts, so those requests run on
  // the device until it restricts, then stay on the host.
  bool NeedLocked(LoomState& loom, const sampling::SamplingConfig& config,
                  sampling::SamplerState& state) const {
    if (loom.sticky)
      return true;
    if (StaticNeed(config)) {
      loom.sticky = true;
      return true;
    }
    if (config.constraint == nullptr)
      return false;
    // A lazy tool grammar admits the whole vocabulary until the call marker
    // is complete; computing its mask (a walk over the vocabulary) every step
    // would tax every tool request, so look at the text first.
    if (loom.context != nullptr && loom.context->lazy_tools &&
        loom.tail.find(loom.context->lazy_open) == std::string::npos)
      return false;
    if (state.NeedsConstraintMask())
      loom.sticky = true;
    return loom.sticky;
  }

  // Chooses the row's pending selection from its logit row with the request's
  // sampler state (mask, penalties, min-p, seed). The draw's RNG state is kept
  // so the real sampler, which has not drawn, continues the same stream.
  void SampleNextLocked(LoomState& loom, sampling::SamplerState& state) const {
    const std::size_t vocabulary = VocabularyLocked();
    const std::size_t width = model_->logits().width;
    loom.logits.assign(std::max(width, vocabulary), -3e38F);
    std::size_t count = 0;
    model_->Check(loom_text_embed_row_logits(model_->embed(), loom.row(),
                                             loom.logits.data(), width, &count),
                  "logits");
    // Ids the head has no row for stay at the padding value; ids past the
    // tokenizer are not tokens.
    const std::span<const float> row(loom.logits.data(), vocabulary);
    const auto token = state.Sample(row);
    LoomState::HostNext next;
    next.token = token;
    next.stop = loom.stop_at_eos() &&
                loom_text_embed_is_stop_token(model_->embed(),
                                              static_cast<std::int32_t>(token));
    next.rng = state.rng_state();
    next.logprobs =
        sampling::CollectLogprobs(row, token, state.config().logprobs);
    loom.next = std::move(next);
  }

  std::size_t VocabularyLocked() const {
    return static_cast<std::size_t>(
        loom_text_embed_vocabulary_size(model_->embed()));
  }

  // Binds the request's sampling (with the row's logits flag).
  void BindLocked(LoomState& loom,
                  const sampling::SamplingConfig& config) const {
    auto* embed = model_->embed();
    std::string body = SamplingBody(model_->info().model_id, config,
                                    loom.flag && model_->logits().enabled);
    if (body != loom.bound) {
      std::size_t count = 0;
      loom_text_embed_request_t* request = nullptr;
      model_->Check(
          loom_text_embed_chat_prepare(embed, body.data(), body.size(), nullptr,
                                       0, &count, &request),
          "sampling options");
      const int status = loom_text_embed_row_bind(embed, loom.row(), request);
      loom_text_embed_request_free(request);
      model_->Check(status, "bind sampling");
      loom.bound = std::move(body);
    }
  }

  // Binds the request's sampling and consumes a deferred prompt token.
  void PrepareLocked(LoomState& loom, sampling::SamplerState& sampler) const {
    auto* embed = model_->embed();
    // Without the logits capability (or a grammar this package's tokens can
    // take) the package samples on device and its parser reads the output.
    if (!CanConstrain(loom))
      Unconstrain(loom, sampler);
    EnsureExtentLocked(loom, 2);
    const bool hosted = HostedLocked(loom, sampler.config());
    const bool consuming = loom.slot->deferred >= 0;
    // The epoch that consumes the prompt's last token makes the first
    // selection: it leaves its logits when the host will pick it.
    if (hosted && consuming)
      loom.flag = NeedLocked(loom, sampler.config(), sampler);
    BindLocked(loom, sampler.config());
    if (consuming) {
      const std::int32_t token = static_cast<std::int32_t>(loom.slot->deferred);
      loom.slot->deferred = -1;
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), &token, 1),
                    "prefill");
      if (hosted) {
        if (loom.flag)
          SampleNextLocked(loom, sampler);
        else
          loom.next.reset();
      }
    }
  }

  // The row's pending selection: the host's when it chose one from the
  // logits (the sampler then continues the draw's RNG stream), else the
  // device's.
  ext::TextDecodeSelection SelectionLocked(
      LoomState& loom, sampling::SamplerState* sampler = nullptr) const {
    auto* embed = model_->embed();
    if (loom.next) {
      auto next = std::move(*loom.next);
      loom.next.reset();
      if (sampler != nullptr)
        sampler->SetRngState(next.rng);
      if (next.stop)
        return {.stop = true,
                .token = next.token,
                .logprobs = std::move(next.logprobs)};
      return {.token = next.token,
              .piece = PieceLocked(loom, next.token),
              .logprobs = std::move(next.logprobs)};
    }
    const auto token = static_cast<TextRunnerToken>(
        loom_text_embed_row_token(embed, loom.row()));
    if (loom.stop_at_eos() && loom_text_embed_row_is_eos(embed, loom.row()))
      return {.stop = true, .token = token};
    return {.token = token, .piece = PieceLocked(loom, token)};
  }

  void AdvanceLocked(LoomState& loom, TextRunnerToken token) const {
    auto* embed = model_->embed();
    // A forced token (reasoning budget) replaces whatever the host chose; the
    // next selection comes from the device.
    loom.next.reset();
    loom.snapshot.reset();
    EnsureExtentLocked(loom, 2);
    if (loom.slot->deferred < 0 &&
        static_cast<TextRunnerToken>(
            loom_text_embed_row_token(embed, loom.row())) == token) {
      model_->Check(loom_text_embed_row_decode(embed, loom.row()), "decode");
    } else {
      // A token other than the pending selection replaces it, after any
      // prompt token still deferred.
      std::vector<std::int32_t> input;
      if (loom.slot->deferred >= 0)
        input.push_back(static_cast<std::int32_t>(loom.slot->deferred));
      loom.slot->deferred = -1;
      input.push_back(static_cast<std::int32_t>(token));
      model_->Check(loom_text_embed_row_prefill(embed, loom.row(), input.data(),
                                                input.size()),
                    "advance");
    }
    CommitLocked(loom);
  }

  // Consumes the token the host chose (SelectNext/Advance flow): the epoch
  // leaves the next logit row when the host needs it, and the host picks the
  // next selection right away, as it would after a device selection.
  void AdvanceHostedLocked(LoomState& loom, TextRunnerToken token) const {
    auto* embed = model_->embed();
    EnsureExtentLocked(loom, 2);
    sampling::SamplerState probe(*loom.snapshot);
    loom.snapshot.reset();
    probe.Accept(static_cast<sampling::TokenId>(token));
    loom.flag = NeedLocked(loom, probe.config(), probe);
    BindLocked(loom, probe.config());
    const std::uint32_t row = loom.row();
    if (loom.slot->deferred < 0) {
      const std::int32_t input = static_cast<std::int32_t>(token);
      model_->Check(loom_text_embed_decode_batch_tokens(embed, &row, 1, &input),
                    "decode");
    } else {
      const std::int32_t input[2] = {
          static_cast<std::int32_t>(loom.slot->deferred),
          static_cast<std::int32_t>(token)};
      loom.slot->deferred = -1;
      model_->Check(loom_text_embed_row_prefill(embed, row, input, 2),
                    "advance");
    }
    CommitLocked(loom);
    if (loom.flag)
      SampleNextLocked(loom, probe);
    else
      loom.next.reset();
  }

  void CommitLocked(LoomState& loom) const {
    ++loom.generated;
    ++loom.slot->position;
  }

  // Text this token settles: decoded by the row's streaming detokenizer,
  // then read by the request's output reader into canonical markup.
  std::string PieceLocked(LoomState& loom, std::int32_t token) const {
    const std::pair<std::size_t, std::int32_t> key{loom.generated, token};
    if (loom.piece_key == key)
      return loom.piece;
    if (loom.piece_key && loom.piece_key->first == key.first)
      throw std::logic_error("loom selection changed before it was consumed");
    std::string raw(256, '\0');
    std::size_t size = 0;
    model_->Check(loom_text_embed_decoder_feed(loom.decoder, &token, 1,
                                               raw.data(), raw.size(), &size),
                  "decode");
    raw.resize(size);
    loom.tail += raw;
    if (loom.tail.size() > 160)
      loom.tail.erase(0, loom.tail.size() - 160);
    loom.piece_key = key;
    loom.piece = OutputLocked(loom, raw, false);
    // The prompt opened reasoning and the model closed it at once: the
    // output reader consumed the close as framing, so restore it for
    // Gufo's parser, which reads this response as reasoning until a close.
    if (!loom.emitted && !loom.piece.empty()) {
      loom.emitted = true;
      if (loom.reasoning && !loom.piece.starts_with("<think>") &&
          loom.piece.find("</think>") == std::string::npos)
        loom.piece.insert(0, "</think>");
    }
    loom.produced = true;
    if (close_tokens_.empty() && loom.context != nullptr)
      LearnCloseLocked(loom.context->request);
    // Track reasoning from the canonical markers the output reader emits.
    const auto open = loom.piece.rfind("<think>");
    const auto close = loom.piece.rfind("</think>");
    if (open != std::string::npos || close != std::string::npos)
      loom.reasoning = open != std::string::npos &&
                       (close == std::string::npos || open > close);
    if (loom.reasoning)
      ++loom.reasoning_tokens;
    return loom.piece;
  }

  std::string OutputLocked(LoomState& loom, std::string_view raw,
                           bool final) const {
    if (loom.context == nullptr || loom.flushed)
      return std::string(raw);
    std::string text(raw.size() + kTextBuffer, '\0');
    std::size_t size = 0;
    model_->Check(loom_text_embed_output_feed(loom.context->request, raw.data(),
                                              raw.size(), final, text.data(),
                                              text.size(), &size),
                  "read output");
    text.resize(size);
    return text;
  }

  // Ends the response when the next selection is EOS or the token budget is
  // spent: releases withheld text and the parsed tool calls.
  void FinishLocked(LoomState& loom, ext::TextDecodeStep& step,
                    std::size_t remaining) const {
    if (loom.context == nullptr || loom.flushed || step.selections.empty())
      return;
    auto* embed = model_->embed();
    const bool eos = loom.next
                         ? loom.next->stop
                         : loom.stop_at_eos() &&
                               loom_text_embed_row_is_eos(embed, loom.row());
    const bool spent = step.selections.size() >= remaining;
    if (!eos && !spent)
      return;
    std::string tail = OutputLocked(loom, {}, true);
    std::string json(kTextBuffer, '\0');
    std::size_t size = 0;
    if (loom_text_embed_output_tools(loom.context->request, eos ? 0 : 1,
                                     json.data(), json.size(), &size) == 0) {
      json.resize(size);
      tail += ToolMarkup(json);
    }
    loom.flushed = true;
    step.selections.back().piece += tail;
  }

  // Grows the row's reservation before a step that consumes `step` tokens.
  void EnsureExtentLocked(LoomState& loom, std::uint64_t step) const {
    auto* embed = model_->embed();
    const std::uint64_t position =
        loom_text_embed_row_position(embed, loom.row()) +
        (loom.slot->deferred >= 0 ? 1 : 0);
    const std::uint64_t context = max_context_;
    if (position + step <= loom_text_embed_row_extent(embed, loom.row()))
      return;
    const std::uint64_t extent =
        std::min<std::uint64_t>(context, position + step + kExtentSlack);
    if (position + step > extent)
      throw std::length_error("loom row reached the context capacity");
    if (!ReserveLocked([&] {
          return loom_text_embed_row_extend(embed, loom.row(), extent);
        }))
      throw cache::ResourceExhausted();
    // The reservation restarted the request; bind its sampling again.
    loom.bound.clear();
  }

  // The package's reasoning close marker: render an earlier turn with known
  // reasoning and answer, and read what the chat program puts between them.
  std::vector<std::int32_t> DiscoverReasoningClose() const {
    static constexpr std::string_view kThought = "Q7reasoning9";
    static constexpr std::string_view kAnswer = "Q7answer9";
    for (const bool trailing_user : {false, true}) {
      ext::ChatRequest request;
      request.messages.emplace_back(ext::ChatRole::kUser, "Hi.");
      request.messages.emplace_back(ext::ChatRole::kAssistant,
                                    std::string(kAnswer), std::string(),
                                    std::string(kThought));
      if (trailing_user)
        request.messages.emplace_back(ext::ChatRole::kUser, "Thanks.");
      request.reasoning.enabled = true;
      try {
        const auto body = ChatBody(model_->info().model_id, request);
        std::lock_guard lock(model_->mutex());
        std::vector<std::int32_t> tokens(model_->info().context_capacity);
        std::size_t count = 0;
        if (loom_text_embed_chat_prepare(model_->embed(), body.data(),
                                         body.size(), tokens.data(),
                                         tokens.size(), &count, nullptr) != 0)
          continue;
        tokens.resize(count);
        const std::string text = DecodeLocked(tokens);
        const auto thought = text.find(kThought);
        const auto answer = text.find(kAnswer);
        if (thought == std::string::npos || answer == std::string::npos ||
            answer <= thought + kThought.size())
          continue;
        const std::string close = text.substr(
            thought + kThought.size(), answer - thought - kThought.size());
        std::vector<std::int32_t> close_tokens(close.size() + 16);
        std::size_t close_count = 0;
        if (close.find_first_not_of(" \t\r\n") == std::string::npos ||
            loom_text_embed_tokenize(model_->embed(), close.data(),
                                     close.size(), 1, close_tokens.data(),
                                     close_tokens.size(), &close_count) != 0)
          continue;
        close_tokens.resize(close_count);
        return close_tokens;
      } catch (const std::exception&) {
      }
    }
    // Templates that drop earlier reasoning: compare the generation prompt
    // with thinking on (it ends open) and off (it adds the close).
    try {
      std::string rendered[2];
      for (int on = 0; on < 2; ++on) {
        ext::ChatRequest request;
        request.messages.emplace_back(ext::ChatRole::kUser, "Hi.");
        request.reasoning.enabled = on == 1;
        const auto body = ChatBody(model_->info().model_id, request);
        std::lock_guard lock(model_->mutex());
        std::vector<std::int32_t> tokens(model_->info().context_capacity);
        std::size_t count = 0;
        if (loom_text_embed_chat_prepare(model_->embed(), body.data(),
                                         body.size(), tokens.data(),
                                         tokens.size(), &count, nullptr) != 0)
          return {};
        tokens.resize(count);
        rendered[on] = DecodeLocked(tokens);
      }
      const std::string& off = rendered[0];
      const std::string& on = rendered[1];
      if (on.size() >= off.size() || off.compare(0, on.size(), on) != 0)
        return {};
      const std::string close = off.substr(on.size());
      if (close.find("</") == std::string::npos &&
          close.find("<|") == std::string::npos)
        return {};
      std::lock_guard lock(model_->mutex());
      std::vector<std::int32_t> close_tokens(close.size() + 16);
      std::size_t close_count = 0;
      if (loom_text_embed_tokenize(model_->embed(), close.data(), close.size(),
                                   1, close_tokens.data(), close_tokens.size(),
                                   &close_count) != 0)
        return {};
      close_tokens.resize(close_count);
      return close_tokens;
    } catch (const std::exception&) {
    }
    return {};
  }

  // Takes the model's own reasoning close from a response that reached its
  // answer: the framing the package's output reader consumed in between.
  void LearnCloseLocked(loom_text_embed_request_t* request) const {
    if (request == nullptr)
      return;
    std::string close(kTextBuffer, '\0');
    std::size_t size = 0;
    if (loom_text_embed_output_close(request, close.data(), close.size(),
                                     &size) != 0 ||
        size == 0)
      return;
    close.resize(size);
    std::vector<std::int32_t> tokens(close.size() + 16);
    std::size_t count = 0;
    if (loom_text_embed_tokenize(model_->embed(), close.data(), close.size(), 1,
                                 tokens.data(), tokens.size(), &count) != 0 ||
        count == 0)
      return;
    tokens.resize(count);
    close_tokens_ = std::move(tokens);
  }

  // Packages whose templates show neither a reasoning close nor a way to turn
  // reasoning off (they always reason): learn the close from one short greedy
  // generation at load.
  void ProbeReasoningClose() {
    ext::ChatRequest request;
    request.messages.emplace_back(ext::ChatRole::kUser,
                                  "What is 2+2? Answer briefly.");
    request.reasoning.enabled = true;
    std::string body;
    try {
      body = ChatBody(model_->info().model_id, request);
    } catch (const std::exception&) {
      return;
    }
    std::lock_guard lock(model_->mutex());
    auto* embed = model_->embed();
    std::vector<std::int32_t> tokens(model_->info().context_capacity);
    std::size_t count = 0;
    loom_text_embed_request_t* handle = nullptr;
    if (loom_text_embed_chat_prepare(embed, body.data(), body.size(),
                                     tokens.data(), tokens.size(), &count,
                                     &handle) != 0)
      return;
    loom_text_embed_decoder_t* decoder = nullptr;
    constexpr std::size_t kProbeTokens = 768;
    const std::uint64_t extent = std::min<std::uint64_t>(
        model_->info().context_capacity, count + kProbeTokens + 1);
    if (loom_text_embed_decoder_open(embed, &decoder) == 0 &&
        count + kProbeTokens < model_->info().context_capacity &&
        loom_text_embed_row_begin(embed, 0, handle, extent, 1) == 0) {
      if (loom_text_embed_row_prefill(embed, 0, tokens.data(), count) == 0) {
        std::string raw(256, '\0'), text(256 + kTextBuffer, '\0');
        for (std::size_t i = 0; i < kProbeTokens && close_tokens_.empty();
             ++i) {
          if (loom_text_embed_row_is_eos(embed, 0))
            break;
          const std::int32_t token = loom_text_embed_row_token(embed, 0);
          std::size_t size = 0, out = 0;
          if (loom_text_embed_decoder_feed(decoder, &token, 1, raw.data(),
                                           raw.size(), &size) != 0 ||
              loom_text_embed_output_feed(handle, raw.data(), size, 0,
                                          text.data(), text.size(), &out) != 0)
            break;
          LearnCloseLocked(handle);
          if (!close_tokens_.empty() ||
              loom_text_embed_row_decode(embed, 0) != 0)
            break;
        }
      }
      loom_text_embed_row_end(embed, 0);
      loom_text_embed_row_reset(embed, 0);
    }
    loom_text_embed_decoder_free(decoder);
    loom_text_embed_request_free(handle);
  }

  // Forces the reasoning close once the request's budget is spent.
  bool CloseReasoningLocked(LoomState& loom, ext::TextDecodeStep& step,
                            std::size_t remaining) const {
    if (loom.context == nullptr || !loom.context->reasoning_budget ||
        loom.budget_closed || !loom.reasoning ||
        loom.reasoning_tokens < *loom.context->reasoning_budget)
      return false;
    loom.budget_closed = true;
    for (const auto token : close_tokens_) {
      if (step.selections.size() >= remaining)
        break;
      const auto forced = static_cast<TextRunnerToken>(token);
      step.selections.push_back(
          {.token = forced, .piece = PieceLocked(loom, forced)});
      AdvanceLocked(loom, forced);
    }
    FinishLocked(loom, step, remaining);
    return true;
  }

  std::shared_ptr<LoomModel> model_;
  // --context, at most the package's own context.
  std::uint32_t max_context_;
  std::uint32_t rows_;
  std::uint32_t width_;
  // Guarded by the model mutex; learned from output when templates hide it.
  mutable std::vector<std::int32_t> close_tokens_;
  std::size_t ram_cache_bytes_{0};
  std::optional<ImageLayout> images_;
  loom_text_embed_memory_t memory_{};
  // Logits load only: the model's own call syntax and reasoning markers, and
  // whether grammars may run around its reasoning at all.
  NativeDialect native_;
  bool reasoning_ok_{false};
  // Guarded by the model mutex: whether recently prepared prompts end inside
  // reasoning, by the SHA-256 of their chat body (InitialOutputState).
  static constexpr std::size_t kRecentPrompts = 32;
  mutable std::vector<std::pair<BodyDigest, bool>> recent_reasoning_;

  // Device bytes of the KV pool's pages.
  [[nodiscard]] std::uint64_t KvBytes() const {
    return memory_.block_size
               ? memory_.state_tokens / memory_.block_size * memory_.page_bytes
               : 0;
  }

  // Host bytes the RAM tier may hold as images: --ram-cache, or room for a
  // few full-context images.
  [[nodiscard]] std::size_t ImageCacheBytes() const {
    if (ram_cache_bytes_ > 0)
      return ram_cache_bytes_;
    const std::size_t pages =
        (model_->info().context_capacity + images_->block_size - 1) /
        images_->block_size;
    return std::clamp<std::size_t>(
        4 * (images_->private_bytes() + pages * images_->page_bytes),
        std::size_t{1} << 30, std::size_t{16} << 30);
  }
};

std::shared_ptr<ext::TextModelRunner> CreateLoomRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  try {
    auto model = std::make_shared<LoomModel>(options);
    return std::make_shared<LoomRunner>(
        std::move(model), options.max_context,
        !options.disk_cache.directory.empty(),
        options.ram_cache.enabled ? options.ram_cache.capacity_bytes : 0);
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = exception.what();
    return nullptr;
  }
}

}  // namespace
}  // namespace gufo::loom

void loom_text_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_loom::kProvider_loom_text);
  registry.AddTextProvider({
      .id = "loom-text",
      .probe = &gufo::loom::ProbeLoomText,
      .create = &gufo::loom::CreateLoomRunner,
  });
  // Every loom-fun package installs by its repository.
  registry.AddModelSource({
      .owner = std::string(gufo::loom::source::kOwner),
      .provider = "loom-text",
      .resolve =
          [](std::string_view repo, const gufo::ext::ModelSourceHub& hub) {
            return gufo::loom::source::Resolve(repo, hub,
                                               gufo::loom::kRuntimeSdk);
          },
  });
}

void loom_register(gufo::ext::Registry& registry) {
  (void)registry;
}
