#include "src/cli/serve/lfm3_runner.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/models/lfm3/chat_template.hpp"
#include "src/models/lfm3_vl/prompt.hpp"
namespace gufo::server {
namespace {
struct ImageContext final : TextPromptContext {
  models::lfm3_vl::Prompt prompt;
};
class State final : public TextRunnerState {
public:
  std::unique_ptr<models::lfm3::Session> session;
  std::vector<float> logits;
  bool logits_ready{};
  std::shared_ptr<const ImageContext> input;
  std::vector<std::vector<float>> image_rows;
  void Invalidate() noexcept override {
    (void)session->Reset();
    logits_ready = false;
  }
};
State& Get(TextRunnerState& state) {
  return dynamic_cast<State&>(state);
}
const State& Get(const TextRunnerState& state) {
  return dynamic_cast<const State&>(state);
}
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
class Lfm3Snapshot final : public TextRunnerSnapshot {
public:
  Lfm3Snapshot(std::shared_ptr<models::lfm3::Model> owner,
               std::vector<std::uint8_t> payload)
      : owner(std::move(owner)), payload(std::move(payload)) {}
  std::size_t PayloadBytes() const noexcept override { return payload.size(); }
  std::shared_ptr<models::lfm3::Model> owner;
  std::vector<std::uint8_t> payload;
};
class Runner final : public TextModelRunner {
public:
  Runner(std::shared_ptr<models::lfm3::Model> model, std::uint32_t context,
         std::string artifact_fingerprint,
         std::shared_ptr<models::lfm3_vl::Vision> vision)
      : model_(std::move(model)),
        vision_(std::move(vision)),
        context_(context) {
    if (!artifact_fingerprint.empty()) {
      if (artifact_fingerprint.size() != 64 ||
          artifact_fingerprint.find_first_not_of("0123456789abcdef") !=
              std::string::npos)
        throw std::invalid_argument("invalid LFM3 artifact fingerprint");
      auto identity =
          "schema=gufo-text-continuation-v2\nmodel_kind=lfm3\nartifact_id=" +
          std::string(core::kGgufIdentityScheme) + ':' + artifact_fingerprint +
          "\nstate_abi=lfm3-fp32-v2\npayload_layout=lfm3-fp32-compact-v1\n" +
          "tokenizer=lfm2-byte-bpe-v2-gguf-controls\nchat_template=lfm3-native-"
          "v2\n" +
          "numerics=fixed-point-three-digit-v1-wave32-attention-v1\ncontext_"
          "tokens=" +
          std::to_string(context_) +
          "\nposition_policy=absolute-ring-v1\nadapters=none\n";
      if (vision_)
        identity += "vision=lfm3-vl-image-v1:" + vision_->Identity() + "\n";
      persistence_ = TextRunnerPersistenceDescriptor{
          {identity.begin(), identity.end()}, 1};
    }
  }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = vision_ ? "LFM3-VL-2B" : "LFM3-1.7B",
            .state_abi = "lfm3-fp32-v2",
            .max_context = context_,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = true,
                             .incremental_text_is_exact = true,
                             .prefix_reuse = true,
                             .partial_prefill_retention = true},
            .persistence = persistence_};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
            .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes()};
  }
  std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        VocabularySize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              std::string(model_->tokenizer().Piece(id)),
              model_->tokenizer().IsStop(id)};
        });
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "</think>", {"</think>", "<tool_call>"}};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{TextExecutionPlanKind::kSerial, 1}};
    for (std::size_t width = 2; width <= models::lfm3::Session::kMaxBatch;
         ++width)
      plans.push_back({TextExecutionPlanKind::kBatched, width});
    return plans;
  }
  std::uint32_t VocabularySize() const override {
    return model_->config().vocab;
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    return model_->tokenizer().Encode(text, false);
  }
  std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }
  std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    if (vision_)
      return std::nullopt;
    return models::lfm3::RenderChat(request.messages, Tools(request),
                                    request.reasoning);
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    if (vision_)
      return models::lfm3_vl::Prepare(model_->tokenizer(), request.messages,
                                      Tools(request), request.reasoning,
                                      context_)
          .tokens;
    return models::lfm3::EncodeChat(model_->tokenizer(), request.messages,
                                    Tools(request), request.reasoning);
  }
  std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (vision_) {
      auto context = std::make_shared<ImageContext>();
      context->prompt =
          models::lfm3_vl::Prepare(model_->tokenizer(), request.messages,
                                   Tools(request), request.reasoning, context_);
      crypto::Sha256Hasher hash;
      const auto& identity = vision_->Identity();
      hash.Update({reinterpret_cast<const std::uint8_t*>(identity.data()),
                   identity.size()});
      for (const auto& image : context->prompt.images) {
        context->cache_prefixes.push_back(
            {image.offset, context->cache_identity});
        const std::uint64_t geometry[] = {image.offset,
                                          std::uint64_t(image.picture.height),
                                          std::uint64_t(image.picture.width)};
        hash.Update({reinterpret_cast<const std::uint8_t*>(geometry),
                     sizeof(geometry)});
        const auto& patches = image.picture.patches;
        hash.Update({reinterpret_cast<const std::uint8_t*>(patches.data()),
                     patches.size() * sizeof(float)});
        const auto digest = hash.Digest();
        context->cache_identity.assign(digest.begin(), digest.end());
      }
      const auto stable = context->prompt.stable_prefix;
      auto tokens = context->prompt.tokens;
      return TextPreparedPrompt{std::move(tokens), std::move(context), stable};
    }
    auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    // Keep the history before the mutable assistant opening. A client that
    // discards a cancelled reply appends a new user turn at this boundary.
    const auto history =
        models::lfm3::EncodeChat(model_->tokenizer(), request.messages,
                                 Tools(request), request.reasoning, false);
    if (history.size() > tokens->size() ||
        !std::equal(history.begin(), history.end(), tokens->begin()))
      throw std::logic_error("LFM3 history is not a prompt prefix");
    return TextPreparedPrompt{std::move(*tokens), {}, history.size()};
  }
  std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (vision_)
      return models::lfm3_vl::Prepare(model_->tokenizer(), request.messages,
                                      Tools(request), request.reasoning,
                                      model_->config().context)
          .tokens.size();
    return TextModelRunner::CountPromptTokens(request);
  }
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& s = Get(state);
    s.input = std::dynamic_pointer_cast<const ImageContext>(context);
    if (context && (!vision_ || !s.input))
      throw std::invalid_argument("invalid LFM3-VL prompt context");
    s.image_rows.clear();
    if (s.input)
      s.image_rows.resize(s.input->prompt.images.size());
  }
  TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& r) const override {
    return models::lfm3::ThinkingEnabled(r.reasoning)
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    std::string error;
    auto s = std::make_unique<State>();
    s->session = model_->CreateSession(context_, &error);
    Require(bool(s->session), error);
    s->logits.resize(model_->config().vocab);
    return s;
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t max_input_tokens) const override {
    auto& s = Get(state);
    if (offset != s.session->Position() || offset >= prompt.size())
      throw std::logic_error("LFM3 prefill frontier mismatch");
    auto count =
        std::min<std::size_t>({models::lfm3::Session::kPrefillChunk,
                               max_input_tokens, prompt.size() - offset});
    if (!count)
      throw std::invalid_argument("LFM3 prefill budget must be positive");
    std::string error;
    bool visual = false;
    if (s.input)
      for (std::size_t i = 0; i < s.input->prompt.images.size(); ++i) {
        const auto& image = s.input->prompt.images[i];
        const auto rows =
            std::size_t(image.picture.height) * image.picture.width / 4;
        if (offset < image.offset) {
          count = std::min(count, image.offset - offset);
          break;
        }
        if (offset >= image.offset + rows)
          continue;
        if (s.image_rows[i].empty())
          s.image_rows[i] = vision_->Encode(image.picture);
        count = std::min(count, image.offset + rows - offset);
        const auto hidden = model_->config().hidden;
        Require(
            s.session->ForwardEmbeddings(
                std::span(s.image_rows[i])
                    .subspan((offset - image.offset) * hidden, count * hidden),
                false,
                offset + count == prompt.size() ? std::span(s.logits)
                                                : std::span<float>{},
                &error),
            error);
        visual = true;
        break;
      }
    if (!visual)
      Require(s.session->Forward(prompt.subspan(offset, count),
                                 offset + count == prompt.size()
                                     ? std::span(s.logits)
                                     : std::span<float>{},
                                 &error),
              error);
    s.logits_ready = offset + count == prompt.size();
    return {count, s.logits_ready};
  }
  TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = Get(state);
    if (!s.logits_ready) {
      std::string error;
      Require(s.session->ReadLogits(s.logits, &error), error);
      s.logits_ready = true;
    }
    auto token = sampler.Sample(s.logits);
    if (state.stop_at_eos() && model_->tokenizer().IsStop(token))
      return {.stop = true, .token = token};
    return {.stop = false,
            .token = token,
            .piece = std::string(model_->tokenizer().Piece(token))};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = Get(state);
    std::string error;
    Require(s.session->Forward(std::span(&token, 1), s.logits, &error), error);
    s.logits_ready = true;
  }
  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      TextModelRunner::AdvanceBatch(advances);
      return;
    }
    std::vector<models::lfm3::Session*> sessions;
    std::vector<std::uint32_t> tokens;
    std::vector<std::span<float>> outputs;
    for (const auto& advance : advances) {
      auto& state = Get(advance.state.get());
      sessions.push_back(state.session.get());
      tokens.push_back(advance.token);
      outputs.emplace_back(state.logits);
    }
    std::string error;
    Require(
        models::lfm3::Session::ForwardBatch(sessions, tokens, outputs, &error),
        error);
    for (const auto& advance : advances)
      Get(advance.state.get()).logits_ready = true;
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return Get(state).session->Position();
  }

  std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return Get(state).session->SnapshotBytes();
  }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    return std::make_unique<Lfm3Snapshot>(model_,
                                          Get(state).session->SaveSnapshot());
  }
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& value) const override {
    const auto& snapshot = CheckedSnapshot(value);
    RestorePersistentSnapshot(state, snapshot.payload);
  }
  std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& value) const override {
    return CheckedSnapshot(value).payload.size();
  }
  std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& value,
      std::span<std::uint8_t> destination) const override {
    const auto& payload = CheckedSnapshot(value).payload;
    if (payload.size() != destination.size())
      throw std::invalid_argument("invalid LFM3 snapshot destination");
    std::copy(payload.begin(), payload.end(), destination.begin());
    return payload.size();
  }
  void StreamPersistentSnapshot(const TextRunnerSnapshot& value,
                                const SnapshotSink& sink) const override {
    sink(CheckedSnapshot(value).payload);
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& s = Get(state);
    std::string error;
    Require(s.session->RestoreSnapshot(payload, &error), error);
    s.logits_ready = s.session->Position() != 0;
    if (s.logits_ready)
      Require(s.session->ReadLogits(s.logits, &error), error);
  }

private:
  const Lfm3Snapshot& CheckedSnapshot(
      const TextRunnerSnapshot& snapshot) const {
    const auto* value = dynamic_cast<const Lfm3Snapshot*>(&snapshot);
    if (!value || value->owner != model_)
      throw std::invalid_argument("LFM3 snapshot belongs to a different model");
    return *value;
  }
  static std::span<const tokenization::ChatTool> Tools(const ChatRequest& r) {
    return r.tool_choice == ChatRequest::ToolChoice::kNone
               ? std::span<const tokenization::ChatTool>{}
               : r.tools;
  }

  std::shared_ptr<models::lfm3::Model> model_;
  std::shared_ptr<models::lfm3_vl::Vision> vision_;
  std::uint32_t context_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};
}  // namespace
std::shared_ptr<TextModelRunner> CreateLfm3Runner(
    std::shared_ptr<models::lfm3::Model> model, std::uint32_t context,
    std::string artifact_fingerprint,
    std::shared_ptr<models::lfm3_vl::Vision> vision) {
  return std::make_shared<Runner>(std::move(model), context,
                                  std::move(artifact_fingerprint),
                                  std::move(vision));
}
}  // namespace gufo::server
