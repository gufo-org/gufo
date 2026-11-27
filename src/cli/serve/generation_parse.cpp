#include "src/cli/serve/generation_parse.hpp"

#include <array>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

#include "gufo/generation_parse.h"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/rust/callback.hpp"
#include "src/families/qwen/text/chat_template.hpp"

namespace gufo::server {

namespace {

// Client envelope syntax is removed only when it names a declared tool, or
// immediately echoes a call the parser accepted. Other XML stays content.
constexpr std::array<std::string_view, 1> kEnvelopeHeads{"<invoke name="};
constexpr std::array<std::string_view, 1> kEnvelopeParameters{
    "<parameter name="};
constexpr std::array<std::string_view, 2> kEnvelopeClosers{"</invoke>",
                                                           "</parameter>"};

GufoGenerationBytes Bytes(std::string_view text) noexcept {
  return {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}

std::string_view View(GufoGenerationBytes bytes) noexcept {
  return bytes.len == 0
             ? std::string_view{}
             : std::string_view(reinterpret_cast<const char*>(bytes.data),
                                bytes.len);
}

std::vector<GufoGenerationBytes> List(
    std::span<const std::string_view> values) {
  std::vector<GufoGenerationBytes> list;
  list.reserve(values.size());
  for (const auto value : values)
    list.push_back(Bytes(value));
  return list;
}

std::uint32_t Initial(TextGenerationBackend::InitialOutputState state) {
  using State = TextGenerationBackend::InitialOutputState;
  switch (state) {
    case State::kAuto:
      return GUFO_GENERATION_INITIAL_AUTO;
    case State::kReasoning:
      return GUFO_GENERATION_INITIAL_REASONING;
    case State::kContent:
      return GUFO_GENERATION_INITIAL_CONTENT;
    case State::kContentAfterReasoning:
      return GUFO_GENERATION_INITIAL_CONTENT_AFTER_REASONING;
  }
  throw std::logic_error("unknown initial output state");
}

std::uint32_t Choice(ChatRequest::ToolChoice choice) {
  switch (choice) {
    case ChatRequest::ToolChoice::kAuto:
      return GUFO_GENERATION_TOOL_CHOICE_AUTO;
    case ChatRequest::ToolChoice::kNone:
      return GUFO_GENERATION_TOOL_CHOICE_NONE;
    case ChatRequest::ToolChoice::kRequired:
      return GUFO_GENERATION_TOOL_CHOICE_REQUIRED;
  }
  throw std::logic_error("unknown tool choice");
}

// Receives the parser's events for one call into Rust.
struct Events {
  const GenerationEventFn& on_event;
  core::ffi::CallbackFailure failure{};
  std::vector<TextSourceSpan> spans{};
  bool not_completed{false};

  static int Receive(void* context, const GufoGenerationEvent* raw) noexcept {
    auto& events = *static_cast<Events*>(context);
    return events.failure.Guard(1, [&] {
      if (raw->kind == GUFO_GENERATION_UNSATISFIED) {
        events.not_completed = raw->complete != 0;
        return 0;
      }
      events.spans.clear();
      for (std::size_t i = 0; i < raw->span_count; ++i)
        events.spans.push_back({raw->spans[i].begin, raw->spans[i].size});
      GenerationEvent event{
          .kind = static_cast<GenerationEvent::Kind>(raw->kind),
          .index = raw->index,
          .text = View(raw->text),
          .name = View(raw->name),
          .arguments = View(raw->arguments),
          .spans = events.spans,
          .begin = raw->begin != 0,
          .complete = raw->complete != 0,
          .hide_tool_markup = raw->hide_tool_markup != 0,
      };
      return events.on_event(event) ? 0 : 1;
    });
  }
};

}  // namespace

// The config and the C++ side of the callbacks: the Qwen scanner, the
// dialect's scanner and the tool grammar, with storage for their last
// results.
struct GenerationStream::Host {
  explicit Host(const GenerationSetup& setup)
      : dialect(setup.markup.dialect),
        tools(setup.tools),
        reasoning(setup.reasoning),
        markers(List(setup.markup.markers)),
        reasoning_markers(List(setup.markup.reasoning_markers)),
        closers(List(setup.markup.closers)),
        heads(List(kEnvelopeHeads)),
        parameters(List(kEnvelopeParameters)),
        envelope_closers(List(kEnvelopeClosers)) {
    tool_list.reserve(tools.size());
    for (const auto& tool : tools)
      tool_list.push_back({Bytes(tool.name), Bytes(tool.parameters_json)});
    std::uint32_t flags = 0;
    if (setup.markup.qwen_calls)
      flags |= GUFO_GENERATION_QWEN_CALLS;
    if (setup.markup.native)
      flags |= GUFO_GENERATION_NATIVE;
    if (dialect != nullptr && dialect->scan_tool_calls != nullptr)
      flags |= GUFO_GENERATION_DIALECT;
    if (setup.track_spans)
      flags |= GUFO_GENERATION_TRACK_SPANS;
    if (setup.buffered)
      flags |= GUFO_GENERATION_BUFFERED;
    if (reasoning.repeats && !reasoning.open.empty())
      flags |= GUFO_GENERATION_REASONING_REPEATS;
    std::uint32_t mode = GUFO_GENERATION_MODE_PLAIN;
    if (setup.response_format)
      mode = GUFO_GENERATION_MODE_STRUCTURED;
    else if (setup.constrained_tools)
      mode = GUFO_GENERATION_MODE_TOOL_ONLY;
    config = {
        .initial = Initial(setup.initial),
        .mode = mode,
        .tool_choice = Choice(setup.tool_choice),
        .flags = flags,
        .reasoning_open = Bytes(reasoning.open),
        .reasoning_close = Bytes(reasoning.close),
        .reasoning_separator = Bytes(setup.markup.reasoning_separator),
        .qwen_open = Bytes(tokenization::QwenToolSyntax::kCallOpen),
        .markers = markers.data(),
        .marker_count = markers.size(),
        .reasoning_markers = reasoning_markers.data(),
        .reasoning_marker_count = reasoning_markers.size(),
        .closers = closers.data(),
        .closer_count = closers.size(),
        .envelope_heads = heads.data(),
        .envelope_head_count = heads.size(),
        .envelope_parameters = parameters.data(),
        .envelope_parameter_count = parameters.size(),
        .envelope_closers = envelope_closers.data(),
        .envelope_closer_count = envelope_closers.size(),
        .tools = tool_list.data(),
        .tool_count = tool_list.size(),
        .qwen_scan = &Host::QwenScan,
        .dialect_scan = &Host::DialectScan,
        .grammar_accepts = &Host::GrammarAccepts,
        .context = this,
    };
  }
  Host(const Host&) = delete;
  Host& operator=(const Host&) = delete;

  static GufoQwenParameter Parameter(
      const tokenization::QwenToolCallParameter& parameter) noexcept {
    return {.name = Bytes(parameter.name),
            .value = Bytes(parameter.value),
            .framed = parameter.framed,
            .progress = parameter.progress};
  }

  static int QwenScan(void* context, const std::uint8_t* text, std::size_t len,
                      GufoQwenScan* out) noexcept {
    auto& host = *static_cast<Host*>(context);
    return host.failure.Guard(1, [&] {
      host.qwen = tokenization::ScanQwenToolCall(
          {reinterpret_cast<const char*>(text), len});
      host.qwen_parameters.clear();
      *out = {};
      out->matched = host.qwen.matched;
      out->progress = host.qwen.progress;
      out->end = host.qwen.end;
      if (const auto& call = host.qwen.call) {
        for (const auto& parameter : call->parameters)
          host.qwen_parameters.push_back(Parameter(parameter));
        out->has_call = 1;
        out->native = call->native;
        out->canonical_header = call->canonical_header;
        out->complete = call->complete;
        out->name = Bytes(call->name);
        out->json_payload = Bytes(call->json_payload);
        out->parameters = host.qwen_parameters.data();
        out->parameter_count = host.qwen_parameters.size();
        if (call->unfinished) {
          out->has_unfinished = 1;
          out->unfinished = Parameter(*call->unfinished);
        }
      }
      return 0;
    });
  }

  static GufoScannedArgument Argument(
      const ext::ScannedToolArgument& argument) noexcept {
    return {.name = Bytes(argument.name),
            .value = Bytes(argument.value),
            .is_string = argument.is_string};
  }

  static int DialectScan(void* context, const std::uint8_t* text,
                         std::size_t len, const GufoGenerationBytes* markers,
                         std::size_t marker_count, int partial,
                         GufoDialectScan* out) noexcept {
    auto& host = *static_cast<Host*>(context);
    return host.failure.Guard(1, [&] {
      host.dialect_markers.clear();
      for (std::size_t i = 0; i < marker_count; ++i)
        host.dialect_markers.push_back(View(markers[i]));
      host.scanned.clear();
      host.scanned_spans.clear();
      host.dialect->scan_tool_calls({reinterpret_cast<const char*>(text), len},
                                    host.dialect_markers, &host.scanned,
                                    &host.scanned_spans, partial != 0);
      host.scanned_arguments.assign(host.scanned.size(), {});
      host.scanned_calls.clear();
      for (std::size_t i = 0; i < host.scanned.size(); ++i) {
        const auto& call = host.scanned[i];
        auto& arguments = host.scanned_arguments[i];
        for (const auto& argument : call.arguments)
          arguments.push_back(Argument(argument));
        host.scanned_calls.push_back({
            .name = Bytes(call.name),
            .arguments = arguments.data(),
            .argument_count = arguments.size(),
            .has_unfinished = call.unfinished.has_value(),
            .valid = call.valid,
            .complete = call.complete,
            .unfinished = call.unfinished ? Argument(*call.unfinished)
                                          : GufoScannedArgument{},
        });
      }
      host.ranges.clear();
      for (const auto& [begin, end] : host.scanned_spans)
        host.ranges.push_back({begin, end});
      *out = {.calls = host.scanned_calls.data(),
              .call_count = host.scanned_calls.size(),
              .spans = host.ranges.data(),
              .span_count = host.ranges.size()};
      return 0;
    });
  }

  // An unfinished quotation is ambiguous: the parser keeps such a call only
  // when its complete arguments satisfy the tool's non-strict grammar.
  static int GrammarAccepts(void* context, std::size_t tool,
                            const std::uint8_t* arguments, std::size_t len,
                            int* accepted) noexcept {
    auto& host = *static_cast<Host*>(context);
    return host.failure.Guard(1, [&] {
      if (tool >= host.tools.size())
        throw std::logic_error("generation parser named an unknown tool");
      try {
        const auto grammar = sampling::JsonConstraint::Compile(
            json::parse(host.tools[tool].parameters_json), false);
        auto state = grammar->Start();
        for (std::size_t i = 0; i < len; ++i)
          state = grammar->Advance(state, arguments[i]);
        *accepted = grammar->Complete(state);
      } catch (const std::exception&) {
        *accepted = 0;
      }
      return 0;
    });
  }

  // Rethrows a callback's exception, then reports the status: false when
  // the event callback stopped the parse.
  bool Check(GufoGenerationStatus status, const Events& events) const {
    failure.Rethrow();
    events.failure.Rethrow();
    switch (status) {
      case GUFO_GENERATION_COMPLETE:
        return true;
      case GUFO_GENERATION_STOPPED:
        return false;
      case GUFO_GENERATION_TOOL_CHOICE_UNSATISFIED:
        throw TextGenerationError(
            TextGenerationErrorCode::kToolChoiceUnsatisfied,
            events.not_completed
                ? "model did not complete a declared tool call"
                : "model did not produce a declared tool call");
      case GUFO_GENERATION_UNSAFE_STREAM:
        throw std::runtime_error("streamed tool call did not complete safely");
      case GUFO_GENERATION_NO_MEMORY:
        throw std::bad_alloc();
      case GUFO_GENERATION_INTERNAL:
        break;
    }
    throw std::logic_error("generation parser failed");
  }

  const ext::ToolDialect* dialect;
  std::span<const tokenization::ChatTool> tools;
  ext::ReasoningMarkup reasoning;
  std::vector<GufoGenerationBytes> markers;
  std::vector<GufoGenerationBytes> reasoning_markers;
  std::vector<GufoGenerationBytes> closers;
  std::vector<GufoGenerationBytes> heads;
  std::vector<GufoGenerationBytes> parameters;
  std::vector<GufoGenerationBytes> envelope_closers;
  std::vector<GufoGenerationTool> tool_list;
  GufoGenerationConfig config{};
  core::ffi::CallbackFailure failure;

  tokenization::QwenToolCallScan qwen;
  std::vector<GufoQwenParameter> qwen_parameters;
  std::vector<std::string_view> dialect_markers;
  std::vector<ext::ScannedToolCall> scanned;
  std::vector<std::pair<std::size_t, std::size_t>> scanned_spans;
  std::vector<std::vector<GufoScannedArgument>> scanned_arguments;
  std::vector<GufoScannedCall> scanned_calls;
  std::vector<GufoGenerationRange> ranges;
};

GenerationStream::GenerationStream(const GenerationSetup& setup)
    : host_(std::make_unique<Host>(setup)) {
  const auto status = gufo_generation_new(&host_->config, &handle_);
  if (status == GUFO_GENERATION_NO_MEMORY)
    throw std::bad_alloc();
  if (status != GUFO_GENERATION_COMPLETE)
    throw std::logic_error("generation parser rejected its config");
}

GenerationStream::~GenerationStream() {
  gufo_generation_free(handle_);
}

bool GenerationStream::Feed(std::string_view bytes,
                            const GenerationEventFn& on_event) {
  Events events{.on_event = on_event};
  const auto status = gufo_generation_feed(
      handle_, reinterpret_cast<const std::uint8_t*>(bytes.data()),
      bytes.size(), &Events::Receive, &events);
  return host_->Check(status, events);
}

bool GenerationStream::Finish(bool enforce_required, bool interrupted,
                              const GenerationEventFn& on_event) {
  Events events{.on_event = on_event};
  const auto status = gufo_generation_finish(
      handle_, enforce_required, interrupted, &Events::Receive, &events);
  return host_->Check(status, events);
}

GeneratedText ParseGeneratedText(const GenerationSetup& setup,
                                 std::string_view bytes,
                                 bool enforce_required) {
  GenerationStream::Host host(setup);
  GeneratedText generated;
  const GenerationEventFn collect = [&](const GenerationEvent& event) {
    switch (event.kind) {
      case GenerationEvent::Kind::kFinalReasoning:
        generated.reasoning = event.text;
        break;
      case GenerationEvent::Kind::kFinalContent:
        generated.text = event.text;
        generated.spans.assign(event.spans.begin(), event.spans.end());
        break;
      case GenerationEvent::Kind::kFinalCall:
        generated.calls.push_back(
            {std::string(event.name), std::string(event.arguments)});
        break;
      case GenerationEvent::Kind::kFinalEnd:
        generated.hide_tool_markup = event.hide_tool_markup;
        break;
      default:
        break;
    }
    return true;
  };
  Events events{.on_event = collect};
  const auto status = gufo_generation_parse(
      &host.config, reinterpret_cast<const std::uint8_t*>(bytes.data()),
      bytes.size(), enforce_required, &Events::Receive, &events);
  if (!host.Check(status, events))
    throw std::logic_error("generation parser stopped");
  return generated;
}

}  // namespace gufo::server
