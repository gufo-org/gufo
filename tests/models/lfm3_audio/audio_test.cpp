// Host-side LFM3-Audio helpers: WAV round trip, resampling, prompt framing
// and task prompts. Expected strings come from the model card and the
// official chat template, not from production constants.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

#include "src/models/lfm3_audio/generation.hpp"
namespace audio = gufo::models::lfm3_audio;
namespace {
int failures = 0;
void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}
std::string Render(const std::vector<audio::PromptPiece>& pieces) {
  std::string out;
  for (const auto& p : pieces)
    out += p.kind == audio::PromptPiece::Kind::kAudio ? "[audio]" : p.text;
  return out;
}
}  // namespace
int main() {
  // WAV: PCM16 round trip within one quantization step.
  std::vector<float> tone(2400);
  for (std::size_t i = 0; i < tone.size(); ++i)
    tone[i] = 0.5F * std::sin(2 * std::numbers::pi * 440.0 * i / 24000);
  const auto wav = audio::EncodeWav(tone, 24000);
  std::vector<float> decoded;
  std::uint32_t rate = 0;
  std::string error;
  Expect(audio::DecodeWav(wav, &decoded, &rate, &error), "decode: " + error);
  Expect(rate == 24000 && decoded.size() == tone.size(), "wav shape");
  double worst = 0;
  for (std::size_t i = 0; i < tone.size() && i < decoded.size(); ++i)
    worst = std::max(worst, std::abs(double(decoded[i]) - tone[i]));
  Expect(worst <= 1.0 / 32767, "wav quantization error");
  Expect(!audio::DecodeWav(std::span(wav).first(20), &decoded, &rate, &error),
         "truncated wav rejected");

  // Resampling keeps an in-band tone: 1 kHz at 24 kHz to 16 kHz.
  std::vector<float> source(24000);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = std::sin(2 * std::numbers::pi * 1000.0 * i / 24000);
  const auto resampled = audio::Resample(source, 24000, 16000);
  Expect(resampled.size() == 16000, "resampled length");
  double error_sum = 0;
  for (std::size_t i = 2000; i < 14000; ++i)
    error_sum += std::abs(resampled[i] -
                          std::sin(2 * std::numbers::pi * 1000.0 * i / 16000));
  Expect(error_sum / 12000 < 1e-3, "resampled tone error");

  // Mel framing of the official extractor: right padding to whole hops.
  // One second: 16080 padded samples, 99 frames (the reference reports 99).
  Expect(audio::MelFrames(16000) == 99 && audio::MelFrames(1) == 1 &&
             audio::MelFrames(560) == 2 && audio::EncoderTokens(99) == 13,
         "mel frame counts");

  // Task prompts from the model card.
  Expect(audio::AsrSystemPrompt("English") == "Perform ASR in English.",
         "asr prompt");
  Expect(audio::AsrSystemPrompt() == "Perform ASR.", "asr auto prompt");
  std::string tts;
  Expect(audio::TtsSystemPrompt("en", "female", &tts, &error) &&
             tts == "Perform TTS in American English using a female voice.",
         "tts en prompt");
  Expect(audio::TtsSystemPrompt("en-gb", "MALE", &tts, &error) &&
             tts == "Perform TTS in British English using a male voice.",
         "tts en-gb prompt");
  Expect(audio::TtsSystemPrompt("French", "", &tts, &error) &&
             tts == "Perform TTS in French using a female voice.",
         "tts language name");
  Expect(!audio::TtsSystemPrompt("klingon", "female", &tts, &error),
         "unsupported language rejected");
  Expect(!audio::TtsSystemPrompt("en", "child", &tts, &error),
         "unsupported voice rejected");

  // Chat framing of the official template with an audio placeholder.
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].parts.push_back({true, {}, std::vector<float>(160)});
  messages[0].parts.push_back({false, "What is happening?", {}});
  const auto pieces =
      audio::BuildPrompt("Perform ASR.", messages, "<|mixed_start|>");
  Expect(Render(pieces) ==
             "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n"
             "<|im_start|>user\n[audio]What is happening?<|im_end|>\n"
             "<|im_start|>assistant\n<|mixed_start|>",
         "chat framing: " + Render(pieces));
  // Client text stays literal: control tokens in it are not template.
  Expect(pieces[1].kind == audio::PromptPiece::Kind::kLiteral &&
             pieces[3].kind == audio::PromptPiece::Kind::kAudio &&
             pieces[4].kind == audio::PromptPiece::Kind::kLiteral,
         "piece kinds");

  // Streaming resampling: arbitrary pieces give the one-call samples.
  {
    std::vector<float> noise(7919);
    std::uint32_t state = 1;
    for (auto& v : noise) {
      state = state * 1664525U + 1013904223U;
      v = float(state >> 8) / float(1U << 24) - 0.5F;
    }
    for (auto [from, to] :
         {std::pair{24000U, 16000U}, {44100U, 16000U}, {16000U, 24000U}}) {
      const auto whole = audio::Resample(noise, from, to);
      audio::Resampler stream(from, to);
      std::vector<float> pieces;
      std::size_t at = 0, step = 1;
      while (at < noise.size()) {
        const auto n = std::min(step, noise.size() - at);
        auto out = stream.Push(std::span(noise).subspan(at, n));
        pieces.insert(pieces.end(), out.begin(), out.end());
        at += n;
        step = step * 3 % 487 + 1;
      }
      auto out = stream.Push({}, true);
      pieces.insert(pieces.end(), out.begin(), out.end());
      Expect(pieces == whole, "streaming resampler " + std::to_string(from) +
                                  " -> " + std::to_string(to));
    }
  }

  // Tools in the system turn and tool results, as the official template
  // renders them (tojson spelling, one definition per line).
  {
    const std::string tool =
        R"({"type":"function","function":{"name":"set_volume","description":"Set the device speaker volume.","parameters":{"type":"object","properties":{"volume":{"type":"integer"}},"required":["volume"]},"defer_loading":false}})";
    const auto definition = audio::ToolDefinition(tool);
    Expect(
        definition ==
            R"({"name": "set_volume", "description": "Set the device speaker volume.", "parameters": {"type": "object", "properties": {"volume": {"type": "integer"}}, "required": ["volume"]}})",
        "tool definition: " + definition);
    const std::vector<std::string> tools{definition};
    std::vector<audio::ChatMessage> turn(2);
    turn[0].role = "user";
    turn[0].parts.push_back({false, "Turn it down.", {}});
    turn[1].role = "tool";
    turn[1].parts.push_back({false, R"([{"volume": 30}])", {}});
    const auto framed = Render(audio::BuildPrompt(
        audio::kToolsInterleavedSystemPrompt, turn, {}, tools));
    Expect(framed ==
               "<|startoftext|><|im_start|>system\nRespond with function "
               "calls, and interleaved text and audio.\n\nList of tools:"
               "\n\n<tools>\n" +
                   definition +
                   "\n</tools><|im_end|>\n<|im_start|>user\nTurn it "
                   "down.<|im_end|>\n<|im_start|>tool\n<tool_response>\n"
                   "[{\"volume\": 30}]\n</tool_response><|im_end|>\n"
                   "<|im_start|>assistant\n",
           "tool framing: " + framed);
  }

  // Reply parsing: reasoning, content and calls, with tags split anywhere.
  {
    const std::string reply =
        "Let me think.</think>\n\nSure.<tool_call>\n<function=set_volume>\n"
        "<parameter=volume>\n30\n</parameter>\n<parameter=room>\nkitchen\n"
        "</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n"
        "<function=mute>\n</function>\n</tool_call>";
    for (std::size_t step : {1U, 2U, 5U, 1000U}) {
      audio::ReplyParser parser(true);
      std::string reasoning, content;
      for (std::size_t at = 0; at < reply.size(); at += step) {
        auto delta = parser.Feed(reply.substr(at, step));
        reasoning += delta.reasoning;
        content += delta.content;
      }
      std::vector<audio::ToolCall> calls;
      auto delta = parser.Finish(&calls);
      reasoning += delta.reasoning;
      content += delta.content;
      Expect(reasoning == "Let me think." && content == "Sure.",
             "reply split at " + std::to_string(step) + ": [" + reasoning +
                 "] [" + content + "]");
      Expect(calls.size() == 2 && calls[0].name == "set_volume" &&
                 calls[0].arguments.size() == 2 &&
                 calls[0].arguments[0].value == "30" &&
                 !calls[0].arguments[0].is_string &&
                 calls[0].arguments[1].value == "kitchen" &&
                 calls[0].arguments[1].is_string && calls[1].name == "mute" &&
                 calls[1].arguments.empty(),
             "tool calls at " + std::to_string(step));
    }
    // A reply may open its own reasoning; plain replies pass through.
    audio::ReplyParser opened;
    auto a = opened.Feed("<think>hm</think>Yes");
    std::vector<audio::ToolCall> calls;
    auto b = opened.Finish(&calls);
    Expect(a.reasoning + b.reasoning == "hm" &&
               a.content + b.content == "Yes" && calls.empty(),
           "self-opened reasoning");
    audio::ReplyParser plain;
    auto c = plain.Feed("Hello <tool");
    auto d = plain.Finish(&calls);
    Expect(c.content == "Hello " && d.content == "<tool" && calls.empty(),
           "partial tag released at the end");
  }
  std::printf(failures ? "FAILED %d\n" : "PASSED\n", failures);
  return failures ? 1 : 0;
}
