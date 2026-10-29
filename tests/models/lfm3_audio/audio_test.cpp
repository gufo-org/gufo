// Host-side LFM3-Audio helpers: WAV round trip, resampling, prompt framing
// and task prompts. Expected strings come from the model card and the
// official chat template, not from production constants.
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
  std::printf(failures ? "FAILED %d\n" : "PASSED\n", failures);
  return failures ? 1 : 0;
}
