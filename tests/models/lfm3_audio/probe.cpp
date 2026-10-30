// Compares the native LFM3-Audio stages with fixtures of the independent
// FP32 PyTorch reference (reference/lfm3_audio_ref.py):
//   lfm3_audio_probe BACKBONE.gguf MMPROJ.gguf REF_DIR WAV
// REF_DIR holds the reference's asr/, depth/, codec/ and tts/ outputs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/lfm3_audio/generation.hpp"
namespace audio = gufo::models::lfm3_audio;
namespace fs = std::filesystem;
namespace {
template<class T>
std::vector<T> Load(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("missing fixture " + path.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}
gufo::json::Value Json(const fs::path& path) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)), {});
  return gufo::json::parse(text);
}
int failures = 0;
// Reports max |a - b| relative to max |b|.
void Compare(const char* name, std::span<const float> a,
             std::span<const float> b, double tolerance) {
  if (a.size() != b.size()) {
    std::printf("%-22s FAIL size %zu vs %zu\n", name, a.size(), b.size());
    ++failures;
    return;
  }
  double diff = 0, scale = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(double(a[i]) - b[i]));
    scale = std::max(scale, std::abs(double(b[i])));
  }
  const double relative = diff / std::max(scale, 1e-30);
  const bool ok = relative <= tolerance;
  failures += !ok;
  std::printf("%-22s %s max_abs %.3g rel %.3g (tol %.1g)\n", name,
              ok ? "ok  " : "FAIL", diff, relative, tolerance);
}
void Check(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: lfm3_audio_probe BACKBONE MMPROJ REF_DIR WAV\n";
      return 2;
    }
    const fs::path ref = argv[3];
    std::string error;
    auto t0 = std::chrono::steady_clock::now();
    auto model = audio::Model::Load(argv[1], argv[2], &error);
    Check(bool(model), error);
    std::printf("load %.0f ms\n", std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
    // Audio input.
    std::ifstream in(argv[4], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> pcm;
    std::uint32_t rate = 0;
    Check(audio::DecodeWav(
              {reinterpret_cast<const std::byte*>(raw.data()), raw.size()},
              &pcm, &rate, &error),
          error);
    Check(rate == 16000, "probe WAV must be 16 kHz");
    std::vector<float> mel;
    std::uint32_t frames = 0;
    Check(model->Features(pcm, &mel, &frames, &error), error);
    const auto ref_mel = Load<float>(ref / "asr/mel.f32");
    std::vector<float> mel_t(mel.size());
    for (std::uint32_t t = 0; t < frames; ++t)
      for (int b = 0; b < 128; ++b)
        mel_t[std::size_t(b) * frames + t] = mel[std::size_t(t) * 128 + b];
    // FP32 DFT against the reference FFT; log10 amplifies near-floor power.
    Compare("mel", mel_t, ref_mel, 1e-3);
    std::vector<float> embeddings;
    Check(model->Encode(pcm, &embeddings, &error), error);
    Compare("audio embeddings", embeddings,
            Load<float>(ref / "asr/audio_embeds.f32"), 2e-3);
    // ASR prompt: hidden state and first logits.
    auto asr = Json(ref / "asr/asr.json");
    std::vector<std::uint32_t> ids;
    for (const auto& v : asr["prompt_ids"].items())
      ids.push_back(std::uint32_t(v.as_double()));
    const auto& tokens = model->tokens();
    const auto hidden = model->backbone().config().hidden;
    {
      auto session = model->backbone().CreateSession(4096, &error);
      Check(bool(session), error);
      std::size_t i = 0;
      while (i < ids.size()) {
        std::size_t j = i;
        if (ids[i] == tokens.audio) {
          while (j < ids.size() && ids[j] == tokens.audio)
            ++j;
          Check(j - i == embeddings.size() / hidden, "audio token count");
          Check(session->ForwardEmbeddings(embeddings, false, {}, &error),
                error);
        } else {
          while (j < ids.size() && ids[j] != tokens.audio)
            ++j;
          Check(session->Forward({ids.data() + i, j - i}, {}, &error), error);
        }
        i = j;
      }
      std::vector<float> h(hidden), logits(model->backbone().config().vocab);
      Check(session->ReadHidden(h, &error), error);
      Check(session->ReadLogits(logits, &error), error);
      Compare("asr prompt hidden", h,
              Load<float>(ref / "asr/prompt_hidden.f32"), 2e-2);
      auto ref_logits = Load<float>(ref / "asr/first_logits.f32");
      for (auto id : {tokens.audio, tokens.audio_out, tokens.mixed_start,
                      tokens.audio_start})
        logits[id] = ref_logits[id] = 0;
      Compare("asr first logits", logits, ref_logits, 2e-2);
    }
    // Greedy ASR end to end.
    auto generator = audio::Generator::Create(model, 4096, &error);
    Check(bool(generator), error);
    {
      std::vector<audio::ChatMessage> messages(1);
      messages[0].role = "user";
      messages[0].parts.push_back({true, {}, pcm});
      const auto prompt =
          audio::BuildPrompt(audio::AsrSystemPrompt("English"), messages);
      audio::GenerateOptions options;
      options.max_new_tokens = 96;
      options.suppress_mixed = options.suppress_audio_out = true;
      audio::GenerateResult result;
      Check(generator->Generate(prompt, options, {}, &result, &error), error);
      const auto expected = asr["text"].str();
      const bool ok = result.text == expected;
      failures += !ok;
      std::printf("%-22s %s \"%s\" (prompt %zu, audio %zu, %.1f ms)\n",
                  "asr transcript", ok ? "ok  " : "FAIL", result.text.c_str(),
                  result.prompt_tokens, result.audio_tokens,
                  result.timings.total_ms);
      std::printf(
          "  encode %.1f prefill %.1f first_text %.1f total %.1f ms "
          "(%.1fx real time)\n",
          result.timings.encode_ms, result.timings.prefill_ms,
          result.timings.first_text_ms, result.timings.total_ms,
          pcm.size() / 16.0 / result.timings.total_ms);
    }
    // Streaming encoder: microphone-sized and irregular pieces give the
    // offline embeddings.
    std::vector<float> streamed_rows;
    {
      auto input = model->CreateAudioInput(&error);
      Check(bool(input), error);
      std::size_t at = 0, step = 1600;
      std::vector<float> rows;
      while (at < pcm.size()) {
        const auto n = std::min(step, pcm.size() - at);
        Check(input->Push({pcm.data() + at, n}, false, &rows, &error), error);
        streamed_rows.insert(streamed_rows.end(), rows.begin(), rows.end());
        at += n;
        step = step == 1600 ? 37 : step == 37 ? 4001 : 1600;
      }
      Check(input->Push({}, true, &rows, &error), error);
      streamed_rows.insert(streamed_rows.end(), rows.begin(), rows.end());
      Compare("streamed embeddings", streamed_rows, embeddings, 1e-5);
    }
    // Incremental prefill: the same ASR turn built from streamed rows, a
    // discarded turn erased from the context, then continued.
    {
      std::vector<audio::ChatMessage> none;
      auto head = audio::SystemPieces(audio::AsrSystemPrompt("English"));
      head.push_back({audio::PromptPiece::Kind::kTemplate,
                      "<|im_start|>user\nNever mind.<|im_end|>\n",
                      {}});
      Check(
          generator->Reset(&error) && generator->Append(head, nullptr, &error),
          error);
      const auto junk_end = generator->Position();
      const auto junk_begin =
          junk_end - std::uint32_t(model->tokenizer()
                                       .Encode("<|im_start|>user\nNever "
                                               "mind.<|im_end|>\n")
                                       .size());
      const std::vector<audio::PromptPiece> open{
          {audio::PromptPiece::Kind::kTemplate, "<|im_start|>user\n", {}}};
      Check(generator->Append(open, nullptr, &error), error);
      for (std::size_t r = 0; r < streamed_rows.size(); r += 7 * hidden)
        Check(generator->AppendEmbeddings(
                  {streamed_rows.data() + r,
                   std::min<std::size_t>(7 * hidden, streamed_rows.size() - r)},
                  &error),
              error);
      const std::vector<audio::PromptPiece> close{
          {audio::PromptPiece::Kind::kTemplate,
           "<|im_end|>\n<|im_start|>assistant\n",
           {}}};
      Check(generator->Append(close, nullptr, &error), error);
      Check(generator->Replace(junk_begin, junk_end, {}, &error), error);
      audio::GenerateOptions options;
      options.max_new_tokens = 96;
      options.suppress_mixed = options.suppress_audio_out = true;
      audio::GenerateResult result;
      Check(generator->Continue(options, {}, &result, &error), error);
      const bool ok = result.text == asr["text"].str();
      failures += !ok;
      std::printf("%-22s %s \"%s\" (first text %.1f ms after the turn)\n",
                  "incremental asr", ok ? "ok  " : "FAIL", result.text.c_str(),
                  result.timings.first_text_ms);
    }
    // Live ASR in 100 ms pieces against the reference's chunked decode.
    if (fs::exists(ref / "live/live.json")) {
      auto live_ref = Json(ref / "live/live.json");
      auto live = audio::LiveTranscriber::Create(model, &error);
      Check(bool(live), error);
      std::vector<double> lag;
      auto start = std::chrono::steady_clock::now();
      auto on_text = [&](std::string_view) {
        // Lag behind the audio pushed so far, were it spoken in real time.
        const double now = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
        lag.push_back(now);
        return true;
      };
      double compute = 0;
      for (std::size_t at = 0; at < pcm.size(); at += 1600) {
        const auto t = std::chrono::steady_clock::now();
        Check(live->Push({pcm.data() + at,
                          std::min<std::size_t>(1600, pcm.size() - at)},
                         on_text, &error),
              error);
        compute = std::max(compute, std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t)
                                        .count());
      }
      const auto t = std::chrono::steady_clock::now();
      Check(live->Finish(on_text, &error), error);
      const double finish = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t)
                                .count();
      const auto expected = live_ref["text"].str();
      const bool ok = live->text() == expected;
      failures += !ok;
      std::printf("%-22s %s \"%s\"\n", "live asr", ok ? "ok  " : "FAIL",
                  live->text().c_str());
      if (!ok)
        std::printf("  reference \"%s\"\n", expected.c_str());
      std::printf("  %u chunks; slowest 100 ms piece %.1f ms; finish %.1f ms\n",
                  live->chunks(), compute, finish);
    } else {
      std::printf("%-22s skipped (no live fixture)\n", "live asr");
    }
    // Depth decoder and frame embedding.
    auto voice = model->CreateVoice(&error);
    Check(bool(voice), error);
    {
      const auto h = Load<float>(ref / "depth/depth_hidden.f32");
      const auto codes = Load<std::int32_t>(ref / "depth/depth_codes.i32");
      std::vector<std::int32_t> masked(12);
      for (int i = 0; i < 12; ++i)
        masked[i] = i % 2 == 0;
      std::vector<float> logits(12 * 2017);
      Check(voice->DepthLogits(h, codes, masked, logits, &error), error);
      Compare("depth logits", logits,
              Load<float>(ref / "depth/depth_logits.f32"), 2e-2);
      float* device{};
      Check(hipMalloc(&device, hidden * 4) == hipSuccess, "hipMalloc");
      Check(voice->FrameEmbedding(codes, device, &error), error);
      std::vector<float> e(hidden);
      Check(hipMemcpy(e.data(), device, hidden * 4, hipMemcpyDeviceToHost) ==
                hipSuccess,
            "copy");
      (void)hipFree(device);
      Compare("frame embedding", e,
              Load<float>(ref / "depth/frame_embedding.f32"), 2e-2);
    }
    // Codec: whole decode and chunked streaming decode.
    {
      const auto codes = Load<std::int32_t>(ref / "codec/codec_codes.i32");
      const auto expected = Load<float>(ref / "codec/codec_audio.f32");
      std::vector<float> whole;
      voice->Reset();
      auto c0 = std::chrono::steady_clock::now();
      Check(voice->Decode(codes, true, &whole, &error), error);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - c0)
                            .count();
      Compare("codec audio", whole, expected, 2e-3);
      std::printf("  codec %zu frames in %.1f ms\n", codes.size() / 12, ms);
      voice->Reset();
      std::vector<float> streamed, piece;
      const std::size_t frames_total = codes.size() / 12;
      std::size_t at = 0, step = 1;
      while (at < frames_total) {
        const auto n = std::min(step, frames_total - at);
        Check(voice->Decode({codes.data() + at * 12, n * 12},
                            at + n == frames_total, &piece, &error),
              error);
        streamed.insert(streamed.end(), piece.begin(), piece.end());
        at += n;
        step = step % 5 + 1;
      }
      Compare("codec streamed", streamed, whole, 1e-4);
    }
    // Greedy TTS with the reference's fixed reveal order.
    {
      const auto ref_codes = Load<std::int32_t>(ref / "tts/tts_codes.i32");
      auto tts = Json(ref / "tts/tts.json");
      std::vector<std::uint32_t> prompt_ids;
      for (const auto& v : tts["prompt_ids"].items())
        prompt_ids.push_back(std::uint32_t(v.as_double()));
      auto session = model->backbone().CreateSession(4096, &error);
      Check(bool(session), error);
      Check(session->Forward(prompt_ids, {}, &error), error);
      std::vector<float> h(hidden);
      Check(session->ReadHidden(h, &error), error);
      Compare("tts prompt hidden", h,
              Load<float>(ref / "tts/tts_prompt_hidden.f32"), 2e-2);
      voice->Reset();
      audio::Voice::Sampling greedy{0.0F, 0, 1.0F, 0};
      std::mt19937_64 rng(0);
      std::vector<std::int32_t> codes;
      std::array<std::int32_t, 12> frame;
      float* device{};
      Check(hipMalloc(&device, hidden * 4) == hipSuccess, "hipMalloc");
      std::size_t matched = 0;
      const std::size_t ref_frames = ref_codes.size() / 12;
      for (std::size_t f = 0; f < ref_frames + 1; ++f) {
        Check(voice->SampleFrame(session->DeviceHidden(), greedy, rng, frame,
                                 device, true, &error),
              error);
        if (std::find(frame.begin(), frame.end(), 2016) != frame.end())
          break;
        codes.insert(codes.end(), frame.begin(), frame.end());
        if (f < ref_frames &&
            std::equal(frame.begin(), frame.end(), ref_codes.begin() + f * 12))
          ++matched;
        Check(session->ForwardEmbeddings({device, hidden}, true, {}, &error),
              error);
      }
      (void)hipFree(device);
      std::printf("%-22s %zu/%zu reference frames identical, %zu native\n",
                  "tts greedy frames", matched, ref_frames, codes.size() / 12);
    }
    // Sampled TTS through the generator, transcribed back by the model.
    {
      std::string system;
      Check(audio::TtsSystemPrompt("en", "female", &system, &error), error);
      std::vector<audio::ChatMessage> messages(1);
      messages[0].role = "user";
      messages[0].parts.push_back(
          {false,
           "The quick brown fox jumps over the lazy dog. Then it runs back "
           "home through the forest, happy and tired.",
           {}});
      const auto prompt =
          audio::BuildPrompt(system, messages, "<|audio_start|>");
      audio::GenerateOptions options;
      options.max_new_tokens = 600;
      options.seed = 7;
      audio::GenerateResult speech;
      Check(generator->Generate(prompt, options, {}, &speech, &error), error);
      const double seconds = speech.samples.size() / 24000.0;
      const auto& t = speech.timings;
      std::printf(
          "tts: %zu frames, %.2f s audio in %.1f ms (%.1fx real "
          "time); prefill %.1f first_audio %.1f backbone %.1f depth "
          "%.1f codec %.1f ms\n",
          speech.frames, seconds, t.total_ms, seconds * 1000 / t.total_ms,
          t.prefill_ms, t.first_audio_ms, t.backbone_ms, t.depth_ms,
          t.codec_ms);
      const auto wav = audio::EncodeWav(speech.samples, 24000);
      std::ofstream(ref / "native_tts.wav", std::ios::binary)
          .write(reinterpret_cast<const char*>(wav.data()), wav.size());
      std::vector<audio::ChatMessage> asr_messages(1);
      asr_messages[0].role = "user";
      asr_messages[0].parts.push_back(
          {true, {}, audio::Resample(speech.samples, 24000, 16000)});
      audio::GenerateOptions asr_options;
      asr_options.max_new_tokens = 128;
      asr_options.suppress_mixed = asr_options.suppress_audio_out = true;
      audio::GenerateResult heard;
      Check(generator->Generate(
                audio::BuildPrompt(audio::AsrSystemPrompt("English"),
                                   asr_messages),
                asr_options, {}, &heard, &error),
            error);
      std::printf("tts round trip: \"%s\"\n", heard.text.c_str());
    }
    // Batched synthesis: greedy streams match the lone generator's codes up
    // to batched rounding; sampled streams report aggregate throughput.
    {
      std::string system;
      Check(audio::TtsSystemPrompt("en", "female", &system, &error), error);
      std::vector<audio::ChatMessage> messages(1);
      messages[0].role = "user";
      messages[0].parts.push_back(
          {false,
           "The quick brown fox jumps over the lazy dog. Then it runs back "
           "home through the forest, happy and tired.",
           {}});
      const auto prompt =
          audio::BuildPrompt(system, messages, "<|audio_start|>");
      auto batch = audio::SpeechBatch::Create(model, 1024, 8, &error);
      Check(bool(batch), error);
      audio::GenerateOptions greedy;
      greedy.max_new_tokens = 600;
      greedy.audio.temperature = 0;
      audio::GenerateResult lone;
      Check(generator->Generate(prompt, greedy, {}, &lone, &error), error);
      for (std::size_t streams : {1U, 2U, 4U, 8U}) {
        for (bool sampled : {false, true}) {
          std::vector<audio::GenerateResult> results(streams);
          std::size_t done = 0;
          const auto start = std::chrono::steady_clock::now();
          for (std::size_t i = 0; i < streams; ++i) {
            auto options = greedy;
            if (sampled) {
              options.audio = {};
              options.seed = 100 + i;
            }
            Check(batch->Add(
                      prompt, options, {},
                      [&, i](bool ok, audio::GenerateResult result,
                             std::string failure) {
                        Check(ok, failure);
                        results[i] = std::move(result);
                        ++done;
                      },
                      &error),
                  error);
          }
          while (done < streams)
            Check(batch->Step(&error), error);
          const double ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
          double seconds = 0;
          std::size_t same = 0, frames = 0;
          for (const auto& r : results) {
            seconds += r.samples.size() / 24000.0;
            if (!sampled) {
              frames += lone.frames;
              for (std::size_t f = 0; f < std::min(r.frames, lone.frames); ++f)
                same += std::equal(r.codes.begin() + f * 12,
                                   r.codes.begin() + f * 12 + 12,
                                   lone.codes.begin() + f * 12);
            }
          }
          if (sampled)
            std::printf(
                "batched tts x%zu: %.2f s audio in %.0f ms (%.1fx real time "
                "aggregate); per step depth %.2f backbone %.2f ms\n",
                streams, seconds, ms, seconds * 1000 / ms,
                results[0].timings.depth_ms / results[0].frames,
                results[0].timings.backbone_ms / results[0].frames);
          else
            std::printf(
                "batched greedy x%zu: %zu/%zu frames as the lone "
                "stream\n",
                streams, same, frames);
        }
      }
    }
    std::printf(failures ? "FAILED %d\n" : "PASSED\n", failures);
    return failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
