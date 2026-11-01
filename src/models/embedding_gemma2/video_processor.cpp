#include "video_processor.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/library.hpp>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "config.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {
// Thread-safe first-use loading preserves text/image/audio startup latency.
// Hold handles before the typed symbols so every function pointer stays valid.
struct FFmpegApi {
  platform::Library avformat{GUFO_AVFORMAT_LIBRARY};
  platform::Library avcodec{GUFO_AVCODEC_LIBRARY};
  platform::Library avutil{GUFO_AVUTIL_LIBRARY};
  platform::Library swscale{GUFO_SWSCALE_LIBRARY};
  decltype(&::avio_alloc_context) avio_alloc_context{
      reinterpret_cast<decltype(&::avio_alloc_context)>(
          avformat.Symbol("avio_alloc_context"))};
  decltype(&::avio_context_free) avio_context_free{
      reinterpret_cast<decltype(&::avio_context_free)>(
          avformat.Symbol("avio_context_free"))};
  decltype(&::avformat_alloc_context) avformat_alloc_context{
      reinterpret_cast<decltype(&::avformat_alloc_context)>(
          avformat.Symbol("avformat_alloc_context"))};
  decltype(&::avformat_open_input) avformat_open_input{
      reinterpret_cast<decltype(&::avformat_open_input)>(
          avformat.Symbol("avformat_open_input"))};
  decltype(&::avformat_close_input) avformat_close_input{
      reinterpret_cast<decltype(&::avformat_close_input)>(
          avformat.Symbol("avformat_close_input"))};
  decltype(&::avformat_find_stream_info) avformat_find_stream_info{
      reinterpret_cast<decltype(&::avformat_find_stream_info)>(
          avformat.Symbol("avformat_find_stream_info"))};
  decltype(&::av_read_frame) av_read_frame{
      reinterpret_cast<decltype(&::av_read_frame)>(
          avformat.Symbol("av_read_frame"))};
  decltype(&::avcodec_alloc_context3) avcodec_alloc_context3{
      reinterpret_cast<decltype(&::avcodec_alloc_context3)>(
          avcodec.Symbol("avcodec_alloc_context3"))};
  decltype(&::avcodec_parameters_to_context) avcodec_parameters_to_context{
      reinterpret_cast<decltype(&::avcodec_parameters_to_context)>(
          avcodec.Symbol("avcodec_parameters_to_context"))};
  decltype(&::avcodec_open2) avcodec_open2{
      reinterpret_cast<decltype(&::avcodec_open2)>(
          avcodec.Symbol("avcodec_open2"))};
  decltype(&::avcodec_find_decoder) avcodec_find_decoder{
      reinterpret_cast<decltype(&::avcodec_find_decoder)>(
          avcodec.Symbol("avcodec_find_decoder"))};
  decltype(&::avcodec_free_context) avcodec_free_context{
      reinterpret_cast<decltype(&::avcodec_free_context)>(
          avcodec.Symbol("avcodec_free_context"))};
  decltype(&::avcodec_send_packet) avcodec_send_packet{
      reinterpret_cast<decltype(&::avcodec_send_packet)>(
          avcodec.Symbol("avcodec_send_packet"))};
  decltype(&::avcodec_receive_frame) avcodec_receive_frame{
      reinterpret_cast<decltype(&::avcodec_receive_frame)>(
          avcodec.Symbol("avcodec_receive_frame"))};
  decltype(&::av_packet_alloc) av_packet_alloc{
      reinterpret_cast<decltype(&::av_packet_alloc)>(
          avcodec.Symbol("av_packet_alloc"))};
  decltype(&::av_packet_free) av_packet_free{
      reinterpret_cast<decltype(&::av_packet_free)>(
          avcodec.Symbol("av_packet_free"))};
  decltype(&::av_packet_unref) av_packet_unref{
      reinterpret_cast<decltype(&::av_packet_unref)>(
          avcodec.Symbol("av_packet_unref"))};
  decltype(&::av_malloc) av_malloc{
      reinterpret_cast<decltype(&::av_malloc)>(avutil.Symbol("av_malloc"))};
  decltype(&::av_free) av_free{
      reinterpret_cast<decltype(&::av_free)>(avutil.Symbol("av_free"))};
  decltype(&::av_freep) av_freep{
      reinterpret_cast<decltype(&::av_freep)>(avutil.Symbol("av_freep"))};
  decltype(&::av_dict_set) av_dict_set{
      reinterpret_cast<decltype(&::av_dict_set)>(avutil.Symbol("av_dict_set"))};
  decltype(&::av_dict_free) av_dict_free{
      reinterpret_cast<decltype(&::av_dict_free)>(
          avutil.Symbol("av_dict_free"))};
  decltype(&::av_frame_alloc) av_frame_alloc{
      reinterpret_cast<decltype(&::av_frame_alloc)>(
          avutil.Symbol("av_frame_alloc"))};
  decltype(&::av_frame_free) av_frame_free{
      reinterpret_cast<decltype(&::av_frame_free)>(
          avutil.Symbol("av_frame_free"))};
  decltype(&::av_frame_unref) av_frame_unref{
      reinterpret_cast<decltype(&::av_frame_unref)>(
          avutil.Symbol("av_frame_unref"))};
  decltype(&::sws_getContext) sws_getContext{
      reinterpret_cast<decltype(&::sws_getContext)>(
          swscale.Symbol("sws_getContext"))};
  decltype(&::sws_freeContext) sws_freeContext{
      reinterpret_cast<decltype(&::sws_freeContext)>(
          swscale.Symbol("sws_freeContext"))};
  decltype(&::sws_scale) sws_scale{
      reinterpret_cast<decltype(&::sws_scale)>(swscale.Symbol("sws_scale"))};
};
FFmpegApi& FFmpeg() {
  static FFmpegApi api;
  return api;
}

struct Budget {
  const std::function<bool()>& cancelled;
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::now() + std::chrono::seconds(15)};
  std::size_t pixels{0};
  bool Stop() const {
    return (cancelled && cancelled()) ||
           std::chrono::steady_clock::now() > deadline;
  }
  void Check() const {
    if (Stop())
      throw std::invalid_argument(
          "embedding video decoding cancelled or timed out");
  }
};
struct Decoder {
  std::span<const std::uint8_t> bytes;
  Budget& budget;
  std::size_t position{0};
  AVIOContext* io{nullptr};
  AVFormatContext* format{nullptr};
  AVCodecContext* codec{nullptr};
  AVFrame* frame{nullptr};
  AVPacket* packet{nullptr};
  int stream{-1};
  ~Decoder() {
    FFmpeg().av_packet_free(&packet);
    FFmpeg().av_frame_free(&frame);
    FFmpeg().avcodec_free_context(&codec);
    FFmpeg().avformat_close_input(&format);
    if (io) {
      FFmpeg().av_freep(&io->buffer);
      FFmpeg().avio_context_free(&io);
    }
  }
  static int Read(void* context, std::uint8_t* out, int count) {
    auto& self = *static_cast<Decoder*>(context);
    if (self.budget.Stop())
      return AVERROR_EXIT;
    const auto n =
        std::min<std::size_t>(count, self.bytes.size() - self.position);
    if (!n)
      return AVERROR_EOF;
    std::memcpy(out, self.bytes.data() + self.position, n);
    self.position += n;
    return static_cast<int>(n);
  }
  static std::int64_t Seek(void* context, std::int64_t offset, int whence) {
    auto& self = *static_cast<Decoder*>(context);
    if (self.budget.Stop())
      return AVERROR_EXIT;
    if (whence == AVSEEK_SIZE)
      return static_cast<std::int64_t>(self.bytes.size());
    whence &= ~AVSEEK_FORCE;
    const auto base =
        whence == SEEK_SET   ? 0LL
        : whence == SEEK_CUR ? static_cast<long long>(self.position)
        : whence == SEEK_END ? static_cast<long long>(self.bytes.size())
                             : -1LL;
    if (base < 0 || offset < -base ||
        offset > static_cast<std::int64_t>(self.bytes.size()) - base)
      return AVERROR(EINVAL);
    self.position = static_cast<std::size_t>(base + offset);
    return static_cast<std::int64_t>(self.position);
  }
  void Open() {
    budget.Check();
    auto* buffer = static_cast<unsigned char*>(FFmpeg().av_malloc(32768));
    if (!buffer)
      throw std::bad_alloc();
    io = FFmpeg().avio_alloc_context(buffer, 32768, 0, this, Read, nullptr,
                                     Seek);
    if (!io) {
      FFmpeg().av_free(buffer);
      throw std::bad_alloc();
    }
    format = FFmpeg().avformat_alloc_context();
    if (!format)
      throw std::bad_alloc();
    format->pb = io;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    format->interrupt_callback = {
        [](void* p) { return static_cast<Budget*>(p)->Stop() ? 1 : 0; },
        &budget};
    format->max_streams = 8;
    format->probesize = 1024 * 1024;
    format->max_analyze_duration = 2 * AV_TIME_BASE;
    AVDictionary* options = nullptr;
    FFmpeg().av_dict_set(&options, "format_whitelist", "mov,matroska,webm", 0);
    FFmpeg().av_dict_set(&options, "protocol_whitelist", "", 0);
    // MOV external data references must never trigger additional IO.
    FFmpeg().av_dict_set(&options, "enable_drefs", "0", 0);
    const int opened =
        FFmpeg().avformat_open_input(&format, nullptr, nullptr, &options);
    FFmpeg().av_dict_free(&options);
    if (opened < 0)
      throw std::invalid_argument(
          "embedding video container could not be opened");
    // Probing may itself decode a frame. Apply limits before that decoder is
    // created, not only to the persistent codec used for the two passes.
    AVDictionary* probe_options[8]{};
    for (auto& option : probe_options) {
      FFmpeg().av_dict_set(&option, "max_pixels", "4194304", 0);
      FFmpeg().av_dict_set(&option, "threads", "1", 0);
    }
    const int probed =
        FFmpeg().avformat_find_stream_info(format, probe_options);
    for (auto& option : probe_options)
      FFmpeg().av_dict_free(&option);
    if (probed < 0)
      throw std::invalid_argument(
          "embedding video supports valid MP4/MOV and WebM/Matroska "
          "containers");
    for (unsigned i = 0; i < format->nb_streams; ++i) {
      if (format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        stream = static_cast<int>(i);
        break;
      }
    }
    if (format->duration != AV_NOPTS_VALUE &&
        format->duration > 300LL * AV_TIME_BASE)
      throw std::invalid_argument("embedding video exceeds 300 seconds");
    if (stream < 0)
      throw std::invalid_argument("embedding video has no video stream");
    const auto* parameters = format->streams[stream]->codecpar;
    const auto* implementation =
        FFmpeg().avcodec_find_decoder(parameters->codec_id);
    if (!implementation)
      throw std::invalid_argument("unsupported embedding video codec");
    codec = FFmpeg().avcodec_alloc_context3(implementation);
    if (!codec)
      throw std::bad_alloc();
    if (FFmpeg().avcodec_parameters_to_context(codec, parameters) < 0)
      throw std::invalid_argument("invalid embedding video codec parameters");
    codec->thread_count = 1;
    codec->max_pixels = 4 * 1024 * 1024;
    codec->err_recognition = AV_EF_EXPLODE;
    if (FFmpeg().avcodec_open2(codec, implementation, nullptr) < 0)
      throw std::invalid_argument("embedding video decoder could not open");
    frame = FFmpeg().av_frame_alloc();
    packet = FFmpeg().av_packet_alloc();
    if (!frame || !packet)
      throw std::bad_alloc();
  }
  std::size_t Decode(
      const std::function<void(std::size_t, const AVFrame*)>& consume) {
    std::size_t count = 0;
    int width = 0, height = 0;
    auto receive = [&] {
      for (;;) {
        budget.Check();
        const int status = FFmpeg().avcodec_receive_frame(codec, frame);
        if (status == AVERROR(EAGAIN) || status == AVERROR_EOF)
          break;
        if (status < 0)
          throw std::invalid_argument("embedding video frame decoding failed");
        if (count == 0) {
          width = frame->width;
          height = frame->height;
        }
        if (frame->width != width || frame->height != height)
          throw std::invalid_argument(
              "embedding video frame dimensions change within clip");
        const auto pixels =
            static_cast<std::size_t>(frame->width) * frame->height;
        if (++count > 4096 || frame->width < 1 || frame->height < 1 ||
            pixels > 4 * 1024 * 1024 ||
            pixels > 512 * 1024 * 1024 - budget.pixels)
          throw std::invalid_argument(
              "embedding video exceeds decoded work bounds");
        budget.pixels += pixels;
        consume(count - 1, frame);
        FFmpeg().av_frame_unref(frame);
      }
    };
    int status;
    while ((status = FFmpeg().av_read_frame(format, packet)) >= 0) {
      budget.Check();
      if (packet->stream_index == stream) {
        if (FFmpeg().avcodec_send_packet(codec, packet) < 0)
          throw std::invalid_argument("embedding video packet decoding failed");
        receive();
      }
      FFmpeg().av_packet_unref(packet);
    }
    if (status != AVERROR_EOF ||
        FFmpeg().avcodec_send_packet(codec, nullptr) < 0)
      throw std::invalid_argument(
          "embedding video is truncated or decoding failed");
    receive();
    return count;
  }
};
}  // namespace

std::vector<std::size_t> SampleVideoFrames(std::size_t frames, double fps,
                                           const VideoOptions& options) {
  if (!frames || frames > 4096 || !std::isfinite(fps) || fps <= 0 ||
      !std::isfinite(options.fps) || options.fps <= 0 || options.fps > 120 ||
      !options.max_frames || options.max_frames > 32 || frames / fps > 300)
    throw std::invalid_argument(
        "invalid embedding video rate, duration or frame count");
  const auto sampled = std::max<std::size_t>(
      1, static_cast<std::size_t>(frames / fps * options.fps));
  const auto kept = std::min(sampled, options.max_frames);
  std::vector<std::size_t> indices;
  // NumPy linspace multiplies by a double step, then floors for dtype=int.
  // Rational integer division differs at floating rounding boundaries.
  const double step =
      kept > 1 ? static_cast<double>(sampled - 1) / (kept - 1) : 0;
  for (std::size_t j = 0; j < kept; ++j) {
    const auto i = sampled <= kept || options.truncate_overflow ? j
                   : j + 1 == kept && kept > 1
                       ? sampled - 1
                       : static_cast<std::size_t>(j * step);
    const auto index =
        i == 0 ? 0
               : std::min(frames - 1,
                          static_cast<std::size_t>(i * (fps / options.fps)));
    indices.push_back(index);
  }
  return indices;
}

bool ValidateVideoProcessor(const std::string& model_dir, std::string* error) {
  try {
    std::ifstream file(std::filesystem::path(model_dir) /
                       "processor_config.json");
    const auto root =
        json::parse(std::string(std::istreambuf_iterator<char>(file), {}));
    const auto* c = root.find("video_processor");
    const auto flag = [&](const char* name, bool expected) {
      const auto* v = c ? c->find(name) : nullptr;
      return v && v->is_bool() && v->as_bool() == expected;
    };
    if (c &&
        c->member_str("video_processor_type") ==
            "EmbeddingGemma2VideoProcessor" &&
        flag("do_convert_rgb", true) && flag("do_resize", true) &&
        flag("do_rescale", true) && flag("do_normalize", false) &&
        flag("do_sample_frames", true) && flag("add_timestamps", false) &&
        c->member_double("fps") == 1 && c->member_size("max_frames") == 32 &&
        c->member_str("overflow_strategy") == "uniform" &&
        c->member_size("max_soft_tokens") == 140 &&
        c->member_size("patch_size") == 16 &&
        c->member_size("pooling_kernel_size") == 3 &&
        c->member_size("resample") == 3 &&
        std::abs(c->member_double("rescale_factor") - 1.0 / 255) < 1e-15)
      return true;
  } catch (const std::exception&) {
  }
  if (error)
    *error =
        "embedding_gemma2: unsupported or missing video processor "
        "configuration";
  return false;
}

std::vector<VideoFrame> ProcessVideo(std::span<const std::uint8_t> bytes,
                                     const VideoOptions& options,
                                     const std::function<bool()>& is_cancelled,
                                     std::size_t frame_budget) {
  if (bytes.empty() || bytes.size() > 20 * 1024 * 1024)
    throw std::invalid_argument("embedding video is empty or exceeds 20 MiB");
  Budget budget{is_cancelled};
  budget.Check();
  (void)FFmpeg();  // Finish loading before a Decoder can need cleanup.
  std::size_t count;
  double fps;
  {
    Decoder decoder{bytes, budget};
    decoder.Open();
    fps = av_q2d(decoder.format->streams[decoder.stream]->avg_frame_rate);
    count = decoder.Decode([](std::size_t, const AVFrame*) {});
    const auto declared = decoder.format->streams[decoder.stream]->nb_frames;
    if (declared > 0 && static_cast<std::size_t>(declared) != count)
      throw std::invalid_argument(
          "embedding video frame count differs from container; truncated "
          "input");
  }
  const auto selected = SampleVideoFrames(count, fps, options);
  if (selected.size() > frame_budget)
    throw std::invalid_argument(
        "embedding request video frame budget exhausted");
  Decoder decoder{bytes, budget};
  decoder.Open();
  std::vector<VideoFrame> frames;
  decoder.Decode([&](std::size_t index, const AVFrame* frame) {
    if (frames.size() >= selected.size() || index != selected[frames.size()])
      return;
    std::unique_ptr<SwsContext, decltype(FFmpeg().sws_freeContext)> conversion(
        FFmpeg().sws_getContext(frame->width, frame->height,
                                static_cast<AVPixelFormat>(frame->format),
                                frame->width, frame->height, AV_PIX_FMT_RGB24,
                                SWS_BICUBIC, nullptr, nullptr, nullptr),
        FFmpeg().sws_freeContext);
    if (!conversion)
      throw std::invalid_argument("embedding video RGB conversion failed");
    core::Image image{
        static_cast<std::uint32_t>(frame->width),
        static_cast<std::uint32_t>(frame->height),
        std::vector<std::uint8_t>(static_cast<std::size_t>(frame->width) *
                                  frame->height * 3)};
    std::uint8_t* output[] = {image.pixels.data()};
    int stride[] = {frame->width * 3};
    if (FFmpeg().sws_scale(conversion.get(), frame->data, frame->linesize, 0,
                           frame->height, output, stride) != frame->height)
      throw std::invalid_argument("embedding video RGB conversion failed");
    auto patches = ProcessImage(image, 140);
    do {
      frames.push_back({index, index / fps, patches});
    } while (frames.size() < selected.size() &&
             selected[frames.size()] == index);
    budget.Check();
  });
  if (frames.size() != selected.size())
    throw std::invalid_argument("embedding video lost selected frames");
  return frames;
}

std::string FrameVideo(const std::vector<VideoFrame>& frames,
                       const VideoOptions& options) {
  std::string text;
  for (const auto& frame : frames) {
    if (options.timestamps) {
      if (!text.empty())
        text += ' ';
      char timestamp[32];
      const int seconds = static_cast<int>(frame.timestamp);
      std::snprintf(timestamp, sizeof(timestamp), "%02d:%02d ", seconds / 60,
                    seconds % 60);
      text += timestamp;
    }
    text += kBeginImageToken;
    for (std::size_t j = 0; j < frame.patches.soft_tokens(); ++j)
      text += kVideoToken;
    text += kEndImageToken;
  }
  return text;
}
}  // namespace gufo::models::embedding_gemma2
