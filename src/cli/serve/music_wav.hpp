#ifndef GUFO_SERVER_MUSIC_WAV_HPP_
#define GUFO_SERVER_MUSIC_WAV_HPP_

#include <string>

#include "src/ext/music.hpp"

namespace gufo::server {

/// RIFF WAV of frame-interleaved music: PCM16 by default, IEEE float32 with
/// `float32`. Empty on invalid input (no channels, NaN, over 4 GiB).
[[nodiscard]] std::string EncodeMusicWav(const ext::MusicAudio& audio,
                                         bool float32);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MUSIC_WAV_HPP_
