# Audio decoder fixtures

These generated mono Vorbis tones exercise chained Ogg decoding in
`audio_decode_test`. The first stream has three audio pages, so damaging the
middle audio page leaves the final page's declared length available for the
truncation check. The second stream uses a different tone and duration.

Generated with FFmpeg 8.1.2 and libvorbis:

```sh
ffmpeg -v error -f lavfi \
  -i 'sine=frequency=440:sample_rate=48000:duration=2.2' \
  -c:a libvorbis -fflags +bitexact -y tests/fixtures/audio/vorbis-first.ogg
ffmpeg -v error -f lavfi \
  -i 'sine=frequency=880:sample_rate=48000:duration=0.2' \
  -c:a libvorbis -fflags +bitexact -y tests/fixtures/audio/vorbis-second.ogg
```

The test concatenates the streams in memory and checks that only the first
stream is returned, matching libsndfile. No encoder is needed to run the test.
