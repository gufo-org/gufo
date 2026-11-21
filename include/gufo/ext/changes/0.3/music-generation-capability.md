# Music generation capability

Publish music generation contracts (`include/gufo/ext/music.hpp`):
- `Capability::kMusicGeneration` and `Registry::AddMusicProvider`; manifest
  providers declare `"capability": "music_generation"`, served by
  `gufo serve music` and run once by `gufo music`.
- `MusicProvider` (`probe`, `resolve`, `create_with_options`),
  `MusicRunnerOptions`, `RegisterMusicProvider`, `FindMusicProvider`,
  `ProbeMusicProvider`, and the commands `kMusicServeCommand` and
  `kMusicCommand` whose manifest options reach the runner.
- `MusicRunner`: `Generate`, `Capabilities`, `ValidateRequest`, `SampleRate`,
  `Channels`, `ModelId` and `Concurrent`.
- `MusicCapabilities` with `MusicSupport` and `MusicDurationControl`: which
  request fields a runner takes, whether it returns or accepts a symbolic
  score, plans without audio, and whether the length is exact, a maximum or
  the model's choice.
- `MusicRequest` (an absent seed leaves the default to the runner),
  `MusicResult`, `MusicFinish`, `MusicAudio` (an alias of
  `TtsAudioBuffer`, frame-interleaved) and `MusicProgress` (phase plus counts;
  a total of zero is unknown).
- `ValidateMusicRequest`, the capability check the core applies before
  inference.

Additive; existing extensions rebuild unchanged.
