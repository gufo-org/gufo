# Media provider contracts

Publish capability-neutral contracts and provider registries:
- ASR (`include/gufo/ext/asr.hpp`): `AsrSegment`, `AsrProgress`, `AsrRunner`, `AsrProvider`,
  `RegisterAsrProvider`, `FindAsrProvider`, `ProbeAsrProvider`, `Registry::AddAsrProvider`.
- TTS (`include/gufo/ext/tts.hpp`): `TtsAudioFormat`, `TtsAudioChunk`, `TtsProgress`,
  `TtsRunner`, `TtsProvider`, `RegisterTtsProvider`, `FindTtsProvider`, `ProbeTtsProvider`,
  `Registry::AddTtsProvider`.
- Video (`include/gufo/ext/video.hpp`): `VideoParameters`, `VideoRequest`, `VideoTelemetry`,
  `VideoRunner`, `VideoProvider`, `RegisterVideoProvider`, `FindVideoProvider`, `ProbeVideoProvider`,
  `Registry::AddVideoProvider`.
- Image (`include/gufo/ext/image.hpp`): Provider registration and discovery functions
  `RegisterImageProvider`, `FindImageProvider`, `ProbeImageProvider`, `Registry::AddImageProvider`.
- Platform Process (`include/gufo/ext/platform/process.hpp`): Subprocess execution contracts
  `RunMediaMux`, `RunCapture`.
- Extension migrations for Cards 15–18: Qwen3-ASR (`qwen3_asr`), Qwen3-TTS (`qwen3_tts`),
  Qwen-Image-2.1 (`qwen_image_21`), MiniMax H3 (`minimax_h3`).
