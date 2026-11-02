# Realtime speech and decision contracts

Add realtime speech, speech CLI and decision contracts:
- `asr.hpp`: `LiveTranscription` and `VoiceConversation` stream contracts; optional `AsrRunner` methods `CreateLiveTranscription`, `CreateVoiceConversation`, their `Supports*` probes and `ModelId`; `AsrRunnerOptions` with a projector sidecar path and `AsrProvider::create_with_options`. `gufo serve asr` serves `/v1/realtime` live transcription and conversations (with the `/voice` page) for runners that support them, and `gufo transcribe --live` uses live transcription.
- `tts.hpp`: optional `TtsRunner` methods `ModelId` and `Concurrent` (the runner admits concurrent requests itself); `TtsRunnerOptions` and `TtsProvider::create_with_options`; `SpeakOptions` and `TtsProvider::speak_driver` behind `gufo speak`.
- `decision.hpp` and `registry.hpp`: the `decision` capability (`Capability::kDecision`, `"capability": "decision"` in metadata), `DecisionRunner`, `DecisionRunnerOptions`, `DecisionProvider` and `Registry::AddDecisionProvider`, behind `gufo decide` and `gufo serve decision` (`/v1/systemone`).
- `serve asr`, `serve tts` and `transcribe` accept `--mmproj` for providers with `create_with_options`.
All additions are optional for existing providers.
