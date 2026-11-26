# Voice conversation voices and turn deletion

Voice conversations (`src/ext/asr.hpp`):
- `VoiceConversation::Settings::voice` carries a Realtime client's
  `audio.output.voice`. It is empty for the conversation's own voice, or one
  of `voices()`.
- `VoiceConversation::voices()` lists the voices a client may select. The
  default lists none, and `session.update` rejects any named
  `audio.output.voice`.
- `VoiceConversation::DeleteUserTurn()` removes the latest user turn when it
  is closed and nothing followed it (`conversation.item.delete`). The default
  fails, and the client receives an error event.

Existing conversations keep compiling and behave as before. They must be
rebuilt against the new header.
