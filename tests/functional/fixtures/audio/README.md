# English speech fixture

The repository maintainer supplied this English recording and authorized its
publication for ASR and TTS voice-cloning functional tests. The sibling text is
the reviewed expected transcript. No additional recording is needed: ASR
concurrency pairs the speech with silence generated in memory.

- WAV SHA-256: `41ff28fbea935f45dae8a760100609bf39a64dccbedc742edb4f0711d516ef5a`
- Mono PCM16, 16 kHz, 10 seconds.

Tests never download voice recordings or depend on a mounted audio folder.

`clear-english-voice.mp3` is the same recording encoded as 16 kHz mono MPEG
layer III (SHA-256
`0ca4a34db89f3faaaa01001a38ac77a7059414872994cfb0924d1ae235938991`), so ASR
can check that an MP3 upload gives the WAV's transcript. Generated with
python-soundfile 0.14.0 (libsndfile 1.2.2, LAME):

```python
import soundfile as sf
data, rate = sf.read('clear-english-voice.wav', dtype='int16')
sf.write('clear-english-voice.mp3', data, rate, format='MP3',
         subtype='MPEG_LAYER_III', compression_level=0.5,
         bitrate_mode='CONSTANT')
```
