"""Real-model Qwen3-ASR HTTP transcription, isolation and streaming contracts."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import wave

from service_http import events, sha256, transcript_events, word_error_rate


def run(evidence, url, audio_root, model):
    fixtures = []
    for language in ('english',):
        path = audio_root / f'clear-{language}-voice.wav'
        text = path.with_suffix('.txt').read_text().strip()
        assert text, 'missing reviewed transcript'
        fixtures.append({'path': path, 'text': text, 'language': language.title(),
                         'sha256': sha256(path)})
    evidence.report['fixtures'] = [{**item, 'path': str(item['path'])} for item in fixtures]
    endpoint = url + '/v1/audio/transcriptions'

    def fields(item, **extra):
        return {'model': model, 'language': item['language'], 'max_tokens': '128', **extra}

    def request(item, label, **extra):
        return evidence.json(endpoint, label=label, data=fields(item, **extra),
                             files={'file': ('speech.wav', item['path'].read_bytes(), 'audio/wav')})

    references = []
    for item in fixtures:
        with evidence.case('transcribe-' + item['language']) as case:
            result = request(item, case['name'])
            wer = word_error_rate(result['text'], item['text'])
            case.update(transcript=result['text'], expected=item['text'], word_error_rate=wer,
                        maximum_word_error_rate=.15)
            assert wer <= .15, case
            references.append(result['text'])

    item = fixtures[0]
    with evidence.case('formats-and-repeat'):
        assert request(item, 'repeat')['text'] == references[0]
        verbose = request(item, 'verbose', response_format='verbose_json')
        assert verbose['text'] == references[0] and verbose['duration'] > 0, verbose
        status, body, _ = evidence.call(endpoint, label='text',
            data=fields(item, response_format='text'),
            files={'file': ('speech.wav', item['path'].read_bytes(), 'audio/wav')})
        assert status == 200 and body.decode().strip() == references[0].strip(), body

    with evidence.case('streaming-equals-buffered'):
        with evidence.request(endpoint, label='stream', data=fields(item, stream='true'),
                files={'file': ('speech.wav', item['path'].read_bytes(), 'audio/wav')}) as (response, row, path):
            assert response.status_code == 200 and 'text/event-stream' in response.headers['content-type']
            items = list(events(response, row, path))
            path.write_text(json.dumps(items, indent=2))
            assert transcript_events(items) == references[0]

    with evidence.case('silence'):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(32000))
        result = evidence.json(endpoint, label='silence', data={'model': model},
            files={'file': ('silence.wav', buffer.getvalue(), 'audio/wav')})
        assert not result['text'].strip(), result

    with evidence.case('concurrent-speech-and-silence'):
        def silent_peer():
            return evidence.json(endpoint, label='silent-peer', data={'model': model},
                files={'file': ('silence.wav', buffer.getvalue(), 'audio/wav')})
        with ThreadPoolExecutor(max_workers=2) as pool:
            speech = pool.submit(request, item, 'speech-peer')
            silence = pool.submit(silent_peer)
            assert speech.result()['text'] == references[0]
            assert not silence.result()['text'].strip()

    with evidence.case('rejections-and-recovery'):
        for options, code in [({'model': 'unknown'}, 'invalid_model'),
                              ({'temperature': '1'}, 'invalid_temperature'),
                              ({'response_format': 'srt'}, 'invalid_response_format'),
                              ({'max_tokens': '0'}, 'invalid_max_tokens')]:
            evidence.reject(endpoint, code, label=code, data=fields(item, **options),
                files={'file': ('speech.wav', item['path'].read_bytes(), 'audio/wav')})
        status, body, _ = evidence.call(endpoint, label='malformed-audio', data={'model': model},
            files={'file': ('broken.wav', b'not a WAV', 'audio/wav')})
        # Current API maps native WAV decode failures to transcription_failed.
        # Keep this limitation explicit rather than accepting arbitrary errors.
        assert status == 500 and json.loads(body)['error']['code'] == 'transcription_failed', (status, body)
        evidence.report['known_limitations'] = [
            'Malformed WAV returns HTTP 500 transcription_failed rather than HTTP 400.']
        assert request(item, 'recovered')['text'] == references[0]

    with evidence.case('cancel-stream-and-recover'):
        with evidence.request(endpoint, label='cancel', cancel=True, data=fields(item, stream='true'),
                files={'file': ('speech.wav', item['path'].read_bytes(), 'audio/wav')}) as (response, row, path):
            assert response.status_code == 200
            first = next(events(response, row, path))
            assert first['type'] == 'transcript.text.delta' and first['delta'], first
            path.write_text(json.dumps(first))
        assert request(item, 'after-cancel')['text'] == references[0]
