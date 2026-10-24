"""Real-model Qwen3-TTS audio, cloning, replay and HTTP streaming contracts."""

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

from service_http import audio_check, events, pcm_wav, sha256, speech_events, word_error_rate


TEXTS = ('The little brown dog ran across the garden.',
         'Please bring a glass of water to the table.')


def run(evidence, url, model, voice, variant, audio_root, asr_binary, asr_model):
    endpoint = url + '/v1/audio/speech'
    base = {'model': model, 'voice': voice, 'input': TEXTS[0], 'seed': 42,
            'language': 'English', 'max_new_tokens': 180}
    if variant == 'design':
        base['instructions'] = 'A clear English voice speaking calmly and naturally.'
    if variant == 'base':
        reference = audio_root / 'clear-english-voice.wav'
        evidence.report['reference_voice'] = {'path': str(reference), 'sha256': sha256(reference),
            'transcript': reference.with_suffix('.txt').read_text().strip()}
    evidence.report['evaluator'] = {'binary': str(asr_binary), 'binary_sha256': sha256(asr_binary),
                                     'model': str(asr_model)}

    def synth(label, **options):
        status, data, row = evidence.call(endpoint, label=label, json={**base, **options})
        assert status == 200, (status, data[:1000])
        pcm, measures = audio_check(data, wav=options.get('response_format', 'wav') == 'wav')
        row['audio'] = measures
        return pcm

    def intelligible(pcm, text, label):
        path = evidence.output / (label + '.wav')
        path.write_bytes(pcm_wav(pcm))
        command = [str(asr_binary), 'transcribe', '--model', str(asr_model),
                   '--audio', str(path), '--format', 'json']
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, timeout=180)
        (evidence.output / (label + '-asr.stdout')).write_bytes(result.stdout)
        (evidence.output / (label + '-asr.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0, result.stderr.decode(errors='replace')
        actual = json.loads(result.stdout)['text']
        wer = word_error_rate(actual, text)
        row = {'command': command, 'expected': text, 'transcript': actual,
               'word_error_rate': wer, 'maximum_word_error_rate': .25,
               'elapsed_ms': (time.perf_counter() - started) * 1000}
        evidence.report.setdefault('intelligibility', []).append(row)
        evidence.save()
        assert wer <= .25, row

    with evidence.case('voice-discovery'):
        voices = evidence.json(url + '/v1/audio/voices', label='voices', method='GET')
        assert voice in voices['voices'] and voice in [item['id'] for item in voices['data']], voices

    with evidence.case('synthesis-and-intelligibility'):
        first = synth('cold')
        intelligible(first, TEXTS[0], 'cold')

    with evidence.case('seeded-replay'):
        assert synth('warm') == first, 'same-seed warm replay changed PCM'

    with evidence.case('pcm-equals-wav'):
        assert synth('pcm', response_format='pcm') == first

    with evidence.case('sse-equals-wav'):
        with evidence.request(endpoint, label='sse', json={**base, 'response_format': 'pcm',
                'stream_format': 'sse'}) as (response, row, path):
            assert response.status_code == 200 and 'text/event-stream' in response.headers['content-type']
            items = list(events(response, row, path))
            path.write_text(json.dumps(items, indent=2))
            pcm = speech_events(items)
            audio_check(pcm, wav=False)
            assert pcm == first, 'SSE changed PCM'

    with evidence.case('concurrent-different-inputs'):
        second = synth('second-control', input=TEXTS[1], seed=43)
        intelligible(second, TEXTS[1], 'second')
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(synth, 'peer-first')
            b = pool.submit(synth, 'peer-second', input=TEXTS[1], seed=43)
            assert a.result() == first and b.result() == second, 'request state leaked between peers'
        assert first != second, 'different texts produced identical audio'

    with evidence.case('rejections-and-recovery'):
        for options, code in [({'model': 'unknown'}, 'invalid_model'),
                              ({'voice': 'unknown'}, 'invalid_voice'),
                              ({'input': ''}, 'missing_input'),
                              ({'max_new_tokens': 0}, 'invalid_max_new_tokens')]:
            evidence.reject(endpoint, code, label=code, json={**base, **options})
        assert synth('recovered') == first

    with evidence.case('cancel-stream-and-recover'):
        with evidence.request(endpoint, label='cancel', cancel=True,
                json={**base, 'response_format': 'pcm', 'stream_format': 'sse'}) as (response, row, path):
            assert response.status_code == 200
            first_event = next(events(response, row, path))
            assert first_event['type'] == 'speech.audio.delta' and first_event['audio'], first_event
            path.write_text(json.dumps(first_event))
        assert synth('after-cancel') == first
