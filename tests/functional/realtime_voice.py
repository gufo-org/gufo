#!/usr/bin/env python3
"""Realtime WebSocket checks for a model that hears and speaks (LFM3-Audio).

Starts an owned `gufo serve asr` with an LFM3-Audio GGUF pair and drives
`/v1/realtime` as a client streaming a microphone would:

  * live transcription (intent=transcription): text deltas arrive while the
    audio is still being appended; clear discards an utterance;
  * conversation sessions: a spoken question is prefilled while it arrives,
    the reply streams transcript and 24 kHz speech, which the same server
    transcribes back; a typed follow-up continues the context; a spoken
    request calls a function tool whose output the model then answers;
    response.cancel stops a reply; text replies may reason first.

The spoken prompts are synthesized first with `gufo speak` (greedy speech
codes), so no audio is downloaded. Every event is retained in the evidence
directory. Timings are observations, not a qualified comparison.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import traceback
import wave

from run import provenance, server
from service_http import Evidence, sha256

HERE = Path(__file__).resolve().parent
RATE = 24000


class WebSocket:
    """Minimal RFC 6455 text client (masked frames, no extensions)."""

    def __init__(self, port, path, timeout=120):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                           'Upgrade: websocket\r\nConnection: Upgrade\r\n'
                           f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        head = b''
        while b'\r\n\r\n' not in head:
            chunk = self.sock.recv(4096)
            assert chunk, 'connection closed during the upgrade'
            head += chunk
        head, self.buffer = head.split(b'\r\n\r\n', 1)
        assert head.startswith(b'HTTP/1.1 101'), head.decode(errors='replace')
        self.events = []

    def send(self, event):
        data = json.dumps(event).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(data) < 126:
            header += bytes([0x80 | len(data)])
        elif len(data) < 65536:
            header += bytes([0x80 | 126]) + struct.pack('>H', len(data))
        else:
            header += bytes([0x80 | 127]) + struct.pack('>Q', len(data))
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n):
        while len(self.buffer) < n:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError('server closed the WebSocket')
            self.buffer += chunk
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def receive(self, timeout=None):
        """Next text event as a dict, or None on timeout."""
        self.sock.settimeout(timeout)
        try:
            while True:
                b0, b1 = self._read(2)
                n = b1 & 0x7F
                if n == 126:
                    n = struct.unpack('>H', self._read(2))[0]
                elif n == 127:
                    n = struct.unpack('>Q', self._read(8))[0]
                payload = self._read(n)
                opcode = b0 & 0x0F
                if opcode == 1:
                    event = json.loads(payload)
                    event['_received'] = time.perf_counter()
                    self.events.append(event)
                    return event
                if opcode == 8:
                    raise ConnectionError('server closed the WebSocket')
        except socket.timeout:
            return None
        finally:
            self.sock.settimeout(120)

    def until(self, kind, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = self.receive(timeout=max(0.01, deadline - time.monotonic()))
            if event is None:
                break
            if event['type'] == kind:
                return event
            if event['type'] == 'error':
                raise AssertionError(f'server error: {event["error"]}')
        raise AssertionError(f'no {kind} within {timeout} s')

    def drain(self):
        while self.receive(timeout=0.05) is not None:
            pass

    def close(self):
        self.sock.close()


def read_pcm(path):
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        frames = w.readframes(w.getnframes())
        rate = w.getframerate()
    if rate != RATE:
        raise ValueError(f'{path} must be 24 kHz mono PCM16')
    return frames


def write_wav(path, pcm):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def stream(ws, pcm, pace=0.25):
    """Appends 100 ms pieces, receiving events meanwhile; pace is wall seconds
    per audio second. Returns when the last piece was sent."""
    piece = RATE // 10 * 2
    for at in range(0, len(pcm), piece):
        ws.send({'type': 'input_audio_buffer.append',
                 'audio': base64.b64encode(pcm[at:at + piece]).decode()})
        deadline = time.perf_counter() + 0.1 * pace
        while ws.receive(timeout=max(0.0005, deadline - time.perf_counter())) is not None:
            if ws.events[-1]['type'] == 'error':
                raise AssertionError(f'server error: {ws.events[-1]["error"]}')
    return time.perf_counter()


def silence(seconds):
    """PCM16 zeros."""
    return bytes(2 * int(RATE * seconds))


def seen(ws, kind, since, timeout=120):
    """The first kind event received after since, waiting for it if needed."""
    for event in ws.events:
        if event['type'] == kind and event['_received'] >= since:
            return event
    return ws.until(kind, timeout)


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def save_events(output, name, ws):
    with (output / f'{name}.events.jsonl').open('w') as f:
        for event in ws.events:
            event = dict(event)
            for key in ('delta',):
                if event.get('type') == 'response.output_audio.delta':
                    event[key] = f'<{len(event[key])} base64 bytes>'
            f.write(json.dumps(event) + '\n')


def respond(ws, **response):
    """Sends response.create and collects the reply until response.done."""
    sent = time.perf_counter()
    ws.send({'type': 'response.create', **({'response': response} if response else {})})
    audio, transcript, text, reasoning, first = bytearray(), '', '', '', {}
    while True:
        event = ws.receive(timeout=180)
        assert event is not None, 'no response.done'
        kind = event['type']
        if kind == 'error':
            raise AssertionError(f'server error: {event["error"]}')
        if kind.endswith('.delta'):
            first.setdefault(kind, (event['_received'] - sent) * 1000)
        if kind == 'response.output_audio.delta':
            audio += base64.b64decode(event['delta'])
        elif kind == 'response.output_audio_transcript.delta':
            transcript += event['delta']
        elif kind == 'response.output_text.delta':
            text += event['delta']
        elif kind == 'response.reasoning_text.delta':
            reasoning += event['delta']
        elif kind == 'response.done':
            return {'response': event['response'], 'audio': bytes(audio),
                    'transcript': transcript, 'text': text, 'reasoning': reasoning,
                    'first_ms': first, 'total_ms': (event['_received'] - sent) * 1000}


def transcribe(port, pcm, pace=0):
    ws = WebSocket(port, '/v1/realtime?intent=transcription')
    ws.until('session.created')
    stream(ws, pcm, pace)
    ws.send({'type': 'input_audio_buffer.commit'})
    done = ws.until('conversation.item.input_audio_transcription.completed', timeout=180)
    ws.close()
    return done['transcript']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='LFM3-Audio backbone GGUF')
    parser.add_argument('--mmproj', type=Path, required=True, help='LFM3-Audio audio towers GGUF')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, default=HERE / 'fixtures/audio')
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--browser', type=Path,
                        help='Chromium binary: also drive the /voice page headless with the spoken '
                             'question as a fake microphone')
    args = parser.parse_args()
    for path in (args.binary, args.model, args.mmproj):
        if not path.is_file():
            parser.error(f'{path} must exist; missing weights are not a pass')
    binary, model, mmproj = (p.resolve() for p in (args.binary, args.model, args.mmproj))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {**provenance(), 'schema': 1, 'service': 'realtime-voice', 'status': 'running',
              'model_sha256': {p.name: sha256(p) for p in (model, mmproj)},
              'binary_sha256': sha256(binary),
              'source_revision': subprocess.check_output(
                  ['git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
              'harness_sha256': sha256(Path(__file__)),
              'timing_qualification': 'observations only; no matched baseline comparison'}
    evidence = Evidence(output, report)
    evidence.save()
    try:
        prompts = {'question': 'What is the capital of France?',
                   'volume': 'Set the volume to thirty, please.'}
        with evidence.case('synthesize-prompts'):
            for name, text in prompts.items():
                command = [str(binary), 'speak', '--model', str(model), '--mmproj', str(mmproj),
                           '--text', text, '--temperature', '0', '--output', str(output / f'{name}.wav')]
                subprocess.run(command, check=True, capture_output=True, timeout=300)
            report['prompts'] = prompts
        question = read_pcm(output / 'question.wav')
        volume = read_pcm(output / 'volume.wav')
        expected = (args.audio_root / 'clear-english-voice.txt').read_text().strip()
        with wave.open(str(args.audio_root / 'clear-english-voice.wav')) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            fixture16 = w.readframes(w.getnframes())
        # 16 kHz fixture to 24 kHz PCM16 by linear interpolation: the check is
        # the live protocol, not resampling fidelity.
        samples = struct.unpack(f'<{len(fixture16) // 2}h', fixture16)
        upsampled = []
        for i in range(len(samples) * 3 // 2):
            t = i * 2 / 3
            j = int(t)
            a, b = samples[j], samples[min(j + 1, len(samples) - 1)]
            upsampled.append(int(round(a + (b - a) * (t - j))))
        fixture = struct.pack(f'<{len(upsampled)}h', *upsampled)

        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        command = [str(binary), 'serve', 'asr', '--model', str(model), '--mmproj', str(mmproj),
                   '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'functional-voice']
        report['command'] = command
        with server(command, output / 'server.log', args.startup_timeout) as process:
            with evidence.case('live-transcription') as row:
                ws = WebSocket(port, '/v1/realtime?intent=transcription')
                ws.until('session.created')
                # Four times real time: deltas must precede the commit.
                last = stream(ws, fixture, pace=0.25)
                early = [e for e in ws.events
                         if e['type'] == 'conversation.item.input_audio_transcription.delta']
                ws.send({'type': 'input_audio_buffer.commit'})
                done = ws.until('conversation.item.input_audio_transcription.completed')
                row.update(transcript=done['transcript'], early_deltas=len(early),
                           completed_after_last_append_ms=(done['_received'] - last) * 1000)
                save_events(output, 'live-transcription', ws)
                assert early, 'no transcript delta arrived while audio was streaming'
                heard, want = words(done['transcript']), words(expected)
                assert heard[:len(want)] == want and len(heard) <= len(want) + 2, (heard, want)
                ids = {e.get('item_id') for e in ws.events if 'transcription' in e['type']}
                assert len(ids) == 1, ids
                # Clearing drops an utterance; the next one is transcribed alone.
                # Live ASR writes a word once audio after it has arrived, as a
                # microphone delivers before the client commits.
                stream(ws, fixture[:RATE * 2], pace=0)
                ws.send({'type': 'input_audio_buffer.clear'})
                ws.until('input_audio_buffer.cleared')
                stream(ws, question + silence(0.5), pace=0)
                ws.send({'type': 'input_audio_buffer.commit'})
                again = ws.until('conversation.item.input_audio_transcription.completed')
                row['after_clear'] = again['transcript']
                assert words(again['transcript']) == words(prompts['question']), again['transcript']
                # A transcription prompt selects the offline path at commit.
                ws.send({'type': 'session.update', 'session': {'type': 'transcription', 'audio': {
                    'input': {'transcription': {'prompt': 'An audiobook.'}}}}})
                ws.until('session.updated')
                stream(ws, fixture, pace=0)
                ws.send({'type': 'input_audio_buffer.commit'})
                offline = ws.until('conversation.item.input_audio_transcription.completed')
                row['offline'] = offline['transcript']
                assert words(offline['transcript']) == want, offline['transcript']
                ws.close()

            with evidence.case('transcription-turn-detection') as row:
                ws = WebSocket(port, '/v1/realtime?intent=transcription')
                ws.until('session.created')
                ws.send({'type': 'session.update', 'session': {'type': 'transcription', 'audio': {
                    'input': {'turn_detection': {'type': 'server_vad', 'silence_duration_ms': 800}}}}})
                ws.until('session.updated')
                stream(ws, silence(0.5) + fixture + silence(1.5), pace=0.25)
                # Every turn the server ends is transcribed without a commit.
                count = lambda kind: sum(e['type'] == kind for e in ws.events)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and (
                        not count('input_audio_buffer.speech_stopped') or
                        count('conversation.item.input_audio_transcription.completed') <
                        count('input_audio_buffer.speech_stopped')):
                    ws.receive(timeout=1)
                done = [e['transcript'] for e in ws.events
                        if e['type'] == 'conversation.item.input_audio_transcription.completed']
                kinds = [e['type'] for e in ws.events]
                row.update(turns=len(done), transcripts=done)
                save_events(output, 'transcription-turn-detection', ws)
                assert 'input_audio_buffer.speech_started' in kinds, kinds
                assert 'input_audio_buffer.commit' not in kinds
                heard = words(' '.join(done))
                assert heard[:len(want)] == want and len(heard) <= len(want) + 2, (heard, want)
                ws.close()

            with evidence.case('hands-free-question') as row:
                ws = WebSocket(port, '/v1/realtime')
                ws.until('session.created')
                ws.send({'type': 'session.update', 'session': {'audio': {'input': {
                    'turn_detection': {'type': 'server_vad', 'silence_duration_ms': 500}}}}})
                ws.until('session.updated')
                last = stream(ws, silence(0.5) + question + silence(1.0), pace=1.0)
                stopped = seen(ws, 'input_audio_buffer.speech_stopped', 0)
                event = seen(ws, 'response.done', stopped['_received'])
                deltas = [e for e in ws.events if e['_received'] >= stopped['_received']]
                audio = [e for e in deltas if e['type'] == 'response.output_audio.delta']
                transcript = ''.join(e['delta'] for e in deltas
                                     if e['type'] == 'response.output_audio_transcript.delta')
                row.update(transcript=transcript, speech_stopped_ms=stopped['audio_end_ms'],
                           stop_to_first_audio_ms=(audio[0]['_received'] - stopped['_received']) * 1000
                           if audio else None)
                assert event['response']['status'] == 'completed', event['response']
                assert 'paris' in words(transcript), transcript

                # Barge-in: speaking over a reply stops it; the new question is
                # answered and the interrupted reply keeps what was heard.
                ws.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user', 'content': [{
                        'type': 'input_text', 'text': 'Tell me a long story about a dragon.'}]}})
                ws.until('conversation.item.created')
                ws.send({'type': 'response.create'})
                story = ws.until('response.output_item.added')['item']['id']
                ws.until('response.output_audio.delta')
                sent = time.perf_counter()
                # The client truncates as soon as the server hears speech,
                # while the user is still talking.
                half = len(volume) // 2 // 2 * 2
                stream(ws, volume[:half], pace=1.0)
                started = seen(ws, 'input_audio_buffer.speech_started', sent)
                ws.send({'type': 'conversation.item.truncate', 'item_id': story,
                         'content_index': 0, 'audio_end_ms': 400})
                stream(ws, volume[half:] + silence(1.0), pace=1.0)
                cancelled = seen(ws, 'response.done', sent)
                row['barge_in_cancel_ms'] = (cancelled['_received'] - started['_received']) * 1000
                assert cancelled['response']['status'] == 'cancelled', cancelled['response']
                seen(ws, 'conversation.item.truncated', sent)
                answer = ws.until('response.done', timeout=120)
                said = ''.join(e['delta'] for e in ws.events
                               if e['type'] == 'response.output_audio_transcript.delta'
                               and e['_received'] > cancelled['_received'])
                row['after_barge_in'] = said
                assert answer['response']['status'] == 'completed', answer['response']
                assert said, 'no reply after the interruption'
                save_events(output, 'hands-free', ws)
                ws.close()

            if args.browser:
                with evidence.case('voice-page') as row:
                    mic = output / 'page-microphone.wav'
                    write_wav(mic, silence(1.0) + question + silence(4.0))
                    profile = output / 'browser-profile'
                    log = output / 'browser.log'
                    with log.open('w') as stderr:
                        try:
                            subprocess.run(
                                [str(args.browser), '--headless=new', '--no-sandbox',
                                 f'--user-data-dir={profile}', '--use-fake-ui-for-media-stream',
                                 '--use-fake-device-for-media-stream',
                                 f'--use-file-for-fake-audio-capture={mic}',
                                 '--autoplay-policy=no-user-gesture-required',
                                 '--enable-logging=stderr', '--v=0', '--remote-debugging-port=0',
                                 f'http://localhost:{port}/voice#connect'],
                                stdout=subprocess.DEVNULL, stderr=stderr, timeout=25)
                        except subprocess.TimeoutExpired:
                            pass  # the page runs until the browser is stopped
                    replies = re.findall(r'"reply (\w+): ([^"]*)', log.read_text(errors='replace'))
                    row['replies'] = replies
                    assert replies and replies[0][0] == 'completed', replies
                    assert 'paris' in words(replies[0][1]), replies

            ws = WebSocket(port, '/v1/realtime')
            created = ws.until('session.created')
            assert created['session']['type'] == 'realtime', created
            with evidence.case('spoken-question') as row:
                last = stream(ws, question, pace=1.0)
                ws.send({'type': 'input_audio_buffer.commit'})
                ws.until('conversation.item.created')
                reply = respond(ws)
                write_wav(output / 'reply.wav', reply['audio'])
                row.update(transcript=reply['transcript'], first_ms=reply['first_ms'],
                           total_ms=reply['total_ms'], audio_seconds=len(reply['audio']) / 2 / RATE)
                assert reply['response']['status'] == 'completed', reply['response']
                assert 'paris' in words(reply['transcript']), reply['transcript']
                assert len(reply['audio']) > RATE, 'reply speech is shorter than 0.5 s'
                kinds = [item['type'] for item in reply['response']['output']]
                assert kinds == ['message'], kinds
            with evidence.case('reply-speech-transcribes-back') as row:
                heard = transcribe(port, reply['audio'])
                row['heard'] = heard
                said, got = words(reply['transcript']), words(heard)
                common = sum(1 for w in said if w in got)
                assert common >= 0.8 * len(said), (said, got)
            with evidence.case('typed-follow-up-text') as row:
                ws.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'And what about Italy?'}]}})
                ws.until('conversation.item.created')
                reply = respond(ws, output_modalities=['text'])
                row.update(text=reply['text'], first_ms=reply['first_ms'])
                assert reply['response']['status'] == 'completed', reply['response']
                assert 'rome' in words(reply['text']), reply['text']
                assert '<' not in reply['text'], reply['text']
                assert not reply['audio'], 'text reply carried audio'
            with evidence.case('spoken-tool-call') as row:
                ws.send({'type': 'session.update', 'session': {'tools': [{
                    'type': 'function', 'name': 'set_volume',
                    'description': 'Set the device speaker volume.',
                    'parameters': {'type': 'object', 'properties': {
                        'volume': {'type': 'integer', 'description': 'Volume from 0 to 100.'}},
                        'required': ['volume']}}]}})
                ws.until('session.updated')
                stream(ws, volume, pace=0)
                ws.send({'type': 'input_audio_buffer.commit'})
                ws.until('conversation.item.created')
                reply = respond(ws)
                calls = [item for item in reply['response']['output'] if item['type'] == 'function_call']
                row['calls'] = calls
                assert len(calls) == 1 and calls[0]['name'] == 'set_volume', reply['response']
                assert json.loads(calls[0]['arguments']) == {'volume': 30}, calls[0]
                ws.send({'type': 'conversation.item.create', 'item': {
                    'type': 'function_call_output', 'call_id': calls[0]['call_id'],
                    'output': json.dumps({'volume': 30, 'status': 'ok'})}})
                ws.until('conversation.item.created')
                answer = respond(ws)
                write_wav(output / 'tool-answer.wav', answer['audio'])
                row['answer'] = answer['transcript']
                assert answer['response']['status'] == 'completed', answer['response']
                assert answer['transcript'] and answer['audio'], answer['response']
            with evidence.case('cancel-reply') as row:
                ws.send({'type': 'session.update', 'session': {'tools': []}})
                ws.until('session.updated')
                ws.send({'type': 'conversation.item.create', 'item': {
                    'type': 'message', 'role': 'user', 'content': [{
                        'type': 'input_text', 'text': 'Tell me a long story about a dragon.'}]}})
                ws.until('conversation.item.created')
                ws.send({'type': 'response.create'})
                ws.until('response.output_audio.delta')
                cancelled = time.perf_counter()
                ws.send({'type': 'response.cancel'})
                done = ws.until('response.done')
                row['cancel_to_done_ms'] = (done['_received'] - cancelled) * 1000
                assert done['response']['status'] == 'cancelled', done['response']
                assert row['cancel_to_done_ms'] < 1000, row
            with evidence.case('text-reasoning') as row:
                ws.send({'type': 'session.update', 'session': {
                    'output_modalities': ['text'], 'reasoning': {'effort': 'low'}}})
                ws.until('session.updated')
                stream(ws, question, pace=0)
                ws.send({'type': 'input_audio_buffer.commit'})
                ws.until('conversation.item.created')
                reply = respond(ws)
                row.update(reasoning=reply['reasoning'], text=reply['text'])
                assert reply['response']['status'] == 'completed', reply['response']
                assert reply['reasoning'].strip(), 'no reasoning was streamed'
                assert 'paris' in words(reply['text']), reply['text']
            with evidence.case('rejections') as row:
                ws.send({'type': 'session.update', 'session': {
                    'audio': {'input': {'turn_detection': {'type': 'semantic_vad'}}}}})
                row['turn_detection'] = ws.until('error')['error']['message']
                assert 'semantic_vad' in row['turn_detection'], row
                ws.send({'type': 'response.cancel'})
                row['idle_cancel'] = ws.until('error')['error']['code']
                assert row['idle_cancel'] == 'response_cancel_not_active', row
            save_events(output, 'conversation', ws)
            ws.close()
            assert process.poll() is None, 'server exited unexpectedly during checks'
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        evidence.save()
    print('PASS realtime-voice', output, flush=True)


if __name__ == '__main__':
    main()
