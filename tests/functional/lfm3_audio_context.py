#!/usr/bin/env python3
"""Exercise LFM3-Audio realtime history near its native 32768-slot limit.

Runs separately from the short voice suite: verifies actual input-token usage,
encoder prefill after 30K text tokens, and eviction/recovery at capacity.
"""
import argparse
from pathlib import Path
import socket
import time

from realtime_voice import WebSocket, respond, save_events, silence, source_revision, stream
from run import provenance, server
from service_http import Evidence, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'mmproj', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    binary, model, mmproj = (p.resolve() for p in (args.binary, args.model, args.mmproj))
    for path in (binary, model, mmproj):
        if not path.is_file():
            parser.error(f'missing {path}; missing weights are not a pass')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {**provenance(), 'service': 'lfm3-audio-context', 'status': 'running',
              'source_revision': source_revision(), 'binary_sha256': sha256(binary), 'harness_sha256': sha256(Path(__file__)),
              'model_sha256': {p.name: sha256(p) for p in (model, mmproj)},
              'timing_qualification': 'observations only; no matched speed claim'}
    evidence = Evidence(output, report)
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    command = [str(binary), 'serve', 'asr', '--model', str(model), '--mmproj', str(mmproj),
               '--host', '127.0.0.1', '--port', str(port)]
    report['command'] = command
    ws = None

    def text(message):
        sent = time.perf_counter()
        ws.send({'type': 'conversation.item.create', 'item': {
            'type': 'message', 'role': 'user',
            'content': [{'type': 'input_text', 'text': message}]}})
        ws.until('conversation.item.created', timeout=300)
        return (time.perf_counter() - sent) * 1000

    def answer(row):
        reply = respond(ws, output_modalities=['text'], max_output_tokens=32)
        row.update(text=reply['text'], response=reply['response'],
                   first_ms=reply['first_ms'], reply_ms=reply['total_ms'])
        assert reply['response']['status'] == 'completed', reply['response']
        row['usage'] = reply['response']['usage']
        assert reply['text'].strip() == 'OK', reply['text']
        assert row['usage']['total_tokens'] <= 32768, row
        return row['usage']['input_tokens']

    try:
        with server(command, output / 'server.log', 300):
            ws = WebSocket(port, '/v1/realtime', timeout=300)
            ws.until('session.created')
            ws.send({'type': 'session.update', 'session': {
                'instructions': 'Always reply with only OK.', 'output_modalities': ['text'],
                'max_output_tokens': 32, 'reasoning': {'effort': 'none'}}})
            ws.until('session.updated')
            with evidence.case('native-32k-history') as row:
                row['prefill_ms'] = text(' quiet' * 32000 + '\nReply with only OK.')
                depth = answer(row)
                assert 32000 < depth < 32768, row
            with evidence.case('audio-after-32k-text') as row:
                # Real encoder rows must append at positions above the old 8K
                # allocation, without resetting the retained text history.
                sent = time.perf_counter()
                stream(ws, silence(1), pace=0)
                ws.send({'type': 'input_audio_buffer.commit'})
                ws.until('conversation.item.created', timeout=300)
                row['prefill_ms'] = (time.perf_counter() - sent) * 1000
                audio_depth = answer(row)
                assert depth < audio_depth < 32768, row
            with evidence.case('evict-oldest-at-capacity') as row:
                # Fill the remaining headroom, then force room reclamation on
                # the following turn. The current input must remain usable.
                row['prefill_ms'] = text(' quiet' * 100 + '\nReply with only OK.')
                near_limit = answer(row)
                assert audio_depth < near_limit < 32768, row
                row['next_prefill_ms'] = text(' quiet' * 400 + '\nReply with only OK.')
                recovered = {}
                after = answer(recovered)
                row['recovered'] = recovered
                assert after < depth, row
            save_events(output, 'context', ws)
            ws.close()
            ws = WebSocket(port, '/v1/realtime', timeout=300)
            ws.until('session.created')
            # Leave the default 4096-token reply budget installed. A single
            # near-capacity input must survive even though it leaves less
            # room than that budget; response.create can ask for less.
            ws.send({'type': 'session.update', 'session': {
                'instructions': 'Always reply with only OK.',
                'output_modalities': ['text'], 'reasoning': {'effort': 'none'}}})
            ws.until('session.updated')
            with evidence.case('keep-current-input-near-32768') as row:
                row['prefill_ms'] = text(' quiet' * 32730 + '\nReply with only OK.')
                assert 32700 < answer(row) < 32768, row
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if ws:
            save_events(output, 'context', ws)
            ws.close()
        evidence.save()


if __name__ == '__main__':
    main()
