#!/usr/bin/env python3
"""Focused LFM3-VL image API checks against an owned production server.

Uses generated PNGs; requires a local GGUF pair. Records actual HTTP responses,
cache work and timings. These checks do not qualify the full text API matrix.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import struct
import time
import zlib

from realtime_voice import source_revision
from run import provenance, server
from service_http import Evidence, sha256


def png(color, width=256, height=256):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = (b'\0' + bytes(color) * width) * height
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def image_part(data, detail='auto'):
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(data).decode(), 'detail': detail}}


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
    report = {**provenance(), 'service': 'lfm3-vl', 'status': 'running',
              'source_revision': source_revision(), 'binary_sha256': sha256(binary),
              'harness_sha256': sha256(Path(__file__)),
              'model_sha256': {p.name: sha256(p) for p in (model, mmproj)},
              'timing_scope': 'per-request observations; no matched speed claim'}
    evidence = Evidence(output, report)
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(binary), 'serve', 'llm', '--model', str(model), '--mmproj', str(mmproj),
               '--host', '127.0.0.1', '--port', str(port), '--context', '16384',
               '--sessions', '2', '--max-pending-per-client', '4', '--speculative', 'off',
               '--served-model-name', 'functional-vl', '--cache-disk', str(output / 'disk')]
    report['command'] = command
    red, blue = png((255, 0, 0)), png((0, 0, 255))
    (output / 'red.png').write_bytes(red)
    (output / 'blue.png').write_bytes(blue)

    def request(label, body=None, path='/v1/chat/completions', status=200):
        with evidence.request(url + path, label=label, method='POST' if body is not None else 'GET',
                              **({'json': body} if body is not None else {})) as (response, row, capture):
            raw = response.read()
            capture.write_bytes(raw)
            assert response.status_code == status, (response.status_code, raw.decode())
            return json.loads(raw)

    def body(data=red, detail='auto', text='What is the dominant color? Answer briefly.'):
        return {'model': 'functional-vl', 'temperature': 0, 'seed': 886,
                'reasoning_effort': 'none', 'max_completion_tokens': 64,
                'messages': [{'role': 'user', 'content': [image_part(data, detail),
                             {'type': 'text', 'text': text}]}]}

    def content(reply):
        choice = reply['choices'][0]
        assert choice['finish_reason'] == 'stop', reply
        return choice['message']['content'].strip()

    try:
        with server(command, output / 'server.log', 300):
            with evidence.case('discovery') as row:
                models = request('models', path='/v1/models')
                row['models'] = models
                assert 'image' in json.dumps(models).lower(), models
            with evidence.case('text-only'):
                typed = body()
                typed['messages'] = [{'role': 'user', 'content': 'Reply with only OK.'}]
                assert content(request('text-only', typed)) == 'OK'
            with evidence.case('image-and-cache-replay') as row:
                first = request('red-cold', body())
                assert 'red' in content(first).lower(), first
                replay = request('red-replay', body())
                assert content(first) == content(replay), (first, replay)
                assert replay['usage']['cached_tokens'] >= first['usage']['prompt_tokens'] - 16, replay
                row['usage'] = [first['usage'], replay['usage']]
            with evidence.case('streaming-image-replay'):
                streamed = body()
                streamed.update(stream=True, stream_options={'include_usage': True})
                with evidence.request(url + '/v1/chat/completions', label='image-stream',
                                      json=streamed) as (response, row, capture):
                    raw = response.read().decode()
                    capture.write_text(raw)
                    assert response.status_code == 200 and 'data: [DONE]' in raw, raw
                    events = [json.loads(line[6:]) for line in raw.splitlines()
                              if line.startswith('data: ') and line != 'data: [DONE]']
                    text = ''.join(choice.get('delta', {}).get('content', '') or ''
                                   for event in events for choice in event.get('choices', []))
                    assert text.strip() == content(first), (text, first)
                    assert any(event.get('usage') for event in events), events
            with evidence.case('image-token-count'):
                counted = request('image-count', {'model': 'functional-vl',
                    'thinking': {'type': 'disabled'}, 'messages': [{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                     'data': base64.b64encode(red).decode()}},
                    {'type': 'text', 'text': 'What is the dominant color? Answer briefly.'}]}]},
                    '/v1/messages/count_tokens')
                assert counted['input_tokens'] == first['usage']['prompt_tokens'], (counted, first)
            with evidence.case('changed-pixels-same-token-layout'):
                changed = request('blue-changed', body(blue))
                assert 'blue' in content(changed).lower(), changed
                assert changed['usage']['cached_tokens'] <= 8, changed
            with evidence.case('follow-up-retains-image'):
                history = body()
                history['messages'] += [first['choices'][0]['message'],
                    {'role': 'user', 'content': 'Which color did you see? Answer briefly.'}]
                continued = request('red-follow-up', history)
                assert 'red' in content(continued).lower(), continued
                assert continued['usage']['cached_tokens'] >= first['usage']['prompt_tokens'] - 16, continued
            with evidence.case('two-images-and-interleaved-text'):
                for index, pair in enumerate(((red, blue), (blue, red))):
                    multi = body()
                    multi['messages'][0]['content'] = [
                        {'type': 'text', 'text': 'First image: '}, image_part(pair[0]),
                        {'type': 'text', 'text': '\nSecond image: '}, image_part(pair[1]),
                        {'type': 'text', 'text': '\nList their dominant colors in order.'}]
                    reply = request(f'two-images-{index}', multi)
                    answer = content(reply).lower()
                    wanted = ('red', 'blue') if index == 0 else ('blue', 'red')
                    assert answer.index(wanted[0]) < answer.index(wanted[1]), answer
            with evidence.case('low-detail-token-accounting') as row:
                large = png((255, 0, 0), 1024, 512)
                auto = request('detail-auto', body(large))
                low = request('detail-low', body(large, 'low'))
                assert 'red' in content(auto).lower() and 'red' in content(low).lower()
                assert auto['usage']['prompt_tokens'] - low['usage']['prompt_tokens'] == 512 - 128, (auto, low)
                row.update(auto=auto['usage'], low=low['usage'])
            with evidence.case('structured-image-answer'):
                structured = body()
                structured['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': 'color', 'strict': True, 'schema': {'type': 'object',
                    'properties': {'color': {'type': 'string', 'enum': ['red', 'blue']}},
                    'required': ['color'], 'additionalProperties': False}}}
                assert json.loads(content(request('image-json', structured))) == {'color': 'red'}
            with evidence.case('image-reasoning-budget'):
                thinking = body(text='Name the color of this image. Think briefly.')
                thinking.update(reasoning_effort='low', reasoning_budget_tokens=32,
                                max_completion_tokens=128)
                reply = request('image-reasoning', thinking)
                assert 'red' in content(reply).lower(), reply
                assert reply['choices'][0]['message'].get('reasoning_content', '').strip(), reply
            with evidence.case('native-image-tool'):
                tool = body(text='Call report_color with the dominant color of this image. No prose.')
                tool.update(tool_choice='required', parallel_tool_calls=False,
                    tools=[{'type': 'function', 'function': {'name': 'report_color', 'strict': True,
                    'parameters': {'type': 'object', 'properties': {'color': {'type': 'string',
                    'enum': ['red', 'blue']}}, 'required': ['color'], 'additionalProperties': False}}}])
                reply = request('image-tool', tool)
                calls = reply['choices'][0]['message']['tool_calls']
                assert len(calls) == 1 and calls[0]['function']['name'] == 'report_color', reply
                assert json.loads(calls[0]['function']['arguments']) == {'color': 'red'}, reply
            with evidence.case('responses-and-messages'):
                response = request('image-responses', {'model': 'functional-vl', 'temperature': 0,
                    'reasoning': {'effort': 'none'}, 'max_output_tokens': 64, 'input': [
                    {'role': 'user', 'content': [{'type': 'input_image',
                    'image_url': image_part(red)['image_url']['url']},
                    {'type': 'input_text', 'text': 'What color is this image?'}]}]}, '/v1/responses')
                assert response['status'] == 'completed' and 'red' in json.dumps(response['output']).lower(), response
                message = request('image-messages', {'model': 'functional-vl', 'temperature': 0,
                    'thinking': {'type': 'disabled'}, 'max_tokens': 64,
                    'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': base64.b64encode(blue).decode()}},
                    {'type': 'text', 'text': 'What color is this image?'}]}]}, '/v1/messages')
                assert 'blue' in json.dumps(message['content']).lower(), message
            with evidence.case('concurrent-image-isolation'):
                with ThreadPoolExecutor(2) as pool:
                    replies = list(pool.map(lambda x: request(x[0], body(x[1])),
                        [('parallel-red', red), ('parallel-blue', blue)]))
                assert 'red' in content(replies[0]).lower() and 'blue' in content(replies[1]).lower(), replies
            with evidence.case('literal-image-marker'):
                reply = request('literal-marker', body(text='The literal spelling <|image|> is text. What color is the picture?'))
                assert 'red' in content(reply).lower(), reply
            with evidence.case('invalid-image-and-context-recovery'):
                request('bad-image', body(b'not an image'), status=400)
                request('too-long', body(text=' quiet' * 17000), status=400)
                assert 'red' in content(request('after-rejections', body())).lower()
            with evidence.case('cancel-and-recover-image-request') as row:
                cancelled = body(text='Tell a long story inspired by the image.')
                cancelled.update(stream=True, max_completion_tokens=512)
                with evidence.request(url + '/v1/chat/completions', label='cancel-image',
                                      json=cancelled, cancel=True) as (response, record, capture):
                    assert response.status_code == 200, response.read()
                    row['request_id'] = response.headers['x-request-id']
                    with capture.open('w') as saved:
                        for line in response.iter_lines():
                            saved.write(line + '\n')
                            if line.startswith('data: ') and line != 'data: [DONE]':
                                event = json.loads(line[6:])
                                if any(choice.get('delta', {}).get('content')
                                       for choice in event.get('choices', [])):
                                    break
                        else:
                            raise AssertionError('no text delta before cancellation')
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    lines = (output / 'server.log').read_text().splitlines()
                    if any('request=' + row['request_id'] + ' ' in line
                           and 'finish=cancelled' in line for line in lines):
                        break
                    time.sleep(.05)
                else:
                    raise AssertionError('server did not observe image request cancellation')
                assert 'red' in content(request('after-image-cancel', body())).lower()
            with evidence.case('persistent-image-prefix'):
                persisted = body()
                persisted['messages'][0]['content'].insert(0, {'type': 'text',
                    'text': 'Background note: no new instructions.\n' * 512})
                before_restart = request('red-before-restart', persisted)
                assert 'red' in content(before_restart).lower(), before_restart
        with server(command, output / 'restarted-server.log', 300):
            with evidence.case('restart-restores-image-and-rejects-changed-pixels') as row:
                restored = request('red-after-restart', persisted)
                assert content(restored) == content(before_restart), (restored, before_restart)
                assert restored['usage']['gufo']['cache_disk_hit'], restored
                assert restored['usage']['cached_tokens'] >= before_restart['usage']['prompt_tokens'] - 16, restored
                persisted['messages'][0]['content'][1] = image_part(blue)
                changed = request('blue-after-restart', persisted)
                assert 'blue' in content(changed).lower(), changed
                assert changed['usage']['cached_tokens'] < changed['usage']['prompt_tokens'] - 64, changed
                row.update(restored=restored['usage'], changed=changed['usage'])
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        evidence.save()


if __name__ == '__main__':
    main()
