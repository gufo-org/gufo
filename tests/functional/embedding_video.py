#!/usr/bin/env python3
"""Video HTTP reference parity; --smoke is a bounded multimodal service check."""
import argparse
import base64
import concurrent.futures
import io
import http.client
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import socket
import re
import urllib.parse

from embedding_images import cosine, picture

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'embedding_gemma2'


def source(path):
    return base64.b64encode(path.read_bytes()).decode()


def mixed_audio():
    import numpy as np
    samples = np.sin(np.arange(3200, dtype=np.float32)*0.07)*0.1
    # IEEE float WAV retains the reference float32 waveform exactly.
    raw = samples.astype('<f4').tobytes()
    return base64.b64encode(b'RIFF' + struct.pack('<I', 36+len(raw)) + b'WAVEfmt ' +
        struct.pack('<IHHIIHH', 16, 3, 1, 16000, 64000, 4, 32) + b'data' +
        struct.pack('<I', len(raw)) + raw).decode()


def content(case):
    from PIL import Image
    text = case['text']
    parts, video = [], 0
    while text:
        next_marker = min(((text.find(marker), marker) for marker in
            ('<|video|>', '<|image|>', '<|audio|>') if marker in text), default=None)
        if next_marker is None:
            parts.append({'type': 'text', 'text': text})
            break
        at, marker = next_marker
        if at:
            parts.append({'type': 'text', 'text': text[:at]})
        text = text[at+len(marker):]
        if marker == '<|video|>':
            parts.append({'type': 'input_video', 'input_video': {'data': source(FIXTURES / case['videos'][video]['path'])}})
            video += 1
        elif marker == '<|audio|>':
            parts.append({'type': 'input_audio', 'input_audio': {'data': mixed_audio()}})
        else:
            output = io.BytesIO()
            Image.new('RGB', (32, 32), (255, 0, 0)).save(output, 'PNG')
            parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(output.getvalue()).decode()}})
    return {'content': parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model', default='embeddinggemma-2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--server-log', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must not exist')
    fixture = json.loads((FIXTURES / 'video_golden.json').read_text())
    cases = {c['name']: c for c in fixture['cases']}
    results = []
    report = {'status': 'running', 'smoke': args.smoke, 'requests': results,
              'reference': {k: v for k, v in fixture.items() if k != 'cases'}}

    def post(inputs, **options):
        request = urllib.request.Request(args.base_url.rstrip('/') + '/embeddings',
            data=json.dumps({'model': args.model, 'input': inputs, **options}).encode(),
            headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body, status = json.load(response), response.status
        except urllib.error.HTTPError as e:
            body, status = json.load(e), e.code
        results.append({'status': status, 'elapsed_ms': (time.perf_counter()-start)*1000,
                        'options': options, 'input': inputs, 'response': body})
        return status, body

    def check(body, expected):
        assert [item['index'] for item in body['data']] == list(range(len(expected))), body
        assert body['usage']['prompt_tokens'] == sum(len(c['token_ids']) for c in expected), body['usage']
        for item, case in zip(body['data'], expected):
            vector = item['embedding']
            assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
            assert abs(sum(x*x for x in vector)-1) < 1e-5
            agreement = cosine(vector, case['embedding'])
            assert agreement >= 0.9999, (case['name'], agreement)
            item['reference_cosine'] = agreement

    try:
        # One small real batch covers all four towers and interleaving. Text and
        # image references remain independent of the newly introduced video path.
        text = next(c for c in json.loads((FIXTURES/'embedding_golden.json').read_text())['cases'] if c['name'] == 'plain')
        image = json.loads((FIXTURES/'image_golden.json').read_text())['cases'][0]
        probes = [text, image, cases['short'], cases['mixed']]
        inputs = [text['text'], {'content': [{'type': 'image_url', 'image_url': {'url': picture(image['images'][0])}}]},
                  content(cases['short']), content(cases['mixed'])]
        status, body = post(inputs)
        assert status == 200, body
        check(body, probes)
        # Image budgets must not change the distinct 140-token video default.
        # Explicit default-image options also preserve the joint multimodal
        # reference, including both placeholder kinds in one input.
        for budget in (70, 1120):
            status, body = post([content(cases['short'])],
                                mm_processor_kwargs={'max_soft_tokens': budget})
            assert status == 200, body
            check(body, [cases['short']])
        status, body = post([content(cases['mixed'])],
                            mm_processor_kwargs={'max_soft_tokens': 280})
        assert status == 200, body
        check(body, [cases['mixed']])
        if not args.smoke:
            for case in fixture['cases']:
                status, body = post([content(case)], video_options=case['options'])
                assert status == 200, body
                check(body, [case])
            for dimensions in (128, 256, 512):
                status, body = post([content(cases['short'])], dimensions=dimensions)
                assert status == 200, body
                vec = body['data'][0]['embedding']
                assert len(vec) == dimensions and abs(sum(v*v for v in vec)-1) < 1e-5
                assert cosine(vec, cases['short']['embedding'][:dimensions]) >= 0.9999
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                replies = list(pool.map(lambda _: post([content(cases['short'])]), range(2)))
            for status, body in replies:
                assert status == 200, body
                check(body, [cases['short']])
            invalid = ['file:///etc/passwd', 'https://127.0.0.1/x', 'data:video/mp4;base64,AAAA']
            for url in invalid:
                status, body = post([{'content': [{'type': 'input_video', 'input_video': {'url': url}}]}])
                assert status in (400, 413), body
            # The HTTP layer refuses oversized Content-Length before reading a
            # body. Sending the rejected body can race the server's close.
            endpoint = urllib.parse.urlsplit(args.base_url)
            client = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=30)
            try:
                client.putrequest('POST', endpoint.path.rstrip('/') + '/embeddings')
                client.putheader('Content-Type', 'application/json')
                client.putheader('Content-Length', str(30*1024*1024))
                client.endheaders()
                response = client.getresponse()
                rejected = response.read().decode()
                assert response.status == 413, rejected
                report['oversized_http'] = {'content_length': 30*1024*1024, 'status': response.status, 'response': rejected}
            finally:
                client.close()
            for name in ('oversized', 'overduration', 'overframes'):
                status, body = post([{'content': [{'type': 'input_video', 'input_video': {
                    'data': source(FIXTURES/'video'/f'{name}.mp4')}}]}])
                assert status == 400 and body['error']['code'] == 'invalid_video_input', (name, body)
            limited = {'content': [content(cases['overflow'])['content'][0]] * 3}
            status, body = post([limited])
            assert status == 400 and body['error']['code'] == 'invalid_video_input', body
            too_long = content(cases['overflow'])
            too_long['content'].insert(0, {'type': 'text', 'text': 'hello '*8500})
            for truncate in (False, True):
                status, body = post([too_long], truncate=truncate)
                assert status == 400 and body['error']['code'] == 'context_length_exceeded', body
        if not args.smoke:
            assert args.server_log and args.server_log.is_file(), 'full suite needs --server-log to verify cancellation'
            before = len(args.server_log.read_text().splitlines())
            endpoint = urllib.parse.urlsplit(args.base_url)
            payload = json.dumps({'model': args.model, 'input': [content(cases['overflow'])]}).encode()
            with socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=30) as client:
                client.sendall((f"POST {endpoint.path.rstrip('/')}/embeddings HTTP/1.1\r\n"
                    f"Host: {endpoint.netloc}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()+payload)
                time.sleep(0.02)
                client.shutdown(socket.SHUT_RDWR)
            deadline, cancellation = time.monotonic()+15, None
            while time.monotonic() < deadline:
                lines = args.server_log.read_text().splitlines()[before:]
                received = [re.search(r'request=(\S+)', line).group(1) for line in lines
                            if 'event=received ' in line and 'path=/v1/embeddings ' in line]
                if received:
                    assert len(received) == 1, received
                    cancellation = next((line for line in lines if f'request={received[0]} ' in line
                        and 'event=completed ' in line and 'error_code=cancelled' in line), None)
                    if cancellation:
                        break
                time.sleep(0.02)
            assert cancellation, 'server must observe cancellation for this request'
            report['cancelled_request'] = {'request': json.loads(payload), 'completion': cancellation}
        status, body = post([{'content': [{'type': 'input_video', 'input_video': {'data': 'AAAA'}}]}])
        assert status == 400 and body['error']['code'] == 'invalid_video_input', body
        status, body = post([text['text']])
        assert status == 200, body
        check(body, [text])
        report['status'] = 'passed'
    finally:
        if report['status'] != 'passed':
            report['status'] = 'failed'
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"PASS embedding {'smoke' if args.smoke else 'video'}: {len(results)} requests")


if __name__ == '__main__':
    main()
