import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import time
import urllib.request

root = Path('/home/mixer/gufo')
binary = root / 'build/gpu-test/gufo'
model = root / 'models/lfm3/LFM3-1.7B-Q8_0.gguf'
with tempfile.TemporaryDirectory(prefix='gufo-card25-') as temporary:
    package = Path(temporary)
    (package / 'bin').mkdir()
    (package / 'bin/gufo').symlink_to(binary)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    args = {'model': str(model), 'provider': 'lfm3', 'context': 1024,
            'host': '127.0.0.1', 'port': port, 'servedModelName': 'card25-lfm3'}
    (package / 'args.json').write_text(json.dumps(args))
    expression = '''let
      lib = import /nix/store/6nwzgfvznah5rg192x0hnl5ykchdgi59-source/lib;
      metadata = import /home/mixer/gufo/.devops/nix/extension-metadata.nix {
        inherit lib; extensions = [ /home/mixer/gufo/lfm3-gufo-extension ];
      };
      serve = import /home/mixer/gufo/.devops/nix/mk-serve.nix {
        inherit lib; gufo = { outPath = "PACKAGE"; extensionMetadata = metadata; };
      };
    in serve (builtins.fromJSON (builtins.readFile "PACKAGE/args.json"))
    '''.replace('PACKAGE', str(package))
    command = subprocess.check_output(['nix', 'eval', '--impure', '--raw', '--expr', expression], text=True)
    unit = 'gufo-card25-smoke-' + str(port)
    report = {'command': command, 'binary_sha256': hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
              'external_revision': subprocess.check_output(['git', '-C', str(root / 'lfm3-gufo-extension'), 'rev-parse', 'HEAD'], text=True).strip()}
    try:
        subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--property=Type=exec', *shlex.split(command)], check=True)
        base = f'http://127.0.0.1:{port}'
        for attempt in range(120):
            try:
                with urllib.request.urlopen(base + '/v1/models', timeout=1) as response:
                    models = json.load(response)
                break
            except Exception:
                if subprocess.check_output(['systemctl', '--user', 'is-active', unit], text=True).strip() == 'failed':
                    raise RuntimeError('service failed')
                time.sleep(0.5)
        else:
            raise RuntimeError('readiness timeout')
        assert models['data'][0]['id'] == 'card25-lfm3', models
        report['models'] = models
        body = {'model': 'card25-lfm3', 'messages': [{'role': 'user', 'content': 'What is the capital of France?'}],
                'max_tokens': 128, 'temperature': 0, 'enable_thinking': False}
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            report['chat_status'] = response.status
            report['chat'] = json.load(response)
        assert report['chat']['choices'][0]['message']['content'], report['chat']
        report['service_active'] = subprocess.check_output(['systemctl', '--user', 'is-active', unit], text=True).strip()
        report['verdict'] = 'PASS'
    finally:
        report['journal'] = subprocess.check_output(['journalctl', '--user', '--unit=' + unit, '--no-pager', '-n', '40'], text=True)
        subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
        (root / 'docs/model-extensions/evidence/card25/systemd-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(report['verdict'])
