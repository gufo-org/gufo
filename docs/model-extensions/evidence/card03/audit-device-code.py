#!/usr/bin/env python3
"""Compare device ISA/constants/resource notes, excluding host/debug identifiers."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
SOURCE = Path(__file__).with_name('audit-builds.py').read_text()
namespace = {}
exec(SOURCE[:SOURCE.index('rows = []')], namespace)
section = namespace['section']
ROOT = Path('/home/mixer/gufo')
SPIKE = Path('/tmp/gufo-card03-spike')
BIN = Path('/nix/store/92bpbb2rmrkszhw83x63x7qzj9jsz324-rocm-toolchain/bin')
rows = []
with tempfile.TemporaryDirectory(prefix='card03-device-audit-') as temporary:
    temporary = Path(temporary)
    for path in sorted((ROOT / 'build/release').rglob('*.o')):
        relative = path.relative_to(ROOT / 'build/release')
        other = SPIKE / 'build/release' / relative
        if not other.is_file():
            continue
        before = section(path, '.hip_fatbin')
        if before is None:
            continue
        images = []
        for name, data in [('baseline', before), ('candidate', section(other, '.hip_fatbin'))]:
            source = temporary / (name + '.bundle')
            destination = temporary / (name + '.hsaco')
            source.write_bytes(data)
            subprocess.run([str(BIN / 'clang-offload-bundler'), '--unbundle', '--type=o',
                '--input=' + str(source), '--targets=hip-amdgcn-amd-amdhsa--gfx1151',
                '--output=' + str(destination)], check=True, capture_output=True)
            images.append(destination)
        result = {'object': str(relative), 'sections': {}}
        for name in ('.text', '.rodata', '.note'):
            left, right = [section(path, name) for path in images]
            result['sections'][name] = {'identical': left == right,
                'baseline_sha256': hashlib.sha256(left or b'').hexdigest(),
                'candidate_sha256': hashlib.sha256(right or b'').hexdigest(),
                'baseline_bytes': len(left or b''), 'candidate_bytes': len(right or b'')}
        rows.append(result)
print(json.dumps({'objects': rows,
    'scope': 'gfx1151 device .text ISA, .rodata constants and .note resource metadata; excludes debug sections and symbol/string tables whose path identifiers differ between source roots'}, indent=2))
