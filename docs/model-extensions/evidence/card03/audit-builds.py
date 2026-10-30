#!/usr/bin/env python3
"""Compare production flags and unchanged embedded HIP images after timing jobs."""
import hashlib
import json
import struct
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
SPIKE = Path('/tmp/gufo-card03-spike')

def section(path, wanted):
    data = path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError(f'not ELF64 little endian: {path}')
    offset = struct.unpack_from('<Q', data, 40)[0]
    stride, count, strings = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * stride)
               for i in range(count)]
    names = data[headers[strings][4]:headers[strings][4] + headers[strings][5]]
    for header in headers:
        name = names[header[0]:].split(b'\0', 1)[0].decode()
        if name == wanted:
            return data[header[4]:header[4] + header[5]]
    return None

rows = []
for path in sorted((ROOT / 'build/release').rglob('*.o')):
    relative = path.relative_to(ROOT / 'build/release')
    other = SPIKE / 'build/release' / relative
    if not other.is_file():
        continue
    before = section(path, '.hip_fatbin')
    if before is None:
        continue
    after = section(other, '.hip_fatbin')
    rows.append({'object': str(relative), 'identical_embedded_image': before == after,
                 'baseline_sha256': hashlib.sha256(before).hexdigest(),
                 'candidate_sha256': hashlib.sha256(after or b'').hexdigest()})

def commands(root):
    records = json.loads((root / 'build/release/compile_commands.json').read_text())
    return {str(Path(r['file']).resolve().relative_to(root)): r['command'] for r in records}
left, right = commands(ROOT), commands(SPIKE)
command_rows = []
for path in sorted(left.keys() & right.keys()):
    command_rows.append({'source': path,
        'identical_normalized_command': left[path].replace(str(ROOT), '<SOURCE>') ==
            right[path].replace(str(SPIKE), '<SOURCE>')})
report = {'embedded_hip_images': rows, 'common_compile_commands': command_rows,
          'note': 'Exact embedded-image equality is stronger than stripped ISA/resource equality; any mismatch remains explicit.'}
print(json.dumps(report, indent=2))
