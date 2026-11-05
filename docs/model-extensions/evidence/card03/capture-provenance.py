#!/usr/bin/env python3
"""Audit actual binaries/caches, overriding model-bench host autodetection."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mixer/gufo')
SPIKE = Path('/tmp/gufo-card03-spike')
roots = {'baseline': ROOT, 'candidate': SPIKE}
revisions = {'baseline': '79930b75383fcda678daee98d796e6cc6dfdb63f',
             'candidate': '50ba836fc4ab2fe7003371f3f290bdd48ea68b0a'}
arms = {}
for arm, root in roots.items():
    cache = (root / 'build/release/CMakeCache.txt').read_text().splitlines()
    fields = {line.split(':')[0]: line.split('=', 1)[1] for line in cache
              if line.startswith(('BUILD_TESTING:', 'CMAKE_BUILD_TYPE:',
                                  'CMAKE_CXX_COMPILER:', 'CMAKE_HIP_COMPILER:',
                                  'CMAKE_CXX_FLAGS_RELWITHDEBINFO:',
                                  'CMAKE_HIP_FLAGS_RELWITHDEBINFO:'))}
    arms[arm] = {'source_revision': revisions[arm],
                 'binary': str(root / 'build/release/gufo'),
                 'binary_sha256': hashlib.sha256((root / 'build/release/gufo').read_bytes()).hexdigest(),
                 'cache': fields}
models = []
for path in sorted(Path('/persist/models/qwen38-flash-next').glob('*.gguf')):
    stat = path.stat()
    stamp = f'{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}'
    cache_key = hashlib.sha256(stamp.encode()).hexdigest()
    cache_path = Path.home() / '.cache/gufo/gguf-sha256-v1' / cache_key
    if cache_path.exists():
        digest = cache_path.read_text().strip()
        method = 'stat_validated_production_identity_cache'
    else:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        method = 'full_file_read_after_timing_jobs'
    assert len(digest) == 64
    assert path.stat() == stat
    models.append({'path': str(path), 'size': stat.st_size, 'stamp': stamp,
                   'sha256': digest, 'hash_method': method})
print(json.dumps({'arms': arms, 'model_files': models,
                  'model_hash_scope': 'Existing production gguf-sha256-v1 cached full-file digests indexed by current device/inode/size/mtime/ctime stamp, or full file read when cache absent (projector); this audit does not reread the 111 GB main model.',
                  'raw_model_bench_provenance_limit': 'Raw artifacts infer GCC15.3 and source79930b75 from editing host/root even when candidate binary is supplied; actual per-arm audited GCC16.2/HIPClang24 production metadata above takes precedence. Raw artifacts retained unchanged.',
                  'toolchain': 'Pinned flake GCC16.2.0 / HIP Clang24 / ROCm10.1; same dependency paths and normalized common compile commands.'}, indent=2))
