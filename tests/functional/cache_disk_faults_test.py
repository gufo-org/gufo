"""Check the process-local publication gate without loading a model."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

LIBRARY = Path(sys.argv.pop(1)).resolve()
SCRIPT = """import os,sys
with open(sys.argv[1], 'wb') as output:
    output.write(b'checkpoint')
    output.flush()
    os.fsync(output.fileno())
"""


class DiskGateTest(unittest.TestCase):
    def run_gate(self, armed, inside, release=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, gate = root / "cache", root / "gate"
            cache.mkdir()
            gate.mkdir()
            if armed:
                (gate / "armed").touch()
            target = (cache if inside else root) / "checkpoint"
            environment = {**os.environ, "LD_PRELOAD": str(LIBRARY),
                           "GUFO_TEST_DISK_GATE": str(gate),
                           "GUFO_TEST_DISK_DIRECTORY": str(cache)}
            process = subprocess.Popen([sys.executable, "-c", SCRIPT, str(target)],
                                       env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            try:
                if armed and inside:
                    deadline = time.monotonic() + 5
                    while not (gate / "blocked").exists() and time.monotonic() < deadline:
                        self.assertIsNone(process.poll())
                        time.sleep(.01)
                    self.assertTrue((gate / "blocked").exists())
                    self.assertIsNone(process.poll())
                    if release:
                        (gate / "release").touch()
                    else:
                        process.kill()
                out, error = process.communicate(timeout=5)
                self.assertEqual(process.returncode, -9 if armed and inside and not release else 0,
                                 error.decode())
                self.assertEqual(target.read_bytes(), b"checkpoint")
                if not (armed and inside):
                    self.assertFalse((gate / "blocked").exists())
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_disarmed_gate_and_files_outside_private_cache_are_unaffected(self):
        self.run_gate(False, True)
        self.run_gate(True, False)

    def test_private_write_is_held_until_its_process_is_killed(self):
        self.run_gate(True, True)

    def test_releasing_the_gate_completes_the_write(self):
        self.run_gate(True, True, release=True)


if __name__ == "__main__":
    unittest.main()
