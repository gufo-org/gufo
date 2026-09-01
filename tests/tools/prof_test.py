#!/usr/bin/env python3

import importlib.util
import sqlite3
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prof" / "prof.py"
SPEC = importlib.util.spec_from_file_location("gufo_prof", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load profiler module")
prof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prof
SPEC.loader.exec_module(prof)


with tempfile.TemporaryDirectory() as directory:
    database = Path(directory) / "profile.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE rocpd_info_kernel_symbol (
            id INTEGER PRIMARY KEY,
            display_name TEXT
        );
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY,
            kernel_id INTEGER,
            start INTEGER,
            end INTEGER,
            grid_size_x INTEGER,
            grid_size_y INTEGER,
            grid_size_z INTEGER,
            workgroup_size_x INTEGER,
            workgroup_size_y INTEGER,
            workgroup_size_z INTEGER
        );
        INSERT INTO rocpd_info_kernel_symbol VALUES (1, 'valid_kernel');
        INSERT INTO rocpd_info_kernel_symbol VALUES (2, 'invalid_kernel');
        INSERT INTO rocpd_kernel_dispatch
        VALUES (1, 1, 100, 300, 64, 1, 1, 32, 1, 1);
        INSERT INTO rocpd_kernel_dispatch
        VALUES (2, 2, 500, 400, 64, 1, 1, 32, 1, 1);
        """
    )
    connection.commit()
    connection.close()

    result = prof.load(str(database))

assert result.dispatches == 1
assert result.invalid_dispatches == 1
assert result.total_ns == 200.0
assert result.busy_ns == 200.0
assert result.wall_ns == 200.0
assert "valid_kernel" in result.kernels
assert "invalid_kernel" not in result.kernels

print("Profiler tests passed.")
