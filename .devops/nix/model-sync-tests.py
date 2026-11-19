"""Exercise sync decisions with an isolated CLI fixture; no model downloads."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("model_sync", Path(__file__).with_name("model-sync.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SyncTest(unittest.TestCase):
    def test_install_update_scoping_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "gufo"
            binary.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
root = Path(os.environ["SYNC_FIXTURE"])
if sys.argv[1:] == ["model", "list", "--json"]:
    print((root / "listing.json").read_text())
else:
    with (root / "calls").open("a") as out:
        out.write(json.dumps(sys.argv[1:]) + "\\n")
    sys.exit(1 if sys.argv[-1] == "broken" else 0)
''')
            binary.chmod(0o755)
            rows = [{"id": name, "extension": "fixture", "extensionStatus": "installed",
                     "downloadable": True, **({"path": "/fixture/model"} if installed else {})}
                    for name, installed in [("new", False), ("existing", True),
                                            ("unmanaged", True), ("broken", True)]]
            listing = root / "listing.json"
            listing.write_text(json.dumps({"models": rows}))
            with patch.dict(os.environ, {"SYNC_FIXTURE": str(root)}):
                self.assertEqual(module.sync(str(binary), ["new", "existing"]), 0)
                calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
                self.assertEqual(calls, [["model", "install", "new"], ["model", "update", "existing"]])
                self.assertEqual(module.sync(str(binary), ["broken", "existing"]), 1)
                (root / "calls").unlink()
                rows[0]["extensionStatus"] = "not installed"
                listing.write_text(json.dumps({"models": rows}))
                with self.assertRaisesRegex(RuntimeError, "requires extension fixture"):
                    module.sync(str(binary), ["existing", "new"])
                self.assertFalse((root / "calls").exists())
                rows[0]["extensionStatus"] = "installed"
                rows[0]["downloadable"] = False
                listing.write_text(json.dumps({"models": rows}))
                with self.assertRaisesRegex(RuntimeError, "no pinned download metadata"):
                    module.sync(str(binary), ["new"])
                with self.assertRaisesRegex(RuntimeError, "not in gufo model list"):
                    module.sync(str(binary), ["unknown"])


if __name__ == "__main__":
    unittest.main()
