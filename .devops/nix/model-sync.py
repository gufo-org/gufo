"""Reconcile declared IDs through the Gufo CLI without managing extensions."""
import json
import subprocess
import sys


def sync(binary, models):
    listing = json.loads(subprocess.check_output(
        [binary, "model", "list", "--json"], text=True))
    rows = {row["id"]: row for row in listing["models"]}
    errors = []
    for model in models:
        row = rows.get(model)
        if row is None:
            errors.append(f"{model}: not in gufo model list")
        elif row.get("extensionStatus") not in ("built-in", "installed"):
            errors.append(f"{model}: requires extension {row['extension']}; "
                          f"run gufo extension install {row['extension']} as the sync user")
        elif not row.get("downloadable", False):
            errors.append(f"{model}: extension {row['extension']} has no pinned download metadata")
    if errors:
        raise RuntimeError("\n".join(errors))
    failed = False
    for model in models:
        row = rows[model]
        command = "update" if "path" in row else "install"
        result = subprocess.run([binary, "model", command, model], check=False)
        failed = result.returncode != 0 or failed
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(sync(sys.argv[1], json.loads(sys.argv[2])))
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"gufo model sync: {error}", file=sys.stderr)
        sys.exit(1)
