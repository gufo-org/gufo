# Model README table metadata

Extend `schemas/extension.schema.json` with optional `readme` metadata for
extensions with a `docs` entry. Metadata declares a weights license and task
section rows, with optional model title, guide and per-row license overrides.
Table cells accept literal strings or templates populated from named benchmark
tables, rows and columns. Unknown metadata fields are rejected.

Existing manifests remain valid without `readme`. Consumers validating manifests
with this metadata must use the updated schema. Public C++ declarations are
unchanged; this addition introduces no new rebuilding requirements.
