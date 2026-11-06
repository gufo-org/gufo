# Extension API change records

Record every public extension API declaration or contract change here. This
includes public headers and their transitive `SURFACE` closure, the extension
metadata schema, and public CMake contracts. The PR requires the `ext-api` label.

## Create a record

1. Read `include/gufo/ext/api-series.json` for the active major/minor series.
2. Add one uniquely named Markdown file under that series directory. Choose a
   descriptive filename, such as `embedding-video-token-budget.md` or
   `serving-metadata.md`. Use a feature or issue identifier if needed for
   uniqueness; do not use an API version or revision number.
3. Use a descriptive heading and describe the concrete public contracts changed.
   Do not assign an API version in the filename, heading or description.

```markdown
# Capability-neutral embedding contracts

Publish capability-neutral embedding contracts (`include/gufo/ext/embeddings.hpp`):
- `EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`,
  `EmbeddingVideoOptions`, `EmbeddingRequestOptions`.
```

Describe the contracts and any compatibility or rebuilding implications. Keep
implementation history, test logs and unrelated changes out of the record.
Existing merged records remain permanent and immutable. The one-time conversion
of the original records to descriptive filenames/headings preserves their
contract descriptions and does not advance the API revision.

## Versioning and checks

CMake derives the API revision from `revision_base` plus the number of records
in the active series. `baseline.md` describes the initial contracts and does
not advance the count. Do not edit version constants for additive changes.
The README and its `AGENTS.md` and `CLAUDE.md` symlinks are guidance, not revision
records.

Concurrent PRs add independent, uniquely named files. Their merge counts both
records without a shared counter, assigned version heading or filename conflict.
A breaking API change explicitly advances the compatibility series in
`api-series.json` and adds a record in that series; retain older series.

After header/include changes, run:

```sh
python3 tools/ci/check-extension-api.py --write
```

Run `python3 tools/ci/check-extension-api.py` to verify the surface and record
inventory. See [the release policy](../../../../docs/RELEASING.md) for the
source compatibility guarantee and PR requirements.
