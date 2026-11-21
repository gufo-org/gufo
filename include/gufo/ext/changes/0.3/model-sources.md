# Model sources

Let an extension resolve models when they are installed instead of declaring
each one in its metadata (`include/gufo/ext/model_source.hpp`):
- `ModelSource`: a Hugging Face owner, the provider that serves its models
  and a `resolve` function from `<owner>/<name>` to a `ModelSourceModel`.
- `ModelSourceModel` and `ModelSourceComponent`: the model ID, license,
  description and each component's repository, pinned revision, option,
  entry and included or excluded paths.
- `ModelSourceHub`: Hub API and file reads, with the endpoint and token
  `gufo model` uses.
- `Registry::AddModelSource` and `FindModelSource`.

`gufo model install <owner>/<name>` resolves the model through the owner's
source, pins every file's size and SHA-256 from the Hub (LFS SHA-256, or the
Git object ID of a downloaded file), and records the repository so that
`gufo model update` resolves it again. A model an extension declares for the
same package is installed from its declaration instead.

Existing extensions are unchanged; registering a source is optional.
