# Text models from model directories

`serve llm --model` accepts a directory. Gufo asks each text provider's
`probe(model_root)` about it, before opening any GGUF, and creates the claiming
provider's runner through `create` with `TextRunnerOptions::model_path` set to
the directory and `reader` null. A provider whose model is a package of files
(for example a manifest with programs and parameters) uses `probe` to claim it
and reads its own files. Providers that only probe GGUFs are unaffected.
