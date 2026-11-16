# Text models from model directories

`serve llm --model` accepts a directory. The text provider whose
`probe(model_root)` accepts it creates the runner, with
`TextRunnerOptions::model_path` set to the directory and no GGUF reader
(`PrepareRunnerOptions` receives a null `reader`). GGUF paths are unchanged.
