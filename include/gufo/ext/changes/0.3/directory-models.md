# Text models from model directories

`--model` accepts a directory in `serve llm`, `prompt`, `chat` and `bench`. The
text provider whose `probe(model_root)` accepts it creates the runner, with
`TextRunnerOptions::model_path` set to the directory and no GGUF reader
(`PrepareRunnerOptions` receives a null `reader`). A provider with a runner but
no drivers of its own gets `prompt`, `chat` and `bench` through the server's
backend. GGUF paths are unchanged.
