// Teacher-forced full-vocabulary logits from the pinned llama.cpp reference.
//
// Reads little-endian int32 token ids, evaluates them in order on the GPU
// (f16 KV, flash attention) and writes the logits of the selected positions:
//   magic "G4LG", u32 version=1, u32 rows, u32 vocab, u32 positions[rows],
//   f32 logits[rows][vocab]
// Build: tools/gemma4/build_llama_logits.sh
#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

/// Records named per-layer tensors ("l_out-<layer>") for every row, and
/// optionally every F32 intermediate of one layer into a directory.
struct Trace {
  std::string prefix;
  std::vector<std::vector<float>> layers;  // [layer][rows * width]
  std::uint32_t width{0};
  std::string dump_dir;
  std::string dump_suffix;  // "-<layer>"
};

bool DumpTensor(struct ggml_tensor* t, bool ask, Trace* trace,
                const std::string& name) {
  const bool wanted =
      t->type == GGML_TYPE_F32 &&
      ((name.size() > trace->dump_suffix.size() &&
        name.compare(name.size() - trace->dump_suffix.size(),
                     trace->dump_suffix.size(), trace->dump_suffix) == 0) ||
       name.rfind("inp_", 0) == 0);
  if (ask || !wanted) {
    return wanted;
  }
  std::vector<float> data(ggml_nelements(t));
  ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
  std::string file = name;
  for (char& ch : file) {
    if (ch == ' ' || ch == '/' || ch == '(' || ch == ')') ch = '_';
  }
  std::ofstream out(trace->dump_dir + "/" + file + ".f32", std::ios::binary);
  const std::uint32_t dims[4] = {static_cast<std::uint32_t>(t->ne[0]),
                                 static_cast<std::uint32_t>(t->ne[1]),
                                 static_cast<std::uint32_t>(t->ne[2]),
                                 static_cast<std::uint32_t>(t->ne[3])};
  out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
  return true;
}

bool TraceCallback(struct ggml_tensor* t, bool ask, void* user) {
  auto* trace = static_cast<Trace*>(user);
  const std::string name = ggml_get_name(t);
  if (!trace->dump_dir.empty() && name.rfind(trace->prefix, 0) != 0) {
    return DumpTensor(t, ask, trace, name);
  }
  if (name.rfind(trace->prefix, 0) != 0 || t->type != GGML_TYPE_F32) {
    return ask ? false : true;
  }
  if (ask) {
    return true;
  }
  const auto layer = static_cast<std::size_t>(
      std::stoul(name.substr(trace->prefix.size())));
  if (trace->layers.size() <= layer) trace->layers.resize(layer + 1);
  trace->width = static_cast<std::uint32_t>(t->ne[0]);
  std::vector<float> data(ggml_nelements(t));
  ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
  auto& dst = trace->layers[layer];
  dst.insert(dst.end(), data.begin(), data.end());
  return true;
}

[[noreturn]] void Die(const std::string& message) {
  std::fprintf(stderr, "llama_logits: %s\n", message.c_str());
  std::exit(1);
}

void Usage() {
  std::fprintf(stderr,
               "usage: llama_logits --model GGUF --tokens IN.i32 --output OUT "
               "[--first N] [--stride N] [--batch N] [--trace-out F]\n"
               "       [--dump-dir DIR [--dump-layer L]]\n"
               "  rows at positions first, first+stride, ... (default: all)\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, output_path, trace_path, dump_dir;
  int dump_layer = 0;
  bool flash_attention = true;
  std::uint32_t first = 0, stride = 1, batch = 512;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) Usage();
      return argv[++i];
    };
    if (arg == "--model") model_path = value();
    else if (arg == "--tokens") tokens_path = value();
    else if (arg == "--output") output_path = value();
    else if (arg == "--first") first = std::stoul(value());
    else if (arg == "--stride") stride = std::stoul(value());
    else if (arg == "--batch") batch = std::stoul(value());
    else if (arg == "--trace-out") trace_path = value();
    else if (arg == "--dump-dir") dump_dir = value();
    else if (arg == "--dump-layer") dump_layer = std::stoi(value());
    else if (arg == "--no-flash-attn") flash_attention = false;
    else Usage();
  }
  if (model_path.empty() || tokens_path.empty() || output_path.empty() ||
      stride == 0 || batch == 0) {
    Usage();
  }

  std::ifstream in(tokens_path, std::ios::binary);
  std::vector<llama_token> tokens;
  for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), sizeof(t));) {
    tokens.push_back(t);
  }
  if (tokens.empty()) Die("no tokens in " + tokens_path);
  const auto n = static_cast<std::uint32_t>(tokens.size());

  ggml_backend_load_all_from_path(GUFO_LLAMA_BACKEND_DIR);
  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (model == nullptr) Die("cannot load " + model_path);
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const auto n_vocab = static_cast<std::uint32_t>(llama_vocab_n_tokens(vocab));

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = (n + 255) / 256 * 256;
  cparams.n_batch = batch;
  cparams.n_ubatch = batch;
  cparams.n_seq_max = 1;
  cparams.flash_attn_type = flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  Trace trace{"l_out-"};
  trace.dump_dir = dump_dir;
  trace.dump_suffix = "-" + std::to_string(dump_layer);
  if (!trace_path.empty() || !dump_dir.empty()) {
    cparams.cb_eval = TraceCallback;
    cparams.cb_eval_user_data = &trace;
  }
  llama_context* ctx = llama_init_from_model(model, cparams);
  if (ctx == nullptr) Die("cannot create context");

  std::vector<std::uint32_t> positions;
  for (std::uint32_t p = first; p < n; p += stride) positions.push_back(p);
  std::ofstream out(output_path, std::ios::binary);
  const std::uint32_t header[4] = {0x474C3447U, 1U,
                                   static_cast<std::uint32_t>(positions.size()),
                                   n_vocab};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(positions.data()),
            static_cast<std::streamsize>(positions.size() * 4));

  llama_batch b = llama_batch_init(static_cast<std::int32_t>(batch), 0, 1);
  std::size_t next = 0;
  for (std::uint32_t start = 0; start < n; start += batch) {
    const std::uint32_t count = std::min(batch, n - start);
    b.n_tokens = static_cast<std::int32_t>(count);
    for (std::uint32_t j = 0; j < count; ++j) {
      const std::uint32_t pos = start + j;
      b.token[j] = tokens[pos];
      b.pos[j] = static_cast<llama_pos>(pos);
      b.n_seq_id[j] = 1;
      b.seq_id[j][0] = 0;
      b.logits[j] = pos >= first && (pos - first) % stride == 0;
    }
    if (llama_decode(ctx, b) != 0) Die("decode failed at " + std::to_string(start));
    for (std::uint32_t j = 0; j < count; ++j) {
      if (!b.logits[j]) continue;
      const float* row = llama_get_logits_ith(ctx, static_cast<std::int32_t>(j));
      if (row == nullptr) Die("missing logits");
      out.write(reinterpret_cast<const char*>(row),
                static_cast<std::streamsize>(n_vocab) * 4);
      ++next;
    }
  }
  if (next != positions.size()) Die("row count mismatch");
  if (!trace_path.empty()) {
    // [layer][row][width] float32 after a u32 header: layers, rows, width.
    std::ofstream t(trace_path, std::ios::binary);
    const std::uint32_t th[3] = {static_cast<std::uint32_t>(trace.layers.size()),
                                 n, trace.width};
    t.write(reinterpret_cast<const char*>(th), sizeof(th));
    for (const auto& layer : trace.layers) {
      if (layer.size() != static_cast<std::size_t>(n) * trace.width)
        Die("trace rows incomplete; use --batch >= token count");
      t.write(reinterpret_cast<const char*>(layer.data()),
              static_cast<std::streamsize>(layer.size() * 4));
    }
  }
  llama_batch_free(b);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  std::fprintf(stderr, "llama_logits: wrote %zu rows x %u\n", next, n_vocab);
  return 0;
}
