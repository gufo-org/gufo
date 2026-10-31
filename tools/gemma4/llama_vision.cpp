// Gemma 4 image oracle from the pinned llama.cpp reference (mtmd, gemma4v).
//
// Images are raw RGB8 files already resized by Gufo (sides multiples of 48);
// the token budget is pinned to their size so mtmd keeps the pixels as-is.
//
// Encode mode writes each image's projector output:
//   magic "G4VE", u32 version=1, u32 rows, u32 width, f32 embd[rows][width]
// With --tokens it also teacher-forces a Gufo prompt (little-endian int32
// ids): each run of `<|image|>` soft tokens takes the next image's
// embeddings through mtmd's non-causal image decode, and every other token
// is decoded as text. Logits of all text positions are written as in
// llama_logits: "G4LG", u32 1, u32 rows, u32 vocab, u32 positions[rows],
// f32 logits[rows][vocab].
// Build: tools/gemma4/build_llama_logits.sh
#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>
#include <mtmd-helper.h>
#include <mtmd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Dump {
  std::string directory;
};

bool DumpCallback(struct ggml_tensor* t, bool ask, void* user) {
  if (t->type != GGML_TYPE_F32) {
    return false;
  }
  if (ask) {
    return true;
  }
  const auto* dump = static_cast<Dump*>(user);
  std::string file = ggml_get_name(t);
  for (char& ch : file) {
    if (ch == ' ' || ch == '/' || ch == '(' || ch == ')') ch = '_';
  }
  std::vector<float> data(ggml_nelements(t));
  ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
  std::ofstream out(dump->directory + "/" + file + ".f32", std::ios::binary);
  const std::uint32_t dims[4] = {static_cast<std::uint32_t>(t->ne[0]),
                                 static_cast<std::uint32_t>(t->ne[1]),
                                 static_cast<std::uint32_t>(t->ne[2]),
                                 static_cast<std::uint32_t>(t->ne[3])};
  out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * 4));
  return true;
}

[[noreturn]] void Die(const std::string& message) {
  std::fprintf(stderr, "llama_vision: %s\n", message.c_str());
  std::exit(1);
}

void Usage() {
  std::fprintf(stderr,
               "usage: llama_vision --model GGUF --mmproj GGUF "
               "--image RGB WIDTH HEIGHT [--image ...]\n"
               "       [--embd-out PREFIX] [--dump-dir DIR]\n"
               "       [--tokens IN.i32 --logits-out OUT]\n"
               "  PREFIX<i>.f32 receives image i's embeddings\n");
  std::exit(2);
}

struct Image {
  std::string path;
  std::uint32_t width{0};
  std::uint32_t height{0};
};

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, mmproj_path, embd_prefix, dump_dir, tokens_path,
      logits_path;
  std::vector<Image> images;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) Usage();
      return argv[++i];
    };
    if (arg == "--model") model_path = value();
    else if (arg == "--mmproj") mmproj_path = value();
    else if (arg == "--image") {
      Image image;
      image.path = value();
      image.width = std::stoul(value());
      image.height = std::stoul(value());
      images.push_back(image);
    } else if (arg == "--embd-out") embd_prefix = value();
    else if (arg == "--dump-dir") dump_dir = value();
    else if (arg == "--tokens") tokens_path = value();
    else if (arg == "--logits-out") logits_path = value();
    else Usage();
  }
  if (model_path.empty() || mmproj_path.empty() || images.empty() ||
      tokens_path.empty() != logits_path.empty()) {
    Usage();
  }

  ggml_backend_load_all_from_path(GUFO_LLAMA_BACKEND_DIR);
  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  // Encoding alone reads only the text model's hyperparameters; keep its
  // weights mapped rather than offloaded.
  mparams.n_gpu_layers = tokens_path.empty() ? 0 : 999;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (model == nullptr) Die("cannot load " + model_path);

  std::vector<std::vector<unsigned char>> pixels;
  std::uint32_t max_rows = 0;
  for (const auto& image : images) {
    if (image.width % 48 != 0 || image.height % 48 != 0) {
      Die(image.path + ": sides must be multiples of 48");
    }
    std::ifstream in(image.path, std::ios::binary);
    std::vector<unsigned char> data(std::size_t{image.width} * image.height *
                                    3);
    if (!in.read(reinterpret_cast<char*>(data.data()),
                 static_cast<std::streamsize>(data.size()))) {
      Die("cannot read " + image.path);
    }
    pixels.push_back(std::move(data));
    max_rows = std::max(max_rows, (image.width / 48) * (image.height / 48));
  }

  // One mtmd context per distinct budget keeps each image's size unchanged.
  Dump dump{dump_dir};
  const auto open_mtmd = [&](std::uint32_t rows) {
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.print_timings = false;
    params.warmup = false;
    params.image_min_tokens = static_cast<int>(rows);
    params.image_max_tokens = static_cast<int>(rows);
    if (!dump_dir.empty()) {
      params.cb_eval = DumpCallback;
      params.cb_eval_user_data = &dump;
    }
    mtmd_context* ctx = mtmd_init_from_file(mmproj_path.c_str(), model, params);
    if (ctx == nullptr) Die("cannot load " + mmproj_path);
    return ctx;
  };

  // Tokenizing a lone media marker yields "<|image>", the image chunk and
  // "<image|>"; only the image chunk is kept.
  struct Encoded {
    mtmd_context* mtmd;
    mtmd_input_chunks* chunks;
    const mtmd_input_chunk* chunk;
    std::vector<float> embd;
  };
  std::vector<Encoded> encoded;
  for (std::size_t i = 0; i < images.size(); ++i) {
    const std::uint32_t rows =
        (images[i].width / 48) * (images[i].height / 48);
    mtmd_context* mtmd = open_mtmd(rows);
    mtmd_bitmap* bitmap =
        mtmd_bitmap_init(images[i].width, images[i].height, pixels[i].data());
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const std::string marker = mtmd_get_marker(mtmd);
    const mtmd_input_text text{marker.c_str(), marker.size(), false, true};
    const mtmd_bitmap* bitmaps[] = {bitmap};
    if (mtmd_tokenize(mtmd, chunks, &text, bitmaps, 1) != 0) {
      Die("mtmd_tokenize failed for " + images[i].path);
    }
    mtmd_bitmap_free(bitmap);
    const mtmd_input_chunk* chunk = nullptr;
    for (std::size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
      const auto* candidate = mtmd_input_chunks_get(chunks, c);
      if (mtmd_input_chunk_get_type(candidate) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        chunk = candidate;
      }
    }
    if (chunk == nullptr || mtmd_input_chunk_get_n_tokens(chunk) != rows) {
      Die(images[i].path + ": mtmd changed the image size");
    }
    if (mtmd_encode_chunk(mtmd, chunk) != 0) Die("encode failed");
    const std::size_t width = llama_model_n_embd_inp(model);
    const float* out = mtmd_get_output_embd(mtmd);
    encoded.push_back({mtmd, chunks, chunk,
                       std::vector<float>(out, out + rows * width)});
    if (!embd_prefix.empty()) {
      std::ofstream file(embd_prefix + std::to_string(i) + ".f32",
                         std::ios::binary);
      const std::uint32_t header[4] = {0x45563447U, 1U, rows,
                                       static_cast<std::uint32_t>(width)};
      file.write(reinterpret_cast<const char*>(header), sizeof(header));
      file.write(reinterpret_cast<const char*>(encoded.back().embd.data()),
                 static_cast<std::streamsize>(encoded.back().embd.size() * 4));
    }
    std::fprintf(stderr, "llama_vision: image %zu: %u rows x %zu\n", i, rows,
                 width);
  }

  if (!tokens_path.empty()) {
    std::ifstream in(tokens_path, std::ios::binary);
    std::vector<llama_token> tokens;
    for (std::int32_t t; in.read(reinterpret_cast<char*>(&t), sizeof(t));) {
      tokens.push_back(t);
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const llama_token soft = [&] {
      llama_token id = -1;
      const std::string text = "<|image|>";
      if (llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()),
                         &id, 1, false, true) != 1) {
        Die("vocabulary lacks <|image|>");
      }
      return id;
    }();
    const auto n = static_cast<std::uint32_t>(tokens.size());
    const auto n_vocab = static_cast<std::uint32_t>(llama_vocab_n_tokens(vocab));
    const std::uint32_t batch = std::max<std::uint32_t>(2048, max_rows);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (n + 255) / 256 * 256;
    cparams.n_batch = batch;
    cparams.n_ubatch = batch;
    cparams.n_seq_max = 1;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) Die("cannot create context");

    std::vector<std::uint32_t> positions;
    std::vector<float> logits;
    llama_batch b = llama_batch_init(static_cast<std::int32_t>(batch), 0, 1);
    std::size_t next_image = 0;
    std::uint32_t pos = 0;
    while (pos < n) {
      if (tokens[pos] == soft) {
        if (next_image >= encoded.size()) Die("more image spans than images");
        auto& image = encoded[next_image++];
        const auto rows =
            static_cast<std::uint32_t>(mtmd_input_chunk_get_n_tokens(image.chunk));
        for (std::uint32_t j = 0; j < rows; ++j) {
          if (pos + j >= n || tokens[pos + j] != soft) {
            Die("image span length differs from its image");
          }
        }
        llama_pos new_pos = 0;
        if (mtmd_helper_decode_image_chunk(
                image.mtmd, ctx, image.chunk, image.embd.data(),
                static_cast<llama_pos>(pos), 0, static_cast<std::int32_t>(batch),
                &new_pos, nullptr, nullptr) != 0) {
          Die("image decode failed");
        }
        pos += rows;
        continue;
      }
      std::uint32_t count = 0;
      while (pos + count < n && tokens[pos + count] != soft && count < batch) {
        ++count;
      }
      b.n_tokens = static_cast<std::int32_t>(count);
      for (std::uint32_t j = 0; j < count; ++j) {
        b.token[j] = tokens[pos + j];
        b.pos[j] = static_cast<llama_pos>(pos + j);
        b.n_seq_id[j] = 1;
        b.seq_id[j][0] = 0;
        b.logits[j] = true;
      }
      if (llama_decode(ctx, b) != 0) Die("decode failed at " + std::to_string(pos));
      for (std::uint32_t j = 0; j < count; ++j) {
        const float* row = llama_get_logits_ith(ctx, static_cast<std::int32_t>(j));
        if (row == nullptr) Die("missing logits");
        positions.push_back(pos + j);
        logits.insert(logits.end(), row, row + n_vocab);
      }
      pos += count;
    }
    if (next_image != encoded.size()) Die("fewer image spans than images");
    std::ofstream out(logits_path, std::ios::binary);
    const std::uint32_t header[4] = {0x474C3447U, 1U,
                                     static_cast<std::uint32_t>(positions.size()),
                                     n_vocab};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(positions.data()),
              static_cast<std::streamsize>(positions.size() * 4));
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * 4));
    std::fprintf(stderr, "llama_vision: wrote %zu rows x %u\n", positions.size(),
                 n_vocab);
    llama_batch_free(b);
    llama_free(ctx);
  }

  for (auto& image : encoded) {
    mtmd_input_chunks_free(image.chunks);
    mtmd_free(image.mtmd);
  }
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
