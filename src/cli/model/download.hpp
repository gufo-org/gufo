#pragma once
// Parallel, resumable downloads of pinned Hugging Face files into the model
// store's blobs. Each file is resolved once to its CDN location, then fetched
// as byte ranges over several connections; a file is hashed as soon as its
// last range lands, while the others keep downloading. Files already in the
// store or in the Hugging Face cache are not downloaded again.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gufo::cli::model_download {

struct Item {
  std::string repo;
  std::string revision;
  std::string path;
  std::uint64_t size = 0;
  std::string sha256;
};

struct Options {
  std::size_t connections = 16;
  std::uint64_t chunk_bytes = 64ULL << 20U;
};

// Brings every item into <store>/blobs/<sha256>, keeping partial downloads in
// <store>/downloads for the next call. Throws when an item cannot be fetched
// or does not match its SHA-256.
void Fetch(const std::vector<Item>& items, const std::filesystem::path& store,
           const Options& options);

// HF_ENDPOINT, else https://huggingface.co.
std::string Endpoint();
// HF_TOKEN, else the token `hf auth login` saved; empty when there is none.
std::string Token();
// HF_HUB_CACHE, else $HF_HOME/hub, else <user cache>/huggingface/hub.
std::filesystem::path HubCache();

}  // namespace gufo::cli::model_download
