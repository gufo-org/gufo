#ifndef GUFO_EXT_MODEL_SOURCE_HPP_
#define GUFO_EXT_MODEL_SOURCE_HPP_

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

/// Reads from the Hugging Face Hub for a model source's resolver, with the
/// endpoint and token `gufo model` uses. Each call throws std::runtime_error
/// when the request fails.
struct ModelSourceHub {
  /// The body of `<endpoint>/api/<path>`, such as `models/<repo>/refs`.
  std::function<std::string(std::string_view path)> api;
  /// The text of one file of `repo` at a 40-hex `revision`.
  std::function<std::string(std::string_view repo, std::string_view revision,
                            std::string_view path)>
      file;
};

/// One component of a resolved model: the files of one Hugging Face
/// repository at a pinned revision, installed at their repository paths in
/// the model directory.
struct ModelSourceComponent {
  std::string role;
  /// The command option the component's path fills; `--model` for the one
  /// the model is served from, empty when the runner finds the component in
  /// the model directory on its own.
  std::string option;
  std::string repo;
  /// 40-hex commit.
  std::string revision;
  /// The path passed to `option`, relative to the model directory; "." is
  /// the model directory itself.
  std::string entry = ".";
  /// Exact repository paths to install; empty installs every file except
  /// those matching `exclude`.
  std::vector<std::string> include;
  /// Patterns (`*` matches within one path part) a file's path or name must
  /// not match when `include` is empty.
  std::vector<std::string> exclude;
};

struct ModelSourceModel {
  /// The model ID: lowercase letters, digits, '.', '_' and '-'.
  std::string id;
  std::string license;
  std::string description;
  std::vector<ModelSourceComponent> components;
};

/// Models resolved when they are installed rather than declared in the
/// extension's metadata: `gufo model install <owner>/<name>` asks the source
/// registered for `<owner>`, whose every repository is approved.
struct ModelSource {
  /// The Hugging Face owner whose repositories this source resolves.
  std::string owner;
  /// The provider that serves the resolved models.
  std::string provider;
  /// Resolves `repo` (`<owner>/<name>`) to a model with pinned revisions, or
  /// throws std::runtime_error saying why it cannot be installed.
  std::function<ModelSourceModel(std::string_view repo,
                                 const ModelSourceHub& hub)>
      resolve;
};

/// The source registered for a Hugging Face owner, or nullptr.
const ModelSource* FindModelSource(std::string_view owner);
const std::vector<ModelSource>& ModelSources();

}  // namespace gufo::ext

#endif  // GUFO_EXT_MODEL_SOURCE_HPP_
