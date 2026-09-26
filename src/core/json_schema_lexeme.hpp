#ifndef GUFO_CORE_JSON_SCHEMA_LEXEME_HPP_
#define GUFO_CORE_JSON_SCHEMA_LEXEME_HPP_

#include <memory>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::sampling {

// Incremental constraints on a single primitive. A complete value can also be
// a prefix (e.g. 1 / 1.5); the grammar retains both continuations.
class JsonSchemaLexeme {
public:
  struct Match {
    bool prefix{false};
    bool complete{false};
  };
  virtual ~JsonSchemaLexeme() = default;
  virtual Match Check(std::string_view bytes) const = 0;
  virtual bool AcceptValue(const json::Value& value) const {
    return Check(value.dump()).complete;
  }
  static std::shared_ptr<const JsonSchemaLexeme> String(
      const json::Value& schema);
  static std::shared_ptr<const JsonSchemaLexeme> Number(
      const json::Value& schema, bool integer);
};

}  // namespace gufo::sampling
#endif
