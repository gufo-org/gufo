#ifndef GUFO_CORE_JSON_SCHEMA_LEXEME_HPP_
#define GUFO_CORE_JSON_SCHEMA_LEXEME_HPP_

#include <bitset>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::sampling {

class JsonSchemaEmpty : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

// Incremental constraints on a single primitive. A complete value can also be
// a prefix (e.g. 1 / 1.5); the grammar retains both continuations.
class JsonSchemaLexeme {
public:
  struct Match {
    bool prefix{false};
    bool complete{false};
  };
  virtual ~JsonSchemaLexeme() = default;
  // Reject impossible bytes before copying request state. This alphabet is a
  // superset of every valid transition, not a replacement for the predicate.
  bool AllowsByte(unsigned char byte) const { return alphabet_.test(byte); }
  virtual Match Check(std::string_view bytes) const = 0;
  virtual Match Advance(std::string& state, unsigned char byte) const {
    state += static_cast<char>(byte);
    return Check(state);
  }
  virtual bool CacheTransitions() const { return false; }
  virtual void CanonicalMaskState(std::string&, std::size_t) const {}
  virtual bool AcceptValue(const json::Value& value) const {
    return Check(value.dump()).complete;
  }
  static std::shared_ptr<const JsonSchemaLexeme> String(
      const json::Value& schema);
  static std::shared_ptr<const JsonSchemaLexeme> RawString(
      const json::Value& schema, std::string delimiter);
  static std::shared_ptr<const JsonSchemaLexeme> Number(
      const json::Value& schema, bool integer);
  static std::shared_ptr<const JsonSchemaLexeme> Whitespace();
  static json::Value IntersectMultipleOf(const json::Value& a,
                                         const json::Value& b);
  static json::Value Format(std::string_view format);

protected:
  JsonSchemaLexeme() : alphabet_(~std::bitset<256>{}) {}
  explicit JsonSchemaLexeme(std::string_view alphabet) {
    for (unsigned char byte : alphabet)
      alphabet_.set(byte);
  }

private:
  std::bitset<256> alphabet_;
};

}  // namespace gufo::sampling
#endif
