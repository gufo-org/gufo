#ifndef GUFO_CORE_JSON_SCHEMA_CALL_HPP_
#define GUFO_CORE_JSON_SCHEMA_CALL_HPP_

#include <gufo/json_schema.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_schema_lexeme.hpp"
#include "src/core/rust/callback.hpp"

namespace gufo::sampling::detail {

// One call into the Rust JSON Schema compiler (include/gufo/json_schema.h):
// the context of its callbacks, and the exceptions its status stands for.
struct JsonSchemaCall {
  core::ffi::CallbackFailure failure;
  std::string reason;
  std::vector<std::string> outputs;

  static void Reason(void* context, const std::uint8_t* data,
                     std::size_t len) noexcept {
    auto& call = *static_cast<JsonSchemaCall*>(context);
    call.failure.Guard(
        [&] { call.reason.assign(reinterpret_cast<const char*>(data), len); });
  }
  static void Output(void* context, const std::uint8_t* data,
                     std::size_t len) noexcept {
    auto& call = *static_cast<JsonSchemaCall*>(context);
    call.failure.Guard([&] {
      call.outputs.emplace_back(reinterpret_cast<const char*>(data), len);
    });
  }
  void Check(GufoJsonSchemaStatus status) const {
    failure.Rethrow();
    switch (status) {
      case GUFO_JSON_SCHEMA_COMPLETE:
        return;
      case GUFO_JSON_SCHEMA_REJECTED:
        throw std::invalid_argument("JSON Schema: " + reason);
      case GUFO_JSON_SCHEMA_EMPTY:
        throw JsonSchemaEmpty("JSON Schema: " + reason);
      case GUFO_JSON_SCHEMA_NO_MEMORY:
        throw std::bad_alloc();
      case GUFO_JSON_SCHEMA_STOPPED:
      case GUFO_JSON_SCHEMA_INTERNAL:
        break;
    }
    throw std::logic_error("JSON Schema compiler failed");
  }
};

// Schema keyword values as the compiler reads them. The values point into
// this object, so it stays where it is built.
template<std::size_t N>
class JsonSchemaKeywords {
public:
  JsonSchemaKeywords(const json::Value& schema,
                     const std::array<const char*, N>& keys) {
    for (std::size_t i = 0; i < N; ++i)
      Set(i, schema.find(keys[i]));
  }
  explicit JsonSchemaKeywords(const std::array<const json::Value*, N>& values) {
    for (std::size_t i = 0; i < N; ++i)
      Set(i, values[i]);
  }
  JsonSchemaKeywords(const JsonSchemaKeywords&) = delete;
  JsonSchemaKeywords& operator=(const JsonSchemaKeywords&) = delete;
  const GufoJsonSchemaValue* data() const { return values_.data(); }

private:
  void Set(std::size_t i, const json::Value* value) {
    auto& out = values_[i];
    out = {GUFO_JSON_SCHEMA_ABSENT, 0.0, nullptr, 0};
    if (!value)
      return;
    if (value->is_number()) {
      out.kind = GUFO_JSON_SCHEMA_NUMBER;
      out.number = value->as_double();
      if (std::isfinite(out.number))
        texts_[i] = value->dump();
    } else if (value->is_string()) {
      out.kind = GUFO_JSON_SCHEMA_STRING;
      texts_[i] = value->str();
    } else {
      out.kind = GUFO_JSON_SCHEMA_OTHER;
    }
    out.text = reinterpret_cast<const std::uint8_t*>(texts_[i].data());
    out.len = texts_[i].size();
  }
  std::array<std::string, N> texts_;
  std::array<GufoJsonSchemaValue, N> values_{};
};

}  // namespace gufo::sampling::detail
#endif
