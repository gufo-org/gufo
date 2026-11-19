#include "src/core/json.hpp"

#include <gufo/json.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/rust/callback.hpp"

namespace gufo::json {
namespace detail {

// Builds the Value tree from the parser's events, each value in place: a key
// adds its member at once, and open arrays and objects stay on a stack until
// their end event. A container's address holds while it is open, since only
// the innermost one grows. The Rust parser reports a well-formed sequence;
// assertions check it in test builds.
struct Parser {
  std::vector<Value*> open;
  Value root;
  std::string reason;
  core::ffi::CallbackFailure failure;

  static std::string_view Text(const GufoJsonEvent& event) {
    return {reinterpret_cast<const char*>(event.text), event.len};
  }

  // Where the next value goes: the root, the newest member of the innermost
  // object, or a new item of the innermost array.
  Value& Next() {
    if (open.empty())
      return root;
    Value& parent = *open.back();
    if (parent.type_ == Value::Type::kObject) {
      assert(!parent.obj_.empty());
      return parent.obj_.back().second;
    }
    return parent.arr_.emplace_back();
  }

  void Take(const GufoJsonEvent& event) {
    switch (event.kind) {
      case GUFO_JSON_NULL:
        Next();
        return;
      case GUFO_JSON_FALSE:
      case GUFO_JSON_TRUE: {
        Value& value = Next();
        value.type_ = Value::Type::kBool;
        value.bool_ = event.kind == GUFO_JSON_TRUE;
        return;
      }
      case GUFO_JSON_NUMBER: {
        Value& value = Next();
        value.type_ = Value::Type::kNumber;
        value.num_ = event.number;
        value.str_ = Text(event);
        return;
      }
      case GUFO_JSON_STRING: {
        Value& value = Next();
        value.type_ = Value::Type::kString;
        value.str_ = Text(event);
        return;
      }
      case GUFO_JSON_KEY:
        assert(!open.empty() && open.back()->type_ == Value::Type::kObject);
        open.back()->obj_.emplace_back(std::string(Text(event)), Value());
        return;
      case GUFO_JSON_BEGIN_ARRAY:
      case GUFO_JSON_BEGIN_OBJECT: {
        Value& value = Next();
        value.type_ = event.kind == GUFO_JSON_BEGIN_ARRAY
                          ? Value::Type::kArray
                          : Value::Type::kObject;
        open.push_back(&value);
        return;
      }
      case GUFO_JSON_END:
        assert(!open.empty());
        open.pop_back();
        return;
    }
    throw std::logic_error("unknown JSON parser event");
  }
};

}  // namespace detail

Value parse(std::string_view text) {
  detail::Parser parser;
  const auto status = gufo_json_parse(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size(),
      [](void* context, const GufoJsonEvent* event) noexcept -> int {
        auto& parser = *static_cast<detail::Parser*>(context);
        return parser.failure.Guard(1, [&] {
          parser.Take(*event);
          return 0;
        });
      },
      [](void* context, const std::uint8_t* data, std::size_t len) noexcept {
        auto& parser = *static_cast<detail::Parser*>(context);
        parser.failure.Guard([&] {
          parser.reason.assign(reinterpret_cast<const char*>(data), len);
        });
      },
      &parser);
  parser.failure.Rethrow();
  switch (status) {
    case GUFO_JSON_COMPLETE:
      assert(parser.open.empty());
      return std::move(parser.root);
    case GUFO_JSON_REJECTED:
      throw std::runtime_error("JSON parse error: " + parser.reason);
    case GUFO_JSON_NO_MEMORY:
      throw std::bad_alloc();
    case GUFO_JSON_STOPPED:
    case GUFO_JSON_INTERNAL:
      break;
  }
  throw std::logic_error("JSON parser failed");
}

}  // namespace gufo::json
