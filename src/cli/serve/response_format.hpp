#ifndef GUFO_SERVER_RESPONSE_FORMAT_HPP_
#define GUFO_SERVER_RESPONSE_FORMAT_HPP_

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>

#include "src/core/json_constraint.hpp"

namespace gufo::server {

// Chat Completions wire contract:
// https://developers.openai.com/api/docs/guides/structured-outputs
inline std::shared_ptr<const sampling::JsonConstraint> ParseResponseFormat(
    const json::Value* format, bool responses = false) {
  if (!format || format->is_null())
    return {};
  if (responses && format->is_object() &&
      format->member_str("type") == "json_schema") {
    auto chat = json::Value::object();
    chat["type"] = "json_schema";
    auto specification = json::Value::object();
    for (const auto& [key, value] : format->members())
      if (key != "type")
        specification[key] = value;
    chat["json_schema"] = std::move(specification);
    return ParseResponseFormat(&chat);
  }
  const auto invalid = [](const char* message) {
    throw std::invalid_argument(message);
  };
  if (!format->is_object())
    invalid("'response_format' must be an object");
  const auto type = format->member_str("type");
  if (type == "text" || type == "json_object") {
    if (format->size() != 1)
      invalid("unexpected response_format member");
    return type == "text" ? nullptr : sampling::JsonConstraint::Object();
  }
  if (type != "json_schema" || format->size() != 2)
    invalid("response_format type must be text, json_object or json_schema");
  const auto* specification = format->find("json_schema");
  if (!specification || !specification->is_object())
    invalid("'json_schema' must be an object");
  for (const auto& [key, value] : specification->members()) {
    (void)value;
    if (key != "name" && key != "description" && key != "schema" &&
        key != "strict")
      invalid("unexpected json_schema member");
  }
  const auto name = specification->member_str("name");
  if (name.empty() || name.size() > 64 ||
      !std::ranges::all_of(name, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-';
      }))
    invalid("json_schema.name must contain 1–64 letters, digits, '_' or '-'");
  if (const auto* description = specification->find("description");
      description &&
      (!description->is_string() || description->str().size() > 8192))
    invalid("json_schema.description must be a string of at most 8192 bytes");
  const auto* strict = specification->find("strict");
  if (strict && !strict->is_null() && !strict->is_bool())
    invalid("json_schema.strict must be a boolean or null");
  const auto* schema = specification->find("schema");
  if (!schema)
    invalid("json_schema.schema is required");
  return sampling::JsonConstraint::Compile(*schema,
                                           strict && strict->as_bool());
}
}  // namespace gufo::server

#endif
