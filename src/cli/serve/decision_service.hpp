#ifndef GUFO_SERVER_DECISION_SERVICE_HPP_
#define GUFO_SERVER_DECISION_SERVICE_HPP_
#include <functional>
#include <string>

#include "src/core/json.hpp"
namespace gufo::server {
struct DecisionService {
  std::string model_id;
  bool multimodal{};
  std::function<json::Value(const json::Value&)> decide;
};
}  // namespace gufo::server
#endif
