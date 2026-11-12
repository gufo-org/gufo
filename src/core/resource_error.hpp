#ifndef GUFO_CORE_RESOURCE_ERROR_HPP_
#define GUFO_CORE_RESOURCE_ERROR_HPP_

#include <stdexcept>

namespace gufo::core {

/// A request exceeded a managed execution-memory budget. This is a normal
/// admission refusal, not evidence that the device or model became unhealthy.
class ResourceAdmissionError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

}  // namespace gufo::core

#endif
