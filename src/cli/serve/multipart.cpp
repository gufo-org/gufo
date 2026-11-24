#include "src/cli/serve/multipart.hpp"

#include <gufo/http.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "src/core/rust/callback.hpp"

namespace gufo::server {
namespace {

std::string_view View(GufoHttpSlice slice) {
  return {reinterpret_cast<const char*>(slice.data), slice.len};
}

struct Context {
  const std::function<bool(const MultipartField&)>* on_field;
  std::string detail;
  core::ffi::CallbackFailure failure;
};

}  // namespace

MultipartResult ParseMultipart(
    std::string_view content_type, std::string_view body,
    MultipartLimits limits,
    const std::function<bool(const MultipartField&)>& on_field) {
  Context context{&on_field, {}, {}};
  // No exception may cross the Rust frames; it is rethrown below.
  const auto field = [](void* opaque,
                        const GufoMultipartField* parsed) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    return context.failure.Guard(1, [&] {
      return (*context.on_field)({.name = View(parsed->name),
                                  .has_filename = parsed->has_filename != 0,
                                  .content = View(parsed->content)})
                 ? 0
                 : 1;
    });
  };
  const auto error = [](void* opaque, const std::uint8_t* data,
                        std::size_t len) noexcept {
    auto& context = *static_cast<Context*>(opaque);
    context.failure.Guard([&] {
      context.detail.assign(reinterpret_cast<const char*>(data), len);
    });
  };
  const GufoMultipartParse parsed = gufo_multipart_parse(
      reinterpret_cast<const std::uint8_t*>(content_type.data()),
      content_type.size(), reinterpret_cast<const std::uint8_t*>(body.data()),
      body.size(), limits.max_fields, limits.unique_names ? 1 : 0, field, error,
      &context);
  context.failure.Rethrow();
  const auto result = [&context](MultipartStatus status) {
    return MultipartResult{.status = status,
                           .detail = std::move(context.detail)};
  };
  switch (parsed) {
    case GUFO_MULTIPART_COMPLETE:
      return result(MultipartStatus::kComplete);
    case GUFO_MULTIPART_NOT_FORM_DATA:
      return result(MultipartStatus::kNotFormData);
    case GUFO_MULTIPART_BAD_BOUNDARY:
      return result(MultipartStatus::kBadBoundary);
    case GUFO_MULTIPART_MALFORMED:
      return result(MultipartStatus::kMalformed);
    case GUFO_MULTIPART_TOO_MANY_FIELDS:
      return result(MultipartStatus::kTooManyFields);
    case GUFO_MULTIPART_DUPLICATE_FIELD:
      return result(MultipartStatus::kDuplicateField);
    case GUFO_MULTIPART_STOPPED:
      return result(MultipartStatus::kStopped);
    case GUFO_MULTIPART_INTERNAL:
      break;
  }
  throw std::runtime_error("Rust multipart parser failed internally");
}

}  // namespace gufo::server
