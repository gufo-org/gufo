#ifndef GUFO_SERVER_SDAPI_HPP_
#define GUFO_SERVER_SDAPI_HPP_

// The AUTOMATIC1111 Stable Diffusion WebUI API (`/sdapi/v1/*`) in the dialect
// of stable-diffusion.cpp's server. It has its own request schema, responses
// and `{"error": ...}` errors; it shares only the `ImageService` with the
// OpenAI image API and never translates fields between the two.

#include <string_view>

#include "src/cli/serve/image_service.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

bool IsSdapiPath(std::string_view path);

/// Every `/sdapi/v1/*` route of an image server.
HttpResponse HandleSdapiRequest(const HttpRequest& request,
                                ImageService& service);

/// `GET /sdapi/v1/loras`: registered adapters in registration order.
HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service);

}  // namespace gufo::server
#endif
