#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

// The OpenAI image API: `POST /v1/images/generations` and
// `POST /v1/images/edits`. The A1111 API is in `sdapi.hpp`; both share only
// the `ImageService`.

#include "src/cli/serve/image_service.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);

}  // namespace gufo::server
#endif
