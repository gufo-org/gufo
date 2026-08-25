#ifndef STRIX_SERVER_VIDEO_API_HPP_
#define STRIX_SERVER_VIDEO_API_HPP_

#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/video_jobs.hpp"

namespace strix::server {

inline constexpr std::string_view kVideoApiSchema = "strix.video-api.v1";

[[nodiscard]] bool IsVideoApiPath(std::string_view path) noexcept;
[[nodiscard]] HttpResponse HandleVideoApiRequest(const HttpRequest& request,
                                                 VideoJobService& service);

}  // namespace strix::server

#endif  // STRIX_SERVER_VIDEO_API_HPP_
