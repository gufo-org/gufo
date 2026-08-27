#ifndef GUFO_EVAL_HTTP_CLIENT_HPP_
#define GUFO_EVAL_HTTP_CLIENT_HPP_

#include <string>
#include <string_view>

namespace gufo::eval {

struct HttpResult {
  bool transport_ok{false};
  long status_code{0};
  std::string body;
  std::string transport_code;
  double elapsed_ms{0.0};
};

class HttpClient {
public:
  HttpClient(std::string base_url, std::string bearer_token);

  [[nodiscard]] const std::string& base_url() const noexcept {
    return base_url_;
  }

  [[nodiscard]] HttpResult Get(std::string_view relative_path) const;
  [[nodiscard]] HttpResult PostJson(std::string_view relative_path,
                                    std::string_view body) const;

private:
  [[nodiscard]] HttpResult Request(std::string_view method,
                                   std::string_view relative_path,
                                   std::string_view body) const;

  std::string base_url_;
  std::string bearer_token_;
};

[[nodiscard]] bool ValidateBaseUrl(std::string_view base_url,
                                   std::string* error = nullptr);

[[nodiscard]] std::string NormalizeBaseUrl(std::string base_url);

}  // namespace gufo::eval

#endif  // GUFO_EVAL_HTTP_CLIENT_HPP_
