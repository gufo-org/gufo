#include "src/eval/http_client.hpp"

#include <curl/curl.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace gufo::eval {
namespace {

constexpr std::size_t kMaximumResponseBytes =
    std::size_t{64} * std::size_t{1024} * std::size_t{1024};

struct CurlHandleDeleter {
  void operator()(CURL* handle) const noexcept {
    if (handle != nullptr) {
      curl_easy_cleanup(handle);
    }
  }
};

struct CurlHeadersDeleter {
  void operator()(curl_slist* headers) const noexcept {
    if (headers != nullptr) {
      curl_slist_free_all(headers);
    }
  }
};

using CurlHandle = std::unique_ptr<CURL, CurlHandleDeleter>;
using CurlHeaders = std::unique_ptr<curl_slist, CurlHeadersDeleter>;

void EnsureCurlInitialized() {
  static const int initialized = [] {
    return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK ? 1 : 0;
  }();
  (void)initialized;
}

std::size_t AppendResponse(char* contents, std::size_t size, std::size_t count,
                           void* opaque) {
  auto* output = static_cast<std::string*>(opaque);
  const std::size_t bytes = size * count;
  if (output == nullptr || output->size() + bytes > kMaximumResponseBytes) {
    return 0;
  }
  output->append(contents, bytes);
  return bytes;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

HttpClient::HttpClient(std::string base_url, std::string bearer_token)
    : base_url_(NormalizeBaseUrl(std::move(base_url))),
      bearer_token_(std::move(bearer_token)) {
  EnsureCurlInitialized();
}

HttpResult HttpClient::Get(std::string_view relative_path) const {
  return Request("GET", relative_path, "");
}

HttpResult HttpClient::PostJson(std::string_view relative_path,
                                std::string_view body) const {
  return Request("POST", relative_path, body);
}

HttpResult HttpClient::Request(std::string_view method,
                               std::string_view relative_path,
                               std::string_view body) const {
  HttpResult result;
  const CurlHandle handle(curl_easy_init());
  if (handle == nullptr) {
    result.transport_code = "curl_initialization_failed";
    return result;
  }

  std::string url = base_url_;
  if (relative_path.empty() || relative_path.front() != '/') {
    url.push_back('/');
  }
  url.append(relative_path);

  curl_slist* raw_headers = nullptr;
  raw_headers = curl_slist_append(raw_headers, "Accept: application/json");
  if (method == "POST") {
    raw_headers =
        curl_slist_append(raw_headers, "Content-Type: application/json");
  }
  std::string authorization;
  if (!bearer_token_.empty()) {
    authorization = "Authorization: Bearer " + bearer_token_;
    raw_headers = curl_slist_append(raw_headers, authorization.c_str());
  }
  const CurlHeaders headers(raw_headers);
  const std::string request_body(body);

  curl_easy_setopt(handle.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers.get());
  curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(handle.get(), CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, AppendResponse);
  curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &result.body);
  curl_easy_setopt(handle.get(), CURLOPT_USERAGENT, "gufo-eval/1");
  curl_easy_setopt(handle.get(), CURLOPT_FOLLOWLOCATION, 0L);
  if (method == "POST") {
    curl_easy_setopt(handle.get(), CURLOPT_POST, 1L);
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDS, request_body.c_str());
    curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(body.size()));
  }

  const auto started = std::chrono::steady_clock::now();
  const CURLcode code = curl_easy_perform(handle.get());
  result.elapsed_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
  if (code != CURLE_OK) {
    result.transport_code = "curl_" + std::to_string(static_cast<int>(code));
    return result;
  }
  result.transport_ok = true;
  curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &result.status_code);
  return result;
}

bool ValidateBaseUrl(std::string_view base_url, std::string* error) {
  const bool http = base_url.starts_with("http://");
  const bool https = base_url.starts_with("https://");
  if (!http && !https) {
    SetError(error, "--base-url must use http:// or https://");
    return false;
  }
  if (base_url.find('?') != std::string_view::npos ||
      base_url.find('#') != std::string_view::npos) {
    SetError(error, "--base-url must not contain a query or fragment");
    return false;
  }
  const std::size_t authority_start = base_url.find("://") + 3;
  const std::size_t authority_end = base_url.find('/', authority_start);
  const std::string_view authority =
      base_url.substr(authority_start, authority_end - authority_start);
  if (authority.empty()) {
    SetError(error, "--base-url is missing a host");
    return false;
  }
  if (authority.find('@') != std::string_view::npos) {
    SetError(error, "--base-url must not embed credentials");
    return false;
  }
  return true;
}

std::string NormalizeBaseUrl(std::string base_url) {
  while (!base_url.empty() && base_url.back() == '/') {
    base_url.pop_back();
  }
  return base_url;
}

}  // namespace gufo::eval
