#ifndef GUFO_CORE_HRX_HRX_UTILS_HPP_
#define GUFO_CORE_HRX_HRX_UTILS_HPP_

#include <hrx/hrx_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

#define HRX_CHECK(expr)                                                     \
  do {                                                                      \
    hrx_status_t status = (expr);                                           \
    if (!hrx_status_is_ok(status)) {                                        \
      char* msg = nullptr;                                                  \
      size_t len = 0;                                                       \
      hrx_status_to_string(status, &msg, &len);                             \
      std::string err_msg = std::string("HRX error in ") + __FILE__ + ":" + \
                            std::to_string(__LINE__) +                      \
                            " (" #expr "): " + (msg ? msg : "unknown");     \
      std::cerr << err_msg << std::endl;                                    \
      hrx_status_free_message(msg);                                         \
      hrx_status_ignore(status);                                            \
      std::abort();                                                         \
    }                                                                       \
  } while (0)

#endif  // GUFO_CORE_HRX_HRX_UTILS_HPP_
