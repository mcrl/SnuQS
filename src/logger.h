#pragma once

#include <cstdio>
//#include <format>

namespace snuqs {

class Logger {

  public:
  template<typename... Args>
  static void error(Args... args) {
    printf("[Error] ", args...);
  }

  template<typename... Args>
  static void info(Args... args) {
    printf("[Info] ", args...);
  }

  template<typename... Args>
  static void debug(Args... args) {
    printf("[Debug] ", args...);
  }

};

} // namespace snuqs
