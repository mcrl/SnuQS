#pragma once

#include <sstream>
#include <string>

namespace snuqs{
namespace string {

template <typename T>
std::string parenthesize(const T& x) {
  std::stringstream ss;
  ss << "(" << x << ")";
  return ss.str();
}

template <typename T>
std::string bracketize(const T& x) {
  std::stringstream ss;
  ss << "[" << x << "]";
  return ss.str();
}

template <typename T>
std::string quote(const T& x) {
  std::stringstream ss;
  ss << "\"" << x << "\"";
  return ss.str();
}

template <typename T>
std::string concat(const T& x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

template <typename T, typename... Q>
std::string concat(const T& x, const Q&... y) {
  std::stringstream ss;
  ss << x;
  return ss.str() + concat(y...);
}

} // namespace string
} // namespace snuqs
