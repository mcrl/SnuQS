#pragma once

#include <iostream>
#include <stdexcept>
#include "string.hpp"

namespace snuqs {
namespace error {

template <typename T, typename... Q>
std::exception runtime(const T& x, const Q&... y) {
    return std::runtime_error(string::concat(x, y...));
}

template <typename T, typename... Q>
void exitWithMessage(const T& x, const Q&... y) {
	std::cerr << string::concat(x, y...) << "\n";
	std::exit(EXIT_FAILURE);
}

} // namespace error
} // namespace snuqs
