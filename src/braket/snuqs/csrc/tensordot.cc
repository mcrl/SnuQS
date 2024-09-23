#include "tensordot.h"
#include <iostream>

py::array_t<std::complex<double>>
tensordot(py::array_t<std::complex<double>> gate,
          py::array_t<std::complex<double>> state,
          std::pair<std::vector<size_t>, std::vector<size_t>> axes) {
  std::cout << "gate.ndim(): " << gate.ndim() << "\n";
  std::cout << "state.ndim(): " << state.ndim() << "\n";

  std::cout << "axes.first: [";
  for (int i = 0; i < axes.first.size(); ++i) {
    std::cout << axes.first[i] << " ";
  }
  std::cout << "]\n";
  std::cout << "axes.second: [";
  for (int i = 0; i < axes.second.size(); ++i) {
    std::cout << axes.second[i] << " ";
  }
  std::cout << "]\n";
}
