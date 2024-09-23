#include "tensordot.h"
#include <iostream>

static void contract(py::array_t<std::complex<double>> gate, size_t idx0,
                     py::array_t<std::complex<double>> state, size_t idx1) {
    throw "NOT IMPLEMENTED";
}

py::array_t<std::complex<double>>
tensordot(py::array_t<std::complex<double>> gate,
          py::array_t<std::complex<double>> state,
          std::pair<std::vector<size_t>, std::vector<size_t>> axes) {
  assert(gate.ndim() == state.ndim());
  assert(axes.first.size() == axes.second.size());

  for (int i = 0; i < axes.first.size(); ++i) {
    contract(gate, axes.first[i], state, axes.second[i]);
  }
  return state;
}
