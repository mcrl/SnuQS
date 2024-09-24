#include <iostream>
#include "tensordot.h"

static void contract(py::array_t<std::complex<double>> gate,
                     std::vector<size_t> indices0,
                     py::array_t<std::complex<double>> state,
                     std::vector<size_t> indices1) {
  size_t size = state.size();
  for (size_t i = 0; i < size; ++i) {
  }
}

py::array_t<std::complex<double>>
tensordot(py::array_t<std::complex<double>> gate,
          py::array_t<std::complex<double>> state,
          std::pair<std::vector<size_t>, std::vector<size_t>> axes) {
  assert(gate.ndim() == state.ndim());
  assert(axes.first.size() == axes.second.size());

  contract(gate, axes.first, state, axes.second);
  return state;
}
