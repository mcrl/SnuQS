#ifndef _TENSORDOT_H_
#define _TENSORDOT_H_
#include <utility>
#include <vector>

#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
py::array_t<std::complex<double>>
tensordot(py::array_t<std::complex<double>> gate,
          py::array_t<std::complex<double>> state,
          std::pair<std::vector<size_t>, std::vector<size_t>> axes) {
#endif // _TENSORDOT_H_
