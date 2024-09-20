#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";
  m.def("add", &add, "A function that adds two numbers");
}
