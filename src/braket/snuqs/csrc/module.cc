#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gate_operation.h"
#include "state_vector.h"
#include <complex>

namespace py = pybind11;

#define GATEOP(name)                                                           \
  py::class_<name>(m, #name, py::buffer_protocol())                            \
      .def_buffer([](name &g) -> py::buffer_info {                            \
        return py::buffer_info(                                                \
            g.data(), sizeof(std::complex<double>),                           \
            py::format_descriptor<std::complex<double>>::format(), g.dim(),   \
            g.shape(), g.stride());                                          \
      })                                                                       \
      .def(py::init<>())

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";

  // StateVector
  py::class_<StateVector>(m, "StateVector", py::buffer_protocol())
      .def_buffer([](StateVector &sv) -> py::buffer_info {
        return py::buffer_info(
            sv.data(),                    /* Pointer to buffer */
            sizeof(std::complex<double>), /* Size of one scalar */
            py::format_descriptor<std::complex<double>>::format(), /* Python
                                                       struct-style format
                                                       descriptor */
            sv.dim(), sv.shape(), {sizeof(std::complex<double>)});
      })
      .def(py::init<size_t>());

  // GateOperation
  GATEOP(Identity);
  GATEOP(Hadamard);
}
