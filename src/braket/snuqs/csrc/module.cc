#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gate_operation.h"
#include "interface.h"
#include "state_vector.h"
#include <complex>

namespace py = pybind11;

#define GATEOP(name)                                                           \
  py::class_<name, GateOperation>(m, #name, py::buffer_protocol())             \
      .def_buffer([](name &g) -> py::buffer_info {                             \
        return py::buffer_info(                                                \
            g.data(), sizeof(std::complex<double>),                            \
            py::format_descriptor<std::complex<double>>::format(), g.dim(),    \
            g.shape(), g.stride());                                            \
      })                                                                       \
      .def(py::init<>())

#define GATEOP1(name)                                                          \
  py::class_<name, GateOperation>(m, #name, py::buffer_protocol())             \
      .def_buffer([](name &g) -> py::buffer_info {                             \
        return py::buffer_info(                                                \
            g.data(), sizeof(std::complex<double>),                            \
            py::format_descriptor<std::complex<double>>::format(), g.dim(),    \
            g.shape(), g.stride());                                            \
      })                                                                       \
      .def(py::init<double>())

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
      .def(py::init<size_t>())
      .def("toCPU", &StateVector::toCPU)
      .def("toCUDA", &StateVector::toCUDA);

  m.def("evolve", &evolve);

  // GateOperation
  py::class_<GateOperation>(m, "GateOperation");
  GATEOP(Identity);
  GATEOP(Hadamard);
  GATEOP(PauliX);
  GATEOP(PauliY);
  GATEOP(PauliZ);
  GATEOP(CX);
  GATEOP(CY);
  GATEOP(CZ);
  GATEOP(S);
  GATEOP(Si);
  GATEOP(T);
  GATEOP(Ti);
  GATEOP(V);
  GATEOP(Vi);
  GATEOP1(PhaseShift);
  GATEOP1(CPhaseShift);
  GATEOP1(CPhaseShift00);
  GATEOP1(CPhaseShift01);
  GATEOP1(CPhaseShift10);
  GATEOP1(RotX);
  GATEOP1(RotY);
  GATEOP1(RotZ);
  GATEOP(Swap);
  GATEOP(ISwap);
  GATEOP1(PSwap);
  GATEOP1(XY);
  GATEOP1(XX);
  GATEOP1(YY);
  GATEOP1(ZZ);
  GATEOP(CCNot);
  GATEOP(CSwap);
}
