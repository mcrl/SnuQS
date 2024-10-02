#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>

#include "evolution.h"
#include "gate_operation.h"
#include "initialization.h"
#include "operation.h"
#include "state_vector.h"

namespace py = pybind11;

#define GATEOP(name)                                                          \
  py::class_<name, GateOperation>(m, #name, py::buffer_protocol())            \
      .def_buffer([](name &g) -> py::buffer_info {                            \
        return py::buffer_info(                                               \
            g.data(), sizeof(std::complex<double>),                           \
            py::format_descriptor<std::complex<double>>::format(), g.dim(),   \
            g.shape(), g.stride());                                           \
      })                                                                      \
      .def(py::init<const std::vector<size_t> &, const std::vector<size_t> &, \
                    size_t>(),                                                \
           py::arg("targets"), py::kw_only(),                                 \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),                 \
           py::arg("power") = 1);

#define GATEOP1(name)                                                       \
  py::class_<name, GateOperation>(m, #name, py::buffer_protocol())          \
      .def_buffer([](name &g) -> py::buffer_info {                          \
        return py::buffer_info(                                             \
            g.data(), sizeof(std::complex<double>),                         \
            py::format_descriptor<std::complex<double>>::format(), g.dim(), \
            g.shape(), g.stride());                                         \
      })                                                                    \
      .def(py::init<const std::vector<size_t> &, double,                    \
                    const std::vector<size_t> &, size_t>(),                 \
           py::arg("targets"), py::arg("angle"), py::kw_only(),             \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),               \
           py::arg("power") = 1);

#define GATEOP3(name)                                                       \
  py::class_<name, GateOperation>(m, #name, py::buffer_protocol())          \
      .def_buffer([](name &g) -> py::buffer_info {                          \
        return py::buffer_info(                                             \
            g.data(), sizeof(std::complex<double>),                         \
            py::format_descriptor<std::complex<double>>::format(), g.dim(), \
            g.shape(), g.stride());                                         \
      })                                                                    \
      .def(py::init<const std::vector<size_t> &, double, double, double,    \
                    const std::vector<size_t> &, size_t>(),                 \
           py::arg("targets"), py::arg("angle_1"), py::arg("angle_2"),      \
           py::arg("angle_3"), py::kw_only(),                               \
           py::arg("ctrl_modifiers") = std::vector<size_t>(),               \
           py::arg("power") = 1);

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

  // Functions
  m.def("evolve", &evolve);
  m.def("initialize_zero", &initialize_zero);
  m.def("initialize_basis_z", &initialize_basis_z);

  // Operation
  py::class_<Operation>(m, "Operation")
      .def(py::init<const std::vector<size_t> &>())
      .def_property("targets", &Operation::get_targets,
                    &Operation::set_targets);
  //.def_property("_targets", &Operation::get_targets, &Operation::set_targets);

  // GateOperation
  py::class_<GateOperation, Operation>(m, "GateOperation")
      .def(py::init<const std::vector<size_t> &, const std::vector<size_t> &,
                    size_t>());

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
  GATEOP3(U);
  GATEOP1(GPhase);
}
