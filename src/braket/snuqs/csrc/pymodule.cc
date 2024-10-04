#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>

#include "cuda/runtime.h"
#include "functionals.h"
#include "gate_operations.h"
#include "operation.h"
#include "state_vector.h"

namespace py = pybind11;

#define GATEOP(name)                                                          \
  py::class_<name, GateOperation>(m_gate_operations, #name,                   \
                                  py::buffer_protocol())                      \
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
           py::arg("power") = 1)                                              \
      .def("__repr__", &GateOperation::formatted_string);

#define GATEOP1(name)                                                       \
  py::class_<name, GateOperation>(m_gate_operations, #name,                 \
                                  py::buffer_protocol())                    \
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
           py::arg("power") = 1)                                            \
      .def("__repr__", &GateOperation::formatted_string);

#define GATEOP3(name)                                                       \
  py::class_<name, GateOperation>(m_gate_operations, #name,                 \
                                  py::buffer_protocol())                    \
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
           py::arg("power") = 1)                                            \
      .def("__repr__", &GateOperation::formatted_string);

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
      .def("cpu", &StateVector::cpu)
      .def("cuda", &StateVector::cuda);

  // Operation
  auto m_operation = m.def_submodule("operation");
  py::class_<Operation>(m_operation, "Operation")
      .def(py::init<const std::vector<size_t> &>())
      .def_property("targets", &Operation::get_targets,
                    &Operation::set_targets);

  // GateOperation
  py::class_<GateOperation, Operation>(m_operation, "GateOperation")
      .def(py::init<const std::vector<size_t> &, const std::vector<size_t> &,
                    size_t>());
  auto m_gate_operations = m.def_submodule("gate_operations");
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

  // Functionals
  auto m_functionals = m.def_submodule("functionals");
  m_functionals.def("apply", &functionals::apply);
  m_functionals.def("initialize_zero", &functionals::initialize_zero);
  m_functionals.def("initialize_basis_z", &functionals::initialize_basis_z);

  auto m_cuda = m.def_submodule("cuda");
  m_cuda.def("device_count", &cu::device_count);
}
