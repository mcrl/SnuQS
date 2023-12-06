#include "buffer/buffer.h"
#include "circuit/qop.h"
#include "circuit/circuit.h"
#include "simulator/statevector_simulator.h"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";

  py::class_<snuqs::MemoryBuffer>(m, "MemoryBuffer")
      .def(py::init<size_t>())
      .def("__getitem__", &snuqs::MemoryBuffer::__getitem__)
      .def("__setitem__", &snuqs::MemoryBuffer::__setitem__);

  py::class_<snuqs::StorageBuffer>(m, "StorageBuffer")
      .def(py::init<size_t, std::vector<std::string>>())
      .def("__getitem__", &snuqs::StorageBuffer::__getitem__)
      .def("__setitem__", &snuqs::StorageBuffer::__setitem__);

  py::class_<snuqs::Qop>(m, "Qop")
      .def(py::init<>());

  py::class_<snuqs::Circuit>(m, "Circuit")
      .def(py::init<size_t, size_t>())
      .def("append", &snuqs::Circuit::append);

  py::class_<snuqs::StatevectorSimulator>(m, "StatevectorSimulator")
      .def(py::init<>())
      .def("test", &snuqs::StatevectorSimulator::test);
}
