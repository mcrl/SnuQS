#include "buffer/buffer.h"
#include "launcher/launcher.h"
#include "simulator/statevector_simulator.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";

  py::class_<snuqs::Launcher>(m, "Launcher")
      .def(py::init<>())
      .def("run", &snuqs::Launcher::run, "run");

  py::class_<snuqs::StatevectorSimulator>(m, "StatevectorSimulator")
      .def(py::init<>());

  py::class_<snuqs::MemoryBuffer>(m, "MemoryBuffer")
      .def(py::init<size_t>())
      .def("__getitem__", &snuqs::MemoryBuffer::__getitem__)
      .def("__setitem__", &snuqs::MemoryBuffer::__setitem__);

  py::class_<snuqs::StorageBuffer>(m, "StorageBuffer")
      .def(py::init<size_t>())
      .def("__getitem__", &snuqs::StorageBuffer::__getitem__)
      .def("__setitem__", &snuqs::StorageBuffer::__setitem__);
}
