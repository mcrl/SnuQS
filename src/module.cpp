#include <pybind11/pybind11.h>
#include "executor/executor.h"

namespace py = pybind11;

PYBIND11_MODULE(_snuqs, _m)
{
  _m.doc() = "SnuQS Pybind11 module.";

  auto m = _m.def_submodule("impl", "SnuQS implementation module.");
  m.doc() = "snuqs executor implementation"; 

  py::class_<snuqs::Executor>(m, "Executor")
    .def(py::init<>())
    .def("run", &snuqs::Executor::run, "run");
}
