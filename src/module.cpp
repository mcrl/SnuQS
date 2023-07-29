#include <pybind11/pybind11.h>
#include "executor/executor.h"

namespace py = pybind11;

PYBIND11_MODULE(_snuqs, _m)
{
  _m.doc() = "This is top module - mymodule.";

  auto m = _m.def_submodule("cpp", "This is A.");
  m.doc() = "snuqs executor implementation"; 

  py::class_<snuqs::Executor>(m, "Executor")
    .def(py::init<>())
    .def("run", &snuqs::Executor::run, "run");
}
