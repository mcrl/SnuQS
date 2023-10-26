#include <pybind11/pybind11.h>
#include "launcher/launcher.h"

namespace py = pybind11;

PYBIND11_MODULE(_snuqs_impl, _m)
{
  _m.doc() = "SnuQS Pybind11 module.";

  auto m = _m.def_submodule("impl", "SnuQS implementation module.");
  m.doc() = "SnuQS implementation"; 

  py::class_<snuqs::Launcher>(m, "Launcher")
    .def(py::init<>())
    .def("run", &snuqs::Launcher::run, "run");
}
