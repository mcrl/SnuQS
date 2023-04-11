#include <pybind11/pybind11.h>

#include "compiler/compiler.h"

namespace py = pybind11;


static void test1() {
  printf("test1\n");
}

static void test2() {
  printf("test2\n");
}

PYBIND11_MODULE(compiler, m) {
  m.doc() = "snuqs pybind11 implementation"; 

  m.def("test1", &test1, "");
  m.def("test2", &test2, "");

  py::class_<snuqs::Compiler>(m, "Compiler")
    .def("Compile", &snuqs::Compiler::Compile)
    .def("GetQuantumCircuit", &snuqs::Compiler::GetQuantumCircuit);
}
