#include "buffer/buffer.h"
#include "circuit/arg.h"
#include "circuit/circuit.h"
#include "circuit/parameter.h"
#include "circuit/qop.h"
#include "circuit/reg.h"
#include "simulator/statevector_simulator.h"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "SnuQS Pybind11 module.";

  //
  // Buffer
  // 
  py::class_<snuqs::MemoryBuffer>(m, "MemoryBuffer")
      .def(py::init<size_t>())
      .def("__getitem__", &snuqs::MemoryBuffer::__getitem__)
      .def("__setitem__", &snuqs::MemoryBuffer::__setitem__);

  py::class_<snuqs::StorageBuffer>(m, "StorageBuffer")
      .def(py::init<size_t, std::vector<std::string>>())
      .def("__getitem__", &snuqs::StorageBuffer::__getitem__)
      .def("__setitem__", &snuqs::StorageBuffer::__setitem__);

  //
  // Reg
  //
  py::class_<snuqs::Reg>(m, "Reg");
  py::class_<snuqs::Qreg, snuqs::Reg>(m, "Qreg")
      .def(py::init<std::string, size_t>())
      .def("name", &snuqs::Qreg::name)
      .def("dim", &snuqs::Qreg::dim)
      .def("__repr__", &snuqs::Qreg::__repr__);
  py::class_<snuqs::Creg, snuqs::Reg>(m, "Creg")
      .def(py::init<std::string, size_t>())
      .def("name", &snuqs::Creg::name)
      .def("dim", &snuqs::Creg::dim)
      .def("__repr__", &snuqs::Creg::__repr__)
      .def("__getitem__", &snuqs::Creg::__getitem__)
      .def("__setitem__", &snuqs::Creg::__setitem__);

  //
  // Arg
  //
  py::class_<snuqs::Arg>(m, "Arg");
  py::class_<snuqs::Qarg, snuqs::Arg>(m, "Qarg")
      .def(py::init<const snuqs::Qreg&, size_t>())
      .def("reg", &snuqs::Qarg::reg)
      .def("name", &snuqs::Qarg::index)
      .def("dim", &snuqs::Qarg::dim)
      .def("value", &snuqs::Qarg::value)
      .def("__repr__", &snuqs::Qarg::__repr__);
  py::class_<snuqs::Carg, snuqs::Arg>(m, "Carg")
      .def(py::init<const snuqs::Creg&, size_t>())
      .def("reg", &snuqs::Carg::reg)
      .def("name", &snuqs::Carg::index)
      .def("dim", &snuqs::Carg::dim)
      .def("value", &snuqs::Carg::value)
      .def("__repr__", &snuqs::Carg::__repr__);
      

  py::class_<snuqs::Qop>(m, "Qop").def(py::init<>());

  py::class_<snuqs::Circuit>(m, "Circuit")
      .def(py::init<size_t, size_t>())
      .def("append", &snuqs::Circuit::append);

  py::class_<snuqs::StatevectorSimulator>(m, "StatevectorSimulator")
      .def(py::init<>())
      .def("test", &snuqs::StatevectorSimulator::test);

  //
  // Parameter
  //
  py::class_<snuqs::Parameter>(m, "Parameter");

  py::class_<snuqs::BinOp, snuqs::Parameter>(m, "BinOp")
      .def(py::init<snuqs::BinOpType, const snuqs::Parameter &,
                    const snuqs::Parameter &>())
      .def("eval", &snuqs::BinOp::eval);
  py::enum_<snuqs::BinOpType>(m, "BinOpType")
      .value("ADD", snuqs::BinOpType::ADD)
      .value("SUB", snuqs::BinOpType::SUB)
      .value("MUL", snuqs::BinOpType::MUL)
      .value("DIV", snuqs::BinOpType::DIV);

  py::class_<snuqs::NegOp, snuqs::Parameter>(m, "NegOp")
      .def(py::init<const snuqs::Parameter &>())
      .def("eval", &snuqs::NegOp::eval);

  py::class_<snuqs::UnaryOp, snuqs::Parameter>(m, "UnaryOp")
      .def(py::init<snuqs::UnaryOpType, const snuqs::Parameter &>())
      .def("eval", &snuqs::UnaryOp::eval);
  py::enum_<snuqs::UnaryOpType>(m, "UnaryOpType")
      .value("SIN", snuqs::UnaryOpType::SIN)
      .value("COS", snuqs::UnaryOpType::COS)
      .value("TAN", snuqs::UnaryOpType::TAN)
      .value("EXP", snuqs::UnaryOpType::EXP)
      .value("LN", snuqs::UnaryOpType::LN)
      .value("SQRT", snuqs::UnaryOpType::SQRT);

  py::class_<snuqs::Parenthesis, snuqs::Parameter>(m, "Parenthesis")
      .def(py::init<const snuqs::Parameter &>())
      .def("eval", &snuqs::Parenthesis::eval);

  py::class_<snuqs::Constant, snuqs::Parameter>(m, "Constant")
      .def(py::init<double>())
      .def("eval", &snuqs::Constant::eval);

  py::class_<snuqs::Pi, snuqs::Constant>(m, "Pi")
      .def(py::init<>())
      .def("eval", &snuqs::Pi::eval);
}
