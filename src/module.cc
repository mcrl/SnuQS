#include "buffer/buffer.h"
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
  py::class_<snuqs::Buffer<double>, std::shared_ptr<snuqs::Buffer<double>>>(
      m, "Buffer");
  py::class_<snuqs::MemoryBuffer<double>, snuqs::Buffer<double>,
             std::shared_ptr<snuqs::MemoryBuffer<double>>>(m, "MemoryBuffer")
      .def(py::init<size_t>())
      .def(py::init<size_t, bool>())
      .def("__getitem__", &snuqs::MemoryBuffer<double>::__getitem__)
      .def("__setitem__", &snuqs::MemoryBuffer<double>::__setitem__);

  py::class_<snuqs::StorageBuffer<double>, snuqs::Buffer<double>,
             std::shared_ptr<snuqs::StorageBuffer<double>>>(m, "StorageBuffer")
      .def(py::init<size_t, std::vector<std::string>>())
      .def("__getitem__", &snuqs::StorageBuffer<double>::__getitem__)
      .def("__setitem__", &snuqs::StorageBuffer<double>::__setitem__);

  //
  // Reg
  //
  py::class_<snuqs::Reg, std::shared_ptr<snuqs::Reg>>(m, "Reg");
  py::class_<snuqs::Qreg, snuqs::Reg, std::shared_ptr<snuqs::Qreg>>(m, "Qreg")
      .def(py::init<std::string, size_t>())
      .def("name", &snuqs::Qreg::name)
      .def("dim", &snuqs::Qreg::dim)
      .def("__repr__", &snuqs::Qreg::__repr__);
  py::class_<snuqs::Creg, snuqs::Reg, std::shared_ptr<snuqs::Creg>>(m, "Creg")
      .def(py::init<std::string, size_t>())
      .def("name", &snuqs::Creg::name)
      .def("dim", &snuqs::Creg::dim)
      .def("value", &snuqs::Creg::value)
      .def("__repr__", &snuqs::Creg::__repr__)
      .def("__getitem__", &snuqs::Creg::__getitem__)
      .def("__setitem__", &snuqs::Creg::__setitem__);

  //
  // Arg
  //
  py::class_<snuqs::Qarg, std::shared_ptr<snuqs::Qarg>>(m, "Qarg")
      .def(py::init<std::shared_ptr<const snuqs::Qreg>>())
      .def(py::init<std::shared_ptr<const snuqs::Qreg>, size_t>())
      .def("index", &snuqs::Qarg::index)
      .def("__repr__", &snuqs::Qarg::__repr__);

  py::class_<snuqs::Carg, std::shared_ptr<snuqs::Carg>>(m, "Carg")
      .def(py::init<const snuqs::Creg &>())
      .def(py::init<const snuqs::Creg &, size_t>())
      .def("__repr__", &snuqs::Carg::__repr__);

  py::class_<snuqs::StatevectorSimulator<double>>(m, "StatevectorSimulator")
      .def(py::init<>())
      .def("run", &snuqs::StatevectorSimulator<double>::run);

  //
  // Parameter
  //
  py::class_<snuqs::Parameter, std::shared_ptr<snuqs::Parameter>>(m,
                                                                  "Parameter");

  py::class_<snuqs::Identifier, snuqs::Parameter,
             std::shared_ptr<snuqs::Identifier>>(m, "Identifier")
      .def(py::init<const snuqs::Creg &>())
      .def("eval", &snuqs::Identifier::eval);

  py::class_<snuqs::BinOp, snuqs::Parameter, std::shared_ptr<snuqs::BinOp>>(
      m, "BinOp")
      .def(py::init<snuqs::BinOpType, std::shared_ptr<snuqs::Parameter>,
                    std::shared_ptr<snuqs::Parameter>>())
      .def("eval", &snuqs::BinOp::eval);
  py::enum_<snuqs::BinOpType>(m, "BinOpType")
      .value("ADD", snuqs::BinOpType::ADD)
      .value("SUB", snuqs::BinOpType::SUB)
      .value("MUL", snuqs::BinOpType::MUL)
      .value("DIV", snuqs::BinOpType::DIV);

  py::class_<snuqs::NegOp, snuqs::Parameter, std::shared_ptr<snuqs::NegOp>>(
      m, "NegOp")
      .def(py::init<std::shared_ptr<snuqs::Parameter>>())
      .def("eval", &snuqs::NegOp::eval);

  py::class_<snuqs::UnaryOp, snuqs::Parameter, std::shared_ptr<snuqs::UnaryOp>>(
      m, "UnaryOp")
      .def(py::init<snuqs::UnaryOpType, std::shared_ptr<snuqs::Parameter>>())
      .def("eval", &snuqs::UnaryOp::eval);
  py::enum_<snuqs::UnaryOpType>(m, "UnaryOpType")
      .value("SIN", snuqs::UnaryOpType::SIN)
      .value("COS", snuqs::UnaryOpType::COS)
      .value("TAN", snuqs::UnaryOpType::TAN)
      .value("EXP", snuqs::UnaryOpType::EXP)
      .value("LN", snuqs::UnaryOpType::LN)
      .value("SQRT", snuqs::UnaryOpType::SQRT);

  py::class_<snuqs::Parenthesis, snuqs::Parameter,
             std::shared_ptr<snuqs::Parenthesis>>(m, "Parenthesis")
      .def(py::init<std::shared_ptr<snuqs::Parameter>>())
      .def("eval", &snuqs::Parenthesis::eval);

  py::class_<snuqs::Constant, snuqs::Parameter,
             std::shared_ptr<snuqs::Constant>>(m, "Constant")
      .def(py::init<double>())
      .def("eval", &snuqs::Constant::eval);

  py::class_<snuqs::Pi, snuqs::Constant, std::shared_ptr<snuqs::Pi>>(m, "Pi")
      .def(py::init<>())
      .def("eval", &snuqs::Pi::eval);

  //
  // Circuit
  //
  py::class_<snuqs::Circuit>(m, "Circuit")
      .def(py::init<const std::string &>())
      .def("append_qreg", &snuqs::Circuit::append_qreg)
      .def("append_creg", &snuqs::Circuit::append_creg)
      .def("append", &snuqs::Circuit::append)
      .def("name", &snuqs::Circuit::name)
      .def("__repr__", &snuqs::Circuit::__repr__);

  //
  // Qop
  //
  py::enum_<snuqs::QopType>(m, "QopType")
      .value("BARRIER", snuqs::QopType::BARRIER)
      .value("RESET", snuqs::QopType::RESET)
      .value("MEASURE", snuqs::QopType::MEASURE)
      .value("COND", snuqs::QopType::COND)
      .value("CUSTOM", snuqs::QopType::CUSTOM)
      .value("QGATE", snuqs::QopType::QGATE);

  py::class_<snuqs::Qop, std::shared_ptr<snuqs::Qop>>(m, "Qop")
      .def(py::init<snuqs::QopType>())
      .def(
          py::init<snuqs::QopType, std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<snuqs::QopType, std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("qargs", &snuqs::Qop::qargs)
      .def("params", &snuqs::Qop::params);

  py::class_<snuqs::Barrier, snuqs::Qop, std::shared_ptr<snuqs::Barrier>>(
      m, "Barrier")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>());

  py::class_<snuqs::Reset, snuqs::Qop, std::shared_ptr<snuqs::Reset>>(m,
                                                                      "Reset")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>());

  py::class_<snuqs::Measure, snuqs::Qop, std::shared_ptr<snuqs::Measure>>(
      m, "Measure")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<snuqs::Carg>>());

  py::class_<snuqs::Cond, snuqs::Qop, std::shared_ptr<snuqs::Cond>>(m, "Cond")
      .def(py::init<std::shared_ptr<snuqs::Qop>, std::shared_ptr<snuqs::Creg>,
                    size_t>());

  py::class_<snuqs::Custom, snuqs::Qop, std::shared_ptr<snuqs::Custom>>(
      m, "Custom")
      .def(py::init<std::string, std::vector<std::shared_ptr<snuqs::Qop>>,
                    std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>());

  py::enum_<snuqs::QgateType>(m, "QgateType")
      .value("ID", snuqs::QgateType::ID)
      .value("X", snuqs::QgateType::X)
      .value("Y", snuqs::QgateType::Y)
      .value("Z", snuqs::QgateType::Z)
      .value("H", snuqs::QgateType::H)
      .value("S", snuqs::QgateType::S)
      .value("SDG", snuqs::QgateType::SDG)
      .value("T", snuqs::QgateType::T)
      .value("TDG", snuqs::QgateType::TDG)
      .value("SX", snuqs::QgateType::SX)
      .value("SXDG", snuqs::QgateType::SXDG)
      .value("P", snuqs::QgateType::P)
      .value("RX", snuqs::QgateType::RX)
      .value("RY", snuqs::QgateType::RY)
      .value("RZ", snuqs::QgateType::RZ)
      .value("U0", snuqs::QgateType::U0)
      .value("U1", snuqs::QgateType::U1)
      .value("U2", snuqs::QgateType::U2)
      .value("U3", snuqs::QgateType::U3)
      .value("U", snuqs::QgateType::U)
      .value("CX", snuqs::QgateType::CX)
      .value("CZ", snuqs::QgateType::CZ)
      .value("CY", snuqs::QgateType::CY)
      .value("SWAP", snuqs::QgateType::SWAP)
      .value("CH", snuqs::QgateType::CH)
      .value("CSX", snuqs::QgateType::CSX)
      .value("CRX", snuqs::QgateType::CRX)
      .value("CRY", snuqs::QgateType::CRY)
      .value("CRZ", snuqs::QgateType::CRZ)
      .value("CU1", snuqs::QgateType::CU1)
      .value("CP", snuqs::QgateType::CP)
      .value("RXX", snuqs::QgateType::RXX)
      .value("RZZ", snuqs::QgateType::RZZ)
      .value("CU3", snuqs::QgateType::CU3)
      .value("CU", snuqs::QgateType::CU)
      .value("CCX", snuqs::QgateType::CCX)
      .value("CSWAP", snuqs::QgateType::CSWAP);

  py::class_<snuqs::Qgate, snuqs::Qop, std::shared_ptr<snuqs::Qgate>>(m,
                                                                      "Qgate");

  py::class_<snuqs::ID, snuqs::Qgate, std::shared_ptr<snuqs::ID>>(m, "ID")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::ID::numQargs)
      .def("numParams", &snuqs::ID::numParams);

  py::class_<snuqs::X, snuqs::Qgate, std::shared_ptr<snuqs::X>>(m, "X")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::X::numQargs)
      .def("numParams", &snuqs::X::numParams);

  py::class_<snuqs::Y, snuqs::Qgate, std::shared_ptr<snuqs::Y>>(m, "Y")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::Y::numQargs)
      .def("numParams", &snuqs::Y::numParams);

  py::class_<snuqs::Z, snuqs::Qgate, std::shared_ptr<snuqs::Z>>(m, "Z")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::Z::numQargs)
      .def("numParams", &snuqs::Z::numParams);

  py::class_<snuqs::H, snuqs::Qgate, std::shared_ptr<snuqs::H>>(m, "H")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::H::numQargs)
      .def("numParams", &snuqs::H::numParams);

  py::class_<snuqs::S, snuqs::Qgate, std::shared_ptr<snuqs::S>>(m, "S")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::S::numQargs)
      .def("numParams", &snuqs::S::numParams);

  py::class_<snuqs::SDG, snuqs::Qgate, std::shared_ptr<snuqs::SDG>>(m, "SDG")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::SDG::numQargs)
      .def("numParams", &snuqs::SDG::numParams);

  py::class_<snuqs::T, snuqs::Qgate, std::shared_ptr<snuqs::T>>(m, "T")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::T::numQargs)
      .def("numParams", &snuqs::T::numParams);

  py::class_<snuqs::TDG, snuqs::Qgate, std::shared_ptr<snuqs::TDG>>(m, "TDG")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::TDG::numQargs)
      .def("numParams", &snuqs::TDG::numParams);

  py::class_<snuqs::SX, snuqs::Qgate, std::shared_ptr<snuqs::SX>>(m, "SX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::SX::numQargs)
      .def("numParams", &snuqs::SX::numParams);

  py::class_<snuqs::SXDG, snuqs::Qgate, std::shared_ptr<snuqs::SXDG>>(m, "SXDG")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::SXDG::numQargs)
      .def("numParams", &snuqs::SXDG::numParams);

  py::class_<snuqs::P, snuqs::Qgate, std::shared_ptr<snuqs::P>>(m, "P")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::P::numQargs)
      .def("numParams", &snuqs::P::numParams);

  py::class_<snuqs::RX, snuqs::Qgate, std::shared_ptr<snuqs::RX>>(m, "RX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::RX::numQargs)
      .def("numParams", &snuqs::RX::numParams);

  py::class_<snuqs::RY, snuqs::Qgate, std::shared_ptr<snuqs::RY>>(m, "RY")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::RY::numQargs)
      .def("numParams", &snuqs::RY::numParams);

  py::class_<snuqs::RZ, snuqs::Qgate, std::shared_ptr<snuqs::RZ>>(m, "RZ")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::RZ::numQargs)
      .def("numParams", &snuqs::RZ::numParams);

  py::class_<snuqs::U0, snuqs::Qgate, std::shared_ptr<snuqs::U0>>(m, "U0")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::U0::numQargs)
      .def("numParams", &snuqs::U0::numParams);

  py::class_<snuqs::U1, snuqs::Qgate, std::shared_ptr<snuqs::U1>>(m, "U1")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::U1::numQargs)
      .def("numParams", &snuqs::U1::numParams);

  py::class_<snuqs::U2, snuqs::Qgate, std::shared_ptr<snuqs::U2>>(m, "U2")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::U2::numQargs)
      .def("numParams", &snuqs::U2::numParams);

  py::class_<snuqs::U3, snuqs::Qgate, std::shared_ptr<snuqs::U3>>(m, "U3")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::U3::numQargs)
      .def("numParams", &snuqs::U3::numParams);

  py::class_<snuqs::U, snuqs::Qgate, std::shared_ptr<snuqs::U>>(m, "U")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::U::numQargs)
      .def("numParams", &snuqs::U::numParams);

  py::class_<snuqs::CX, snuqs::Qgate, std::shared_ptr<snuqs::CX>>(m, "CX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CX::numQargs)
      .def("numParams", &snuqs::CX::numParams);

  py::class_<snuqs::CZ, snuqs::Qgate, std::shared_ptr<snuqs::CZ>>(m, "CZ")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CZ::numQargs)
      .def("numParams", &snuqs::CZ::numParams);

  py::class_<snuqs::CY, snuqs::Qgate, std::shared_ptr<snuqs::CY>>(m, "CY")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CY::numQargs)
      .def("numParams", &snuqs::CY::numParams);

  py::class_<snuqs::SWAP, snuqs::Qgate, std::shared_ptr<snuqs::SWAP>>(m, "SWAP")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::SWAP::numQargs)
      .def("numParams", &snuqs::SWAP::numParams);

  py::class_<snuqs::CH, snuqs::Qgate, std::shared_ptr<snuqs::CH>>(m, "CH")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CH::numQargs)
      .def("numParams", &snuqs::CH::numParams);

  py::class_<snuqs::CSX, snuqs::Qgate, std::shared_ptr<snuqs::CSX>>(m, "CSX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CSX::numQargs)
      .def("numParams", &snuqs::CSX::numParams);

  py::class_<snuqs::CRX, snuqs::Qgate, std::shared_ptr<snuqs::CRX>>(m, "CRX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CRX::numQargs)
      .def("numParams", &snuqs::CRX::numParams);

  py::class_<snuqs::CRY, snuqs::Qgate, std::shared_ptr<snuqs::CRY>>(m, "CRY")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CRY::numQargs)
      .def("numParams", &snuqs::CRY::numParams);

  py::class_<snuqs::CRZ, snuqs::Qgate, std::shared_ptr<snuqs::CRZ>>(m, "CRZ")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CRZ::numQargs)
      .def("numParams", &snuqs::CRZ::numParams);

  py::class_<snuqs::CU1, snuqs::Qgate, std::shared_ptr<snuqs::CU1>>(m, "CU1")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CU1::numQargs)
      .def("numParams", &snuqs::CU1::numParams);

  py::class_<snuqs::CP, snuqs::Qgate, std::shared_ptr<snuqs::CP>>(m, "CP")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CP::numQargs)
      .def("numParams", &snuqs::CP::numParams);

  py::class_<snuqs::RXX, snuqs::Qgate, std::shared_ptr<snuqs::RXX>>(m, "RXX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::RXX::numQargs)
      .def("numParams", &snuqs::RXX::numParams);

  py::class_<snuqs::RZZ, snuqs::Qgate, std::shared_ptr<snuqs::RZZ>>(m, "RZZ")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::RZZ::numQargs)
      .def("numParams", &snuqs::RZZ::numParams);

  py::class_<snuqs::CU3, snuqs::Qgate, std::shared_ptr<snuqs::CU3>>(m, "CU3")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CU3::numQargs)
      .def("numParams", &snuqs::CU3::numParams);

  py::class_<snuqs::CU, snuqs::Qgate, std::shared_ptr<snuqs::CU>>(m, "CU")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CU::numQargs)
      .def("numParams", &snuqs::CU::numParams);

  py::class_<snuqs::CCX, snuqs::Qgate, std::shared_ptr<snuqs::CCX>>(m, "CCX")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CCX::numQargs)
      .def("numParams", &snuqs::CCX::numParams);

  py::class_<snuqs::CSWAP, snuqs::Qgate, std::shared_ptr<snuqs::CSWAP>>(m,
                                                                        "CSWAP")
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>>())
      .def(py::init<std::vector<std::shared_ptr<snuqs::Qarg>>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQargs", &snuqs::CSWAP::numQargs)
      .def("numParams", &snuqs::CSWAP::numParams);
}
