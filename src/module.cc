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
  py::class_<snuqs::Arg>(m, "Arg");
  py::class_<snuqs::Qarg, snuqs::Arg>(m, "Qarg")
      .def(py::init<const snuqs::Qreg &>())
      .def(py::init<const snuqs::Qreg &, size_t>())
      .def("reg", &snuqs::Qarg::reg)
      .def("name", &snuqs::Qarg::index)
      .def("dim", &snuqs::Qarg::dim)
      .def("value", &snuqs::Qarg::value)
      .def("__repr__", &snuqs::Qarg::__repr__);
  py::class_<snuqs::Carg, snuqs::Arg>(m, "Carg")
      .def(py::init<const snuqs::Creg &>())
      .def(py::init<const snuqs::Creg &, size_t>())
      .def("reg", &snuqs::Carg::reg)
      .def("name", &snuqs::Carg::index)
      .def("dim", &snuqs::Carg::dim)
      .def("value", &snuqs::Carg::value)
      .def("__repr__", &snuqs::Carg::__repr__);

  py::class_<snuqs::StatevectorSimulator>(m, "StatevectorSimulator")
      .def(py::init<>())
      .def("test", &snuqs::StatevectorSimulator::test);

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
      .def(py::init<snuqs::BinOpType, const snuqs::Parameter &,
                    const snuqs::Parameter &>())
      .def("eval", &snuqs::BinOp::eval);
  py::enum_<snuqs::BinOpType>(m, "BinOpType")
      .value("ADD", snuqs::BinOpType::ADD)
      .value("SUB", snuqs::BinOpType::SUB)
      .value("MUL", snuqs::BinOpType::MUL)
      .value("DIV", snuqs::BinOpType::DIV);

  py::class_<snuqs::NegOp, snuqs::Parameter, std::shared_ptr<snuqs::NegOp>>(
      m, "NegOp")
      .def(py::init<const snuqs::Parameter &>())
      .def("eval", &snuqs::NegOp::eval);

  py::class_<snuqs::UnaryOp, snuqs::Parameter, std::shared_ptr<snuqs::UnaryOp>>(
      m, "UnaryOp")
      .def(py::init<snuqs::UnaryOpType, const snuqs::Parameter &>())
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
      .def(py::init<const snuqs::Parameter &>())
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
      .def("__repr__", &snuqs::Circuit::__repr__);

  //
  // Qop
  //
  py::class_<snuqs::Qop, std::shared_ptr<snuqs::Qop>>(m, "Qop")
      .def(py::init<>())
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>());

  py::class_<snuqs::Barrier, snuqs::Qop, std::shared_ptr<snuqs::Barrier>>(
      m, "Barrier")
      .def(py::init<std::vector<snuqs::Qarg>>());

  py::class_<snuqs::Reset, snuqs::Qop, std::shared_ptr<snuqs::Reset>>(m,
                                                                      "Reset")
      .def(py::init<std::vector<snuqs::Qarg>>());

  py::class_<snuqs::Measure, snuqs::Qop, std::shared_ptr<snuqs::Measure>>(
      m, "Measure")
      .def(py::init<std::vector<snuqs::Qarg>, std::vector<snuqs::Carg>>());

  py::class_<snuqs::Cond, snuqs::Qop, std::shared_ptr<snuqs::Cond>>(m, "Cond")
      .def(py::init<std::shared_ptr<snuqs::Qop>, std::shared_ptr<snuqs::Creg>, size_t>());

  py::class_<snuqs::Custom, snuqs::Qop, std::shared_ptr<snuqs::Custom>>(
      m, "Custom")
      .def(py::init<std::string, std::vector<std::shared_ptr<snuqs::Qop>>,
                    std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>());

  py::class_<snuqs::Qgate, snuqs::Qop, std::shared_ptr<snuqs::Qgate>>(m,
                                                                      "Qgate")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>());

  py::class_<snuqs::ID, snuqs::Qgate, std::shared_ptr<snuqs::ID>>(m, "ID")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::ID::numQubits)
      .def("numParams", &snuqs::ID::numParams);

  py::class_<snuqs::X, snuqs::Qgate, std::shared_ptr<snuqs::X>>(m, "X")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::X::numQubits)
      .def("numParams", &snuqs::X::numParams);

  py::class_<snuqs::Y, snuqs::Qgate, std::shared_ptr<snuqs::Y>>(m, "Y")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::Y::numQubits)
      .def("numParams", &snuqs::Y::numParams);

  py::class_<snuqs::Z, snuqs::Qgate, std::shared_ptr<snuqs::Z>>(m, "Z")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::Z::numQubits)
      .def("numParams", &snuqs::Z::numParams);

  py::class_<snuqs::H, snuqs::Qgate, std::shared_ptr<snuqs::H>>(m, "H")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::H::numQubits)
      .def("numParams", &snuqs::H::numParams);

  py::class_<snuqs::S, snuqs::Qgate, std::shared_ptr<snuqs::S>>(m, "S")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::S::numQubits)
      .def("numParams", &snuqs::S::numParams);

  py::class_<snuqs::SDG, snuqs::Qgate, std::shared_ptr<snuqs::SDG>>(m, "SDG")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::SDG::numQubits)
      .def("numParams", &snuqs::SDG::numParams);

  py::class_<snuqs::T, snuqs::Qgate, std::shared_ptr<snuqs::T>>(m, "T")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::T::numQubits)
      .def("numParams", &snuqs::T::numParams);

  py::class_<snuqs::TDG, snuqs::Qgate, std::shared_ptr<snuqs::TDG>>(m, "TDG")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::TDG::numQubits)
      .def("numParams", &snuqs::TDG::numParams);

  py::class_<snuqs::SX, snuqs::Qgate, std::shared_ptr<snuqs::SX>>(m, "SX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::SX::numQubits)
      .def("numParams", &snuqs::SX::numParams);

  py::class_<snuqs::SXDG, snuqs::Qgate, std::shared_ptr<snuqs::SXDG>>(m, "SXDG")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::SXDG::numQubits)
      .def("numParams", &snuqs::SXDG::numParams);

  py::class_<snuqs::P, snuqs::Qgate, std::shared_ptr<snuqs::P>>(m, "P")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::P::numQubits)
      .def("numParams", &snuqs::P::numParams);

  py::class_<snuqs::RX, snuqs::Qgate, std::shared_ptr<snuqs::RX>>(m, "RX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RX::numQubits)
      .def("numParams", &snuqs::RX::numParams);

  py::class_<snuqs::RY, snuqs::Qgate, std::shared_ptr<snuqs::RY>>(m, "RY")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RY::numQubits)
      .def("numParams", &snuqs::RY::numParams);

  py::class_<snuqs::RZ, snuqs::Qgate, std::shared_ptr<snuqs::RZ>>(m, "RZ")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RZ::numQubits)
      .def("numParams", &snuqs::RZ::numParams);

  py::class_<snuqs::U0, snuqs::Qgate, std::shared_ptr<snuqs::U0>>(m, "U0")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::U0::numQubits)
      .def("numParams", &snuqs::U0::numParams);

  py::class_<snuqs::U1, snuqs::Qgate, std::shared_ptr<snuqs::U1>>(m, "U1")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::U1::numQubits)
      .def("numParams", &snuqs::U1::numParams);

  py::class_<snuqs::U2, snuqs::Qgate, std::shared_ptr<snuqs::U2>>(m, "U2")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::U2::numQubits)
      .def("numParams", &snuqs::U2::numParams);

  py::class_<snuqs::U3, snuqs::Qgate, std::shared_ptr<snuqs::U3>>(m, "U3")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::U3::numQubits)
      .def("numParams", &snuqs::U3::numParams);

  py::class_<snuqs::U, snuqs::Qgate, std::shared_ptr<snuqs::U>>(m, "U")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::U::numQubits)
      .def("numParams", &snuqs::U::numParams);

  py::class_<snuqs::CX, snuqs::Qgate, std::shared_ptr<snuqs::CX>>(m, "CX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CX::numQubits)
      .def("numParams", &snuqs::CX::numParams);

  py::class_<snuqs::CZ, snuqs::Qgate, std::shared_ptr<snuqs::CZ>>(m, "CZ")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CZ::numQubits)
      .def("numParams", &snuqs::CZ::numParams);

  py::class_<snuqs::CY, snuqs::Qgate, std::shared_ptr<snuqs::CY>>(m, "CY")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CY::numQubits)
      .def("numParams", &snuqs::CY::numParams);

  py::class_<snuqs::SWAP, snuqs::Qgate, std::shared_ptr<snuqs::SWAP>>(m, "SWAP")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::SWAP::numQubits)
      .def("numParams", &snuqs::SWAP::numParams);

  py::class_<snuqs::CH, snuqs::Qgate, std::shared_ptr<snuqs::CH>>(m, "CH")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CH::numQubits)
      .def("numParams", &snuqs::CH::numParams);

  py::class_<snuqs::CSX, snuqs::Qgate, std::shared_ptr<snuqs::CSX>>(m, "CSX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CSX::numQubits)
      .def("numParams", &snuqs::CSX::numParams);

  py::class_<snuqs::CRX, snuqs::Qgate, std::shared_ptr<snuqs::CRX>>(m, "CRX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CRX::numQubits)
      .def("numParams", &snuqs::CRX::numParams);

  py::class_<snuqs::CRY, snuqs::Qgate, std::shared_ptr<snuqs::CRY>>(m, "CRY")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CRY::numQubits)
      .def("numParams", &snuqs::CRY::numParams);

  py::class_<snuqs::CRZ, snuqs::Qgate, std::shared_ptr<snuqs::CRZ>>(m, "CRZ")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CRZ::numQubits)
      .def("numParams", &snuqs::CRZ::numParams);

  py::class_<snuqs::CU1, snuqs::Qgate, std::shared_ptr<snuqs::CU1>>(m, "CU1")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CU1::numQubits)
      .def("numParams", &snuqs::CU1::numParams);

  py::class_<snuqs::CP, snuqs::Qgate, std::shared_ptr<snuqs::CP>>(m, "CP")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CP::numQubits)
      .def("numParams", &snuqs::CP::numParams);

  py::class_<snuqs::RXX, snuqs::Qgate, std::shared_ptr<snuqs::RXX>>(m, "RXX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RXX::numQubits)
      .def("numParams", &snuqs::RXX::numParams);

  py::class_<snuqs::RZZ, snuqs::Qgate, std::shared_ptr<snuqs::RZZ>>(m, "RZZ")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RZZ::numQubits)
      .def("numParams", &snuqs::RZZ::numParams);

  py::class_<snuqs::CU3, snuqs::Qgate, std::shared_ptr<snuqs::CU3>>(m, "CU3")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CU3::numQubits)
      .def("numParams", &snuqs::CU3::numParams);

  py::class_<snuqs::CU, snuqs::Qgate, std::shared_ptr<snuqs::CU>>(m, "CU")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CU::numQubits)
      .def("numParams", &snuqs::CU::numParams);

  py::class_<snuqs::CCX, snuqs::Qgate, std::shared_ptr<snuqs::CCX>>(m, "CCX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CCX::numQubits)
      .def("numParams", &snuqs::CCX::numParams);

  py::class_<snuqs::CSWAP, snuqs::Qgate, std::shared_ptr<snuqs::CSWAP>>(m,
                                                                        "CSWAP")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::CSWAP::numQubits)
      .def("numParams", &snuqs::CSWAP::numParams);

  py::class_<snuqs::RCCX, snuqs::Qgate, std::shared_ptr<snuqs::RCCX>>(m, "RCCX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RCCX::numQubits)
      .def("numParams", &snuqs::RCCX::numParams);

  py::class_<snuqs::RC3X, snuqs::Qgate, std::shared_ptr<snuqs::RC3X>>(m, "RC3X")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::RC3X::numQubits)
      .def("numParams", &snuqs::RC3X::numParams);

  py::class_<snuqs::C3X, snuqs::Qgate, std::shared_ptr<snuqs::C3X>>(m, "C3X")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::C3X::numQubits)
      .def("numParams", &snuqs::C3X::numParams);

  py::class_<snuqs::C3SQRTX, snuqs::Qgate, std::shared_ptr<snuqs::C3SQRTX>>(
      m, "C3SQRTX")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::C3SQRTX::numQubits)
      .def("numParams", &snuqs::C3SQRTX::numParams);

  py::class_<snuqs::C4X, snuqs::Qgate, std::shared_ptr<snuqs::C4X>>(m, "C4X")
      .def(py::init<std::vector<snuqs::Qarg>>())
      .def(py::init<std::vector<snuqs::Qarg>,
                    std::vector<std::shared_ptr<snuqs::Parameter>>>())
      .def("numQubits", &snuqs::C4X::numQubits)
      .def("numParams", &snuqs::C4X::numParams);
}
