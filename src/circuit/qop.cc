#include "qop.h"
#include "assertion.h"

#include <iostream>
#include <sstream>

namespace snuqs {

//
// Qop
//
Qop::Qop(QopType type) : type_(type) {}
Qop::Qop(QopType type, std::vector<std::shared_ptr<Qarg>> qargs)
    : type_(type), qargs_(qargs) {}
Qop::Qop(QopType type, std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : type_(type), qargs_(qargs), params_(params) {}
Qop::~Qop() {}
QopType Qop::type() const { return type_; }
void Qop::setQargs(std::vector<std::shared_ptr<Qarg>> qargs) { qargs_ = qargs; }
std::vector<std::shared_ptr<Qarg>> Qop::qargs() { return qargs_; }
std::vector<std::shared_ptr<Parameter>> Qop::params() { return params_; }
std::string Qop::__repr__() const { return "qop"; }

Barrier::Barrier(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qop(QopType::BARRIER, qargs) {}
std::string Barrier::__repr__() const {
  std::ostringstream s;
  s << "barrier ";
  for (auto &q : qargs_) {
    s << q->__repr__();
  }
  return s.str();
}

Reset::Reset(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qop(QopType::RESET, qargs) {}
std::string Reset::__repr__() const {
  std::ostringstream s;
  s << "reset ";
  for (auto &q : qargs_) {
    s << q->__repr__();
  }
  return s.str();
}

Measure::Measure(std::vector<std::shared_ptr<Qarg>> qargs,
                 std::vector<Carg> cbits)
    : Qop(QopType::MEASURE, qargs), cbits_(cbits) {}
std::string Measure::__repr__() const {
  std::ostringstream s;
  s << "Measure ";
  for (auto &q : qargs_) {
    s << q->__repr__();
  }
  s << " -> ";
  for (auto &c : cbits_) {
    s << c.__repr__();
  }

  return s.str();
}

Cond::Cond(std::shared_ptr<Qop> op, std::shared_ptr<Creg> creg, size_t val)
    : Qop(QopType::COND), op_(op), creg_(creg), val_(val) {}
std::string Cond::__repr__() const {
  std::ostringstream s;
  s << "if (" << creg_->__repr__() << " == " << val_ << ") ";
  s << op_->__repr__();

  return s.str();
}

Custom::Custom(const std::string &name, std::vector<std::shared_ptr<Qop>> qops,
               std::vector<std::shared_ptr<Qarg>> qargs,
               std::vector<std::shared_ptr<Parameter>> params)
    : Qop(QopType::CUSTOM, qargs, params), name_(name), qops_(qops) {}
std::vector<std::shared_ptr<Qop>> Custom::qops() { return qops_; }
std::string Custom::__repr__() const {
  std::ostringstream s;

  if (params_.size() > 0) {
    s << "(";
    for (auto &p : params_) {
      s << p->eval() << ", ";
    }
    s << ")";
  }

  s << " ";
  for (auto &q : qargs_) {
    s << q->__repr__() << ", ";
  }
  s << "\n";
  //  for (auto &op : qops_) {
  //    s << op->__repr__() << "\n";
  //  }

  return s.str();
}

static std::string qgateTypeToString(QgateType type) {
  switch (type) {
  case QgateType::ID:
    return "ID";
  case QgateType::X:
    return "X";
  case QgateType::Y:
    return "Y";
  case QgateType::Z:
    return "Z";
  case QgateType::H:
    return "H";
  case QgateType::S:
    return "S";
  case QgateType::SDG:
    return "SDG";
  case QgateType::T:
    return "T";
  case QgateType::TDG:
    return "TDG";
  case QgateType::SX:
    return "SX";
  case QgateType::SXDG:
    return "SXDG";
  case QgateType::P:
    return "P";
  case QgateType::RX:
    return "RX";
  case QgateType::RY:
    return "RY";
  case QgateType::RZ:
    return "RZ";
  case QgateType::U0:
    return "U0";
  case QgateType::U1:
    return "U1";
  case QgateType::U2:
    return "U2";
  case QgateType::U3:
    return "U3";
  case QgateType::U:
    return "U";
  case QgateType::CX:
    return "CX";
  case QgateType::CZ:
    return "CZ";
  case QgateType::CY:
    return "CY";
  case QgateType::SWAP:
    return "SWAP";
  case QgateType::CH:
    return "CH";
  case QgateType::CSX:
    return "CSX";
  case QgateType::CRX:
    return "CRX";
  case QgateType::CRY:
    return "CRY";
  case QgateType::CRZ:
    return "CRZ";
  case QgateType::CU1:
    return "CU1";
  case QgateType::CP:
    return "CP";
  case QgateType::RXX:
    return "RXX";
  case QgateType::RZZ:
    return "RZZ";
  case QgateType::CU3:
    return "CU3";
  case QgateType::CU:
    return "CU";
  case QgateType::CCX:
    return "CCX";
  case QgateType::CSWAP:
    return "CSWAP";
  }
  CANNOT_BE_HERE();
  return "INVALID";
}

Qgate::Qgate(QgateType gate_type, std::vector<std::shared_ptr<Qarg>> qargs)
    : Qop(QopType::QGATE, qargs), gate_type_(gate_type) {}

Qgate::Qgate(QgateType gate_type, std::vector<std::shared_ptr<Qarg>> qargs,
             std::vector<std::shared_ptr<Parameter>> params)
    : Qop(QopType::QGATE, qargs, params), gate_type_(gate_type) {}

QgateType Qgate::gate_type() const { return gate_type_; }
std::string Qgate::__repr__() const {
  std::ostringstream s;
  s << qgateTypeToString(gate_type_);
  if (params_.size() > 0) {
    s << "(";
    for (auto &p : params_) {
      s << p->eval() << ", ";
    }
    s << ")";
  }

  s << " ";
  for (auto &q : qargs_) {
    s << q->__repr__() << ", ";
  }

  return s.str();
}

ID::ID(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::ID, qargs) {}
ID::ID(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::ID, qargs, params) {}
size_t ID::numQargs() const { return 1; }
size_t ID::numParams() const { return 0; }

X::X(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::X, qargs) {}
X::X(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::X, qargs, params) {}
size_t X::numQargs() const { return 1; }
size_t X::numParams() const { return 0; }

Y::Y(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::Y, qargs) {}
Y::Y(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::Y, qargs, params) {}
size_t Y::numQargs() const { return 1; }
size_t Y::numParams() const { return 0; }

Z::Z(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::Z, qargs) {}
Z::Z(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::Z, qargs, params) {}
size_t Z::numQargs() const { return 1; }
size_t Z::numParams() const { return 0; }

H::H(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::H, qargs) {}
H::H(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::H, qargs, params) {}
size_t H::numQargs() const { return 1; }
size_t H::numParams() const { return 0; }

S::S(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::S, qargs) {}
S::S(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::S, qargs, params) {}
size_t S::numQargs() const { return 1; }
size_t S::numParams() const { return 0; }

SDG::SDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SDG, qargs) {}
SDG::SDG(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SDG, qargs, params) {}
size_t SDG::numQargs() const { return 1; }
size_t SDG::numParams() const { return 0; }

T::T(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::T, qargs) {}
T::T(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::T, qargs, params) {}
size_t T::numQargs() const { return 1; }
size_t T::numParams() const { return 0; }

TDG::TDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::TDG, qargs) {}
TDG::TDG(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::TDG, qargs, params) {}
size_t TDG::numQargs() const { return 1; }
size_t TDG::numParams() const { return 0; }

SX::SX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SX, qargs) {}
SX::SX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SX, qargs, params) {}
size_t SX::numQargs() const { return 1; }
size_t SX::numParams() const { return 0; }

SXDG::SXDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SXDG, qargs) {}
SXDG::SXDG(std::vector<std::shared_ptr<Qarg>> qargs,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SXDG, qargs, params) {}
size_t SXDG::numQargs() const { return 1; }
size_t SXDG::numParams() const { return 0; }

P::P(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::P, qargs) {}
P::P(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::P, qargs, params) {}
size_t P::numQargs() const { return 1; }
size_t P::numParams() const { return 1; }

RX::RX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RX, qargs) {}
RX::RX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RX, qargs, params) {}
size_t RX::numQargs() const { return 1; }
size_t RX::numParams() const { return 1; }

RY::RY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RY, qargs) {}
RY::RY(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RY, qargs, params) {}
size_t RY::numQargs() const { return 1; }
size_t RY::numParams() const { return 1; }

RZ::RZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RZ, qargs) {}
RZ::RZ(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RZ, qargs, params) {}
size_t RZ::numQargs() const { return 1; }
size_t RZ::numParams() const { return 1; }

U0::U0(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U0, qargs) {}
U0::U0(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U0, qargs, params) {}
size_t U0::numQargs() const { return 1; }
size_t U0::numParams() const { return 1; }

U1::U1(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U1, qargs) {}
U1::U1(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U1, qargs, params) {}
size_t U1::numQargs() const { return 1; }
size_t U1::numParams() const { return 1; }

U2::U2(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U2, qargs) {}
U2::U2(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U2, qargs, params) {}
size_t U2::numQargs() const { return 1; }
size_t U2::numParams() const { return 2; }

U3::U3(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U3, qargs) {}
U3::U3(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U3, qargs, params) {}
size_t U3::numQargs() const { return 1; }
size_t U3::numParams() const { return 3; }

U::U(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::U, qargs) {}
U::U(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U, qargs, params) {}
size_t U::numQargs() const { return 1; }
size_t U::numParams() const { return 3; }

CX::CX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CX, qargs) {}
CX::CX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CX, qargs, params) {}
size_t CX::numQargs() const { return 2; }
size_t CX::numParams() const { return 0; }

CZ::CZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CZ, qargs) {}
CZ::CZ(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CZ, qargs, params) {}
size_t CZ::numQargs() const { return 2; }
size_t CZ::numParams() const { return 0; }

CY::CY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CY, qargs) {}
CY::CY(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CY, qargs, params) {}
size_t CY::numQargs() const { return 2; }
size_t CY::numParams() const { return 0; }

SWAP::SWAP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SWAP, qargs) {}
SWAP::SWAP(std::vector<std::shared_ptr<Qarg>> qargs,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SWAP, qargs, params) {}
size_t SWAP::numQargs() const { return 2; }
size_t SWAP::numParams() const { return 0; }

CH::CH(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CH, qargs) {}
CH::CH(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CH, qargs, params) {}
size_t CH::numQargs() const { return 2; }
size_t CH::numParams() const { return 0; }

CSX::CSX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CSX, qargs) {}
CSX::CSX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CSX, qargs, params) {}
size_t CSX::numQargs() const { return 2; }
size_t CSX::numParams() const { return 0; }

CRX::CRX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRX, qargs) {}
CRX::CRX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRX, qargs, params) {}
size_t CRX::numQargs() const { return 2; }
size_t CRX::numParams() const { return 1; }

CRY::CRY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRY, qargs) {}
CRY::CRY(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRY, qargs, params) {}
size_t CRY::numQargs() const { return 2; }
size_t CRY::numParams() const { return 1; }

CRZ::CRZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRZ, qargs) {}
CRZ::CRZ(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRZ, qargs, params) {}
size_t CRZ::numQargs() const { return 2; }
size_t CRZ::numParams() const { return 1; }

CU1::CU1(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU1, qargs) {}
CU1::CU1(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU1, qargs, params) {}
size_t CU1::numQargs() const { return 2; }
size_t CU1::numParams() const { return 1; }

CP::CP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CP, qargs) {}
CP::CP(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CP, qargs, params) {}
size_t CP::numQargs() const { return 2; }
size_t CP::numParams() const { return 1; }

RXX::RXX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RXX, qargs) {}
RXX::RXX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RXX, qargs, params) {}
size_t RXX::numQargs() const { return 2; }
size_t RXX::numParams() const { return 1; }

RZZ::RZZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RZZ, qargs) {}
RZZ::RZZ(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RZZ, qargs, params) {}
size_t RZZ::numQargs() const { return 2; }
size_t RZZ::numParams() const { return 1; }

CU3::CU3(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU3, qargs) {}
CU3::CU3(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU3, qargs, params) {}
size_t CU3::numQargs() const { return 2; }
size_t CU3::numParams() const { return 3; }

CU::CU(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU, qargs) {}
CU::CU(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU, qargs, params) {}
size_t CU::numQargs() const { return 2; }
size_t CU::numParams() const { return 4; }

CCX::CCX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CCX, qargs) {}
CCX::CCX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CCX, qargs, params) {}
size_t CCX::numQargs() const { return 3; }
size_t CCX::numParams() const { return 0; }

CSWAP::CSWAP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CSWAP, qargs) {}
CSWAP::CSWAP(std::vector<std::shared_ptr<Qarg>> qargs,
             std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CSWAP, qargs, params) {}
size_t CSWAP::numQargs() const { return 3; }
size_t CSWAP::numParams() const { return 0; }

GlobalSwap::GlobalSwap(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qop(QopType::GLOBAL_SWAP, qargs) {}
std::string GlobalSwap::__repr__() const {
  std::ostringstream s;
  s << "global-swap ";
  s << qargs_[0]->__repr__() << " <-> " << qargs_[1]->__repr__();
  return s.str();
}

} // namespace snuqs
