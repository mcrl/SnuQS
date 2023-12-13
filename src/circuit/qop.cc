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
std::shared_ptr<Qop> Qop::clone() const {
  return std::make_shared<Qop>(type_, qargs_, params_);
}

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

std::shared_ptr<Qop> Barrier::clone() const {
  return std::make_shared<Barrier>(qargs_);
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

std::shared_ptr<Qop> Reset::clone() const {
  return std::make_shared<Reset>(qargs_);
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

std::shared_ptr<Qop> Measure::clone() const {
  return std::make_shared<Measure>(qargs_, cbits_);
}

Cond::Cond(std::shared_ptr<Qop> op, std::shared_ptr<Creg> creg, size_t val)
    : Qop(QopType::COND), op_(op), creg_(creg), val_(val) {}
std::string Cond::__repr__() const {
  std::ostringstream s;
  s << "if (" << creg_->__repr__() << " == " << val_ << ") ";
  s << op_->__repr__();

  return s.str();
}

std::shared_ptr<Qop> Cond::clone() const {
  return std::make_shared<Cond>(op_->clone(), creg_, val_);
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

std::shared_ptr<Qop> Custom::clone() const {
  std::vector<std::shared_ptr<Qop>> qops(qops_.size());
  for (size_t i = 0; i < qops_.size(); ++i) {
    qops[i] = qops_[i]->clone();
  }
  return std::make_shared<Custom>(name_, qops, qargs_, params_);
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
std::shared_ptr<Qop> ID::clone() const {
  return std::make_shared<ID>(qargs_, params_);
}

X::X(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::X, qargs) {}
X::X(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::X, qargs, params) {}
size_t X::numQargs() const { return 1; }
size_t X::numParams() const { return 0; }
std::shared_ptr<Qop> X::clone() const {
  return std::make_shared<X>(qargs_, params_);
}

Y::Y(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::Y, qargs) {}
Y::Y(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::Y, qargs, params) {}
size_t Y::numQargs() const { return 1; }
size_t Y::numParams() const { return 0; }
std::shared_ptr<Qop> Y::clone() const {
  return std::make_shared<Y>(qargs_, params_);
}

Z::Z(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::Z, qargs) {}
Z::Z(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::Z, qargs, params) {}
size_t Z::numQargs() const { return 1; }
size_t Z::numParams() const { return 0; }
std::shared_ptr<Qop> Z::clone() const {
  return std::make_shared<Z>(qargs_, params_);
}

H::H(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::H, qargs) {}
H::H(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::H, qargs, params) {}
size_t H::numQargs() const { return 1; }
size_t H::numParams() const { return 0; }
std::shared_ptr<Qop> H::clone() const {
  return std::make_shared<H>(qargs_, params_);
}

S::S(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::S, qargs) {}
S::S(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::S, qargs, params) {}
size_t S::numQargs() const { return 1; }
size_t S::numParams() const { return 0; }
std::shared_ptr<Qop> S::clone() const {
  return std::make_shared<S>(qargs_, params_);
}

SDG::SDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SDG, qargs) {}
SDG::SDG(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SDG, qargs, params) {}
size_t SDG::numQargs() const { return 1; }
size_t SDG::numParams() const { return 0; }
std::shared_ptr<Qop> SDG::clone() const {
  return std::make_shared<SDG>(qargs_, params_);
}

T::T(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::T, qargs) {}
T::T(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::T, qargs, params) {}
size_t T::numQargs() const { return 1; }
size_t T::numParams() const { return 0; }
std::shared_ptr<Qop> T::clone() const {
  return std::make_shared<T>(qargs_, params_);
}

TDG::TDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::TDG, qargs) {}
TDG::TDG(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::TDG, qargs, params) {}
size_t TDG::numQargs() const { return 1; }
size_t TDG::numParams() const { return 0; }
std::shared_ptr<Qop> TDG::clone() const {
  return std::make_shared<TDG>(qargs_, params_);
}

SX::SX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SX, qargs) {}
SX::SX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SX, qargs, params) {}
size_t SX::numQargs() const { return 1; }
size_t SX::numParams() const { return 0; }
std::shared_ptr<Qop> SX::clone() const {
  return std::make_shared<SX>(qargs_, params_);
}

SXDG::SXDG(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SXDG, qargs) {}
SXDG::SXDG(std::vector<std::shared_ptr<Qarg>> qargs,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SXDG, qargs, params) {}
size_t SXDG::numQargs() const { return 1; }
size_t SXDG::numParams() const { return 0; }
std::shared_ptr<Qop> SXDG::clone() const {
  return std::make_shared<SXDG>(qargs_, params_);
}

P::P(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::P, qargs) {}
P::P(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::P, qargs, params) {}
size_t P::numQargs() const { return 1; }
size_t P::numParams() const { return 1; }
std::shared_ptr<Qop> P::clone() const {
  return std::make_shared<P>(qargs_, params_);
}

RX::RX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RX, qargs) {}
RX::RX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RX, qargs, params) {}
size_t RX::numQargs() const { return 1; }
size_t RX::numParams() const { return 1; }
std::shared_ptr<Qop> RX::clone() const {
  return std::make_shared<RX>(qargs_, params_);
}

RY::RY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RY, qargs) {}
RY::RY(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RY, qargs, params) {}
size_t RY::numQargs() const { return 1; }
size_t RY::numParams() const { return 1; }
std::shared_ptr<Qop> RY::clone() const {
  return std::make_shared<RY>(qargs_, params_);
}

RZ::RZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RZ, qargs) {}
RZ::RZ(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RZ, qargs, params) {}
size_t RZ::numQargs() const { return 1; }
size_t RZ::numParams() const { return 1; }
std::shared_ptr<Qop> RZ::clone() const {
  return std::make_shared<RZ>(qargs_, params_);
}

U0::U0(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U0, qargs) {}
U0::U0(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U0, qargs, params) {}
size_t U0::numQargs() const { return 1; }
size_t U0::numParams() const { return 1; }
std::shared_ptr<Qop> U0::clone() const {
  return std::make_shared<U0>(qargs_, params_);
}

U1::U1(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U1, qargs) {}
U1::U1(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U1, qargs, params) {}
size_t U1::numQargs() const { return 1; }
size_t U1::numParams() const { return 1; }
std::shared_ptr<Qop> U1::clone() const {
  return std::make_shared<U1>(qargs_, params_);
}

U2::U2(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U2, qargs) {}
U2::U2(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U2, qargs, params) {}
size_t U2::numQargs() const { return 1; }
size_t U2::numParams() const { return 2; }
std::shared_ptr<Qop> U2::clone() const {
  return std::make_shared<U2>(qargs_, params_);
}

U3::U3(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::U3, qargs) {}
U3::U3(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U3, qargs, params) {}
size_t U3::numQargs() const { return 1; }
size_t U3::numParams() const { return 3; }
std::shared_ptr<Qop> U3::clone() const {
  return std::make_shared<U3>(qargs_, params_);
}

U::U(std::vector<std::shared_ptr<Qarg>> qargs) : Qgate(QgateType::U, qargs) {}
U::U(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::U, qargs, params) {}
size_t U::numQargs() const { return 1; }
size_t U::numParams() const { return 3; }
std::shared_ptr<Qop> U::clone() const {
  return std::make_shared<U>(qargs_, params_);
}

CX::CX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CX, qargs) {}
CX::CX(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CX, qargs, params) {}
size_t CX::numQargs() const { return 2; }
size_t CX::numParams() const { return 0; }
std::shared_ptr<Qop> CX::clone() const {
  return std::make_shared<CX>(qargs_, params_);
}

CZ::CZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CZ, qargs) {}
CZ::CZ(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CZ, qargs, params) {}
size_t CZ::numQargs() const { return 2; }
size_t CZ::numParams() const { return 0; }
std::shared_ptr<Qop> CZ::clone() const {
  return std::make_shared<CZ>(qargs_, params_);
}

CY::CY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CY, qargs) {}
CY::CY(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CY, qargs, params) {}
size_t CY::numQargs() const { return 2; }
size_t CY::numParams() const { return 0; }
std::shared_ptr<Qop> CY::clone() const {
  return std::make_shared<CY>(qargs_, params_);
}

SWAP::SWAP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::SWAP, qargs) {}
SWAP::SWAP(std::vector<std::shared_ptr<Qarg>> qargs,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::SWAP, qargs, params) {}
size_t SWAP::numQargs() const { return 2; }
size_t SWAP::numParams() const { return 0; }
std::shared_ptr<Qop> SWAP::clone() const {
  return std::make_shared<SWAP>(qargs_, params_);
}

CH::CH(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CH, qargs) {}
CH::CH(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CH, qargs, params) {}
size_t CH::numQargs() const { return 2; }
size_t CH::numParams() const { return 0; }
std::shared_ptr<Qop> CH::clone() const {
  return std::make_shared<CH>(qargs_, params_);
}

CSX::CSX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CSX, qargs) {}
CSX::CSX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CSX, qargs, params) {}
size_t CSX::numQargs() const { return 2; }
size_t CSX::numParams() const { return 0; }
std::shared_ptr<Qop> CSX::clone() const {
  return std::make_shared<CSX>(qargs_, params_);
}

CRX::CRX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRX, qargs) {}
CRX::CRX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRX, qargs, params) {}
size_t CRX::numQargs() const { return 2; }
size_t CRX::numParams() const { return 1; }
std::shared_ptr<Qop> CRX::clone() const {
  return std::make_shared<CRX>(qargs_, params_);
}

CRY::CRY(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRY, qargs) {}
CRY::CRY(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRY, qargs, params) {}
size_t CRY::numQargs() const { return 2; }
size_t CRY::numParams() const { return 1; }
std::shared_ptr<Qop> CRY::clone() const {
  return std::make_shared<CRY>(qargs_, params_);
}

CRZ::CRZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CRZ, qargs) {}
CRZ::CRZ(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CRZ, qargs, params) {}
size_t CRZ::numQargs() const { return 2; }
size_t CRZ::numParams() const { return 1; }
std::shared_ptr<Qop> CRZ::clone() const {
  return std::make_shared<CRZ>(qargs_, params_);
}

CU1::CU1(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU1, qargs) {}
CU1::CU1(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU1, qargs, params) {}
size_t CU1::numQargs() const { return 2; }
size_t CU1::numParams() const { return 1; }
std::shared_ptr<Qop> CU1::clone() const {
  return std::make_shared<CU1>(qargs_, params_);
}

CP::CP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CP, qargs) {}
CP::CP(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CP, qargs, params) {}
size_t CP::numQargs() const { return 2; }
size_t CP::numParams() const { return 1; }
std::shared_ptr<Qop> CP::clone() const {
  return std::make_shared<CP>(qargs_, params_);
}

RXX::RXX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RXX, qargs) {}
RXX::RXX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RXX, qargs, params) {}
size_t RXX::numQargs() const { return 2; }
size_t RXX::numParams() const { return 1; }
std::shared_ptr<Qop> RXX::clone() const {
  return std::make_shared<RXX>(qargs_, params_);
}

RZZ::RZZ(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::RZZ, qargs) {}
RZZ::RZZ(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::RZZ, qargs, params) {}
size_t RZZ::numQargs() const { return 2; }
size_t RZZ::numParams() const { return 1; }
std::shared_ptr<Qop> RZZ::clone() const {
  return std::make_shared<RZZ>(qargs_, params_);
}

CU3::CU3(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU3, qargs) {}
CU3::CU3(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU3, qargs, params) {}
size_t CU3::numQargs() const { return 2; }
size_t CU3::numParams() const { return 3; }
std::shared_ptr<Qop> CU3::clone() const {
  return std::make_shared<CU3>(qargs_, params_);
}

CU::CU(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CU, qargs) {}
CU::CU(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CU, qargs, params) {}
size_t CU::numQargs() const { return 2; }
size_t CU::numParams() const { return 4; }
std::shared_ptr<Qop> CU::clone() const {
  return std::make_shared<CU>(qargs_, params_);
}

CCX::CCX(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CCX, qargs) {}
CCX::CCX(std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CCX, qargs, params) {}
size_t CCX::numQargs() const { return 3; }
size_t CCX::numParams() const { return 0; }
std::shared_ptr<Qop> CCX::clone() const {
  return std::make_shared<CCX>(qargs_, params_);
}

CSWAP::CSWAP(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qgate(QgateType::CSWAP, qargs) {}
CSWAP::CSWAP(std::vector<std::shared_ptr<Qarg>> qargs,
             std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(QgateType::CSWAP, qargs, params) {}
size_t CSWAP::numQargs() const { return 3; }
size_t CSWAP::numParams() const { return 0; }
std::shared_ptr<Qop> CSWAP::clone() const {
  return std::make_shared<CSWAP>(qargs_, params_);
}

InitZeroState::InitZeroState() : Qop(QopType::INIT_ZERO_STATE, {}) {}
std::string InitZeroState::__repr__() const { return "init-zero-state"; }
std::shared_ptr<Qop> InitZeroState::clone() const {
  return std::make_shared<InitZeroState>();
}

SetZero::SetZero() : Qop(QopType::SET_ZERO, {}) {}
std::string SetZero::__repr__() const { return "set-zero"; }
std::shared_ptr<Qop> SetZero::clone() const {
  return std::make_shared<SetZero>();
}

MemcpyH2D::MemcpyH2D() : Qop(QopType::MEMCPY_H2D), slice_(0) {}
MemcpyH2D::MemcpyH2D(std::map<Qarg, Qarg> qarg_map)
    : Qop(QopType::MEMCPY_H2D), qarg_map_(qarg_map), slice_(0) {}
MemcpyH2D::MemcpyH2D(std::map<Qarg, Qarg> qarg_map, size_t slice)
    : Qop(QopType::MEMCPY_H2D), qarg_map_(qarg_map), slice_(slice) {}
std::string MemcpyH2D::__repr__() const {
  std::ostringstream s;
  s << "memcpy-h2d ";
  s << "(" << slice_ << ") ";
  for (auto &kv : qarg_map_) {
    s << kv.first.__repr__() << " <-> " << kv.second.__repr__() << ", ";
  }
  return s.str();
}
std::shared_ptr<Qop> MemcpyH2D::clone() const {
  return std::make_shared<MemcpyH2D>(qarg_map_, slice_);
}

MemcpyD2H::MemcpyD2H() : Qop(QopType::MEMCPY_D2H), slice_(0) {}
MemcpyD2H::MemcpyD2H(std::map<Qarg, Qarg> qarg_map)
    : Qop(QopType::MEMCPY_D2H), qarg_map_(qarg_map), slice_(0) {}
MemcpyD2H::MemcpyD2H(std::map<Qarg, Qarg> qarg_map, size_t slice)
    : Qop(QopType::MEMCPY_D2H), qarg_map_(qarg_map), slice_(slice) {}
std::string MemcpyD2H::__repr__() const {
  std::ostringstream s;
  s << "memcpy-d2h ";
  s << "(" << slice_ << ") ";
  for (auto &kv : qarg_map_) {
    s << kv.first.__repr__() << " <-> " << kv.second.__repr__() << ", ";
  }
  return s.str();
}
std::shared_ptr<Qop> MemcpyD2H::clone() const {
  return std::make_shared<MemcpyD2H>(qarg_map_, slice_);
}

Sync::Sync() : Qop(QopType::SYNC) {}
std::string Sync::__repr__() const { return "sync "; }
std::shared_ptr<Qop> Sync::clone() const { return std::make_shared<Sync>(); }

GlobalSwap::GlobalSwap(std::vector<std::shared_ptr<Qarg>> qargs)
    : Qop(QopType::GLOBAL_SWAP, qargs) {}
std::string GlobalSwap::__repr__() const {
  std::ostringstream s;
  s << "global-swap ";
  s << qargs_[0]->__repr__() << " <-> " << qargs_[1]->__repr__();
  return s.str();
}

std::shared_ptr<Qop> GlobalSwap::clone() const {
  return std::make_shared<GlobalSwap>(qargs_);
}

Slice::Slice(size_t slice) : Qop(QopType::SLICE), slice_(slice) {}
std::string Slice::__repr__() const {
  return "slice " + std::to_string(slice_);
}

std::shared_ptr<Qop> Slice::clone() const {
  return std::make_shared<Slice>(slice_);
}

} // namespace snuqs
