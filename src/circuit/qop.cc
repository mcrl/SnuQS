#include "qop.h"
#include "assertion.h"

#include <sstream>

namespace snuqs {

//
// Qop
//
Qop::Qop() {}
Qop::Qop(std::vector<Qarg> qbits) : qbits_(qbits) {}
Qop::Qop(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : qbits_(qbits), params_(params) {}
Qop::~Qop() {}
std::string Qop::__repr__() const { return "qop"; }

Barrier::Barrier(std::vector<Qarg> qbits) : Qop(qbits) {}
std::string Barrier::__repr__() const {
  std::ostringstream s;
  s << "barrier ";
  for (auto &q : qbits_) {
    s << q.__repr__();
  }
  return s.str();
}

Reset::Reset(std::vector<Qarg> qbits) : Qop(qbits) {}
std::string Reset::__repr__() const {
  std::ostringstream s;
  s << "reset ";
  for (auto &q : qbits_) {
    s << q.__repr__();
  }
  return s.str();
}

Measure::Measure(std::vector<Qarg> qbits, std::vector<Carg> cbits)
    : Qop(qbits), cbits_(cbits) {}
std::string Measure::__repr__() const {
  std::ostringstream s;
  s << "Measure ";
  for (auto &q : qbits_) {
    s << q.__repr__();
  }
  s << " -> ";
  for (auto &c : cbits_) {
    s << c.__repr__();
  }

  return s.str();
}

Cond::Cond(std::shared_ptr<Qop> op, std::shared_ptr<Creg> creg, size_t val)
    : op_(op), creg_(creg), val_(val) {}
std::string Cond::__repr__() const {
  std::ostringstream s;
  s << "if (" << creg_->__repr__() << " == " << val_ << ") ";
  s << op_->__repr__();

  return s.str();
}

Custom::Custom(const std::string &name, std::vector<std::shared_ptr<Qop>> qops,
               std::vector<Qarg> qbits,
               std::vector<std::shared_ptr<Parameter>> params)
    : Qop(qbits, params), name_(name), qops_(qops) {}
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
  for (auto &q : qbits_) {
    s << q.__repr__() << ", ";
  }
  s << "\n";
  for (auto &op : qops_) {
    s << op->__repr__() << "\n";
  }

  return s.str();
}

Qgate::Qgate(std::vector<Qarg> qbits) : Qop(qbits) {}
Qgate::Qgate(std::vector<Qarg> qbits,
             std::vector<std::shared_ptr<Parameter>> params)
    : Qop(qbits, params) {}
size_t Qgate::numQubits() const {
  NOT_IMPLEMENTED();
  return 0;
}
size_t Qgate::numParams() const {
  NOT_IMPLEMENTED();
  return 0;
}
std::string Qgate::name() const {
  NOT_IMPLEMENTED();
  return "Qgate";
}
std::string Qgate::__repr__() const {
  std::ostringstream s;
  s << name();
  if (params_.size() > 0) {
    s << "(";
    for (auto &p : params_) {
      s << p->eval() << ", ";
    }
    s << ")";
  }

  s << " ";
  for (auto &q : qbits_) {
    s << q.__repr__() << ", ";
  }

  return s.str();
}

ID::ID(std::vector<Qarg> qbits) : Qgate(qbits) {}
ID::ID(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t ID::numQubits() const { return 1; }
size_t ID::numParams() const { return 0; }
std::string ID::name() const { return "ID"; }

X::X(std::vector<Qarg> qbits) : Qgate(qbits) {}
X::X(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t X::numQubits() const { return 1; }
size_t X::numParams() const { return 0; }
std::string X::name() const { return "X"; }

Y::Y(std::vector<Qarg> qbits) : Qgate(qbits) {}
Y::Y(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t Y::numQubits() const { return 1; }
size_t Y::numParams() const { return 0; }
std::string Y::name() const { return "Y"; }

Z::Z(std::vector<Qarg> qbits) : Qgate(qbits) {}
Z::Z(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t Z::numQubits() const { return 1; }
size_t Z::numParams() const { return 0; }
std::string Z::name() const { return "Z"; }

H::H(std::vector<Qarg> qbits) : Qgate(qbits) {}
H::H(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t H::numQubits() const { return 1; }
size_t H::numParams() const { return 0; }
std::string H::name() const { return "H"; }

S::S(std::vector<Qarg> qbits) : Qgate(qbits) {}
S::S(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t S::numQubits() const { return 1; }
size_t S::numParams() const { return 0; }
std::string S::name() const { return "S"; }

SDG::SDG(std::vector<Qarg> qbits) : Qgate(qbits) {}
SDG::SDG(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t SDG::numQubits() const { return 1; }
size_t SDG::numParams() const { return 0; }
std::string SDG::name() const { return "SDG"; }

T::T(std::vector<Qarg> qbits) : Qgate(qbits) {}
T::T(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t T::numQubits() const { return 1; }
size_t T::numParams() const { return 0; }
std::string T::name() const { return "T"; }

TDG::TDG(std::vector<Qarg> qbits) : Qgate(qbits) {}
TDG::TDG(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t TDG::numQubits() const { return 1; }
size_t TDG::numParams() const { return 0; }
std::string TDG::name() const { return "TDG"; }

SX::SX(std::vector<Qarg> qbits) : Qgate(qbits) {}
SX::SX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t SX::numQubits() const { return 1; }
size_t SX::numParams() const { return 0; }
std::string SX::name() const { return "SX"; }

SXDG::SXDG(std::vector<Qarg> qbits) : Qgate(qbits) {}
SXDG::SXDG(std::vector<Qarg> qbits,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t SXDG::numQubits() const { return 1; }
size_t SXDG::numParams() const { return 0; }
std::string SXDG::name() const { return "SXDG"; }

P::P(std::vector<Qarg> qbits) : Qgate(qbits) {}
P::P(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t P::numQubits() const { return 1; }
size_t P::numParams() const { return 1; }
std::string P::name() const { return "P"; }

RX::RX(std::vector<Qarg> qbits) : Qgate(qbits) {}
RX::RX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RX::numQubits() const { return 1; }
size_t RX::numParams() const { return 1; }
std::string RX::name() const { return "RX"; }

RY::RY(std::vector<Qarg> qbits) : Qgate(qbits) {}
RY::RY(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RY::numQubits() const { return 1; }
size_t RY::numParams() const { return 1; }
std::string RY::name() const { return "RY"; }

RZ::RZ(std::vector<Qarg> qbits) : Qgate(qbits) {}
RZ::RZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RZ::numQubits() const { return 1; }
size_t RZ::numParams() const { return 1; }
std::string RZ::name() const { return "RZ"; }

U0::U0(std::vector<Qarg> qbits) : Qgate(qbits) {}
U0::U0(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t U0::numQubits() const { return 1; }
size_t U0::numParams() const { return 1; }
std::string U0::name() const { return "U0"; }

U1::U1(std::vector<Qarg> qbits) : Qgate(qbits) {}
U1::U1(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t U1::numQubits() const { return 1; }
size_t U1::numParams() const { return 1; }
std::string U1::name() const { return "U1"; }

U2::U2(std::vector<Qarg> qbits) : Qgate(qbits) {}
U2::U2(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t U2::numQubits() const { return 1; }
size_t U2::numParams() const { return 2; }
std::string U2::name() const { return "U2"; }

U3::U3(std::vector<Qarg> qbits) : Qgate(qbits) {}
U3::U3(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t U3::numQubits() const { return 1; }
size_t U3::numParams() const { return 3; }
std::string U3::name() const { return "U3"; }

U::U(std::vector<Qarg> qbits) : Qgate(qbits) {}
U::U(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t U::numQubits() const { return 1; }
size_t U::numParams() const { return 3; }
std::string U::name() const { return "U"; }

CX::CX(std::vector<Qarg> qbits) : Qgate(qbits) {}
CX::CX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CX::numQubits() const { return 2; }
size_t CX::numParams() const { return 0; }
std::string CX::name() const { return "CX"; }

CZ::CZ(std::vector<Qarg> qbits) : Qgate(qbits) {}
CZ::CZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CZ::numQubits() const { return 2; }
size_t CZ::numParams() const { return 0; }
std::string CZ::name() const { return "CZ"; }

CY::CY(std::vector<Qarg> qbits) : Qgate(qbits) {}
CY::CY(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CY::numQubits() const { return 2; }
size_t CY::numParams() const { return 0; }
std::string CY::name() const { return "CY"; }

SWAP::SWAP(std::vector<Qarg> qbits) : Qgate(qbits) {}
SWAP::SWAP(std::vector<Qarg> qbits,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t SWAP::numQubits() const { return 2; }
size_t SWAP::numParams() const { return 0; }
std::string SWAP::name() const { return "SWAP"; }

CH::CH(std::vector<Qarg> qbits) : Qgate(qbits) {}
CH::CH(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CH::numQubits() const { return 2; }
size_t CH::numParams() const { return 0; }
std::string CH::name() const { return "CH"; }

CSX::CSX(std::vector<Qarg> qbits) : Qgate(qbits) {}
CSX::CSX(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CSX::numQubits() const { return 2; }
size_t CSX::numParams() const { return 0; }
std::string CSX::name() const { return "CSX"; }

CRX::CRX(std::vector<Qarg> qbits) : Qgate(qbits) {}
CRX::CRX(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CRX::numQubits() const { return 2; }
size_t CRX::numParams() const { return 1; }
std::string CRX::name() const { return "CRX"; }

CRY::CRY(std::vector<Qarg> qbits) : Qgate(qbits) {}
CRY::CRY(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CRY::numQubits() const { return 2; }
size_t CRY::numParams() const { return 1; }
std::string CRY::name() const { return "CRY"; }

CRZ::CRZ(std::vector<Qarg> qbits) : Qgate(qbits) {}
CRZ::CRZ(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CRZ::numQubits() const { return 2; }
size_t CRZ::numParams() const { return 1; }
std::string CRZ::name() const { return "CRZ"; }

CU1::CU1(std::vector<Qarg> qbits) : Qgate(qbits) {}
CU1::CU1(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CU1::numQubits() const { return 2; }
size_t CU1::numParams() const { return 1; }
std::string CU1::name() const { return "CU1"; }

CP::CP(std::vector<Qarg> qbits) : Qgate(qbits) {}
CP::CP(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CP::numQubits() const { return 2; }
size_t CP::numParams() const { return 1; }
std::string CP::name() const { return "CP"; }

RXX::RXX(std::vector<Qarg> qbits) : Qgate(qbits) {}
RXX::RXX(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RXX::numQubits() const { return 2; }
size_t RXX::numParams() const { return 1; }
std::string RXX::name() const { return "RXX"; }

RZZ::RZZ(std::vector<Qarg> qbits) : Qgate(qbits) {}
RZZ::RZZ(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RZZ::numQubits() const { return 2; }
size_t RZZ::numParams() const { return 1; }
std::string RZZ::name() const { return "RZZ"; }

CU3::CU3(std::vector<Qarg> qbits) : Qgate(qbits) {}
CU3::CU3(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CU3::numQubits() const { return 2; }
size_t CU3::numParams() const { return 3; }
std::string CU3::name() const { return "CU3"; }

CU::CU(std::vector<Qarg> qbits) : Qgate(qbits) {}
CU::CU(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CU::numQubits() const { return 2; }
size_t CU::numParams() const { return 4; }
std::string CU::name() const { return "CU"; }

CCX::CCX(std::vector<Qarg> qbits) : Qgate(qbits) {}
CCX::CCX(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CCX::numQubits() const { return 3; }
size_t CCX::numParams() const { return 0; }
std::string CCX::name() const { return "CCX"; }

CSWAP::CSWAP(std::vector<Qarg> qbits) : Qgate(qbits) {}
CSWAP::CSWAP(std::vector<Qarg> qbits,
             std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t CSWAP::numQubits() const { return 3; }
size_t CSWAP::numParams() const { return 0; }
std::string CSWAP::name() const { return "CSWAP"; }

RCCX::RCCX(std::vector<Qarg> qbits) : Qgate(qbits) {}
RCCX::RCCX(std::vector<Qarg> qbits,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RCCX::numQubits() const { return 3; }
size_t RCCX::numParams() const { return 0; }
std::string RCCX::name() const { return "RCCX"; }

RC3X::RC3X(std::vector<Qarg> qbits) : Qgate(qbits) {}
RC3X::RC3X(std::vector<Qarg> qbits,
           std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t RC3X::numQubits() const { return 4; }
size_t RC3X::numParams() const { return 0; }
std::string RC3X::name() const { return "RC3X"; }

C3X::C3X(std::vector<Qarg> qbits) : Qgate(qbits) {}
C3X::C3X(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t C3X::numQubits() const { return 4; }
size_t C3X::numParams() const { return 0; }
std::string C3X::name() const { return "C3X"; }

C3SQRTX::C3SQRTX(std::vector<Qarg> qbits) : Qgate(qbits) {}
C3SQRTX::C3SQRTX(std::vector<Qarg> qbits,
                 std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t C3SQRTX::numQubits() const { return 4; }
size_t C3SQRTX::numParams() const { return 0; }
std::string C3SQRTX::name() const { return "C3SQRTX"; }

C4X::C4X(std::vector<Qarg> qbits) : Qgate(qbits) {}
C4X::C4X(std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params)
    : Qgate(qbits, params) {}
size_t C4X::numQubits() const { return 5; }
size_t C4X::numParams() const { return 0; }
std::string C4X::name() const { return "C4X"; }

} // namespace snuqs
