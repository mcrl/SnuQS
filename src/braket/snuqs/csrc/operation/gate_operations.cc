#include "operation/gate_operations.h"

#include <spdlog/spdlog.h>

using namespace std::complex_literals;

Identity::Identity(std::vector<size_t> targets,
                   std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1;
}
Identity::~Identity() {}
std::string Identity::name() const { return "Identity"; }

Hadamard::Hadamard(std::vector<size_t> targets,
                   std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = M_SQRT1_2;
  data_[0 * 2 + 1] = M_SQRT1_2;
  data_[1 * 2 + 0] = M_SQRT1_2;
  data_[1 * 2 + 1] = -M_SQRT1_2;
}
Hadamard::~Hadamard() {}
std::string Hadamard::name() const { return "Hadamard"; }

PauliX::PauliX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = 1;
  data_[1 * 2 + 0] = 1;
  data_[1 * 2 + 1] = 0;
}
PauliX::~PauliX() {}
std::string PauliX::name() const { return "PauliX"; }

PauliY::PauliY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = -1i;
  data_[1 * 2 + 0] = 1i;
  data_[1 * 2 + 1] = 0;
}
PauliY::~PauliY() {}
std::string PauliY::name() const { return "PauliY"; }

PauliZ::PauliZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1;
}
PauliZ::~PauliZ() {}
std::string PauliZ::name() const { return "PauliZ"; }

CX::CX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 0;
  data_[2 * 4 + 3] = 1;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 1;
  data_[3 * 4 + 3] = 0;
}
CX::~CX() {}
std::string CX::name() const { return "CX"; }

CY::CY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 0;
  data_[2 * 4 + 3] = -1i;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 1i;
  data_[3 * 4 + 3] = 0;
}
CY::~CY() {}
std::string CY::name() const { return "CY"; }

CZ::CZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 1;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = -1;
}
CZ::~CZ() {}
std::string CZ::name() const { return "CZ"; }

S::S(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1i;
}
S::~S() {}
std::string S::name() const { return "S"; }

Si::Si(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1i;
}
Si::~Si() {}
std::string Si::name() const { return "Si"; }

T::T(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * M_PI / 4.);
}
T::~T() {}
std::string T::name() const { return "T"; }

Ti::Ti(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(-1i * M_PI / 4.);
}
Ti::~Ti() {}
std::string Ti::name() const { return "Ti"; }

V::V(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0.5 + 0.5i;
  data_[0 * 2 + 1] = 0.5 - 0.5i;
  data_[1 * 2 + 0] = 0.5 - 0.5i;
  data_[1 * 2 + 1] = 0.5 + 0.5i;
}
V::~V() {}
std::string V::name() const { return "V"; }

Vi::Vi(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0.5 - 0.5i;
  data_[0 * 2 + 1] = 0.5 + 0.5i;
  data_[1 * 2 + 0] = 0.5 + 0.5i;
  data_[1 * 2 + 1] = 0.5 - 0.5i;
}
Vi::~Vi() {}
std::string Vi::name() const { return "Vi"; }

PhaseShift::PhaseShift(std::vector<size_t> targets, double angle,
                       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * angle);
}
PhaseShift::~PhaseShift() {}
std::string PhaseShift::name() const {
  return "PhaseShift(" + std::to_string(angle_) + ")";
}

CPhaseShift::CPhaseShift(std::vector<size_t> targets, double angle,
                         std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 1;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = std::exp(1i * angle);
}
CPhaseShift::~CPhaseShift() {}
std::string CPhaseShift::name() const {
  return "CPhaseShift(" + std::to_string(angle_) + ")";
}

CPhaseShift00::CPhaseShift00(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 4 + 0] = std::exp(1i * angle);
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 1;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
CPhaseShift00::~CPhaseShift00() {}
std::string CPhaseShift00::name() const {
  return "CPhaseShift00(" + std::to_string(angle_) + ")";
}

CPhaseShift01::CPhaseShift01(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = std::exp(1i * angle);
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = 1;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
CPhaseShift01::~CPhaseShift01() {}
std::string CPhaseShift01::name() const {
  return "CPhaseShift01(" + std::to_string(angle_) + ")";
}

CPhaseShift10::CPhaseShift10(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 1;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = std::exp(1i * angle);
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
CPhaseShift10::~CPhaseShift10() {}
std::string CPhaseShift10::name() const {
  return "CPhaseShift10(" + std::to_string(angle_) + ")";
}

RotX::RotX(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> i_sin_hanlf_angle = 1i * sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -i_sin_hanlf_angle;
  data_[1 * 2 + 0] = -i_sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotX::~RotX() {}
std::string RotX::name() const {
  return "RotX(" + std::to_string(angle_) + ")";
}

RotY::RotY(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> sin_hanlf_angle = sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -sin_hanlf_angle;
  data_[1 * 2 + 0] = sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotY::~RotY() {}
std::string RotY::name() const {
  return "RotY(" + std::to_string(angle_) + ")";
}

RotZ::RotZ(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  data_[0 * 2 + 0] = negative_phase;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = positive_phase;
}
RotZ::~RotZ() {}
std::string RotZ::name() const {
  return "RotZ(" + std::to_string(angle_) + ")";
}

Swap::Swap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 0;
  data_[1 * 4 + 2] = 1;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 1;
  data_[2 * 4 + 2] = 0;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
Swap::~Swap() {}
std::string Swap::name() const { return "Swap"; }

ISwap::ISwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 0;
  data_[1 * 4 + 2] = 1i;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 1i;
  data_[2 * 4 + 2] = 0;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
ISwap::~ISwap() {}
std::string ISwap::name() const { return "ISwap"; }

PSwap::PSwap(std::vector<size_t> targets, double angle,
             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = 0;
  data_[1 * 4 + 2] = std::exp(1i * angle);
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = std::exp(1i * angle);
  data_[2 * 4 + 2] = 0;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
PSwap::~PSwap() {}
std::string PSwap::name() const {
  return "PSwap(" + std::to_string(angle_) + ")";
}

XY::XY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  data_[0 * 4 + 0] = 1;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = cos_angle;
  data_[1 * 4 + 2] = i_sin_angle;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = i_sin_angle;
  data_[2 * 4 + 2] = cos_angle;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = 1;
}
XY::~XY() {}
std::string XY::name() const { return "XY(" + std::to_string(angle_) + ")"; }

XX::XX(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  data_[0 * 4 + 0] = cos_angle;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = -i_sin_angle;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = cos_angle;
  data_[1 * 4 + 2] = -i_sin_angle;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = -i_sin_angle;
  data_[2 * 4 + 2] = cos_angle;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = -i_sin_angle;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = cos_angle;
}
XX::~XX() {}
std::string XX::name() const { return "XX(" + std::to_string(angle_) + ")"; }

YY::YY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  data_[0 * 4 + 0] = cos_angle;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = i_sin_angle;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = cos_angle;
  data_[1 * 4 + 2] = -i_sin_angle;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = -i_sin_angle;
  data_[2 * 4 + 2] = cos_angle;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = i_sin_angle;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = cos_angle;
}
YY::~YY() {}
std::string YY::name() const { return "YY(" + std::to_string(angle_) + ")"; }

ZZ::ZZ(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  data_[0 * 4 + 0] = negative_phase;
  data_[0 * 4 + 1] = 0;
  data_[0 * 4 + 2] = 0;
  data_[0 * 4 + 3] = 0;
  data_[1 * 4 + 0] = 0;
  data_[1 * 4 + 1] = positive_phase;
  data_[1 * 4 + 2] = 0;
  data_[1 * 4 + 3] = 0;
  data_[2 * 4 + 0] = 0;
  data_[2 * 4 + 1] = 0;
  data_[2 * 4 + 2] = positive_phase;
  data_[2 * 4 + 3] = 0;
  data_[3 * 4 + 0] = 0;
  data_[3 * 4 + 1] = 0;
  data_[3 * 4 + 2] = 0;
  data_[3 * 4 + 3] = negative_phase;
}
ZZ::~ZZ() {}
std::string ZZ::name() const { return "ZZ(" + std::to_string(angle_) + ")"; }

CCNot::CCNot(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 8 + 0] = 1;
  data_[0 * 8 + 1] = 0;
  data_[0 * 8 + 2] = 0;
  data_[0 * 8 + 3] = 0;
  data_[0 * 8 + 4] = 0;
  data_[0 * 8 + 5] = 0;
  data_[0 * 8 + 6] = 0;
  data_[0 * 8 + 7] = 0;
  data_[1 * 8 + 0] = 0;
  data_[1 * 8 + 1] = 1;
  data_[1 * 8 + 2] = 0;
  data_[1 * 8 + 3] = 0;
  data_[1 * 8 + 4] = 0;
  data_[1 * 8 + 5] = 0;
  data_[1 * 8 + 6] = 0;
  data_[1 * 8 + 7] = 0;
  data_[2 * 8 + 0] = 0;
  data_[2 * 8 + 1] = 0;
  data_[2 * 8 + 2] = 1;
  data_[2 * 8 + 3] = 0;
  data_[2 * 8 + 4] = 0;
  data_[2 * 8 + 5] = 0;
  data_[2 * 8 + 6] = 0;
  data_[2 * 8 + 7] = 0;
  data_[3 * 8 + 0] = 0;
  data_[3 * 8 + 1] = 0;
  data_[3 * 8 + 2] = 0;
  data_[3 * 8 + 3] = 1;
  data_[3 * 8 + 4] = 0;
  data_[3 * 8 + 5] = 0;
  data_[3 * 8 + 6] = 0;
  data_[3 * 8 + 7] = 0;
  data_[4 * 8 + 0] = 0;
  data_[4 * 8 + 1] = 0;
  data_[4 * 8 + 2] = 0;
  data_[4 * 8 + 3] = 0;
  data_[4 * 8 + 4] = 1;
  data_[4 * 8 + 5] = 0;
  data_[4 * 8 + 6] = 0;
  data_[4 * 8 + 7] = 0;
  data_[5 * 8 + 0] = 0;
  data_[5 * 8 + 1] = 0;
  data_[5 * 8 + 2] = 0;
  data_[5 * 8 + 3] = 0;
  data_[5 * 8 + 4] = 0;
  data_[5 * 8 + 5] = 1;
  data_[5 * 8 + 6] = 0;
  data_[5 * 8 + 7] = 0;
  data_[6 * 8 + 0] = 0;
  data_[6 * 8 + 1] = 0;
  data_[6 * 8 + 2] = 0;
  data_[6 * 8 + 3] = 0;
  data_[6 * 8 + 4] = 0;
  data_[6 * 8 + 5] = 0;
  data_[6 * 8 + 6] = 0;
  data_[6 * 8 + 7] = 1;
  data_[7 * 8 + 0] = 0;
  data_[7 * 8 + 1] = 0;
  data_[7 * 8 + 2] = 0;
  data_[7 * 8 + 3] = 0;
  data_[7 * 8 + 4] = 0;
  data_[7 * 8 + 5] = 0;
  data_[7 * 8 + 6] = 1;
  data_[7 * 8 + 7] = 0;
}
CCNot::~CCNot() {}
std::string CCNot::name() const { return "CCNot"; }

CSwap::CSwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 8 + 0] = 1;
  data_[0 * 8 + 1] = 0;
  data_[0 * 8 + 2] = 0;
  data_[0 * 8 + 3] = 0;
  data_[0 * 8 + 4] = 0;
  data_[0 * 8 + 5] = 0;
  data_[0 * 8 + 6] = 0;
  data_[0 * 8 + 7] = 0;
  data_[1 * 8 + 0] = 0;
  data_[1 * 8 + 1] = 1;
  data_[1 * 8 + 2] = 0;
  data_[1 * 8 + 3] = 0;
  data_[1 * 8 + 4] = 0;
  data_[1 * 8 + 5] = 0;
  data_[1 * 8 + 6] = 0;
  data_[1 * 8 + 7] = 0;
  data_[2 * 8 + 0] = 0;
  data_[2 * 8 + 1] = 0;
  data_[2 * 8 + 2] = 1;
  data_[2 * 8 + 3] = 0;
  data_[2 * 8 + 4] = 0;
  data_[2 * 8 + 5] = 0;
  data_[2 * 8 + 6] = 0;
  data_[2 * 8 + 7] = 0;
  data_[3 * 8 + 0] = 0;
  data_[3 * 8 + 1] = 0;
  data_[3 * 8 + 2] = 0;
  data_[3 * 8 + 3] = 1;
  data_[3 * 8 + 4] = 0;
  data_[3 * 8 + 5] = 0;
  data_[3 * 8 + 6] = 0;
  data_[3 * 8 + 7] = 0;
  data_[4 * 8 + 0] = 0;
  data_[4 * 8 + 1] = 0;
  data_[4 * 8 + 2] = 0;
  data_[4 * 8 + 3] = 0;
  data_[4 * 8 + 4] = 1;
  data_[4 * 8 + 5] = 0;
  data_[4 * 8 + 6] = 0;
  data_[4 * 8 + 7] = 0;
  data_[5 * 8 + 0] = 0;
  data_[5 * 8 + 1] = 0;
  data_[5 * 8 + 2] = 0;
  data_[5 * 8 + 3] = 0;
  data_[5 * 8 + 4] = 0;
  data_[5 * 8 + 5] = 0;
  data_[5 * 8 + 6] = 1;
  data_[5 * 8 + 7] = 0;
  data_[6 * 8 + 0] = 0;
  data_[6 * 8 + 1] = 0;
  data_[6 * 8 + 2] = 0;
  data_[6 * 8 + 3] = 0;
  data_[6 * 8 + 4] = 0;
  data_[6 * 8 + 5] = 1;
  data_[6 * 8 + 6] = 0;
  data_[6 * 8 + 7] = 0;
  data_[7 * 8 + 0] = 0;
  data_[7 * 8 + 1] = 0;
  data_[7 * 8 + 2] = 0;
  data_[7 * 8 + 3] = 0;
  data_[7 * 8 + 4] = 0;
  data_[7 * 8 + 5] = 0;
  data_[7 * 8 + 6] = 0;
  data_[7 * 8 + 7] = 1;
}
CSwap::~CSwap() {}
std::string CSwap::name() const { return "CSwap"; }

U::U(std::vector<size_t> targets, double theta, double phi, double lambda,
     std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power),
      theta_(theta),
      phi_(phi),
      lambda_(lambda) {
  data_[0 * 2 + 0] = cos(theta / 2);
  data_[0 * 2 + 1] = -std::exp(1i * lambda) * sin(theta / 2);
  data_[1 * 2 + 0] = std::exp(1i * phi) * sin(theta / 2);
  data_[1 * 2 + 1] = std::exp(1i * (phi + lambda)) * cos(theta / 2);
}
U::~U() {}
std::string U::name() const { return "U"; }

GPhase::GPhase(std::vector<size_t> targets, double angle,
               std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0] = std::exp(1i * angle);
}
GPhase::~GPhase() {}
std::string GPhase::name() const {
  return "GPhase(" + std::to_string(angle_) + ")";
}
