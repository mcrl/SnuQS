#include "gate_operation.h"

#include <spdlog/spdlog.h>

using namespace std::complex_literals;

Identity::Identity(const std::vector<size_t> &targets,
                   const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1;
}
Identity::~Identity() {}

Hadamard::Hadamard(const std::vector<size_t> &targets,
                   const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = M_SQRT1_2;
  data_[0 * 2 + 1] = M_SQRT1_2;
  data_[1 * 2 + 0] = M_SQRT1_2;
  data_[1 * 2 + 1] = -M_SQRT1_2;
}
Hadamard::~Hadamard() {}

PauliX::PauliX(const std::vector<size_t> &targets,
               const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = 1;
  data_[1 * 2 + 0] = 1;
  data_[1 * 2 + 1] = 0;
}
PauliX::~PauliX() {}

PauliY::PauliY(const std::vector<size_t> &targets,
               const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = -1i;
  data_[1 * 2 + 0] = 1i;
  data_[1 * 2 + 1] = 0;
}
PauliY::~PauliY() {}

PauliZ::PauliZ(const std::vector<size_t> &targets,
               const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1;
}
PauliZ::~PauliZ() {}

CX::CX(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

CY::CY(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

CZ::CZ(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

S::S(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1i;
}
S::~S() {}

Si::Si(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1i;
}
Si::~Si() {}

T::T(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * M_PI / 4.);
}
T::~T() {}

Ti::Ti(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(-1i * M_PI / 4.);
}
Ti::~Ti() {}

V::V(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0.5 + 0.5i;
  data_[0 * 2 + 1] = 0.5 - 0.5i;
  data_[1 * 2 + 0] = 0.5 - 0.5i;
  data_[1 * 2 + 1] = 0.5 + 0.5i;
}
V::~V() {}

Vi::Vi(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power) {
  data_[0 * 2 + 0] = 0.5 - 0.5i;
  data_[0 * 2 + 1] = 0.5 + 0.5i;
  data_[1 * 2 + 0] = 0.5 + 0.5i;
  data_[1 * 2 + 1] = 0.5 - 0.5i;
}
Vi::~Vi() {}

PhaseShift::PhaseShift(const std::vector<size_t> &targets, double angle,
                       const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * angle);
}
PhaseShift::~PhaseShift() {}

CPhaseShift::CPhaseShift(const std::vector<size_t> &targets, double angle,
                         const std::vector<size_t> &ctrl_modifiers,
                         size_t power)
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

CPhaseShift00::CPhaseShift00(const std::vector<size_t> &targets, double angle,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
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

CPhaseShift01::CPhaseShift01(const std::vector<size_t> &targets, double angle,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
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

CPhaseShift10::CPhaseShift10(const std::vector<size_t> &targets, double angle,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
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

RotX::RotX(const std::vector<size_t> &targets, double angle,
           const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> i_sin_hanlf_angle = 1i * sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -i_sin_hanlf_angle;
  data_[1 * 2 + 0] = -i_sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotX::~RotX() {}

RotY::RotY(const std::vector<size_t> &targets, double angle,
           const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> sin_hanlf_angle = sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -sin_hanlf_angle;
  data_[1 * 2 + 0] = sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotY::~RotY() {}

RotZ::RotZ(const std::vector<size_t> &targets, double angle,
           const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(targets, ctrl_modifiers, power), angle_(angle) {
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  data_[0 * 2 + 0] = negative_phase;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = positive_phase;
}
RotZ::~RotZ() {}

Swap::Swap(const std::vector<size_t> &targets,
           const std::vector<size_t> &ctrl_modifiers, size_t power)
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

ISwap::ISwap(const std::vector<size_t> &targets,
             const std::vector<size_t> &ctrl_modifiers, size_t power)
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

PSwap::PSwap(const std::vector<size_t> &targets, double angle,
             const std::vector<size_t> &ctrl_modifiers, size_t power)
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

XY::XY(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

XX::XX(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

YY::YY(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

ZZ::ZZ(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power)
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

CCNot::CCNot(const std::vector<size_t> &targets,
             const std::vector<size_t> &ctrl_modifiers, size_t power)
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

CSwap::CSwap(const std::vector<size_t> &targets,
             const std::vector<size_t> &ctrl_modifiers, size_t power)
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

U::U(const std::vector<size_t> &targets, double theta, double phi,
     double lambda, const std::vector<size_t> &ctrl_modifiers, size_t power)
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

GPhase::GPhase(const std::vector<size_t> &targets, double angle,
               const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation((targets.size() == 0) ? std::vector<size_t>{0} : targets,
                    ctrl_modifiers, power),
      angle_(angle) {
  data_[0 * 2 + 0] = std::exp(1i * angle);
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * angle);
}
GPhase::~GPhase() {}
