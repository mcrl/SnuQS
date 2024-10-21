#include "operation/gate_operations.h"

#include <spdlog/spdlog.h>

#include <complex>

using namespace std::complex_literals;

Identity::Identity(std::vector<size_t> targets,
                   std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::Identity, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = 1;
}
Identity::~Identity() {}

Hadamard::Hadamard(std::vector<size_t> targets,
                   std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::Hadamard, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = M_SQRT1_2;
  ptr[0 * 2 + 1] = M_SQRT1_2;
  ptr[1 * 2 + 0] = M_SQRT1_2;
  ptr[1 * 2 + 1] = -M_SQRT1_2;
}
Hadamard::~Hadamard() {}

PauliX::PauliX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(GateOperationType::PauliX, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 0;
  ptr[0 * 2 + 1] = 1;
  ptr[1 * 2 + 0] = 1;
  ptr[1 * 2 + 1] = 0;
}
PauliX::~PauliX() {}

PauliY::PauliY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(GateOperationType::PauliY, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 0;
  ptr[0 * 2 + 1] = -1i;
  ptr[1 * 2 + 0] = 1i;
  ptr[1 * 2 + 1] = 0;
}
PauliY::~PauliY() {}

PauliZ::PauliZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
               size_t power)
    : GateOperation(GateOperationType::PauliZ, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = -1;
}
PauliZ::~PauliZ() {}

CX::CX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::CX, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = 1;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 1;
  ptr[3 * 4 + 3] = 0;
}
CX::~CX() {}

CY::CY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::CY, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = -1i;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 1i;
  ptr[3 * 4 + 3] = 0;
}
CY::~CY() {}

CZ::CZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::CZ, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 1;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = -1;
}
CZ::~CZ() {}

S::S(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(GateOperationType::S, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = 1i;
}
S::~S() {}

Si::Si(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::Si, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = -1i;
}
Si::~Si() {}

T::T(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(GateOperationType::T, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = std::exp(1i * M_PI / 4.);
}
T::~T() {}

Ti::Ti(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::Ti, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = std::exp(-1i * M_PI / 4.);
}
Ti::~Ti() {}

V::V(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power)
    : GateOperation(GateOperationType::V, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 0.5 + 0.5i;
  ptr[0 * 2 + 1] = 0.5 - 0.5i;
  ptr[1 * 2 + 0] = 0.5 - 0.5i;
  ptr[1 * 2 + 1] = 0.5 + 0.5i;
}
V::~V() {}

Vi::Vi(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power)
    : GateOperation(GateOperationType::Vi, targets, {}, ctrl_modifiers, power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 0.5 - 0.5i;
  ptr[0 * 2 + 1] = 0.5 + 0.5i;
  ptr[1 * 2 + 0] = 0.5 + 0.5i;
  ptr[1 * 2 + 1] = 0.5 - 0.5i;
}
Vi::~Vi() {}

PhaseShift::PhaseShift(std::vector<size_t> targets, double angle,
                       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::PhaseShift, targets, {angle},
                    ctrl_modifiers, power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = 1;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = std::exp(1i * angle);
}
PhaseShift::~PhaseShift() {}

CPhaseShift::CPhaseShift(std::vector<size_t> targets, double angle,
                         std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::CPhaseShift, targets, {angle},
                    ctrl_modifiers, power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 1;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = std::exp(1i * angle);
}
CPhaseShift::~CPhaseShift() {}

CPhaseShift00::CPhaseShift00(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::CPhaseShift00, targets, {angle},
                    ctrl_modifiers, power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = std::exp(1i * angle);
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 1;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
CPhaseShift00::~CPhaseShift00() {}

CPhaseShift01::CPhaseShift01(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::CPhaseShift01, targets, {angle},
                    ctrl_modifiers, power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = std::exp(1i * angle);
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = 1;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
CPhaseShift01::~CPhaseShift01() {}

CPhaseShift10::CPhaseShift10(std::vector<size_t> targets, double angle,
                             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::CPhaseShift10, targets, {angle},
                    ctrl_modifiers, power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 1;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = std::exp(1i * angle);
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
CPhaseShift10::~CPhaseShift10() {}

RotX::RotX(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::RotX, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> i_sin_hanlf_angle = 1i * sin(angle_ / 2);
  ptr[0 * 2 + 0] = cos_half_angle;
  ptr[0 * 2 + 1] = -i_sin_hanlf_angle;
  ptr[1 * 2 + 0] = -i_sin_hanlf_angle;
  ptr[1 * 2 + 1] = cos_half_angle;
}
RotX::~RotX() {}

RotY::RotY(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::RotY, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> sin_hanlf_angle = sin(angle_ / 2);
  ptr[0 * 2 + 0] = cos_half_angle;
  ptr[0 * 2 + 1] = -sin_hanlf_angle;
  ptr[1 * 2 + 0] = sin_hanlf_angle;
  ptr[1 * 2 + 1] = cos_half_angle;
}
RotY::~RotY() {}

RotZ::RotZ(std::vector<size_t> targets, double angle,
           std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::RotZ, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  ptr[0 * 2 + 0] = negative_phase;
  ptr[0 * 2 + 1] = 0;
  ptr[1 * 2 + 0] = 0;
  ptr[1 * 2 + 1] = positive_phase;
}
RotZ::~RotZ() {}

Swap::Swap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power)
    : GateOperation(GateOperationType::Swap, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 0;
  ptr[1 * 4 + 2] = 1;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 1;
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
Swap::~Swap() {}

ISwap::ISwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(GateOperationType::ISwap, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 0;
  ptr[1 * 4 + 2] = 1i;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 1i;
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
ISwap::~ISwap() {}

PSwap::PSwap(std::vector<size_t> targets, double angle,
             std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::PSwap, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 0;
  ptr[1 * 4 + 2] = std::exp(1i * angle);
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = std::exp(1i * angle);
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
PSwap::~PSwap() {}

XY::XY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::XY, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = cos_angle;
  ptr[1 * 4 + 2] = i_sin_angle;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = i_sin_angle;
  ptr[2 * 4 + 2] = cos_angle;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
XY::~XY() {}

XX::XX(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::XX, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  ptr[0 * 4 + 0] = cos_angle;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = -i_sin_angle;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = cos_angle;
  ptr[1 * 4 + 2] = -i_sin_angle;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = -i_sin_angle;
  ptr[2 * 4 + 2] = cos_angle;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = -i_sin_angle;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = cos_angle;
}
XX::~XX() {}

YY::YY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::YY, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  double cos_angle = cos(angle / 2);
  std::complex<double> i_sin_angle = 1i * sin(angle / 2);
  ptr[0 * 4 + 0] = cos_angle;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = i_sin_angle;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = cos_angle;
  ptr[1 * 4 + 2] = -i_sin_angle;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = -i_sin_angle;
  ptr[2 * 4 + 2] = cos_angle;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = i_sin_angle;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = cos_angle;
}
YY::~YY() {}

ZZ::ZZ(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::ZZ, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  ptr[0 * 4 + 0] = negative_phase;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = positive_phase;
  ptr[1 * 4 + 2] = 0;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 0;
  ptr[2 * 4 + 2] = positive_phase;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = negative_phase;
}
ZZ::~ZZ() {}

CCNot::CCNot(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(GateOperationType::CCNot, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 8 + 0] = 1;
  ptr[0 * 8 + 1] = 0;
  ptr[0 * 8 + 2] = 0;
  ptr[0 * 8 + 3] = 0;
  ptr[0 * 8 + 4] = 0;
  ptr[0 * 8 + 5] = 0;
  ptr[0 * 8 + 6] = 0;
  ptr[0 * 8 + 7] = 0;
  ptr[1 * 8 + 0] = 0;
  ptr[1 * 8 + 1] = 1;
  ptr[1 * 8 + 2] = 0;
  ptr[1 * 8 + 3] = 0;
  ptr[1 * 8 + 4] = 0;
  ptr[1 * 8 + 5] = 0;
  ptr[1 * 8 + 6] = 0;
  ptr[1 * 8 + 7] = 0;
  ptr[2 * 8 + 0] = 0;
  ptr[2 * 8 + 1] = 0;
  ptr[2 * 8 + 2] = 1;
  ptr[2 * 8 + 3] = 0;
  ptr[2 * 8 + 4] = 0;
  ptr[2 * 8 + 5] = 0;
  ptr[2 * 8 + 6] = 0;
  ptr[2 * 8 + 7] = 0;
  ptr[3 * 8 + 0] = 0;
  ptr[3 * 8 + 1] = 0;
  ptr[3 * 8 + 2] = 0;
  ptr[3 * 8 + 3] = 1;
  ptr[3 * 8 + 4] = 0;
  ptr[3 * 8 + 5] = 0;
  ptr[3 * 8 + 6] = 0;
  ptr[3 * 8 + 7] = 0;
  ptr[4 * 8 + 0] = 0;
  ptr[4 * 8 + 1] = 0;
  ptr[4 * 8 + 2] = 0;
  ptr[4 * 8 + 3] = 0;
  ptr[4 * 8 + 4] = 1;
  ptr[4 * 8 + 5] = 0;
  ptr[4 * 8 + 6] = 0;
  ptr[4 * 8 + 7] = 0;
  ptr[5 * 8 + 0] = 0;
  ptr[5 * 8 + 1] = 0;
  ptr[5 * 8 + 2] = 0;
  ptr[5 * 8 + 3] = 0;
  ptr[5 * 8 + 4] = 0;
  ptr[5 * 8 + 5] = 1;
  ptr[5 * 8 + 6] = 0;
  ptr[5 * 8 + 7] = 0;
  ptr[6 * 8 + 0] = 0;
  ptr[6 * 8 + 1] = 0;
  ptr[6 * 8 + 2] = 0;
  ptr[6 * 8 + 3] = 0;
  ptr[6 * 8 + 4] = 0;
  ptr[6 * 8 + 5] = 0;
  ptr[6 * 8 + 6] = 0;
  ptr[6 * 8 + 7] = 1;
  ptr[7 * 8 + 0] = 0;
  ptr[7 * 8 + 1] = 0;
  ptr[7 * 8 + 2] = 0;
  ptr[7 * 8 + 3] = 0;
  ptr[7 * 8 + 4] = 0;
  ptr[7 * 8 + 5] = 0;
  ptr[7 * 8 + 6] = 1;
  ptr[7 * 8 + 7] = 0;
}
CCNot::~CCNot() {}

CSwap::CSwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
             size_t power)
    : GateOperation(GateOperationType::CSwap, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 8 + 0] = 1;
  ptr[0 * 8 + 1] = 0;
  ptr[0 * 8 + 2] = 0;
  ptr[0 * 8 + 3] = 0;
  ptr[0 * 8 + 4] = 0;
  ptr[0 * 8 + 5] = 0;
  ptr[0 * 8 + 6] = 0;
  ptr[0 * 8 + 7] = 0;
  ptr[1 * 8 + 0] = 0;
  ptr[1 * 8 + 1] = 1;
  ptr[1 * 8 + 2] = 0;
  ptr[1 * 8 + 3] = 0;
  ptr[1 * 8 + 4] = 0;
  ptr[1 * 8 + 5] = 0;
  ptr[1 * 8 + 6] = 0;
  ptr[1 * 8 + 7] = 0;
  ptr[2 * 8 + 0] = 0;
  ptr[2 * 8 + 1] = 0;
  ptr[2 * 8 + 2] = 1;
  ptr[2 * 8 + 3] = 0;
  ptr[2 * 8 + 4] = 0;
  ptr[2 * 8 + 5] = 0;
  ptr[2 * 8 + 6] = 0;
  ptr[2 * 8 + 7] = 0;
  ptr[3 * 8 + 0] = 0;
  ptr[3 * 8 + 1] = 0;
  ptr[3 * 8 + 2] = 0;
  ptr[3 * 8 + 3] = 1;
  ptr[3 * 8 + 4] = 0;
  ptr[3 * 8 + 5] = 0;
  ptr[3 * 8 + 6] = 0;
  ptr[3 * 8 + 7] = 0;
  ptr[4 * 8 + 0] = 0;
  ptr[4 * 8 + 1] = 0;
  ptr[4 * 8 + 2] = 0;
  ptr[4 * 8 + 3] = 0;
  ptr[4 * 8 + 4] = 1;
  ptr[4 * 8 + 5] = 0;
  ptr[4 * 8 + 6] = 0;
  ptr[4 * 8 + 7] = 0;
  ptr[5 * 8 + 0] = 0;
  ptr[5 * 8 + 1] = 0;
  ptr[5 * 8 + 2] = 0;
  ptr[5 * 8 + 3] = 0;
  ptr[5 * 8 + 4] = 0;
  ptr[5 * 8 + 5] = 0;
  ptr[5 * 8 + 6] = 1;
  ptr[5 * 8 + 7] = 0;
  ptr[6 * 8 + 0] = 0;
  ptr[6 * 8 + 1] = 0;
  ptr[6 * 8 + 2] = 0;
  ptr[6 * 8 + 3] = 0;
  ptr[6 * 8 + 4] = 0;
  ptr[6 * 8 + 5] = 1;
  ptr[6 * 8 + 6] = 0;
  ptr[6 * 8 + 7] = 0;
  ptr[7 * 8 + 0] = 0;
  ptr[7 * 8 + 1] = 0;
  ptr[7 * 8 + 2] = 0;
  ptr[7 * 8 + 3] = 0;
  ptr[7 * 8 + 4] = 0;
  ptr[7 * 8 + 5] = 0;
  ptr[7 * 8 + 6] = 0;
  ptr[7 * 8 + 7] = 1;
}
CSwap::~CSwap() {}

U::U(std::vector<size_t> targets, double theta, double phi, double lambda,
     std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::U, targets, {theta, phi, lambda},
                    ctrl_modifiers, power),
      theta_(theta),
      phi_(phi),
      lambda_(lambda) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 2 + 0] = cos(theta / 2);
  ptr[0 * 2 + 1] = -std::exp(1i * lambda) * sin(theta / 2);
  ptr[1 * 2 + 0] = std::exp(1i * phi) * sin(theta / 2);
  ptr[1 * 2 + 1] = std::exp(1i * (phi + lambda)) * cos(theta / 2);
}
U::~U() {}

GPhase::GPhase(std::vector<size_t> targets, double angle,
               std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::GPhase, targets, {angle}, ctrl_modifiers,
                    power),
      angle_(angle) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0] = std::exp(1i * angle);
}
GPhase::~GPhase() {}

SwapA2A::SwapA2A(std::vector<size_t> targets,
                 std::vector<size_t> ctrl_modifiers, size_t power)
    : GateOperation(GateOperationType::SwapA2A, targets, {}, ctrl_modifiers,
                    power) {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  ptr[0 * 4 + 0] = 1;
  ptr[0 * 4 + 1] = 0;
  ptr[0 * 4 + 2] = 0;
  ptr[0 * 4 + 3] = 0;
  ptr[1 * 4 + 0] = 0;
  ptr[1 * 4 + 1] = 0;
  ptr[1 * 4 + 2] = 1;
  ptr[1 * 4 + 3] = 0;
  ptr[2 * 4 + 0] = 0;
  ptr[2 * 4 + 1] = 1;
  ptr[2 * 4 + 2] = 0;
  ptr[2 * 4 + 3] = 0;
  ptr[3 * 4 + 0] = 0;
  ptr[3 * 4 + 1] = 0;
  ptr[3 * 4 + 2] = 0;
  ptr[3 * 4 + 3] = 1;
}
SwapA2A::~SwapA2A() {}
