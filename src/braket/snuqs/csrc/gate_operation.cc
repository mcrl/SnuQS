#include "gate_operation.h"
#include "spdlog/spdlog.h"
#include <iostream>

using namespace std::complex_literals;

OneQubitGate::OneQubitGate() { data_ = new std::complex<double>[2 * 2]; }
OneQubitGate::~OneQubitGate() { delete[] data_; }
std::complex<double> *OneQubitGate::data() { return data_; }
size_t OneQubitGate::dim() const { return 2; }
std::vector<size_t> OneQubitGate::shape() const { return {2, 2}; }
std::vector<size_t> OneQubitGate::stride() const {
  return {2 * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

void OneQubitGate::evolve(py::buffer buffer, std::vector<size_t> targets) {
  py::buffer_info info = buffer.request();
  std::complex<double> *buf =
      reinterpret_cast<std::complex<double> *>(info.ptr);
}

TwoQubitGate::TwoQubitGate() { data_ = new std::complex<double>[4 * 4]; }
TwoQubitGate::~TwoQubitGate() { delete[] data_; }
std::complex<double> *TwoQubitGate::data() { return data_; }
size_t TwoQubitGate::dim() const { return 2; }
std::vector<size_t> TwoQubitGate::shape() const { return {4, 4}; }
std::vector<size_t> TwoQubitGate::stride() const {
  return {4 * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

void TwoQubitGate::evolve(py::buffer buffer, std::vector<size_t> targets) {
  throw "NOT IMPLEMENTED";
}

ThreeQubitGate::ThreeQubitGate() { data_ = new std::complex<double>[8 * 8]; }
ThreeQubitGate::~ThreeQubitGate() { delete[] data_; }
std::complex<double> *ThreeQubitGate::data() { return data_; }
size_t ThreeQubitGate::dim() const { return 2; }
std::vector<size_t> ThreeQubitGate::shape() const { return {8, 8}; }
std::vector<size_t> ThreeQubitGate::stride() const {
  return {8 * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

void ThreeQubitGate::evolve(py::buffer buffer, std::vector<size_t> targets) {
  throw "NOT IMPLEMENTED";
}

Identity::Identity() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1;
}
Identity::~Identity() {}

Hadamard::Hadamard() {
  data_ = new std::complex<double>[4];
  data_[0 * 2 + 0] = M_SQRT1_2;
  data_[0 * 2 + 1] = M_SQRT1_2;
  data_[1 * 2 + 0] = M_SQRT1_2;
  data_[1 * 2 + 1] = -M_SQRT1_2;
}
Hadamard::~Hadamard() {}

PauliX::PauliX() {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = 1;
  data_[1 * 2 + 0] = 1;
  data_[1 * 2 + 1] = 0;
}
PauliX::~PauliX() {}

PauliY::PauliY() {
  data_[0 * 2 + 0] = 0;
  data_[0 * 2 + 1] = -1i;
  data_[1 * 2 + 0] = 1i;
  data_[1 * 2 + 1] = 0;
}
PauliY::~PauliY() {}

PauliZ::PauliZ() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1;
}
PauliZ::~PauliZ() {}

CX::CX() {
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

CY::CY() {
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

CZ::CZ() {
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

S::S() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1i;
}
S::~S() {}

Si::Si() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = -1i;
}
Si::~Si() {}

T::T() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * M_PI / 4.);
}
T::~T() {}

Ti::Ti() {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(-1i * M_PI / 4.);
}
Ti::~Ti() {}

V::V() {
  data_[0 * 2 + 0] = 0.5 + 0.5i;
  data_[0 * 2 + 1] = 0.5 - 0.5i;
  data_[1 * 2 + 0] = 0.5 - 0.5i;
  data_[1 * 2 + 1] = 0.5 + 0.5i;
}
V::~V() {}

Vi::Vi() {
  data_[0 * 2 + 0] = 0.5 - 0.5i;
  data_[0 * 2 + 1] = 0.5 + 0.5i;
  data_[1 * 2 + 0] = 0.5 + 0.5i;
  data_[1 * 2 + 1] = 0.5 - 0.5i;
}
Vi::~Vi() {}

PhaseShift::PhaseShift(double angle) : angle_(angle) {
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * angle);
}
PhaseShift::~PhaseShift() {}

CPhaseShift::CPhaseShift(double angle) : angle_(angle) {
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

CPhaseShift00::CPhaseShift00(double angle) : angle_(angle) {
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

CPhaseShift01::CPhaseShift01(double angle) : angle_(angle) {
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

CPhaseShift10::CPhaseShift10(double angle) : angle_(angle) {
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

RotX::RotX(double angle) : angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> i_sin_hanlf_angle = 1i * sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -i_sin_hanlf_angle;
  data_[1 * 2 + 0] = -i_sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotX::~RotX() {}

RotY::RotY(double angle) : angle_(angle) {
  double cos_half_angle = cos(angle_ / 2);
  std::complex<double> sin_hanlf_angle = sin(angle_ / 2);
  data_[0 * 2 + 0] = cos_half_angle;
  data_[0 * 2 + 1] = -sin_hanlf_angle;
  data_[1 * 2 + 0] = sin_hanlf_angle;
  data_[1 * 2 + 1] = cos_half_angle;
}
RotY::~RotY() {}

RotZ::RotZ(double angle) : angle_(angle) {
  std::complex<double> positive_phase = std::exp(1i * angle_ / 2.);
  std::complex<double> negative_phase = std::exp(-1i * angle_ / 2.);
  data_[0 * 2 + 0] = negative_phase;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = positive_phase;
}
RotZ::~RotZ() {}

Swap::Swap() {
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

ISwap::ISwap() {
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

PSwap::PSwap(double angle) : angle_(angle) {
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

XY::XY(double angle) : angle_(angle) {
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

XX::XX(double angle) : angle_(angle) {
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

YY::YY(double angle) : angle_(angle) {
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

ZZ::ZZ(double angle) : angle_(angle) {
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

CCNot::CCNot() {
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

CSwap::CSwap() {
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

Unitary::Unitary() { throw "Not Implemented"; }
Unitary::~Unitary() {}
std::complex<double> *Unitary::data() { return data_; }
size_t Unitary::dim() const { throw "Not Implemeneted"; }
std::vector<size_t> Unitary::shape() const { throw "Not Implemeneted"; }
std::vector<size_t> Unitary::stride() const { throw "Not Implemeneted"; }

U::U(double theta, double phi, double lambda)
    : theta_(theta), phi_(phi), lambda_(lambda) {
  data_[0 * 2 + 0] = cos(theta / 2);
  data_[0 * 2 + 1] = -std::exp(1i * lambda) * sin(theta / 2);
  data_[1 * 2 + 0] = std::exp(1i * phi) * sin(theta / 2);
  data_[1 * 2 + 1] = std::exp(1i * (phi + lambda)) * cos(theta / 2);
}
U::~U() {}

GPhase::GPhase(double angle) : angle_(angle) {
  data_[0 * 2 + 0] = std::exp(1i * angle);
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = std::exp(1i * angle);
}
GPhase::~GPhase() {}
