#include "gate_operation.h"
#include <iostream>

Identity::Identity() {
  data_ = new std::complex<double>[4];
  data_[0 * 2 + 0] = 1;
  data_[0 * 2 + 1] = 0;
  data_[1 * 2 + 0] = 0;
  data_[1 * 2 + 1] = 1;
}
Identity::~Identity() { delete[] data_; }
std::complex<double> *Identity::data() { return data_; }
size_t Identity::dim() const { return 2; }
std::vector<size_t> Identity::shape() const { return {2, 2}; }
std::vector<size_t> Identity::stride() const {
  return {2 * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

Hadamard::Hadamard() {
  data_ = new std::complex<double>[4];
  data_[0 * 2 + 0] = M_SQRT1_2;
  data_[0 * 2 + 1] = M_SQRT1_2;
  data_[1 * 2 + 0] = M_SQRT1_2;
  data_[1 * 2 + 1] = -M_SQRT1_2;
}
Hadamard::~Hadamard() { delete[] data_; }
std::complex<double> *Hadamard::data() { return data_; }
size_t Hadamard::dim() const { return 2; }
std::vector<size_t> Hadamard::shape() const { return {2, 2}; }
std::vector<size_t> Hadamard::stride() const {
  return {2 * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

// PauliX::PauliX() { throw "Not Implemented"; }
// std::complex<double> *PauliX::data() { return data_; }
// size_t PauliX::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PauliX::shape() const { throw "Not Implemeneted"; }
//
// PauliY::PauliY() { throw "Not Implemented"; }
// std::complex<double> *PauliY::data() { return data_; }
// size_t PauliY::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PauliY::shape() const { throw "Not Implemeneted"; }
//
// PauliZ::PauliZ() { throw "Not Implemented"; }
// std::complex<double> *PauliZ::data() { return data_; }
// size_t PauliZ::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PauliZ::shape() const { throw "Not Implemeneted"; }
//
// CV::CV() { throw "Not Implemented"; }
// std::complex<double> *CV::data() return data_; }
// size_t CV::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CV::shape() const { throw "Not Implemeneted"; }
//
// CX::CX() { throw "Not Implemented"; }
// std::complex<double> *CX::data() return data_; }
// size_t CX::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CX::shape() const { throw "Not Implemeneted"; }
//
// CY::CY() { throw "Not Implemented"; }
// std::complex<double> *CY::data() return data_; }
// size_t CY::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CY::shape() const { throw "Not Implemeneted"; }
//
// CZ::CZ() { throw "Not Implemented"; }
// std::complex<double> *CZ::data() return data_; }
// size_t CZ::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CZ::shape() const { throw "Not Implemeneted"; }
//
// ECR::ECR() { throw "Not Implemented"; }
// std::complex<double> *ECR::data() { return data_; }
// size_t ECR::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> ECR::shape() const { throw "Not Implemeneted"; }
//
// S::S() { throw "Not Implemented"; }
// std::complex<double> *S::data() return data_; }
// size_t S::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> S::shape() const { throw "Not Implemeneted"; }
//
// Si::Si() { throw "Not Implemented"; }
// std::complex<double> *Si::data() return data_; }
// size_t Si::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> Si::shape() const { throw "Not Implemeneted"; }
//
// T::T() { throw "Not Implemented"; }
// std::complex<double> *T::data() return data_; }
// size_t T::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> T::shape() const { throw "Not Implemeneted"; }
//
// Ti::Ti() { throw "Not Implemented"; }
// std::complex<double> *Ti::data() return data_; }
// size_t Ti::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> Ti::shape() const { throw "Not Implemeneted"; }
//
// V::V() { throw "Not Implemented"; }
// std::complex<double> *V::data() return data_; }
// size_t V::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> V::shape() const { throw "Not Implemeneted"; }
//
// Vi::Vi() { throw "Not Implemented"; }
// std::complex<double> *Vi::data() return data_; }
// size_t Vi::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> Vi::shape() const { throw "Not Implemeneted"; }
//
// PhaseShift::PhaseShift() { throw "Not Implemented"; }
// std::complex<double> *PhaseShift::data() { throw "return data_; }
// size_t PhaseShift::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PhaseShift::shape() const { throw "Not Implemeneted"; }
//
// CPhaseShift::CPhaseShift() { throw "Not Implemented"; }
// std::complex<double> *CPhaseShift::data() { throw "return data_; }
// size_t CPhaseShift::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CPhaseShift::shape() const { throw "Not Implemeneted"; }
//
// CPhaseShift00::CPhaseShift00() { throw "Not Implemented"; }
// std::complex<double> *CPhaseShift00::data() { throw "return data_; }
// size_t CPhaseShift00::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CPhaseShift00::shape() const { throw "Not Implemeneted";
// }
//
// CPhaseShift01::CPhaseShift01() { throw "Not Implemented"; }
// std::complex<double> *CPhaseShift01::data() { throw "return data_; }
// size_t CPhaseShift01::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CPhaseShift01::shape() const { throw "Not Implemeneted";
// }
//
// CPhaseShift10::CPhaseShift10() { throw "Not Implemented"; }
// std::complex<double> *CPhaseShift10::data() { throw "return data_; }
// size_t CPhaseShift10::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CPhaseShift10::shape() const { throw "Not Implemeneted";
// }
//
// RotX::RotX() { throw "Not Implemented"; }
// std::complex<double> *RotX::data() { return data_; }
// size_t RotX::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> RotX::shape() const { throw "Not Implemeneted"; }
//
// RotY::RotY() { throw "Not Implemented"; }
// std::complex<double> *RotY::data() { return data_; }
// size_t RotY::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> RotY::shape() const { throw "Not Implemeneted"; }
//
// RotZ::RotZ() { throw "Not Implemented"; }
// std::complex<double> *RotZ::data() { return data_; }
// size_t RotZ::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> RotZ::shape() const { throw "Not Implemeneted"; }
//
// Swap::Swap() { throw "Not Implemented"; }
// std::complex<double> *Swap::data() { return data_; }
// size_t Swap::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> Swap::shape() const { throw "Not Implemeneted"; }
//
// ISwap::ISwap() { throw "Not Implemented"; }
// std::complex<double> *ISwap::data() { return data_; }
// size_t ISwap::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> ISwap::shape() const { throw "Not Implemeneted"; }
//
// PSwap::PSwap() { throw "Not Implemented"; }
// std::complex<double> *PSwap::data() { return data_; }
// size_t PSwap::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PSwap::shape() const { throw "Not Implemeneted"; }
//
// XY::XY() { throw "Not Implemented"; }
// std::complex<double> *XY::data() return data_; }
// size_t XY::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> XY::shape() const { throw "Not Implemeneted"; }
//
// XX::XX() { throw "Not Implemented"; }
// std::complex<double> *XX::data() return data_; }
// size_t XX::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> XX::shape() const { throw "Not Implemeneted"; }
//
// YY::YY() { throw "Not Implemented"; }
// std::complex<double> *YY::data() return data_; }
// size_t YY::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> YY::shape() const { throw "Not Implemeneted"; }
//
// ZZ::ZZ() { throw "Not Implemented"; }
// std::complex<double> *ZZ::data() return data_; }
// size_t ZZ::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> ZZ::shape() const { throw "Not Implemeneted"; }
//
// CCNot::CCNot() { throw "Not Implemented"; }
// std::complex<double> *CCNot::data() { return data_; }
// size_t CCNot::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CCNot::shape() const { throw "Not Implemeneted"; }
//
// CSwap::CSwap() { throw "Not Implemented"; }
// std::complex<double> *CSwap::data() { return data_; }
// size_t CSwap::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> CSwap::shape() const { throw "Not Implemeneted"; }
//
// PRx::PRx() { throw "Not Implemented"; }
// std::complex<double> *PRx::data() { return data_; }
// size_t PRx::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> PRx::shape() const { throw "Not Implemeneted"; }
//
// Unitary::Unitary() { throw "Not Implemented"; }
// std::complex<double> *Unitary::data() { return data_; }
// size_t Unitary::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> Unitary::shape() const { throw "Not Implemeneted"; }
//
// U::U(): GateOperation() { throw "Not Implemented"; }
// std::complex<double> *U::data() return data_; }
// size_t U::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> U::shape() const { throw "Not Implemeneted"; }
//
// GPhase::GPhase() { throw "Not Implemented"; }
// std::complex<double> *GPhase::data() { return data_; }
// size_t GPhase::dim() const { throw "Not Implemeneted"; }
// std::vector<size_t> GPhase::shape() const { throw "Not Implemeneted"; }
