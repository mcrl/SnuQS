#include "operation/operation.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <complex>
#include <cstdlib>
#include <sstream>

#include "utils_cuda.h"

// Operation
Operation::Operation(const std::vector<size_t> &targets) : targets_(targets) {}
Operation::~Operation() {}
std::vector<size_t> Operation::get_targets() const { return targets_; }
void Operation::set_targets(const std::vector<size_t> &targets) {
  targets_ = targets;
}

static std::string gate_operation_type_to_string(GateOperationType type) {
  switch (type) {
    case GateOperationType::Identity:
      return "Identity";
    case GateOperationType::Hadamard:
      return "Hadamard";
    case GateOperationType::PauliX:
      return "PauliX";
    case GateOperationType::PauliY:
      return "PauliY";
    case GateOperationType::PauliZ:
      return "PauliZ";
    case GateOperationType::CX:
      return "CX";
    case GateOperationType::CY:
      return "CY";
    case GateOperationType::CZ:
      return "CZ";
    case GateOperationType::S:
      return "S";
    case GateOperationType::Si:
      return "Si";
    case GateOperationType::T:
      return "T";
    case GateOperationType::Ti:
      return "Ti";
    case GateOperationType::V:
      return "V";
    case GateOperationType::Vi:
      return "Vi";
    case GateOperationType::PhaseShift:
      return "PhaseShift";
    case GateOperationType::CPhaseShift:
      return "CPhaseShift";
    case GateOperationType::CPhaseShift00:
      return "CPhaseShift00";
    case GateOperationType::CPhaseShift01:
      return "CPhaseShift01";
    case GateOperationType::CPhaseShift10:
      return "CPhaseShift10";
    case GateOperationType::RotX:
      return "RotX";
    case GateOperationType::RotY:
      return "RotY";
    case GateOperationType::RotZ:
      return "RotZ";
    case GateOperationType::Swap:
      return "Swap";
    case GateOperationType::ISwap:
      return "ISwap";
    case GateOperationType::PSwap:
      return "PSwap";
    case GateOperationType::XY:
      return "XY";
    case GateOperationType::XX:
      return "XX";
    case GateOperationType::YY:
      return "YY";
    case GateOperationType::ZZ:
      return "ZZ";
    case GateOperationType::CCNot:
      return "CCNot";
    case GateOperationType::CSwap:
      return "CSwap";
    case GateOperationType::U:
      return "U";
    case GateOperationType::GPhase:
      return "GPhase";
    case GateOperationType::SwapA2A:
      return "SwapA2A";
  }
  assert(false);
  return "";
}

// GateOperation
GateOperation::GateOperation(GateOperationType type,
                             const std::vector<size_t> &targets,
                             const std::vector<double> &angles,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
    : Operation(targets),
      type_(type),
      angles_(angles),
      ctrl_modifiers_(ctrl_modifiers),
      power_(power) {
  size_t ncols = (1ul << targets.size());
  ptr_ = aligned_alloc(4096, sizeof(std::complex<double>) * ncols * ncols);
  CUDA_CHECK(
      cudaMalloc(&ptr_cuda_, sizeof(std::complex<double>) * ncols * ncols));
}

GateOperation::~GateOperation() {
  CUDA_CHECK(cudaFree(ptr_cuda_));
  free(ptr_);
}

void *GateOperation::data() { return ptr_; }
void *GateOperation::ptr() { return ptr_; }
void *GateOperation::ptr_cuda() {
  if (!copied_to_cuda) {
    CUDA_CHECK(cudaMemcpy(ptr_cuda_, ptr_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
    copied_to_cuda = true;
  }
  return ptr_cuda_;
}

size_t GateOperation::num_elems() const {
  size_t ncols = (1ul << targets_.size());
  return ncols * ncols;
}

size_t GateOperation::dim() const { return 2; }
std::vector<size_t> GateOperation::shape() const {
  size_t ncols = (1ul << targets_.size());
  return {ncols, ncols};
}
std::vector<size_t> GateOperation::stride() const {
  size_t ncols = (1ul << targets_.size());
  return {ncols * sizeof(std::complex<double>), sizeof(std::complex<double>)};
}

std::shared_ptr<GateOperation> GateOperation::slice(size_t idx) const {
  auto g = std::make_shared<GateOperation>(type_, targets_, angles_,
                                           ctrl_modifiers_, power_);
  return g;
}

bool GateOperation::diagonal() const {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  size_t ncols = (1ul << targets_.size());
  for (size_t i = 0; i < ncols; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      if ((i != j) && (ptr[i] == ptr[j])) {
        return false;
      }
    }
  }
  return true;
}

bool GateOperation::anti_diagonal() const {
  std::complex<double> *ptr = reinterpret_cast<std::complex<double> *>(ptr_);
  size_t ncols = (1ul << targets_.size());
  for (size_t i = 0; i < ncols; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      if ((i != (ncols - 1 - j)) && (ptr[i] == ptr[j])) {
        return false;
      }
    }
  }
  return true;
}
bool GateOperation::sliceable() const { return false; }
std::string GateOperation::name() const {
  return gate_operation_type_to_string(type_);
}

std::string GateOperation::formatted_string() const {
  std::stringstream ss;
  ss << name();
  if (angles_.size() > 0) {
    ss << "(";
    for (size_t i = 0; i < angles_.size(); ++i) {
      ss << angles_[i];
      if (i < angles_.size() - 1) ss << ", ";
    }
    ss << ")";
  }

  ss << " {";
  for (size_t i = 0; i < targets_.size(); ++i) {
    ss << targets_[i];
    if (i < targets_.size() - 1) ss << ", ";
  }
  ss << "}";
  return ss.str();
}

GateOperationType GateOperation::type() const { return type_; }
std::vector<double> GateOperation::angles() const { return angles_; }
std::vector<size_t> GateOperation::ctrl_modifiers() const {
  return ctrl_modifiers_;
}
size_t GateOperation::power() const { return power_; }

bool GateOperation::operator==(const GateOperation &other) const {
  auto other_type = other.type();
  auto other_targets = other.get_targets();
  auto other_angles = other.angles();
  auto other_ctrl_modifiers = other.ctrl_modifiers();
  auto other_power = other.power();

  if (targets_.size() != other_targets.size()) return false;
  for (int i = 0; i < targets_.size(); ++i) {
    if (targets_[i] != other_targets[i]) return false;
  }
  if (angles_.size() != other_angles.size()) return false;
  for (int i = 0; i < angles_.size(); ++i) {
    if (angles_[i] != other_angles[i]) return false;
  }
  if (ctrl_modifiers_.size() != other_ctrl_modifiers.size()) return false;
  for (int i = 0; i < ctrl_modifiers_.size(); ++i) {
    if (ctrl_modifiers_[i] != other_ctrl_modifiers[i]) return false;
  }
  return (type_ == other_type) && (power_ == other_power);
}
