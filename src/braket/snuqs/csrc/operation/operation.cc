#include "operation/operation.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <sstream>

#include "operation/gate_operations.h"
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
  size_t ncols = (1ul << targets_.size());
  ptr_ = new std::complex<double>[ncols * ncols];
  CUDA_CHECK(
      cudaMalloc(&ptr_cuda_, sizeof(std::complex<double>) * ncols * ncols));
}

GateOperation::~GateOperation() {
  CUDA_CHECK(cudaFree(ptr_cuda_));
  delete[] ptr_;
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
  size_t ncols = (1ul << targets_.size());
  for (size_t i = 0; i < ncols; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      if ((i != j) && (ptr_[i] == ptr_[j])) {
        return false;
      }
    }
  }
  return true;
}

bool GateOperation::anti_diagonal() const {
  size_t ncols = (1ul << targets_.size());
  for (size_t i = 0; i < ncols; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      if ((i != (ncols - 1 - j)) && (ptr_[i] == ptr_[j])) {
        return false;
      }
    }
  }
  return true;
}
bool GateOperation::sliceable() const { return diagonal(); }
std::string GateOperation::name() const {
  return gate_operation_type_to_string(type_);
}

std::string GateOperation::formatted_string() const {
  std::stringstream ss;
  ss << "<";
  if (angles_.size() == 0) {
    return name();
  } else {
    std::stringstream ss;
    ss << name() << "(";
    for (size_t i = 0; i < angles_.size(); ++i) {
      ss << angles_[i];
      if (i < angles_.size() - 1) ss << ", ";
    }
    ss << ")";
    return ss.str();
  }
  ss << "targets: {";
  for (auto t : targets_) {
    ss << t << ", ";
  }
  ss << "}";
  ss << ">";
  return ss.str();
}
