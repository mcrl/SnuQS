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

// GateOperation
GateOperation::GateOperation(const std::string &name,
                             const std::vector<size_t> &targets,
                             const std::vector<double> &angles,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
    : Operation(targets),
      angles_(angles),
      ctrl_modifiers_(ctrl_modifiers),
      power_(power),
      name_(name) {
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
  auto g = std::make_shared<GateOperation>(name_, targets_, angles_,
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
  if (angles_.size() == 0) {
    return name_;
  } else {
    std::stringstream ss;
    ss << name_ << "(";
    for (size_t i = 0; i < angles_.size(); ++i) {
      ss << angles_[i];
      if (i < angles_.size() - 1) ss << ", ";
    }
    ss << ")";
    return ss.str();
  }
}
std::string GateOperation::formatted_string() const {
  std::stringstream ss;
  ss << "<";
  ss << name() << ": ";
  ss << "targets: {";
  for (auto t : targets_) {
    ss << t << ", ";
  }
  ss << "}";
  ss << ">";
  return ss.str();
}

// GateOperationSliced
GateOperationSliced::GateOperationSliced(
    const std::string &name, const std::vector<size_t> &targets,
    const std::vector<double> &angles,
    const std::vector<size_t> &ctrl_modifiers, size_t power)
    : GateOperation(name, targets, angles, ctrl_modifiers, power) {}
bool GateOperationSliced::sliceable() const { return false; }
GateOperationSliced::~GateOperationSliced() {}
