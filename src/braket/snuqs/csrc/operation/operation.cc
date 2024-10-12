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
GateOperation::GateOperation(const std::vector<size_t> &targets,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
    : Operation(targets), ctrl_modifiers_(ctrl_modifiers), power_(power) {
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

bool GateOperation::sliceable() const { return false; }
void GateOperation::slice(size_t idx) { assert(false); }
bool GateOperation::symmetric() const { return false; }

std::string GateOperation::name() const { return "Unknown"; }
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
