#include "operation.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils.h"

// Operation
Operation::Operation(const std::vector<size_t> &targets) : targets_(targets) {}
Operation::~Operation() {}
std::vector<size_t> Operation::get_targets() const { return targets_; }
void Operation::set_targets(const std::vector<size_t> &targets) {
  spdlog::info("set_targets({})", targets.size());
  targets_ = targets;
}

// GateOperation
GateOperation::GateOperation(const std::vector<size_t> &targets,
                             const std::vector<size_t> &ctrl_modifiers,
                             size_t power)
    : Operation(targets), ctrl_modifiers_(ctrl_modifiers), power_(power) {
  size_t ncols = (1ul << targets_.size());
  data_ = new std::complex<double>[ncols * ncols];
  CUDA_CHECK(
      cudaMalloc(&data_cuda_, sizeof(std::complex<double>) * ncols * ncols));
}

GateOperation::~GateOperation() {
  delete[] data_;
  CUDA_CHECK(cudaFree(data_cuda_));
}

void *GateOperation::data() { return data_; }
void *GateOperation::data_cuda() {
  if (!copied_to_cuda) {
    CUDA_CHECK(cudaMemcpy(data_cuda_, data_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
    copied_to_cuda = true;
  }
  return data_cuda_;
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
