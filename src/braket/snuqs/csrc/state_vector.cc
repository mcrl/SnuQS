#include "state_vector.h"

#include <cuda_runtime.h>

#include "utils.h"

StateVector::StateVector(size_t num_qubits) : num_qubits_(num_qubits) {
  data_ = new std::complex<double>[1 << num_qubits];
  if (data_cuda_ != nullptr) {
    cudaFree(data_cuda_);
  }
}
StateVector::~StateVector() { delete[] data_; }
void *StateVector::data() {
  if (device_ != Device::CPU) {
    toCPU();
  }
  return data_;
}

void *StateVector::data_cuda() {
  if (device_ != Device::CUDA) {
    toCUDA();
  }
  return data_cuda_;
}

void StateVector::toCPU() {
  CUDA_CHECK(cudaMemcpy(data_, data_cuda_,
                        num_elems() * sizeof(std::complex<double>),
                        cudaMemcpyDeviceToHost));
  device_ = Device::CPU;
}

void StateVector::toCUDA() {
  if (data_cuda_ == nullptr) {
    CUDA_CHECK(
        cudaMalloc(&data_cuda_, num_elems() * sizeof(std::complex<double>)));
  }
  CUDA_CHECK(cudaMemcpy(data_cuda_, data_,
                        num_elems() * sizeof(std::complex<double>),
                        cudaMemcpyHostToDevice));
  device_ = Device::CUDA;
}

size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_qubits_); }
size_t StateVector::num_qubits() const { return num_qubits_; }
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
std::string StateVector::__repr__() const { return "StateVector"; }
