#include "result_types/state_vector.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils_cuda.h"

StateVector::StateVector(size_t num_qubits) : num_qubits_(num_qubits) {}

StateVector::~StateVector() {
  if (data_ != nullptr) delete[] data_;
  if (data_cuda_ != nullptr) cudaFree(data_cuda_);
}

void *StateVector::data() {
  if (device_ != DeviceType::CPU) {
    cpu();
  }
  return data_;
}

void *StateVector::data_cuda() {
  if (device_ != DeviceType::CUDA) {
    cuda();
  }
  return data_cuda_;
}

void StateVector::cpu() {
  if (data_ == nullptr) {
    data_ = new std::complex<double>[1 << num_qubits_];
  }

  if (data_cuda_ != nullptr) {
    CUDA_CHECK(cudaMemcpy(data_, data_cuda_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToHost));
  }
  device_ = DeviceType::CPU;
}

void StateVector::cuda() {
  if (data_cuda_ == nullptr) {
    CUDA_CHECK(
        cudaMalloc(&data_cuda_, num_elems() * sizeof(std::complex<double>)));
  }

  if (data_ != nullptr) {
    CUDA_CHECK(cudaMemcpy(data_cuda_, data_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
  }
  device_ = DeviceType::CUDA;
}

bool StateVector::allocated() const {
  if (device_ == DeviceType::CPU) {
    return (data_ != nullptr);
  } else if (device_ == DeviceType::CUDA) {
    return (data_cuda_ != nullptr);
  } else {
    return (data_ != nullptr) && (data_cuda_ != nullptr);
  }
  return false;
}

void StateVector::set_initialized() { initialized_ = true; }
bool StateVector::initialized() const { return initialized_; }
DeviceType StateVector::device() const { return device_; }
size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_qubits_); }
size_t StateVector::num_qubits() const { return num_qubits_; }
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
std::string StateVector::formatted_string() const { return "StateVector"; }
