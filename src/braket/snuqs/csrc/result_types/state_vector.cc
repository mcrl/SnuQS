#include "result_types/state_vector.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "utils_cuda.h"

StateVector::StateVector(size_t num_qubits)
    : num_qubits_(num_qubits), num_effective_qubits_(num_qubits) {}

StateVector::~StateVector() {
  if (data_ != nullptr) delete[] data_;
  if (data_cuda_ != nullptr) cudaFree(data_cuda_);
}

void *StateVector::data() {
  if (device_ != DeviceType::CPU) {
    cpu();
  }
  return &data_[slice_offset_];
}

void *StateVector::data_cuda() {
  if (device_ != DeviceType::CUDA) {
    cuda();
  }
  return data_cuda_;
}

void StateVector::cpu() {
  if (data_ == nullptr) {
    data_ = new std::complex<double>[(1ul << num_qubits_)];
  }

  if (device_ == DeviceType::CUDA) {
    assert(data_cuda_ != nullptr);
    CUDA_CHECK(cudaMemcpy(data_, data_cuda_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToHost));
  }
  device_ = DeviceType::CPU;
}

void StateVector::cuda() {
  if (data_cuda_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&data_cuda_, num_elems() * sizeof(std::complex<double>)));
  }

  if (device_ == DeviceType::CPU) {
    assert(data_ != nullptr);
    CUDA_CHECK(cudaMemcpy(data_cuda_, data_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
  }
  device_ = DeviceType::CUDA;
}

void StateVector::copy(StateVector &from) {
  assert(allocated());
  assert(from.allocated());

  auto from_device = from.device();
  if (device_ == DeviceType::CPU && from_device == DeviceType::CPU) {
    memcpy(data(), from.data(), num_elems() * sizeof(std::complex<double>));
  } else if (device_ == DeviceType::CPU && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(data(), from.data_cuda(),
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToHost));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CPU) {
    CUDA_CHECK(cudaMemcpy(data_cuda(), from.data(),
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(data_cuda(), from.data_cuda(),
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToDevice));
  } else {
    assert(false);
  }
}

void StateVector::copy_slice(StateVector &from) {
  assert(allocated());
  assert(from.allocated());

  auto from_device = from.device();
  if (device_ == DeviceType::CPU && from_device == DeviceType::CPU) {
    memcpy(data(), from.data(),
           num_effective_elems() * sizeof(std::complex<double>));
  } else if (device_ == DeviceType::CPU && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(data(), from.data_cuda(),
                          num_effective_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToHost));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CPU) {
    CUDA_CHECK(cudaMemcpy(data_cuda(), from.data(),
                          num_effective_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(data_cuda(), from.data_cuda(),
                          num_effective_elems() * sizeof(std::complex<double>),
                          cudaMemcpyDeviceToDevice));
  } else {
    assert(false);
  }
}

bool StateVector::allocated() const {
  switch (device_) {
    case DeviceType::UNKNOWN:
      return false;
    case DeviceType::CPU:
      return (data_ != nullptr);
    case DeviceType::CUDA:
      return (data_cuda_ != nullptr);
  }
  return false;
}

void StateVector::cut(size_t num_effective_qubits) {
  num_effective_qubits_ = num_effective_qubits;
}

void StateVector::slice(size_t idx) {
  assert(idx >= 0 && idx < num_effective_qubits_);
  slice_offset_ = (1ull << num_effective_qubits_) * idx;
}

void StateVector::set_initialized() { initialized_ = true; }
bool StateVector::initialized() const { return initialized_; }
DeviceType StateVector::device() const { return device_; }
size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_qubits_); }
size_t StateVector::num_effective_elems() const {
  return (1ul << num_effective_qubits_);
}
size_t StateVector::num_qubits() const { return num_qubits_; }
size_t StateVector::num_effective_qubits() const {
  return num_effective_qubits_;
}
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
std::string StateVector::formatted_string() const { return "StateVector"; }
