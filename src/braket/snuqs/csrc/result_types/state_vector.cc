#include "result_types/state_vector.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cmath>

#include "utils_cuda.h"

StateVector::StateVector(size_t num_qubits)
    : num_qubits_(num_qubits), num_effective_qubits_(num_qubits) {}

StateVector::StateVector(size_t num_qubits, size_t num_effective_qubits)
    : num_qubits_(num_qubits), num_effective_qubits_(num_effective_qubits) {}

StateVector::~StateVector() {}

void *StateVector::data() { return ptr(); }

void *StateVector::ptr() {
  if (ptr_ == nullptr) {
    ptr_ = std::move(std::make_shared<BufferCPU>(1ul << num_effective_qubits_));
  }
  return &ptr_->buffer()[slice_index_ * (1ul << num_effective_qubits_)];
}

void *StateVector::ptr_cuda() {
  if (ptr_cuda_ == nullptr) {
    ptr_cuda_ = std::make_shared<BufferCUDA>(1ul << num_effective_qubits_);
  }
  return &ptr_cuda_->buffer()[slice_index_ * (1ul << num_effective_qubits_)];
}

void StateVector::cpu() {
  if (device_ == DeviceType::CUDA) {
    assert(ptr_cuda_ != nullptr);
    CUDA_CHECK(cudaMemcpy(ptr(), ptr_cuda(), num_elems() * ptr_->itemsize(),
                          cudaMemcpyDeviceToHost));
  } else {
    ptr();
  }
  device_ = DeviceType::CPU;
}

void StateVector::cuda() {
  if (device_ == DeviceType::CPU) {
    assert(ptr_ != nullptr);
    CUDA_CHECK(cudaMemcpy(ptr_cuda(), ptr(), num_elems() * ptr_->itemsize(),
                          cudaMemcpyHostToDevice));
  } else {
    ptr_cuda();
  }
  device_ = DeviceType::CUDA;
}

void StateVector::copy(StateVector &from) {
  assert(allocated());
  assert(from.allocated());

  auto from_device = from.device();
  if (device_ == DeviceType::CPU && from_device == DeviceType::CPU) {
    memcpy(ptr(), from.ptr(), num_elems() * ptr_->itemsize());
  } else if (device_ == DeviceType::CPU && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(ptr(), from.ptr_cuda(),
                          num_elems() * ptr_->itemsize(),
                          cudaMemcpyDeviceToHost));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CPU) {
    CUDA_CHECK(cudaMemcpy(ptr_cuda(), from.ptr(),
                          num_elems() * ptr_->itemsize(),
                          cudaMemcpyHostToDevice));
  } else if (device_ == DeviceType::CUDA && from_device == DeviceType::CUDA) {
    CUDA_CHECK(cudaMemcpy(ptr_cuda(), from.ptr_cuda(),
                          num_elems() * ptr_->itemsize(),
                          cudaMemcpyDeviceToDevice));
  } else {
    assert(false);
  }
}

void StateVector::upload() {
  size_t num_elems = slice_index_ * (1ul << num_effective_qubits_);
}

void StateVector::download() {
  size_t num_elems = slice_index_ * (1ul << num_effective_qubits_);
}

bool StateVector::allocated() const {
  switch (device_) {
    case DeviceType::UNKNOWN:
      return false;
    case DeviceType::CPU:
      return (ptr_ != nullptr);
    case DeviceType::CUDA:
      return (ptr_cuda_ != nullptr);
  }
  return false;
}

void StateVector::cut(size_t num_effective_qubits) {
  assert(allocated());
  num_effective_qubits_ = num_effective_qubits;
}

void StateVector::glue() {
  CUDA_CHECK(cudaDeviceSynchronize());
  num_effective_qubits_ = num_qubits_;
  slice_index_ = 0;
}

void StateVector::slice(size_t idx) {
  assert(idx >= 0 && idx < (num_qubits_ - num_effective_qubits_));
  slice_index_ = idx;
}

void StateVector::set_initialized() { initialized_ = true; }
bool StateVector::initialized() const { return initialized_; }
DeviceType StateVector::device() const { return device_; }
size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_effective_qubits_); }
size_t StateVector::num_qubits() const { return num_qubits_; }
size_t StateVector::num_effective_qubits() const {
  return num_effective_qubits_;
}
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
std::string StateVector::formatted_string() const {
  std::stringstream ss;

  ss << "StateVector:\n"
     << "\tnum_qubits: " << num_qubits() << "\n"
     << "\tnum_effective_qubits: " << num_effective_qubits() << "\n"
     << "\tdevice: " << static_cast<int>(device());
  return ss.str();
}
