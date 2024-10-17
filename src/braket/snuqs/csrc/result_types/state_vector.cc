#include "result_types/state_vector.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <complex>

StateVector::StateVector(size_t num_qubits)
    : device_(DeviceType::CPU),
      num_qubits_(num_qubits),
      buffer_(std::make_shared<PBuffer>(sizeof(std::complex<double>) *
                                        (1ul << num_qubits))) {}

StateVector::StateVector(DeviceType device, size_t num_qubits)
    : device_(device),
      num_qubits_(num_qubits),
      buffer_(std::make_shared<PBuffer>(
          device, sizeof(std::complex<double>) * (1ul << num_qubits))) {}

StateVector::StateVector(DeviceType device, size_t num_qubits,
                         std::shared_ptr<PBuffer> buffer)
    : device_(device), num_qubits_(num_qubits), buffer_(buffer) {}

StateVector::~StateVector() {}

void *StateVector::ptr() { return buffer_->buffer(); }

std::shared_ptr<StateVector> StateVector::cpu() {
  return std::make_shared<StateVector>(DeviceType::CPU, num_qubits_,
                                       buffer_->cpu());
}

std::shared_ptr<StateVector> StateVector::cuda() {
  return std::make_shared<StateVector>(DeviceType::CUDA, num_qubits_,
                                       buffer_->cuda());
}

std::shared_ptr<StateVector> StateVector::slice(size_t num_sliced_qubits,
                                                size_t index) {
  auto sv = std::make_shared<StateVector>(device_, num_sliced_qubits, buffer_);
  return sv;
}

void StateVector::cut(size_t num_effective_qubits) {}
void StateVector::glue() {}

void *StateVector::data() { return buffer_->buffer(); }
size_t StateVector::dim() const { return 1; }
size_t StateVector::num_elems() const { return (1ul << num_qubits_); }
std::vector<size_t> StateVector::shape() const { return {num_elems()}; }
DeviceType StateVector::device() const { return device_; }
size_t StateVector::num_qubits() const { return num_qubits_; }
std::shared_ptr<PBuffer> StateVector::buffer() const { return buffer_; }

std::string StateVector::formatted_string() const {
  std::stringstream ss;

  ss << "StateVector<" << num_qubits_
     << " qubits, device: " << device_to_string(device_) << ">";
  return ss.str();
}
