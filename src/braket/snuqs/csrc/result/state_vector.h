#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <complex>
#include <cstddef>
#include <vector>

#include "device.h"

class StateVector {
 public:
  StateVector(size_t num_qubits);
  ~StateVector();
  void *data();
  void *data_cuda();

  void cpu();
  void cuda();
  bool allocated() const;
  size_t dim() const;
  size_t num_elems() const;
  size_t num_qubits() const;
  std::vector<size_t> shape() const;
  Device device() const;
  std::string formatted_string() const;

 private:
  size_t num_qubits_;
  std::complex<double> *data_ = nullptr;
  std::complex<double> *data_cuda_ = nullptr;
  Device device_ = Device::UNKNOWN;
};

#endif  //_STATE_VECTOR_H_
