#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include "device.h"
#include <complex>
#include <cstddef>
#include <vector>

class StateVector {
public:
  StateVector(size_t num_qubits);
  ~StateVector();
  void *data();
  void *data_cuda();
  void toCPU();
  void toCUDA();
  size_t dim() const;
  size_t num_elems() const;
  size_t num_qubits() const;
  std::vector<size_t> shape() const;
  std::string __repr__() const;

private:
  size_t num_qubits_;
  std::complex<double> *data_ = nullptr;
  std::complex<double> *data_cuda_ = nullptr;
  Device device_ = Device::CPU;
};

#endif //_STATE_VECTOR_H_
