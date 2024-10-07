#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <complex>
#include <cstddef>
#include <vector>

#include "device_type.h"
#include "result_types/result_types.h"

class StateVector : public ResultType {
 public:
  StateVector(size_t num_qubits);
  ~StateVector();
  virtual void *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;

  void *data_cuda();
  void cpu();
  void cuda();
  bool allocated() const;
  size_t num_elems() const;
  size_t num_qubits() const;
  DeviceType device() const;
  std::string formatted_string() const;
  void set_initialized();
  bool initialized() const;

 private:
  size_t num_qubits_;
  std::complex<double> *data_ = nullptr;
  std::complex<double> *data_cuda_ = nullptr;
  DeviceType device_ = DeviceType::UNKNOWN;
  bool initialized_ = false;
};

#endif  //_STATE_VECTOR_H_
