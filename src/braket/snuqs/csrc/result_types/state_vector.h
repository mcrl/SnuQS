#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "buffer/buffer.h"
#include "device_types.h"
#include "result_types/result_types.h"

class StateVector {
 public:
  StateVector(size_t num_qubits);
  StateVector(DeviceType device, size_t num_qubits);
  StateVector(DeviceType device, size_t num_qubits,
              std::shared_ptr<Buffer> buffer);
  virtual ~StateVector();

  void *ptr();
  std::shared_ptr<StateVector> cpu();
  std::shared_ptr<StateVector> cuda();
  std::shared_ptr<StateVector> slice(size_t num_sliced_qubits, size_t index);
  void set_offset(size_t count);
  void cut(size_t num_effective_qubits);
  void glue();

  DeviceType device() const;
  size_t num_qubits() const;
  std::shared_ptr<Buffer> buffer() const;
  void *data();
  size_t dim() const;
  std::vector<size_t> shape() const;
  size_t num_elems() const;
  std::string formatted_string() const;

 private:
  DeviceType device_ = DeviceType::UNKNOWN;
  size_t num_qubits_ = 0;
  std::shared_ptr<Buffer> buffer_ = nullptr;
};

#endif  //_STATE_VECTOR_H_
