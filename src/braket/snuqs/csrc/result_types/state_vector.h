#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "buffer/pbuffer.h"
#include "device_types.h"

class StateVector : public std::enable_shared_from_this<StateVector> {
 public:
  StateVector(size_t num_qubits);
  StateVector(size_t num_qubits, bool pinned);
  StateVector(DeviceType device, size_t num_qubits);
  StateVector(DeviceType device, size_t num_qubits,
              std::shared_ptr<PBuffer> buffer);
  virtual ~StateVector();

  void *ptr();
  std::shared_ptr<StateVector> cpu();
  std::shared_ptr<StateVector> cuda();
  std::shared_ptr<StateVector> storage();
  std::shared_ptr<StateVector> slice(size_t num_sliced_qubits, size_t index);
  void copy(StateVector &other);

  void *data();
  size_t dim() const;
  size_t num_elems() const;
  std::vector<size_t> shape() const;
  DeviceType device() const;
  size_t num_qubits() const;
  std::shared_ptr<PBuffer> buffer() const;
  std::string formatted_string() const;

 private:
  DeviceType device_;
  size_t num_qubits_ = 0;
  std::shared_ptr<PBuffer> buffer_ = nullptr;
};

#endif  //_STATE_VECTOR_H_
