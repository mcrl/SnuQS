#ifndef _STATE_VECTOR_H_
#define _STATE_VECTOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "device_types.h"
#include "result_types/result_types.h"

class StateVector : public ResultType {
 public:
  StateVector(size_t num_qubits);
  StateVector(size_t num_qubits, size_t num_effective_qubits);
  virtual ~StateVector();
  virtual void *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;

  void *ptr();
  void *ptr_cuda();

  void cpu();
  void cuda();
  void copy(StateVector &from);
  void upload();
  void download();
  bool allocated() const;
  size_t num_elems() const;
  size_t num_qubits() const;
  size_t num_effective_qubits() const;
  DeviceType device() const;
  std::string formatted_string() const;
  void set_initialized();
  bool initialized() const;

  void cut(size_t num_effective_qubits);
  void glue();
  void slice(size_t idx);

 private:
  size_t num_qubits_ = 0;
  size_t num_effective_qubits_ = 0;
  size_t slice_index_ = 0;
  std::shared_ptr<BufferCPU> ptr_ = nullptr;
  std::shared_ptr<BufferCUDA> ptr_cuda_ = nullptr;
  DeviceType device_ = DeviceType::UNKNOWN;
  bool initialized_ = false;
};

#endif  //_STATE_VECTOR_H_
