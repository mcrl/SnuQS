#include "assertion.h"
#include "buffer/buffer.h"
#include "circuit/arg.h"
#include "circuit/parameter.h"
#include "circuit/qop.h"
#include "simulator/executor.h"
#include "simulator/qop_impl.h"
#include "simulator/run.h"
#include <iostream>

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runSingleGPU(Circuit &circ) {
  size_t num_qubits = 0;
  for (auto &qreg : circ.qregs()) {
    num_qubits += qreg->dim();
  }
  api::setDevice(0);

  size_t num_states = (1ull << num_qubits);
  std::shared_ptr<Buffer<T>> buffer =
      std::make_shared<CudaBuffer<T>>(num_states);

  QopImpl<T>::initZero(buffer->ptr(), num_states, {}, {});

  for (auto &qop : circ.qops()) {
    exec<T>(qop.get(), buffer.get(), num_states);
  }

  std::shared_ptr<Buffer<T>> mem_buffer =
      std::make_shared<MemoryBuffer<T>>(num_states);

  api::memcpy(mem_buffer->ptr(), buffer->ptr(),
              num_states * sizeof(std::complex<T>), cudaMemcpyDeviceToHost);
  return mem_buffer;
}

template std::shared_ptr<Buffer<float>> runSingleGPU<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runSingleGPU<double>(Circuit &circ);

} // namespace cuda
} // namespace snuqs
