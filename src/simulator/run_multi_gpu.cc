#include "assertion.h"
#include "cuda_api.h"
#include "simulator/executor.h"
#include "simulator/qop_impl.h"
#include "simulator/run.h"
#include "simulator/transpile.h"

#include <iostream>
namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runMultiGPU(Circuit &_circ) {

  size_t num_qubits = 0;
  for (auto &qreg : _circ.qregs()) {
    num_qubits += qreg->dim();
  }

  int num_devices;
  api::getDeviceCount(&num_devices);

  size_t num_states = (1ull << num_qubits);
  assert(num_states % num_devices == 0);

  size_t num_states_per_device = num_states / num_devices;

  std::shared_ptr<Buffer<T>> mem_buffer =
      std::make_shared<MemoryBuffer<T>>(num_states, true);

#pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; ++i) {
    api::setDevice(i);

    std::shared_ptr<Circuit> circ =
        transpileMultiGPU(_circ, num_qubits, i, num_devices);

    std::shared_ptr<Buffer<T>> buffer =
        std::make_shared<CudaBuffer<T>>(num_states_per_device);

    int k = 0;
    for (auto &qop : circ->qops()) {
      exec<T>(qop.get(), buffer.get(), num_states_per_device, mem_buffer.get());
    }

    api::memcpy(mem_buffer->ptr() + i * num_states_per_device, buffer->ptr(),
                num_states_per_device * sizeof(std::complex<T>),
                cudaMemcpyDeviceToHost);
  }

  return mem_buffer;
}

template std::shared_ptr<Buffer<float>> runMultiGPU<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runMultiGPU<double>(Circuit &circ);
} // namespace cuda
} // namespace snuqs
