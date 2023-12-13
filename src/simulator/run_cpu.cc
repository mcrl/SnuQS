#include "assertion.h"
#include "cuda_api.h"
#include "simulator/executor.h"
#include "simulator/qop_impl.h"
#include "simulator/run.h"
#include "transpile/transpile.h"

#include <iostream>
#include <unistd.h>

namespace snuqs {
namespace cuda {

template <typename T>
std::shared_ptr<Buffer<T>> runCPU(Circuit &_circ, size_t mem_per_device) {

  size_t num_qubits = 0;
  for (auto &qreg : _circ.qregs()) {
    num_qubits += qreg->dim();
  }

  int num_devices;
  api::getDeviceCount(&num_devices);

  size_t num_states = (1ull << num_qubits);
  assert(num_states % num_devices == 0);


  size_t num_qubits_per_device = 1;

  while ((1ull << num_qubits_per_device) <= mem_per_device / sizeof(std::complex<T>)) {
    num_qubits_per_device += 1;
  };
  num_qubits_per_device -= 1; // sub 1
  size_t num_states_per_device = (1ull << num_qubits_per_device);
  std::cout << "num_states_per_device: " << num_states_per_device << "\n";

  std::shared_ptr<Buffer<T>> mem_buffer =
      std::make_shared<MemoryBuffer<T>>(num_states);

#pragma omp parallel for num_threads(num_devices)
  for (int i = 0; i < num_devices; ++i) {
    api::setDevice(i);

    std::shared_ptr<Circuit> circ = transpileCPU(_circ, num_qubits_per_device, num_qubits, i, num_devices);

    std::shared_ptr<Buffer<T>> buffer =
        std::make_shared<CudaBuffer<T>>(num_states_per_device);

    if (i == 0) {
        std::cout << circ->__repr__() << "\n";
    }

//    int k = 0;
//    for (auto &qop : circ->qops()) {
//      exec<T>(qop.get(), buffer.get(), num_states_per_device, mem_buffer.get());
//    }
//
//    api::memcpy(mem_buffer->ptr() + i * num_states_per_device, buffer->ptr(),
//                num_states_per_device * sizeof(std::complex<T>),
//                cudaMemcpyDeviceToHost);
  }

  return mem_buffer;
}

template std::shared_ptr<Buffer<float>> runCPU<float>(Circuit &circ,
                                                      size_t mem_per_device);
template std::shared_ptr<Buffer<double>> runCPU<double>(Circuit &circ,
                                                        size_t mem_per_device);
} // namespace cuda
} // namespace snuqs
