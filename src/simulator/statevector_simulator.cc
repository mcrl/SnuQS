#include "simulator/statevector_simulator.h"
#include "assertion.h"
#include "buffer/buffer.h"
#include "circuit/arg.h"
#include "circuit/circuit.h"
#include "circuit/reg.h"
#include "cuda_api.h"
#include "simulator/run.h"

extern "C" {
#include <sys/sysinfo.h>
}

namespace snuqs {

template <typename T> StatevectorSimulator<T>::StatevectorSimulator() {}

template <typename T> StatevectorSimulator<T>::~StatevectorSimulator() {}

template <typename T>
std::shared_ptr<Buffer<T>> StatevectorSimulator<T>::run(Circuit &circ) {
  //
  //
  //
  size_t num_qubits = 0;
  for (auto &qreg : circ.qregs()) {
    num_qubits += qreg->dim();
  }

  //
  //
  //
  size_t state_size = (1ull << num_qubits) * sizeof(std::complex<T>);

  //
  //
  //
  int num_devices;
  cuda::api::getDeviceCount(&num_devices);

  //
  //
  //
  size_t min_cuda_mem = SIZE_MAX;
  for (int i = 0; i < num_devices; ++i) {
    size_t free, total;
    cuda::api::setDevice(i);
    cuda::api::memGetInfo(&free, &total);
    min_cuda_mem = std::min(free, min_cuda_mem);
  }
  size_t min_cuda_mem_md = min_cuda_mem * num_devices;

  //
  //
  //
  struct sysinfo info;
  sysinfo(&info);
  size_t min_cpu_mem = info.freeram;

  if (state_size <= min_cuda_mem) {
      std::shared_ptr<Buffer<T>> buffer = cuda::runSingleGPU<T>(circ);
      return buffer;
  } else if (state_size <= min_cuda_mem_md) {
      std::shared_ptr<Buffer<T>> buffer = cuda::runMultiGPU<T>(circ);
      return buffer;
  } else if (state_size <= min_cpu_mem) {
      std::shared_ptr<Buffer<T>> buffer = cuda::runCPU<T>(circ, min_cuda_mem);
      return buffer;
  } else {
      std::shared_ptr<Buffer<T>> buffer = cuda::runStorage<T>(circ);
      return buffer;
  }
}

template class StatevectorSimulator<float>;
template class StatevectorSimulator<double>;

} // namespace snuqs
