#include <thrust/complex.h>

#include <cassert>
#include <complex>

#include "gate_operation_impl_cuda.h"
#include "utils.h"

static constexpr size_t BLOCKDIM = 256;

namespace cu {

static __global__ void applyGlobalPhase_kernel(thrust::complex<double> *buffer,
                                               thrust::complex<double> *gate,
                                               size_t target, size_t nqubits,
                                               size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= nelems) return;

  thrust::complex<double> gphase = gate[0];
  buffer[i] = buffer[i] * gphase;
}

static __global__ void applyOneQubitGate_kernel(thrust::complex<double> *buffer,
                                                thrust::complex<double> *gate,
                                                size_t target, size_t nqubits,
                                                size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st = (1ull << (nqubits - target - 1));

  if (i >= nelems) return;

  if ((i & st) == 0) {
    thrust::complex<double> a0 = buffer[i];
    thrust::complex<double> a1 = buffer[i + st];
    buffer[i] = gate[0 * 2 + 0] * a0 + gate[0 * 2 + 1] * a1;
    buffer[i + st] = gate[1 * 2 + 0] * a0 + gate[1 * 2 + 1] * a1;
  }
}

static __global__ void applyTwoQubitGate_kernel(thrust::complex<double> *buffer,
                                                thrust::complex<double> *gate,
                                                size_t target0, size_t target1,
                                                size_t nqubits, size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st0 = (1ull << (nqubits - target1 - 1));
  size_t st1 = (1ull << (nqubits - target0 - 1));

  if (i >= nelems) return;

  if ((i & st0) == 0 && (i & st1) == 0) {
    thrust::complex<double> a0 = buffer[i + 0];
    thrust::complex<double> a1 = buffer[i + st0];
    thrust::complex<double> a2 = buffer[i + st1];
    thrust::complex<double> a3 = buffer[i + st1 + st0];
    buffer[i + 0] = gate[0 * 4 + 0] * a0 + gate[0 * 4 + 1] * a1 +
                    gate[0 * 4 + 2] * a2 + gate[0 * 4 + 3] * a3;
    buffer[i + st0] = gate[1 * 4 + 0] * a0 + gate[1 * 4 + 1] * a1 +
                      gate[1 * 4 + 2] * a2 + gate[1 * 4 + 3] * a3;
    buffer[i + st1] = gate[2 * 4 + 0] * a0 + gate[2 * 4 + 1] * a1 +
                      gate[2 * 4 + 2] * a2 + gate[2 * 4 + 3] * a3;
    buffer[i + st1 + st0] = gate[3 * 4 + 0] * a0 + gate[3 * 4 + 1] * a1 +
                            gate[3 * 4 + 2] * a2 + gate[3 * 4 + 3] * a3;
  }
}

static __global__ void applyThreeQubitGate_kernel(
    thrust::complex<double> *buffer, thrust::complex<double> *gate,
    size_t target0, size_t target1, size_t target2, size_t nqubits,
    size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t st0 = (1ull << (nqubits - target2 - 1));
  size_t st1 = (1ull << (nqubits - target1 - 1));
  size_t st2 = (1ull << (nqubits - target0 - 1));

  if (i >= nelems) return;

  if ((i & st0) == 0 && (i & st1) == 0 && (i & st2) == 0) {
    thrust::complex<double> a0 = buffer[i + 0];
    thrust::complex<double> a1 = buffer[i + st0];
    thrust::complex<double> a2 = buffer[i + st1];
    thrust::complex<double> a3 = buffer[i + st1 + st0];
    thrust::complex<double> a4 = buffer[i + st2];
    thrust::complex<double> a5 = buffer[i + st2 + st0];
    thrust::complex<double> a6 = buffer[i + st2 + st1];
    thrust::complex<double> a7 = buffer[i + st2 + st1 + st0];
    buffer[i + 0] = gate[0 * 8 + 0] * a0 + gate[0 * 8 + 1] * a1 +
                    gate[0 * 8 + 2] * a2 + gate[0 * 8 + 3] * a3 +
                    gate[0 * 8 + 4] * a4 + gate[0 * 8 + 5] * a5 +
                    gate[0 * 8 + 6] * a6 + gate[0 * 8 + 7] * a7;
    buffer[i + st0] = gate[1 * 8 + 0] * a0 + gate[1 * 8 + 1] * a1 +
                      gate[1 * 8 + 2] * a2 + gate[1 * 8 + 3] * a3 +
                      gate[1 * 8 + 4] * a4 + gate[1 * 8 + 5] * a5 +
                      gate[1 * 8 + 6] * a6 + gate[1 * 8 + 7] * a7;
    buffer[i + st1] = gate[2 * 8 + 0] * a0 + gate[2 * 8 + 1] * a1 +
                      gate[2 * 8 + 2] * a2 + gate[2 * 8 + 3] * a3 +
                      gate[2 * 8 + 4] * a4 + gate[2 * 8 + 5] * a5 +
                      gate[2 * 8 + 6] * a6 + gate[2 * 8 + 7] * a7;
    buffer[i + st1 + st0] = gate[3 * 8 + 0] * a0 + gate[3 * 8 + 1] * a1 +
                            gate[3 * 8 + 2] * a2 + gate[3 * 8 + 3] * a3 +
                            gate[3 * 8 + 4] * a4 + gate[3 * 8 + 5] * a5 +
                            gate[3 * 8 + 6] * a6 + gate[3 * 8 + 7] * a7;
    buffer[i + st2] = gate[4 * 8 + 0] * a0 + gate[4 * 8 + 1] * a1 +
                      gate[4 * 8 + 2] * a2 + gate[4 * 8 + 3] * a3 +
                      gate[4 * 8 + 4] * a4 + gate[4 * 8 + 5] * a5 +
                      gate[4 * 8 + 6] * a6 + gate[4 * 8 + 7] * a7;
    buffer[i + st2 + st0] = gate[5 * 8 + 0] * a0 + gate[5 * 8 + 1] * a1 +
                            gate[5 * 8 + 2] * a2 + gate[5 * 8 + 3] * a3 +
                            gate[5 * 8 + 4] * a4 + gate[5 * 8 + 5] * a5 +
                            gate[5 * 8 + 6] * a6 + gate[5 * 8 + 7] * a7;
    buffer[i + st2 + st1] = gate[6 * 8 + 0] * a0 + gate[6 * 8 + 1] * a1 +
                            gate[6 * 8 + 2] * a2 + gate[6 * 8 + 3] * a3 +
                            gate[6 * 8 + 4] * a4 + gate[6 * 8 + 5] * a5 +
                            gate[6 * 8 + 6] * a6 + gate[6 * 8 + 7] * a7;
    buffer[i + st2 + st1 + st0] = gate[7 * 8 + 0] * a0 + gate[7 * 8 + 1] * a1 +
                                  gate[7 * 8 + 2] * a2 + gate[7 * 8 + 3] * a3 +
                                  gate[7 * 8 + 4] * a4 + gate[7 * 8 + 5] * a5 +
                                  gate[7 * 8 + 6] * a6 + gate[7 * 8 + 7] * a7;
  }
}

static void applyGlobalPhase(thrust::complex<double> *buffer,
                             thrust::complex<double> *gate,
                             std::vector<size_t> targets, size_t nqubits,
                             size_t nelems) {
  applyGlobalPhase_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      buffer, gate, 0, nqubits, nelems);
  CUDA_CHECK(cudaGetLastError());
}

static void applyOneQubitGate(thrust::complex<double> *buffer,
                              thrust::complex<double> *gate,
                              std::vector<size_t> targets, size_t nqubits,
                              size_t nelems) {
  applyOneQubitGate_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      buffer, gate, targets[0], nqubits, nelems);
  CUDA_CHECK(cudaGetLastError());
}

static void applyTwoQubitGate(thrust::complex<double> *buffer,
                              thrust::complex<double> *gate,
                              std::vector<size_t> targets, size_t nqubits,
                              size_t nelems) {
  applyTwoQubitGate_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      buffer, gate, targets[0], targets[1], nqubits, nelems);
  CUDA_CHECK(cudaGetLastError());
}

static void applyThreeQubitGate(thrust::complex<double> *buffer,
                                thrust::complex<double> *gate,
                                std::vector<size_t> targets, size_t nqubits,
                                size_t nelems) {
  applyThreeQubitGate_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      buffer, gate, targets[0], targets[1], targets[2], nqubits, nelems);
  CUDA_CHECK(cudaGetLastError());
}

void applyGate(void *_buffer, void *_gate, std::vector<size_t> targets,
               size_t nqubits, size_t nelems) {
  assert(targets.size() == 1 || targets.size() == 2 || targets.size() == 3);
  auto buffer = reinterpret_cast<thrust::complex<double> *>(_buffer);
  auto gate = reinterpret_cast<thrust::complex<double> *>(_gate);
  if (targets.size() == 0) {  // global Phase
    applyGlobalPhase(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 1) {
    applyOneQubitGate(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 2) {
    applyTwoQubitGate(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 3) {
    applyThreeQubitGate(buffer, gate, targets, nqubits, nelems);
  } else {
    assert(false);
  }
}

}  // namespace cu
