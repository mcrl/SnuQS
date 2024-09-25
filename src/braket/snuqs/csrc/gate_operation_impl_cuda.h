#ifndef _GATE_OPERATION_IMPL_CUDA_H_
#define _GATE_OPERATION_IMPL_CUDA_H_

#include <thrust/complex.h>
#include <vector>

namespace cu {

void applyOneQubitGate(thrust::complex<double> *buffer,
                       thrust::complex<double> *gate,
                       std::vector<size_t> targets, size_t nqubits,
                       size_t nelem);

void applyTwoQubitGate(thrust::complex<double> *buffer,
                       thrust::complex<double> *gate,
                       std::vector<size_t> targets, size_t nqubits,
                       size_t nelem);

void applyThreeQubitGate(thrust::complex<double> *buffer,
                         thrust::complex<double> *gate,
                         std::vector<size_t> targets, size_t nqubits,
                         size_t nelem);
} // namespace cu

#endif // _GATE_OPERATION_IMPL_CUDA_H_
