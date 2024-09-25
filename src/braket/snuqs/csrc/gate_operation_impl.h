#ifndef _GATE_OPERATION_IMPL_H_
#define _GATE_OPERATION_IMPL_H_

#include <complex>
#include <vector>

void applyOneQubitGate(std::complex<double> *buffer, std::complex<double> *gate,
                       std::vector<size_t> target, size_t nqubits,
                       size_t nelem);
void applyTwoQubitGate(std::complex<double> *buffer, std::complex<double> *gate,
                       std::vector<size_t> targets, size_t nqubits,
                       size_t nelem);

void applyThreeQubitGate(std::complex<double> *buffer,
                         std::complex<double> *gate,
                         std::vector<size_t> targets, size_t nqubits,
                         size_t nelem);

#endif //_GATE_OPERATION_IMPL_H_
