#ifndef _GATE_OPERATION_IMPL_H_
#define _GATE_OPERATION_IMPL_H_

#include <complex>
#include <vector>

void applyGate(void *_buffer, void *_gate,
               std::vector<size_t> target, size_t nqubits, size_t nelem);

#endif //_GATE_OPERATION_IMPL_H_