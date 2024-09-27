#ifndef _GATE_OPERATION_IMPL_CUDA_H_
#define _GATE_OPERATION_IMPL_CUDA_H_

#include <vector>

namespace cu {

void applyGate(void *_buffer, void *_gate, std::vector<size_t> targets,
               size_t nqubits, size_t nelems);

} // namespace cu

#endif // _GATE_OPERATION_IMPL_CUDA_H_
