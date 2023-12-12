#ifndef __RUN_H__
#define __RUN_H__

#include "buffer/buffer.h"
#include "circuit/circuit.h"

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runSingleGPU(Circuit &circ);
template <typename T> std::shared_ptr<Buffer<T>> runMultiGPU(Circuit &circ);
template <typename T> std::shared_ptr<Buffer<T>> runCPU(Circuit &circ);
template <typename T> std::shared_ptr<Buffer<T>> runStorage(Circuit &circ);

} // namespace cuda
} // namespace snuqs

#endif //__RUN_H__
