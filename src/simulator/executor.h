#ifndef __EXECUTOR_H__
#define __EXECUTOR_H__

#include "buffer/buffer.h"
#include "circuit/qop.h"

namespace snuqs {
namespace cuda {
template <typename T> void exec(Qop *qop, Buffer<T> *buffer, size_t num_states);
} // namespace cuda
} // namespace snuqs

#endif //__EXECUTOR_H__
