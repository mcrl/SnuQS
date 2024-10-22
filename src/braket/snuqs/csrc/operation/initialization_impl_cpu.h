#ifndef _INITIALIZATION_IMPL_CPU_H_
#define _INITIALIZATION_IMPL_CPU_H_
#include <cstddef>

namespace cpu {
void initializeZero(void* _buf, size_t nelems);
void initializeBasis_Z(void* _buf, size_t nelems);
};  // namespace cpu
#endif  //_INITIALIZATION_IMPL_CPU_H_
