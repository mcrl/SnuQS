#ifndef _INITIALIZATION_IMPL_CUDA_H_
#define _INITIALIZATION_IMPL_CUDA_H_

#include <complex>
#include <cstddef>
namespace cu {

void initializeZero(void* _buf, size_t nelems);
void initializeBasis_Z(void* _buf, size_t nelems);

};  // namespace cu
#endif  //_INITIALIZATION_IMPL_CUDA_H_
