#ifndef __ASSERTION_H__
#define __ASSERTION_H__

#include <cassert>
#include <stdexcept>

#define ERROR(msg) assert(false && (msg));
#define NOT_IMPLEMENTED() (throw std::domain_error("Not implemented yet"))

#define DO_NOTHING()

#define CUDA_ASSERT(e)                                                         \
  do {                                                                         \
    cudaError_t err = (e);                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("[%s:%d] CUDA ERROR: %s\n", __FILE__, __LINE__,                   \
             cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

#endif // __ASSERTION_H__
