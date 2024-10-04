#ifndef _UTILS_H_
#define _UTILS_H_

#include <cuda_runtime.h>

#include <cstdio>
#define CUDA_CHECK(e)                                                   \
  do {                                                                  \
    cudaError_t r = (e);                                                \
    if (r != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error: %s %s %s:%d\n", cudaGetErrorName(r), \
              cudaGetErrorString(r), __FILE__, __LINE__);               \
    }                                                                   \
  } while (0)

#endif  //_UTILS_H_
