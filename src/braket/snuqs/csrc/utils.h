#ifndef _UTILS_H_
#define _UTILS_H_

#include <pybind11/pybind11.h>
#include <vector>

#define CUDA_CHECK(e)                                                          \
  do {                                                                         \
    cudaError_t r = (e);                                                       \
    if (r != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA Error: %s %s %s:%d\n", cudaGetErrorName(r),        \
              cudaGetErrorString(r), __FILE__, __LINE__);                      \
    }                                                                          \
  } while (0);

namespace py = pybind11;
void multiply_matrix(py::buffer state, py::buffer matrix,
                     std::vector<int> targets);

#endif //_UTILS_H_
