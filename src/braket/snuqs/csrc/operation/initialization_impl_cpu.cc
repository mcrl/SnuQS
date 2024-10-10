#include "operation/initialization_impl_cpu.h"

#include <complex>

namespace cpu {

void initializeZero(void* _buf, size_t nelems) {
  auto buf = reinterpret_cast<std::complex<double>*>(_buf);
#pragma omp parallel for
  for (size_t i = 0; i < nelems; ++i) {
    buf[i] = 0;
  }
}

void initializeBasis_Z(void* _buf, size_t nelems) {
  auto buf = reinterpret_cast<std::complex<double>*>(_buf);
#pragma omp parallel for
  for (size_t i = 0; i < nelems; ++i) {
    buf[i] = 0;
  }
  buf[0] = 1;
}

}  // namespace cpu
