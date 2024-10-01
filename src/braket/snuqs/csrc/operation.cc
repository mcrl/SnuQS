#include "operation.h"

#include <cuda_runtime.h>

#include "utils.h"

void *GateOperation::data() { return data_; }
void *GateOperation::data_cuda() {
  if (!copied_to_cuda) {
    CUDA_CHECK(cudaMemcpy(data_cuda_, data_,
                          num_elems() * sizeof(std::complex<double>),
                          cudaMemcpyHostToDevice));
    copied_to_cuda = true;
  }
  return data_cuda_;
}

size_t GateOperation::num_elems() const {
  size_t nelems = 1;
  auto sh = shape();
  for (size_t d = 0; d < dim(); ++d) {
    nelems *= sh[d];
  }
  return nelems;
}
