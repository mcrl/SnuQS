#include "utils.h"

namespace cu {
int device_count() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}
};  // namespace cu
