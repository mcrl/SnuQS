#include "assertion.h"
#include "simulator/run.h"

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runCPU(Circuit &circ) {
  NOT_IMPLEMENTED();
}

template std::shared_ptr<Buffer<float>> runCPU<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runCPU<double>(Circuit &circ);
} // namespace cuda
} // namespace snuqs
