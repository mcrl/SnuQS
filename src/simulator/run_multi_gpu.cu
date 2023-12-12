#include "assertion.h"
#include "simulator/run.h"

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runMultiGPU(Circuit &circ) {
  NOT_IMPLEMENTED();
}
template std::shared_ptr<Buffer<float>> runMultiGPU<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runMultiGPU<double>(Circuit &circ);
} // namespace cuda
} // namespace snuqs
