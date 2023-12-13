#include "assertion.h"
#include "simulator/run.h"

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runStorage(Circuit &circ) {
  NOT_IMPLEMENTED();
}
//template std::shared_ptr<Buffer<float>> runStorage<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runStorage<double>(Circuit &circ);
} // namespace cuda
} // namespace snuqs
