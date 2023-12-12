#include "assertion.h"
#include "cuda_api.h"
#include "simulator/executor.h"
#include "simulator/qop_impl.h"
#include "simulator/run.h"
#include "simulator/transpile.h"

#include <iostream>
#include <unistd.h>

namespace snuqs {
namespace cuda {

template <typename T> std::shared_ptr<Buffer<T>> runCPU(Circuit &_circ) {
  NOT_IMPLEMENTED();
}

template std::shared_ptr<Buffer<float>> runCPU<float>(Circuit &circ);
template std::shared_ptr<Buffer<double>> runCPU<double>(Circuit &circ);
} // namespace cuda
} // namespace snuqs
