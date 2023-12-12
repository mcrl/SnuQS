#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include "circuit/circuit.h"
#include "buffer/buffer.h"

namespace snuqs {

template <typename T>
class Simulator {
public:
  virtual ~Simulator() = default;
  virtual std::shared_ptr<Buffer<T>> run(Circuit &circ) = 0;
};

} // namespace snuqs

#endif // __SIMULATOR_H__
