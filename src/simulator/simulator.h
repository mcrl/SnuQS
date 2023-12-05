#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include "circuit/circuit.h"

namespace snuqs {

class Simulator {
public:
  virtual ~Simulator() = default;
  virtual void run(Circuit &circ) = 0;
};

} // namespace snuqs

#endif // __SIMULATOR_H__
