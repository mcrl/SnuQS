#ifndef __STATEVECTOR_SIMULATOR_H__
#define __STATEVECTOR_SIMULATOR_H__

#include "circuit/circuit.h"
#include "simulator.h"

namespace snuqs {

class StatevectorSimulator : public Simulator {
public:
  StatevectorSimulator();
  ~StatevectorSimulator();

  virtual void run(Circuit &circ) override;
  void test();
};

} // namespace snuqs
#endif //__STATEVECTOR_SIMULATOR_H__
