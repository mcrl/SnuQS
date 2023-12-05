#ifndef __STATEVECTOR_SIMULATOR_H__
#define __STATEVECTOR_SIMULATOR_H__

#include "quantum_circuit/quantum_circuit.h"
#include "simulator.h"

namespace snuqs {

class StatevectorSimulator : public Simulator {
public:
  StatevectorSimulator();
  ~StatevectorSimulator();

  virtual void run() override;
};

} // namespace snuqs
#endif //__STATEVECTOR_SIMULATOR_H__
