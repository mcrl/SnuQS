#ifndef __STATEVECTOR_SIMULATOR_H__
#define __STATEVECTOR_SIMULATOR_H__

#include "quantum_circuit/quantum_circuit.h"
#include "simulator/simulator.h"

namespace snuqs {

template <typename T>
class StatevectorSimulator : public Simulator<T> {
public:
  StatevectorSimulator();
  virtual ~StatevectorSimulator() override;

  virtual void run() override;
};

} // namespace snuqs
#endif //__STATEVECTOR_SIMULATOR_H__
