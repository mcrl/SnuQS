#ifndef __STATEVECTOR_SIMULATOR_H__
#define __STATEVECTOR_SIMULATOR_H__

#include "quantum-circuit/quantum-circuit.h"

namespace snuqs {

class StatevectorSimulator {
public:
  StatevectorSimulator();
  ~StatevectorSimulator();


  void run(const QuantumCircuit &circ);
};

} // namespace snuqs
#endif //__STATEVECTOR_SIMULATOR_H__
