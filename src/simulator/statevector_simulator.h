#ifndef __STATEVECTOR_SIMULATOR_H__
#define __STATEVECTOR_SIMULATOR_H__

#include "circuit/circuit.h"
#include "simulator.h"

namespace snuqs {

template <typename T> class StatevectorSimulator : public Simulator<T> {
public:
  StatevectorSimulator();
  ~StatevectorSimulator();

  virtual std::shared_ptr<Buffer<T>> run(Circuit &circ) override;
};

} // namespace snuqs
#endif //__STATEVECTOR_SIMULATOR_H__
