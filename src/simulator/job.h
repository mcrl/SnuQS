#pragma once
#include <memory>

#include "circuit/quantum_circuit.h"

namespace snuqs {

class Simulator;

class Job {
  public:
  Job(Simulator *sim);
  ~Job();
  using unique_ptr = std::unique_ptr<Job>;

  void Run(const std::vector<QuantumCircuit> &circs);

  private:
  Simulator *sim_;

};

} // namespace snuqs
