#include "simulator.h"

namespace snuqs {

void NoSimulator::init(size_t num_qubits) {
  throw std::runtime_error("No simulator");
}

void NoSimulator::deinit() {
  throw std::runtime_error("No simulator");
}

void NoSimulator::run(const std::vector<QuantumCircuit> &circs) {
  throw std::runtime_error("No simulator");
}

} // namespace snuqs
