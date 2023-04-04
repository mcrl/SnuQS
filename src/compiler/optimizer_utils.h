#pragma once

#include <memory>
#include "circuit/quantum_circuit.h"

namespace snuqs {

bool lessthan(Gate::shared_ptr &g, Gate::shared_ptr &h);
void sortGatesInv(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, size_t base=0);
void sortGates(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, size_t base=0);
void sortGatesOrder(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, const std::vector<size_t> &order);
void sortGates(QuantumCircuit &circ);

} // namespace snuqs

