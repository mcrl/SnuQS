#include "circuit/quantum_circuit.h"

namespace snuqs {

namespace {
}

QuantumCircuit::QuantumCircuit()
{
}

QuantumCircuit::QuantumCircuit(const QuantumCircuit &other) 
: num_qubits_(other.num_qubits())
{
	auto &gates = other.gates();
	gates_.clear();
	gates_ = gates;
}

QuantumCircuit& QuantumCircuit::operator=(const QuantumCircuit &other) {
	if (this != &other) {
		num_qubits_ = other.num_qubits();

		auto &gates = other.gates();
		gates_.clear();
		gates_ = gates;
	}
	return *this;
}


size_t QuantumCircuit::num_qubits() const {
	return num_qubits_;
}

void QuantumCircuit::set_num_qubits(size_t num_qubits) {
	num_qubits_ = num_qubits;
}

const std::vector<size_t>& QuantumCircuit::permutation() const {
	return perm_;
}

void QuantumCircuit::set_permutation(const std::vector<size_t> &perm) {
	perm_ = perm;
}

void QuantumCircuit::addGate(Gate::shared_ptr gate) {
	gates_.push_back(gate); 
}

void QuantumCircuit::addGate(Gate::shared_ptr gate, size_t i) {
	gates_.insert(gates_.begin()+i, gate);
}

void QuantumCircuit::clearGate() {
	gates_.clear(); 
}

void QuantumCircuit::clearGate(size_t i) {
	gates_.erase(gates_.begin()+i);
}

const std::vector<Gate::shared_ptr>& QuantumCircuit::gates() const {
	return gates_;
}
 
std::vector<Gate::shared_ptr>& QuantumCircuit::gates() {
	return gates_;
}

std::ostream& QuantumCircuit::operator<<(std::ostream &os) const {
	if (!gates_.empty()) {
		size_t i = 0; 
		for (const auto &g : gates_) {
			os << *g;
			if (i++ != (gates_.size()-1)) os << "\n";
		}
	}

	return os;
}

std::ostream &operator<<(std::ostream &os, const QuantumCircuit &circ) {
    return circ.operator<<(os);
}

} // namespace snuqs
