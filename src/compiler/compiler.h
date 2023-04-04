#pragma once

#include <vector>
#include <string>

#include "circuit/quantum_circuit.h"

namespace snuqs {

class Compiler {
	private:
	QuantumCircuit circ_;

	public:
	void Compile(const std::string &filename);
	const snuqs::QuantumCircuit& GetQuantumCircuit();
};

} // namespace snuqsl
