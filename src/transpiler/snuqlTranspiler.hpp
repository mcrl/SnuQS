#pragma once

#include <vector>
#include <string>

#include "simulator.hpp"
#include "transpiler.hpp"
#include "quantumCircuit.hpp"

namespace snuqs {

class SnuQLTranspiler {
	private:
	QuantumCircuit circ_;
	SimulationMethod method_;
	Transpiler *transpiler_;

	public:
	SnuQLTranspiler(SimulationMethod method);
	~SnuQLTranspiler();
	void transpile(const QuantumCircuit &circ);

	const QuantumCircuit& getQuantumCircuit();
};

} // namespace snuqsl
