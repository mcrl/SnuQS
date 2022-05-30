#pragma once

#include <vector>
#include <string>

#include "job.hpp"
#include "quantumCircuit.hpp"

namespace snuqs {

class SnuQLCompiler {
	private:
	QuantumCircuit circ_;

	public:
	void compile(const std::string &filename);

	const snuqs::QuantumCircuit& getQuantumCircuit();
};

} // namespace snuqsl
