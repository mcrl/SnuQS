#pragma once

#include "quantumCircuit.hpp"

namespace snuqs {

class Transpiler {
	public:
	virtual ~Transpiler();
	virtual void transpile(const QuantumCircuit &circ) = 0;
	virtual const QuantumCircuit& getQuantumCircuit() = 0;
};

class NoTranspiler : public Transpiler {
	private:
	QuantumCircuit circ_;

	public:
	~NoTranspiler() override;
	void transpile(const QuantumCircuit &circ) override;
	const QuantumCircuit& getQuantumCircuit() override;
};

class DensityTranspiler : public Transpiler {
	private:
	QuantumCircuit circ_;

	public:
	~DensityTranspiler() override;
	void transpile(const QuantumCircuit &circ) override;

	const QuantumCircuit& getQuantumCircuit() override;
};

} // namespace snuqs
