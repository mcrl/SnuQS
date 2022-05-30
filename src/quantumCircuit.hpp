#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <ostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "gate.hpp"
#include "error.hpp"

namespace snuqs {

enum class InitMethod {
	ZERO,
	UNIFORM,
};

class QuantumCircuit {
    public:

    using gate_iterator = std::vector<Gate::shared_ptr>::iterator;

    QuantumCircuit();
	QuantumCircuit(const QuantumCircuit &other);
	QuantumCircuit& operator=(const QuantumCircuit &other);

    size_t num_qubits() const;
    void set_num_qubits(size_t);

	const std::vector<size_t>& permutation() const;
	void set_permutation(const std::vector<size_t> &perm);

    void addGate(Gate::shared_ptr gate);
    void addGate(Gate::shared_ptr gate, size_t i);
    void clearGate();
    void clearGate(size_t i);

	const std::vector<Gate::shared_ptr> & gates() const;
	std::vector<Gate::shared_ptr>& gates();

    std::ostream& operator<<(std::ostream &os) const;
    friend std::ostream& operator<<(std::ostream &os, const QuantumCircuit &circ);

    private:
    size_t num_qubits_ = 0;
    std::vector<Gate::shared_ptr> gates_;
    std::vector<size_t> perm_;

};

}  // namespace snuqs
