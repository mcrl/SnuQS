#pragma once
#include <ostream>
#include <fstream>
#include <algorithm>
#include <set>

#include "circuit/gate.h"

namespace snuqs {

enum class InitMethod {
	ZERO,
	UNIFORM,
};

class Gate;
class QuantumCircuit {
  public:

    using gate_iterator = std::vector<std::shared_ptr<Gate>>::iterator;

    QuantumCircuit();
    QuantumCircuit(const QuantumCircuit &other);
    QuantumCircuit& operator=(const QuantumCircuit &other);

    size_t num_qubits() const;
    void set_num_qubits(size_t);

    const std::vector<size_t>& permutation() const;
    void set_permutation(const std::vector<size_t> &perm);

    void addGate(std::shared_ptr<Gate> gate);
    void addGate(std::shared_ptr<Gate> gate, size_t i);
    void clearGate();
    void clearGate(size_t i);

    const std::vector<std::shared_ptr<Gate>> & gates() const;
    std::vector<std::shared_ptr<Gate>>& gates();

    std::ostream& operator<<(std::ostream &os) const;
    friend std::ostream& operator<<(std::ostream &os, const QuantumCircuit &circ);

  private:
    size_t num_qubits_ = 0;
    std::vector<std::shared_ptr<Gate>> gates_;
    std::vector<size_t> perm_;

};

}  // namespace snuqs
