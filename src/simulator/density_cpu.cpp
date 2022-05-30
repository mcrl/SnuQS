#include "configure.hpp"
#include "simulator.hpp"

#include "gate_cpu.hpp"

#include <chrono>
#include <malloc.h>

namespace snuqs {

// static methods
namespace {


void simulate(const QuantumCircuit &circ, amp_t *state) {
	size_t num_qubits = circ.num_qubits();
	size_t num_amps = (1ULL << num_qubits);

	cpu::init(state, num_amps);

	for (const auto &g : circ.gates()) {
		std::cout << *g << "\n";
		g->apply(state, num_amps);
	}
}

} // static methods

void DensityCPUSimulator::init(size_t num_qubits) {
	std::cout << "Density CPU Simulator\n";
	std::cout << "Number of qubits: " << num_qubits << "\n";

	num_amps_ = (1ul << num_qubits);

	auto s = std::chrono::system_clock::now();
	state_ = reinterpret_cast<amp_t*>(memalign(4096, sizeof(amp_t) * num_amps_));
	assert(state_ != nullptr);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Setup time: " << sec.count() << "s\n";
}

void DensityCPUSimulator::deinit() {
	auto s = std::chrono::system_clock::now();
	free(state_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Teardown time: " << sec.count() << "s\n";
}

void DensityCPUSimulator::run(const std::vector<QuantumCircuit> &circs) {
	const QuantumCircuit &circ = circs[0];

	std::cout << "Simulation start..\n";
	auto s = std::chrono::system_clock::now();
	simulate(circ, state_);
	auto sec = (std::chrono::system_clock::now() - s);
	std::cout << "Simulation Time: " << sec.count() << "s\n";

	std::cout << state_[0] << "\n";
	std::cout << state_[1] << "\n";
	std::cout << state_[2] << "\n";
	std::cout << state_[3] << "\n";

	std::cout << state_[(1ul << circ.num_qubits()) - 4] << "\n";
	std::cout << state_[(1ul << circ.num_qubits()) - 3] << "\n";
	std::cout << state_[(1ul << circ.num_qubits()) - 2] << "\n";
	std::cout << state_[(1ul << circ.num_qubits()) - 1] << "\n";

}

} // namespace snuqs

