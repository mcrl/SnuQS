#include "simulator.h"

#include <cassert>
#include <cstdlib>
#include <chrono>

#include "circuit/gate_cpu_impl.h"
#include "types.h"
#include "logger.h"


namespace snuqs {

// static methods
namespace {

void simulate(const QuantumCircuit &circ, amp_t *state) {
	size_t num_qubits = circ.num_qubits();
	size_t num_amps = (1ULL << num_qubits);

	assert(false);
	//cpu::init(state, num_amps);

	for (const auto &g : circ.gates()) {
		g->Apply(state, num_amps);
	}
}

} // static methods

void StatevectorCPUSimulator::init(size_t num_qubits) {
	Logger::info("Statevector GPU Simulator\n");
	Logger::info("Number of qubits: {}\n", num_qubits);

	num_amps_ = (1ul << num_qubits);

	auto s = std::chrono::system_clock::now();

	state_ = reinterpret_cast<amp_t*>(aligned_alloc(4096, sizeof(amp_t) * num_amps_));
	assert(state_ != nullptr);

	auto sec = (std::chrono::system_clock::now() - s);
	Logger::info("Setup time: {}\n", sec.count());
}

void StatevectorCPUSimulator::deinit() {
	auto s = std::chrono::system_clock::now();
	free(state_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	Logger::info("Teardown time: {}\n", sec.count());
}

void StatevectorCPUSimulator::run(const std::vector<QuantumCircuit> &circs) {
	Logger::info("Simulation start...\n");
	auto s = std::chrono::system_clock::now();
	for (auto &circ : circs) {
		simulate(circ, state_);
	}
	auto sec = (std::chrono::system_clock::now() - s);
	Logger::info("Simulation Time: {}\n", sec.count());
}

} // namespace snuqs

