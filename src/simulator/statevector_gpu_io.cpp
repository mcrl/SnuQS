#include "simulator.h"

#include <chrono>
#include <cassert>

#include "gpu_utils.h"
#include "circuit/gate_cuda.h"
#include "logger.h"


namespace snuqs {

// static methods
namespace {

void simulate(const QuantumCircuit &circ,
		amp_t *state,
		amp_t *d_state,
		gpu::stream_t stream, 
		size_t num_amps) {
	assert(false);
	//gpu::init(d_state, num_amps, stream);

	for (const auto &g : circ.gates()) {
		g->applyGPU(d_state, num_amps, stream);
	}

	gpu::MemcpyAsyncD2H(state,
			d_state,
			sizeof(amp_t) * num_amps,
			stream
			);
	gpu::deviceSynchronize();
}

} // static methods

void StatevectorGPUSimulator::init(size_t num_qubits) {
	Logger::info("Statevector GPU Simulator\n");
	Logger::info("Number of qubits: {}\n", num_qubits);

	num_amps_ = (1ul << num_qubits);

	auto s = std::chrono::system_clock::now();
	gpu::MallocHost((void**)&state_, sizeof(amp_t) * num_amps_);
	gpu::Malloc((void**)&d_state_, sizeof(amp_t) * num_amps_);
	gpu::streamCreate(&stream_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	Logger::info("Setup time: {}\n", sec.count());
}

void StatevectorGPUSimulator::deinit() {
	auto s = std::chrono::system_clock::now();
	gpu::streamDestroy(stream_);
	gpu::Free(d_state_);
	gpu::FreeHost(state_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	Logger::info("Teardown time: {}\n", sec.count());
}

void StatevectorGPUSimulator::run(const std::vector<QuantumCircuit> &circs) {
	Logger::info("Simulation start...\n");
	auto s = std::chrono::system_clock::now();
	for (auto & circ : circs) {
		simulate(circ, state_, d_state_, stream_, num_amps_);
	}
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	Logger::info("Simulation Time: {}\n", sec.count());
}

} // namespace snuqs

