#include "simulator.hpp"
#include "gpu_utils.hpp"
#include "gate_gpu.hpp"

#include <chrono>

namespace snuqs {

// static methods
namespace {

void simulate(const QuantumCircuit &circ,
		amp_t *state,
		amp_t *d_state,
		gpu::stream_t stream, 
		size_t num_amps) {
	gpu::init(d_state, num_amps, stream);

	for (const auto &g : circ.gates()) {
		//std::cout << *g << "\n";
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
	std::cout << "Statevector GPU Simulator\n";
	std::cout << "Number of qubits: " << num_qubits << "\n";

	num_amps_ = (1ul << num_qubits);

	auto s = std::chrono::system_clock::now();
	gpu::MallocHost((void**)&state_, sizeof(amp_t) * num_amps_);
	gpu::Malloc((void**)&d_state_, sizeof(amp_t) * num_amps_);
	gpu::streamCreate(&stream_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Setup time: " << sec.count() << "s\n";
}

void StatevectorGPUSimulator::deinit() {
	auto s = std::chrono::system_clock::now();
	gpu::streamDestroy(stream_);
	gpu::Free(d_state_);
	gpu::FreeHost(state_);
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Teardown time: " << sec.count() << "s\n";
}

void StatevectorGPUSimulator::run(const std::vector<QuantumCircuit> &circs) {

	std::cout << "Simulation start..\n";
	auto s = std::chrono::system_clock::now();
	for (auto & circ : circs) {
		simulate(circ, state_, d_state_, stream_, num_amps_);
	}
	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Simulation Time: " << sec.count() << "s\n";


//	for (size_t i = 0; i < 32; i++) {
//		std::cout << i << ": " << state_[i] << "\n";
//	}
//	for (size_t i = (1ul << circs[0].num_qubits()) - 32; i < (1ul << circs[0].num_qubits()); i++) {
//		std::cout << i << ": " << state_[i] << "\n";
//	}

}

} // namespace snuqs

