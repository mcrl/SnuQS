#include "simulator.hpp"

#include <chrono>

#include "socl.h"

namespace snuqs {

// static methods
namespace {

/*
inline void checkError(int err)
{
	assert(false);
}

void post_reads(
		amp_t *buf,
		size_t base,
		size_t count,
		socl::CommandQueue &q
) {
	int err;
	err = enqueueS2H(q, buf, base, count);
	checkError(err);
}

void post_writes() {
	int err;
	err = enqueueH2S(q, buf, base, count);
	checkError(err);
}

		std::vector<amp_t*> &state_pair,
		std::vector<socl::CommandQueue> &queues,
		size_t num_amps, 
		size_t nbuf,
		struct baio_handle &hdlr,
		size_t iobuf_elems,
		gpu::stream_t iostream,
		amp_t *temp_buf,
		size_t ioblock
		) {


	size_t num_amps_in_memory = (1ul << BUFSHIFT);
	size_t niter = (1ul << num_qubits) / num_amps;

	cpu::init(states);

	for (size_t c = 0; c < circs.size(); ++c) {
		auto &circ = circs[c];
		auto &perm = circ.permutation();


		for (size_t it = 0; it < ((c == 0) ? 1 : niter); ++it) {

			post_reads(state);

			for (const auto &g : circ.gates()) {
				g->apply(state, num_amps_in_memory, (num_amps_in_memory*(it+1)-1));
			}

			post_writes(state);
		}
	}
}
*/

void simulate(const std::vector<QuantumCircuit> &circs) {
}

} // static methods

void StatevectorCPUIOSimulator::init(size_t num_qubits) {
	std::cout << "Statevector GPU-IO Simulator\n";
	std::cout << "Number of qubits: " << num_qubits << "\n";

	auto s = std::chrono::system_clock::now();


	int err = socl_init();
	assert(!err);

	std::chrono::duration<double> sec = (std::chrono::system_clock::now() - s);
	std::cout << "Setup time: " << sec.count() << "s\n";
}

void StatevectorCPUIOSimulator::deinit() {
	int err = socl_finalize();
	assert(!err);
}

void StatevectorCPUIOSimulator::run(const std::vector<QuantumCircuit> &circs) {
	simulate(circs);
}

} // namespace snuqs

