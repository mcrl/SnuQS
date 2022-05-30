#include "simulator.hpp"
#include "gpu_utils.hpp"
#include "gate_gpu.hpp"
#include "socl.hpp"

#include "baio.h"

#include <chrono>
#include <bitset>
#include <unistd.h>

#define QUEUE_DEPTH (1ul << 15)
#define MAX_REQ (2)

namespace snuqs {

// static methods
namespace {
} // static methods

void StatevectorGPUIOSimulator::init(size_t num_qubits) {
	assert(false);
}

void StatevectorGPUIOSimulator::deinit() {
	assert(false);
}

void StatevectorGPUIOSimulator::run(const std::vector<QuantumCircuit> &circs) {
	assert(false);
}

} // namespace snuqs

