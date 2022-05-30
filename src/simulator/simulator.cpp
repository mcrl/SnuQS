#include "simulator.hpp"

namespace snuqs {

Simulator::shared_ptr getCPUSimulator(SimulationMethod method, size_t nqubits) {
	switch (method) {
		case SimulationMethod::STATEVECTOR:
			return std::make_unique<StatevectorCPUSimulator>();
		case SimulationMethod::DENSITY:
			return std::make_unique<DensityCPUSimulator>();
		case SimulationMethod::CONTRACTION:
			return std::make_unique<ContractionCPUSimulator>();
	}
	return std::make_unique<StatevectorCPUSimulator>();
}

Simulator::shared_ptr getGPUSimulator(SimulationMethod method, size_t nqubits) {
	switch (method) {
		case SimulationMethod::STATEVECTOR:
			return std::make_unique<StatevectorGPUSimulator>();
		case SimulationMethod::DENSITY:
			return std::make_unique<DensityGPUSimulator>();
		case SimulationMethod::CONTRACTION:
			return std::make_unique<ContractionGPUSimulator>();
	}
	return std::make_unique<StatevectorGPUSimulator>();
}

Simulator::shared_ptr getCPUIOSimulator(SimulationMethod method, size_t nqubits) {
	switch (method) {
		case SimulationMethod::STATEVECTOR:
			return std::make_unique<StatevectorCPUIOSimulator>();
		case SimulationMethod::DENSITY:
			return std::make_unique<DensityCPUIOSimulator>();
		case SimulationMethod::CONTRACTION:
			assert(false);
			//return std::make_unique<ContractionCPUIOSimulator>();
	}
	return std::make_unique<StatevectorCPUIOSimulator>();
}

Simulator::shared_ptr getGPUIOSimulator(SimulationMethod method, size_t nqubits) {
	switch (method) {
		case SimulationMethod::STATEVECTOR:
			return std::make_unique<StatevectorGPUIOSimulator>();
		case SimulationMethod::DENSITY:
			return std::make_unique<DensityGPUIOSimulator>();
		case SimulationMethod::CONTRACTION:
			assert(false);
			//return std::make_unique<ContractionGPUIOSimulator>();
	}
	return std::make_unique<StatevectorGPUIOSimulator>();
}

Simulator::shared_ptr getSimulator(
	SimulationMethod method,
	SimulationDevice device,
	bool useio, 
	size_t nqubits) {

	if (!useio) {
		if (device == SimulationDevice::CPU) {
			return getCPUSimulator(method, nqubits);
		} else {
			return getGPUSimulator(method, nqubits);
		}
	} else {
		if (device == SimulationDevice::CPU) {
			return getCPUIOSimulator(method, nqubits);
		} else {
			return getGPUIOSimulator(method, nqubits);
		}
	}
}



} // namespace snuqs
