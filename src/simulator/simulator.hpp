#pragma once

#include "quantumCircuit.hpp"
#include "baio.h"


#include <vector>
#include <memory>

namespace snuqs {

enum class SimulationMethod {
	STATEVECTOR,
	DENSITY,
	CONTRACTION,
};

enum class SimulationDevice {
	CPU,
	GPU,
};

class Simulator {
	public:
	using shared_ptr = std::shared_ptr<Simulator>;

	virtual void init(size_t num_qubits) = 0;
	virtual void deinit() = 0;
	virtual void run(const std::vector<QuantumCircuit> &circs) = 0;
};

class StatevectorCPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
	amp_t *state_;
	size_t num_amps_;
};

class StatevectorCPUIOSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;

	amp_t *state_;
	size_t num_amps_;
};

class StatevectorGPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
	private:
	amp_t *state_;
	amp_t *d_state_;
	gpu::stream_t stream_;
	size_t num_amps_;
};

class StatevectorGPUIOSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;

	private:
	amp_t *temp_buf_;
	std::vector<amp_t*> state_pair_;
	std::vector<amp_t*> d_states_;
	size_t num_amps_;
	size_t nbuf_;
	size_t iobuf_elems_;
	struct baio_handle hdlr_;
	gpu::stream_t iostream_;
	size_t ioblock_;
};

class DensityCPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
	private:
	amp_t *state_;
	amp_t *d_state_;
	gpu::stream_t stream_;
	size_t num_amps_;
};

class DensityCPUIOSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;

	amp_t *state_;
	size_t num_amps_;
};

class DensityGPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
	private:
	amp_t *state_;
	amp_t *d_state_;
	gpu::stream_t stream_;
	size_t num_amps_;
};

class DensityGPUIOSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;

	private:
	amp_t *temp_buf_;
	std::vector<amp_t*> state_pair_;
	std::vector<amp_t*> d_states_;
	size_t num_amps_;
	size_t nbuf_;
	size_t iobuf_elems_;
	struct baio_handle hdlr_;
	gpu::stream_t iostream_;
	size_t ioblock_;
};

class ContractionCPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
};

class ContractionGPUSimulator : public Simulator {
	void init(size_t num_qubits) override;
	void deinit() override;
	void run(const std::vector<QuantumCircuit> &circs) override;
};

Simulator::shared_ptr getSimulator(SimulationMethod method, SimulationDevice device, bool useio, size_t nqubits);

} // namespace snuqs
