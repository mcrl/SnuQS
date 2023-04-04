#pragma once

#include "circuit/quantum_circuit.h"

#include <vector>
#include <memory>


namespace snuqs {

class Simulator {
	public:
	using unique_ptr = std::unique_ptr<Simulator>;
  enum class Method {
    kStateVector,
  };

  enum class Device {
    kCPU,
    kGPU,
  };

  static Simulator::unique_ptr CreateSimulator(Method method, Device device, bool useio); 
	virtual ~Simulator() = default;

	virtual void init(size_t num_qubits) = 0;
	virtual void deinit() = 0;
	virtual void run(const std::vector<QuantumCircuit> &circs) = 0;
};

class NoSimulator : public Simulator {
	virtual void init(size_t num_qubits) override;
	virtual void deinit() override;
	virtual void run(const std::vector<QuantumCircuit> &circs) override;
};

class StatevectorCPUSimulator : public Simulator {
	virtual void init(size_t num_qubits) override;
	virtual void deinit() override;
	virtual void run(const std::vector<QuantumCircuit> &circs) override;

	amp_t *state_;
	size_t num_amps_;
};

class StatevectorCPUIOSimulator : public Simulator {
	virtual void init(size_t num_qubits) override;
	virtual void deinit() override;
	virtual void run(const std::vector<QuantumCircuit> &circs) override;

	amp_t *state_;
	size_t num_amps_;
};

/*
class StatevectorGPUSimulator : public Simulator {
	virtual void init(size_t num_qubits) override;
	virtual void deinit() override;
	virtual void run(const std::vector<QuantumCircuit> &circs) override;

	private:
	amp_t *state_;
	amp_t *d_state_;
	gpu::stream_t stream_;
	size_t num_amps_;
};

class StatevectorGPUIOSimulator : public Simulator {
	virtual void init(size_t num_qubits) override;
	virtual void deinit() override;
	virtual void run(const std::vector<QuantumCircuit> &circs) override;

	private:
	amp_t *temp_buf_;
	std::vector<amp_t*> state_pair_;
	std::vector<amp_t*> d_states_;
	size_t num_amps_;
	size_t nbuf_;
	size_t iobuf_elems_;
	gpu::stream_t iostream_;
	size_t ioblock_;
};
   */

} // namespace snuqs
