#pragma once

#include <vector>
#include <memory>
#include <ostream>


#include "configure.hpp"

#include "gpu_utils.hpp"

namespace snuqs {

class Gate {
    friend std::ostream& operator<<(std::ostream &os, const Gate &gate); 

    public:
	enum class Type {
		ID,
		H,
		X,
		Y,
		Z,
		SX,
		SY,
		S,
		SDG,
		T,
		TDG,

		RX,
		RY,
		RZ,
		U1,
		U2,
		U3,

		SWAP,
		CX,
		CY,
		CZ,
		CH,
		CRX,
		CRY,
		CRZ,
		CU1,
		CU2,
		CU3,
	};


    using shared_ptr = std::shared_ptr<Gate>;

    Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);

	Gate::Type type() const;
	const std::vector<size_t>& qubits() const;
	const std::vector<double>& params() const;

    std::ostream& operator<<(std::ostream &os) const;

    virtual void apply(snuqs::amp_t *state, size_t num_amps) const;
    virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const;

    protected:
	Gate::Type type_;
	std::vector<size_t> qubits_;
	std::vector<double> params_;
    
};

class IDGate : public Gate {
	public:
	IDGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class HGate : public Gate {
	public:
	HGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class XGate : public Gate {
	public:
	XGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class YGate : public Gate {
	public:
	YGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class ZGate : public Gate {
	public:
	ZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class SXGate : public Gate {
	public:
	SXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class SYGate : public Gate {
	public:
	SYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class SGate : public Gate {
	public:
	SGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class SDGGate : public Gate {
	public:
	SDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class TGate : public Gate {
	public:
	TGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class TDGGate : public Gate {
	public:
	TDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class RXGate : public Gate {
	public:
	RXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class RYGate : public Gate {
	public:
	RYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class RZGate : public Gate {
	public:
	RZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class U1Gate : public Gate {
	public:
	U1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class U2Gate : public Gate {
	public:
	U2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class U3Gate : public Gate {
	public:
	U3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class SWAPGate : public Gate {
	public:
	SWAPGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CXGate : public Gate {
	public:
	CXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CYGate : public Gate {
	public:
	CYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CZGate : public Gate {
	public:
	CZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CHGate : public Gate {
	public:
	CHGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CRXGate : public Gate {
	public:
	CRXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CRYGate : public Gate {
	public:
	CRYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CRZGate : public Gate {
	public:
	CRZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CU1Gate : public Gate {
	public:
	CU1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CU2Gate : public Gate {
	public:
	CU2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

class CU3Gate : public Gate {
	public:
	CU3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<double> &params);
	virtual void apply(snuqs::amp_t *state, size_t num_amps) const override;
	virtual void applyGPU(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s) const override;
};

Gate::shared_ptr gateFactory(const std::string &name,
		const std::vector<size_t> &&qubits,
		const std::vector<double> &&params);

} // namespace snuqs
