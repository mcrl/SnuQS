#pragma once

#include <vector>
#include <memory>
#include <ostream>


#include "configure.hpp"

#include "gpu_utils.hpp"

#include "tensor.hpp"

namespace snuqs {

struct GPUGate;

class Gate {
    friend std::ostream& operator<<(std::ostream &os, const Gate &gate); 

    public:
	enum class Type : std::size_t {
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
		FSIM,
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

		ZEROSTATE,
		UNIFORM,

		FUSION,
		BLOCK,

		PLACEHOLDER,
		NSWAP,

	};


    using shared_ptr = std::shared_ptr<Gate>;

    Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);

	Gate::Type type() const;
	std::string name() const;
	const std::vector<size_t>& qubits() const;
	void set_qubits(const std::vector<size_t> &q);
	const std::vector<real_t>& params() const;
	GPUGate* gpu_gate(int d) const;

	virtual double flops() const;
	virtual bool diagonal() const;

    virtual std::ostream& operator<<(std::ostream &os) const;

    void apply(amp_t *state, size_t num_amps) const;
    virtual void apply(amp_t *state, size_t num_amps, size_t mask) const;
    virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const;
    void applyGPU(amp_t *state, size_t num_amps, gpu::stream_t s) const;

    virtual Tensor* toTensor() const;

    protected:
	Gate::Type type_;
	std::vector<size_t> qubits_;
	std::vector<real_t> params_;

	snuqs::GPUGate *gpu_gate_[gpu::MAX_NGPU];
    
};

struct blockParams {
	Gate::Type types[256];
	size_t ntypes;
};

struct GPUGate {
	Gate::Type type;
	size_t *qubits;
	size_t *mapped;
	size_t nqubits;
	real_t *params;
	size_t nparams;
	GPUGate *gates;
	size_t ngates;
	size_t *perm;
};


class IDGate : public Gate {
	public:
	IDGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class HGate : public Gate {
	public:
	HGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class XGate : public Gate {
	public:
	XGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class YGate : public Gate {
	public:
	YGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class ZGate : public Gate {
	public:
	ZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class SXGate : public Gate {
	public:
	SXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class SYGate : public Gate {
	public:
	SYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class SGate : public Gate {
	public:
	SGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class SDGGate : public Gate {
	public:
	SDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class TGate : public Gate {
	public:
	TGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual bool diagonal() const override;
	virtual double flops() const override;
};

class TDGGate : public Gate {
	public:
	TDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class RXGate : public Gate {
	public:
	RXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class RYGate : public Gate {
	public:
	RYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class RZGate : public Gate {
	public:
	RZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class U1Gate : public Gate {
	public:
	U1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class U2Gate : public Gate {
	public:
	U2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class U3Gate : public Gate {
	public:
	U3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class SWAPGate : public Gate {
	public:
	SWAPGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CXGate : public Gate {
	public:
	CXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CYGate : public Gate {
	public:
	CYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CZGate : public Gate {
	public:
	CZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual bool diagonal() const override;
	virtual double flops() const override;
};

class CHGate : public Gate {
	public:
	CHGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CRXGate : public Gate {
	public:
	CRXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CRYGate : public Gate {
	public:
	CRYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CRZGate : public Gate {
	public:
	CRZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CU1Gate : public Gate {
	public:
	CU1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual bool diagonal() const override;
};

class CU2Gate : public Gate {
	public:
	CU2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class CU3Gate : public Gate {
	public:
	CU3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class ZeroGate : public Gate {
	public:
	ZeroGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class UniformGate : public Gate {
	public:
	UniformGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

class FusionGate : public Gate {
	public:
	FusionGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual std::ostream& operator<<(std::ostream &os) const override;

	private:
	std::vector<Gate::shared_ptr> gates_;
};

class BlockGate : public Gate {
	public:
	BlockGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual std::ostream& operator<<(std::ostream &os) const override;
	virtual double flops() const override;


	void addGates(const std::vector<Gate::shared_ptr> &gates);

	private:
	std::vector<size_t> perm_;
	std::vector<Gate::shared_ptr> gates_;
};

class PlaceHolder : public Gate {
	public:
	PlaceHolder(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
	virtual std::ostream& operator<<(std::ostream &os) const override;
	virtual double flops() const override;
};

class NSWAPGate : public Gate {
	public:
	NSWAPGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params);
	virtual void apply(amp_t *state, size_t num_amps, size_t mask) const override;
	virtual void applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const override;
};

Gate::shared_ptr gateFactory(std::string name,
		std::vector<size_t> qubits,
		std::vector<real_t> params);

} // namespace snuqs
