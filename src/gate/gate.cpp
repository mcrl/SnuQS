#include "gate.hpp"
#include "gate_cpu.hpp"
#include "gate_gpu.hpp"

#include <algorithm>
#include <set>
#include <cassert>
#include <iostream>

namespace snuqs {

//
// static methods
//

namespace {

Gate::Type stringToType(const std::string &s) {
	if (s == "id") {
		return Gate::Type::ID;
	} else if (s == "h") {
		return Gate::Type::H;
	} else if (s == "x") {
		return Gate::Type::X;
	} else if (s == "y") {
		return Gate::Type::Y;
	} else if (s == "z") {
		return Gate::Type::Z;
	} else if (s == "sx") {
		return Gate::Type::SX;
	} else if (s == "sy") {
		return Gate::Type::SY;
	} else if (s == "s") {
		return Gate::Type::S;
	} else if (s == "sdg") {
		return Gate::Type::SDG;
	} else if (s == "t") {
		return Gate::Type::T;
	} else if (s == "tdg") {
		return Gate::Type::TDG;
	} else if (s == "rx") {
		return Gate::Type::RX;
	} else if (s == "ry") {
		return Gate::Type::RY;
	} else if (s == "rz") {
		return Gate::Type::RZ;
	} else if (s == "u1") {
		return Gate::Type::U1;
	} else if (s == "u2") {
		return Gate::Type::U2;
	} else if ((s == "U") || (s == "u") || (s == "u3")) {
		return Gate::Type::U3;
	} else if (s == "fsim") {
		return Gate::Type::FSIM;
	} else if (s == "swap") {
		return Gate::Type::SWAP;
	} else if ((s == "CX") || (s == "cx")) {
		return Gate::Type::CX;
	} else if (s == "cy") {
		return Gate::Type::CY;
	} else if (s == "cz") {
		return Gate::Type::CZ;
	} else if (s == "ch") {
		return Gate::Type::CH;
	} else if (s == "crx") {
		return Gate::Type::CRX;
	} else if (s == "cry") {
		return Gate::Type::CRY;
	} else if (s == "crz") {
		return Gate::Type::CRZ;
	} else if (s == "cu1") {
		return Gate::Type::CU1;
	} else if (s == "cu2") {
		return Gate::Type::CU2;
	} else if (s == "cu3") {
		return Gate::Type::CU3;
	} else if (s == "zerostate") {
		return Gate::Type::ZEROSTATE;
	} else if (s == "uniform") {
		return Gate::Type::UNIFORM;
	} else if (s == "block") {
		return Gate::Type::BLOCK;
	} else if (s == "fusion") {
		return Gate::Type::FUSION;
	} else if (s == "placeholder") {
		return Gate::Type::PLACEHOLDER;
	} else if (s == "nswap") {
		return Gate::Type::NSWAP;
	} else {
		assert(false);
	}
}

std::string typeToString(Gate::Type t) {
	switch (t) {
		case Gate::Type::ID:  return "id";
		case Gate::Type::H:   return "h";
		case Gate::Type::X:   return "x";
		case Gate::Type::Y:   return "y";
		case Gate::Type::Z:   return "z";
		case Gate::Type::SX:  return "sx";
		case Gate::Type::SY:  return "sy";
		case Gate::Type::S:   return "s";
		case Gate::Type::SDG: return "sdg";
		case Gate::Type::T:   return "t";
		case Gate::Type::TDG: return "tdg";
		case Gate::Type::RX:  return "rx";
		case Gate::Type::RY:  return "ry";
		case Gate::Type::RZ:  return "rz";
		case Gate::Type::U1:  return "u1";
		case Gate::Type::U2:  return "u2";
		case Gate::Type::U3:  return "u3";
		case Gate::Type::FSIM:  return "fsim";
		case Gate::Type::SWAP:  return "swap";
		case Gate::Type::CX:  return "cx";
		case Gate::Type::CY:  return "cy";
		case Gate::Type::CZ:  return "cz";
		case Gate::Type::CH:  return "ch";
		case Gate::Type::CRX: return "crx";
		case Gate::Type::CRY: return "cry";
		case Gate::Type::CRZ: return "crz";
		case Gate::Type::CU1: return "cu1";
		case Gate::Type::CU2: return "cu2";
		case Gate::Type::CU3: return "cu3";

		case Gate::Type::ZEROSTATE: return "zerostate";
		case Gate::Type::UNIFORM: return "uniform";

		case Gate::Type::FUSION: return "fusion";
		case Gate::Type::BLOCK: return "block";

		case Gate::Type::PLACEHOLDER: return "placeholder";
		case Gate::Type::NSWAP:  return "nswap";

		default: assert(false);
	}
}

} // static methods

//
// Gate
//
Gate::Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: type_(type),
  qubits_(qubits),
  params_(params) {
#ifdef __CUDA_ARCH__
	  for (size_t d = 1; d < gpu::getNumDevices(); d++) {
		  gpu::setDevice(d);
		  gpu::Malloc(&gpu_gate_[d], sizeof(GPUGate));

		  GPUGate gpu_gate;
		  gpu_gate.type = type_;
		  gpu_gate.qubits = nullptr;
		  gpu_gate.nqubits = qubits_.size();
		  gpu_gate.params = nullptr;
		  gpu_gate.nparams = params_.size();

		  if (gpu_gate.nqubits > 0) {
			  gpu::Malloc(&gpu_gate.qubits, sizeof(size_t) * gpu_gate.nqubits);
			  gpu::MemcpyAsyncH2D(gpu_gate.qubits, &qubits_[0], sizeof(size_t) * gpu_gate.nqubits);
		  }

		  if (gpu_gate.nparams > 0) {
			  gpu::Malloc(&gpu_gate.params, sizeof(size_t) * gpu_gate.nparams);
			  gpu::MemcpyAsyncH2D(gpu_gate.params, &params_[0], sizeof(size_t) * gpu_gate.nparams);
		  }

		  gpu::MemcpyAsyncH2D(gpu_gate_[d], &gpu_gate, sizeof(GPUGate));
	  }
#endif
}

Gate::Type Gate::type() const {
	return type_;
}

std::string Gate::name() const {
	return typeToString(type_);
}

const std::vector<size_t>& Gate::qubits() const {
	return qubits_;
}

void Gate::set_qubits(const std::vector<size_t> &q) {
	qubits_ = q;
}

const std::vector<real_t>& Gate::params() const {
	return params_;
}

GPUGate* Gate::gpu_gate(int d) const {
	return gpu_gate_[d];
}

double Gate::flops() const {
	size_t n = (1ul << qubits_.size());
	return (6 * n + 2 * (n-1));
}

void Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	assert(false);
}

void Gate::apply(amp_t *state, size_t num_amps) const {
	apply(state, num_amps, -1);
}

void Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	assert(false);
}

void Gate::applyGPU(amp_t *state, size_t num_amps, gpu::stream_t s) const {
	applyGPU(state, num_amps, -1, s);
}

bool Gate::diagonal() const {
	return false;
}

std::ostream& Gate::operator<<(std::ostream &os) const {
	os << typeToString(type_);

	if (!params_.empty()) {
		os << "(";
		for (size_t i = 0; i < params_.size(); i++) {
			os << params_[i];
			if (i+1 != params_.size()) {
				os << ",";
			}
		}
		os << ")";
	}

	os << " ";
	for (size_t i = 0; i < qubits_.size(); i++) {
		os << qubits_[i];
		if (i+1 != qubits_.size()) {
			os << " ";
		}
	}
	return os;
}

Tensor* Gate::toTensor() const {
	switch (type_) {
		case Gate::Type::ID:  
			return new Tensor({'i'}, {'j'}, {1., 0.,
											0., 1.});
		case Gate::Type::H:
			return new Tensor({'i'}, {'j'}, {M_SQRT1_2, M_SQRT1_2,
											 M_SQRT1_2, -M_SQRT1_2});
		case Gate::Type::X:
			return new Tensor({'i'}, {'j'}, {0., 1.,
											1., 0.});
		case Gate::Type::Y:
			return new Tensor({'i'}, {'j'}, {0., {0., -1.},
											 {0., 1.}, 0.});
		case Gate::Type::Z:
			return new Tensor({'i'}, {'j'}, {1., 0.,
											 0., -1.});
		case Gate::Type::SX:
			return new Tensor({'i'}, {'j'}, {{0.5, 0.5}, {0.5, -0.5},
											{0.5, -0.5}, {0.5, 0.5}});
		case Gate::Type::SY:
			return new Tensor({'i'}, {'j'}, {{0.5, 0.5}, {0.5, 0.5},
											{-0.5, -0.5}, {0.5, 0.5}});
		case Gate::Type::S:
			return new Tensor({'i'}, {'j'}, {1., 0.,
											0., {0, 1}});
		case Gate::Type::SDG:
			return new Tensor({'i'}, {'j'}, {1., 0.,
											0., {0, -1}});
		case Gate::Type::T:
			return new Tensor({'i'}, {'j'}, {1., 0.,
											 0., {M_SQRT1_2, M_SQRT1_2}});
		case Gate::Type::TDG:
			return new Tensor({'i'}, {'j'}, {1., 0.,
											0., {M_SQRT1_2, -M_SQRT1_2}});
		case Gate::Type::RX:
			return new Tensor({'i'}, {'j'}, {
					{cos(params_[0]/2), 0}, {0, -sin(params_[0]/2)},
					{0, -sin(params_[0]/2)}, {cos(params_[0]/2), 0}
					});
		case Gate::Type::RY:
			return new Tensor({'i'}, {'j'}, {
					{cos(params_[0]/2), 0}, {-sin(params_[0]/2), 0},
					{sin(params_[0]/2), 0}, {cos(params_[0]/2), 0}
					});
		case Gate::Type::RZ:
			return new Tensor({'i'}, {'j'}, {
					{cos(params_[0]/2), -sin(params_[0]/2)}, 0,
					0, {cos(params_[0]/2), sin(params_[0]/2)},
					});
		case Gate::Type::U1:
			return new Tensor({'i'}, {'j'}, {
					1., 0.,
					0., {cos(params_[0]), sin(params_[0])}
			});
		case Gate::Type::U2:
			return new Tensor({'i'}, {'j'}, {
					M_SQRT1_2,
					{M_SQRT1_2 * -cos(params_[1]), M_SQRT1_2 * -sin(params_[1])},
					{M_SQRT1_2 * cos(params_[0]), M_SQRT1_2 * sin(params_[0])},
					{M_SQRT1_2 * cos(params_[0]+params_[1]), M_SQRT1_2 * sin(params_[0]+params_[1])}
			});
		case Gate::Type::U3:
			return new Tensor({'i'}, {'j'}, {
					{cos(params_[0]/2), 0},
					{-cos(params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)},
					{cos(params_[1])*sin(params_[0]/2), sin(params_[1])*sin(params_[0]/2)},
					{cos(params_[1] + params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)}
			});
		case Gate::Type::FSIM:
			{
				real_t theta_rads = M_PI * params_[0];
				real_t phi_rads = M_PI * params_[1];

				snuqs::amp_t c1, c2, c3, c4, d;
				c1 = snuqs::amp_t(0.5) * (thrust::exp(snuqs::amp_t(0.0, -1 * phi_rads / 2)) + snuqs::amp_t(std::cos(theta_rads), 0.0));
				c2 = snuqs::amp_t(0.5) * (thrust::exp(snuqs::amp_t(0.0, -1 * phi_rads / 2)) - snuqs::amp_t(std::cos(theta_rads), 0.0));
				c3 = snuqs::amp_t(0.0, -1 / 2.) * snuqs::amp_t(std::sin(theta_rads), 0.0);
				c4 = c3;
				snuqs::amp_t s1, s2, s3, s4;
				s1 = thrust::sqrt(c1);
				s2 = thrust::sqrt(c2);
				s3 = thrust::sqrt(c3);
				s4 = thrust::sqrt(c4);
				d = thrust::exp(snuqs::amp_t(0.0, phi_rads / 4.));

				snuqs::amp_t 
				b0000, b0001, b0010, b0011, b0100, b0101, b0110, b0111,
				b1000, b1001, b1010, b1011, b1100, b1101, b1110, b1111,
				a000, a001, a010, a011, a020, a021, a030, a031,
				a100, a101, a110, a111, a120, a121, a130, a131;
				a000 = d * s1;
				a001 = 0;
				a010 = d * s2;
				a011 = 0;
				a020 = 0;
				a021 = s3;
				a030 = 0;
				a031 = snuqs::amp_t{0.0, 1.0} * s4;
				a100 = 0;
				a101 = conj(d) * s1;
				a110 = 0;
				a111 = -conj(d) * s2;
				a120 = s3;
				a121 = 0;
				a130 = snuqs::amp_t{0.0, -1.0} * s4;
				a131 = 0.;

				b0000 = a000 * a000 + a010 * a010;
				b0001 = a000 * a000 + a010 * a011;
				b0010 = a000 * a100 + a010 * a110;
				b0011 = a000 * a100 + a010 * a111;
				b0100 = a001 * a000 + a011 * a010;
				b0101 = a001 * a000 + a011 * a011;
				b0110 = a001 * a100 + a011 * a110;
				b0111 = a001 * a100 + a011 * a111;
				b1000 = a100 * a000 + a110 * a010;
				b1001 = a100 * a000 + a110 * a011;
				b1010 = a100 * a100 + a110 * a110;
				b1011 = a100 * a100 + a110 * a111;
				b1100 = a101 * a000 + a111 * a010;
				b1101 = a101 * a000 + a111 * a011;
				b1110 = a101 * a100 + a111 * a110;
				b1111 = a101 * a100 + a111 * a111;

				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						b0000, b0001, b0010, b0011,
						b0100, b0101, b0110, b0111,
						b1000, b1001, b1010, b1011,
						b1100, b1101, b1110, b1111,
						});
			}
		case Gate::Type::SWAP:
			return new Tensor({'a', 'b'}, {'c', 'd'}, {
					1, 0, 0, 0,
					0, 0, 1, 0,
					0, 1, 0, 0,
					0, 0, 0, 1
			});
		case Gate::Type::CX: 
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 0, 0, 1,
						0, 0, 1, 0,
						0, 1, 0, 0
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1,
						0, 0, 1, 0
						});
			}
		case Gate::Type::CY:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 0, 0, {0, -1},
						0, 0, 1, 0,
						0, {0, 1}, 0, 0
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, {0, -1},
						0, 0, {0, 1}, 0
						});
			}
		case Gate::Type::CZ:
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, -1
						});
		case Gate::Type::CH:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, M_SQRT1_2, 0, M_SQRT1_2,
						0, 0, 1, 0,
						0, M_SQRT1_2, 0, -M_SQRT1_2
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 0, 0, 0,
						0, 0, M_SQRT1_2, M_SQRT1_2,
						0, 0, M_SQRT1_2, -M_SQRT1_2
						});
			}
		case Gate::Type::CRX:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
					1, 0, 0, 0,
						0, {cos(params_[0]/2), 0}, 0, {0, -sin(params_[0]/2)},
						0, 0, 1, 0,
						0, {0, -sin(params_[0]/2)}, 0,  {cos(params_[0]/2), 0}
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
					1, 0, 0, 0, 
					0, 1, 0, 0, 
						0, 0, {cos(params_[0]/2), 0}, {0, -sin(params_[0]/2)},
						0, 0, {0, -sin(params_[0]/2)}, {cos(params_[0]/2), 0}
						});
			}
		case Gate::Type::CRY:
			if (qubits_[0] < qubits_[1]) {
			return new Tensor({'a', 'b'}, {'c', 'd'}, {
				1, 0, 0, 0,
					0, {cos(params_[0]/2), 0}, 0, {-sin(params_[0]/2), 0},
					0, 0, 1, 0,
					0, {sin(params_[0]/2), 0}, 0,  {cos(params_[0]/2), 0}
					});
			} else {
			return new Tensor({'a', 'b'}, {'c', 'd'}, {
				1, 0, 0, 0, 
				0, 1, 0, 0, 
					0, 0, {cos(params_[0]/2), 0}, {-sin(params_[0]/2), 0},
					0, 0, {sin(params_[0]/2), 0}, {cos(params_[0]/2), 0}
					});
			}
		case Gate::Type::CRZ:
			if (qubits_[0] < qubits_[1]) {
			return new Tensor({'a', 'b'}, {'c', 'd'}, {
				1, 0, 0, 0,
					0, {cos(params_[0]/2), -sin(params_[0]/2)}, 0, 0,
					0, 0, 1, 0,
					0, 0, {cos(params_[0]/2), sin(params_[0]/2)}, 0, 
					});
			} else {
			return new Tensor({'a', 'b'}, {'c', 'd'}, {
				1, 0, 0, 0, 
				0, 1, 0, 0, 
					0, 0, {cos(params_[0]/2), -sin(params_[0]/2)}, 0,
					0, 0, 0, {cos(params_[0]/2), sin(params_[0]/2)},
					});
			}
		case Gate::Type::CU1:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0, 
						0, 1., 0, 0.,
						0, 0, 1, 0, 
						0, 0., 0, {cos(params_[0]), sin(params_[0])}
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0, 
						0, 1, 0, 0, 
						0, 0, 1., 0.,
						0, 0, 0., {cos(params_[0]), sin(params_[0])}
						});
			}
		case Gate::Type::CU2:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, M_SQRT1_2, 0, {M_SQRT1_2 * -cos(params_[1]), M_SQRT1_2 * -sin(params_[1])},
						0, 0, 1, 0,
						0, {M_SQRT1_2 * cos(params_[0]), M_SQRT1_2 * sin(params_[0])},
						0, {M_SQRT1_2 * cos(params_[0]+params_[1]), M_SQRT1_2 * sin(params_[0]+params_[1])}
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, M_SQRT1_2, {M_SQRT1_2 * -cos(params_[1]), M_SQRT1_2 * -sin(params_[1])},
						0, 0, {M_SQRT1_2 * cos(params_[0]), M_SQRT1_2 * sin(params_[0])}, {M_SQRT1_2 * cos(params_[0]+params_[1]), M_SQRT1_2 * sin(params_[0]+params_[1])}
						});
			}
		case Gate::Type::CU3:
			if (qubits_[0] < qubits_[1]) {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
					1, 0, 0, 0,
					0, {cos(params_[0]/2), 0}, 0, {-cos(params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)},
					0, 0, 1, 0, 
					0, {cos(params_[1])*sin(params_[0]/2), sin(params_[1])*sin(params_[0]/2)}, 0, {cos(params_[1] + params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)}
						});
			} else {
				return new Tensor({'a', 'b'}, {'c', 'd'}, {
					1, 0, 0, 0,
					0, 1, 0, 0,
						0, 0, {cos(params_[0]/2), 0}, {-cos(params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)},
						0, 0, {cos(params_[1])*sin(params_[0]/2), sin(params_[1])*sin(params_[0]/2)}, {cos(params_[1] + params_[2])*sin(params_[0]/2), sin(params_[2])*sin(params_[0]/2)}
						});
			}
		default:
			assert(false);
	return new Tensor({'i'}, {'j'}, {1., 0., 0., 1.});
	}
	return new Tensor({'i'}, {'j'}, {1., 0., 0., 1.});

}

std::ostream &operator<<(std::ostream &os, const Gate &gate) 
{
    return gate.operator<<(os);
}


//
// IDGate
//
IDGate::IDGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void IDGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::idgate(state, num_amps, qubits_, params_);
}

void IDGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::idgate(state, num_amps, s, qubits_, params_);
}

//
// HGate
//
HGate::HGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void HGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::hgate(state, num_amps, qubits_, params_);
}

void HGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::hgate(state, num_amps, s, qubits_, params_);
}

//
// XGate
//
XGate::XGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void XGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::xgate(state, num_amps, qubits_, params_);
}

void XGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::xgate(state, num_amps, s, qubits_, params_);
}

//
// YGate
//
YGate::YGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void YGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::ygate(state, num_amps, qubits_, params_);
}

void YGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::ygate(state, num_amps, s, qubits_, params_);
}

//
// ZGate
//
ZGate::ZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void ZGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::zgate(state, num_amps, qubits_, params_);
}

void ZGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::zgate(state, num_amps, s, qubits_, params_);
}

//
// SXGate
//
SXGate::SXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void SXGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::sxgate(state, num_amps, qubits_, params_);
}

void SXGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::sxgate(state, num_amps, s, qubits_, params_);
}

//
// SYGate
//
SYGate::SYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void SYGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::sygate(state, num_amps, qubits_, params_);
}

void SYGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::sygate(state, num_amps, s, qubits_, params_);
}

//
// SGate
//
SGate::SGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void SGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::sgate(state, num_amps, qubits_, params_);
}

void SGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::sgate(state, num_amps, s, qubits_, params_);
}

//
// SDGGate
//
SDGGate::SDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void SDGGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::sdggate(state, num_amps, qubits_, params_);
}

void SDGGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::sdggate(state, num_amps, s, qubits_, params_);
}

//
// TGate
//
TGate::TGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void TGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	size_t tgt = qubits_[0];
	if ((1ul << tgt) & mask) {
		cpu::tgate(state, num_amps, qubits_, params_);
	}
}

void TGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	size_t tgt = qubits_[0];
	if ((1ul << tgt) & mask) {
		gpu::tgate(state, num_amps, s, qubits_, params_);
	}
}

bool TGate::diagonal() const {
	return true;
}

double TGate::flops() const {
	return 3;
}

//
// TDGGate
//
TDGGate::TDGGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void TDGGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::tdggate(state, num_amps, qubits_, params_);
}

void TDGGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::tdggate(state, num_amps, s, qubits_, params_);
}

//
// RXGate
//
RXGate::RXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void RXGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::rxgate(state, num_amps, qubits_, params_);
}

void RXGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::rxgate(state, num_amps, s, qubits_, params_);
}

//
// RYGate
//
RYGate::RYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void RYGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::rygate(state, num_amps, qubits_, params_);
}

void RYGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::rygate(state, num_amps, s, qubits_, params_);
}

//
// RZGate
//
RZGate::RZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void RZGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::rzgate(state, num_amps, qubits_, params_);
}

void RZGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::rzgate(state, num_amps, s, qubits_, params_);
}

//
// U1Gate
//
U1Gate::U1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void U1Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::u1gate(state, num_amps, qubits_, params_);
}

void U1Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::u1gate(state, num_amps, s, qubits_, params_);
}

//
// U2Gate
//
U2Gate::U2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void U2Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::u2gate(state, num_amps, qubits_, params_);
}

void U2Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::u2gate(state, num_amps, s, qubits_, params_);
}

//
// U3Gate
//
U3Gate::U3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void U3Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::u3gate(state, num_amps, qubits_, params_);
}

void U3Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::u3gate(state, num_amps, s, qubits_, params_);
}

//
// SWAPGate
//
SWAPGate::SWAPGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void SWAPGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::swapgate(state, num_amps, qubits_, params_);
}

void SWAPGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::swapgate(state, num_amps, s, qubits_, params_);
}

//
// CXGate
//
CXGate::CXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CXGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::cxgate(state, num_amps, qubits_, params_);
}

void CXGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::cxgate(state, num_amps, s, qubits_, params_);
}

//
// CYGate
//
CYGate::CYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CYGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::cygate(state, num_amps, qubits_, params_);
}

void CYGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::cygate(state, num_amps, s, qubits_, params_);
}

//
// CZGate
//
CZGate::CZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CZGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	size_t tgt0 = qubits_[0];
	size_t tgt1 = qubits_[1];
	if (((1ul << tgt0) & mask) && ((1ul << tgt1) & mask)) {
		cpu::czgate(state, num_amps, qubits_, params_);
	}
}

void CZGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	size_t tgt0 = qubits_[0];
	size_t tgt1 = qubits_[1];
	if (((1ul << tgt0) & mask) && ((1ul << tgt1) & mask)) {
		gpu::czgate(state, num_amps, s, qubits_, params_);
	} 
}

bool CZGate::diagonal() const {
	return true;
}

double CZGate::flops() const {
	return 0.5;
}

//
// CHGate
//
CHGate::CHGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CHGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::chgate(state, num_amps, qubits_, params_);
}

void CHGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::chgate(state, num_amps, s, qubits_, params_);
}

//
// CRXGate
//
CRXGate::CRXGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CRXGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::crxgate(state, num_amps, qubits_, params_);
}

void CRXGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::crxgate(state, num_amps, s, qubits_, params_);
}

//
// CRYGate
//
CRYGate::CRYGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CRYGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::crygate(state, num_amps, qubits_, params_);
}

void CRYGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::crygate(state, num_amps, s, qubits_, params_);
}

//
// CRZGate
//
CRZGate::CRZGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CRZGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::crzgate(state, num_amps, qubits_, params_);
}

void CRZGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::crzgate(state, num_amps, s, qubits_, params_);
}

//
// CU1Gate
//
CU1Gate::CU1Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CU1Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::cu1gate(state, num_amps, qubits_, params_);
}

void CU1Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::cu1gate(state, num_amps, s, qubits_, params_);
}

bool CU1Gate::diagonal() const {
	return true;
}


//
// CU2Gate
//
CU2Gate::CU2Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CU2Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::cu2gate(state, num_amps, qubits_, params_);
}

void CU2Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::cu2gate(state, num_amps, s, qubits_, params_);
}

//
// CU3Gate
//
CU3Gate::CU3Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void CU3Gate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::cu3gate(state, num_amps, qubits_, params_);
}

void CU3Gate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::cu3gate(state, num_amps, s, qubits_, params_);
}

//
// ZeroGate
//
ZeroGate::ZeroGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void ZeroGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::zerostate(state, num_amps, qubits_, params_);
}

void ZeroGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::zerostate(state, num_amps, s, qubits_, params_);
}

//
// UniformGate
//
UniformGate::UniformGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void UniformGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::uniform(state, num_amps, qubits_, params_);
}

void UniformGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::uniform(state, num_amps, s, qubits_, params_);
}

//
// FusionGate
//
FusionGate::FusionGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void FusionGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	assert(false);
}

void FusionGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	assert(false);
}

std::ostream& FusionGate::operator<<(std::ostream &os) const {
	assert(false);
	return os;
}

//
// BlockGate
//
BlockGate::BlockGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void BlockGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	cpu::blockgate(state, num_amps, gates_);
}

void BlockGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::blockgate(state, num_amps, s, gpu_gate_[gpu::getDevice()]);
}

std::ostream& BlockGate::operator<<(std::ostream &os) const {
	os << typeToString(type_) << " (" <<  gates_.size() << " gates, " << qubits_.size()<< " effective qubits)\n";
	for (auto q : perm_) {
		os << q << " ";
	}
	os << "\n";
	for (size_t i = 0; i < gates_.size(); i++) {
		auto &&g = gates_[i];
		os << "|---- " << *g;
		if (i+1 < gates_.size())
			os << "\n";
	}
	return os;
}

double BlockGate::flops() const {
	size_t flops = 0;
	for (auto g : gates_) {
		flops += g->flops();
	}
	return flops;
}

void BlockGate::addGates(const std::vector<Gate::shared_ptr> &gates) {
#ifdef __CUDA_ARCH__
	for (int d = 0; d < gpu::getNumDevices(); d++) {
		gpu::setDevice(d);

		GPUGate gpugate;
		gpu::MemcpyAsyncD2H(&gpugate, gpu_gate_[d], sizeof(GPUGate));
		gpu::deviceSynchronize();


		gpugate.nqubits = qubits_.size();
		if (gpugate.qubits != nullptr)
			gpu::Free(gpugate.qubits);
		gpu::Malloc(&gpugate.qubits, sizeof(size_t) * gpugate.nqubits);
		gpu::MemcpyAsyncH2D(gpugate.qubits, &qubits_[0], sizeof(size_t) * gpugate.nqubits);

		gpugate.ngates = gates.size();
		gpu::Malloc(&gpugate.gates, sizeof(GPUGate) * gpugate.ngates);

		std::set<size_t> equbits;
		equbits.insert(qubits_.begin(), qubits_.end());

		size_t q = 0;
		while (true) {
			if (equbits.find(q) == equbits.end()) {
				equbits.insert(q);
			}
			if (equbits.size() >= (gpu::WARPSHIFT+gpu::MAX_CACHE))
				break;
			q++;
		}

		perm_.resize(MAX_INMEM);
		for (size_t i = 0; i < MAX_INMEM; i++) {
			perm_[i] = i;
		}
		std::vector<size_t> inhole;
		std::vector<size_t> outhole;
		for (size_t i = 0; i < gpu::SMEM_SHIFT; i++) {
			if (equbits.find(i) == equbits.end())
				inhole.push_back(i);
		}

		for (size_t i = gpu::SMEM_SHIFT; i < MAX_INMEM; i++) {
			if (equbits.find(i) != equbits.end())
				outhole.push_back(i);
		}

		assert(inhole.size() == outhole.size());
		for (size_t i = 0; i < inhole.size(); i++) {
			std::swap(perm_[inhole[i]], perm_[outhole[i]]);
		}


		gpu::Malloc(&gpugate.perm, sizeof(size_t) * MAX_INMEM);
		gpu::MemcpyAsyncH2D(gpugate.perm, &perm_[0], sizeof(size_t) * MAX_INMEM);

		for (size_t i = 0; i < gpugate.ngates; i++) {
			std::vector<size_t> qubits = gates[i]->qubits();
			std::vector<real_t> params = gates[i]->params();

			for (size_t i = 0; i < qubits.size(); i++) {
				if (qubits[i] < MAX_INMEM) {
					qubits[i] = perm_[qubits[i]];
				}
			}

			if (d == 0) {
				gates_.push_back(gateFactory(typeToString(gates[i]->type()),
							std::move(qubits),
							std::move(params)));
			}

			gpu::MemcpyAsyncH2D(&gpugate.gates[i], gates_[i]->gpu_gate(d), sizeof(GPUGate));
		}

		gpu::MemcpyAsyncH2D(gpu_gate_[d], &gpugate, sizeof(GPUGate));
	}
#endif
}

//
// PlaceHoder
//
PlaceHolder::PlaceHolder(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
}

void PlaceHolder::apply(amp_t *state, size_t num_amps, size_t mask) const {
	// Do nothing
}

void PlaceHolder::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	// Do nothing
}

std::ostream& PlaceHolder::operator<<(std::ostream &os) const {
	os << typeToString(type_);
	for (auto q : qubits_) {
		std::cout << " " << q;
	} 
	return os;
}

double PlaceHolder::flops() const {
	return 0.;
}

//
// NSWAPGate
//
NSWAPGate::NSWAPGate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: Gate(type, qubits, params) {
#ifdef __CUDA_ARCH__
	std::vector<size_t> sorted = qubits;

	std::sort(sorted.begin(), sorted.end());
	for (int d = 0; d < gpu::getNumDevices(); d++) {
		gpu::setDevice(d);
		GPUGate gpu_gate;

		gpu::MemcpyAsyncD2H(&gpu_gate, gpu_gate_[d], sizeof(GPUGate));
		gpu::deviceSynchronize();

		gpu::Malloc(&gpu_gate.perm, sizeof(size_t) * sorted.size());
		gpu::MemcpyAsyncH2D(gpu_gate.perm, &sorted[0], sizeof(size_t) * sorted.size());

		gpu::MemcpyAsyncH2D(gpu_gate_[d], &gpu_gate, sizeof(GPUGate));
	}
#endif
}

void NSWAPGate::apply(amp_t *state, size_t num_amps, size_t mask) const {
	assert(false);
	//cpu::nswapgate(state, num_amps, qubits_, params_);
}

void NSWAPGate::applyGPU(amp_t *state, size_t num_amps, size_t mask, gpu::stream_t s) const {
	gpu::nswapgate(state, num_amps, s, qubits_.size(), gpu_gate_[gpu::getDevice()]);
}

//
// gateFactory
//
Gate::shared_ptr gateFactory(std::string name,
		std::vector<size_t> qubits,
		std::vector<real_t> params) {
	switch (stringToType(name)) {
		case Gate::Type::ID:   return std::make_shared<IDGate>(stringToType(name), qubits, params);
		case Gate::Type::H:    return std::make_shared<HGate>(stringToType(name), qubits, params);
		case Gate::Type::X:    return std::make_shared<XGate>(stringToType(name), qubits, params);
		case Gate::Type::Y:    return std::make_shared<YGate>(stringToType(name), qubits, params);
		case Gate::Type::Z:    return std::make_shared<ZGate>(stringToType(name), qubits, params);
		case Gate::Type::SX:   return std::make_shared<SXGate>(stringToType(name), qubits, params);
		case Gate::Type::SY:   return std::make_shared<SYGate>(stringToType(name), qubits, params);
		case Gate::Type::S:    return std::make_shared<SGate>(stringToType(name), qubits, params);
		case Gate::Type::SDG:  return std::make_shared<SDGGate>(stringToType(name), qubits, params);
		case Gate::Type::T:    return std::make_shared<TGate>(stringToType(name), qubits, params);
		case Gate::Type::TDG:  return std::make_shared<TDGGate>(stringToType(name), qubits, params);
		case Gate::Type::RX:   return std::make_shared<RXGate>(stringToType(name), qubits, params);
		case Gate::Type::RY:   return std::make_shared<RYGate>(stringToType(name), qubits, params);
		case Gate::Type::RZ:   return std::make_shared<RZGate>(stringToType(name), qubits, params);
		case Gate::Type::U1:   return std::make_shared<U1Gate>(stringToType(name), qubits, params);
		case Gate::Type::U2:   return std::make_shared<U2Gate>(stringToType(name), qubits, params);
		case Gate::Type::U3:   return std::make_shared<U3Gate>(stringToType(name), qubits, params);
		case Gate::Type::FSIM: assert(false); 
		case Gate::Type::SWAP: return  std::make_shared<SWAPGate>(stringToType(name), qubits, params);
		case Gate::Type::CX:   return std::make_shared<CXGate>(stringToType(name), qubits, params);
		case Gate::Type::CY:   return std::make_shared<CYGate>(stringToType(name), qubits, params);
		case Gate::Type::CZ:   return std::make_shared<CZGate>(stringToType(name), qubits, params);
		case Gate::Type::CH:   return std::make_shared<CHGate>(stringToType(name), qubits, params);
		case Gate::Type::CRX:  return std::make_shared<CRXGate>(stringToType(name), qubits, params);
		case Gate::Type::CRY:  return std::make_shared<CRYGate>(stringToType(name), qubits, params);
		case Gate::Type::CRZ:  return std::make_shared<CRZGate>(stringToType(name), qubits, params);
		case Gate::Type::CU1:  return std::make_shared<CU1Gate>(stringToType(name), qubits, params);
		case Gate::Type::CU2:  return std::make_shared<CU2Gate>(stringToType(name), qubits, params);
		case Gate::Type::CU3:  return std::make_shared<CU3Gate>(stringToType(name), qubits, params);

		case Gate::Type::ZEROSTATE: return std::make_shared<ZeroGate>(stringToType(name), qubits, params);
		case Gate::Type::UNIFORM: return std::make_shared<UniformGate>(stringToType(name), qubits, params);

		case Gate::Type::FUSION: return std::make_shared<FusionGate>(stringToType(name), qubits, params);
		case Gate::Type::BLOCK: return std::make_shared<BlockGate>(stringToType(name), qubits, params);

		case Gate::Type::PLACEHOLDER: return std::make_shared<PlaceHolder>(stringToType(name), qubits, params);
		case Gate::Type::NSWAP:return std::make_shared<NSWAPGate>(stringToType(name), qubits, params);
	}

	assert(false);
	return std::make_shared<IDGate>(Gate::Type::ID, qubits, params);
}


} // namespace snuqs
