#include "gate.h"

#include <algorithm>
#include <set>
#include <cassert>


#include "gate_factory.h"
#include "gate_cpu_impl.h"
#include "gate_cuda.h"

namespace snuqs {

//
// Gate
//
Gate::Gate(Gate::Type type, const std::vector<size_t> &qubits, const std::vector<real_t> &params)
: type_(type),
  qubits_(qubits),
  params_(params) {
    /*
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
*/
}

Gate::Type Gate::type() const {
	return type_;
}

std::string Gate::name() const {
	return GateFactory::TypeToString(type_);
}

void Gate::set_qubits(const std::vector<size_t> &q) {
	qubits_ = q;
}

const std::vector<size_t>& Gate::qubits() const {
	return qubits_;
}

const std::vector<real_t>& Gate::params() const {
	return params_;
}

bool Gate::diagonal() const {
  switch (type_) {
    case Gate::Type::ID: 
    case Gate::Type::Z: 
    case Gate::Type::S: 
    case Gate::Type::SDG: 
    case Gate::Type::T: 
    case Gate::Type::TDG: 
    case Gate::Type::RZ: 
    case Gate::Type::U1: 
    case Gate::Type::U2: 
    case Gate::Type::CZ: 
    case Gate::Type::CRZ: 
    case Gate::Type::CU1: 
      return true;
    default:
      return false;
  }
  return false;
}


void Gate::AddSubgate(std::shared_ptr<Gate> gate) {
  subgates_.push_back(gate);
}

std::ostream& Gate::operator<<(std::ostream &os) const {
	os << name(); 

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
	if (qubits_.size() == 0) {
    os << "all ";
  } else {
    for (size_t i = 0; i < qubits_.size(); i++) {
      os << qubits_[i];
      if (i+1 != qubits_.size()) {
        os << " ";
      }
    }
  }
	return os;
}

std::ostream &operator<<(std::ostream &os, const Gate &gate) 
{
    return gate.operator<<(os);
}

void Gate::Apply(amp_t *state, size_t num_amps) const {
	std::unique_ptr<GateInterface> op = std::make_unique<GateCPUImpl>();

  switch (type_) {
    //
    // Normal gates
    //
    case Gate::Type::ID: 
      op->idgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::H: 
      op->hgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::X: 
      op->xgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::Y: 
      op->ygate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::Z: 
      op->zgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::SX: 
      op->sxgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::SY: 
      op->sygate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::S: 
      op->sgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::SDG: 
      op->sdggate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::T: 
      op->tgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::TDG: 
      op->tdggate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::RX: 
      op->rxgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::RY: 
      op->rygate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::RZ: 
      op->rzgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::U1: 
      op->u1gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::U2: 
      op->u2gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::U3: 
      op->u3gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::SWAP: 
      op->swapgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CX: 
      op->cxgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CY: 
      op->cygate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CZ: 
      op->czgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CH: 
      op->chgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CRX: 
      op->crxgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CRY: 
      op->crygate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CRZ: 
      op->crzgate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CU1: 
      op->cu1gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CU2: 
      op->cu2gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CU3: 
      op->cu3gate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::CCX: 
      op->ccxgate(state, num_amps, qubits(), params());
      break;

    // 
    // Measurement
    //
    case Gate::Type::MEASURE: 
      op->measure(state, num_amps, qubits(), params());
      break;

    // 
    // Initializers
    //
    case Gate::Type::ZEROSTATE: 
      op->zerostate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::UNIFORM: 
      op->uniform(state, num_amps, qubits(), params());
      break;

    //
    // Optimized gates
    //
      /*
    case Gate::Type::FUSION: 
      op->fusiongate(state, num_amps, qubits(), params());
      break;
    case Gate::Type::BLOCK: 
      op->blockgate(state, num_amps, subgates_);
      break;
    case Gate::Type::NSWAP: 
      op->nswapgate(state, num_amps, qubits(), params());
      break;
      */

    // Special gates
    case Gate::Type::PLACEHOLDER: 
      /* Do nothing */
      break;
  }
}

} // namespace snuqs
