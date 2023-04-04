#include "gate_cuda.h"

#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cooperative_groups.h>

#include <memory>
#include <complex>
#include <algorithm>
#include <vector>
#include <cassert>


#include "types.h"
#include "constants.h"


#include "gpu_utils.h"

using namespace cooperative_groups;

namespace snuqs {

GateCUDAImpl::~GateCUDAImpl() {
  assert(false);
}

void GateCUDAImpl::idgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::hgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::xgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::ygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::zgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::sxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::sygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::sgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::sdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::tgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::tdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::rxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::rygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::rzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::u1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::u2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::u3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::swapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::cxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::cygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::czgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::chgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::crxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::crygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::crzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::cu1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::cu2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::cu3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::ccxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::measure(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}


void GateCUDAImpl::init(amp_t *buf, size_t num_amps) {
  assert(false);
}

void GateCUDAImpl::zerostate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::uniform(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}


void GateCUDAImpl::fusiongate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::nswapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
  assert(false);
}

void GateCUDAImpl::blockgate(amp_t *state, size_t num_amps, const std::vector<Gate::shared_ptr> &gates) {
  assert(false);
}


/*
namespace kernel {

__device__ void group_apply(amp_t *state, size_t start, size_t step, size_t num_amps, GPUGate *gate, amp_t *amps);

__global__ void init(amp_t *state)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	state[i] = 0;
	if (i == 0)
		state[i] = 1;
}

__global__ void zero(amp_t *state)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	state[i] = 0;
}

__global__ void gate_zerostate(amp_t *state)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	state[i] = 0;
	if (i == 0)
		state[i] = 1;
}

__global__ void gate_uniform(amp_t *state, real_t val)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	state[i] = val;
}

template<typename F0, typename F1>
__global__ void gate_unitary_l(amp_t *state, size_t num_amps, size_t tgt, F0 f0, F1 f1)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int mask = (1u << tgt);

	amp_t a0 = state[i];
	real_t real = __shfl_xor_sync(0xffffffff, a0.real(), mask, 0);
	real_t imag = __shfl_xor_sync(0xffffffff, a0.imag(), mask, 0);
	amp_t a1 = {real, imag};

	state[i] = (threadIdx.x & mask) ? f1(a1, a0) : f0(a0, a1);
}

template<typename F0, typename F1>
__global__ void gate_unitary(amp_t *state, size_t num_amps, size_t tgt, F0 f0, F1 f1)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t s = (1ul << tgt);
	size_t idx = ((i >> tgt) << (tgt+1)) + (i & (s-1));

	amp_t a0 = state[idx];
	amp_t a1 = state[idx+s];
	state[idx] = f0(a0, a1);
	state[idx+s] = f1(a0, a1);
}

template<typename F>
__global__ void gate_phase(amp_t *state, size_t num_amps, size_t tgt, F f)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t s = (1ul << tgt);
	size_t idx = ((i >> tgt) << (tgt+1)) + (i & (s-1));

	amp_t a = state[idx+s];
	state[idx+s] = f(a);
}

__global__ void gate_swap(amp_t *state, size_t num_amps, size_t tgt0, size_t tgt1)
{
	if (tgt0 > tgt1) {
		size_t tmp = tgt0;
		tgt0 = tgt1;
		tgt1 = tmp;
	}

	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);

	size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

	amp_t a0 = state[idx+st0];
	amp_t a1 = state[idx+st1];
	state[idx+st0] = a1;
	state[idx+st1] = a0;
}

template<typename F0, typename F1>
__global__ void gate_controlled_unitary(amp_t *state, size_t num_amps, size_t ctrl, size_t tgt, F0 f0, F1 f1)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t cst = 1ul << ctrl;
	size_t st = st0 + st1;

	size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

	amp_t a0 = state[idx+cst];
	amp_t a1 = state[idx+st];
	state[idx+cst] = f0(a0, a1);
	state[idx+st] = f1(a0, a1);
}

template<typename F0, typename F1>
__global__ void gate_controlled_unitary_tgt0(amp_t *state, size_t num_amps, size_t ctrl, size_t tgt, F0 f0, F1 f1)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t cst = 1ul << ctrl;
	size_t st = st0 + st1;

	size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

	amp_t a0 = state[idx+cst];
	amp_t a1 = state[idx+st];

	// 
	// Assuming blockDim == 256, we have:
	// thread 0  1  2  3  ... 255
	//        a0 a2 a4 a6 ... a510
	//        a1 a3 a5 a7 ... a511
	// We transpose this into:
	// thread 0  1  2  3  ... 255
	//        b0 b1 b2 b3 ... b255
	//        b256        ... b511
	// hoping for coalesced write.
	//

	extern __shared__ amp_t shmem[];
	shmem[threadIdx.x * 2 + 0] = f0(a0, a1);
	shmem[threadIdx.x * 2 + 1] = f1(a0, a1);
	__syncthreads();

	amp_t b0 = shmem[threadIdx.x];
	i = blockDim.x * blockIdx.x + threadIdx.x / 2;
	ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));
	state[idx+(threadIdx.x % 2 == 0 ? cst : st)] = b0;

	amp_t b1 = shmem[blockDim.x + threadIdx.x];
	i = blockDim.x * blockIdx.x + blockDim.x / 2 + threadIdx.x / 2;
	ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));
	state[idx+(threadIdx.x % 2 == 0 ? cst : st)] = b1;
}

template<typename F>
__global__ void gate_controlled_phase(amp_t *state, size_t num_amps, size_t ctrl, size_t tgt, F f)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t st = st0 + st1;

	size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
	size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

	state[idx+st] = f(state[idx+st]);

}

template<typename F>
__global__ void gate_all_amplitudes(amp_t *state, size_t num_amps, F f)
{
	// tgt < ctrl
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	state[i] = f(state[i]);
}

__device__ void hadamard(amp_t *state, size_t num_amps, size_t tgt)
{
	size_t s = (1ul << tgt);
	for (size_t i = threadIdx.x; i < num_amps/2; i += blockDim.x) {
		size_t idx = ((i >> tgt) << (tgt+1)) + (i & (s-1));

		amp_t a0 = state[idx];
		amp_t a1 = state[idx+s];
		state[idx] = ((real_t)M_SQRT1_2) * (a0 + a1);
		state[idx+s] = ((real_t)M_SQRT1_2) * (a0 - a1);
	}
}

__device__ void cx(amp_t *state, size_t num_amps, size_t ctrl, size_t tgt)
{
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t cst = 1ul << ctrl;
	size_t st = st0 + st1;
	for (size_t i = threadIdx.x; i < num_amps/4; i += blockDim.x) {
		size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
		size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

		amp_t a0 = state[idx+cst];
		amp_t a1 = state[idx+st];
		state[idx+cst] = a1;
		state[idx+st] = a0;
	}
}

template<typename F0, typename F1>
__device__ void group_unitary(amp_t *state, size_t tgt, F0 f0, F1 f1)
{
	size_t s = (1ul << tgt);
	for (size_t i = threadIdx.x; i < (SMEM_NELEM >> 1); i += blockDim.x) {
		size_t idx = ((i >> tgt) << (tgt+1)) + (i & (s-1));

		amp_t a0 = state[idx];
		amp_t a1 = state[idx+s];
		state[idx] = f0(a0, a1);
		state[idx+s] = f1(a0, a1);
	}
}

template<typename F>
__device__ void group_phase(amp_t *state, size_t tgt, F f)
{
	size_t s = (1ul << tgt);
	for (size_t i = threadIdx.x; i < (SMEM_NELEM >> 1); i += blockDim.x) {
		size_t idx = ((i >> tgt) << (tgt+1)) + (i & (s-1));

		amp_t a = state[idx+s];
		state[idx+s] = f(a);
	}
}

template<typename F>
__device__ void group_all(amp_t *state, F f)
{
	for (size_t i = threadIdx.x; i < SMEM_NELEM; i += blockDim.x) {
		state[i] = f(state[i]);
	}
}

template<typename F0, typename F1>
__device__ void group_controlled_unitary(amp_t *state, size_t ctrl, size_t tgt, F0 f0, F1 f1)
{
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t cst = 1ul << ctrl;
	size_t st = st0 + st1;
	for (size_t i = threadIdx.x; i < (SMEM_NELEM>>2); i += blockDim.x) {
		size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
		size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

		amp_t a0 = state[idx+cst];
		amp_t a1 = state[idx+st];
		state[idx+cst] = f0(a0, a1);
		state[idx+st] = f1(a0, a1);
	}
}

template<typename F>
__device__ void group_controlled_phase(amp_t *state, size_t ctrl, size_t tgt, F f)
{
	size_t tgt0 = (ctrl < tgt) ? ctrl : tgt;
	size_t tgt1 = (ctrl > tgt) ? ctrl : tgt;
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	size_t st = st0 + st1;

	for (size_t i = threadIdx.x; i < (SMEM_NELEM>>2); i += blockDim.x) {
		size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
		size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));
		state[idx+st] = f(state[idx+st]);
	}
}

template<typename F0, typename F1>
__device__ void apply_unitary(
		amp_t *state,
		size_t tgt,
		F0 f0, 
		F1 f1) {
	group_unitary(state, tgt, f0, f1);
}

template<typename F>
__device__ void apply_phase(
		amp_t *state,
		size_t tgt,
		F f
		) {
	if (tgt < MAX_INMEM) {
		group_phase(state, tgt, f);
	} else {
		group_all(state, f);
	}
}

template<typename F>
__device__ void apply_tphase(
		amp_t *state,
		size_t tgt,
		F f,
		size_t base,
		size_t *perm
		) {
	size_t st = (1ul << tgt);
	if (st < SMEM_NELEM) {
		for (size_t i = threadIdx.x; i < (SMEM_NELEM >> 1); i += blockDim.x) {
			size_t idx = ((i >> tgt) << (tgt+1)) + (i & (st-1));

			amp_t a = state[idx+st];
			state[idx+st] = f(a);
		}
	} else {
		for (size_t j = threadIdx.x; j < SMEM_NELEM; j += blockDim.x) {
			if ((((base + j + 32) - 1) & (1ul << perm[tgt])) != 0) {
				state[j] = f(state[j]);
			}
		}
	}
}

template<typename F0, typename F1>
__device__ void apply_controlled_unitary(
		amp_t *state,
		size_t ctrl,
		size_t tgt,
		F0 f0,
		F1 f1) {
	group_controlled_unitary(state, ctrl, tgt, f0, f1);
}

template<typename F>
__device__ void apply_controlled_phase(
		amp_t *state,
		size_t ctrl,
		size_t tgt,
		F f) {
	if (ctrl < tgt) {
		size_t tmp = tgt;
		tgt = ctrl;
		ctrl = tmp;
	}
	if (ctrl < MAX_INMEM) {
		group_controlled_phase(state, ctrl, tgt, f);
	} else if (tgt < MAX_INMEM) {
		apply_phase(state, tgt, f);
	} else {
		group_all(state, f);
	}
}

__device__ void group_hgate(amp_t *state, GPUGate *gate)
{
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 + a1) * ((real_t)M_SQRT1_2);
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 - a1) * ((real_t)M_SQRT1_2);
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_xgate(amp_t *state, GPUGate *gate)
{
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a0;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_ygate(amp_t *state, GPUGate *gate)
{
	amp_t coef = {0, 1};
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return -coef*a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef*a0;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_zgate(amp_t *state, GPUGate *gate)
{
	auto f = [=] __host__ __device__ (amp_t a) {
		return -a;
	};

	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_sxgate(amp_t *state, GPUGate *gate) {
	amp_t coef0 = {0, -1};
	amp_t coef1 = {-1, 1};
//	amp_t coef0 = {0.5, 0.5};
//	amp_t coef1 = {0.5, -0.5};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return ((real_t)M_SQRT1_2) * (a0 + coef0 * a1);
//		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return ((real_t)M_SQRT1_2) * (coef0*a0 + a1);
//		return coef1 * a0 + coef0 * a1;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_sygate(amp_t *state, GPUGate *gate) {
//	amp_t coef = {0.5, 0.5};

	auto f0 = [=] __host__ __device__ (
			amp_t a0,
			amp_t a1) {
//		return coef * (a0 + a1);
		return ((real_t)M_SQRT1_2) * (a0 - a1);
	};

	auto f1 = [=] __host__ __device__ (
			amp_t a0,
			amp_t a1) {
//		return coef * (a1 - a0);
		return ((real_t)M_SQRT1_2) * (a0 + a1);
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_sgate(amp_t *state, GPUGate *gate) {
	amp_t coef = {0, 1};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};
	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_sdggate(amp_t *state, GPUGate *gate) {
	auto f = [=] __host__ __device__ (amp_t a) {
		amp_t coef = {0, -1};
		return coef * a;
	};
	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_tgate(amp_t *state, GPUGate *gate) {
	amp_t coef = {((real_t)M_SQRT1_2), ((real_t)M_SQRT1_2)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};
	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_tdggate(amp_t *state, GPUGate *gate) {
	amp_t coef = {((real_t)M_SQRT1_2), -((real_t)M_SQRT1_2)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};
	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_rxgate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {0, -sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_rygate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {sin(theta/2), 0};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 - coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_rzgate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	amp_t coef1 = {cos(theta/2), sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a1;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_u1gate(amp_t *state, GPUGate *gate) {
	real_t lambda = gate->params[0];
	amp_t coef = {cos(lambda), sin(lambda)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};
	apply_phase(state, gate->qubits[0], f);
}

__device__ void group_u2gate(amp_t *state, GPUGate *gate) {
	real_t phi = gate->params[0];
	real_t lambda = gate->params[1];
	amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_u3gate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	real_t phi = gate->params[1];
	real_t lambda = gate->params[2];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	apply_unitary(state, gate->qubits[0], f0, f1);
}

__device__ void group_swapgate(amp_t *state, GPUGate *gate) {
	size_t tgt0 = (gate->qubits[0] < gate->qubits[1]) ? gate->qubits[0] : gate->qubits[1];
	size_t tgt1 = (gate->qubits[0] > gate->qubits[1]) ? gate->qubits[0] : gate->qubits[1];
	size_t st0 = (1ul << tgt0);
	size_t st1 = (1ul << tgt1);
	for (size_t i = threadIdx.x; i < (SMEM_NELEM>>2); i += blockDim.x) {
		size_t ci = ((i >> tgt0) << (tgt0+1)) + (i & (st0-1));
		size_t idx = ((ci >> tgt1) << (tgt1+1)) + (ci & (st1-1));

		amp_t a0 = state[idx+st0];
		amp_t a1 = state[idx+st1];
		state[idx+st0] = a1;
		state[idx+st1] = a0;
	}
}

__device__ void group_cxgate(amp_t *state, GPUGate *gate)
{
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a1;
	};
	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a0;
	};

	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_cygate(amp_t *state, GPUGate *gate) {
	amp_t coef = {0, 1};
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return -coef*a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef*a0;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_czgate(amp_t *state, GPUGate *gate) {
	auto f = [=] __host__ __device__ (amp_t a) {
		return -a;
	};
	apply_controlled_phase(state, gate->qubits[0], gate->qubits[1], f);
}

__device__ void group_chgate(amp_t *state, GPUGate *gate) {
	real_t coef = ((real_t)M_SQRT1_2);
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 + a1) * coef;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 - a1) * coef;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_crxgate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {0, -sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_crygate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {sin(theta/2), 0};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 - coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_crzgate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	amp_t coef1 = {cos(theta/2), sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a1;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_cu1gate(amp_t *state, GPUGate *gate) {
	real_t lambda = gate->params[0];
	amp_t coef = {cos(lambda), sin(lambda)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};
	apply_controlled_phase(state, gate->qubits[0], gate->qubits[1], f);
}

__device__ void group_cu2gate(amp_t *state, GPUGate *gate) {
	real_t phi = gate->params[0];
	real_t lambda = gate->params[1];
	amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_cu3gate(amp_t *state, GPUGate *gate) {
	real_t theta = gate->params[0];
	real_t phi = gate->params[1];
	real_t lambda = gate->params[2];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	apply_controlled_unitary(state, gate->qubits[0], gate->qubits[1], f0, f1);
}

__device__ void group_nswap(amp_t *state, GPUGate *gate) {
	for (size_t i = threadIdx.x; i < (SMEM_NELEM>>gate->nqubits); i += blockDim.x) {
		for (int j = 0; j < gate->nqubits; j++) {
			size_t tgt = gate->perm[j];
			size_t st = (1ul << tgt);
			i = ((i >> tgt) << (tgt+1)) + (i & (st - 1));
		}

		for (size_t j = 0; j < (1ul << gate->nqubits); j++) {
			size_t from_mask = 0;
			size_t to_mask = 0;
			for (size_t k = 0; k < gate->nqubits; k++) {
				size_t bit = ((j & (1ul << k)) >> k);
				from_mask |= (bit << gate->qubits[k]);
				to_mask |= (bit << gate->qubits[(k+1)%gate->nqubits]);
			}
			size_t from_addr = i + from_mask;
			size_t to_addr = i + to_mask;

			if (from_addr < to_addr) {
				amp_t amp = state[from_addr];
				state[from_addr] = state[to_addr];
				state[to_addr] = amp;
			}
		}
	}
}


__device__ bool applyable(size_t start, size_t step, GPUGate *gate) {
	if (gate->type == Gate::Type::T) {
		size_t tgt = gate->qubits[0];
		return ((1ul << tgt) < SMEM_NELEM) || (((start + SMEM_NELEM - 1) & (1ul << tgt)) != 0);
	} else if (gate->type == Gate::Type::CZ) {
		size_t st0 = (1ul << max(gate->qubits[0], gate->qubits[1]));
		size_t st1 = (1ul << min(gate->qubits[0], gate->qubits[1]));
		size_t mask = (start + SMEM_NELEM-1);

		return (st0 < SMEM_NELEM)
			|| ((st1 < SMEM_NELEM) && (mask & st0))
			|| ((st1 >= SMEM_NELEM) && (mask & st0) && (mask & st1));
	} else {
		return true;
	}
}

__device__ void group_gate(amp_t *state, GPUGate *gate)
{
	switch (gate->type) {
		case Gate::Type::ID:
			break;
		case Gate::Type::H:
			group_hgate(state, gate);
			break;
		case Gate::Type::X:
			group_xgate(state, gate);
			break;
		case Gate::Type::Y:
			group_ygate(state, gate);
			break;
		case Gate::Type::Z:
			group_zgate(state, gate);
			break;
		case Gate::Type::SX:
			group_sxgate(state, gate);
			break;
		case Gate::Type::SY:
			group_sygate(state, gate);
			break;
		case Gate::Type::S:
			group_sgate(state, gate);
			break;
		case Gate::Type::SDG:
			group_sdggate(state, gate);
			break;
		case Gate::Type::T:
			group_tgate(state, gate);
			break;
		case Gate::Type::TDG:
			group_tdggate(state, gate);
			break;
		case Gate::Type::RX:
			group_rxgate(state, gate);
			break;
		case Gate::Type::RY:
			group_rygate(state, gate);
			break;
		case Gate::Type::RZ:
			group_rzgate(state, gate);
			break;
		case Gate::Type::U1:
			group_u1gate(state, gate);
			break;
		case Gate::Type::U2:
			group_u2gate(state, gate);
			break;
		case Gate::Type::U3:
			group_u3gate(state, gate);
			break;
		case Gate::Type::SWAP:
			group_swapgate(state, gate);
			break;
		case Gate::Type::CX:
			group_cxgate(state, gate);
			break;
		case Gate::Type::CY:
			group_cygate(state, gate);
			break;
		case Gate::Type::CZ:
			group_czgate(state, gate);
			break;
		case Gate::Type::CH:
			group_chgate(state, gate);
			break;
		case Gate::Type::CRX:
			group_crxgate(state, gate);
			break;
		case Gate::Type::CRY:
			group_crygate(state, gate);
			break;
		case Gate::Type::CRZ:
			group_crzgate(state, gate);
			break;
		case Gate::Type::CU1:
			group_cu1gate(state, gate);
			break;
		case Gate::Type::CU2:
			group_cu2gate(state, gate);
			break;
		case Gate::Type::CU3:
			group_cu3gate(state, gate);
			break;
		case Gate::Type::NSWAP:
			group_nswap(state, gate);
			break;
		default:
			assert(false);
			break;
	}
}

__global__ void gate_block(amp_t *state, size_t num_amps, const GPUGate *gate)
{
	__shared__ amp_t amps[SMEM_NELEM];

	size_t slimit = min(num_amps, SMEM_NELEM);
	for (size_t s = blockIdx.x * SMEM_NELEM;
				s < num_amps;
				s += gridDim.x * SMEM_NELEM) {
		for (size_t j = threadIdx.x; j < slimit; j += blockDim.x) {
			size_t idx = s + j;
			for (size_t k = WARP_SHIFT; k < SMEM_SHIFT; k++) {
				if (k != gate->perm[k]) {
					size_t q = gate->perm[k];
					size_t m0 = (idx & (1ul << k));
					size_t m1 = (idx & (1ul << q)) >> (q - k);
					size_t b = (m0 ^ m1);
					idx = idx ^ ((b << (q - k)) | b);
				}
			}
			amps[j] = state[idx];
		}
		__syncthreads();

		for (size_t j = 0; j < gate->ngates; j++) {
			group_gate(amps, &gate->gates[j]);
			__syncthreads();
		}

		for (size_t j = threadIdx.x; j < slimit; j += blockDim.x) {
			size_t idx = s + j;
			for (size_t k = WARP_SHIFT; k < SMEM_SHIFT; k++) {
				if (k != gate->perm[k]) {
					size_t q = gate->perm[k];
					size_t m0 = (idx & (1ul << k));
					size_t m1 = (idx & (1ul << q)) >> (q - k);
					size_t b = (m0 ^ m1);
					idx = idx ^ ((b << (q - k)) | b);
				}
			}
			state[idx] = amps[j];
		}
		__syncthreads();
	}
}

__global__ void gate_nswap(amp_t *state, size_t num_amps, const GPUGate *gate) 
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < gate->nqubits; i++) {
		size_t tgt = gate->perm[i];
		size_t st = (1ul << tgt);
		idx = ((idx >> tgt) << (tgt+1)) + (idx & (st - 1));
	}

	for (size_t i = 0; i < (1ul << gate->nqubits); i++) {
		size_t from_mask = 0;
		size_t to_mask = 0;
		for (size_t j = 0; j < gate->nqubits; j++) {
			from_mask |= (((i & (1ul << j)) >> j) << gate->qubits[j]);
			to_mask |= (((i & (1ul << j)) >> j) << gate->qubits[(j+1)%gate->nqubits]);
		}
		size_t from_addr = idx + from_mask;
		size_t to_addr = idx + to_mask;

		if (from_addr < to_addr) {
			amp_t amp = state[from_addr];
			state[from_addr] = state[to_addr];
			state[to_addr] = amp;
		}
	}
}

}

namespace gpu {

namespace {
template<typename F0, typename F1>
void applyUnitary(
		amp_t *state,
		size_t num_amps,
		size_t tgt,
		cudaStream_t s,
		F0 f0, 
		F1 f1) {
	//	if (tgt < 5) {
	//		size_t nthreads = NTHREADS;
	//		size_t nblocks = std::max(1ul, num_amps / NTHREADS);
	//		kernel::gate_unitary_l<<<nblocks, nthreads, 0, s>>>(state, num_amps, tgt, f0, f1);
	//	} else {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / NTHREADS / 2);
	kernel::gate_unitary<<<nblocks, nthreads, 0, s>>>(state, num_amps, tgt, f0, f1);
	//	}
	gpu::checkLastError();
}

template<typename F>
void applyAllAmptitudes(
		amp_t *state,
		size_t num_amps,
		cudaStream_t s,
		F f) {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / NTHREADS);
	kernel::gate_all_amplitudes<<<nblocks, nthreads, 0, s>>>(
			state,
			num_amps,
			f);
	gpu::checkLastError();
}

template<typename F>
void applyPhase(
		amp_t *state,
		size_t num_amps,
		size_t tgt,
		cudaStream_t s,
		F f) {

	if (tgt < MAX_INMEM) {
		size_t nthreads = NTHREADS;
		size_t nblocks = std::max(1ul, num_amps / NTHREADS / 2);
		kernel::gate_phase<<<nblocks, nthreads, 0, s>>> ((amp_t*)state, num_amps, tgt, f);
		gpu::checkLastError();
	} else {
		applyAllAmptitudes(state, num_amps, s, f);
	}
}

template<typename F0, typename F1>
void applyControlledUnitary(
		amp_t *state,
		size_t num_amps,
		size_t ctrl,
		size_t tgt,
		cudaStream_t s,
		F0 f0, 
		F1 f1) {
	if (tgt == 0) {
		size_t nthreads = NTHREADS;
		size_t nblocks = std::max(1ul, num_amps / NTHREADS / 4);
		kernel::gate_controlled_unitary_tgt0<<<nblocks, nthreads, 2 * nthreads * sizeof(amp_t), s>>>(
				state,
				num_amps,
				ctrl,
				tgt,
				f0,
				f1);
		gpu::checkLastError();
	} else {
		size_t nthreads = NTHREADS;
		size_t nblocks = std::max(1ul, num_amps / NTHREADS / 4);
		kernel::gate_controlled_unitary<<<nblocks, nthreads, 0, s>>>(
				state,
				num_amps,
				ctrl,
				tgt,
				f0,
				f1);
		gpu::checkLastError();
	}
}

template<typename F>
void applyControlledPhase(
		amp_t *state,
		size_t num_amps,
		size_t ctrl,
		size_t tgt,
		cudaStream_t s,
		F f) {
	if (ctrl < tgt) {
		size_t tmp = tgt;
		tgt = ctrl;
		ctrl = tmp;
	}
	if (ctrl < MAX_INMEM) {
		// 1. tgt < ctrl < GPU_NQUBIT
		size_t nthreads = NTHREADS;
		size_t nblocks = std::max(1ul, num_amps / NTHREADS / 4);
		kernel::gate_controlled_phase<<<nblocks, nthreads, 0, s>>>(
				state,
				num_amps,
				ctrl, 
				tgt,
				f);
		gpu::checkLastError();
	} else if (tgt < MAX_INMEM) {
		// 2. tgt < GPU_NQUBIT <= GPU_NQUBIT
		applyPhase(state, num_amps, tgt, s, f);
	} else {
		// 3. GPU_NQUBIT <= tgt < ctrl
		applyAllAmptitudes(state, num_amps, s, f);
	}
}

}

void init(amp_t *state, size_t count, cudaStream_t s) {
	size_t namps = count;
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, namps / NTHREADS);
	kernel::init<<<nblocks, nthreads, 0, s>>>(state);
	gpu::checkLastError();
}

void zero(amp_t *state, size_t count, cudaStream_t s) {
	size_t namps = count;
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, namps / NTHREADS);
	kernel::zero<<<nblocks, nthreads, 0, s>>>(state);
	gpu::checkLastError();
}

void zerostate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / NTHREADS);
	kernel::gate_zerostate<<<nblocks, nthreads, 0, s>>>(state);
	gpu::checkLastError();
}

void uniform(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) { 
	size_t nqubit = tgts.size();
	real_t val = 1.;
	for (size_t i = 0; i < nqubit; i++) {
		val *= ((real_t)M_SQRT1_2);
	}

	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / NTHREADS);
	kernel::gate_uniform<<<nblocks, nthreads, 0, s>>>(state, val);
	gpu::checkLastError();
}

void idgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	// Do nothing 
}

void hgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t coef = ((real_t)M_SQRT1_2);
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 + a1) * coef;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 - a1) * coef;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void xgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a0;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void ygate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {0, 1};
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return -coef*a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef*a0;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void zgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	auto f = [=] __host__ __device__ (amp_t a) {
		return -a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void sxgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef0 = {0.5, 0.5};
	amp_t coef1 = {0.5, -0.5};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void sygate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {0.5, 0.5};

	auto f0 = [=] __host__ __device__ (
			amp_t a0,
			amp_t a1) {
		return coef * (a0 + a1);
	};

	auto f1 = [=] __host__ __device__ (
			amp_t a0,
			amp_t a1) {
		return coef * (a1 - a0);
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void sgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {0, 1};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void sdggate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	auto f = [=] __host__ __device__ (amp_t a) {
		amp_t coef = {0, -1};
		return coef * a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void tgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {((real_t)M_SQRT1_2), ((real_t)M_SQRT1_2)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void tdggate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {((real_t)M_SQRT1_2), -((real_t)M_SQRT1_2)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void rxgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {0, -sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void rygate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {sin(theta/2), 0};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 - coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void rzgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	amp_t coef1 = {cos(theta/2), sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void u1gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t lambda = params[0];
	amp_t coef = {cos(lambda), sin(lambda)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};

	applyPhase(state, num_amps, tgts[0], s, f);
}

void u2gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t phi = params[0];
	real_t lambda = params[1];
	amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void u3gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	real_t phi = params[1];
	real_t lambda = params[2];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};

	applyUnitary(state, num_amps, tgts[0], s, f0, f1);
}

void swapgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / NTHREADS / 4);
	kernel::gate_swap<<<nblocks, nthreads, 0, s>>>(
			state,
			num_amps,
			tgts[0],
			tgts[1]);
	gpu::checkLastError();
}

void cxgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a1;
	};
	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return a0;
	};

	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void cygate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	amp_t coef = {0, 1};
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return -coef*a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef*a0;
	};

	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void czgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	auto f = [=] __host__ __device__ (amp_t a) {
		return -a;
	};

	applyControlledPhase(state, num_amps, tgts[0], tgts[1], s, f);
}

void chgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t coef = ((real_t)M_SQRT1_2);
	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 + a1) * coef;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return (a0 - a1) * coef;
	};
	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void crxgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {0, -sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};
	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void crygate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {sin(theta/2), 0};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 - coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a0 + coef0 * a1;
	};

	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void crzgate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	amp_t coef1 = {cos(theta/2), sin(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef1 * a1;
	};
	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void cu1gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t lambda = params[0];
	amp_t coef = {cos(lambda), sin(lambda)};
	auto f = [=] __host__ __device__ (amp_t a) {
		return coef * a;
	};

	applyControlledPhase(state, num_amps, tgts[0], tgts[1], s, f);
}

void cu2gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t phi = params[0];
	real_t lambda = params[1];
	amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);

}

void cu3gate(amp_t *state, size_t num_amps, cudaStream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	real_t theta = params[0];
	real_t phi = params[1];
	real_t lambda = params[2];
	amp_t coef0 = {cos(theta/2), 0};
	amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};

	auto f0 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef0 * a0 + coef1 * a1;
	};

	auto f1 = [=] __host__ __device__ (amp_t a0, amp_t a1) {
		return coef2 * a0 + coef3 * a1;
	};
	applyControlledUnitary(state, num_amps, tgts[0], tgts[1], s, f0, f1);
}

void blockgate(amp_t *state, size_t num_amps, cudaStream_t s, const GPUGate *gate) {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, num_amps / SMEM_NELEM);
	kernel::gate_block<<<nblocks, nthreads, 0, s>>>(state, num_amps, gate);
	gpu::checkLastError();
}

void nswapgate(amp_t *state, size_t num_amps, cudaStream_t s, size_t nqubits, const GPUGate *gate) {
	size_t nthreads = NTHREADS;
	size_t nblocks = std::max(1ul, (num_amps >> nqubits) / NTHREADS);
	kernel::gate_nswap<<<nblocks, nthreads, 0, s>>>(state, num_amps, gate);
	gpu::checkLastError();
}

} // namespace gpu
*/
} // namespace snuqs
