#pragma once


#include "configure.hpp"
#include "gpu_utils.hpp"
#include "gate.hpp"

namespace snuqs {
namespace gpu {

	void init(snuqs::amp_t *buf, size_t count, gpu::stream_t s);
	void zero(snuqs::amp_t *buf, size_t count, gpu::stream_t s);

	void zerostate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void uniform(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);

	void idgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void hgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void xgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void ygate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void zgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void sxgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void sygate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void sgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void sdggate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void tgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void tdggate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void rxgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void rygate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void rzgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void u1gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void u2gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void u3gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void swapgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void cxgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void cygate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void czgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void chgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void crxgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void crygate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void crzgate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void cu1gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void cu2gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void cu3gate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);

	void fusiongate(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const std::vector<size_t> &tgts, const std::vector<real_t> &params);
	void blockgate(snuqs::amp_t *state, size_t num_amps, cudaStream_t s, const snuqs::GPUGate *gate);

	void apply(snuqs::amp_t *state, size_t num_amps, gpu::stream_t s, const Gate *gate);
	void nswapgate(snuqs::amp_t *state, size_t num_amps, cudaStream_t s, size_t nqubits, const snuqs::GPUGate *gate);

} // namespace gpu
} // namespace snuqs
