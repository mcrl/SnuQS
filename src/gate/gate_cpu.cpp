#include <iostream>
#include <cstddef>
#include <cmath>
#include <bitset>
#include <algorithm>

#include <cassert>

#include "gate_cpu.hpp"
#include "configure.hpp"

namespace snuqs {
namespace cpu {

void init(snuqs::amp_t *buf, size_t num_amps) {
#pragma omp parallel for
	for (size_t i = 0; i < num_amps; i++) {
		buf[i] = 0;
	}
	buf[0] = 1.;
}

void zerostate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
#pragma omp parallel for
	for (size_t i = 0; i < num_amps; i++) {
		state[i] = 0.;
	}
}

void uniform(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t nqubit = tgts.size();
	real_t val = 1.;
	for (size_t i = 0; i < nqubit; i++) {
		val *= ((real_t)M_SQRT1_2);
	}

#pragma omp parallel for firstprivate(val)
	for (size_t i = 0; i < num_amps; i++) {
		state[i] = val;
	}
}

void idgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	/* Do Nothing */
}

void hgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

#pragma omp parallel for collapse(2)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = g + i;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = ((real_t)M_SQRT1_2) * (a0 + a1);
			state[j+st] = ((real_t)M_SQRT1_2) * (a0 - a1);
		}
	}
}

void xgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

#pragma omp parallel for collapse(2)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = g + i;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = a1;
			state[j+st] = a0;
		}
	}
}

void ygate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = { 0, 1 };
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = g + i;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = -coef*a1;
			state[j+st] = coef*a0;
		}
	}
}

void zgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

#pragma omp parallel for collapse(2)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = - a;
		}
	}
}

void sxgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef0 = { 0.5, 0.5 };
	snuqs::amp_t coef1 = { 0.5, -0.5 };
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0 + coef1 * a1;
			state[j+st] = coef1 * a0 + coef0 * a1;
		}
	}
}

void sygate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = { 0.5, 0.5 };
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef * (a0 + a1);
			state[j+st] = coef * (a1 - a0);
		}
	}
}

void sgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = {0, 1};
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = coef * a;
		}
	}
}

void sdggate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = {0, -1};
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = coef * a;
		}
	}
}

void tgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = { ((real_t)M_SQRT1_2), ((real_t)M_SQRT1_2) };
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = coef * a;
		}
	}
}

void tdggate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	snuqs::amp_t coef = { ((real_t)M_SQRT1_2), -((real_t)M_SQRT1_2) };
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = coef * a;
		}
	}
}

void rxgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {0, -sin(theta/2)};
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0 + coef1 * a1;
			state[j+st] = coef1 * a0 + coef0 * a1;
		}
	}
}

void rygate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {sin(theta/2), 0};
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0 - coef1 * a1;
			state[j+st] = coef1 * a0 + coef0 * a1;
		}
	}
}

void rzgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	snuqs::amp_t coef1 = {cos(theta/2), sin(theta/2)};
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0;
			state[j+st] = coef1 * a1;
		}
	}
}

void u1gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t lambda = params[0];
	snuqs::amp_t coef = { cos(lambda), sin(lambda) };
#pragma omp parallel for collapse(2) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a = state[j+st];
			state[j+st] = coef * a;
		}
	}
}

void u2gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t phi = params[0];
	real_t lambda = params[1];
	snuqs::amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	snuqs::amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	snuqs::amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	snuqs::amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1, coef2, coef3)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0 + coef1 * a1;
			state[j+st] = coef2 * a0 + coef3 * a1;
		}
	}
}

void u3gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt = tgts[0];
	size_t st = (1ul << tgt);
	size_t step = st*2;

	real_t theta = params[0];
	real_t phi = params[1];
	real_t lambda = params[2];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	snuqs::amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	snuqs::amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};
#pragma omp parallel for collapse(2) firstprivate(coef0, coef1, coef2, coef3)
	for (size_t g = 0; g < num_amps; g += step) {
		for (size_t i = 0; i < st; i++) {
			size_t j = i + g;

			snuqs::amp_t a0 = state[j];
			snuqs::amp_t a1 = state[j+st];
			state[j] = coef0 * a0 + coef1 * a1;
			state[j+st] = coef2 * a0 + coef3 * a1;
		}
	}
}

void swapgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt0 = tgts[0];
	size_t tgt1 = tgts[1];

	size_t st0 = 1ul << std::max(tgt0, tgt1);
	size_t st1 = 1ul << std::min(tgt0, tgt1);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

#pragma omp parallel for collapse(3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+st0];
				snuqs::amp_t a1 = state[j+st1];
				state[j+st0] = a1;
				state[j+st1] = a0;
			}
		}
	}
}

void cxgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

#pragma omp parallel for collapse(3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = a1;
				state[j+st] = a0;
			}
		}
	}
}

void cygate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	snuqs::amp_t coef = { 0, 1 };
#pragma omp parallel for collapse(3) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = -coef*a1;
				state[j+st] = coef*a0;
			}
		}
	}
}

void czgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t tgt0 = tgts[0];
	size_t tgt1 = tgts[1];

	size_t st0 = 1ul << std::max(tgt0, tgt1);
	size_t st1 = 1ul << std::min(tgt0, tgt1);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

#pragma omp parallel for collapse(3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				state[j+st] = -state[j+st];
			}
		}
	}
//#endif
}

void chgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

#pragma omp parallel for collapse(3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = ((real_t)M_SQRT1_2) * (a0 + a1);
				state[j+st] = ((real_t)M_SQRT1_2) * (a0 - a1);
			}
		}
	}
}

void crxgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {0, -sin(theta/2)};
#pragma omp parallel for collapse(3) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = coef0 * a0 + coef1 * a1;
				state[j+st] = coef1 * a0 + coef0 * a1;
			}
		}
	}
}

void crygate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {sin(theta/2), 0};
#pragma omp parallel for collapse(3) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = coef0 * a0 - coef1 * a1;
				state[j+st] = coef1 * a0 + coef0 * a1;
			}
		}
	}
}

void crzgate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t theta = params[0];
	snuqs::amp_t coef0 = {cos(theta/2), -sin(theta/2)};
	snuqs::amp_t coef1 = {cos(theta/2), sin(theta/2)};
#pragma omp parallel for collapse(3) firstprivate(coef0, coef1)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = coef0 * a0;
				state[j+st] = coef1 * a1;
			}
		}
	}
}

void cu1gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t lambda = params[0];
	snuqs::amp_t coef = { cos(lambda), sin(lambda) };
#pragma omp parallel for collapse(3) firstprivate(coef)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a = state[j+st];
				state[j+st] = coef * a;
			}
		}
	}
}

void cu2gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;

	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t phi = params[0];
	real_t lambda = params[1];
	snuqs::amp_t coef0 = {((real_t)M_SQRT1_2), 0};
	snuqs::amp_t coef1 = {((real_t)M_SQRT1_2) * -cos(lambda), ((real_t)M_SQRT1_2) * -sin(lambda)};
	snuqs::amp_t coef2 = {((real_t)M_SQRT1_2) * cos(phi), ((real_t)M_SQRT1_2) * sin(phi)};
	snuqs::amp_t coef3 = {((real_t)M_SQRT1_2) * cos(phi+lambda), ((real_t)M_SQRT1_2) * sin(phi+lambda)};
#pragma omp parallel for collapse(3) firstprivate(coef0, coef1, coef2, coef3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = coef0 * a0 + coef1 * a1;
				state[j+st] = coef2 * a0 + coef3 * a1;;
			}
		}
	}
}

void cu3gate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	size_t ctrl = tgts[0];
	size_t tgt = tgts[1];

	size_t cst = 1ul << ctrl;
	size_t st0 = 1ul << std::max(ctrl, tgt);
	size_t st1 = 1ul << std::min(ctrl, tgt);

	size_t step0 = st0*2;
	size_t step1 = st1*2;

	size_t st = st0+st1;

	real_t theta = params[0];
	real_t phi = params[1];
	real_t lambda = params[2];
	snuqs::amp_t coef0 = {cos(theta/2), 0};
	snuqs::amp_t coef1 = {-cos(lambda)*sin(theta/2), sin(lambda)*sin(theta/2)};
	snuqs::amp_t coef2 = {cos(phi)*sin(theta/2), sin(phi)*sin(theta/2)};
	snuqs::amp_t coef3 = {cos(phi+lambda)*cos(theta/2), sin(phi+lambda)*cos(theta/2)};
#pragma omp parallel for collapse(3) firstprivate(coef0, coef1, coef2, coef3)
	for (size_t g = 0; g < num_amps; g += step0) {
		for (size_t h = 0; h < st0; h += step1) {
			for (size_t i = 0; i < st1; i++) {
				size_t j = g + h + i;
				snuqs::amp_t a0 = state[j+cst];
				snuqs::amp_t a1 = state[j+st];
				state[j+cst] = coef0 * a0 + coef1 * a1;
				state[j+st] = coef2 * a0 + coef3 * a1;;
			}
		}
	}
}

void diagonal(snuqs::amp_t *state, const snuqs::amp_t *vec, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	std::vector<size_t> sorted_tgts = tgts;
	std::sort(sorted_tgts.begin(), sorted_tgts.end());

	size_t ntgt = tgts.size();
	std::vector<size_t> sts(1ul << ntgt);
	for (size_t i = 0; i < (1ul << ntgt); i++) {
		sts[i] = 0;
		for (size_t j = 0; j < ntgt; j++) {
			if ((i & (1ul << j)) != 0)
				sts[i] += (1ul << tgts[j]);
		}
	}

#pragma omp parallel for
	for (size_t g = 0; g < num_amps / (1ul << tgts.size()); g++) {
		// index
		size_t j = g;
		for (size_t k = 0; k < ntgt; k++) {
			j = ((j >> sorted_tgts[k]) << (sorted_tgts[k]+1)) + (j & ((1ul << tgts[k])-1));
		}

		// gather
		snuqs::amp_t buf[(1ul << ntgt)];
		for (size_t k = 0; k < (1ul << ntgt); k++) {
			buf[k] = state[j + sts[k]];
		}

		// dotprodcut + scatter
		for (size_t k = 0; k < (1ul << ntgt); k++) {
			state[j + sts[k]] = vec[k] * buf[k];
		}
	}
}

void unitary(snuqs::amp_t *state, const snuqs::amp_t *mat, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	std::vector<size_t> sorted_tgts = tgts;
	std::sort(sorted_tgts.begin(), sorted_tgts.end());

	size_t ntgt = tgts.size();
	std::vector<size_t> sts(1ul << ntgt);
	for (size_t i = 0; i < (1ul << ntgt); i++) {
		sts[i] = 0;
		for (size_t j = 0; j < ntgt; j++) {
			if ((i & (1ul << j)) != 0)
				sts[i] += (1ul << tgts[j]);
		}
	}

#pragma omp parallel for
	for (size_t g = 0; g < num_amps / (1ul << tgts.size()); g++) {
		// index
		size_t j = g;
		for (size_t k = 0; k < ntgt; k++) {
			j = ((j >> sorted_tgts[k]) << (sorted_tgts[k]+1)) + (j & ((1ul << tgts[k])-1));
		}

		// gather
		snuqs::amp_t buf[(1ul << ntgt)];
		for (size_t k = 0; k < (1ul << ntgt); k++) {
			buf[k] = state[j + sts[k]];
		}

		// matmul + scatter
		for (size_t c = 0; c < (1ul << ntgt); c++) {
			for (size_t k = 0; k < (1ul << ntgt); k++) {
				state[j + sts[k]] = mat[c * (1ul << ntgt) + k] * buf[k];
			}
		}
	}
}

void fusiongate(snuqs::amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) {
	assert(false);
}

void blockgate(snuqs::amp_t *state, size_t num_amps, const std::vector<Gate::shared_ptr> &gates) {
	for (auto &g : gates) {
		std::cout << g << "\n";
	}
	assert(false);
}



} // namespace cpu
} // namespace snuqs
