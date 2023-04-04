#pragma once

#include "types.h"
#include "gate.h"

namespace snuqs {

class GateInterface {
  public:
  virtual ~GateInterface() = 0;

  virtual void idgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void hgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void xgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void ygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void zgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void sxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void sygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void sgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void sdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void tgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void tdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void rxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void rygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void rzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void u1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void u2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void u3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void swapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void cxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void cygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void czgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void chgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void crxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void crygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void crzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void cu1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void cu2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void cu3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void ccxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;

  virtual void measure(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;

  virtual void init(amp_t *buf, size_t num_amps) = 0;
  virtual void zerostate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void uniform(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;

  virtual void fusiongate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void nswapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) = 0;
  virtual void blockgate(amp_t *state, size_t num_amps, const std::vector<Gate::shared_ptr> &gates) = 0;
};

} // namespace snuqs
