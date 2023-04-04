#pragma once

#include <vector>

#include "gate_interface.h"
#include "types.h"
#include "gate.h"

namespace snuqs {

class GateCUDAImpl : public GateInterface {
  public:
  virtual ~GateCUDAImpl() override;
  virtual void idgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void hgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void xgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void ygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void zgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void sxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void sygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void sgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void sdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void tgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void tdggate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void rxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void rygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void rzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void u1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void u2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void u3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void swapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void cxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void cygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void czgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void chgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void crxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void crygate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void crzgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void cu1gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void cu2gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void cu3gate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void ccxgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;

  virtual void measure(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;

  virtual void init(amp_t *buf, size_t num_amps) override;
  virtual void zerostate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void uniform(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;

  virtual void fusiongate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void nswapgate(amp_t *state, size_t num_amps, const std::vector<size_t> &tgts, const std::vector<real_t> &params) override;
  virtual void blockgate(amp_t *state, size_t num_amps, const std::vector<Gate::shared_ptr> &gates) override;
};

} // namespace snuqs
