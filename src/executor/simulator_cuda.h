#ifndef __SIMULATOR_CUDA_H__
#define __SIMULATOR_CUDA_H__

#include "simulator_interface.h"
#include "runtime/rt.h"

namespace snuqs {

class SimulatorCUDA: public SimulatorInterface {
  public:
    SimulatorCUDA();
    virtual ~SimulatorCUDA();
    virtual void run(std::vector<Qop*> &circ, Qreg &qreg, Creg &creg) override;
    virtual void run_op(Qop* qop, Qreg &qreg, Creg &creg) override;
    virtual void init(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void fini(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void cond(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void measure(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void reset(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void barrier(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void ugate(Qop *qop, Qreg &qreg, Creg &creg) override;
    virtual void cxgate(Qop *qop, Qreg &qreg, Creg &creg) override;

    rt::handle_t rt;
    rt::stream_t *stream;
};

} // namespace snuqs

#endif // __SIMULATOR_CUDA_H__
