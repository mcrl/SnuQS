#ifndef __SIMULATOR_OMP_H__
#define __SIMULATOR_OMP_H__

#include "simulator_interface.h"

namespace snuqs {

class SimulatorOMP: public SimulatorInterface {
  public:
    SimulatorOMP();
    virtual ~SimulatorOMP();
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
};

} // namespace snuqs

#endif // __SIMULATOR_OMP_H__
