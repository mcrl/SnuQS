#ifndef __SIMULATOR_OMP_IO_H__
#define __SIMULATOR_OMP_IO_H__

#include "simulator_interface.h"

#include "runtime/rt.h"

namespace snuqs {

class SimulatorOMPIO: public SimulatorInterface {
  public:
    SimulatorOMPIO(std::vector<std::string> paths);
    virtual ~SimulatorOMPIO();
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
};

} // namespace snuqs

#endif // __SIMULATOR_OMP_H__
