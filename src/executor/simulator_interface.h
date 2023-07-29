#ifndef __SIMULATOR_INTERFACE_H__
#define __SIMULATOR_INTERFACE_H__

#include <vector>
#include <complex>
#include "qop.h"
#include "qreg.h"
#include "creg.h"

namespace snuqs {
  /*
    Barrier = 0,
    Reset = 1,
    Measure = 2,
    Cond = 3,
    Init = 4,
    Fini = 5,
    UGate = 6,
    CXGate = 7
    */

class SimulatorInterface {
  public:
  virtual ~SimulatorInterface() {}
  virtual void run(std::vector<Qop*> &circ, Qreg &qreg, Creg &creg) = 0;
  virtual void run_op(Qop* qop, Qreg &qreg, Creg &creg) = 0;
  virtual void init(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void fini(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void cond(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void measure(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void reset(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void barrier(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void ugate(Qop *qop, Qreg &qreg, Creg &creg) = 0;
  virtual void cxgate(Qop *qop, Qreg &qreg, Creg &creg) = 0;
};

} // namespace snuqs

#endif // __SIMULATOR_INTERFACE_H__
