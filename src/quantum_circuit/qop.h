#ifndef __QOP_H__
#define __QOP_H__

#include "types.h"
#include <vector>

namespace snuqs {

class Qop {
public:
  Qop(std::vector<qidx> qubits);
  Qop(std::vector<qidx> qubits, std::vector<double> params);
  virtual ~Qop();

protected:
  std::vector<qidx> qubits_;
  std::vector<double> params_;
};

class UGate : public Qop {
public:
  UGate(qidx target, double theta, double phi, double lambda);
};

class CXGate : public Qop {
public:
  CXGate(qidx control, qidx target);
};

class Measurement : public Qop {
  Measurement(qidx target);
};

class Reset : public Qop {
  Reset(qidx target);
};

} // namespace snuqs

#endif //__QOP_H__
