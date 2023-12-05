#ifndef __QOP_H__
#define __QOP_H__

#include "types.h"
#include <vector>

namespace snuqs {

class Qop {
public:
  Qop();
  virtual ~Qop();

protected:
  std::vector<qidx> qubits_;
  std::vector<double> params_;
};

} // namespace snuqs

#endif //__QOP_H__
