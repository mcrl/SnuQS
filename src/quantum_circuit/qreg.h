#ifndef __QREG_H__
#define __QREG_H__

#include "types.h"

namespace snuqs {

class Qreg {
public:
  Qreg(qidx num_qubits);

private:
  qidx num_qubits_;
};

} // namespace snuqs

#endif // __QREG_H__
