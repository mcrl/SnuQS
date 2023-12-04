#ifndef __CREG_H__
#define __CREG_H__

#include "types.h"

namespace snuqs {
class Creg {
public:
  Creg(qidx num_bits);

private:
  qidx num_bits_;
};
} // namespace snuqs

#endif //__CREG_H__
