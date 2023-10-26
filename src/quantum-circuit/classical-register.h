#ifndef __CLASSICAL_REGISTER_H__
#define __CLASSICAL_REGISTER_H__

#include "misc/types.h"

namespace snuqs {
class ClassicalRegister {
public:
  ClassicalRegister(qidx num_bits);

private:
  qidx num_bits_;
};
} // namespace snuqs

#endif //__CLASSICAL_REGISTER_H__
