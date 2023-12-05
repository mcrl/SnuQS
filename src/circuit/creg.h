#ifndef __CREG_H__
#define __CREG_H__

#include <cstddef>

namespace snuqs {
class Creg {
public:
  Creg(size_t num_bits);

private:
  size_t num_bits_;
};

} // namespace snuqs

#endif //__CREG_H__
