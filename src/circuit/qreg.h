#ifndef __QREG_H__
#define __QREG_H__

#include <cstddef>
#include <string>
#include <vector>

namespace snuqs {

class Qbit;
class Qreg {
public:
  Qreg(std::string name, size_t dim);
  virtual ~Qreg();

  std::string name_;
  size_t dim_;
  std::vector<Qbit *> qbits_;
};

class Qbit : public Qreg {
public:
  Qbit(Qreg *qreg, size_t index);

  Qreg *qreg_;
  size_t index_;
};

} // namespace snuqs

#endif // __QREG_H__
