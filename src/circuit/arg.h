#ifndef __ARG_H__
#define __ARG_H__

#include "reg.h"
#include <cstddef>
#include <memory>

namespace snuqs {

class Qarg {
public:
  Qarg(std::shared_ptr<const Qreg> qreg);
  Qarg(std::shared_ptr<const Qreg> qreg, size_t index);
  std::shared_ptr<const Qreg> qreg() const;
  std::string __repr__() const;
  size_t index() const;
  size_t globalIndex() const;
  bool operator<(const Qarg &other) const;
  bool operator==(const Qarg &other) const;

  std::shared_ptr<const Qreg> qreg_;

private:
  size_t index_;
  size_t base_;
};

class Carg {
public:
  Carg(const Creg &creg);
  Carg(const Creg &creg, size_t index);
  const Creg &creg() const;
  std::string __repr__() const;

  const Creg &creg_;
  size_t index_;
};

} // namespace snuqs

#endif //__ARG_H__
