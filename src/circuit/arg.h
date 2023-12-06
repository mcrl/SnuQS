#ifndef __ARG_H__
#define __ARG_H__

#include "reg.h"
#include <cstddef>

namespace snuqs {

class Arg {
protected:
  Arg(const Reg &qreg, size_t index);

public:
  const Reg &reg() const;
  size_t index() const;
  size_t dim() const;
  size_t value() const;
  std::string __repr__() const;

protected:
  const Reg &reg_;
  size_t index_;
  size_t dim_;
  size_t value_;
};

class Qarg : public Arg {
public:
  Qarg(const Qreg &qreg, size_t index);
};

class Carg : public Arg {
public:
  Carg(const Creg &creg, size_t index);
};

} // namespace snuqs

#endif //__ARG_H__
