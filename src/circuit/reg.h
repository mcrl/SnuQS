#ifndef __REG_H__
#define __REG_H__

#include <bitset>
#include <cstddef>
#include <string>

namespace snuqs {

class Reg {
protected:
  Reg(std::string name, size_t dim);

public:
  static constexpr size_t MAX_BITS = 64;

  std::string name() const;
  size_t dim() const;
  virtual std::string __repr__() const;

protected:
  std::string name_;
  size_t dim_;
};

class Qreg : public Reg {
public:
  Qreg(std::string name, size_t dim);
};

template <size_t nbits> class _BitsetImpl {};

class Creg : public Reg {
public:
  Creg(std::string name, size_t dim);
  virtual std::string __repr__() const override;
  bool __getitem__(size_t key);
  void __setitem__(size_t key, bool val);

private:
  std::bitset<Reg::MAX_BITS> bitset_;
};

} // namespace snuqs

#endif // __QREG_H__
