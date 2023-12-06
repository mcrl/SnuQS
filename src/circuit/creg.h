#ifndef __CREG_H__
#define __CREG_H__

#include <bitset>
#include <cstddef>
#include <string>
#include <vector>

namespace snuqs {

class Cbit;
class Creg {
public:
  static constexpr size_t MAX_BITS = 64;
  using bitset = std::bitset<Creg::MAX_BITS>;

  Creg(std::string name, size_t num_bits);
  ~Creg();
  std::string name() const;
  size_t value() const;
  Cbit &__getitem__(size_t key);
  void __setitem__(size_t key, bool val);

private:
  std::string name_;
  size_t num_bits_;
  bitset bitset_;
  std::vector<Cbit *> cbits_;

protected:
  Creg();
};

class Cbit {
public:
  Cbit(const Creg &creg, size_t index, Creg::bitset::reference bit);
  ~Cbit();

private:
  const Creg &creg_;
  size_t index_;
  Creg::bitset::reference bit_;
};

} // namespace snuqs

#endif //__CREG_H__
