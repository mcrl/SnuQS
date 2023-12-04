#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

namespace snuqs {

template<typename T>
class Simulator {
public:
  virtual ~Simulator();
  virtual void run() = 0;
};

} // namespace snuqs

#endif // __SIMULATOR_H__
