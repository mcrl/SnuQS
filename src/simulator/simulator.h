#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

namespace snuqs {

class Simulator {
public:
  virtual ~Simulator() = default;
  virtual void run() = 0;
};

} // namespace snuqs

#endif // __SIMULATOR_H__
