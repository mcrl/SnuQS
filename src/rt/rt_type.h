#ifndef __RT_TYPE_H__
#define __RT_TYPE_H__

#include <cstdint>
#include <cuda_runtime.h>

namespace snuqs {
namespace rt {

using addr_t = void*;
using dim3 = dim3;

/*
struct dim3 {
  int x;
  int y;
  int z;
};
*/

} // namespace snuqs
} // namespace rt

#endif //__RT_TYPE_H__
