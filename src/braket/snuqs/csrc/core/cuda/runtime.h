#ifndef _CUDA_RUNTIME_H_
#define _CUDA_RUNTIME_H_
#include <tuple>
namespace cu {
std::pair<size_t, size_t> mem_info();
int device_count();
void set_device(int device);
int get_device();
void device_synchronize();
};  // namespace cu
#endif  // _CUDA_RUNTIME_H_
