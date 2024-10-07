#ifndef _CUDA_RUNTIME_H_
#define _CUDA_RUNTIME_H_
namespace cu {
std::pair<size_t, size_t> mem_info();
int device_count();
};  // namespace cu
#endif  // _CUDA_RUNTIME_H_
