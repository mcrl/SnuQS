#ifndef _STREAM_H_
#define _STREAM_H_
#include <cuda_runtime.h>

class Stream {
 public:
  Stream();
  ~Stream();
  cudaStream_t get();

 private:
  cudaStream_t stream_;
};
#endif  //_STREAM_H_
