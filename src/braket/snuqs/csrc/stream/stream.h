#ifndef _STREAM_H_
#define _STREAM_H_
#include <cuda_runtime.h>

class Stream {
 public:
  Stream(cudaStream_t stream);
  ~Stream();
  cudaStream_t get();
  static Stream& null();

 private:
  cudaStream_t stream_;
};

#endif  //_STREAM_H_
