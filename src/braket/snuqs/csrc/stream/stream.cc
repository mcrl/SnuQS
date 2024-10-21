#include "stream/stream.h"

#include "utils_cuda.h"

Stream::Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
Stream::~Stream() { CUDA_CHECK(cudaStreamDestroy(stream_)); }
cudaStream_t Stream::get() { return stream_; }
