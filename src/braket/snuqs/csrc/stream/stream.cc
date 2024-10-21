#include "stream/stream.h"

#include "utils_cuda.h"

Stream::Stream(cudaStream_t stream) : stream_(stream) {}

Stream::~Stream() {
  if (stream_ != nullptr) CUDA_CHECK(cudaStreamDestroy(stream_));
}

cudaStream_t Stream::get() { return stream_; }

static Stream* null_stream = new Stream(nullptr);
Stream& Stream::null() { return *null_stream; }
