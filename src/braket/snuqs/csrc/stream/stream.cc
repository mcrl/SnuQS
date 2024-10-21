#include "stream/stream.h"

#include "utils_cuda.h"

Stream::Stream(void* stream) : stream_(stream) {}

Stream::~Stream() {
  if (stream_ != nullptr)
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_)));
}

void* Stream::get() { return stream_; }

std::shared_ptr<Stream> Stream::create() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  return std::make_shared<Stream>(reinterpret_cast<void*>(stream));
}
