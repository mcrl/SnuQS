#include "stream/stream.h"

#include "utils_cuda.h"

Stream::Stream(void* stream) : stream_(stream) {}

Stream::~Stream() {
  if (stream_ != nullptr)
    CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_)));
}

void* Stream::get() { return stream_; }

static std::shared_ptr<Stream> null_stream = std::make_shared<Stream>(nullptr);
std::shared_ptr<Stream> Stream::null() { return null_stream; }
