#include "stream/stream.h"

#include "event/event.h"
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

std::shared_ptr<Stream> Stream::create_nonblocking() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return std::make_shared<Stream>(reinterpret_cast<void*>(stream));
}

void Stream::synchronize() {
  CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)));
}

void Stream::wait_event(std::shared_ptr<Event> event) {
  CUDA_CHECK(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_),
                                 reinterpret_cast<cudaEvent_t>(event->get())));
}
