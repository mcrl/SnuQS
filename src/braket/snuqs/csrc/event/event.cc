#include "event/event.h"

#include "utils_cuda.h"

Event::Event(void* event) : event_(event) {}

Event::~Event() {
  if (event_ != nullptr)
    CUDA_CHECK(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event_)));
}

void* Event::get() { return event_; }
std::shared_ptr<Event> Event::create() {
  cudaEvent_t event;
  CUDA_CHECK(cudaEventCreate(&event));
  return std::make_shared<Event>(reinterpret_cast<void*>(event));
}

void Event::record(std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event_),
                             reinterpret_cast<cudaStream_t>(stream->get())));
}

void Event::synchronize() {
  CUDA_CHECK(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event_)));
}
