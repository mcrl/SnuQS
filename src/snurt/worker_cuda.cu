#include "worker_cuda.h"
#include "command.h"

#include <mutex>
#include <cstdlib>

#include "logger.h"

namespace snurt {

WorkerCUDA::WorkerCUDA(int dev_id)
: dev_id_(dev_id)
{
  cudaError_t
  err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess)
    throw std::invalid_argument("Cannot create cuda stream");
}

WorkerCUDA::~WorkerCUDA()
{
  cudaStreamDestroy(stream_);
}

int WorkerCUDA::Init() {
  running_ = true;
  return 0;
}

static void callback(void *userData) {
  Command *comm = static_cast<Command*>(userData);
  comm->retval = CommandRetval::kSuccess;
  comm->status = CommandStatus::kTerminated;
}

void WorkerCUDA::process(Command *comm) {
  cudaError_t err;
  switch (comm->opcode) {
    case CommandOpcode::kMemcpyH2D:
    case CommandOpcode::kMemcpyD2H:
      err = cudaMemcpyAsync(
          static_cast<void*>(comm->dst),
          static_cast<void*>(comm->src), 
          comm->count, 
          comm->opcode == CommandOpcode::kMemcpyH2D
          ? cudaMemcpyHostToDevice 
          : cudaMemcpyDeviceToHost,
          stream_);
      if (err != cudaSuccess) {
        comm->retval = CommandRetval::kFailure;
        comm->status = CommandStatus::kTerminated;
      }
      // FIXME
      err = cudaLaunchHostFunc(stream_, callback, comm);
      if (err != cudaSuccess) {
        comm->retval = CommandRetval::kFailure;
        comm->status = CommandStatus::kTerminated;
      }
      break;

    case CommandOpcode::kMemcpyS2D:
    case CommandOpcode::kMemcpyD2S:
      throw std::invalid_argument("Not supported yet");

    default:
      throw std::invalid_argument("Invalid command");
      break;
  }
}

void WorkerCUDA::Loop() {
  cudaError_t err = cudaSetDevice(dev_id_);
  if (err != cudaSuccess) {
    //TODO: Logging
    std::exit(EXIT_FAILURE);
  }

  while (running_) {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return !running_ || !commands_.empty(); });

    if (!commands_.empty()) {
      Command *comm = commands_.front();
      if (comm != nullptr) {
        process(comm);
        commands_.pop_front();
      }
    }
  }
}

void WorkerCUDA::Deinit() {
  running_ = false;
  cv_.notify_one();
}

void WorkerCUDA::Enqueue(Command *comm) {
  std::lock_guard<std::mutex> guard(mutex_);
  commands_.push_back(comm);
  cv_.notify_one();
}

} // namespace snurt
