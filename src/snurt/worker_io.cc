#include "worker_io.h"

#include <mutex>
#include <cstdlib>

#include "command.h"
#include "blkio.h"

namespace snurt {
//
// WorkerIO
//
WorkerIO::WorkerIO(const char *config_file)
: config_file_(config_file)
{
}

WorkerIO::~WorkerIO()
{
}

int WorkerIO::Init() {
  io_handle_ = blkio::CreateIOHandle(config_file_);
  running_ = true;
  return 0;
}

static void callback(void *user_data) {
  Command *comm = static_cast<Command*>(user_data);
  comm->retval = CommandRetval::kSuccess;
  comm->status = CommandStatus::kTerminated;
}

void WorkerIO::process(Command *comm) {
  int err;
  switch (comm->opcode) {
    case CommandOpcode::kMemcpyS2H:
      err = io_handle_->Read(
          static_cast<void*>(comm->dst), 
          static_cast<uint64_t>(comm->src), 
          comm->count,
          callback,
          comm
          );
      if (err) {
        comm->retval = CommandRetval::kFailure;
        comm->status = CommandStatus::kTerminated;
      }
      break;
    case CommandOpcode::kMemcpyH2S:
      err = io_handle_->Write(
          static_cast<void*>(comm->src), 
          static_cast<uint64_t>(comm->dst), 
          comm->count,
          callback,
          comm
      );
      if (err) {
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

void WorkerIO::Loop() {
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

void WorkerIO::Deinit() {
  running_ = false;
  cv_.notify_one();
}

void WorkerIO::Enqueue(Command *comm) {
  std::lock_guard<std::mutex> guard(mutex_);
  commands_.push_back(comm);
  cv_.notify_one();
}

} // namespace snurt
