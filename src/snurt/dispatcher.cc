#include "dispatcher.h"
#include "context.h"
#include "worker.h"


#include <cassert>


#include <unistd.h>


namespace snurt {

void Dispatcher::RegisterQueue(CommandQueue *queue) {
	std::lock_guard<std::mutex> guard(mutex_);
  registered_queues_.push_back(queue);
}

int Dispatcher::Init() {
  running_ = true;
  return 0;
}

void Dispatcher::Deinit() {
  running_ = false;
}

void Dispatcher::Loop() {
  while (running_) {
    Dispatch();
  }
}

void Dispatcher::Dispatch() {
  for (auto queue : registered_queues_) {
    queue->Lock();
    Command *comm = queue->FrontLocked();
    if (comm != nullptr) {
      switch (comm->status) {
        case CommandStatus::kEnqueued:
          switch (comm->opcode) {
            case CommandOpcode::kMemcpyH2D: 
            case CommandOpcode::kMemcpyD2H: 
              {
                comm->status = CommandStatus::kRunning;
                auto worker = context::GetCUDAWorker(comm->dev_id);
                assert(worker != nullptr);
                worker->Enqueue(comm);
              }
              break;

            case CommandOpcode::kMemcpyH2S: 
            case CommandOpcode::kMemcpyS2H: 
              {
                comm->status = CommandStatus::kRunning;
                auto worker = context::GetIOWorker();
                assert(worker != nullptr);
                worker->Enqueue(comm);
              }
              break;

            case CommandOpcode::kMemcpyS2D: 
            case CommandOpcode::kMemcpyD2S: 
              throw std::invalid_argument("Not supported command in the queue");

            case CommandOpcode::kSynchronize: 
              {
                comm->retval = CommandRetval::kSuccess;
                comm->status = CommandStatus::kTerminated;
                std::condition_variable *cv = static_cast<std::condition_variable*>(comm->data);
                cv->notify_one();

                std::mutex mutex;
                std::unique_lock<std::mutex> lk(mutex);
                cv->wait(lk, [&] { return comm->data == nullptr; });
              }
              break;

            default:
              throw std::invalid_argument("Unexpected command in the queue");
          }
          break;

        case CommandStatus::kTerminated:
          if (comm->retval != CommandRetval::kSuccess) {
            queue->SetRetval(-EINVAL);
          }
          queue->DequeueLocked();
          break;
        default:
          break;
      }
    }
    queue->Unlock();
  }
}

} // namespace snurt
