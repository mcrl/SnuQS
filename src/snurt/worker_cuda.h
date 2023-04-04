#pragma once

#include "worker.h"
#include <list>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <cuda_runtime.h>

namespace snurt {

class Command;

class WorkerCUDA : public Worker {
  public:
  WorkerCUDA(int dev_id);
  virtual ~WorkerCUDA() override;

  virtual int Init() override;
  virtual void Loop() override;
  virtual void Deinit() override;

  virtual void Enqueue(Command *comm) override;

  private:
  void process(Command *comm);

  int dev_id_;

	std::mutex mutex_;
	std::condition_variable cv_;
  std::list<Command*> commands_;
  std::atomic<bool> running_;

  cudaStream_t stream_;
};

} // namespace snurt
