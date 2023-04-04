#pragma once

#include "worker.h"
#include <list>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <cuda_runtime.h>

#include "blkio.h"

namespace snurt {

class Command;

class WorkerIO : public Worker {
  public:
  WorkerIO(const char *config_file);
  virtual ~WorkerIO() override;

  virtual int Init() override;
  virtual void Loop() override;
  virtual void Deinit() override;

  virtual void Enqueue(Command *comm) override;

  private:
  void process(Command *comm);

	std::mutex mutex_;
	std::condition_variable cv_;
  std::list<Command*> commands_;
  std::atomic<bool> running_;

  const char *config_file_;
  std::unique_ptr<blkio::IOHandle> io_handle_;
};

} // namespace snurt
