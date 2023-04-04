#include "context.h"
#include "worker.h"

#include <vector>
#include <thread>


#include "worker_cuda.h"

namespace snurt {
namespace context {

static std::vector<Worker*> workers;
static std::vector<std::thread> threads;

static void WorkerLoop(Worker *worker)
{
  worker->Loop();
}

int ContextCUDAInit() {
  cudaError_t err;
  int count;
  err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess)
    return -EFAULT;

  for (int i = 0; i < count; ++i) {
    workers.push_back(new WorkerCUDA(i));
  }

  for (auto &worker : workers) {
    int err = worker->Init();
    if (err) return err;

    auto t = std::thread(WorkerLoop, worker);
    t.detach();
    threads.push_back(std::move(t));
  }

  return 0;
}

void ContextCUDADeinit() {
  for (auto &worker : workers) {
    worker->Deinit();
  }

  for (auto &worker : workers) {
    delete worker;
  }
}

Worker* GetCUDAWorker(int dev_id)
{
  if (dev_id < workers.size())
    return workers[dev_id];
  return nullptr;
}



} // namespace context

} // namespace snurt
