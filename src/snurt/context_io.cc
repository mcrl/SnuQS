#include "context.h"
#include "worker.h"

#include <vector>
#include <thread>


#include "worker_cuda.h"
#include "worker_io.h"

namespace snurt {
namespace context {


const char *config_file = "io_config.cfg";

static std::vector<Worker*> workers;
static std::vector<std::thread> threads;

static void WorkerLoop(Worker *worker)
{
  worker->Loop();
}

int ContextIOInit() {
  workers.push_back(new WorkerIO(config_file));

  for (auto &worker : workers) {
    int err = worker->Init();
    if (err) return err;

    auto t = std::thread(WorkerLoop, worker);
    t.detach();
    threads.push_back(std::move(t));
  }

  return 0;
}

void ContextIODeinit() {
  for (auto &worker : workers) {
    worker->Deinit();
  }

  for (auto &worker : workers) {
    delete worker;
  }
}

Worker* GetIOWorker()
{
  if (workers.empty())
    return nullptr;
  return workers[0];
}

} // namespace context
} // namespace snurt
