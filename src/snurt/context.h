#pragma once

class CommandQueue;

namespace snurt {
class Worker;
namespace context {

int Init();
void Deinit();

void RegisterQueue(CommandQueue *queue);

int ContextIOInit();
void ContextIODeinit();
Worker* GetIOWorker();

int ContextCUDAInit();
void ContextCUDADeinit();
Worker* GetCUDAWorker(int dev_id);

} // namespace context

} // namespace snurt
