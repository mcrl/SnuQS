#pragma once


namespace snurt {

class Command;

class Worker {
  public:
  virtual ~Worker() = default;

  virtual int Init() = 0;
  virtual void Loop() = 0;
  virtual void Deinit() = 0;
  virtual void Enqueue(Command *comm) = 0;
};

} // namespace snurt
