#include <iostream>
#include <chrono>
#include <thread>

#include "snurt/api.h"
#include "snurt/context.h"
#include "snurt/command_queue.h"

#include <unistd.h>

#define NELEM (1e9)

int main()
{
  int err;
  err = snurt::Init();
  if (err) {
    std::cout << "ContextInit failed\n";
    return err;
  }

  snurt::CommandQueue queue;
  snurt::CommandQueue queue1;

  snurt::addr_t haddr = snurt::MallocHost(sizeof(size_t)*NELEM);
  snurt::addr_t daddr = snurt::MallocDevice(sizeof(size_t)*NELEM, 0);

  size_t *hbuf = static_cast<size_t*>(static_cast<void*>(haddr));

  std::cout << static_cast<void*>(haddr) << "\n";
  std::cout << static_cast<void*>(daddr) << "\n";

  std::cout << "========================================\n";
  std::cout << "=============== CUDA TEST ==============\n";
  std::cout << "========================================\n";

  for (size_t i = 0; i < NELEM; ++i) {
    hbuf[i] = i;
  }

  struct timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  err = snurt::MemcpyH2D(queue, daddr, haddr, sizeof(size_t)*NELEM);
  if (err) {
    std::cout << "H2D failed\n";
    return err;
  }

  err = snurt::Synchronize(queue);
  if (err) {
    std::cout << "Sync1 failed\n";
    return err;
  }
  clock_gettime(CLOCK_MONOTONIC, &e);

  double elapsed_time = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec)/1000000000.;
  std::cout << "Elapsed time: " << elapsed_time << "s\n";
  std::cout << "Bandwidth: " << sizeof(size_t) * NELEM / elapsed_time / 1000000000<< "GB/s\n";

  for (size_t i = 0; i < NELEM; ++i) {
    hbuf[i] = 0;
  }

  clock_gettime(CLOCK_MONOTONIC, &s);
  err = snurt::MemcpyD2H(queue, haddr, daddr, sizeof(size_t)*NELEM);
  if (err) {
    std::cout << "D2H failed\n";
    return err;
  }
  err = snurt::Synchronize(queue);
  if (err) {
    std::cout << "Sync2 failed\n";
    return err;
  }
  clock_gettime(CLOCK_MONOTONIC, &e);

  elapsed_time = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec)/1000000000.;
  std::cout << "Elapsed time: " << elapsed_time << "s\n";
  std::cout << "Bandwidth: " << sizeof(size_t) * NELEM / elapsed_time / 1000000000<< "GB/s\n";

  for (size_t i = 0; i < NELEM; ++i) {
    if (hbuf[i] != i) {
      std::cout << "Memcpy Failed " <<  hbuf[i] << " " << i << "\n";
      return 0;
    }
  }
  std::cout << "====>>> OK\n";

  snurt::Deinit();

  return 0;
}
