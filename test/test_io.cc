#include <iostream>
#include <chrono>
#include <thread>

#include "snurt/api.h"
#include "snurt/context.h"
#include "snurt/command_queue.h"

#include <unistd.h>

//#define NELEM (512 * 2049)
#define NELEM (1ul << 25)
#define NCOPY 1

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

  snurt::addr_t haddr = snurt::MallocHost(sizeof(int)*NELEM);
  snurt::addr_t ioaddr = snurt::MallocIO(sizeof(int)*NELEM);

  int *hbuf = static_cast<int*>(static_cast<void*>(haddr));


  std::cout << "========================================\n";
  std::cout << "=============== IO TEST ================\n";
  std::cout << "========================================\n";

  std::cout << "Address of host buffer: " << static_cast<void*>(haddr) << "\n";
  std::cout << "Address of io buffer: " << static_cast<void*>(ioaddr) << "\n";

  for (size_t i = 0; i < NELEM; ++i) {
    hbuf[i] = 0xdeadbeaf;
  }

  double elapsed_time;
  struct timespec s, e;
  snurt::addr_t ha, ia;

  for (size_t i = 0; i < NELEM; ++i) {
    hbuf[i] = 1;
  }

  ha = haddr;
  ia = ioaddr;
  clock_gettime(CLOCK_MONOTONIC, &s);
  for (int i = 0; i < NCOPY; ++i) {
    size_t count = sizeof(int) * NELEM / NCOPY;
    err = snurt::MemcpyS2H(queue, ha, ia, count);
    if (err) {
      std::cout << "S2H failed\n";
      return err;
    }
    ha += count;
    ia += count;
  }

  err = snurt::Synchronize(queue);
  if (err) {
    std::cout << "Sync4 failed\n";
    return err;
  }

  clock_gettime(CLOCK_MONOTONIC, &e);

  elapsed_time = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec)/1000000000.;
  std::cout << "Elapsed time: " << elapsed_time << "s\n";
  std::cout << "Bandwidth: " << sizeof(int) * NELEM / elapsed_time / 1000000000<< "GB/s\n";

  ha = haddr;
  ia = ioaddr;
  clock_gettime(CLOCK_MONOTONIC, &s);
  for (int i = 0; i < NCOPY; ++i) {
    size_t count = sizeof(int) * NELEM / NCOPY;
    err = snurt::MemcpyH2S(queue, ia, ha, count);
    if (err) {
      std::cout << "H2S failed\n";
      return err;
    }
    ha += count;
    ia += count;
  }

  err = snurt::Synchronize(queue);
  if (err) {
    std::cout << "Sync3 failed\n";
    return err;
  }

  clock_gettime(CLOCK_MONOTONIC, &e);

  elapsed_time = (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec)/1000000000.;
  std::cout << "Elapsed time: " << elapsed_time << "s\n";
  std::cout << "Bandwidth: " << sizeof(int) * NELEM / elapsed_time / 1000000000<< "GB/s\n";

  for (size_t i = 0; i < NELEM; ++i) {
    if (hbuf[i] != (int)0xdeadbeaf) {
      std::cout << "Memcpy Failed " <<  hbuf[i] << " " << i << "\n";
      return 0;
    }
  }
  std::cout << "====>>> OK\n";

  snurt::Deinit();

  return 0;
}
