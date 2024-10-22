#ifndef _RUNTIME_TYPES_H_
#define _RUNTIME_TYPES_H_

#include <stdint.h>
namespace runtime {

struct grid {
  int x, y, z;
};

struct addr_t {
  union {
    uint64_t addr_u64;
    void *addr_p;
  };
};

typedef addr_t host_addr_t;
typedef addr_t device_addr_t;
typedef addr_t storage_addr_t;

} // namespace runtime

#endif //_RUNTIME_TYPES_H_
