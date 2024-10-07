#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "runtime.h"

#include <stdio.h>
#include <sys/sysinfo.h>
#include <unistd.h>

std::pair<size_t, size_t> mem_info() {
  /* PAGESIZE is POSIX: http://pubs.opengroup.org/onlinepubs/9699919799/
   * but PHYS_PAGES and AVPHYS_PAGES are glibc extensions. I bet those are
   * parsed from /proc/meminfo. */
  size_t free, total;
  total = get_phys_pages() * sysconf(_SC_PAGESIZE);
  free = get_avphys_pages() * sysconf(_SC_PAGESIZE);
  return {free, total};
}
