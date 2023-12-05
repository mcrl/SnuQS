#include "assertion.h"
#include "buffer.h"
#include "buffer/mt_raid.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace snuqs {
static std::vector<std::string> devices{
    "/dev/sdb",  "/dev/sdc",  "/dev/sdd",  "/dev/sde",  "/dev/sdf",
    "/dev/sdg",  "/dev/sdh",  "/dev/sdi",  "/dev/sdj",  "/dev/sdk",
    "/dev/sdl",  "/dev/sdm",  "/dev/sdn",  "/dev/sdo",  "/dev/sdp",
    "/dev/sdq",  "/dev/sdr",  "/dev/sds",  "/dev/sdt",  "/dev/sdu",
    "/dev/sdv",  "/dev/sdw",  "/dev/sdx",  "/dev/sdy",  "/dev/sdz",
    "/dev/sdaa", "/dev/sdab", "/dev/sdac", "/dev/sdad", "/dev/sdae",
    "/dev/sdaf", "/dev/sdag", "/dev/sdah", "/dev/sdai", "/dev/sdaj",
    "/dev/sdak", "/dev/sdal", "/dev/sdam", "/dev/sdan", "/dev/sdao",
    "/dev/sdap", "/dev/sdaq", "/dev/sdar", "/dev/sdas", "/dev/sdat",
    "/dev/sdau", "/dev/sdav", "/dev/sdaw", "/dev/sdax", "/dev/sday",
    "/dev/sdaz", "/dev/sdba", "/dev/sdbb", "/dev/sdbc", "/dev/sdbd",
    //"/dev/sdbe", "/dev/sdbf", "/dev/sdbg", "/dev/sdbh", "/dev/sdbi",
    "/dev/sdbj", "/dev/sdbk", "/dev/sdbl", "/dev/sdbm", "/dev/sdbn",
    "/dev/sdbo", "/dev/sdbp", "/dev/sdbq", "/dev/sdbr", "/dev/sdbs",
    "/dev/sdbt", "/dev/sdbu", "/dev/sdbv", "/dev/sdbw", "/dev/sdbx",
    "/dev/sdby", "/dev/sdbz", "/dev/sdca", "/dev/sdcb", "/dev/sdcc",
};


//
// Storage Buffer
//
StorageBuffer::StorageBuffer(size_t count) : count_(count), raid_(devices) {
  // TODO: Make device lists as parameter

  small_buf_ = reinterpret_cast<double *>(aligned_alloc(512, 512));
  if (small_buf_ == nullptr) {
    throw std::bad_alloc();
  }
  raid_.alloc(reinterpret_cast<void **>(&buf_), sizeof(double) * count);
}

StorageBuffer::~StorageBuffer() { raid_.free(reinterpret_cast<void **>(buf_)); }

double StorageBuffer::__getitem__(size_t key) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(double);

  int err = raid_.read(small_buf_, (void *)base_addr, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }

  return small_buf_[idx];
}

void StorageBuffer::__setitem__(size_t key, double val) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(double);

  int err = raid_.read(small_buf_, (void *)base_addr, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }
  small_buf_[idx] = val;

  err = raid_.write((void *)base_addr, small_buf_, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }
}

} // namespace snuqs
