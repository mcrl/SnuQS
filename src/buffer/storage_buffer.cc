#include "assertion.h"
#include "buffer.h"

namespace snuqs {
//
// Storage Buffer
//
StorageBuffer::StorageBuffer(size_t size) : size_(size) {}
StorageBuffer::~StorageBuffer() { NOT_IMPLEMENTED(); }
double StorageBuffer::__getitem__(size_t key) { NOT_IMPLEMENTED(); }
void StorageBuffer::__setitem__(size_t key, double val) { NOT_IMPLEMENTED(); }

} // namespace snuqs
