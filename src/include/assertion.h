#ifndef __ASSERTION_H__
#define __ASSERTION_H__

#include <cassert>
#include <stdexcept>

#define ERROR(msg) assert(false && (msg));
#define NOT_IMPLEMENTED() (throw std::domain_error("Not implemented yet"))

#define DO_NOTHING()

#endif // __ASSERTION_H__
