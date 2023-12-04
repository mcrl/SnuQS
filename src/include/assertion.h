#ifndef __ASSERTION_H__
#define __ASSERTION_H__

#include <cassert>

#define ERROR(msg) assert(false && (msg));
#define NOT_IMPLEMENTED() ERROR("NOT IMPLEMENTED")

#define DO_NOTHING() 

#endif // __ASSERTION_H__
