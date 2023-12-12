#ifndef __ASSERTION_H__
#define __ASSERTION_H__

#include <cassert>
#include <stdexcept>

#define ERROR(msg) assert(false && (msg));
#define POS(msg) ("["+std::string(__FILE__)+":"+std::to_string(__LINE__)+"] "+std::string(msg))
#define NOT_IMPLEMENTED() (throw std::domain_error(POS("Not implemented yet")))
#define NOT_SUPPORTED() (throw std::domain_error(POS("Not supported")))
#define CANNOT_BE_HERE() (throw std::domain_error(POS("Cannot be here")))
#define DO_NOTHING()

#endif // __ASSERTION_H__
