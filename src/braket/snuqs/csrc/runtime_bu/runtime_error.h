#ifndef _RUNTIME_ERROR_H
#define _RUNTIME_ERROR_H

namespace runtime {

enum RuntimeError {
  RT_SUCCESS = 0,
  RT_NULL_POINTER,
  RT_ILLEGAL_TASK,
  RT_ILLEGAL_DESTROY,
  RT_NOT_IMPLEMENTED,
  RT_INVALID_DEVICE,
  RT_INVALID_ADDRESS,
  RT_ZERO_ALLOCATION,
  RT_ZERO_IO_REQUEST,
};

static const char *error_to_string(RuntimeError error) {
  switch (error) {
  case RT_SUCCESS:
    return "RT_SUCCESS";
  case RT_NULL_POINTER:
    return "RT_NULL_POINTER";
  case RT_ILLEGAL_TASK:
    return "RT_ILLEGAL_TASK";
  case RT_ILLEGAL_DESTROY:
    return "RT_ILLEGAL_DESTROY";
  case RT_NOT_IMPLEMENTED:
    return "RT_NOT_IMPLEMENTED";
  case RT_INVALID_DEVICE:
    return "RT_INVALID_DEVICE";
  case RT_INVALID_ADDRESS:
    return "RT_INVALID_ADDRESS";
  case RT_ZERO_ALLOCATION:
    return "RT_ZERO_ALLOCATION";
  case RT_ZERO_IO_REQUEST:
    return "RT_ZERO_IO_REQUEST";
  }
  return "UNKNOWN";
}

} // namespace runtime

#endif //_RUNTIME_ERROR_H
