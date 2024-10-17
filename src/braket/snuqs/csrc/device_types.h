#ifndef _DEVICE_H_
#define _DEVICE_H_
#include <cassert>
#include <string>
enum class DeviceType {
  UNKNOWN = 0,
  CPU = 1,
  CUDA = 2,
  STORAGE = 3,
};

static std::string device_to_string(DeviceType device) {
  switch (device) {
    case DeviceType::UNKNOWN:
      return "UNKNOWN";
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::STORAGE:
      return "STORAGE";
  }
  assert(false);
  return "";
}

#endif  //_DEVICE_H_
