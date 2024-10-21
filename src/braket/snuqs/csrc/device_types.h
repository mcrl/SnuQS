#ifndef _DEVICE_H_
#define _DEVICE_H_
#include <cassert>
#include <string>
enum class DeviceType {
  CPU = 0,
  CUDA = 1,
  STORAGE = 2,
};

static std::string device_to_string(DeviceType device) {
  switch (device) {
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
