#ifndef __RT_EVENT_H__
#define __RT_EVENT_H__
#include "rt.h"

namespace snuqs {
namespace rt {

event_t create_event(handle_t handle);
void destroy_event(handle_t handle, event_t event);

} // namespace rt
} // namespace snuqs

#endif //__RT_EVENT_H__
